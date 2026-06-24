import base64
import json
import os
import re

import cv2
import numpy as np
import torch
from openai import OpenAI
from ultralytics import SAM

IMAGE_PATH = "001024.png"
MODEL = os.environ.get("GEMINI_MODEL", "google/gemini-2.5-flash")
SAM_MODEL = os.environ.get("SAM_MODEL", "models/mobile_sam.pt")
SAM_DEVICE = os.environ.get("SAM_DEVICE", "0" if torch.cuda.is_available() else "cpu")
TIMEOUT_SECONDS = 60
MAX_FEATURES = 1
MAX_POINTS_PER_FEATURE = 4
MAX_TOTAL_POINTS = 4
USE_COORDINATE_GRID = True
GRID_SPACING = 100

client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)


image_for_dims = cv2.imread(IMAGE_PATH)
if image_for_dims is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image_height, image_width = image_for_dims.shape[:2]


def make_coordinate_reference_image(image, spacing=100):
    output = image.copy()
    overlay = output.copy()
    height, width = output.shape[:2]

    for x in range(0, width, spacing):
        cv2.line(overlay, (x, 0), (x, height - 1), (255, 255, 255), 1)
        cv2.putText(
            overlay,
            str(x),
            (min(x + 3, width - 35), 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    for y in range(0, height, spacing):
        cv2.line(overlay, (0, y), (width - 1, y), (255, 255, 255), 1)
        cv2.putText(
            overlay,
            str(y),
            (4, max(16, y - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        overlay,
        f"x: 0..{width - 1}, y: 0..{height - 1}",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return cv2.addWeighted(overlay, 0.28, output, 0.72, 0)


image_for_gemini = (
    make_coordinate_reference_image(image_for_dims, GRID_SPACING)
    if USE_COORDINATE_GRID
    else image_for_dims
)
ok, buffer = cv2.imencode(".png", image_for_gemini)
if not ok:
    raise RuntimeError("Failed to encode image for Gemini.")

image_b64 = base64.b64encode(buffer).decode("utf-8")

prompt = f"""
This is an aerial image.

Return sparse pixel coordinates for the visible road network in the image.
The image size is width={image_width}, height={image_height}.

The image you see includes a faint coordinate grid and numeric pixel labels added only to help localization.
Ignore the grid as an object. Use it only to estimate accurate coordinates on the original image.

Use this exact image coordinate system:
- Pixel coordinates are [x, y].
- The top-left pixel is [0, 0].
- The top-right pixel is [{image_width - 1}, 0].
- The bottom-left pixel is [0, {image_height - 1}].
- The bottom-right pixel is [{image_width - 1}, {image_height - 1}].
- The image center is approximately [{image_width // 2}, {image_height // 2}].
- x increases left to right.
- y increases top to bottom.

Focus only on roads. Ignore buildings, vehicles, trees, fields, water, and all other features.
Do not invent roads. Only return points on clearly visible road pixels.

This is a sparse point prompt task for segmentation.
Do not trace outlines, perimeters, image borders, or dense paths.
Do not return contour points.
Return a road skeleton: only centerline anchor points on visible road surfaces.
Choose points that summarize the road geometry, such as start, bend, intersection, branch, and end.
Every point must lie near the center of a visible road, not on the road edge.

Strict output rules:
- Return valid JSON only.
- Do not use markdown fences.
- Do not include comments, explanations, or trailing commas.
- Coordinates must be integer pixel coordinates in this image.
- x must be between 0 and {image_width - 1}.
- y must be between 0 and {image_height - 1}.
- Return exactly one feature object.
- The feature label must be "road skeleton".
- Return at most {MAX_POINTS_PER_FEATURE} points total.
- If you are unsure, return fewer points.

Return exactly this JSON shape:
{{
  "features": [
    {{
      "label": "short label",
      "points": [[x, y], [x, y]]
    }}
  ]
}}
"""


def parse_json_response(text):
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as first_exc:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass

        print(f"Gemini returned invalid JSON, using partial point extraction: {first_exc}")
        return parse_partial_feature_points(text)


def parse_partial_feature_points(text):
    features = []
    label_matches = list(re.finditer(r'"label"\s*:\s*"([^"]+)"', text))

    for index, match in enumerate(label_matches[:MAX_FEATURES]):
        label = match.group(1)
        block_start = match.end()
        block_end = label_matches[index + 1].start() if index + 1 < len(label_matches) else len(text)
        block = text[block_start:block_end]
        pairs = re.findall(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]", block)
        points = [[float(x), float(y)] for x, y in pairs]

        if points:
            features.append(
                {
                    "label": label,
                    "points": simplify_points(points, MAX_POINTS_PER_FEATURE),
                }
            )

    return {"features": features}


def simplify_points(points, max_points):
    if len(points) <= max_points:
        return points

    if max_points <= 1:
        return [points[len(points) // 2]]

    indexes = np.linspace(0, len(points) - 1, max_points)
    return [points[int(round(index))] for index in indexes]


def normalize_feature_data(data, image_width, image_height):
    normalized = {"features": []}
    total_points = 0

    for feature in data.get("features", [])[:MAX_FEATURES]:
        label = str(feature.get("label", "feature")).strip() or "feature"
        raw_points = feature.get("points", [])
        points = []

        for point in raw_points:
            if total_points >= MAX_TOTAL_POINTS:
                break
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue

            try:
                px = int(round(float(point[0])))
                py = int(round(float(point[1])))
            except (TypeError, ValueError):
                continue

            if not (0 <= px < image_width and 0 <= py < image_height):
                print(f"Skipping out-of-bounds point for {label}: x={px}, y={py}")
                continue

            points.append([px, py])

        points = simplify_points(points, MAX_POINTS_PER_FEATURE)
        remaining_points = MAX_TOTAL_POINTS - total_points
        points = points[:remaining_points]
        if points:
            normalized["features"].append({"label": label, "points": points})
            total_points += len(points)

    return normalized


def iter_feature_points(image, data):
    height, width = image.shape[:2]
    features = data.get("features", [])

    for feature_index, feature in enumerate(features):
        label = str(feature.get("label", f"feature {feature_index}"))
        points = feature.get("points", [])

        for point_index, point in enumerate(points):
            if len(point) < 2:
                continue

            x, y = point[:2]
            px = int(round(float(x)))
            py = int(round(float(y)))

            if not (0 <= px < width and 0 <= py < height):
                print(f"Skipping out-of-bounds point for {label}: x={px}, y={py}")
                continue

            yield {
                "feature_index": feature_index,
                "point_index": point_index,
                "label": label,
                "x": px,
                "y": py,
            }


def color_for_label(label, fallback_index):
    palette = [
        (64, 180, 255),
        (80, 220, 120),
        (255, 140, 80),
        (210, 110, 255),
        (255, 220, 80),
        (120, 220, 255),
        (255, 110, 150),
        (180, 255, 120),
    ]
    color_index = (sum(ord(char) for char in label) + fallback_index) % len(palette)
    return palette[color_index]


def normalize_mask(mask, image_shape):
    if mask.shape[:2] != image_shape[:2]:
        return cv2.resize(
            mask.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask.astype(bool)


def overlay_mask(image, mask, color, alpha=0.36):
    colored = np.zeros_like(image)
    colored[mask] = color
    blended = image.copy()
    blended[mask] = cv2.addWeighted(colored[mask], alpha, image[mask], 1.0 - alpha, 0)
    return blended


def label_mask(image, mask, label, color, point):
    mask_uint8 = mask.astype(np.uint8)
    moments = cv2.moments(mask_uint8)
    if moments["m00"]:
        label_x = int(moments["m10"] / moments["m00"])
        label_y = int(moments["m01"] / moments["m00"])
    else:
        label_x, label_y = point

    label_text = label[:28]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    padding = 4
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text,
        font,
        scale,
        thickness,
    )

    x1 = max(0, min(image.shape[1] - text_width - padding * 2 - 1, label_x))
    y2 = max(text_height + baseline + padding * 2, min(image.shape[0] - 1, label_y))
    y1 = y2 - text_height - baseline - padding * 2
    x2 = x1 + text_width + padding * 2

    cv2.rectangle(image, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), 1)
    cv2.putText(
        image,
        label_text,
        (x1 + padding, y2 - baseline - padding),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def draw_point_anchor(image, point, color):
    px, py = point
    cv2.circle(image, (px, py), 5, (255, 255, 255), -1)
    cv2.circle(image, (px, py), 7, color, 2)


def extract_first_mask(results):
    if not results:
        return None

    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    return result.masks.data[0].detach().cpu().numpy() > 0


def segment_and_draw_feature_points(image_path, data):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    points = list(iter_feature_points(image, data))
    if not points:
        print("No valid Gemini points to segment.")
        cv2.imshow(f"Gemini + MobileSAM - {image_path}", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 0

    print(f"Loading MobileSAM: {SAM_MODEL} on device={SAM_DEVICE}")
    sam = SAM(SAM_MODEL)

    output = image.copy()
    point_count = 0

    for item in points:
        px = item["x"]
        py = item["y"]
        label = item["label"]
        feature_index = item["feature_index"]
        point_index = item["point_index"]
        color = color_for_label(label, feature_index)

        print(f"Segmenting {label}: x={px}, y={py}")
        try:
            results = sam.predict(
                image_path,
                points=[[px, py]],
                labels=[1],
                device=SAM_DEVICE,
                retina_masks=True,
                verbose=False,
            )
            mask = extract_first_mask(results)
        except Exception as exc:
            print(f"MobileSAM failed for {label} at x={px}, y={py}: {exc}")
            mask = None

        if mask is not None:
            mask = normalize_mask(mask, output.shape)
            output = overlay_mask(output, mask, color)
            contours, _hierarchy = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(output, contours, -1, color, 2, cv2.LINE_AA)
            label_mask(output, mask, label, color, (px, py))
        else:
            print(f"No mask returned for {label}: x={px}, y={py}")

        draw_point_anchor(output, (px, py), color)
        point_count += 1

    cv2.imshow(f"Gemini + MobileSAM - {image_path}", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return point_count

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_b64}"
                    },
                },
            ],
        }
    ],
    response_format={"type": "json_object"},
    max_tokens=900,
    timeout=TIMEOUT_SECONDS,
)

content = response.choices[0].message.content
print(content)

data = parse_json_response(content)
data = normalize_feature_data(data, image_width, image_height)
print("Normalized sparse points:")
print(json.dumps(data, indent=2))
point_count = segment_and_draw_feature_points(IMAGE_PATH, data)
print(f"Segmented and visualized {point_count} point prompt(s).")
