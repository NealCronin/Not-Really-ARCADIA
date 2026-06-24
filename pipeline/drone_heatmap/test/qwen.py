import base64
import json
import os
import re

import cv2
from openai import OpenAI


IMAGE_PATH = "001024.png"
MODEL = os.environ.get("QWEN_MODEL", "qwen/qwen3-vl-235b-a22b-instruct")
TIMEOUT_SECONDS = 60
MAX_FEATURES = 6
MAX_POINTS_PER_FEATURE = 8


def parse_json_response(text):
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
        raise


def draw_feature_points(image_path, data):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    features = data.get("features", [])
    point_count = 0

    for feature_index, feature in enumerate(features):
        label = str(feature.get("label", f"feature {feature_index}"))
        points = feature.get("points", [])

        for point_index, point in enumerate(points):
            if len(point) < 2:
                continue

            x, y = point[:2]
            px = int(round(float(x)))
            py = int(round(float(y)))

            if not (0 <= px < image.shape[1] and 0 <= py < image.shape[0]):
                print(f"Skipping out-of-bounds point for {label}: x={px}, y={py}")
                continue

            cv2.circle(image, (px, py), 7, (0, 0, 255), -1)
            cv2.circle(image, (px, py), 9, (255, 255, 255), 2)
            cv2.putText(
                image,
                f"{feature_index}.{point_index} {label}",
                (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            print(f"{label}: x={px}, y={py}")
            point_count += 1

    cv2.imshow(f"Qwen points - {image_path}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return point_count


client = OpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
)

image = cv2.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

image_height, image_width = image.shape[:2]

with open(IMAGE_PATH, "rb") as image_file:
    image_b64 = base64.b64encode(image_file.read()).decode("utf-8")

prompt = f"""
This is an aerial image.

Return pixel coordinates for visually important features in the image.
The image size is width={image_width}, height={image_height}.

Use this exact image coordinate system:
- Pixel coordinates are [x, y].
- The top-left pixel is [0, 0].
- The top-right pixel is [{image_width - 1}, 0].
- The bottom-left pixel is [0, {image_height - 1}].
- The bottom-right pixel is [{image_width - 1}, {image_height - 1}].
- The image center is approximately [{image_width // 2}, {image_height // 2}].
- x increases left to right.
- y increases top to bottom.

Focus on features useful for scene understanding and search:
roads, trails, buildings, vehicles, tree clusters, open fields, water, and other visible landmarks.

Do not invent objects. Only return visible features.

Strict output rules:
- Return valid JSON only.
- Do not use markdown fences.
- Do not include comments, explanations, or trailing commas.
- Coordinates must be integer pixel coordinates in this image.
- x must be between 0 and {image_width - 1}.
- y must be between 0 and {image_height - 1}.
- Return at most {MAX_FEATURES} feature objects.
- Return at most {MAX_POINTS_PER_FEATURE} points per feature.

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
    max_tokens=2000,
    timeout=TIMEOUT_SECONDS,
)

content = response.choices[0].message.content
print(content)

data = parse_json_response(content)
point_count = draw_feature_points(IMAGE_PATH, data)
print(f"Visualized {point_count} point(s).")
