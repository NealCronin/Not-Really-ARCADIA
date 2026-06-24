import os

import cv2
import numpy as np
import torch
from ultralytics import SAM


IMAGE_PATH = os.environ.get("SAM_IMAGE", "001024.png")
SAM_MODEL = os.environ.get("SAM_MODEL", "models/mobile_sam.pt")
SAM_DEVICE = os.environ.get("SAM_DEVICE", "0" if torch.cuda.is_available() else "cpu")


def extract_first_mask(results):
    if not results:
        return None

    result = results[0]
    if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
        return None

    return result.masks.data[0].detach().cpu().numpy() > 0


def normalize_mask(mask, image_shape):
    if mask.shape[:2] != image_shape[:2]:
        return cv2.resize(
            mask.astype(np.uint8),
            (image_shape[1], image_shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    return mask.astype(bool)


def show_mask(image, mask, points, labels):
    mask = normalize_mask(mask, image.shape)
    output = image.copy()
    color = np.array([0, 180, 255], dtype=np.uint8)

    output[mask] = cv2.addWeighted(
        output[mask],
        0.55,
        np.full_like(output[mask], color),
        0.45,
        0,
    )

    contours, _hierarchy = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    cv2.drawContours(output, contours, -1, (0, 255, 255), 2, cv2.LINE_AA)

    output = draw_prompt_points(output, points, labels)
    cv2.imshow("MobileSAM segmentation", output)


def draw_prompt_points(image, points, labels):
    output = image.copy()

    for index, (point, label) in enumerate(zip(points, labels)):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        marker = "+" if label == 1 else "-"
        cv2.circle(output, point, 5, (255, 255, 255), -1)
        cv2.circle(output, point, 8, color, 2)
        cv2.putText(
            output,
            f"{marker}{index}",
            (point[0] + 10, point[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def main():
    image = cv2.imread(IMAGE_PATH)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")

    print(f"Loading MobileSAM: {SAM_MODEL} on device={SAM_DEVICE}")
    sam = SAM(SAM_MODEL)
    points = []
    labels = []

    def on_mouse(event, x, y, _flags, _param):
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return

        label = 1 if event == cv2.EVENT_LBUTTONDOWN else 0
        points.append((x, y))
        labels.append(label)
        preview = draw_prompt_points(image, points, labels)
        cv2.imshow("Click road point", preview)
        point_type = "positive" if label == 1 else "negative"
        print(f"Added {point_type} point: x={x}, y={y}")

    cv2.imshow("Click road point", image)
    cv2.setMouseCallback("Click road point", on_mouse)
    print("Left-click positive points. Right-click negative points.")
    print("Press Enter/Space to segment, Backspace to undo, or Esc/q to quit.")

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            return
        if key in (8, 127) and points:
            removed_point = points.pop()
            removed_label = labels.pop()
            point_type = "positive" if removed_label == 1 else "negative"
            print(f"Removed {point_type} point: x={removed_point[0]}, y={removed_point[1]}")
            cv2.imshow("Click road point", draw_prompt_points(image, points, labels))
        if key in (13, 32) and points:
            break

    print("Running MobileSAM point prompts:")
    for point, label in zip(points, labels):
        point_type = "positive" if label == 1 else "negative"
        print(f"  {point_type}: x={point[0]}, y={point[1]}")

    results = sam.predict(
        IMAGE_PATH,
        points=[[x, y] for x, y in points],
        labels=labels,
        device=SAM_DEVICE,
        retina_masks=True,
        verbose=False,
    )

    mask = extract_first_mask(results)
    if mask is None:
        print("No mask returned.")
        cv2.destroyAllWindows()
        return

    show_mask(image, mask, points, labels)
    print("Press any key in the segmentation window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
