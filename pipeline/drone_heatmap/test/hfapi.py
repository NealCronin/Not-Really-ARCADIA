import ast
import os
import re

import cv2
from gradio_client import Client, handle_file


IMAGE_PATH = "001024.png"
POINTS_INDEX = 1


def parse_points(points_text):
    if not points_text:
        return []

    if not isinstance(points_text, str):
        return points_text

    try:
        points = ast.literal_eval(points_text)
        if isinstance(points, list):
            return points
    except (SyntaxError, ValueError):
        pass

    numbers = [
        float(value)
        for value in re.findall(r"-?\d+(?:\.\d+)?", points_text)
    ]
    return [
        numbers[index:index + 4]
        for index in range(0, len(numbers) - 3, 4)
    ]


def draw_points(image_path, points):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    for index, point in enumerate(points):
        if len(point) < 4:
            continue

        object_id, image_index, x, y = point[:4]
        px = int(round(float(x)))
        py = int(round(float(y)))
        label = f"{int(object_id)}"

        cv2.circle(image, (px, py), 7, (0, 0, 255), -1)
        cv2.circle(image, (px, py), 9, (255, 255, 255), 2)
        cv2.putText(
            image,
            label,
            (px + 10, py - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        print(f"Point {index}: object_id={object_id}, image_index={image_index}, x={x}, y={y}")

    cv2.imshow(f"MolmoPoint - {image_path}", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

client = Client(
    "allenai/MolmoPoint-8B-Demo",
    token=os.environ["HF_TOKEN"],
)

client.predict(api_name="/_select_image_tab")

result = client.predict(
    user_text="""
        Identify visually distinct regions, objects, structures, terrain features, and environmental elements in this aerial image.

        Point to representative locations for each identified element. Include both broad environmental features and specific objects when visible.

        Favor features that are spatially meaningful for scene understanding, navigation, search and rescue, or environment characterization.

        Do not limit yourself to predefined categories. Use the most appropriate labels based on what is actually visible in the image.

        For each point, provide a concise semantic label describing what it refers to.
    """,
    video_tracking_path=None,
    video_pointing_path=None,
    input_images=[
        {
            "image": handle_file(IMAGE_PATH),
            "caption": None,
        }
    ],
    fsm="uniform_last_frame",
    mf=384,
    mfps=10,
    max_tok=2048,
    api_name="/dispatch_submit",
)

generated_text = result[0]
points_text = result[POINTS_INDEX]
points = parse_points(points_text)

print("Generated text:")
print(generated_text)
print("Extracted points:")
print(points)

draw_points(IMAGE_PATH, points)
