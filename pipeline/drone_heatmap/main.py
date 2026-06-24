import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import time
import sys  
from pathlib import Path
import cv2
import pandas as pd
import traceback
from dotenv import load_dotenv
import numpy as np
import base64

from modules.SceneUnderstanding import SceneUnderstanding
from modules.Heatmap import Heatmap
from modules.Segment import Segment, Segmentation
from modules.GeoLocalizer import GeoLocalizer

from scripts.video_helper import (
    compose_video_frame,
    get_video_writer,
    release_video_writer,
)

# Pull central dataset directories directly from the config subfolder package
from config.lm_config import DATASET_ROOT  

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Find cars")
    parser.add_argument("--mask", default=None)
    return parser.parse_args()


class DroneHeatmap: 
    def __init__(self, dataset_root: str, task="Find cars", mask=None, sam_step=15):
        self.dataset_root = Path(dataset_root)
        self.task = task
        self.mask = mask
        self.sam_step = sam_step
        
        self.query_csv = pd.read_csv(self.dataset_root / "query.csv")
        self.query_images_dir = (self.dataset_root / "query_images")

        self.index = 0
        self.output_dir = Path("examples") / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.scene_understanding = SceneUnderstanding()
        self.segmentation = Segment()
        self.heatmap = Heatmap()
        self.geo_localizer = GeoLocalizer()

        self.video_writer = None
        self.video_path = None

    def has_next(self) -> bool:
        return self.index < len(self.query_csv)

    def reset(self):
        self.index = 0

    def should_run_sam(self, frame):
        return frame["frame_index"] % self.sam_step == 0

    def get_next_frame(self):
        while self.has_next():
            frame_index = self.index
            row = self.query_csv.iloc[frame_index]
            self.index += 1

            image_path = (self.query_images_dir / row["name"])
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"Skipping unreadable image: {image_path}")
                continue

            image[:, :, 1] = (image[:, :, 1] * 0.65).astype(image.dtype)
            image[:, :, 2:3] = (image[:, :, 2:3] * 0.8).astype(image.dtype)

            return {
                "image": image,
                "image_path": str(image_path),
                "easting": row["easting"],
                "northing": row["northing"],
                "altitude": row["altitude"],
                "orientation": (
                    row["orient_x"],
                    row["orient_y"],
                    row["orient_z"],
                    row["orient_w"],
                ),
                "frame_index": frame_index,
            }

        return None
    
    def close_video(self):
        release_video_writer(self.video_writer)
        self.video_writer = None

    def show_video(self, image, header, side_image=None, side_header=None):
        image = compose_video_frame(
            image,
            header,
            side_image=side_image,
            side_header=side_header,
        )

        self.video_writer, video_path = get_video_writer(
            self.video_writer,
            image,
            self.output_dir,
        )
        if video_path is not None:
            self.video_path = video_path

        self.video_writer.write(image)
        return image
        
    def label_mask(self, image: np.ndarray, segmentations: list[Segmentation]):
        if self.mask is None: return None

        mask_frame = np.zeros_like(image)

        for segmentation in segmentations:
            if segmentation.label.lower() != self.mask.lower():
                continue

            mask_bool = segmentation.mask.astype(bool)
            mask_frame[mask_bool] = image[mask_bool]

            mask = segmentation.mask.astype(np.uint8)
            blurred = cv2.GaussianBlur(mask * 255, (51, 51), 0)
            alpha = (blurred.astype(np.float32) / 255.0)[..., None]

            feathered = (
                mask_frame.astype(np.float32) * (1.0 - alpha)
                + image.astype(np.float32) * alpha
            ).astype(np.uint8)

            mask_frame[~mask_bool] = feathered[~mask_bool]
                        
        return mask_frame

    def run(self):
        if self.has_next():
            frame = self.get_next_frame()
            if frame is None:
                return None

            image = frame["image"]
            out = image

            position = (
                frame["easting"],
                frame["northing"],
                frame["altitude"]
            )

            scene_dict = None
            if self.should_run_sam(frame):
                scene_dict = self.scene_understanding.get_labels(image, self.task)
            
            segmentations = self.segmentation.get_segmentations(image, scene_dict)
            if segmentations is None:
                segmentations = []

            for segmentation in segmentations:
                segmentation.geo_pos = self.geo_localizer.get_location(
                    image,
                    segmentation.mask,
                    position
                )

            mask_frame = None
            if self.mask is not None:
                mask_frame = self.label_mask(image, segmentations)

            heatmap = self.heatmap.draw_heatmap(image, segmentations)

            if heatmap is not None: 
                out = heatmap

            video_frame = self.show_video(
                out,
                header=f"Task: {self.task}",
                side_image=mask_frame,
                side_header=f"Mask: {self.mask}",
            )
            return video_frame

        return None
        

if __name__ == "__main__":
    args = parse_args()
    
    # FIX: Cast DATASET_ROOT Path object directly to a string representation
    drone = DroneHeatmap(
        str(DATASET_ROOT),
        task=args.task,
        mask=args.mask,
    )

    # Instantiate clean native window wrapper
    window_name = "Drone Target Tracking Heatmap"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"🚀 Initializing tracking loop for task: '{args.task}'")
    print("👉 Click on the OpenCV video window and press 'q' to stop processing completely.")

    try:
        while drone.has_next():
            video_frame = drone.run()
            if video_frame is None:
                break
            
            # Directly display the composited heatmap on the main thread execution line
            cv2.imshow(window_name, video_frame)
            
            # Process frames continuously but monitor for single keystroke terminations
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[User Interrupt] Processing loop stopped via 'q' window key.")
                break

    except KeyboardInterrupt:
        print("\n[Terminal Interrupt] Stopping drone tracking stream via user signal...")

    except Exception:
        traceback.print_exc()

    finally:
        print("Releasing streaming assets and video buffers...")
        drone.close_video()
        cv2.destroyAllWindows()
        print("All processes closed safely. Goodbye.")
        sys.exit(0)