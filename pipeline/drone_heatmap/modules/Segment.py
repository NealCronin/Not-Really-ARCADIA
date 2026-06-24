import base64
import cv2
import numpy as np
from dataclasses import dataclass
from modules.llama_request_helper import Sam3Client

@dataclass
class Segmentation:
    mask: np.ndarray
    label: str
    id: str
    score: float   
    geo_pos: tuple[float, float, float] | None = None
    
class Segment():

    def __init__(self):
        self.segmentations = []
        self.predictor = Sam3Client(host="127.0.0.1", port=8701)
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.prev_gray = None

    def _parse_dict(self, scene_dict):
        return list(scene_dict.keys())

    def _get_flow_map(self, curr_image):
        curr_gray = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
        flow = self.dis.calc(self.prev_gray, curr_gray, None)

        h, w = curr_image.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        map_x = (x - flow[..., 0]).astype(np.float32)
        map_y = (y - flow[..., 1]).astype(np.float32)

        self.prev_gray = curr_gray
        return map_x, map_y

    def get_segmentations(self, image, scene_dict):
        # --- OPTICAL FLOW TRACKING STEP ---
        if scene_dict is None:
            if self.prev_gray is None: 
                return self.segmentations

            map_x, map_y = self._get_flow_map(image)

            for segmentation in self.segmentations:
                segmentation.mask = cv2.remap(
                    segmentation.mask.astype(np.uint8),
                    map_x,
                    map_y,
                    interpolation=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
            return self.segmentations

        # --- NEW SAM REMOTE INFERENCE STEP ---
        else:
            self.prev_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            self.segmentations = []  

            prompts = self._parse_dict(scene_dict)
            if len(prompts) < 1:
                return self.segmentations

            response = self.predictor.request(
                image_base64=image,
                prompts=prompts,
            )
        
            if "error" in response:
                print(f"❌ SAM 3 Server Error: {response['error']}")
                return self.segmentations
                
            predictions = response.get("predictions", [])
            h, w = image.shape[:2]

            # Reconstruct polygons locally into binary arrays & initialize native objects
            for pred in predictions:
                label_name = pred.get("label", "unknown")
                sam3_confidence = pred.get("score", 0.0)  
                mask_xy = pred.get("mask_xy", [])

                # Pull the fuzzy-matched combined score
                matched_scene_info = scene_dict.get(label_name, {})
                final_heatmap_score = matched_scene_info.get("combined_score", None)

                if final_heatmap_score is None:
                    # Key lookup fallback chain
                    final_heatmap_score = 0

                final_heatmap_score = float(np.clip(final_heatmap_score, 0.0, 1.0))

                # Create an isolated single-channel mask array for this detection
                binary_mask = np.zeros((h, w), dtype=np.uint8)
                if len(mask_xy) > 0:
                    pts = np.array(mask_xy, dtype=np.int32)
                    cv2.fillPoly(binary_mask, [pts], 1)

                self.segmentations.append(
                    Segmentation(
                        mask=binary_mask,
                        label=label_name,
                        score=final_heatmap_score,  
                        id=""
                    )
                )
            
            return self.segmentations