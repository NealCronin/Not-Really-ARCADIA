import base64
import io
import time
import argparse
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics.models.sam import SAM3SemanticPredictor

app = FastAPI(title="Ultralytics SAM 3 GPU Inference Server")

# Resolve model path safely relative to the project structure
project_root = Path(__file__).resolve().parent.parent.parent.parent
model_path = project_root / "pipeline" / "drone_heatmap" / "models" / "sam3.pt"

overrides = {
    "conf": 0.25,
    "task": "segment",
    "mode": "predict",
    "model": str(model_path),
    "device": "cuda",
    "half": True,
    "save": False,
}
predictor = SAM3SemanticPredictor(overrides=overrides)

class SAM3InferenceRequest(BaseModel):
    image: str
    prompts: List[str]
    conf: float = 0.25

@app.post("/v1/predict")
async def predict_sam3(req: SAM3InferenceRequest):
    start_time = time.perf_counter()
    try:
        image = req.image
        prompts = req.prompts
        predictor.args.conf = req.conf

        if "," in image:
            image = image.split(",", 1)[1]
            
        image_bytes = base64.b64decode(image)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_array = np.array(pil_image)

        predictor.set_image(image_array)
        results = predictor(text=prompts)

        serialized_predictions = []
        for result in results:
            has_boxes = hasattr(result, "boxes") and result.boxes is not None
            has_masks = hasattr(result, "masks") and result.masks is not None
            if not has_boxes:
                continue

            boxes = result.boxes.xyxy.tolist()
            scores = result.boxes.conf.tolist()
            classes = result.boxes.cls.tolist()
            names = result.names if hasattr(result, "names") else {}
            masks_xy = result.masks.xy if (has_masks and result.masks.xy is not None) else []

            for i in range(len(boxes)):
                cls_id = int(classes[i])
                label_string = names.get(cls_id, prompts[cls_id] if cls_id < len(prompts) else "object")

                serialized_predictions.append({
                    "box": boxes[i],
                    "score": scores[i],
                    "label": label_string,
                    "mask_xy": masks_xy[i].tolist() if i < len(masks_xy) else []
                })

        elapsed_time = time.perf_counter() - start_time
        print(f"⏱️ [SAM 3] Inference completed in: {elapsed_time:.4f}s")
        return {"status": "success", "predictions": serialized_predictions}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Failure: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8701)
    args = parser.parse_args()
    
    uvicorn.run("sam3_server:app", host=args.host, port=args.port, reload=False)