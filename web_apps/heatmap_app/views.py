import sys
from pathlib import Path
from django.conf import settings

# Dynamically add the drone heatmap pipeline directory to the python path
project_root = Path(__file__).resolve().parent.parent.parent
pipeline_dir = project_root / "pipeline" / "drone_heatmap"
if str(pipeline_dir) not in sys.path:
    sys.path.insert(0, str(pipeline_dir))

# Now Python can successfully resolve the import
from main import DroneHeatmap

# 2. Rest of your standard imports
import cv2
import traceback
from django.shortcuts import render, redirect
from django.http import StreamingHttpResponse, HttpResponseBadRequest
from django.contrib import messages

from main import DroneHeatmap


def heatmap(request):
    """Renders the dedicated drone tracking and heatmap analysis mission control workspace."""
    context = {
        "default_dataset": str(getattr(settings, "DRONE_DATASET_DIR", "")),
        "default_task": "Find cars",
        "default_mask": ""
    }
    # FIXED: Explicitly targets your dedicated app-specific template file name
    return render(request, "heatmap_app/heatmap.html", context)


def gen_frames(dataset_root, task, mask):
    """Generator function that runs the drone loop and yields encoded JPEGs."""
    drone = DroneHeatmap(
        dataset_root=dataset_root,
        task=task,
        mask=mask if mask else None
    )
    
    print(f"🎬 Starting Django live stream for Task: '{task}' on path: {dataset_root}")
    
    try:
        while drone.has_next():
            video_frame = drone.run()
            if video_frame is None:
                break
                
            ret, buffer = cv2.imencode('.jpg', video_frame)
            if not ret:
                continue
                
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
    except Exception as e:
        print(f"❌ Exception in streaming generator loop:")
        traceback.print_exc()
        
    finally:
        print("💾 Releasing streaming asset writes cleanly.")
        drone.close_video()


def video_stream(request):
    """Endpoint that returns the real-time multipart image stream."""
    dataset_root = request.GET.get("dataset_root") or str(settings.DRONE_DATASET_DIR)
    task = request.GET.get("task", "Find cars")
    mask = request.GET.get("mask", None)

    if not dataset_root:
        return HttpResponseBadRequest("Missing 'dataset_root' parameter configuration.")

    return StreamingHttpResponse(
        gen_frames(dataset_root, task, mask),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )