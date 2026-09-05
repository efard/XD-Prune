from pathlib import Path
from ultralytics import YOLO

import layer_replacement_adapter

# "C:/a USask/yolo_project/1_models/full/best.pt"
model_path = Path(
    "C:/a USask/yolo_project/1_models/2_full/best2n.pt"
)

if not model_path.is_file():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(str(model_path))

export_path = model.export(
    format="ncnn",
    imgsz=640,
    #quantize=16,
    batch=1,
    device="cpu",
    half=False,
    int8=False,
    end2end=False,
)

print(f"NCNN export completed: {export_path}")