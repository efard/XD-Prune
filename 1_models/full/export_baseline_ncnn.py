from pathlib import Path
from ultralytics import YOLO

# "C:/a USask/yolo_project/1_models/full/best.pt"
model_path = Path(
    "C:/a USask/yolo_project/1_models/trained_baseline/yolo26n_mio_11000_e100_img640_b16_s42_best.pt"
)

if not model_path.is_file():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = YOLO(str(model_path))

export_path = model.export(
    format="ncnn",
    imgsz=640,
    batch=1,
    device="cpu",
    half=False,
    int8=False,
    end2end=False,
)

print(f"NCNN export completed: {export_path}")