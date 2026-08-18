from ultralytics import YOLO

model = YOLO("C:/a USask/yolo_project/1_models/2_full/best.pt")

model.export(
    format="ncnn",
    imgsz=640,
    quantize=16,
    device="cpu",
)