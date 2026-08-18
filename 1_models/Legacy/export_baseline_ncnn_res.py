from ultralytics import YOLO
from pathlib import Path
import shutil

PT_MODEL = Path(r"C:/a USask/yolo_project/1_models/2_full/best.pt")
OUTPUT_ROOT = Path(r"C:/a USask/yolo_project/1_models/2_full/resolution_tests")

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

for imgsz in [480, 320]:
    model = YOLO(str(PT_MODEL))

    exported = Path(
        model.export(
            format="ncnn",
            imgsz=imgsz,
            batch=1,
            device="cpu",
        )
    )

    target = OUTPUT_ROOT / f"baseline_full_{imgsz}"

    if target.exists():
        shutil.rmtree(target)

    shutil.move(str(exported), str(target))

    print(f"{imgsz}: {target}")