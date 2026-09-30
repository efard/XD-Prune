cd /home/afm176/yolo_project
module load apptainer/1.4.5

PYTHONPATH=/project/ko/afm176/yolo-packages:/project/ko/afm176/python_packages_ncnn_minimal \
apptainer exec \
--bind /home/afm176:/home/afm176 \
--bind /project/ko:/project/ko \
/opt/software/apptainer-images/pytorch-25.06.sif \
python - <<'PY'
import csv
from pathlib import Path
from ultralytics import YOLO

# Input / output paths ------------------------------------------------------------
MODEL = Path("/home/afm176/yolo_project/1_models/V5_ncnn_model")

# Use the exact ACDC/SNOW validation dataset used by the PT experiment
# so the NCNN result is directly comparable.
DATA = Path(
    "/project/ko/afm176/datasets/1_data/"
    "reproduction_exact_v1/SNOW_ACDC_exact/dataset_SNOW_local.yaml"
)

OUTPUT_CSV = Path("/home/afm176/yolo_project/v5_ncnn_validation.csv")


# Validate the NCNN model ------------------------------------------------------------
model = YOLO(str(MODEL))

metrics = model.val(
    data=str(DATA),
    split="val",
    imgsz=640,
    batch=1,
    device="cpu",
    workers=4,
    rect=True,
    conf=0.001,
    iou=0.70,
    max_det=300,
    augment=False,
    plots=False,
)


# Extract accuracy and timing ------------------------------------------------------------
speed = metrics.speed

preprocess_ms = float(speed.get("preprocess", 0.0))
inference_ms = float(speed.get("inference", 0.0))
postprocess_ms = float(speed.get("postprocess", 0.0))

# Keep the same total-latency definition used elsewhere in the project:
# preprocess + inference + postprocess.
avg_latency_ms = preprocess_ms + inference_ms + postprocess_ms

# Convert milliseconds/image to frames per second.
fps = 0.0 if avg_latency_ms <= 0.0 else 1000.0 / avg_latency_ms


# Save one summary row to CSV ------------------------------------------------------------
row = {
    "model": str(MODEL),
    "dataset": str(DATA),
    "validation_split": "val",
    "imgsz": 640,
    "batch": 1,
    "mAP50_95": float(metrics.box.map),
    "mAP50": float(metrics.box.map50),
    "mAP75": float(metrics.box.map75),
    "precision": float(metrics.box.mp),
    "recall": float(metrics.box.mr),
    "preprocess_ms": preprocess_ms,
    "inference_ms": inference_ms,
    "postprocess_ms": postprocess_ms,
    "avg_latency_ms": avg_latency_ms,
    "fps": fps,
}

with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=row.keys())
    writer.writeheader()
    writer.writerow(row)


# Print the same key values to terminal for quick checking ------------------------------------------------------------
print("\n===== V5 NCNN RESULT =====")
print(f"mAP50-95      = {row['mAP50_95']:.8f}")
print(f"mAP50         = {row['mAP50']:.8f}")
print(f"mAP75         = {row['mAP75']:.8f}")
print(f"Precision     = {row['precision']:.8f}")
print(f"Recall        = {row['recall']:.8f}")
print(f"Avg latency   = {row['avg_latency_ms']:.4f} ms")
print(f"FPS           = {row['fps']:.4f}")
print(f"CSV saved to  = {OUTPUT_CSV}")
PY