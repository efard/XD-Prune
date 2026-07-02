"""
INT8 post-training quantization (PTQ) experiment for YOLO26n.

Purpose
-------
This script matches:
- keep the FP32 .pt baseline as the reference
- export an INT8 PTQ model using calibration data from the dataset YAML
- validate the exported INT8 model on the same validation split
- report model size, mAP50-95, mAP50, latency, FPS, and accuracy drop

Important experiment rule
-------------------------
The default split is "val". Do not use "test" unless this is the final evaluation.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from ultralytics import YOLO


def file_size_mb(path: Path) -> float:
    # File size is used as a simple deployment-size metric for the report table.
    if not path.is_file():
        raise FileNotFoundError(f"Expected model file does not exist: {path}")
    return path.stat().st_size / (1024 * 1024)


def summarize_val_results(results) -> dict[str, float]:
    # Ultralytics val() returns mAP metrics and speed values in milliseconds per image.
    speed = getattr(results, "speed", {})
    preprocess_ms = float(speed.get("preprocess", 0.0))
    inference_ms = float(speed.get("inference", 0.0))
    postprocess_ms = float(speed.get("postprocess", 0.0))
    latency_ms = preprocess_ms + inference_ms + postprocess_ms
    fps = 0.0 if latency_ms == 0.0 else 1000.0 / latency_ms

    return {
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "preprocess_ms": preprocess_ms,
        "inference_ms": inference_ms,
        "postprocess_ms": postprocess_ms,
        "latency_ms": latency_ms,
        "fps": fps,
    }


def validate_model(
    model_path: Path,
    data: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
) -> dict[str, float]:
    # A new YOLO object is created for each model path so FP32 and INT8 are evaluated separately.
    model = YOLO(str(model_path))
    results = model.val(
        data=str(data),
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(project),
        name=name,
        exist_ok=True,
    )
    return summarize_val_results(results)


parser = argparse.ArgumentParser(description="Run YOLO26n INT8 PTQ export and validation.")
parser.add_argument("--model", required=True, type=Path, help="Path to trained FP32 YOLO26n .pt model.")
parser.add_argument("--data", required=True, type=Path, help="Path to dataset YAML, e.g. mio_tcd.yaml.")
parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split. Use test only for final evaluation.")
parser.add_argument("--format", default="onnx", choices=["onnx", "tflite", "engine"], help="Export format for INT8 PTQ.")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
parser.add_argument("--batch", type=int, default=16, help="Validation/export batch size.")
parser.add_argument("--device", default="0", help="GPU id such as 0, or cpu.")
parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of calibration data used by Ultralytics INT8 export.")
parser.add_argument("--project", default=Path("runs/quantization"), type=Path, help="Ultralytics validation output folder.")
parser.add_argument("--out-dir", required=True, type=Path, help="Folder for exported model and CSV summary.")
parser.add_argument("--summary-csv", default="quantization_summary.csv", help="Summary CSV filename inside --out-dir.")
parser.add_argument("--copy-export", action="store_true", help="Copy exported model into --out-dir for easier tracking.")
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(f"Model file not found: {args.model}")

if not args.data.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

if not (0.0 < args.fraction <= 1.0):
    raise ValueError("--fraction must be in the range (0, 1].")

args.out_dir.mkdir(parents=True, exist_ok=True)
args.project.mkdir(parents=True, exist_ok=True)

print("Validating FP32 baseline model...")
fp32_metrics = validate_model(
    model_path=args.model,
    data=args.data,
    split=args.split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    project=args.project,
    name="fp32_baseline_val",
)

print("Exporting INT8 PTQ model...")
fp32_model = YOLO(str(args.model))

# INT8 PTQ requires the dataset YAML for calibration. The calibration data estimates
# activation ranges before values are represented with lower precision.
exported_path_raw = fp32_model.export(
    format=args.format,
    int8=True,
    data=str(args.data),
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    fraction=args.fraction,
)

exported_path = Path(exported_path_raw)
if not exported_path.exists():
    raise FileNotFoundError(f"Ultralytics export returned a path that does not exist: {exported_path}")

tracked_export_path = exported_path
if args.copy_export:
    tracked_export_path = args.out_dir / exported_path.name
    shutil.copy2(exported_path, tracked_export_path)

print("Validating INT8 exported model...")
int8_metrics = validate_model(
    model_path=tracked_export_path,
    data=args.data,
    split=args.split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    project=args.project,
    name=f"int8_{args.format}_val",
)

fp32_size_mb = file_size_mb(args.model)
int8_size_mb = file_size_mb(tracked_export_path)

summary_path = args.out_dir / args.summary_csv
fieldnames = [
    "method",
    "model_path",
    "format",
    "split",
    "size_mb",
    "map50_95",
    "map50",
    "drop_map50_95",
    "drop_map50",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "latency_ms",
    "fps",
]

rows = [
    {
        "method": "FP32 baseline",
        "model_path": str(args.model),
        "format": "pt",
        "split": args.split,
        "size_mb": f"{fp32_size_mb:.6f}",
        "map50_95": f"{fp32_metrics['map50_95']:.6f}",
        "map50": f"{fp32_metrics['map50']:.6f}",
        "drop_map50_95": "0.000000",
        "drop_map50": "0.000000",
        "preprocess_ms": f"{fp32_metrics['preprocess_ms']:.6f}",
        "inference_ms": f"{fp32_metrics['inference_ms']:.6f}",
        "postprocess_ms": f"{fp32_metrics['postprocess_ms']:.6f}",
        "latency_ms": f"{fp32_metrics['latency_ms']:.6f}",
        "fps": f"{fp32_metrics['fps']:.6f}",
    },
    {
        "method": "INT8 PTQ",
        "model_path": str(tracked_export_path),
        "format": args.format,
        "split": args.split,
        "size_mb": f"{int8_size_mb:.6f}",
        "map50_95": f"{int8_metrics['map50_95']:.6f}",
        "map50": f"{int8_metrics['map50']:.6f}",
        "drop_map50_95": f"{fp32_metrics['map50_95'] - int8_metrics['map50_95']:.6f}",
        "drop_map50": f"{fp32_metrics['map50'] - int8_metrics['map50']:.6f}",
        "preprocess_ms": f"{int8_metrics['preprocess_ms']:.6f}",
        "inference_ms": f"{int8_metrics['inference_ms']:.6f}",
        "postprocess_ms": f"{int8_metrics['postprocess_ms']:.6f}",
        "latency_ms": f"{int8_metrics['latency_ms']:.6f}",
        "fps": f"{int8_metrics['fps']:.6f}",
    },
]

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("INT8 PTQ experiment completed.")
print(f"Exported INT8 model: {tracked_export_path.resolve()}")
print(f"Summary CSV: {summary_path.resolve()}")
print(f"FP32 mAP50-95: {fp32_metrics['map50_95']:.6f}")
print(f"INT8 mAP50-95: {int8_metrics['map50_95']:.6f}")
print(f"mAP50-95 drop: {fp32_metrics['map50_95'] - int8_metrics['map50_95']:.6f}")
