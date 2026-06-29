"""
Train the baseline YOLO26n model on the converted MIO-TCD YOLO dataset.

Default behavior:
    - train using train split
    - validate using val split
    - do NOT use test split unless --run-test is explicitly passed

This protects the locked test set for final evaluation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Train baseline YOLO26n.")
parser.add_argument("--model", required=True, type=Path, help="Path to YOLO26n .pt file.")
parser.add_argument("--data", required=True, type=Path, help="Path to dataset YAML file.")
parser.add_argument("--epochs", type=int, default=100, help="Training epochs.")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
parser.add_argument("--batch", type=int, default=16, help="Batch size.")
parser.add_argument("--device", default="0", help="GPU id such as 0, or cpu.")
parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
parser.add_argument("--seed", type=int, default=42, help="Training seed.")
parser.add_argument("--project", default="runs/baseline", help="Ultralytics output project folder.")
parser.add_argument("--name", default="baseline_yolo26n_mio_11000", help="Run name.")
parser.add_argument("--run-test", action="store_true", help="Evaluate test split after training. Use only for final evaluation.")
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(f"Model file not found: {args.model}")

if not args.data.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

model = YOLO(str(args.model))

# Training uses the YAML train/val split. The validation set is used for epoch selection
# and model comparison, while the test set remains locked by default.
train_results = model.train(
    data=str(args.data),
    epochs=args.epochs,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    seed=args.seed,
    project=args.project,
    name=args.name,
    exist_ok=False,
)

print("Training completed.")
print(train_results)

# Run a clean validation pass on the validation split after training. This records mAP50-95
# for the baseline without touching the locked test split.
val_results = model.val(
    data=str(args.data),
    split="val",
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    project=args.project,
    name=f"{args.name}_val",
    exist_ok=False,
)

print("Validation completed.")
print(f"Validation mAP50-95: {val_results.box.map:.6f}")
print(f"Validation mAP50: {val_results.box.map50:.6f}")

if args.run_test:
    # This is intentionally behind an explicit flag so the test split is not used during
    # training decisions, profiling decisions, pruning decisions, or quantization choices.
    test_results = model.val(
        data=str(args.data),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=f"{args.name}_test",
        exist_ok=False,
    )
    print("Test evaluation completed.")
    print(f"Test mAP50-95: {test_results.box.map:.6f}")
    print(f"Test mAP50: {test_results.box.map50:.6f}")
