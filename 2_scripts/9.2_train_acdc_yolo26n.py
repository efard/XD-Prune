#!/usr/bin/env python3
"""
Train a YOLO26n baseline on the ACDC snow 400/100 split.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def metric_value(metrics: Any, path: str) -> float | None:
    value = metrics
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    try:
        return float(value)
    except Exception:
        return None


def write_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def validation_row(
    model_path: Path,
    data_yaml: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict[str, Any]:
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        half=False,
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        augment=False,
        plots=False,
        save_json=False,
        verbose=True,
        seed=42,
    )

    speed = getattr(metrics, "speed", {}) or {}
    preprocess = float(speed.get("preprocess", 0.0) or 0.0)
    inference = float(speed.get("inference", 0.0) or 0.0)
    postprocess = float(speed.get("postprocess", 0.0) or 0.0)
    avg_latency = preprocess + inference + postprocess

    return {
        "model": str(model_path.resolve()),
        "map50_95": metric_value(metrics, "box.map"),
        "map50": metric_value(metrics, "box.map50"),
        "map75": metric_value(metrics, "box.map75"),
        "precision": metric_value(metrics, "box.mp"),
        "recall": metric_value(metrics, "box.mr"),
        "preprocess_ms_per_image": preprocess,
        "inference_ms_per_image": inference,
        "postprocess_ms_per_image": postprocess,
        "avg_latency_ms": avg_latency,
        "fps": (1000.0 / avg_latency) if avg_latency > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--source-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    if not args.source_model.is_file():
        raise SystemExit(f"YOLO26n source model not found: {args.source_model}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.source_model.resolve()))

    started = time.time()
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=1.0,
        seed=args.seed,
        deterministic=True,
        amp=True,
        pretrained=True,
        project=str(args.output_dir),
        name="baseline_train",
        exist_ok=True,
        plots=False,
        verbose=True,
    )
    elapsed = time.time() - started

    train_dir = args.output_dir / "baseline_train"
    best = train_dir / "weights/best.pt"
    last = train_dir / "weights/last.pt"
    train_args = train_dir / "args.yaml"
    train_results = train_dir / "results.csv"

    if not best.is_file():
        raise RuntimeError(f"Training did not produce best.pt: {best}")

    baseline_best = args.output_dir / "ACDC_YOLO26n_baseline_best.pt"
    shutil.copy2(best, baseline_best)

    if last.is_file():
        shutil.copy2(last, args.output_dir / "ACDC_YOLO26n_baseline_last.pt")
    if train_args.is_file():
        shutil.copy2(train_args, args.output_dir / "baseline_training_args.yaml")
    if train_results.is_file():
        shutil.copy2(train_results, args.output_dir / "baseline_training_results.csv")

    row = validation_row(
        baseline_best,
        args.data.resolve(),
        args.imgsz,
        args.batch,
        args.workers,
        args.device,
    )
    row["training_elapsed_seconds"] = elapsed
    row["epochs"] = args.epochs
    row["seed"] = args.seed
    write_csv(args.output_dir / "baseline_validation.csv", row)

    summary = {
        "status": "PASSED_ACDC_BASELINE_TRAINING",
        "dataset_yaml": str(args.data.resolve()),
        "source_model": str(args.source_model.resolve()),
        "baseline_best": str(baseline_best.resolve()),
        "training_elapsed_seconds": elapsed,
        "training_args": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,
            "seed": args.seed,
            "deterministic": True,
            "amp": True,
        },
        "baseline_validation": row,
    }
    (args.output_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
