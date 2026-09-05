"""
End-to-end profiling for YOLO26n.

This script measures validation-time efficiency metrics:
- mAP50-95
- mAP50
- preprocessing / inference / postprocessing time
- FPS
- CPU usage
- process memory usage
- CUDA peak memory, if using GPU

This is different from 2.1_profile_yolo26n_layers.py:
- 2.1 measures internal layer latency using a fixed random tensor.
- 2.2 measures full validation pipeline behavior using a dataset YAML.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

import psutil
import torch
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Profile YOLO26n end-to-end validation speed and resource usage.")
parser.add_argument("--model", required=True, type=Path, help="Path to trained or official YOLO .pt model.")
parser.add_argument("--data", required=True, type=str, help="Dataset YAML path, e.g. mio_tcd.yaml, coco8.yaml, or coco128.yaml.")
parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split used for profiling.")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
parser.add_argument("--batch", type=int, default=16, help="Batch size.")
parser.add_argument("--device", default="cpu", help="GPU id such as 0, or cpu.")
parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
parser.add_argument("--project", default="profiling", help="Ultralytics output project folder.")
parser.add_argument("--name", default="end_to_end_profile", help="Ultralytics run name.")
parser.add_argument("--out", required=True, type=Path, help="Output CSV path.")
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(f"Model file not found: {args.model}")

# A real dataset YAML is required because this script measures end-to-end validation behavior.
# For Ultralytics built-in datasets, values such as coco8.yaml and coco128.yaml are accepted.
if args.data.endswith(".yaml") is False:
    raise ValueError("--data must be a YOLO dataset YAML name or path, e.g. mio_tcd.yaml or coco128.yaml.")

process = psutil.Process(os.getpid())

model = YOLO(str(args.model))

using_cuda = args.device != "cpu"
if using_cuda:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but CUDA is not available. Use --device cpu.")
    torch.cuda.reset_peak_memory_stats()

# Measure Python process CPU time and wall time.
# CPU percent is computed from process CPU seconds / wall seconds / CPU core count.
cpu_times_before = process.cpu_times()
rss_before_mb = process.memory_info().rss / (1024 * 1024)
start_time = time.perf_counter()

results = model.val(
    data=args.data,
    split=args.split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    project=args.project,
    name=args.name,
    exist_ok=True,
)

elapsed_s = time.perf_counter() - start_time
cpu_times_after = process.cpu_times()
rss_after_mb = process.memory_info().rss / (1024 * 1024)

cpu_seconds = (
    cpu_times_after.user
    + cpu_times_after.system
    - cpu_times_before.user
    - cpu_times_before.system
)

cpu_percent = 0.0
if elapsed_s > 0:
    cpu_percent = cpu_seconds / elapsed_s / os.cpu_count() * 100.0

peak_cuda_memory_mb = 0.0
if using_cuda:
    peak_cuda_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

# Ultralytics reports speed in milliseconds per image.
# The key names usually include preprocess, inference, loss, and postprocess depending on mode.
speed = getattr(results, "speed", {})
preprocess_ms = float(speed.get("preprocess", 0.0))
inference_ms = float(speed.get("inference", 0.0))
postprocess_ms = float(speed.get("postprocess", 0.0))
latency_ms_per_image = preprocess_ms + inference_ms + postprocess_ms
fps = 0.0 if latency_ms_per_image == 0 else 1000.0 / latency_ms_per_image

args.out.parent.mkdir(parents=True, exist_ok=True)

with args.out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "data",
            "split",
            "imgsz",
            "batch",
            "device",
            "map50_95",
            "map50",
            "preprocess_ms_per_image",
            "inference_ms_per_image",
            "postprocess_ms_per_image",
            "latency_ms_per_image",
            "fps",
            "elapsed_seconds",
            "cpu_percent_process",
            "rss_before_mb",
            "rss_after_mb",
            "rss_delta_mb",
            "peak_cuda_memory_mb",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "model": str(args.model),
            "data": args.data,
            "split": args.split,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "map50_95": f"{float(results.box.map):.6f}",
            "map50": f"{float(results.box.map50):.6f}",
            "preprocess_ms_per_image": f"{preprocess_ms:.6f}",
            "inference_ms_per_image": f"{inference_ms:.6f}",
            "postprocess_ms_per_image": f"{postprocess_ms:.6f}",
            "latency_ms_per_image": f"{latency_ms_per_image:.6f}",
            "fps": f"{fps:.6f}",
            "elapsed_seconds": f"{elapsed_s:.6f}",
            "cpu_percent_process": f"{cpu_percent:.6f}",
            "rss_before_mb": f"{rss_before_mb:.6f}",
            "rss_after_mb": f"{rss_after_mb:.6f}",
            "rss_delta_mb": f"{rss_after_mb - rss_before_mb:.6f}",
            "peak_cuda_memory_mb": f"{peak_cuda_memory_mb:.6f}",
        }
    )

print("End-to-end profiling completed.")
print(f"Output CSV: {args.out.resolve()}")
print(f"mAP50-95: {float(results.box.map):.6f}")
print(f"mAP50: {float(results.box.map50):.6f}")
print(f"Latency: {latency_ms_per_image:.6f} ms/image")
print(f"FPS: {fps:.6f}")
print(f"Process CPU usage: {cpu_percent:.6f}%")
print(f"RSS memory delta: {rss_after_mb - rss_before_mb:.6f} MB")
print(f"Peak CUDA memory: {peak_cuda_memory_mb:.6f} MB")