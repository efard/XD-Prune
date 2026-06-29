"""
Profile YOLO26n layer-wise latency.

The script runs a manual forward pass through the Ultralytics model graph and measures
how long each layer takes. This is useful for finding bottleneck layers before pruning,
quantization, or FPGA acceleration decisions.

The input is a fixed random tensor. This measures model-layer compute latency, not dataset
accuracy. For accuracy, use validation/error injection scripts.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Profile YOLO26n layer-wise latency.")
parser.add_argument("--model", required=True, type=Path, help="Path to trained or official YOLO .pt model.")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
parser.add_argument("--device", default="0", help="GPU id such as 0, or cpu.")
parser.add_argument("--warmup", type=int, default=20, help="Warmup iterations not recorded.")
parser.add_argument("--iters", type=int, default=100, help="Measured iterations.")
parser.add_argument("--seed", type=int, default=42, help="Seed for the fixed random input.")
parser.add_argument("--out", required=True, type=Path, help="Output CSV path.")
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(f"Model file not found: {args.model}")

if args.device == "cpu":
    device = torch.device("cpu")
else:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device was requested, but CUDA is not available. Use --device cpu for CPU profiling.")
    device = torch.device(f"cuda:{args.device}")

torch.manual_seed(args.seed)
yolo = YOLO(str(args.model))
detection_model = yolo.model.to(device).eval()
layers = detection_model.model
save_indices = set(detection_model.save)

# The fixed random input keeps the profiling run reproducible. It avoids using validation
# or test images because latency profiling does not require labels.
input_tensor = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)

layer_time_ms = [0.0 for _ in layers]
layer_output_shape = ["" for _ in layers]

def format_shape(value):
    # This helper is used for every layer output. It supports Tensor, list, and tuple outputs.
    if torch.is_tensor(value):
        return str(list(value.shape))
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_shape(item) for item in value) + "]"
    return type(value).__name__


total_iterations = args.warmup + args.iters

with torch.no_grad():
    for iteration in range(total_iterations):
        saved_outputs = []
        x = input_tensor

        for layer_index, layer in enumerate(layers):
            # Ultralytics YOLO layers can receive input from previous layers through layer.f.
            # -1 means the immediate previous output. A list means feature fusion, such as Concat.
            if layer.f != -1:
                if isinstance(layer.f, int):
                    x_in = saved_outputs[layer.f]
                else:
                    x_in = [x if source_index == -1 else saved_outputs[source_index] for source_index in layer.f]
            else:
                x_in = x

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            x = layer(x_in)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            if iteration >= args.warmup:
                layer_time_ms[layer_index] += elapsed_ms
                layer_output_shape[layer_index] = format_shape(x)

            # Only layers listed in detection_model.save are needed later by skip/concat layers.
            # Non-saved outputs are replaced by None to match Ultralytics graph behavior.
            saved_outputs.append(x if layer.i in save_indices else None)

rows = []
total_latency_ms = sum(layer_time_ms[layer_index] / args.iters for layer_index in range(len(layers)))

for layer_index, layer in enumerate(layers):
    mean_ms = layer_time_ms[layer_index] / args.iters
    param_count = sum(param.numel() for param in layer.parameters())
    latency_percent = 0.0 if total_latency_ms == 0 else mean_ms / total_latency_ms * 100.0
    rows.append(
        {
            "layer_id": layer_index,
            "type": layer.__class__.__name__,
            "from": str(layer.f),
            "params": param_count,
            "mean_latency_ms": f"{mean_ms:.6f}",
            "latency_percent": f"{latency_percent:.6f}",
            "output_shape": layer_output_shape[layer_index],
        }
    )

args.out.parent.mkdir(parents=True, exist_ok=True)
with args.out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["layer_id", "type", "from", "params", "mean_latency_ms", "latency_percent", "output_shape"],
    )
    writer.writeheader()
    writer.writerows(rows)

print("Layer profiling completed.")
print(f"Output CSV: {args.out.resolve()}")
print(f"Total measured model latency estimate: {total_latency_ms:.6f} ms")
