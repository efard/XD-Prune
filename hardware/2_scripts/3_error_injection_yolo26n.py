"""
Layer-based error injection for YOLO26n.

the script adds Gaussian noise to layer's output activation and evaluates mAP50-95 on the validation split.
A larger mAP drop means the layer is more sensitive to errors.

Default behavior:
- uses split=val
- excludes Detect layers by default
- never uses the test split unless passes --split test
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from ultralytics import YOLO


parser = argparse.ArgumentParser(description="Run layer-based activation error injection.")
parser.add_argument("--model", required=True, type=Path, help="Path to trained YOLO .pt model.")
parser.add_argument("--data", required=True, type=Path, help="Path to dataset YAML file.")
parser.add_argument("--split", default="val", choices=["val", "test"], help="Dataset split for evaluation.")
parser.add_argument("--target-layers", default="", help="Comma-separated layer ids, e.g. 0,1,2. Empty means all non-Detect layers.")
parser.add_argument("--include-detect", action="store_true", help="Also test Detect layers.")
parser.add_argument("--noise-std-ratio", type=float, default=0.01, help="Noise std = activation std * this ratio.")
parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
parser.add_argument("--device", default="0", help="GPU id such as 0, or cpu.")
parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
parser.add_argument("--seed", type=int, default=42, help="Seed used for injected noise.")
parser.add_argument("--out", required=True, type=Path, help="Output CSV path.")
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(f"Model file not found: {args.model}")

if not args.data.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

if args.noise_std_ratio < 0:
    raise ValueError("noise-std-ratio must be non-negative.")

yolo = YOLO(str(args.model))
layers = yolo.model.model

if args.target_layers.strip():
    target_layer_ids = [int(item.strip()) for item in args.target_layers.split(",") if item.strip()]
else:
    target_layer_ids = []
    for layer_index, layer in enumerate(layers):
        if args.include_detect or layer.__class__.__name__ != "Detect":
            target_layer_ids.append(layer_index)

for layer_id in target_layer_ids:
    if layer_id < 0 or layer_id >= len(layers):
        raise ValueError(f"Layer id {layer_id} is outside valid range 0 to {len(layers) - 1}.")

args.out.parent.mkdir(parents=True, exist_ok=True)

print("Running baseline validation without injected error...")
baseline_results = yolo.val(
    data=str(args.data),
    split=args.split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    project="error_inj",
    name="baseline_no_injection",
    exist_ok=True,
)
baseline_map = float(baseline_results.box.map)
baseline_map50 = float(baseline_results.box.map50)
print(f"Baseline {args.split} mAP50-95: {baseline_map:.6f}")
print(f"Baseline {args.split} mAP50: {baseline_map50:.6f}")

rows = [
    {
        "layer_id": "baseline",
        "type": "none",
        "noise_std_ratio": args.noise_std_ratio,
        "map50_95": f"{baseline_map:.6f}",
        "map50": f"{baseline_map50:.6f}",
        "drop_map50_95": "0.000000",
        "drop_map50": "0.000000",
    }
]

with args.out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["layer_id", "type", "noise_std_ratio", "map50_95", "map50", "drop_map50_95", "drop_map50"],
    )
    writer.writeheader()
    writer.writerows(rows)

for layer_id in target_layer_ids:
    layer = layers[layer_id]
    layer_type = layer.__class__.__name__
    print(f"Injecting layer {layer_id}: {layer_type}")

    def inject_noise(_module, _inputs, output):
        # Noise scale is relative to the activation standard deviation so layers with
        # different numerical ranges receive comparable relative disturbance.
        torch.manual_seed(args.seed + layer_id)

        def perturb(value):
            if torch.is_tensor(value):
                activation_std = value.detach().float().std()
                noise = torch.randn_like(value) * activation_std * args.noise_std_ratio
                return value + noise
            if isinstance(value, list):
                return [perturb(item) for item in value]
            if isinstance(value, tuple):
                return tuple(perturb(item) for item in value)
            return value

        return perturb(output)

    hook_handle = layer.register_forward_hook(inject_noise)
    try:
        injected_results = yolo.val(
            data=str(args.data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project="error_inj",
            name=f"layer_{layer_id}_noise_{args.noise_std_ratio}",
            exist_ok=True,
        )
    finally:
        hook_handle.remove()

    injected_map = float(injected_results.box.map)
    injected_map50 = float(injected_results.box.map50)
    drop_map = baseline_map - injected_map
    drop_map50 = baseline_map50 - injected_map50

    rows.append(
        {
            "layer_id": layer_id,
            "type": layer_type,
            "noise_std_ratio": args.noise_std_ratio,
            "map50_95": f"{injected_map:.6f}",
            "map50": f"{injected_map50:.6f}",
            "drop_map50_95": f"{drop_map:.6f}",
            "drop_map50": f"{drop_map50:.6f}",
        }
    )

    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["layer_id", "type", "noise_std_ratio", "map50_95", "map50", "drop_map50_95", "drop_map50"],
        )
        writer.writeheader()
        writer.writerows(rows)

print("Error injection completed.")
print(f"Output CSV: {args.out.resolve()}")
