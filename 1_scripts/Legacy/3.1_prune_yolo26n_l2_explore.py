"""
L2-norm structured pruning exploration for YOLO26n.

Purpose
-------
This script matches:
- choose target layers using profiling + error-injection sensitivity results
- apply L2-norm structured pruning to Conv2d output channels inside those layers
- remove the PyTorch pruning mask so the zeroed weights become permanent
- validate the pruned model on the same validation split
- report model size, parameter count, zero-weight percentage, mAP, latency, FPS, and accuracy drop

Important limitation
--------------------
This is pruning-mask exploration, not final dense-architecture rebuilding.
The tensor shapes remain unchanged, so normal hardware may not show real latency speedup.
For final deployment speedup, the selected pruning result should later be converted into
a smaller dense YOLO architecture and exported again.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO


def parse_layer_ids(text: str) -> list[int]:
    # The user must explicitly choose layers from profiling/sensitivity results.
    # This prevents accidentally pruning all layers without experimental justification.
    ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not ids:
        raise ValueError("--target-layers must contain at least one layer id, e.g. 10,13,16.")
    return ids


def file_size_mb(path: Path) -> float:
    if not path.is_file():
        raise FileNotFoundError(f"Expected model file does not exist: {path}")
    return path.stat().st_size / (1024 * 1024)


def summarize_val_results(results) -> dict[str, float]:
    # Ultralytics val() reports speed in milliseconds per image.
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


def validate_yolo(
    yolo: YOLO,
    data: Path,
    split: str,
    imgsz: int,
    batch: int,
    device: str,
    workers: int,
    project: Path,
    name: str,
) -> dict[str, float]:
    results = yolo.val(
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


def count_parameters_and_zeros(model: nn.Module) -> tuple[int, int]:
    # Total parameters remain the same after pruning-mask exploration.
    # The zero count shows how many weights became zero after pruning.
    total_params = 0
    zero_params = 0
    with torch.no_grad():
        for param in model.parameters():
            total_params += param.numel()
            zero_params += int((param == 0).sum().item())
    return total_params, zero_params


def convs_inside_layer(layer: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    # Ultralytics layers may be wrappers such as Conv or C3k2.
    # named_modules() finds all internal Conv2d modules inside the selected YOLO layer.
    return [(name, module) for name, module in layer.named_modules() if isinstance(module, nn.Conv2d)]


parser = argparse.ArgumentParser(description="Run YOLO26n L2-norm structured pruning exploration.")
parser.add_argument("--model", required=True, type=Path, help="Path to trained FP32 YOLO26n .pt model.")
parser.add_argument("--data", required=True, type=Path, help="Path to dataset YAML, e.g. mio_tcd.yaml.")
parser.add_argument("--split", default="val", choices=["val", "test"], help="Evaluation split. Use test only for final evaluation.")
parser.add_argument("--target-layers", required=True, help="Comma-separated YOLO layer ids selected from profiling/sensitivity, e.g. 10,13,16.")
parser.add_argument("--amount", type=float, default=0.10, help="Fraction of output channels pruned in each selected Conv2d module.")
parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
parser.add_argument("--device", default="0", help="GPU id such as 0, or cpu.")
parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
parser.add_argument("--project", default=Path("runs/pruning"), type=Path, help="Ultralytics validation output folder.")
parser.add_argument("--out-dir", required=True, type=Path, help="Folder for pruned model and CSV summary.")
parser.add_argument("--save-model", default="yolo26n_l2_pruned.pt", help="Filename for saved pruned model inside --out-dir.")
parser.add_argument("--summary-csv", default="pruning_summary.csv", help="Summary CSV filename inside --out-dir.")
parser.add_argument("--details-csv", default="pruned_layers_detail.csv", help="Detailed pruning CSV filename inside --out-dir.")
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(f"Model file not found: {args.model}")

if not args.data.is_file():
    raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

if not (0.0 < args.amount < 1.0):
    raise ValueError("--amount must be in the range (0, 1).")

target_layer_ids = parse_layer_ids(args.target_layers)

args.out_dir.mkdir(parents=True, exist_ok=True)
args.project.mkdir(parents=True, exist_ok=True)

yolo = YOLO(str(args.model))
layers = yolo.model.model

for layer_id in target_layer_ids:
    if layer_id < 0 or layer_id >= len(layers):
        raise ValueError(f"Layer id {layer_id} is outside valid range 0 to {len(layers) - 1}.")

print("Validating FP32 baseline model...")
baseline_metrics = validate_yolo(
    yolo=yolo,
    data=args.data,
    split=args.split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    project=args.project,
    name="fp32_baseline_val",
)

baseline_total_params, baseline_zero_params = count_parameters_and_zeros(yolo.model)

pruned_detail_rows = []

print("Applying L2-norm structured pruning...")
for layer_id in target_layer_ids:
    layer = layers[layer_id]
    layer_type = layer.__class__.__name__

    if layer_type == "Detect":
        raise ValueError("Detect layer pruning is blocked in this exploration script. Select internal Conv/C3k2/SPPF/neck layers.")

    conv_modules = convs_inside_layer(layer)
    if not conv_modules:
        raise ValueError(f"Layer {layer_id} ({layer_type}) does not contain any Conv2d module to prune.")

    for conv_name, conv in conv_modules:
        original_weight = conv.weight.detach().clone()

        # L2-norm structured pruning removes output channels with the smallest L2 norm.
        # dim=0 means output-channel pruning for Conv2d weights shaped as
        # [out_channels, in_channels, kernel_h, kernel_w].
        prune.ln_structured(
            module=conv,
            name="weight",
            amount=args.amount,
            n=2,
            dim=0,
        )

        mask = conv.weight_mask.detach().clone()
        pruned_output_channels = int((mask.view(mask.shape[0], -1).sum(dim=1) == 0).sum().item())

        # prune.remove() makes the masked weight tensor permanent by removing the
        # reparameterization. The architecture shape is still unchanged.
        prune.remove(conv, "weight")

        zeroed_weights = int((conv.weight.detach() == 0).sum().item() - (original_weight == 0).sum().item())

        pruned_detail_rows.append(
            {
                "layer_id": layer_id,
                "layer_type": layer_type,
                "conv_name": conv_name if conv_name else "<layer_root>",
                "out_channels": conv.out_channels,
                "pruned_output_channels": pruned_output_channels,
                "amount": args.amount,
                "new_zero_weights": zeroed_weights,
            }
        )

print("Validating pruned model...")
pruned_metrics = validate_yolo(
    yolo=yolo,
    data=args.data,
    split=args.split,
    imgsz=args.imgsz,
    batch=args.batch,
    device=args.device,
    workers=args.workers,
    project=args.project,
    name="l2_structured_pruned_val",
)

pruned_total_params, pruned_zero_params = count_parameters_and_zeros(yolo.model)

save_path = args.out_dir / args.save_model
if not hasattr(yolo, "save"):
    raise RuntimeError(
        "This Ultralytics version does not expose YOLO.save(). "
        "Please save through the version-supported checkpoint method before using this script."
    )

# The saved file records the pruned-weight state for traceability.
# It is not a smaller dense architecture.
yolo.save(str(save_path))

baseline_size_mb = file_size_mb(args.model)
pruned_size_mb = file_size_mb(save_path)

details_path = args.out_dir / args.details_csv
with details_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "layer_id",
            "layer_type",
            "conv_name",
            "out_channels",
            "pruned_output_channels",
            "amount",
            "new_zero_weights",
        ],
    )
    writer.writeheader()
    writer.writerows(pruned_detail_rows)

summary_path = args.out_dir / args.summary_csv
fieldnames = [
    "method",
    "model_path",
    "split",
    "size_mb",
    "total_params",
    "zero_params",
    "zero_param_percent",
    "map50_95",
    "map50",
    "drop_map50_95",
    "drop_map50",
    "preprocess_ms",
    "inference_ms",
    "postprocess_ms",
    "latency_ms",
    "fps",
    "target_layers",
    "amount",
    "note",
]

baseline_zero_percent = 0.0 if baseline_total_params == 0 else baseline_zero_params / baseline_total_params * 100.0
pruned_zero_percent = 0.0 if pruned_total_params == 0 else pruned_zero_params / pruned_total_params * 100.0

rows = [
    {
        "method": "FP32 baseline",
        "model_path": str(args.model),
        "split": args.split,
        "size_mb": f"{baseline_size_mb:.6f}",
        "total_params": baseline_total_params,
        "zero_params": baseline_zero_params,
        "zero_param_percent": f"{baseline_zero_percent:.6f}",
        "map50_95": f"{baseline_metrics['map50_95']:.6f}",
        "map50": f"{baseline_metrics['map50']:.6f}",
        "drop_map50_95": "0.000000",
        "drop_map50": "0.000000",
        "preprocess_ms": f"{baseline_metrics['preprocess_ms']:.6f}",
        "inference_ms": f"{baseline_metrics['inference_ms']:.6f}",
        "postprocess_ms": f"{baseline_metrics['postprocess_ms']:.6f}",
        "latency_ms": f"{baseline_metrics['latency_ms']:.6f}",
        "fps": f"{baseline_metrics['fps']:.6f}",
        "target_layers": "",
        "amount": "",
        "note": "original trained model",
    },
    {
        "method": "L2 structured pruning exploration",
        "model_path": str(save_path),
        "split": args.split,
        "size_mb": f"{pruned_size_mb:.6f}",
        "total_params": pruned_total_params,
        "zero_params": pruned_zero_params,
        "zero_param_percent": f"{pruned_zero_percent:.6f}",
        "map50_95": f"{pruned_metrics['map50_95']:.6f}",
        "map50": f"{pruned_metrics['map50']:.6f}",
        "drop_map50_95": f"{baseline_metrics['map50_95'] - pruned_metrics['map50_95']:.6f}",
        "drop_map50": f"{baseline_metrics['map50'] - pruned_metrics['map50']:.6f}",
        "preprocess_ms": f"{pruned_metrics['preprocess_ms']:.6f}",
        "inference_ms": f"{pruned_metrics['inference_ms']:.6f}",
        "postprocess_ms": f"{pruned_metrics['postprocess_ms']:.6f}",
        "latency_ms": f"{pruned_metrics['latency_ms']:.6f}",
        "fps": f"{pruned_metrics['fps']:.6f}",
        "target_layers": args.target_layers,
        "amount": args.amount,
        "note": "mask-based pruning exploration; architecture shape unchanged",
    },
]

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("Pruning exploration completed.")
print(f"Saved pruned model: {save_path.resolve()}")
print(f"Summary CSV: {summary_path.resolve()}")
print(f"Details CSV: {details_path.resolve()}")
print(f"Baseline mAP50-95: {baseline_metrics['map50_95']:.6f}")
print(f"Pruned mAP50-95: {pruned_metrics['map50_95']:.6f}")
print(f"mAP50-95 drop: {baseline_metrics['map50_95'] - pruned_metrics['map50_95']:.6f}")
print("Note: this pruned model keeps the same architecture shape; real speedup may require dense model rebuilding.")
