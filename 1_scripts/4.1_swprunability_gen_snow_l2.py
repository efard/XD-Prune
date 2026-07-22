from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from ultralytics import YOLO


def summarize(results) -> dict[str, float]:
    speed = getattr(results, "speed", {})
    pre = float(speed.get("preprocess", 0.0))
    inf = float(speed.get("inference", 0.0))
    post = float(speed.get("postprocess", 0.0))
    latency = pre + inf + post
    fps = 0.0 if latency == 0.0 else 1000.0 / latency
    return {
        "map50_95": float(results.box.map),
        "map50": float(results.box.map50),
        "latency_ms": latency,
        "fps": fps,
    }


def validate_model(yolo: YOLO, data: Path, split: str, imgsz: int, batch: int, device: str, workers: int, project: Path, name: str):
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
        plots=False,
    )
    return summarize(results)


def convs_inside_layer(layer: nn.Module):
    return [(name, module) for name, module in layer.named_modules() if isinstance(module, nn.Conv2d)]


def prune_layer_l2(layer: nn.Module, amount: float):
    detail_rows = []
    convs = convs_inside_layer(layer)

    if not convs:
        raise ValueError(f"Layer {layer.__class__.__name__} has no Conv2d modules.")

    for conv_name, conv in convs:
        original_zero = int((conv.weight.detach() == 0).sum().item())

        prune.ln_structured(
            module=conv,
            name="weight",
            amount=amount,
            n=2,
            dim=0,
        )

        mask = conv.weight_mask.detach().clone()
        pruned_channels = int((mask.view(mask.shape[0], -1).sum(dim=1) == 0).sum().item())

        prune.remove(conv, "weight")

        new_zero = int((conv.weight.detach() == 0).sum().item()) - original_zero

        detail_rows.append(
            {
                "conv_name": conv_name if conv_name else "<layer_root>",
                "out_channels": conv.out_channels,
                "pruned_output_channels": pruned_channels,
                "new_zero_weights": new_zero,
            }
        )

    return detail_rows


parser = argparse.ArgumentParser(description="GEN/SNOW SWPrunability with L2 structured pruning.")
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--gen-data", required=True, type=Path)
parser.add_argument("--snow-data", required=True, type=Path)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--target-layers", required=True)
parser.add_argument("--amount", required=True, type=float)
parser.add_argument("--imgsz", default=640, type=int)
parser.add_argument("--batch", default=16, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--out-dir", required=True, type=Path)
parser.add_argument("--epsilon", default=1e-6, type=float)
args = parser.parse_args()

if not args.model.is_file():
    raise FileNotFoundError(args.model)
if not args.gen_data.is_file():
    raise FileNotFoundError(args.gen_data)
if not args.snow_data.is_file():
    raise FileNotFoundError(args.snow_data)
if not (0.0 < args.amount < 1.0):
    raise ValueError("--amount must be between 0 and 1.")

args.out_dir.mkdir(parents=True, exist_ok=True)
args.project.mkdir(parents=True, exist_ok=True)

target_layers = [int(x.strip()) for x in args.target_layers.split(",") if x.strip()]

print("Loading baseline model for metadata...")
baseline_yolo = YOLO(str(args.model))
layers = baseline_yolo.model.model
total_params = sum(p.numel() for p in baseline_yolo.model.parameters())

metadata = {}

for layer_id in target_layers:
    layer = layers[layer_id]
    layer_type = layer.__class__.__name__

    if layer_type == "Detect":
        raise ValueError("Detect layer is not used for this structured-pruning experiment.")

    conv_count = len(convs_inside_layer(layer))
    layer_params = sum(p.numel() for p in layer.parameters())

    if conv_count == 0 or layer_params == 0:
        raise ValueError(f"Layer {layer_id} {layer_type} is not a valid pruning layer.")

    metadata[layer_id] = {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "layer_params": layer_params,
        "params_percent": layer_params / total_params * 100.0,
        "conv_count": conv_count,
    }

print("Evaluating baseline on GEN...")
gen_base = validate_model(
    baseline_yolo,
    args.gen_data,
    args.split,
    args.imgsz,
    args.batch,
    args.device,
    args.workers,
    args.project,
    "baseline_GEN",
)

print("Evaluating baseline on SNOW...")
snow_base = validate_model(
    baseline_yolo,
    args.snow_data,
    args.split,
    args.imgsz,
    args.batch,
    args.device,
    args.workers,
    args.project,
    "baseline_SNOW",
)

baseline_rows = [
    {"dataset": "GEN", "split": args.split, **gen_base},
    {"dataset": "SNOW", "split": args.split, **snow_base},
]

with (args.out_dir / "baseline_GEN_SNOW_metrics.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
    writer.writeheader()
    writer.writerows(baseline_rows)

with (args.out_dir / "layer_metadata.csv").open("w", newline="", encoding="utf-8") as f:
    fieldnames = ["layer_id", "layer_type", "layer_params", "params_percent", "conv_count"]
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for layer_id in target_layers:
        row = dict(metadata[layer_id])
        row["params_percent"] = f"{row['params_percent']:.6f}"
        writer.writerow(row)

raw_rows = []
detail_rows = []

for layer_id in target_layers:
    meta = metadata[layer_id]
    print(f"Testing layer {layer_id} {meta['layer_type']} amount={args.amount}")

    # Reload original checkpoint every time.
    yolo = YOLO(str(args.model))
    layer = yolo.model.model[layer_id]

    details = prune_layer_l2(layer, args.amount)

    for d in details:
        detail_rows.append(
            {
                "layer_id": layer_id,
                "layer_type": meta["layer_type"],
                "amount": args.amount,
                **d,
            }
        )

    gen_pruned = validate_model(
        yolo,
        args.gen_data,
        args.split,
        args.imgsz,
        args.batch,
        args.device,
        args.workers,
        args.project,
        f"L{layer_id}_GEN",
    )

    snow_pruned = validate_model(
        yolo,
        args.snow_data,
        args.split,
        args.imgsz,
        args.batch,
        args.device,
        args.workers,
        args.project,
        f"L{layer_id}_SNOW",
    )

    ad_gen = gen_base["map50_95"] - gen_pruned["map50_95"]
    ad_snow = snow_base["map50_95"] - snow_pruned["map50_95"]
    sensitivity = 0.5 * ad_gen + 0.5 * ad_snow

    # SWPrunability = Parameters% / sensitivity.
    swp_score = meta["params_percent"] / max(sensitivity, args.epsilon)

    raw_rows.append(
        {
            "layer_id": layer_id,
            "layer_type": meta["layer_type"],
            "amount": args.amount,
            "total_model_params": total_params,
            "layer_params": meta["layer_params"],
            "params_percent": f"{meta['params_percent']:.6f}",
            "gen_baseline_map50_95": f"{gen_base['map50_95']:.6f}",
            "gen_pruned_map50_95": f"{gen_pruned['map50_95']:.6f}",
            "ad_gen_map50_95": f"{ad_gen:.6f}",
            "snow_baseline_map50_95": f"{snow_base['map50_95']:.6f}",
            "snow_pruned_map50_95": f"{snow_pruned['map50_95']:.6f}",
            "ad_snow_map50_95": f"{ad_snow:.6f}",
            "combined_sensitivity": f"{sensitivity:.6f}",
            "swp_score": f"{swp_score:.6f}",
            "gen_latency_ms": f"{gen_pruned['latency_ms']:.6f}",
            "snow_latency_ms": f"{snow_pruned['latency_ms']:.6f}",
            "gen_fps": f"{gen_pruned['fps']:.6f}",
            "snow_fps": f"{snow_pruned['fps']:.6f}",
        }
    )

    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

with (args.out_dir / "raw_GEN_SNOW_individual_layer_pruning.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
    writer.writeheader()
    writer.writerows(raw_rows)

with (args.out_dir / "structured_pruning_details_by_layer.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
    writer.writeheader()
    writer.writerows(detail_rows)

t1_rows = sorted(raw_rows, key=lambda r: float(r["ad_gen_map50_95"]))
t2_rows = sorted(raw_rows, key=lambda r: float(r["ad_snow_map50_95"]))
t3_rows = sorted(raw_rows, key=lambda r: float(r["swp_score"]), reverse=True)

def write_table(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})

write_table(
    args.out_dir / "T1_GEN_sorted_by_AD_GEN.csv",
    t1_rows,
    [
        "layer_id",
        "layer_type",
        "layer_params",
        "params_percent",
        "gen_baseline_map50_95",
        "gen_pruned_map50_95",
        "ad_gen_map50_95",
        "amount",
    ],
)

write_table(
    args.out_dir / "T2_SNOW_sorted_by_AD_SNOW.csv",
    t2_rows,
    [
        "layer_id",
        "layer_type",
        "layer_params",
        "params_percent",
        "snow_baseline_map50_95",
        "snow_pruned_map50_95",
        "ad_snow_map50_95",
        "amount",
    ],
)

write_table(
    args.out_dir / "T3_SW_prunability_sorted_by_SWP.csv",
    t3_rows,
    [
        "layer_id",
        "layer_type",
        "layer_params",
        "params_percent",
        "ad_gen_map50_95",
        "ad_snow_map50_95",
        "combined_sensitivity",
        "swp_score",
        "amount",
    ],
)

print("Done.")
print(f"Output folder: {args.out_dir.resolve()}")
print("Main outputs:")
print("  baseline_GEN_SNOW_metrics.csv")
print("  layer_metadata.csv")
print("  raw_GEN_SNOW_individual_layer_pruning.csv")
print("  T1_GEN_sorted_by_AD_GEN.csv")
print("  T2_SNOW_sorted_by_AD_SNOW.csv")
print("  T3_SW_prunability_sorted_by_SWP.csv")
print("  structured_pruning_details_by_layer.csv")
