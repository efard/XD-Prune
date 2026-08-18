from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
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


def validate_model(
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
        plots=False,
    )
    return summarize(results)


def zero_like_output(output):
    # This removes the information produced by the selected layer
    # while preserving tensor shape so the YOLO graph can continue.
    if torch.is_tensor(output):
        return torch.zeros_like(output)

    if isinstance(output, tuple):
        return tuple(zero_like_output(item) for item in output)

    if isinstance(output, list):
        return [zero_like_output(item) for item in output]

    raise TypeError(f"Unsupported output type for zero ablation: {type(output)}")


def same_shape(a, b) -> bool:
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape

    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(same_shape(x, y) for x, y in zip(a, b))

    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(same_shape(x, y) for x, y in zip(a, b))

    return False


def make_hook(mode: str):
    def hook(module, inputs, output):
        if mode == "zero":
            return zero_like_output(output)

        if mode == "bypass":
            # True bypass means the layer output is replaced by its input.
            # This only works when input and output have exactly the same shape.
            if len(inputs) != 1:
                raise RuntimeError("Bypass failed: selected layer has multiple inputs.")

            candidate = inputs[0]

            if not same_shape(candidate, output):
                raise RuntimeError(
                    f"Bypass failed: input shape and output shape do not match. "
                    f"input={getattr(candidate, 'shape', type(candidate))}, "
                    f"output={getattr(output, 'shape', type(output))}"
                )

            return candidate

        raise ValueError(f"Unknown mode: {mode}")

    return hook


parser = argparse.ArgumentParser(description="Whole-layer deletion/ablation sensitivity for GEN and SNOW.")
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--gen-data", required=True, type=Path)
parser.add_argument("--snow-data", required=True, type=Path)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--target-layers", required=True)
parser.add_argument("--mode", default="zero", choices=["zero", "bypass"])
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

args.out_dir.mkdir(parents=True, exist_ok=True)
args.project.mkdir(parents=True, exist_ok=True)

target_layers = [int(x.strip()) for x in args.target_layers.split(",") if x.strip()]

print("Loading baseline model for metadata...")
baseline_yolo = YOLO(str(args.model))
layers = baseline_yolo.model.model
total_params = sum(p.numel() for p in baseline_yolo.model.parameters())

metadata = {}
for layer_id in target_layers:
    if layer_id < 0 or layer_id >= len(layers):
        raise ValueError(f"Layer {layer_id} is outside model layer range.")

    layer = layers[layer_id]
    layer_type = layer.__class__.__name__

    if layer_type == "Detect":
        raise ValueError("Detect layer is excluded from this whole-layer deletion experiment.")

    layer_params = sum(p.numel() for p in layer.parameters())
    metadata[layer_id] = {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "layer_params": layer_params,
        "params_percent": layer_params / total_params * 100.0,
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
    f"baseline_GEN_{args.mode}",
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
    f"baseline_SNOW_{args.mode}",
)

baseline_rows = [
    {"dataset": "GEN", "split": args.split, "mode": args.mode, **gen_base},
    {"dataset": "SNOW", "split": args.split, "mode": args.mode, **snow_base},
]

with (args.out_dir / f"baseline_GEN_SNOW_{args.mode}.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(baseline_rows[0].keys()))
    writer.writeheader()
    writer.writerows(baseline_rows)

with (args.out_dir / f"layer_metadata_{args.mode}.csv").open("w", newline="", encoding="utf-8") as f:
    fields = ["layer_id", "layer_type", "layer_params", "params_percent"]
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for layer_id in target_layers:
        row = dict(metadata[layer_id])
        row["params_percent"] = f"{row['params_percent']:.6f}"
        writer.writerow(row)

raw_rows = []

for layer_id in target_layers:
    meta = metadata[layer_id]
    print(f"Testing layer {layer_id} ({meta['layer_type']}) with mode={args.mode}")

    yolo = YOLO(str(args.model))
    target_layer = yolo.model.model[layer_id]

    handle = target_layer.register_forward_hook(make_hook(args.mode))

    row = {
        "layer_id": layer_id,
        "layer_type": meta["layer_type"],
        "mode": args.mode,
        "total_model_params": total_params,
        "layer_params": meta["layer_params"],
        "params_percent": f"{meta['params_percent']:.6f}",
        "status": "ok",
        "error": "",
        "gen_baseline_map50_95": f"{gen_base['map50_95']:.6f}",
        "snow_baseline_map50_95": f"{snow_base['map50_95']:.6f}",
    }

    try:
        gen_deleted = validate_model(
            yolo,
            args.gen_data,
            args.split,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.project,
            f"L{layer_id}_GEN_{args.mode}",
        )

        snow_deleted = validate_model(
            yolo,
            args.snow_data,
            args.split,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.project,
            f"L{layer_id}_SNOW_{args.mode}",
        )

        ad_gen = gen_base["map50_95"] - gen_deleted["map50_95"]
        ad_snow = snow_base["map50_95"] - snow_deleted["map50_95"]
        sensitivity = 0.5 * ad_gen + 0.5 * ad_snow

        # Whole-layer deletion prunability:
        # high parameter percentage and low combined accuracy drop is better.
        deletion_score = meta["params_percent"] / max(sensitivity, args.epsilon)

        row.update(
            {
                "gen_deleted_map50_95": f"{gen_deleted['map50_95']:.6f}",
                "ad_gen_map50_95": f"{ad_gen:.6f}",
                "snow_deleted_map50_95": f"{snow_deleted['map50_95']:.6f}",
                "ad_snow_map50_95": f"{ad_snow:.6f}",
                "combined_sensitivity": f"{sensitivity:.6f}",
                "deletion_score": f"{deletion_score:.6f}",
                "gen_latency_ms": f"{gen_deleted['latency_ms']:.6f}",
                "snow_latency_ms": f"{snow_deleted['latency_ms']:.6f}",
                "gen_fps": f"{gen_deleted['fps']:.6f}",
                "snow_fps": f"{snow_deleted['fps']:.6f}",
            }
        )

    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": str(exc).replace("\n", " ")[:500],
                "gen_deleted_map50_95": "",
                "ad_gen_map50_95": "",
                "snow_deleted_map50_95": "",
                "ad_snow_map50_95": "",
                "combined_sensitivity": "",
                "deletion_score": "",
                "gen_latency_ms": "",
                "snow_latency_ms": "",
                "gen_fps": "",
                "snow_fps": "",
            }
        )

    finally:
        handle.remove()
        raw_rows.append(row)
        del yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

raw_path = args.out_dir / f"raw_GEN_SNOW_layer_deletion_{args.mode}.csv"
with raw_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
    writer.writeheader()
    writer.writerows(raw_rows)

ok_rows = [r for r in raw_rows if r["status"] == "ok"]

if ok_rows:
    t1_rows = sorted(ok_rows, key=lambda r: float(r["ad_gen_map50_95"]))
    t2_rows = sorted(ok_rows, key=lambda r: float(r["ad_snow_map50_95"]))
    t3_rows = sorted(ok_rows, key=lambda r: float(r["deletion_score"]), reverse=True)

    def write_table(path: Path, rows: list[dict], fields: list[str]):
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r[k] for k in fields})

    write_table(
        args.out_dir / f"T1_GEN_layer_deletion_sorted_by_AD_GEN_{args.mode}.csv",
        t1_rows,
        [
            "layer_id",
            "layer_type",
            "layer_params",
            "params_percent",
            "gen_baseline_map50_95",
            "gen_deleted_map50_95",
            "ad_gen_map50_95",
            "mode",
        ],
    )

    write_table(
        args.out_dir / f"T2_SNOW_layer_deletion_sorted_by_AD_SNOW_{args.mode}.csv",
        t2_rows,
        [
            "layer_id",
            "layer_type",
            "layer_params",
            "params_percent",
            "snow_baseline_map50_95",
            "snow_deleted_map50_95",
            "ad_snow_map50_95",
            "mode",
        ],
    )

    write_table(
        args.out_dir / f"T3_layer_deletion_prunability_sorted_{args.mode}.csv",
        t3_rows,
        [
            "layer_id",
            "layer_type",
            "layer_params",
            "params_percent",
            "ad_gen_map50_95",
            "ad_snow_map50_95",
            "combined_sensitivity",
            "deletion_score",
            "mode",
        ],
    )

print("Done.")
print(f"Output folder: {args.out_dir.resolve()}")
print(f"Raw result: {raw_path.resolve()}")
