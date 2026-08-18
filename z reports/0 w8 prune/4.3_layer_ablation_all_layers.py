from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from ultralytics import YOLO


def summarize(results) -> dict[str, float]:
    """Convert Ultralytics validation results into the metrics used by this experiment."""
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
    """Run one GEN or SNOW validation pass."""
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


def zero_like_output(output: Any) -> Any:
    """Replace all tensors in a layer output with zeros while preserving structure and shape."""
    if torch.is_tensor(output):
        return torch.zeros_like(output)

    if isinstance(output, tuple):
        return tuple(zero_like_output(item) for item in output)

    if isinstance(output, list):
        return [zero_like_output(item) for item in output]

    raise TypeError(f"Unsupported output type for zero ablation: {type(output)}")


def same_shape(a: Any, b: Any) -> bool:
    """Check whether nested tensor structures have identical shapes."""
    if torch.is_tensor(a) and torch.is_tensor(b):
        return a.shape == b.shape

    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return all(same_shape(x, y) for x, y in zip(a, b))

    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return all(same_shape(x, y) for x, y in zip(a, b))

    return False


def collect_tensors(value: Any) -> list[torch.Tensor]:
    """Recursively collect tensor inputs, including inputs supplied as lists to Concat-like layers."""
    if torch.is_tensor(value):
        return [value]

    if isinstance(value, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(collect_tensors(item))
        return tensors

    return []


def shape_text(value: Any) -> str:
    """Create compact shape text for the CSV audit trail."""
    if torch.is_tensor(value):
        return str(tuple(value.shape))

    if isinstance(value, tuple):
        return "(" + ", ".join(shape_text(item) for item in value) + ")"

    if isinstance(value, list):
        return "[" + ", ".join(shape_text(item) for item in value) + "]"

    return type(value).__name__


def resize_spatial(source: torch.Tensor, target_hw: tuple[int, int]) -> tuple[torch.Tensor, str]:
    """
    Match H and W with nearest-neighbour interpolation.

    Nearest-neighbour is deliberately used because it adds no learned parameters and does not
    silently apply a trained projection. This is an ablation adapter, not a replacement layer.
    """
    source_hw = tuple(source.shape[-2:])
    if source_hw == target_hw:
        return source, "spatial_unchanged"

    resized = F.interpolate(source, size=target_hw, mode="nearest")
    return resized, f"spatial_resize_{source_hw}_to_{target_hw}"


def match_channels_crop_pad(
    source: torch.Tensor,
    target_channels: int,
) -> tuple[torch.Tensor, str]:
    """
    Match the output channel count without learned weights.

    - Too many channels: keep the first target_channels channels.
    - Too few channels: append zero-valued channels.

    This policy is deterministic and explicit, but it is not mathematically equivalent to a
    learned 1x1 convolution. The resulting experiment must be reported as adapted bypass.
    """
    source_channels = source.shape[1]

    if source_channels == target_channels:
        return source, "channels_unchanged"

    if source_channels > target_channels:
        return source[:, :target_channels, ...], (
            f"channels_crop_{source_channels}_to_{target_channels}"
        )

    pad_channels = target_channels - source_channels
    padding = source.new_zeros(
        (source.shape[0], pad_channels, *source.shape[2:])
    )
    return torch.cat((source, padding), dim=1), (
        f"channels_zero_pad_{source_channels}_to_{target_channels}"
    )


def build_source_tensor(
    inputs: tuple[Any, ...],
    target: torch.Tensor,
    multi_input_policy: str,
) -> tuple[torch.Tensor, list[str]]:
    """
    Build one source tensor from a layer's incoming tensor(s).

    For a multi-input layer, "concat" preserves all incoming branches by resizing them to the
    target spatial size and concatenating them along the channel dimension. For an actual Concat
    layer this may reproduce its normal output, so such a result is not evidence that the layer is
    removable; Concat normally has no trainable parameters to prune.
    """
    sources = collect_tensors(inputs)
    if not sources:
        raise RuntimeError("Adapted bypass failed: no tensor input was found.")

    if target.ndim != 4:
        raise RuntimeError(
            f"Adapted bypass currently supports 4D NCHW outputs only; got {target.ndim}D."
        )

    target_batch = target.shape[0]
    target_hw = tuple(target.shape[-2:])

    for source in sources:
        if source.ndim != 4:
            raise RuntimeError(
                f"Adapted bypass currently supports 4D NCHW inputs only; got {source.ndim}D."
            )
        if source.shape[0] != target_batch:
            raise RuntimeError(
                "Adapted bypass failed: input and output batch dimensions do not match."
            )

    actions = [f"input_tensors={len(sources)}", f"multi_input_policy={multi_input_policy}"]

    if len(sources) == 1:
        return sources[0], actions

    if multi_input_policy == "first":
        actions.append("selected_first_input")
        return sources[0], actions

    if multi_input_policy == "concat":
        resized_sources: list[torch.Tensor] = []
        for index, source in enumerate(sources):
            resized, action = resize_spatial(source, target_hw)
            resized_sources.append(resized)
            actions.append(f"source_{index}_{action}")

        actions.append("concatenated_inputs_along_channels")
        return torch.cat(resized_sources, dim=1), actions

    raise ValueError(f"Unknown multi-input policy: {multi_input_policy}")


def adapted_bypass_output(
    inputs: tuple[Any, ...],
    output: Any,
    multi_input_policy: str,
    hook_state: dict[str, str],
) -> Any:
    """Replace a layer output with shape-matched incoming information."""
    if torch.is_tensor(output):
        source, actions = build_source_tensor(inputs, output, multi_input_policy)
        source, spatial_action = resize_spatial(source, tuple(output.shape[-2:]))
        source, channel_action = match_channels_crop_pad(source, output.shape[1])

        if source.shape != output.shape:
            raise RuntimeError(
                f"Adapted bypass produced the wrong shape: {tuple(source.shape)} != {tuple(output.shape)}"
            )

        # Record the first batch only; subsequent validation batches use the same model geometry.
        if "adapter_details" not in hook_state:
            actions.extend((spatial_action, channel_action))
            hook_state["input_shapes"] = shape_text(inputs)
            hook_state["output_shape"] = shape_text(output)
            hook_state["adapter_details"] = "; ".join(actions)

        return source

    if isinstance(output, tuple):
        return tuple(
            adapted_bypass_output(inputs, item, multi_input_policy, hook_state)
            for item in output
        )

    if isinstance(output, list):
        return [
            adapted_bypass_output(inputs, item, multi_input_policy, hook_state)
            for item in output
        ]

    raise TypeError(f"Unsupported output type for adapted bypass: {type(output)}")


def make_hook(
    mode: str,
    multi_input_policy: str,
    hook_state: dict[str, str],
):
    """Create the selected whole-layer ablation hook."""
    def hook(module, inputs, output):
        if mode == "zero":
            return zero_like_output(output)

        if mode == "bypass":
            # Strict identity bypass: no resize, no channel crop, and no padding.
            if len(inputs) != 1:
                raise RuntimeError("Strict bypass failed: selected layer has multiple arguments.")

            candidate = inputs[0]
            if not same_shape(candidate, output):
                raise RuntimeError(
                    "Strict bypass failed: input and output shapes do not match. "
                    f"input={shape_text(candidate)}, output={shape_text(output)}"
                )

            if "adapter_details" not in hook_state:
                hook_state["input_shapes"] = shape_text(inputs)
                hook_state["output_shape"] = shape_text(output)
                hook_state["adapter_details"] = "strict_identity_no_shape_adapter"

            return candidate

        if mode == "adapted_bypass":
            return adapted_bypass_output(
                inputs=inputs,
                output=output,
                multi_input_policy=multi_input_policy,
                hook_state=hook_state,
            )

        raise ValueError(f"Unknown mode: {mode}")

    return hook


parser = argparse.ArgumentParser(
    description="Whole-layer zero, strict-bypass, or adapted-bypass sensitivity for GEN and SNOW."
)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--gen-data", required=True, type=Path)
parser.add_argument("--snow-data", required=True, type=Path)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument(
    "--target-layers",
    required=True,
    help='Comma-separated layer IDs, or "all" to test every non-Detect top-level layer.',
)
parser.add_argument(
    "--mode",
    default="zero",
    choices=["zero", "bypass", "adapted_bypass"],
)
parser.add_argument(
    "--multi-input-policy",
    default="concat",
    choices=["concat", "first"],
    help="How adapted_bypass handles a layer receiving multiple tensors.",
)
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

print("Loading baseline model for metadata...")
baseline_yolo = YOLO(str(args.model))
layers = baseline_yolo.model.model
total_params = sum(p.numel() for p in baseline_yolo.model.parameters())

if args.target_layers.strip().lower() == "all":
    target_layers = [
        layer_id
        for layer_id, layer in enumerate(layers)
        if layer.__class__.__name__ != "Detect"
    ]
else:
    target_layers = [
        int(value.strip())
        for value in args.target_layers.split(",")
        if value.strip()
    ]

if not target_layers:
    raise ValueError("No target layers were selected.")

metadata = {}
for layer_id in target_layers:
    if layer_id < 0 or layer_id >= len(layers):
        raise ValueError(f"Layer {layer_id} is outside model layer range.")

    layer = layers[layer_id]
    layer_type = layer.__class__.__name__

    if layer_type == "Detect":
        raise ValueError(
            "Detect layer is excluded because replacing detection-head outputs invalidates normal validation."
        )

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

with (args.out_dir / f"baseline_GEN_SNOW_{args.mode}.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=list(baseline_rows[0].keys()))
    writer.writeheader()
    writer.writerows(baseline_rows)

with (args.out_dir / f"layer_metadata_{args.mode}.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    fields = ["layer_id", "layer_type", "layer_params", "params_percent"]
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    for layer_id in target_layers:
        row = dict(metadata[layer_id])
        row["params_percent"] = f"{row['params_percent']:.6f}"
        writer.writerow(row)

raw_rows = []

for layer_id in target_layers:
    meta = metadata[layer_id]
    print(f"Testing layer {layer_id} ({meta['layer_type']}) with mode={args.mode}")

    # Reload the untouched checkpoint so every layer is tested independently.
    yolo = YOLO(str(args.model))
    target_layer = yolo.model.model[layer_id]
    hook_state: dict[str, str] = {}

    handle = target_layer.register_forward_hook(
        make_hook(
            mode=args.mode,
            multi_input_policy=args.multi_input_policy,
            hook_state=hook_state,
        )
    )

    row = {
        "layer_id": layer_id,
        "layer_type": meta["layer_type"],
        "mode": args.mode,
        "multi_input_policy": args.multi_input_policy if args.mode == "adapted_bypass" else "",
        "total_model_params": total_params,
        "layer_params": meta["layer_params"],
        "params_percent": f"{meta['params_percent']:.6f}",
        "status": "ok",
        "error": "",
        "input_shapes": "",
        "output_shape": "",
        "adapter_details": "",
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

        # This score ranks ablation tolerance, not guaranteed physical deletion speed-up.
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
        row["input_shapes"] = hook_state.get("input_shapes", "")
        row["output_shape"] = hook_state.get("output_shape", "")
        row["adapter_details"] = hook_state.get("adapter_details", "")
        raw_rows.append(row)
        del yolo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

raw_path = args.out_dir / f"raw_GEN_SNOW_layer_deletion_{args.mode}.csv"
with raw_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(raw_rows[0].keys()))
    writer.writeheader()
    writer.writerows(raw_rows)

ok_rows = [row for row in raw_rows if row["status"] == "ok"]

if ok_rows:
    t1_rows = sorted(ok_rows, key=lambda row: float(row["ad_gen_map50_95"]))
    t2_rows = sorted(ok_rows, key=lambda row: float(row["ad_snow_map50_95"]))
    t3_rows = sorted(ok_rows, key=lambda row: float(row["deletion_score"]), reverse=True)

    def write_table(path: Path, rows: list[dict], fields: list[str]):
        """Write a selected, consistently ordered subset of result columns."""
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for result_row in rows:
                writer.writerow({key: result_row[key] for key in fields})

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
            "adapter_details",
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
            "adapter_details",
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
            "adapter_details",
        ],
    )

print("Done.")
print(f"Output folder: {args.out_dir.resolve()}")
print(f"Raw result: {raw_path.resolve()}")
