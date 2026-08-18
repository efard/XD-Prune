from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


class NoParamLayerAdapter(nn.Module):
    """
    Replace a whole YOLO layer with a no-parameter adapter.

    This removes the selected layer's trainable parameters from the saved model while keeping
    the graph runnable by matching the original layer output shape.
    """

    def __init__(
        self,
        target_channels: int,
        target_hw: tuple[int, int],
        multi_input_policy: str = "concat",
    ):
        super().__init__()
        self.target_channels = int(target_channels)
        self.target_hw = tuple(target_hw)
        self.multi_input_policy = multi_input_policy

    def _collect_tensors(self, value: Any) -> list[torch.Tensor]:
        if torch.is_tensor(value):
            return [value]

        if isinstance(value, (tuple, list)):
            tensors = []
            for item in value:
                tensors.extend(self._collect_tensors(item))
            return tensors

        return []

    def _match_spatial(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) == self.target_hw:
            return x
        return F.interpolate(x, size=self.target_hw, mode="nearest")

    def _match_channels(self, x: torch.Tensor) -> torch.Tensor:
        current_channels = x.shape[1]

        if current_channels == self.target_channels:
            return x

        if current_channels > self.target_channels:
            return x[:, : self.target_channels, :, :]

        pad_channels = self.target_channels - current_channels
        padding = x.new_zeros((x.shape[0], pad_channels, x.shape[2], x.shape[3]))
        return torch.cat((x, padding), dim=1)

    def forward(self, x):
        tensors = self._collect_tensors(x)

        if not tensors:
            raise RuntimeError("NoParamLayerAdapter received no tensor input.")

        if self.multi_input_policy == "first" or len(tensors) == 1:
            out = tensors[0]
            out = self._match_spatial(out)
            out = self._match_channels(out)
            return out

        if self.multi_input_policy == "concat":
            resized = [self._match_spatial(t) for t in tensors]
            out = torch.cat(resized, dim=1)
            out = self._match_channels(out)
            return out

        raise ValueError(f"Unknown multi_input_policy: {self.multi_input_policy}")


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

        # Important for physical layer replacement:
        # use fixed square validation images so the adapter shape captured
        # from the 640x640 dummy forward pass matches real validation batches.
        rect=False,
    )
    return summarize(results)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def shape_text(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, tuple):
        return "(" + ", ".join(shape_text(v) for v in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(shape_text(v) for v in value) + "]"
    return type(value).__name__


def capture_layer_io_shapes(model: nn.Module, layer_id: int, imgsz: int, device: str):
    """
    Run one dummy forward pass and capture the selected layer's input/output shapes.
    """
    hook_state = {}

    def hook(module, inputs, output):
        hook_state["input_shapes"] = shape_text(inputs)
        hook_state["output_shape"] = shape_text(output)

        if not torch.is_tensor(output):
            raise RuntimeError(
                f"Layer {layer_id} output is not a single tensor. Output={shape_text(output)}"
            )

        if output.ndim != 4:
            raise RuntimeError(
                f"Layer {layer_id} output is not 4D NCHW. Output shape={tuple(output.shape)}"
            )

        hook_state["target_channels"] = int(output.shape[1])
        hook_state["target_hw"] = (int(output.shape[2]), int(output.shape[3]))

    handle = model.model[layer_id].register_forward_hook(hook)

    try:
        model.eval()
        dummy = torch.randn(1, 3, imgsz, imgsz, device=next(model.parameters()).device)
        with torch.no_grad():
            model(dummy)
    finally:
        handle.remove()

    return hook_state


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None):
    if not rows:
        return

    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            filtered_row = {key: row.get(key, "") for key in fieldnames}
            writer.writerow(filtered_row)


parser = argparse.ArgumentParser(
    description="Physical whole-layer replacement for real parameter/model-size reduction."
)
parser.add_argument("--model", required=True, type=Path)
parser.add_argument("--gen-data", required=True, type=Path)
parser.add_argument("--snow-data", required=True, type=Path)
parser.add_argument("--split", default="val", choices=["val", "test"])
parser.add_argument("--target-layers", required=True, help='Comma-separated layer IDs or "all".')
parser.add_argument("--imgsz", default=640, type=int)
parser.add_argument("--batch", default=16, type=int)
parser.add_argument("--device", default="0")
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--project", required=True, type=Path)
parser.add_argument("--out-dir", required=True, type=Path)
parser.add_argument("--multi-input-policy", default="concat", choices=["concat", "first"])
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

models_dir = args.out_dir / "saved_layer_replaced_models"
models_dir.mkdir(parents=True, exist_ok=True)

print("Loading baseline model...")
baseline_yolo = YOLO(str(args.model))
baseline_yolo.model.eval()

baseline_params = count_params(baseline_yolo.model)

baseline_saved_path = args.out_dir / "baseline_resaved.pt"
baseline_yolo.save(str(baseline_saved_path))
baseline_size_mb = file_size_mb(baseline_saved_path)

layers = baseline_yolo.model.model

if args.target_layers.strip().lower() == "all":
    target_layers = [
        layer_id
        for layer_id, layer in enumerate(layers)
        if layer.__class__.__name__ != "Detect"
    ]
else:
    target_layers = [int(x.strip()) for x in args.target_layers.split(",") if x.strip()]

metadata_rows = []
for layer_id in target_layers:
    layer = layers[layer_id]
    layer_type = layer.__class__.__name__
    layer_params = count_params(layer)

    metadata_rows.append(
        {
            "layer_id": layer_id,
            "layer_type": layer_type,
            "layer_params": layer_params,
            "layer_params_percent": f"{layer_params / baseline_params * 100.0:.6f}",
        }
    )

write_csv(args.out_dir / "layer_metadata_physical_layer_replacement.csv", metadata_rows)

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
    "baseline_GEN_layer_replacement",
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
    "baseline_SNOW_layer_replacement",
)

baseline_rows = [
    {
        "dataset": "GEN",
        "split": args.split,
        "baseline_params": baseline_params,
        "baseline_size_mb": f"{baseline_size_mb:.6f}",
        **gen_base,
    },
    {
        "dataset": "SNOW",
        "split": args.split,
        "baseline_params": baseline_params,
        "baseline_size_mb": f"{baseline_size_mb:.6f}",
        **snow_base,
    },
]
write_csv(args.out_dir / "baseline_GEN_SNOW_physical_layer_replacement.csv", baseline_rows)

raw_rows = []

for meta in metadata_rows:
    layer_id = int(meta["layer_id"])
    layer_type = meta["layer_type"]

    print(f"Testing physical whole-layer replacement for layer {layer_id} ({layer_type})")

    row = {
        "layer_id": layer_id,
        "layer_type": layer_type,
        "status": "ok",
        "error": "",
        "baseline_params": baseline_params,
        "baseline_size_mb": f"{baseline_size_mb:.6f}",
        "layer_params": meta["layer_params"],
        "layer_params_percent": meta["layer_params_percent"],
        "gen_baseline_map50_95": f"{gen_base['map50_95']:.6f}",
        "snow_baseline_map50_95": f"{snow_base['map50_95']:.6f}",
        "input_shapes": "",
        "output_shape": "",
        "adapter": "",
    }

    try:
        yolo = YOLO(str(args.model))
        yolo.model.eval()

        io_info = capture_layer_io_shapes(
            model=yolo.model,
            layer_id=layer_id,
            imgsz=args.imgsz,
            device=args.device,
        )

        row["input_shapes"] = io_info["input_shapes"]
        row["output_shape"] = io_info["output_shape"]

        target_channels = io_info["target_channels"]
        target_hw = io_info["target_hw"]

        original_layer_params = count_params(yolo.model.model[layer_id])

        old_layer = yolo.model.model[layer_id]
        
        adapter = NoParamLayerAdapter(
            target_channels=target_channels,
            target_hw=target_hw,
            multi_input_policy=args.multi_input_policy,
        )
        
        # Ultralytics YOLO needs these top-level graph attributes during forward validation.
        # Without .f, .i, .type, or .np, the model may save but fail during validation.
        if hasattr(old_layer, "f"):
            adapter.f = old_layer.f
        else:
            adapter.f = -1
        
        if hasattr(old_layer, "i"):
            adapter.i = old_layer.i
        else:
            adapter.i = layer_id
        
        adapter.type = f"NoParamLayerAdapter_for_{layer_type}"
        adapter.np = 0
        
        # Put adapter on the same device as the model.
        adapter = adapter.to(next(yolo.model.parameters()).device)
        
        yolo.model.model[layer_id] = adapter
        
        row["adapter"] = (
            f"NoParamLayerAdapter(target_channels={target_channels}, "
            f"target_hw={target_hw}, multi_input_policy={args.multi_input_policy}, "
            f"f={adapter.f}, i={adapter.i})"
        )

        replaced_params = count_params(yolo.model)
        params_reduced = baseline_params - replaced_params
        params_reduction_percent = params_reduced / baseline_params * 100.0

        save_path = models_dir / f"replace_L{layer_id}_{layer_type}.pt"
        yolo.save(str(save_path))

        replaced_size_mb = file_size_mb(save_path)
        size_reduced_mb = baseline_size_mb - replaced_size_mb
        size_reduction_percent = size_reduced_mb / baseline_size_mb * 100.0

        gen_result = validate_model(
            yolo,
            args.gen_data,
            args.split,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.project,
            f"L{layer_id}_GEN_layer_replacement",
        )

        snow_result = validate_model(
            yolo,
            args.snow_data,
            args.split,
            args.imgsz,
            args.batch,
            args.device,
            args.workers,
            args.project,
            f"L{layer_id}_SNOW_layer_replacement",
        )

        ad_gen = gen_base["map50_95"] - gen_result["map50_95"]
        ad_snow = snow_base["map50_95"] - snow_result["map50_95"]
        combined_sensitivity = 0.5 * ad_gen + 0.5 * ad_snow

        physical_layer_score = params_reduction_percent / max(combined_sensitivity, args.epsilon)

        row.update(
            {
                "replaced_model_path": str(save_path),
                "original_layer_params_removed": original_layer_params,
                "params_after": replaced_params,
                "params_reduced": params_reduced,
                "params_reduction_percent": f"{params_reduction_percent:.6f}",
                "replaced_size_mb": f"{replaced_size_mb:.6f}",
                "size_reduced_mb": f"{size_reduced_mb:.6f}",
                "size_reduction_percent": f"{size_reduction_percent:.6f}",
                "gen_replaced_map50_95": f"{gen_result['map50_95']:.6f}",
                "ad_gen_map50_95": f"{ad_gen:.6f}",
                "snow_replaced_map50_95": f"{snow_result['map50_95']:.6f}",
                "ad_snow_map50_95": f"{ad_snow:.6f}",
                "combined_sensitivity": f"{combined_sensitivity:.6f}",
                "physical_layer_score": f"{physical_layer_score:.6f}",
                "gen_latency_ms": f"{gen_result['latency_ms']:.6f}",
                "snow_latency_ms": f"{snow_result['latency_ms']:.6f}",
                "gen_fps": f"{gen_result['fps']:.6f}",
                "snow_fps": f"{snow_result['fps']:.6f}",
            }
        )

    except Exception as exc:
        row.update(
            {
                "status": "failed",
                "error": str(exc).replace("\n", " ")[:500],
                "replaced_model_path": "",
                "original_layer_params_removed": "",
                "params_after": "",
                "params_reduced": "",
                "params_reduction_percent": "",
                "replaced_size_mb": "",
                "size_reduced_mb": "",
                "size_reduction_percent": "",
                "gen_replaced_map50_95": "",
                "ad_gen_map50_95": "",
                "snow_replaced_map50_95": "",
                "ad_snow_map50_95": "",
                "combined_sensitivity": "",
                "physical_layer_score": "",
                "gen_latency_ms": "",
                "snow_latency_ms": "",
                "gen_fps": "",
                "snow_fps": "",
            }
        )

    finally:
        raw_rows.append(row)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

write_csv(args.out_dir / "raw_GEN_SNOW_physical_layer_replacement.csv", raw_rows)

ok_rows = [r for r in raw_rows if r["status"] == "ok"]

if ok_rows:
    t1_rows = sorted(ok_rows, key=lambda r: float(r["ad_gen_map50_95"]))
    t2_rows = sorted(ok_rows, key=lambda r: float(r["ad_snow_map50_95"]))
    t3_rows = sorted(ok_rows, key=lambda r: float(r["physical_layer_score"]), reverse=True)

    t1_fields = [
        "layer_id",
        "layer_type",
        "layer_params",
        "layer_params_percent",
        "params_reduced",
        "params_reduction_percent",
        "size_reduction_percent",
        "gen_baseline_map50_95",
        "gen_replaced_map50_95",
        "ad_gen_map50_95",
        "replaced_model_path",
    ]

    t2_fields = [
        "layer_id",
        "layer_type",
        "layer_params",
        "layer_params_percent",
        "params_reduced",
        "params_reduction_percent",
        "size_reduction_percent",
        "snow_baseline_map50_95",
        "snow_replaced_map50_95",
        "ad_snow_map50_95",
        "replaced_model_path",
    ]

    t3_fields = [
        "layer_id",
        "layer_type",
        "layer_params",
        "layer_params_percent",
        "params_reduced",
        "params_reduction_percent",
        "size_reduction_percent",
        "ad_gen_map50_95",
        "ad_snow_map50_95",
        "combined_sensitivity",
        "physical_layer_score",
        "replaced_model_path",
    ]

    write_csv(args.out_dir / "T1_GEN_physical_layer_replacement_sorted_by_AD_GEN.csv", t1_rows, t1_fields)
    write_csv(args.out_dir / "T2_SNOW_physical_layer_replacement_sorted_by_AD_SNOW.csv", t2_rows, t2_fields)
    write_csv(args.out_dir / "T3_physical_layer_replacement_SWPrunability_sorted.csv", t3_rows, t3_fields)

print("Done.")
print(f"Output folder: {args.out_dir.resolve()}")
print("Important outputs:")
print("  baseline_GEN_SNOW_physical_layer_replacement.csv")
print("  layer_metadata_physical_layer_replacement.csv")
print("  raw_GEN_SNOW_physical_layer_replacement.csv")
print("  T1_GEN_physical_layer_replacement_sorted_by_AD_GEN.csv")
print("  T2_SNOW_physical_layer_replacement_sorted_by_AD_SNOW.csv")
print("  T3_physical_layer_replacement_SWPrunability_sorted.csv")
print("  saved_layer_replaced_models/")
