from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from ultralytics import YOLO

from layer_replacement_adapter import DynamicNoParamLayerAdapter, reduced_fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_params(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def file_size_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def safe_metric(value: Any, default: float = float("nan")) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def summarize(results: Any) -> dict[str, float]:
    speed = getattr(results, "speed", {}) or {}
    preprocess = safe_metric(speed.get("preprocess", 0.0), 0.0)
    inference = safe_metric(speed.get("inference", 0.0), 0.0)
    postprocess = safe_metric(speed.get("postprocess", 0.0), 0.0)
    latency = preprocess + inference + postprocess
    box = results.box
    return {
        "map50_95": safe_metric(box.map),
        "map50": safe_metric(box.map50),
        "map75": safe_metric(box.map75),
        "precision": safe_metric(getattr(box, "mp", float("nan"))),
        "recall": safe_metric(getattr(box, "mr", float("nan"))),
        "latency_ms": latency,
        "fps": 0.0 if latency <= 0 else 1000.0 / latency,
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
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        half=False,
        dnn=False,
        augment=False,
        agnostic_nms=False,
        plots=False,
        save_json=False,
        verbose=False,
    )
    return summarize(results)


def torch_device(device: str) -> torch.device:
    if device.lower() == "cpu":
        return torch.device("cpu")
    return torch.device(f"cuda:{device.split(',')[0].strip()}")


def shape_text(value: Any) -> str:
    if torch.is_tensor(value):
        return str(tuple(value.shape))
    if isinstance(value, tuple):
        return "(" + ", ".join(shape_text(item) for item in value) + ")"
    if isinstance(value, list):
        return "[" + ", ".join(shape_text(item) for item in value) + "]"
    return type(value).__name__


def collect_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        tensors: list[torch.Tensor] = []
        for item in value:
            tensors.extend(collect_tensors(item))
        return tensors
    return []


def capture_interface(model: nn.Module, layer_id: int, imgsz: int) -> dict[str, Any]:
    state: dict[str, Any] = {}

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        sources = collect_tensors(inputs)
        if not sources:
            raise RuntimeError(f"Layer {layer_id} has no tensor input")
        if not torch.is_tensor(output) or output.ndim != 4:
            raise RuntimeError(
                f"Layer {layer_id} must have one 4D tensor output; got {shape_text(output)}"
            )
        reference = sources[0]
        if reference.ndim != 4:
            raise RuntimeError(f"Layer {layer_id} reference input is not 4D: {tuple(reference.shape)}")
        h_num, h_den = reduced_fraction(output.shape[-2], reference.shape[-2])
        w_num, w_den = reduced_fraction(output.shape[-1], reference.shape[-1])
        state.update(
            {
                "input_shapes": shape_text(inputs),
                "output_shape": shape_text(output),
                "input_tensor_count": len(sources),
                "target_channels": int(output.shape[1]),
                "h_num": h_num,
                "h_den": h_den,
                "w_num": w_num,
                "w_den": w_den,
            }
        )

    handle = model.model[layer_id].register_forward_hook(hook)
    try:
        model.eval()
        device = next(model.parameters()).device
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)
        with torch.no_grad():
            model(dummy)
    finally:
        handle.remove()
    if not state:
        raise RuntimeError(f"Failed to capture layer {layer_id} interface")
    return state


def replace_layer(
    yolo: YOLO,
    layer_id: int,
    imgsz: int,
    device: torch.device,
    multi_input_policy: str,
) -> dict[str, Any]:
    yolo.model.to(device).eval()
    old_layer = yolo.model.model[layer_id]
    interface = capture_interface(yolo.model, layer_id, imgsz)
    original_layer_params = count_params(old_layer)

    adapter = DynamicNoParamLayerAdapter(
        target_channels=interface["target_channels"],
        spatial_h_num=interface["h_num"],
        spatial_h_den=interface["h_den"],
        spatial_w_num=interface["w_num"],
        spatial_w_den=interface["w_den"],
        multi_input_policy=multi_input_policy,
    ).to(device)

    adapter.f = getattr(old_layer, "f", -1)
    adapter.i = getattr(old_layer, "i", layer_id)
    adapter.type = f"DynamicNoParamLayerAdapter_for_{old_layer.__class__.__name__}"
    adapter.np = 0
    yolo.model.model[layer_id] = adapter

    return {
        **interface,
        "old_layer_type": old_layer.__class__.__name__,
        "original_layer_params": original_layer_params,
        "adapter_repr": repr(adapter),
        "graph_f": adapter.f,
        "graph_i": adapter.i,
    }


def parse_stage1(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["domain"].upper(): row for row in rows}
    missing = {"GEN", "SNOW"} - set(result)
    if missing:
        raise RuntimeError(f"Stage-1 CSV is missing domains: {sorted(missing)}")
    return result


def parse_targets(text: str, layers: nn.ModuleList) -> list[int]:
    if text.strip().lower() == "all":
        return [i for i, layer in enumerate(layers) if layer.__class__.__name__ != "Detect"]
    values = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("No target layers selected")
    return values


def environment_record() -> dict[str, Any]:
    try:
        import torchvision
        torchvision_version = torchvision.__version__
    except Exception as exc:
        torchvision_version = f"unavailable: {exc}"
    try:
        import ultralytics
        ultralytics_version = ultralytics.__version__
    except Exception as exc:
        ultralytics_version = f"unavailable: {exc}"
    try:
        import torch_pruning
        torch_pruning_version = getattr(torch_pruning, "__version__", "unknown")
    except Exception as exc:
        torch_pruning_version = f"unavailable: {exc}"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "ultralytics": ultralytics_version,
        "torch_pruning": torch_pruning_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dual-checkpoint whole-layer replacement sweep using exact GEN and SNOW validation settings."
    )
    parser.add_argument("--gen-model", required=True, type=Path)
    parser.add_argument("--snow-model", required=True, type=Path)
    parser.add_argument("--gen-data", required=True, type=Path)
    parser.add_argument("--snow-data", required=True, type=Path)
    parser.add_argument("--stage1-csv", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--target-layers", default="all")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--multi-input-policy", default="concat", choices=["concat", "first"])
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--repeat-tolerance", default=0.002, type=float)
    parser.add_argument("--epsilon", default=1e-6, type=float)
    args = parser.parse_args()

    for path in [args.gen_model, args.snow_model, args.gen_data, args.snow_data, args.stage1_csv]:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "saved_replaced_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    evidence = environment_record()
    evidence["input_sha256"] = {
        "gen_model": sha256(args.gen_model),
        "snow_model": sha256(args.snow_model),
        "gen_data": sha256(args.gen_data),
        "snow_data": sha256(args.snow_data),
        "stage1_csv": sha256(args.stage1_csv),
        "script": sha256(Path(__file__).resolve()),
        "adapter_module": sha256(Path(__file__).with_name("layer_replacement_adapter.py")),
    }
    write_json(args.out_dir / "experiment_environment_and_checksums.json", evidence)

    stage1 = parse_stage1(args.stage1_csv)
    device = torch_device(args.device)

    print("Loading supplied baselines...")
    gen_baseline_yolo = YOLO(str(args.gen_model), task="detect")
    snow_baseline_yolo = YOLO(str(args.snow_model), task="detect")
    gen_baseline_params = count_params(gen_baseline_yolo.model)
    snow_baseline_params = count_params(snow_baseline_yolo.model)

    if len(gen_baseline_yolo.names) != 11:
        raise RuntimeError(f"GEN checkpoint has {len(gen_baseline_yolo.names)} classes, expected 11")
    if len(snow_baseline_yolo.names) != 8:
        raise RuntimeError(f"SNOW checkpoint has {len(snow_baseline_yolo.names)} classes, expected 8")

    gen_layers = gen_baseline_yolo.model.model
    snow_layers = snow_baseline_yolo.model.model
    if len(gen_layers) != len(snow_layers):
        raise RuntimeError(f"GEN/SNOW top-level layer counts differ: {len(gen_layers)} != {len(snow_layers)}")

    target_layers = parse_targets(args.target_layers, gen_layers)
    for layer_id in target_layers:
        if layer_id < 0 or layer_id >= len(gen_layers):
            raise ValueError(f"Layer ID out of range: {layer_id}")
        if gen_layers[layer_id].__class__.__name__ == "Detect":
            raise ValueError(f"Detect layer {layer_id} is protected")
        if gen_layers[layer_id].__class__.__name__ != snow_layers[layer_id].__class__.__name__:
            raise RuntimeError(
                f"Layer {layer_id} type differs: {gen_layers[layer_id].__class__.__name__} vs "
                f"{snow_layers[layer_id].__class__.__name__}"
            )

    metadata_rows: list[dict[str, Any]] = []
    for layer_id in target_layers:
        gen_layer_params = count_params(gen_layers[layer_id])
        snow_layer_params = count_params(snow_layers[layer_id])
        metadata_rows.append(
            {
                "layer_id": layer_id,
                "layer_type": gen_layers[layer_id].__class__.__name__,
                "gen_layer_params": gen_layer_params,
                "snow_layer_params": snow_layer_params,
                "gen_layer_params_percent": gen_layer_params / gen_baseline_params * 100.0,
                "snow_layer_params_percent": snow_layer_params / snow_baseline_params * 100.0,
                "candidate_for_size_reduction": gen_layer_params > 0 and snow_layer_params > 0,
            }
        )
    write_csv(args.out_dir / "layer_metadata_dual_domain.csv", metadata_rows)

    print("Evaluating in-run GEN baseline with frozen FP32 settings...")
    gen_base = validate_model(
        gen_baseline_yolo, args.gen_data, args.split, args.imgsz, args.batch,
        args.device, args.workers, args.project, "baseline_GEN_stage2"
    )
    print("Evaluating in-run SNOW baseline with frozen FP32 settings...")
    snow_base = validate_model(
        snow_baseline_yolo, args.snow_data, args.split, args.imgsz, args.batch,
        args.device, args.workers, args.project, "baseline_SNOW_stage2"
    )

    stage1_gen = float(stage1["GEN"]["map50_95"])
    stage1_snow = float(stage1["SNOW"]["map50_95"])
    repeat_rows = [
        {
            "domain": "GEN",
            "stage1_map50_95": stage1_gen,
            "stage2_map50_95": gen_base["map50_95"],
            "absolute_difference": abs(stage1_gen - gen_base["map50_95"]),
        },
        {
            "domain": "SNOW",
            "stage1_map50_95": stage1_snow,
            "stage2_map50_95": snow_base["map50_95"],
            "absolute_difference": abs(stage1_snow - snow_base["map50_95"]),
        },
    ]
    for row in repeat_rows:
        row["status"] = "PASS" if row["absolute_difference"] <= args.repeat_tolerance else "FAIL"
    write_csv(args.out_dir / "baseline_repeatability_gate.csv", repeat_rows)
    if any(row["status"] != "PASS" for row in repeat_rows):
        raise RuntimeError(
            f"Stage-2 baseline repeatability gate failed. See {args.out_dir / 'baseline_repeatability_gate.csv'}"
        )

    baseline_rows = [
        {
            "domain": "GEN",
            "classes": len(gen_baseline_yolo.names),
            "parameters": gen_baseline_params,
            "checkpoint_mib": file_size_mib(args.gen_model),
            **gen_base,
        },
        {
            "domain": "SNOW",
            "classes": len(snow_baseline_yolo.names),
            "parameters": snow_baseline_params,
            "checkpoint_mib": file_size_mib(args.snow_model),
            **snow_base,
        },
    ]
    write_csv(args.out_dir / "baseline_GEN_SNOW_stage2.csv", baseline_rows)

    # Baselines are no longer needed on GPU.
    del gen_baseline_yolo, snow_baseline_yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_rows: list[dict[str, Any]] = []
    for meta in metadata_rows:
        layer_id = int(meta["layer_id"])
        layer_type = str(meta["layer_type"])
        print(f"\n===== LAYER {layer_id} ({layer_type}) =====")

        row: dict[str, Any] = {
            **meta,
            "status": "ok",
            "error": "",
            "multi_input_policy": args.multi_input_policy,
            "gen_baseline_map50_95": gen_base["map50_95"],
            "snow_baseline_map50_95": snow_base["map50_95"],
        }

        if not bool(meta["candidate_for_size_reduction"]):
            row["status"] = "skipped_zero_parameter_layer"
            row["error"] = "Top-level layer has no trainable parameters; replacing it cannot reduce model parameters."
            raw_rows.append(row)
            write_csv(args.out_dir / "raw_dual_domain_layer_replacement.csv", raw_rows)
            continue

        gen_yolo: YOLO | None = None
        snow_yolo: YOLO | None = None
        try:
            gen_yolo = YOLO(str(args.gen_model), task="detect")
            snow_yolo = YOLO(str(args.snow_model), task="detect")

            gen_info = replace_layer(gen_yolo, layer_id, args.imgsz, device, args.multi_input_policy)
            snow_info = replace_layer(snow_yolo, layer_id, args.imgsz, device, args.multi_input_policy)

            gen_params_after = count_params(gen_yolo.model)
            snow_params_after = count_params(snow_yolo.model)
            gen_params_reduced = gen_baseline_params - gen_params_after
            snow_params_reduced = snow_baseline_params - snow_params_after
            gen_reduction_pct = gen_params_reduced / gen_baseline_params * 100.0
            snow_reduction_pct = snow_params_reduced / snow_baseline_params * 100.0
            mean_reduction_pct = 0.5 * (gen_reduction_pct + snow_reduction_pct)

            gen_save = models_dir / f"GEN_replace_L{layer_id}_{layer_type}.pt"
            snow_save = models_dir / f"SNOW_replace_L{layer_id}_{layer_type}.pt"
            gen_yolo.save(str(gen_save))
            snow_yolo.save(str(snow_save))

            # Verify that both saved checkpoints can be deserialized while the adapter module is importable.
            gen_reload = YOLO(str(gen_save), task="detect")
            snow_reload = YOLO(str(snow_save), task="detect")
            if count_params(gen_reload.model) != gen_params_after:
                raise RuntimeError("GEN saved-model reload parameter count mismatch")
            if count_params(snow_reload.model) != snow_params_after:
                raise RuntimeError("SNOW saved-model reload parameter count mismatch")
            del gen_reload, snow_reload

            print("Validating GEN replaced model...")
            gen_result = validate_model(
                gen_yolo, args.gen_data, args.split, args.imgsz, args.batch,
                args.device, args.workers, args.project, f"GEN_replace_L{layer_id}"
            )
            print("Validating SNOW replaced model...")
            snow_result = validate_model(
                snow_yolo, args.snow_data, args.split, args.imgsz, args.batch,
                args.device, args.workers, args.project, f"SNOW_replace_L{layer_id}"
            )

            ad_gen = gen_base["map50_95"] - gen_result["map50_95"]
            ad_snow = snow_base["map50_95"] - snow_result["map50_95"]
            nad_gen = ad_gen / gen_base["map50_95"]
            nad_snow = ad_snow / snow_base["map50_95"]
            combined_nad_signed = 0.5 * nad_gen + 0.5 * nad_snow
            score_denominator = max(combined_nad_signed, args.epsilon)
            score = mean_reduction_pct / score_denominator

            row.update(
                {
                    "gen_input_shapes": gen_info["input_shapes"],
                    "gen_output_shape": gen_info["output_shape"],
                    "snow_input_shapes": snow_info["input_shapes"],
                    "snow_output_shape": snow_info["output_shape"],
                    "spatial_scale": f"{gen_info['h_num']}/{gen_info['h_den']} x {gen_info['w_num']}/{gen_info['w_den']}",
                    "adapter_repr": gen_info["adapter_repr"],
                    "gen_model_path": str(gen_save),
                    "snow_model_path": str(snow_save),
                    "gen_model_sha256": sha256(gen_save),
                    "snow_model_sha256": sha256(snow_save),
                    "gen_params_after": gen_params_after,
                    "snow_params_after": snow_params_after,
                    "gen_params_reduced": gen_params_reduced,
                    "snow_params_reduced": snow_params_reduced,
                    "gen_params_reduction_percent": gen_reduction_pct,
                    "snow_params_reduction_percent": snow_reduction_pct,
                    "mean_params_reduction_percent": mean_reduction_pct,
                    "gen_replaced_model_mib": file_size_mib(gen_save),
                    "snow_replaced_model_mib": file_size_mib(snow_save),
                    "gen_replaced_map50_95": gen_result["map50_95"],
                    "gen_replaced_map50": gen_result["map50"],
                    "gen_replaced_map75": gen_result["map75"],
                    "gen_precision": gen_result["precision"],
                    "gen_recall": gen_result["recall"],
                    "gen_latency_ms": gen_result["latency_ms"],
                    "gen_fps": gen_result["fps"],
                    "snow_replaced_map50_95": snow_result["map50_95"],
                    "snow_replaced_map50": snow_result["map50"],
                    "snow_replaced_map75": snow_result["map75"],
                    "snow_precision": snow_result["precision"],
                    "snow_recall": snow_result["recall"],
                    "snow_latency_ms": snow_result["latency_ms"],
                    "snow_fps": snow_result["fps"],
                    "ad_gen_map50_95": ad_gen,
                    "ad_snow_map50_95": ad_snow,
                    "nad_gen": nad_gen,
                    "nad_snow": nad_snow,
                    "combined_nad_signed": combined_nad_signed,
                    "negative_or_zero_sensitivity": combined_nad_signed <= 0,
                    "score_denominator": score_denominator,
                    "physical_layer_score": score,
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            row["traceback"] = traceback.format_exc(limit=8).replace("\n", " | ")[:4000]
            print(row["error"])
        finally:
            raw_rows.append(row)
            write_csv(args.out_dir / "raw_dual_domain_layer_replacement.csv", raw_rows)
            del gen_yolo, snow_yolo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ok_rows = [row for row in raw_rows if row.get("status") == "ok"]
    if not ok_rows:
        raise RuntimeError("No layer replacement completed successfully; inspect the raw CSV")

    t1_rows = sorted(ok_rows, key=lambda row: float(row["ad_gen_map50_95"]))
    t2_rows = sorted(ok_rows, key=lambda row: float(row["ad_snow_map50_95"]))
    t3_rows = sorted(ok_rows, key=lambda row: float(row["physical_layer_score"]), reverse=True)

    t1_fields = [
        "layer_id", "layer_type", "gen_layer_params", "gen_params_reduced",
        "gen_params_reduction_percent", "gen_baseline_map50_95", "gen_replaced_map50_95",
        "ad_gen_map50_95", "gen_replaced_map50", "gen_replaced_map75", "gen_model_path",
    ]
    t2_fields = [
        "layer_id", "layer_type", "snow_layer_params", "snow_params_reduced",
        "snow_params_reduction_percent", "snow_baseline_map50_95", "snow_replaced_map50_95",
        "ad_snow_map50_95", "snow_replaced_map50", "snow_replaced_map75", "snow_model_path",
    ]
    t3_fields = [
        "layer_id", "layer_type", "gen_params_reduction_percent", "snow_params_reduction_percent",
        "mean_params_reduction_percent", "ad_gen_map50_95", "ad_snow_map50_95",
        "nad_gen", "nad_snow", "combined_nad_signed", "negative_or_zero_sensitivity",
        "score_denominator", "physical_layer_score", "gen_model_path", "snow_model_path",
    ]
    write_csv(args.out_dir / "T1_GEN_layer_replacement.csv", t1_rows, t1_fields)
    write_csv(args.out_dir / "T2_SNOW_layer_replacement.csv", t2_rows, t2_fields)
    write_csv(args.out_dir / "T3_dual_domain_SWPrunability.csv", t3_rows, t3_fields)

    status_counts: dict[str, int] = {}
    for row in raw_rows:
        status = str(row.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
    summary = {
        "local_baselines_used_for_accuracy_drops": {"GEN": gen_base, "SNOW": snow_base},
        "source_reference_metrics_are_not_used_as_denominators": True,
        "status_counts": status_counts,
        "successful_layer_count": len(ok_rows),
        "output_files": {
            "raw": str(args.out_dir / "raw_dual_domain_layer_replacement.csv"),
            "T1": str(args.out_dir / "T1_GEN_layer_replacement.csv"),
            "T2": str(args.out_dir / "T2_SNOW_layer_replacement.csv"),
            "T3": str(args.out_dir / "T3_dual_domain_SWPrunability.csv"),
        },
    }
    write_json(args.out_dir / "stage2_summary.json", summary)

    print("\n===== STAGE 2 COMPLETE =====")
    print(f"Successful parameterized layers: {len(ok_rows)}")
    print(f"Output folder: {args.out_dir.resolve()}")
    print("T1: T1_GEN_layer_replacement.csv")
    print("T2: T2_SNOW_layer_replacement.csv")
    print("T3: T3_dual_domain_SWPrunability.csv")


if __name__ == "__main__":
    main()
