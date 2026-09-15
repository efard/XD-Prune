from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        return float(value)
    except Exception:
        return default


def summarize(results: Any) -> dict[str, float]:
    speed = getattr(results, "speed", {}) or {}
    preprocess = safe_float(speed.get("preprocess", 0.0), 0.0)
    inference = safe_float(speed.get("inference", 0.0), 0.0)
    postprocess = safe_float(speed.get("postprocess", 0.0), 0.0)
    latency = preprocess + inference + postprocess
    box = results.box
    return {
        "map50_95": safe_float(box.map),
        "map50": safe_float(box.map50),
        "map75": safe_float(box.map75),
        "precision": safe_float(getattr(box, "mp", float("nan"))),
        "recall": safe_float(getattr(box, "mr", float("nan"))),
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
            raise RuntimeError(f"Layer {layer_id} reference input is not 4D")
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


def replace_one_layer(
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
        "layer_id": layer_id,
        "old_layer_type": old_layer.__class__.__name__,
        "original_layer_params": original_layer_params,
        "input_shapes": interface["input_shapes"],
        "output_shape": interface["output_shape"],
        "adapter_repr": repr(adapter),
    }


def apply_combination(
    yolo: YOLO,
    layer_ids: tuple[int, ...],
    imgsz: int,
    device: torch.device,
    multi_input_policy: str,
) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    # Ascending order makes candidate construction deterministic.
    for layer_id in sorted(layer_ids):
        details.append(
            replace_one_layer(yolo, layer_id, imgsz, device, multi_input_policy)
        )
    return details


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
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision_version,
        "ultralytics": ultralytics_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
    }


def rows_by_domain(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    result = {row["domain"].upper(): row for row in rows}
    missing = {"GEN", "SNOW"} - set(result)
    if missing:
        raise RuntimeError(f"Missing domains in baseline CSV: {sorted(missing)}")
    return result


def build_individual_report(
    stage2_raw: list[dict[str, str]],
    stage2_baselines: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    all_layers: list[dict[str, Any]] = []
    parameterized: list[dict[str, Any]] = []
    gen_base_mib = float(stage2_baselines["GEN"]["checkpoint_mib"])
    snow_base_mib = float(stage2_baselines["SNOW"]["checkpoint_mib"])

    for source in stage2_raw:
        row: dict[str, Any] = {
            "layer_id": source.get("layer_id", ""),
            "layer_type": source.get("layer_type", ""),
            "status": source.get("status", ""),
            "error": source.get("error", ""),
            "gen_layer_params": source.get("gen_layer_params", ""),
            "snow_layer_params": source.get("snow_layer_params", ""),
            "gen_params_reduction_percent": source.get("gen_params_reduction_percent", ""),
            "snow_params_reduction_percent": source.get("snow_params_reduction_percent", ""),
            "mean_params_reduction_percent": source.get("mean_params_reduction_percent", ""),
            "gen_baseline_map50_95": source.get("gen_baseline_map50_95", ""),
            "gen_replaced_map50_95": source.get("gen_replaced_map50_95", ""),
            "ad_gen_map50_95_signed": source.get("ad_gen_map50_95", ""),
            "snow_baseline_map50_95": source.get("snow_baseline_map50_95", ""),
            "snow_replaced_map50_95": source.get("snow_replaced_map50_95", ""),
            "ad_snow_map50_95_signed": source.get("ad_snow_map50_95", ""),
            "gen_replaced_model_mib": source.get("gen_replaced_model_mib", ""),
            "snow_replaced_model_mib": source.get("snow_replaced_model_mib", ""),
            "gen_model_path": source.get("gen_model_path", ""),
            "snow_model_path": source.get("snow_model_path", ""),
        }
        if source.get("gen_replaced_model_mib", ""):
            row["gen_saved_size_reduction_percent"] = (
                (gen_base_mib - float(source["gen_replaced_model_mib"])) / gen_base_mib * 100.0
            )
        else:
            row["gen_saved_size_reduction_percent"] = ""
        if source.get("snow_replaced_model_mib", ""):
            row["snow_saved_size_reduction_percent"] = (
                (snow_base_mib - float(source["snow_replaced_model_mib"])) / snow_base_mib * 100.0
            )
        else:
            row["snow_saved_size_reduction_percent"] = ""
        all_layers.append(row)
        if source.get("status") == "ok":
            parameterized.append(row)

    parameterized.sort(key=lambda row: int(row["layer_id"]))
    all_layers.sort(key=lambda row: int(row["layer_id"]))
    return all_layers, parameterized


def generate_candidates(
    metadata: list[dict[str, str]],
    target_percent: float,
    relative_tolerance: float,
    max_combination_size: int,
) -> tuple[list[dict[str, Any]], float, float]:
    eligible = [
        row for row in metadata
        if str(row.get("candidate_for_size_reduction", "")).lower() == "true"
        and int(row["gen_layer_params"]) > 0
        and int(row["snow_layer_params"]) > 0
    ]
    eligible.sort(key=lambda row: int(row["layer_id"]))

    lower = target_percent * (1.0 - relative_tolerance)
    upper = target_percent * (1.0 + relative_tolerance)
    candidates: list[dict[str, Any]] = []

    for size in range(1, max_combination_size + 1):
        for combo in itertools.combinations(eligible, size):
            layer_ids = tuple(int(row["layer_id"]) for row in combo)
            layer_types = tuple(str(row["layer_type"]) for row in combo)
            gen_est = sum(float(row["gen_layer_params_percent"]) for row in combo)
            snow_est = sum(float(row["snow_layer_params_percent"]) for row in combo)
            mean_est = 0.5 * (gen_est + snow_est)
            # Both domain-specific models must fall inside the declared tolerance band.
            if lower <= gen_est <= upper and lower <= snow_est <= upper:
                candidates.append(
                    {
                        "layer_ids_tuple": layer_ids,
                        "layer_types_tuple": layer_types,
                        "replaced_layers": ",".join(str(value) for value in layer_ids),
                        "layer_types": ",".join(layer_types),
                        "combination_size": size,
                        "estimated_gen_reduction_percent": gen_est,
                        "estimated_snow_reduction_percent": snow_est,
                        "estimated_mean_reduction_percent": mean_est,
                        "estimated_target_distance_percentage_points": abs(mean_est - target_percent),
                    }
                )

    candidates.sort(
        key=lambda row: (
            float(row["estimated_target_distance_percentage_points"]),
            int(row["combination_size"]),
            tuple(row["layer_ids_tuple"]),
        )
    )
    for index, row in enumerate(candidates, start=1):
        row["candidate_id"] = f"C{index:03d}"
    return candidates, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every 1-to-N layer-replacement combination whose estimated GEN and SNOW "
            "parameter reductions are near a target. Accuracy drops remain signed; no T3 score is used."
        )
    )
    parser.add_argument("--gen-model", required=True, type=Path)
    parser.add_argument("--snow-model", required=True, type=Path)
    parser.add_argument("--gen-data", required=True, type=Path)
    parser.add_argument("--snow-data", required=True, type=Path)
    parser.add_argument("--stage2-dir", required=True, type=Path)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--target-percent", default=10.25, type=float)
    parser.add_argument("--relative-tolerance", default=0.05, type=float)
    parser.add_argument("--max-combination-size", default=3, type=int)
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--multi-input-policy", default="concat", choices=["concat", "first"])
    parser.add_argument("--repeat-tolerance", default=0.002, type=float)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.relative_tolerance < 1.0:
        raise ValueError("--relative-tolerance must be a fraction such as 0.05")
    if args.max_combination_size < 1:
        raise ValueError("--max-combination-size must be at least 1")

    required = [args.gen_model, args.snow_model, args.gen_data, args.snow_data]
    stage2_files = {
        "metadata": args.stage2_dir / "layer_metadata_dual_domain.csv",
        "raw": args.stage2_dir / "raw_dual_domain_layer_replacement.csv",
        "baseline": args.stage2_dir / "baseline_GEN_SNOW_stage2.csv",
        "repeatability": args.stage2_dir / "baseline_repeatability_gate.csv",
    }
    required.extend(stage2_files.values())
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.project.mkdir(parents=True, exist_ok=True)
    models_dir = args.out_dir / "saved_near_target_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    metadata = read_csv(stage2_files["metadata"])
    stage2_raw = read_csv(stage2_files["raw"])
    stage2_baselines = rows_by_domain(read_csv(stage2_files["baseline"]))

    all_individual, parameterized_individual = build_individual_report(
        stage2_raw, stage2_baselines
    )
    write_csv(args.out_dir / "all_top_level_layer_status_from_stage2.csv", all_individual)
    write_csv(
        args.out_dir / "individual_parameterized_layer_results.csv",
        parameterized_individual,
    )

    candidates, lower, upper = generate_candidates(
        metadata,
        args.target_percent,
        args.relative_tolerance,
        args.max_combination_size,
    )
    if not candidates:
        raise RuntimeError("No candidates fall inside the requested target range")

    candidate_public_fields = [
        "candidate_id", "replaced_layers", "layer_types", "combination_size",
        "estimated_gen_reduction_percent", "estimated_snow_reduction_percent",
        "estimated_mean_reduction_percent", "estimated_target_distance_percentage_points",
    ]
    write_csv(
        args.out_dir / "near_target_candidate_search_space.csv",
        candidates,
        candidate_public_fields,
    )

    evidence = environment_record()
    evidence.update(
        {
            "method": "whole top-level layer replacement with parameter-free dynamic adapter",
            "negative_accuracy_drop_policy": "retain signed values",
            "T3_formula_status": "pending; no T3 score used in Stage 3",
            "target_percent": args.target_percent,
            "relative_tolerance": args.relative_tolerance,
            "accepted_range_percent": [lower, upper],
            "max_combination_size": args.max_combination_size,
            "candidate_count": len(candidates),
            "input_sha256": {
                "gen_model": sha256(args.gen_model),
                "snow_model": sha256(args.snow_model),
                "gen_data": sha256(args.gen_data),
                "snow_data": sha256(args.snow_data),
                **{f"stage2_{key}": sha256(path) for key, path in stage2_files.items()},
                "script": sha256(Path(__file__).resolve()),
                "adapter_module": sha256(Path(__file__).with_name("layer_replacement_adapter.py")),
            },
        }
    )
    write_json(args.out_dir / "experiment_environment_and_checksums.json", evidence)

    device = torch_device(args.device)
    print("Loading supplied baselines...")
    gen_baseline_yolo = YOLO(str(args.gen_model), task="detect")
    snow_baseline_yolo = YOLO(str(args.snow_model), task="detect")
    gen_baseline_params = count_params(gen_baseline_yolo.model)
    snow_baseline_params = count_params(snow_baseline_yolo.model)
    gen_baseline_file_mib = file_size_mib(args.gen_model)
    snow_baseline_file_mib = file_size_mib(args.snow_model)

    if len(gen_baseline_yolo.names) != 11:
        raise RuntimeError(f"GEN checkpoint has {len(gen_baseline_yolo.names)} classes, expected 11")
    if len(snow_baseline_yolo.names) != 8:
        raise RuntimeError(f"SNOW checkpoint has {len(snow_baseline_yolo.names)} classes, expected 8")

    print("Evaluating in-run GEN baseline...")
    gen_base = validate_model(
        gen_baseline_yolo, args.gen_data, args.split, args.imgsz, args.batch,
        args.device, args.workers, args.project, "baseline_GEN_stage3"
    )
    print("Evaluating in-run SNOW baseline...")
    snow_base = validate_model(
        snow_baseline_yolo, args.snow_data, args.split, args.imgsz, args.batch,
        args.device, args.workers, args.project, "baseline_SNOW_stage3"
    )

    repeat_rows = [
        {
            "domain": "GEN",
            "stage2_map50_95": float(stage2_baselines["GEN"]["map50_95"]),
            "stage3_map50_95": gen_base["map50_95"],
            "absolute_difference": abs(float(stage2_baselines["GEN"]["map50_95"]) - gen_base["map50_95"]),
        },
        {
            "domain": "SNOW",
            "stage2_map50_95": float(stage2_baselines["SNOW"]["map50_95"]),
            "stage3_map50_95": snow_base["map50_95"],
            "absolute_difference": abs(float(stage2_baselines["SNOW"]["map50_95"]) - snow_base["map50_95"]),
        },
    ]
    for row in repeat_rows:
        row["status"] = "PASS" if row["absolute_difference"] <= args.repeat_tolerance else "FAIL"
    write_csv(args.out_dir / "baseline_repeatability_gate.csv", repeat_rows)
    if any(row["status"] != "PASS" for row in repeat_rows):
        raise RuntimeError("Stage-3 baseline repeatability gate failed")

    write_csv(
        args.out_dir / "baseline_GEN_SNOW_stage3.csv",
        [
            {
                "domain": "GEN", "classes": len(gen_baseline_yolo.names),
                "parameters": gen_baseline_params, "checkpoint_mib": gen_baseline_file_mib,
                **gen_base,
            },
            {
                "domain": "SNOW", "classes": len(snow_baseline_yolo.names),
                "parameters": snow_baseline_params, "checkpoint_mib": snow_baseline_file_mib,
                **snow_base,
            },
        ],
    )

    del gen_baseline_yolo, snow_baseline_yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_path = args.out_dir / "raw_near_target_combination_results.csv"
    raw_rows: list[dict[str, Any]] = []
    if args.resume and raw_path.is_file():
        raw_rows = read_csv(raw_path)
        print(f"Resume enabled: loaded {len(raw_rows)} existing candidate rows")

    completed: set[str] = set()
    for row in raw_rows:
        status = row.get("status", "")
        if status == "ok" or (status == "failed" and not args.retry_failed):
            completed.add(row.get("candidate_id", ""))

    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in completed:
            print(f"Skipping completed candidate {candidate_id}")
            continue

        layer_ids = tuple(int(value) for value in candidate["layer_ids_tuple"])
        print(
            f"\n===== {candidate_id}: layers {candidate['replaced_layers']} "
            f"(estimated mean reduction {candidate['estimated_mean_reduction_percent']:.6f}%) ====="
        )

        row: dict[str, Any] = {
            **{key: candidate[key] for key in candidate_public_fields},
            "target_percent": args.target_percent,
            "accepted_lower_percent": lower,
            "accepted_upper_percent": upper,
            "negative_accuracy_drop_policy": "retain_signed",
            "status": "ok",
            "error": "",
            "gen_baseline_map50_95": gen_base["map50_95"],
            "snow_baseline_map50_95": snow_base["map50_95"],
        }

        gen_yolo: YOLO | None = None
        snow_yolo: YOLO | None = None
        gen_save = models_dir / f"GEN_{candidate_id}_layers_{'-'.join(map(str, layer_ids))}.pt"
        snow_save = models_dir / f"SNOW_{candidate_id}_layers_{'-'.join(map(str, layer_ids))}.pt"

        try:
            gen_yolo = YOLO(str(args.gen_model), task="detect")
            snow_yolo = YOLO(str(args.snow_model), task="detect")

            gen_details = apply_combination(
                gen_yolo, layer_ids, args.imgsz, device, args.multi_input_policy
            )
            snow_details = apply_combination(
                snow_yolo, layer_ids, args.imgsz, device, args.multi_input_policy
            )

            gen_params_after = count_params(gen_yolo.model)
            snow_params_after = count_params(snow_yolo.model)
            gen_params_reduced = gen_baseline_params - gen_params_after
            snow_params_reduced = snow_baseline_params - snow_params_after
            gen_reduction_pct = gen_params_reduced / gen_baseline_params * 100.0
            snow_reduction_pct = snow_params_reduced / snow_baseline_params * 100.0
            mean_reduction_pct = 0.5 * (gen_reduction_pct + snow_reduction_pct)

            gen_yolo.save(str(gen_save))
            snow_yolo.save(str(snow_save))
            del gen_yolo, snow_yolo
            gen_yolo = YOLO(str(gen_save), task="detect")
            snow_yolo = YOLO(str(snow_save), task="detect")

            if count_params(gen_yolo.model) != gen_params_after:
                raise RuntimeError("GEN saved-model reload parameter count mismatch")
            if count_params(snow_yolo.model) != snow_params_after:
                raise RuntimeError("SNOW saved-model reload parameter count mismatch")

            print("Validating GEN candidate...")
            gen_result = validate_model(
                gen_yolo, args.gen_data, args.split, args.imgsz, args.batch,
                args.device, args.workers, args.project, f"{candidate_id}_GEN"
            )
            print("Validating SNOW candidate...")
            snow_result = validate_model(
                snow_yolo, args.snow_data, args.split, args.imgsz, args.batch,
                args.device, args.workers, args.project, f"{candidate_id}_SNOW"
            )

            # Signed accuracy drops are retained exactly as decided by the research team.
            ad_gen = gen_base["map50_95"] - gen_result["map50_95"]
            ad_snow = snow_base["map50_95"] - snow_result["map50_95"]

            gen_model_mib = file_size_mib(gen_save)
            snow_model_mib = file_size_mib(snow_save)
            gen_size_reduction_pct = (
                (gen_baseline_file_mib - gen_model_mib) / gen_baseline_file_mib * 100.0
            )
            snow_size_reduction_pct = (
                (snow_baseline_file_mib - snow_model_mib) / snow_baseline_file_mib * 100.0
            )

            row.update(
                {
                    "gen_params_before": gen_baseline_params,
                    "snow_params_before": snow_baseline_params,
                    "gen_params_after": gen_params_after,
                    "snow_params_after": snow_params_after,
                    "gen_params_reduced": gen_params_reduced,
                    "snow_params_reduced": snow_params_reduced,
                    "gen_params_reduction_percent": gen_reduction_pct,
                    "snow_params_reduction_percent": snow_reduction_pct,
                    "mean_params_reduction_percent": mean_reduction_pct,
                    "target_distance_percentage_points": abs(mean_reduction_pct - args.target_percent),
                    "gen_exact_within_tolerance": lower <= gen_reduction_pct <= upper,
                    "snow_exact_within_tolerance": lower <= snow_reduction_pct <= upper,
                    "both_exact_within_tolerance": (
                        lower <= gen_reduction_pct <= upper and lower <= snow_reduction_pct <= upper
                    ),
                    "gen_model_path": str(gen_save),
                    "snow_model_path": str(snow_save),
                    "gen_model_sha256": sha256(gen_save),
                    "snow_model_sha256": sha256(snow_save),
                    "gen_model_mib": gen_model_mib,
                    "snow_model_mib": snow_model_mib,
                    "gen_saved_size_reduction_percent": gen_size_reduction_pct,
                    "snow_saved_size_reduction_percent": snow_size_reduction_pct,
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
                    "ad_gen_map50_95_signed": ad_gen,
                    "ad_snow_map50_95_signed": ad_snow,
                    "gen_accuracy_improved": ad_gen < 0,
                    "snow_accuracy_improved": ad_snow < 0,
                    "gen_adapter_details_json": json.dumps(gen_details, default=str),
                    "snow_adapter_details_json": json.dumps(snow_details, default=str),
                }
            )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"[:1000]
            row["traceback"] = traceback.format_exc(limit=10).replace("\n", " | ")[:5000]
            print(row["error"])
            for partial in (gen_save, snow_save):
                if partial.exists():
                    partial.unlink()
        finally:
            # Replace a previous row with the same candidate ID when retrying.
            raw_rows = [old for old in raw_rows if old.get("candidate_id") != candidate_id]
            raw_rows.append(row)
            raw_rows.sort(key=lambda item: str(item.get("candidate_id", "")))
            write_csv(raw_path, raw_rows)
            del gen_yolo, snow_yolo
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    ok_rows = [row for row in raw_rows if row.get("status") == "ok"]
    failed_rows = [row for row in raw_rows if row.get("status") == "failed"]
    if not ok_rows:
        raise RuntimeError("No near-target combination completed successfully")

    by_distance = sorted(
        ok_rows,
        key=lambda row: (
            float(row["target_distance_percentage_points"]),
            float(row["ad_gen_map50_95_signed"]),
            float(row["ad_snow_map50_95_signed"]),
        ),
    )
    by_gen = sorted(
        ok_rows,
        key=lambda row: (
            float(row["ad_gen_map50_95_signed"]),
            float(row["target_distance_percentage_points"]),
        ),
    )
    by_snow = sorted(
        ok_rows,
        key=lambda row: (
            float(row["ad_snow_map50_95_signed"]),
            float(row["target_distance_percentage_points"]),
        ),
    )

    report_fields = [
        "candidate_id", "replaced_layers", "layer_types", "combination_size",
        "gen_params_reduction_percent", "snow_params_reduction_percent",
        "mean_params_reduction_percent", "target_distance_percentage_points",
        "gen_saved_size_reduction_percent", "snow_saved_size_reduction_percent",
        "gen_baseline_map50_95", "gen_replaced_map50_95", "ad_gen_map50_95_signed",
        "snow_baseline_map50_95", "snow_replaced_map50_95", "ad_snow_map50_95_signed",
        "gen_accuracy_improved", "snow_accuracy_improved",
        "gen_replaced_map50", "snow_replaced_map50",
        "gen_latency_ms", "snow_latency_ms", "gen_fps", "snow_fps",
        "gen_model_path", "snow_model_path", "status",
    ]
    write_csv(args.out_dir / "near_target_results_sorted_by_target_distance.csv", by_distance, report_fields)
    write_csv(args.out_dir / "near_target_results_sorted_by_GEN_signed_AD.csv", by_gen, report_fields)
    write_csv(args.out_dir / "near_target_results_sorted_by_SNOW_signed_AD.csv", by_snow, report_fields)
    write_csv(args.out_dir / "failed_near_target_candidates.csv", failed_rows)

    summary = {
        "target_percent": args.target_percent,
        "relative_tolerance": args.relative_tolerance,
        "accepted_range_percent": [lower, upper],
        "candidate_generation_rule": (
            "all parameterized combinations of size 1 through max_combination_size for which "
            "both estimated GEN and estimated SNOW reductions fall inside the accepted range"
        ),
        "candidate_count": len(candidates),
        "successful_candidate_count": len(ok_rows),
        "failed_candidate_count": len(failed_rows),
        "combination_size_counts": {
            str(size): sum(1 for row in candidates if int(row["combination_size"]) == size)
            for size in range(1, args.max_combination_size + 1)
        },
        "negative_accuracy_drop_policy": "retain signed values; no abs() and no zero clamp",
        "T3_formula_used": False,
        "local_baselines": {"GEN": gen_base, "SNOW": snow_base},
        "best_by_target_distance": by_distance[0]["candidate_id"] if by_distance else None,
        "best_by_GEN_signed_AD": by_gen[0]["candidate_id"] if by_gen else None,
        "best_by_SNOW_signed_AD": by_snow[0]["candidate_id"] if by_snow else None,
        "outputs": {
            "individual_parameterized_layers": str(args.out_dir / "individual_parameterized_layer_results.csv"),
            "candidate_search_space": str(args.out_dir / "near_target_candidate_search_space.csv"),
            "raw": str(raw_path),
            "by_target_distance": str(args.out_dir / "near_target_results_sorted_by_target_distance.csv"),
            "by_GEN_AD": str(args.out_dir / "near_target_results_sorted_by_GEN_signed_AD.csv"),
            "by_SNOW_AD": str(args.out_dir / "near_target_results_sorted_by_SNOW_signed_AD.csv"),
        },
    }
    write_json(args.out_dir / "stage3_summary.json", summary)

    print("\n===== STAGE 3 COMPLETE =====")
    print(f"Candidate search space: {len(candidates)}")
    print(f"Successful candidates:  {len(ok_rows)}")
    print(f"Failed candidates:      {len(failed_rows)}")
    print(f"Accepted range:         {lower:.6f}% to {upper:.6f}%")
    print(f"Output folder:          {args.out_dir.resolve()}")
    print("No T3 formula was used. GEN and SNOW accuracy drops are retained as signed values.")


if __name__ == "__main__":
    main()
