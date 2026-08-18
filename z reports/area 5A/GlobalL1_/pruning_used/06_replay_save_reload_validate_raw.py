#!/usr/bin/env python3
"""
Stage 5C: replay the approved Global L1 plans, save raw structurally pruned
checkpoints, reload them, verify architecture, and run raw validation.

Inputs
------
- Exact GEN and SNOW baseline checkpoints.
- Stage 5B V2 chosen_replay_plan.csv files.
- T4 final group table.
- Protected-root manifest.
- Exact GEN and SNOW dataset YAML files.

Method
------
- 42 validated GENERIC T4 roots.
- 9 CUSTOM C3k2/C2PSA groups excluded.
- Deterministic incremental Global L1 replay.
- 50% maximum pruning per eligible root, minimum 4 channels.
- No training or recovery in this stage.

Outputs
-------
- Raw GEN and SNOW .pt checkpoints.
- Save/reload architecture audits.
- Raw mAP50-95, mAP50, mAP75, precision and recall.
- Signed accuracy drop and retention against the frozen local baselines.
"""

from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import hashlib
import json
import math
import platform
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
import ultralytics


EXPECTED_BASELINE_PARAMS = {
    "GEN": 2_508_090,
    "SNOW": 2_506_920,
}

FROZEN_BASELINE_METRICS = {
    "GEN": {
        "map50_95": 0.6414207461,
        "map50": 0.8300363459,
    },
    "SNOW": {
        "map50_95": 0.1899135424,
        "map50": 0.3241117518,
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(item))
    return tensors


def all_tensor_output_transform(output: Any) -> tuple[torch.Tensor, ...]:
    tensors = flatten_tensors(output)
    if not tensors:
        raise RuntimeError("Model output contained no tensors.")
    return tuple(tensors)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_dependency_graph(
    model: nn.Module,
    imgsz: int,
    device: torch.device,
) -> tp.DependencyGraph:
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(True)

    model.zero_grad(set_to_none=True)

    example = torch.zeros(
        1,
        3,
        imgsz,
        imgsz,
        device=device,
        requires_grad=True,
    )

    with torch.enable_grad():
        graph = tp.DependencyGraph().build_dependency(
            model,
            example_inputs=example,
            output_transform=all_tensor_output_transform,
        )

    return graph


def architecture_rows(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name, module in model.named_modules():
        row: dict[str, Any] = {
            "module_name": name or "<root>",
            "module_type": module.__class__.__name__,
            "direct_parameter_count": sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
            ),
        }

        for attribute in (
            "in_channels",
            "out_channels",
            "groups",
            "in_features",
            "out_features",
            "num_features",
            "nc",
            "reg_max",
        ):
            value = getattr(module, attribute, "")
            row[attribute] = (
                value
                if isinstance(value, (str, int, float, bool))
                else ""
            )

        rows.append(row)

    return rows


def architecture_sha256(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(
        rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_ultralytics_checkpoint(
    model: nn.Module,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    model_to_save = copy.deepcopy(model).float().cpu()
    model_to_save.eval()

    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": model_to_save,
        "ema": None,
        "updates": 0,
        "optimizer": None,
        "train_args": getattr(model_to_save, "args", {}),
        "train_metrics": None,
        "train_results": None,
        "date": dt.datetime.now(dt.timezone.utc).isoformat(),
        "version": getattr(
            ultralytics,
            "__version__",
            "unknown",
        ),
        "license": "AGPL-3.0",
        "docs": "https://docs.ultralytics.com",
    }

    torch.save(checkpoint, destination)


def metric_value(metrics: Any, path: str) -> float | None:
    value = metrics

    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def replay_and_validate_domain(
    *,
    domain: str,
    checkpoint: Path,
    plan_path: Path,
    data_yaml: Path,
    generic_rows: list[dict[str, str]],
    protected_rows: list[dict[str, str]],
    search_summary: dict[str, Any],
    output_dir: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: torch.device,
    val_device: str,
    seed: int,
    score_tolerance: float,
    forward_check_interval: int,
) -> dict[str, Any]:
    started = time.time()
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(seed)

    plan = read_csv(plan_path)
    if not plan:
        raise RuntimeError(f"{domain}: replay plan is empty.")

    expected_steps = int(
        search_summary["chosen_replay_steps"]
    )
    expected_final_parameters = int(
        search_summary["chosen_remaining_parameters"]
    )

    if len(plan) != expected_steps:
        raise RuntimeError(
            f"{domain}: plan rows {len(plan)} != "
            f"summary steps {expected_steps}"
        )

    shutil.copy2(
        plan_path,
        domain_dir / "chosen_replay_plan_used.csv",
    )

    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(device).eval()

    baseline_parameters = count_parameters(model)
    if baseline_parameters != EXPECTED_BASELINE_PARAMS[domain]:
        raise RuntimeError(
            f"{domain}: baseline parameter mismatch: "
            f"{baseline_parameters} != "
            f"{EXPECTED_BASELINE_PARAMS[domain]}"
        )

    modules = dict(model.named_modules())
    allowed_roots = {
        row["representative_root"]
        for row in generic_rows
    }
    protected_roots = {
        row["module_path"]
        for row in protected_rows
    }

    original_out_channels = {
        name: modules[name].out_channels
        for name in allowed_roots
    }
    protected_out_channels = {
        name: modules[name].out_channels
        for name in protected_roots
        if isinstance(modules.get(name), nn.Conv2d)
    }

    replay_rows: list[dict[str, Any]] = []
    architecture_before = architecture_rows(model)
    write_csv(
        domain_dir / "baseline_architecture.csv",
        architecture_before,
    )

    with torch.no_grad():
        output = model(
            torch.zeros(1, 3, imgsz, imgsz, device=device)
        )
        if not flatten_tensors(output):
            raise RuntimeError(
                f"{domain}: baseline forward produced no tensors"
            )

    for row_number, plan_row in enumerate(plan, start=1):
        step = int(plan_row["step"])
        if step != row_number:
            raise RuntimeError(
                f"{domain}: non-sequential plan at row "
                f"{row_number}: step={step}"
            )

        graph = build_dependency_graph(model, imgsz, device)
        modules = dict(model.named_modules())

        root_name = plan_row["root_module"]
        group_id = plan_row["group_id"]
        channel_index = int(
            plan_row["current_channel_index_pruned"]
        )

        if root_name not in allowed_roots:
            raise RuntimeError(
                f"{domain}: plan selected non-generic root "
                f"{root_name}"
            )

        module = modules[root_name]
        if not isinstance(module, nn.Conv2d):
            raise RuntimeError(
                f"{domain}: selected root is not Conv2d: "
                f"{root_name}"
            )

        expected_before_out = int(
            plan_row["root_out_channels_before"]
        )
        expected_after_out = int(
            plan_row["root_out_channels_after"]
        )
        expected_before_parameters = int(
            plan_row["parameters_before"]
        )
        expected_after_parameters = int(
            plan_row["parameters_after"]
        )
        expected_l1 = float(
            plan_row["l1_score_at_selection"]
        )

        actual_before_parameters = count_parameters(model)
        actual_before_out = module.out_channels

        if actual_before_parameters != expected_before_parameters:
            raise RuntimeError(
                f"{domain} step {step}: parameters before "
                f"{actual_before_parameters} != expected "
                f"{expected_before_parameters}"
            )

        if actual_before_out != expected_before_out:
            raise RuntimeError(
                f"{domain} step {step}: root out before "
                f"{actual_before_out} != expected "
                f"{expected_before_out}"
            )

        if not 0 <= channel_index < module.out_channels:
            raise RuntimeError(
                f"{domain} step {step}: invalid channel index "
                f"{channel_index} for {module.out_channels} channels"
            )

        weights = module.weight.detach().float()
        scores = weights.abs().sum(
            dim=tuple(range(1, weights.ndim))
        )
        actual_l1 = float(scores[channel_index].cpu())

        allowed_error = score_tolerance * max(
            1.0,
            abs(expected_l1),
        )
        if abs(actual_l1 - expected_l1) > allowed_error:
            raise RuntimeError(
                f"{domain} step {step}: L1 score mismatch "
                f"{actual_l1} != {expected_l1}"
            )

        group = graph.get_pruning_group(
            module,
            tp.prune_conv_out_channels,
            idxs=[channel_index],
        )

        if not graph.check_pruning_group(group):
            raise RuntimeError(
                f"{domain} step {step}: DepGraph rejected "
                f"the planned group."
            )

        names_by_id = {
            id(current_module): name
            for name, current_module
            in model.named_modules()
        }

        for dependency, _ in group:
            target_module = dependency.target.module
            target_name = names_by_id.get(
                id(target_module),
                "<unresolved>",
            )

            if (
                graph.is_out_channel_pruning_fn(
                    dependency.handler
                )
                and target_name in protected_roots
            ):
                raise RuntimeError(
                    f"{domain} step {step}: plan would prune "
                    f"protected output root {target_name}"
                )

        group.prune()

        modules_after = dict(model.named_modules())
        actual_after_out = (
            modules_after[root_name].out_channels
        )
        actual_after_parameters = count_parameters(model)

        if actual_after_out != expected_after_out:
            raise RuntimeError(
                f"{domain} step {step}: root out after "
                f"{actual_after_out} != expected "
                f"{expected_after_out}"
            )

        if actual_after_parameters != expected_after_parameters:
            raise RuntimeError(
                f"{domain} step {step}: parameters after "
                f"{actual_after_parameters} != expected "
                f"{expected_after_parameters}"
            )

        should_check_forward = (
            step == 1
            or step == expected_steps
            or step % forward_check_interval == 0
        )
        forward_ok = ""

        if should_check_forward:
            model.eval()
            with torch.no_grad():
                output = model(
                    torch.zeros(
                        1,
                        3,
                        imgsz,
                        imgsz,
                        device=device,
                    )
                )
                forward_ok = bool(flatten_tensors(output))

            if not forward_ok:
                raise RuntimeError(
                    f"{domain} step {step}: forward failed"
                )

        replay_rows.append(
            {
                "step": step,
                "group_id": group_id,
                "root_module": root_name,
                "channel_index": channel_index,
                "expected_l1": expected_l1,
                "actual_l1": actual_l1,
                "parameters_before": actual_before_parameters,
                "parameters_after": actual_after_parameters,
                "root_out_before": actual_before_out,
                "root_out_after": actual_after_out,
                "forward_checked": should_check_forward,
                "forward_ok": forward_ok,
            }
        )

        if step % 25 == 0 or step == expected_steps:
            write_csv(
                domain_dir / "replay_progress.csv",
                replay_rows,
            )
            print(
                f"{domain}: replayed {step}/{expected_steps}, "
                f"parameters={actual_after_parameters:,}"
            )

    final_parameters = count_parameters(model)
    if final_parameters != expected_final_parameters:
        raise RuntimeError(
            f"{domain}: final parameters {final_parameters} "
            f"!= expected {expected_final_parameters}"
        )

    modules_final = dict(model.named_modules())

    protected_rows_out: list[dict[str, Any]] = []
    for name, expected_out in sorted(
        protected_out_channels.items()
    ):
        module = modules_final.get(name)
        actual_out = (
            module.out_channels
            if isinstance(module, nn.Conv2d)
            else None
        )
        same = actual_out == expected_out

        protected_rows_out.append(
            {
                "module_path": name,
                "expected_out_channels": expected_out,
                "actual_out_channels": actual_out,
                "output_unchanged": same,
                "current_in_channels": (
                    module.in_channels
                    if isinstance(module, nn.Conv2d)
                    else ""
                ),
            }
        )

        if not same:
            raise RuntimeError(
                f"{domain}: protected output changed: {name}"
            )

    write_csv(
        domain_dir / "protected_output_verification.csv",
        protected_rows_out,
    )

    final_root_rows: list[dict[str, Any]] = []
    for name in sorted(allowed_roots):
        module = modules_final[name]
        original = original_out_channels[name]
        remaining = module.out_channels
        final_root_rows.append(
            {
                "root_module": name,
                "original_out_channels": original,
                "remaining_out_channels": remaining,
                "removed_out_channels": original - remaining,
                "root_pruning_fraction": (
                    (original - remaining) / original
                ),
            }
        )

    write_csv(
        domain_dir / "final_42_root_inventory.csv",
        final_root_rows,
    )

    architecture_source_rows = architecture_rows(model)
    architecture_source_hash = architecture_sha256(
        architecture_source_rows
    )
    write_csv(
        domain_dir / "source_pruned_architecture.csv",
        architecture_source_rows,
    )

    raw_checkpoint = (
        domain_dir
        / f"{domain}_global_L1_42root_raw.pt"
    )
    save_ultralytics_checkpoint(model, raw_checkpoint)

    checkpoint_hash = sha256_file(raw_checkpoint)

    del model, yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    reloaded_yolo = YOLO(str(raw_checkpoint))
    reloaded_model = reloaded_yolo.model.to(device).eval()

    reload_parameters = count_parameters(reloaded_model)
    reload_architecture_rows = architecture_rows(
        reloaded_model
    )
    reload_architecture_hash = architecture_sha256(
        reload_architecture_rows
    )
    write_csv(
        domain_dir / "reloaded_architecture.csv",
        reload_architecture_rows,
    )

    if reload_parameters != final_parameters:
        raise RuntimeError(
            f"{domain}: reload parameters {reload_parameters} "
            f"!= source {final_parameters}"
        )

    if reload_architecture_hash != architecture_source_hash:
        raise RuntimeError(
            f"{domain}: architecture hash changed after reload"
        )

    with torch.no_grad():
        output = reloaded_model(
            torch.zeros(
                1,
                3,
                imgsz,
                imgsz,
                device=device,
            )
        )
        reload_forward_ok = bool(flatten_tensors(output))

    if not reload_forward_ok:
        raise RuntimeError(
            f"{domain}: reloaded checkpoint forward failed"
        )

    del reloaded_model, reloaded_yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"{domain}: starting raw validation")

    validation_yolo = YOLO(str(raw_checkpoint))
    metrics = validation_yolo.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=val_device,
        workers=workers,
        half=False,
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        augment=False,
        plots=False,
        save_json=False,
        verbose=True,
        seed=seed,
    )

    raw_map = metric_value(metrics, "box.map")
    raw_map50 = metric_value(metrics, "box.map50")
    raw_map75 = metric_value(metrics, "box.map75")
    raw_precision = metric_value(metrics, "box.mp")
    raw_recall = metric_value(metrics, "box.mr")

    if raw_map is None or raw_map50 is None:
        raise RuntimeError(
            f"{domain}: could not extract validation metrics"
        )

    baseline_map = FROZEN_BASELINE_METRICS[
        domain
    ]["map50_95"]
    baseline_map50 = FROZEN_BASELINE_METRICS[
        domain
    ]["map50"]

    result = {
        "domain": domain,
        "status": "ok",
        "baseline_checkpoint": str(checkpoint),
        "baseline_checkpoint_sha256": sha256_file(
            checkpoint
        ),
        "data_yaml": str(data_yaml),
        "plan_path": str(plan_path),
        "plan_sha256": sha256_file(plan_path),
        "replay_steps": expected_steps,
        "baseline_parameters": baseline_parameters,
        "raw_parameters": final_parameters,
        "parameters_removed": (
            baseline_parameters - final_parameters
        ),
        "parameter_reduction_percent": (
            100.0
            * (baseline_parameters - final_parameters)
            / baseline_parameters
        ),
        "raw_checkpoint": str(raw_checkpoint),
        "raw_checkpoint_sha256": checkpoint_hash,
        "raw_checkpoint_size_bytes": (
            raw_checkpoint.stat().st_size
        ),
        "source_architecture_sha256": (
            architecture_source_hash
        ),
        "reloaded_architecture_sha256": (
            reload_architecture_hash
        ),
        "architecture_unchanged_after_reload": (
            architecture_source_hash
            == reload_architecture_hash
        ),
        "parameters_unchanged_after_reload": (
            final_parameters == reload_parameters
        ),
        "reload_forward_ok": reload_forward_ok,
        "baseline_map50_95": baseline_map,
        "raw_map50_95": raw_map,
        "signed_accuracy_drop_map50_95": (
            baseline_map - raw_map
        ),
        "raw_accuracy_retention_percent": (
            100.0 * raw_map / baseline_map
        ),
        "baseline_map50": baseline_map50,
        "raw_map50": raw_map50,
        "signed_accuracy_drop_map50": (
            baseline_map50 - raw_map50
        ),
        "raw_map75": raw_map75,
        "raw_precision": raw_precision,
        "raw_recall": raw_recall,
        "training_performed": False,
        "elapsed_seconds": time.time() - started,
    }

    (domain_dir / "raw_result.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-model", type=Path, required=True)
    parser.add_argument("--snow-model", type=Path, required=True)
    parser.add_argument("--gen-data", type=Path, required=True)
    parser.add_argument("--snow-data", type=Path, required=True)
    parser.add_argument("--search-dir", type=Path, required=True)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument(
        "--protected-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-tolerance",
        type=float,
        default=1e-5,
    )
    parser.add_argument(
        "--forward-check-interval",
        type=int,
        default=25,
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--val-device",
        default="0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    required = [
        ("GEN model", args.gen_model),
        ("SNOW model", args.snow_model),
        ("GEN data", args.gen_data),
        ("SNOW data", args.snow_data),
        ("search summary", args.search_dir / "stage5b_v2_summary.json"),
        ("GEN plan", args.search_dir / "GEN/chosen_replay_plan.csv"),
        ("SNOW plan", args.search_dir / "SNOW/chosen_replay_plan.csv"),
        ("T4", args.t4),
        ("protected manifest", args.protected_manifest),
    ]

    for label, path in required:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t4_rows = read_csv(args.t4)
    generic_rows = [
        row
        for row in t4_rows
        if row["group_kind"] == "GENERIC"
    ]
    custom_rows = [
        row
        for row in t4_rows
        if row["group_kind"] == "CUSTOM"
    ]
    protected_rows = read_csv(
        args.protected_manifest
    )

    if len(generic_rows) != 42 or len(custom_rows) != 9:
        raise RuntimeError(
            f"Unexpected T4 scope: generic={len(generic_rows)}, "
            f"custom={len(custom_rows)}"
        )

    search_summary = json.loads(
        (
            args.search_dir
            / "stage5b_v2_summary.json"
        ).read_text(encoding="utf-8")
    )

    if (
        search_summary.get("status")
        != "PASSED_INCREMENTAL_TARGET_SEARCH"
    ):
        raise RuntimeError(
            "Stage 5B V2 did not pass."
        )

    search_by_domain = {
        item["domain"]: item
        for item in search_summary["domains"]
    }

    protocol = {
        "stage": "5C raw model creation and validation",
        "method": (
            "Deterministic replay of incremental Global L1 "
            "dependency-aware structured pruning plans"
        ),
        "generic_roots_included": 42,
        "custom_roots_excluded": 9,
        "maximum_root_pruning_fraction": 0.50,
        "absolute_minimum_channels": 4,
        "validation": {
            "split": "val",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "half": False,
            "rect": True,
            "conf": 0.001,
            "iou": 0.70,
            "max_det": 300,
            "augment": False,
        },
        "recovery_training_performed": False,
        "frozen_baselines": FROZEN_BASELINE_METRICS,
    }

    (output_dir / "stage5c_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    environment = {
        "generated_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": getattr(
            ultralytics,
            "__version__",
            "unknown",
        ),
        "torch_pruning": getattr(
            tp,
            "__version__",
            "unknown",
        ),
        "device": args.device,
        "val_device": args.val_device,
        "cuda_available": torch.cuda.is_available(),
    }

    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    device = torch.device(args.device)

    results = [
        replay_and_validate_domain(
            domain="GEN",
            checkpoint=args.gen_model.resolve(),
            plan_path=(
                args.search_dir
                / "GEN/chosen_replay_plan.csv"
            ),
            data_yaml=args.gen_data.resolve(),
            generic_rows=generic_rows,
            protected_rows=protected_rows,
            search_summary=search_by_domain["GEN"],
            output_dir=output_dir,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            val_device=args.val_device,
            seed=args.seed,
            score_tolerance=args.score_tolerance,
            forward_check_interval=(
                args.forward_check_interval
            ),
        ),
        replay_and_validate_domain(
            domain="SNOW",
            checkpoint=args.snow_model.resolve(),
            plan_path=(
                args.search_dir
                / "SNOW/chosen_replay_plan.csv"
            ),
            data_yaml=args.snow_data.resolve(),
            generic_rows=generic_rows,
            protected_rows=protected_rows,
            search_summary=search_by_domain["SNOW"],
            output_dir=output_dir,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=device,
            val_device=args.val_device,
            seed=args.seed,
            score_tolerance=args.score_tolerance,
            forward_check_interval=(
                args.forward_check_interval
            ),
        ),
    ]

    write_csv(
        output_dir / "raw_global_l1_results.csv",
        results,
    )

    checks = {
        "both_domains_ok": all(
            result["status"] == "ok"
            for result in results
        ),
        "both_architectures_survive_reload": all(
            result[
                "architecture_unchanged_after_reload"
            ]
            for result in results
        ),
        "both_parameter_counts_survive_reload": all(
            result[
                "parameters_unchanged_after_reload"
            ]
            for result in results
        ),
        "both_reload_forward_passes_ok": all(
            result["reload_forward_ok"]
            for result in results
        ),
        "both_raw_validations_completed": all(
            result["raw_map50_95"] is not None
            for result in results
        ),
        "no_recovery_training_performed": True,
    }

    final = {
        "status": (
            "PASSED_RAW_MODEL_CREATION_AND_VALIDATION"
            if all(checks.values())
            else "FAILED_RAW_MODEL_CREATION_OR_VALIDATION"
        ),
        "checks": checks,
        "results": results,
        "methodological_limit": (
            "The 9 custom C3k2/C2PSA groups remain excluded."
        ),
        "safe_next_step": (
            "Run a one-epoch recovery smoke test on the raw SNOW "
            "checkpoint, then run matched 20-epoch GEN and SNOW recovery."
        ),
    }

    (output_dir / "stage5c_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )

    print("\n===== STAGE 5C COMPLETE =====")
    print(json.dumps(final, indent=2))
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
