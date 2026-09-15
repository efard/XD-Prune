"""Shared engine for exhaustive ratio-aware YOLO26n T1/T2 sweeps.

The public launchers freeze the ratio at 25% or 37.5%. Every worker loads one
untouched domain baseline, prunes exactly one of the 51 validated dependency
groups, performs structural and save/reload checks, evaluates the validation
split, writes one atomic JSON record, then exits. No fine-tuning, BatchNorm
update, cumulative pruning, or test data is used.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import types
from typing import Any

import psutil

import run_t1_t2 as base
import run_t1_t2_full_sweep as full
import run_t1_t2_custom_sweep as custom
from yolo26n_custom_pruning_rules import (
    attention_c3k2_head_aware_importance,
    c2psa_head_aware_importance,
    nonattention_c3k2_logical_importance,
    prune_attention_c3k2_head_aware_units,
    prune_c2psa_head_aware_units,
    prune_nonattention_c3k2_logical_channels,
    validate_attention_c3k2_invariants,
    validate_c2psa_invariants,
    validate_nonattention_c3k2_invariants,
)


PROJECT_ROOT = base.PROJECT_ROOT
STUDY_ROOT = base.STUDY_ROOT
GENERIC_FREEZE = full.FREEZE_PATH
CUSTOM_FREEZE = custom.FREEZE_PATH
GENERIC_CATALOGUE = full.CATALOGUE_PATH
GENERIC_OPERATIONS = full.OPERATIONS_PATH
CUSTOM_CATALOGUE = custom.CATALOGUE_PATH
VALIDATION_SIZE = 640
TRACE_SIZE = 32


def ratio_slug(fraction: Fraction) -> str:
    return "25pct" if fraction == Fraction(1, 4) else "375pct" if fraction == Fraction(3, 8) else f"{fraction.numerator}of{fraction.denominator}"


def ratio_percent(fraction: Fraction) -> float:
    return 100.0 * fraction.numerator / fraction.denominator


def exact_count(width: int, fraction: Fraction, label: str) -> int:
    numerator = width * fraction.numerator
    if numerator % fraction.denominator:
        raise ValueError(f"{label}: width {width} does not support exact fraction {fraction}")
    count = numerator // fraction.denominator
    if count < 1 or count >= width:
        raise ValueError(f"{label}: invalid removal count {count} for width {width}")
    return count


def generic_rows() -> dict[str, dict[str, str]]:
    return {row["canonical_group_id"]: row for row in base.read_csv(GENERIC_CATALOGUE)}


def custom_rows() -> dict[str, dict[str, str]]:
    return {row["custom_group_id"]: row for row in base.read_csv(CUSTOM_CATALOGUE)}


def all_group_ids() -> list[str]:
    generic = [str(value) for value in base.read_json(GENERIC_FREEZE)["sweep_groups"]]
    special = [str(value) for value in base.read_json(CUSTOM_FREEZE)["sweep_groups"]]
    groups = [*generic, *special]
    if len(groups) != 51 or len(set(groups)) != 51:
        raise RuntimeError("Ratio sweep must contain exactly 51 unique groups")
    return groups


def baseline_config(domain: str) -> dict[str, Any]:
    config = dict(base.read_json(CUSTOM_FREEZE)["baseline_models"][domain])
    config["baseline_reference"] = base.relative(custom.resolved_baseline_reference(config))
    return config


def preflight(fraction: Fraction, require_cuda: bool = True) -> dict[str, Any]:
    """Verify frozen evidence and exact arithmetic for both requested ratios."""

    import torch
    from ultralytics import YOLO

    if fraction not in {Fraction(1, 4), Fraction(3, 8)}:
        raise ValueError("This study freezes ratio sweeps at 25% and 37.5%")
    frozen = custom.preflight(require_cuda=require_cuda)
    generic = generic_rows()
    special = custom_rows()
    groups = all_group_ids()
    for group_id, row in generic.items():
        exact_count(int(row["root_out_channels"]), fraction, f"{group_id} root")
    for group_id, row in special.items():
        exact_count(int(row["hidden_channels_before"]), fraction, f"{group_id} logical width")

    # Check per-head divisibility for the two attention-aware custom groups.
    checkpoint = PROJECT_ROOT / baseline_config("GEN")["path"]
    model = YOLO(str(checkpoint), task="detect").model.float().cpu().eval()
    _, c2_layout = c2psa_head_aware_importance(model.model[10])
    _, c3_layout = attention_c3k2_head_aware_importance(model.model[22])
    exact_count(c2_layout.key_dim, fraction, "C2PSA units per head")
    exact_count(c3_layout.key_dim, fraction, "attention-C3k2 units per head")
    del model
    gc.collect()
    return {
        "schema": f"t1_t2_ratio_{ratio_slug(fraction)}_preflight_v1",
        "ratio_fraction": str(fraction),
        "ratio_percent": ratio_percent(fraction),
        "groups": len(groups),
        "domains": ["GEN", "SNOW"],
        "expected_group_domain_runs": len(groups) * 2,
        "generic_groups": len(generic),
        "custom_groups": len(special),
        "exact_group_width_arithmetic": True,
        "exact_attention_head_arithmetic": True,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "source_preflight": frozen,
    }


def tensor_comparison(left: Any, right: Any) -> dict[str, Any]:
    import torch

    left_tensors = list(base.flatten_tensors(left))
    right_tensors = list(base.flatten_tensors(right))
    if len(left_tensors) != len(right_tensors):
        raise RuntimeError("Saved model output tensor count changed")
    maximum = 0.0
    total = 0.0
    elements = 0
    for first, second in zip(left_tensors, right_tensors):
        if tuple(first.shape) != tuple(second.shape):
            raise RuntimeError("Saved model output shape changed")
        difference = (first.float() - second.float()).abs()
        maximum = max(maximum, float(difference.max()))
        total += float(difference.sum())
        elements += difference.numel()
    return {
        "tensor_count": len(left_tensors),
        "max_absolute_error": maximum,
        "mean_absolute_error": total / elements if elements else 0.0,
        "exact": maximum == 0.0,
    }


def balanced_attention_indices(scores: Any, layout: Any, fraction: Fraction) -> list[int]:
    """Remove an exact, equal number of least-important units per head."""

    import torch

    if scores.numel() != layout.total_units or not bool(torch.isfinite(scores).all()):
        raise ValueError("Invalid attention-unit importance vector")
    remove_per_head = exact_count(layout.key_dim, fraction, "attention units per head")
    selected: list[int] = []
    for head in range(layout.num_heads):
        offset = head * layout.key_dim
        local = scores[offset : offset + layout.key_dim]
        ranked = torch.argsort(local, stable=True)[:remove_per_head].tolist()
        selected.extend(offset + int(position) for position in ranked)
    return sorted(selected)


def apply_custom(model: Any, group_id: str, fraction: Fraction) -> dict[str, Any]:
    import torch

    row = custom_rows()[group_id]
    block_index = int(row["block_index"])
    block = model.model[block_index]
    hidden_before = int(block.c)
    remove_count = exact_count(hidden_before, fraction, f"{group_id} hidden width")
    if block_index in custom.NONATTENTION_BLOCKS:
        before = validate_nonattention_c3k2_invariants(block)
        importance = nonattention_c3k2_logical_importance(block)
        rank_order = [int(value) for value in torch.argsort(importance, stable=True)[:remove_count].tolist()]
        selected = sorted(rank_order)
        result = prune_nonattention_c3k2_logical_channels(block, selected, module_path=f"model.{block_index}")
        after = validate_nonattention_c3k2_invariants(block)
        selection_unit = "hidden_channel"
    elif block_index == 10:
        before = validate_c2psa_invariants(block)
        importance, layout = c2psa_head_aware_importance(block)
        selected = balanced_attention_indices(importance, layout, fraction)
        rank_order = sorted(selected, key=lambda index: (float(importance[index]), index))
        result = prune_c2psa_head_aware_units(block, selected, module_path=f"model.{block_index}")
        after = validate_c2psa_invariants(block)
        selection_unit = "paired_attention_unit"
    elif block_index == 22:
        before = validate_attention_c3k2_invariants(block)
        importance, layout = attention_c3k2_head_aware_importance(block)
        selected = balanced_attention_indices(importance, layout, fraction)
        rank_order = sorted(selected, key=lambda index: (float(importance[index]), index))
        result = prune_attention_c3k2_head_aware_units(block, selected, module_path=f"model.{block_index}")
        after = validate_attention_c3k2_invariants(block)
        selection_unit = "paired_attention_unit"
    else:
        raise ValueError(f"Unsupported custom block index: {block_index}")
    actual = result.hidden_channels_removed / hidden_before
    if abs(actual - float(fraction)) > 1e-12:
        raise RuntimeError(f"{group_id}: custom rule produced fraction {actual}, expected {float(fraction)}")
    return {
        "method": "validated custom block-aware logical-width rule",
        "representative_root": row["block_path"],
        "rule_family": row["rule_family"],
        "selection_unit": selection_unit,
        "root_channels_before": hidden_before,
        "root_channels_removed": result.hidden_channels_removed,
        "root_channels_after": result.hidden_channels_after,
        "actual_root_fraction": actual,
        "selected_indices": selected,
        "selection_rank_order": rank_order,
        "selected_scores_rank_order": [float(importance[index]) for index in rank_order],
        "operations": result.to_dict()["operations"],
        "operation_count": len(result.operations),
        "invariants_before": before,
        "invariants_after": after,
    }


def apply_generic(model: Any, group_id: str, fraction: Fraction) -> dict[str, Any]:
    import torch
    import torch_pruning as tp

    row = generic_rows()[group_id]
    canonical = base.read_json(GENERIC_OPERATIONS)[group_id]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    modules = dict(model.named_modules())
    module_to_path = {id(module): path for path, module in modules.items()}
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: generic root is not Conv2d")
    channels_before = int(root.out_channels)
    if channels_before != int(row["root_out_channels"]):
        raise RuntimeError(f"{group_id}: root width differs from frozen catalogue")
    remove_count = exact_count(channels_before, fraction, f"{group_id} root")
    head = model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("Expected YOLO26 end-to-end Detect head")
    head.forward = types.MethodType(base.trace_detect_forward, head)

    class TraceWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, images: Any) -> tuple[Any, ...]:
            tensors = tuple(base.flatten_tensors(self.inner(images)))
            if not tensors or not all(tensor.requires_grad for tensor in tensors):
                raise RuntimeError("Trace outputs did not retain Autograd dependencies")
            return tensors

    try:
        graph = tp.DependencyGraph().build_dependency(
            TraceWrapper(model), example_inputs=torch.zeros(1, 3, TRACE_SIZE, TRACE_SIZE)
        )
        full_group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=list(range(channels_before)))
        importance_fn = tp.importance.GroupMagnitudeImportance(
            p=1, group_reduction="mean", normalizer="mean", bias=False
        )
        importance = importance_fn(full_group)
        if importance is None or importance.numel() != channels_before or not bool(torch.isfinite(importance).all()):
            raise RuntimeError("Invalid group-aware importance vector")
        rank_order = [int(value) for value in torch.argsort(importance, stable=True)[:remove_count].tolist()]
        selected = sorted(rank_order)
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=selected)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"DepGraph rejected {group_id} at {ratio_percent(fraction)}%")
        operations = base.operation_records(group, module_to_path)
        if base.operation_skeleton(operations) != base.operation_skeleton(canonical["operations"]):
            raise RuntimeError("Live DepGraph operation family differs from canonical evidence")
        group.prune()
    finally:
        if "forward" in head.__dict__:
            delattr(head, "forward")
    model.eval()
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "method": "live DepGraph generic output-channel group",
        "representative_root": row["representative_root"],
        "rule_family": "GENERIC_DEPGRAPH",
        "selection_unit": "output_channel",
        "root_channels_before": channels_before,
        "root_channels_removed": remove_count,
        "root_channels_after": int(root.out_channels),
        "actual_root_fraction": remove_count / channels_before,
        "selected_indices": selected,
        "selection_rank_order": rank_order,
        "selected_scores_rank_order": [float(importance[index]) for index in rank_order],
        "operations": operations,
        "operation_count": len(operations),
        "operation_family_match": True,
    }


def group_worker(domain: str, group_id: str, output: Path, fraction: Fraction) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    started = time.perf_counter()
    output = output.resolve()
    result_path = output / "runs" / f"{domain}_{group_id}.json"
    config = baseline_config(domain)
    checkpoint = PROJECT_ROOT / config["path"]
    dataset = PROJECT_ROOT / config["dataset_yaml"]
    baseline_reference_path = PROJECT_ROOT / config["baseline_reference"]
    record: dict[str, Any] = {
        "schema": f"t1_t2_ratio_{ratio_slug(fraction)}_run_v1",
        "run_id": f"{domain}_{group_id}_{ratio_slug(fraction)}",
        "domain": domain,
        "group_id": group_id,
        "group_kind": "CUSTOM" if group_id.startswith("CDG") else "GENERIC",
        "status": "FAIL",
        "requested_fraction": float(fraction),
        "requested_percent": ratio_percent(fraction),
        "policy": {
            "fresh_unpruned_baseline": True,
            "isolated_group": True,
            "fine_tuning": False,
            "batchnorm_update": False,
            "test_data_used": False,
        },
        "inputs": {
            "checkpoint": base.relative(checkpoint),
            "checkpoint_sha256": config["sha256"],
            "dataset_yaml": base.relative(dataset),
            "dataset_yaml_sha256": config["dataset_yaml_sha256"],
            "baseline_reference": base.relative(baseline_reference_path),
            "baseline_reference_sha256": config["baseline_reference_sha256"],
            "evaluation_config": base.relative(full.EVAL_PATH),
        },
    }
    monitor = base.MemoryMonitor()
    try:
        preflight(fraction, require_cuda=True)
        baseline_reference = base.read_json(baseline_reference_path)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with monitor:
            checkpoint_hash = base.sha256(checkpoint)
            yolo = YOLO(str(checkpoint), task="detect")
            model = yolo.model.float().cpu().eval()
            parameters_before = sum(parameter.numel() for parameter in model.parameters())
            gflops_before = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_before")
            with torch.inference_mode():
                before_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                public_before = base.public_prediction_summary(before_output)
                del before_output

            intervention = apply_custom(model, group_id, fraction) if group_id.startswith("CDG") else apply_generic(model, group_id, fraction)
            parameters_after = sum(parameter.numel() for parameter in model.parameters())
            gflops_after = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_after")
            if parameters_after >= parameters_before or gflops_after >= gflops_before:
                raise RuntimeError("Pruning did not reduce both parameters and measured GFLOPs")
            with torch.inference_mode():
                after_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_after = base.output_summary(after_output)
                public_after = base.public_prediction_summary(after_output)
            if not native_after["all_finite"] or public_after["shapes"] != public_before["shapes"]:
                raise RuntimeError("Pruned model violated the public output contract")

            with tempfile.TemporaryDirectory(prefix=f"ratio_reload_{domain.lower()}_{group_id.lower()}_") as temporary:
                model_path = Path(temporary) / "model.pth"
                torch.save(model, model_path)
                reloaded = torch.load(model_path, map_location="cpu", weights_only=False).float().cpu().eval()
                with torch.inference_mode():
                    reloaded_output = reloaded(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                reload_comparison = tensor_comparison(after_output, reloaded_output)
            if not reload_comparison["exact"]:
                raise RuntimeError("Saved/reloaded output is not exactly equal")
            del after_output, reloaded_output, reloaded

            yolo.model = model
            overall, per_class = full.evaluate(yolo, dataset, domain, record["run_id"])
            changes: dict[str, dict[str, float]] = {}
            metric_keys = [
                "map50_95", *(full.metric_key(iou) for iou in full.IOU_THRESHOLDS),
                "precision", "recall", "mean_class_f1", "f1_from_mean_precision_recall",
            ]
            for key in metric_keys:
                baseline_value = float(baseline_reference["metrics"][key])
                current = float(overall[key])
                signed_drop = baseline_value - current
                changes[key] = {
                    "baseline": baseline_value,
                    "current": current,
                    "signed_drop": signed_drop,
                    "normalized_signed_drop": signed_drop / baseline_value if abs(baseline_value) >= 1e-12 else 0.0,
                }
            if base.sha256(checkpoint) != checkpoint_hash or checkpoint_hash != config["sha256"]:
                raise RuntimeError("Canonical baseline checkpoint changed")
            record.update({
                "status": "PASS",
                "canonical_checkpoint_modified": False,
                "intervention": intervention,
                "structure": {
                    "parameters_before": parameters_before,
                    "parameters_after": parameters_after,
                    "parameters_removed": parameters_before - parameters_after,
                    "parameter_reduction_percent": 100.0 * (parameters_before - parameters_after) / parameters_before,
                    "gflops_before": gflops_before,
                    "gflops_after": gflops_after,
                    "gflops_removed": gflops_before - gflops_after,
                    "gflops_reduction_percent": 100.0 * (gflops_before - gflops_after) / gflops_before,
                    "native_output_after": native_after,
                    "public_prediction_before": public_before,
                    "public_prediction_after": public_after,
                    "save_reload_comparison": reload_comparison,
                },
                "metrics": overall,
                "metric_changes": changes,
                "per_class_metrics": per_class,
            })
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        record["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_memory_bytes": monitor.peak,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        base.atomic_json(result_path, record)
        try:
            del model, yolo
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if record["status"] == "PASS" else 2


TABLE_FIELDS = [
    "rank_by_AD_ascending", "ratio_percent", "domain", "group_id", "group_kind", "representative_root",
    "rule_family", "root_channels_before", "root_channels_removed", "root_channels_after", "actual_root_fraction",
    "parameters_before", "parameters_after", "parameters_removed", "parameter_reduction_percent",
    "gflops_before", "gflops_after", "gflops_removed", "gflops_reduction_percent",
    "baseline_map50_95", "pruned_map50_95", "AD_map50_95", "NAD_map50_95",
    "map50", "map75", "map95", "precision", "recall", "mean_class_f1",
    "operation_count", "save_reload_exact", "save_reload_max_absolute_error", "status", "record_path",
]

PER_CLASS_FIELDS = [
    "ratio_percent", "domain", "group_id", "group_kind", "class_id", "class_name",
    "validation_images_with_class", "validation_instances", "precision", "recall", "f1",
    "map50_95", "ap50", "ap55", "ap60", "ap65", "ap70", "ap75", "ap80", "ap85",
    "ap90", "ap95",
]


def table_row(record: dict[str, Any], output: Path) -> dict[str, Any]:
    intervention = record["intervention"]
    structure = record["structure"]
    metrics = record["metrics"]
    change = record["metric_changes"]["map50_95"]
    return {
        "rank_by_AD_ascending": "",
        "ratio_percent": record["requested_percent"],
        "domain": record["domain"],
        "group_id": record["group_id"],
        "group_kind": record["group_kind"],
        "representative_root": intervention["representative_root"],
        "rule_family": intervention["rule_family"],
        "root_channels_before": intervention["root_channels_before"],
        "root_channels_removed": intervention["root_channels_removed"],
        "root_channels_after": intervention["root_channels_after"],
        "actual_root_fraction": intervention["actual_root_fraction"],
        "parameters_before": structure["parameters_before"],
        "parameters_after": structure["parameters_after"],
        "parameters_removed": structure["parameters_removed"],
        "parameter_reduction_percent": structure["parameter_reduction_percent"],
        "gflops_before": structure["gflops_before"],
        "gflops_after": structure["gflops_after"],
        "gflops_removed": structure["gflops_removed"],
        "gflops_reduction_percent": structure["gflops_reduction_percent"],
        "baseline_map50_95": change["baseline"],
        "pruned_map50_95": change["current"],
        "AD_map50_95": change["signed_drop"],
        "NAD_map50_95": change["normalized_signed_drop"],
        "map50": metrics["ap50"],
        "map75": metrics["ap75"],
        "map95": metrics["ap95"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "mean_class_f1": metrics["mean_class_f1"],
        "operation_count": intervention["operation_count"],
        "save_reload_exact": structure["save_reload_comparison"]["exact"],
        "save_reload_max_absolute_error": structure["save_reload_comparison"]["max_absolute_error"],
        "status": record["status"],
        "record_path": base.relative(output / "runs" / f"{record['domain']}_{record['group_id']}.json"),
    }


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    try:
        temporary.replace(path)
    except PermissionError:
        live = path.with_name(f"{path.stem}_LIVE{path.suffix}")
        temporary.replace(live)
        print(f"WARNING: {path.name} is locked; wrote {live.name}", flush=True)


def build_tables(output: Path, fraction: Fraction) -> dict[str, Any]:
    output = output.resolve()
    groups = all_group_ids()
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted((output / "runs").glob("*.json")):
        record = base.read_json(path)
        if record.get("schema") != f"t1_t2_ratio_{ratio_slug(fraction)}_run_v1":
            raise RuntimeError(f"Unexpected record schema: {path}")
        if record.get("status") == "PASS":
            records.append(record)
        else:
            failures.append(record)
    expected_keys = {(domain, group_id) for group_id in groups for domain in ("GEN", "SNOW")}
    actual_keys = {(record["domain"], record["group_id"]) for record in [*records, *failures]}
    if len(actual_keys) != len(records) + len(failures) or not actual_keys <= expected_keys:
        raise RuntimeError("Ratio sweep contains duplicate or unknown records")
    rows = [table_row(record, output) for record in records]
    domain_tables: dict[str, list[dict[str, Any]]] = {}
    for domain in ("GEN", "SNOW"):
        selected = [row for row in rows if row["domain"] == domain]
        selected.sort(key=lambda row: (float(row["AD_map50_95"]), row["group_id"]))
        for rank, row in enumerate(selected, start=1):
            row["rank_by_AD_ascending"] = rank
        domain_tables[domain] = selected
    all_rows = sorted(rows, key=lambda row: (row["group_id"], row["domain"]))
    label = ratio_slug(fraction).upper()
    atomic_csv(output / "tables" / f"T1_GEN_{label}.csv", TABLE_FIELDS, domain_tables["GEN"])
    atomic_csv(output / "tables" / f"T2_SNOW_{label}.csv", TABLE_FIELDS, domain_tables["SNOW"])
    atomic_csv(output / "tables" / f"ALL_RESULTS_{label}.csv", TABLE_FIELDS, all_rows)

    per_class: list[dict[str, Any]] = []
    for record in records:
        for metric in record["per_class_metrics"]:
            per_class.append({
                "ratio_percent": record["requested_percent"],
                "domain": record["domain"],
                "group_id": record["group_id"],
                "group_kind": record["group_kind"],
                **metric,
            })
    per_class.sort(key=lambda row: (row["group_id"], row["domain"], int(row["class_id"])))
    atomic_csv(output / "tables" / f"PER_CLASS_RESULTS_{label}.csv", PER_CLASS_FIELDS, per_class)

    by_key = {(row["domain"], row["group_id"]): row for row in rows}
    paired: list[dict[str, Any]] = []
    for group_id in groups:
        gen = by_key.get(("GEN", group_id))
        snow = by_key.get(("SNOW", group_id))
        if gen and snow:
            paired.append({
                "ratio_percent": ratio_percent(fraction),
                "group_id": group_id,
                "group_kind": gen["group_kind"],
                "representative_root": gen["representative_root"],
                "parameters_removed": gen["parameters_removed"],
                "gflops_removed": gen["gflops_removed"],
                "gen_pruned_map50_95": gen["pruned_map50_95"],
                "gen_AD_map50_95": gen["AD_map50_95"],
                "gen_NAD_map50_95": gen["NAD_map50_95"],
                "snow_pruned_map50_95": snow["pruned_map50_95"],
                "snow_AD_map50_95": snow["AD_map50_95"],
                "snow_NAD_map50_95": snow["NAD_map50_95"],
                "directional_NAD_difference_GEN_minus_SNOW": float(gen["NAD_map50_95"]) - float(snow["NAD_map50_95"]),
            })
    paired_fields = list(paired[0]) if paired else [
        "ratio_percent", "group_id", "group_kind", "representative_root", "parameters_removed", "gflops_removed",
        "gen_pruned_map50_95", "gen_AD_map50_95", "gen_NAD_map50_95", "snow_pruned_map50_95",
        "snow_AD_map50_95", "snow_NAD_map50_95", "directional_NAD_difference_GEN_minus_SNOW",
    ]
    atomic_csv(output / "tables" / f"PAIRED_GEN_SNOW_{label}.csv", paired_fields, paired)

    progress = {
        "schema": f"t1_t2_ratio_{ratio_slug(fraction)}_progress_v1",
        "ratio_percent": ratio_percent(fraction),
        "expected_runs": 102,
        "successful_runs": len(records),
        "failed_runs": len(failures),
        "remaining_runs": 102 - len(records) - len(failures),
        "complete_pairs": len(paired),
        "complete": len(records) == 102 and not failures and len(paired) == 51,
    }
    base.atomic_json(output / "progress.json", progress)
    return progress


def execute_worker(command: list[str], log_path: Path, timeout_seconds: int, min_free_gb: float) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(command, cwd=str(STUDY_ROOT), stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - started > timeout_seconds:
                base.terminate_tree(process)
                raise TimeoutError(f"Worker exceeded {timeout_seconds} seconds; inspect {log_path}")
            if psutil.virtual_memory().available / (1024**3) < min_free_gb:
                base.terminate_tree(process)
                raise MemoryError(f"Available RAM fell below {min_free_gb:.1f} GiB; inspect {log_path}")
            time.sleep(2)
    return int(process.returncode or 0)


def write_manifest(output: Path, fraction: Fraction, evidence: dict[str, Any], launcher: Path) -> None:
    manifest = {
        "schema": f"t1_t2_ratio_{ratio_slug(fraction)}_manifest_v1",
        "status": "RUNNING",
        "ratio_fraction": str(fraction),
        "ratio_percent": ratio_percent(fraction),
        "expected_runs": 102,
        "groups": 51,
        "domains": ["GEN", "SNOW"],
        "policy": {
            "fresh_baseline_per_run": True,
            "isolated_group": True,
            "fine_tuning": False,
            "batchnorm_update": False,
            "cumulative_pruning": False,
            "test_data_used": False,
            "exact_ratio_required": True,
        },
        "preflight": evidence,
        "scripts": {
            base.relative(Path(__file__).resolve()): base.sha256(Path(__file__).resolve()),
            base.relative(launcher.resolve()): base.sha256(launcher.resolve()),
        },
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    base.atomic_json(output / "experiment_manifest.json", manifest)
    base.atomic_text(output / "README.md", f"""# Exhaustive {ratio_percent(fraction):g}% Isolated T1/T2 Sweep

This versioned experiment evaluates all 51 validated YOLO26n dependency groups independently on GEN and SNOW at an exact {ratio_percent(fraction):g}% logical/root pruning ratio. Every run starts from the untouched domain checkpoint. Generic groups rebuild DepGraph live; custom C3k2/C2PSA groups use their validated block-aware rules with ratio-aware channel or attention-unit selection.

No fine-tuning, BatchNorm update, cumulative pruning, or test data is used. `runs/` contains one atomic record per domain-group, `logs/` contains worker output, and `tables/` contains T1, T2, complete and paired supervisor-facing CSVs. Use the same launcher with `--retry-failed` to resume a failed record.
""")


def parent(
    fraction: Fraction,
    output: Path,
    launcher: Path,
    retry_failed: bool,
    timeout_seconds: int,
    min_free_gb: float,
) -> int:
    output = output.resolve()
    allowed = (STUDY_ROOT / "results" / "pruning").resolve()
    if allowed != output and allowed not in output.parents:
        raise RuntimeError(f"Output must stay below {allowed}")
    evidence = preflight(fraction, require_cuda=True)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, fraction, evidence, launcher)
    build_tables(output, fraction)
    queue = [(domain, group_id) for group_id in all_group_ids() for domain in ("SNOW", "GEN")]
    for position, (domain, group_id) in enumerate(queue, start=1):
        result_path = output / "runs" / f"{domain}_{group_id}.json"
        if result_path.is_file():
            existing = base.read_json(result_path)
            if existing.get("status") == "PASS":
                print(f"[{position}/102] {domain} {group_id}: already PASS", flush=True)
                continue
            if not retry_failed:
                print(f"[{position}/102] {domain} {group_id}: previous FAIL; rerun with --retry-failed", flush=True)
                return 2
        print(f"[{position}/102] {domain} {group_id}: launching", flush=True)
        command = [
            sys.executable, str(launcher.resolve()), "--group-worker", "--domain", domain,
            "--group", group_id, "--output", str(output),
        ]
        code = execute_worker(
            command, output / "logs" / f"{domain}_{group_id}.log", timeout_seconds, min_free_gb
        )
        progress = build_tables(output, fraction)
        if code != 0:
            print(f"{domain} {group_id}: FAIL; evidence retained in {result_path}", flush=True)
            return code
        print(f"[{position}/102] {domain} {group_id}: PASS", flush=True)
    progress = build_tables(output, fraction)
    if not progress["complete"]:
        raise RuntimeError(f"Ratio sweep ended incomplete: {progress}")
    manifest = base.read_json(output / "experiment_manifest.json")
    manifest.update({"status": "PASS", "completed_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    base.atomic_json(output / "experiment_manifest.json", manifest)
    print(json.dumps(progress, sort_keys=True), flush=True)
    return 0


def main_for_fraction(fraction: Fraction, default_output: Path, launcher: Path) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--group-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", choices=("GEN", "SNOW"), help=argparse.SUPPRESS)
    parser.add_argument("--group", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.group_worker:
        if args.domain is None or args.group is None:
            parser.error("--group-worker requires --domain and --group")
        if args.group not in all_group_ids():
            parser.error(f"Unknown group: {args.group}")
        return group_worker(args.domain, args.group, args.output, fraction)
    if args.preflight:
        print(json.dumps(preflight(fraction, require_cuda=False), indent=2, sort_keys=True))
        return 0
    return parent(fraction, args.output, launcher, args.retry_failed, args.timeout_seconds, args.min_free_gb)
