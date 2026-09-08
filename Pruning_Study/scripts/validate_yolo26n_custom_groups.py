"""Memory-safe physical validation for YOLO26 custom logical-width groups.

Each domain/block pair runs in a fresh subprocess.  This first validator covers
the non-attention C3k2/C3k family and records structural, forward, FLOP,
parameter, save/reload and reproducibility evidence.  It does not evaluate
detection accuracy and does not modify the sealed generic DepGraph catalogue.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Iterable

import psutil

from depgraph_yolo26n_safe_audit import CHECKPOINTS, PROJECT_ROOT, atomic_json, sha256
from yolo26n_custom_pruning_rules import (
    attention_c3k2_head_aware_importance,
    c2psa_head_aware_importance,
    nonattention_c3k2_logical_importance,
    prune_attention_c3k2_head_aware_units,
    prune_c2psa_head_aware_units,
    prune_nonattention_c3k2_logical_channels,
    select_balanced_attention_units,
    validate_attention_c3k2_invariants,
    validate_c2psa_invariants,
    validate_nonattention_c3k2_invariants,
)


STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "depgraph" / "custom_groups_v1"
SCHEMA_VERSION = "YOLO26N_CUSTOM_GROUP_PHYSICAL_V1"
NONATTENTION_BLOCKS = (2, 4, 6, 8, 13, 16, 19)
SUPPORTED_BLOCKS = (*NONATTENTION_BLOCKS, 10, 22)
SMALL_SIZE = 64
FULL_SIZE = 640


def flatten_tensors(value: Any) -> Iterable[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_tensors(item)


def output_summary(value: Any) -> dict[str, Any]:
    import torch

    tensors = list(flatten_tensors(value))
    return {
        "tensor_count": len(tensors),
        "shapes": [list(tensor.shape) for tensor in tensors],
        "all_finite": bool(tensors) and all(bool(torch.isfinite(tensor).all()) for tensor in tensors),
    }


def forward_summary(model: Any, size: int) -> dict[str, Any]:
    import torch

    with torch.inference_mode():
        output = model(torch.zeros(1, 3, size, size))
    summary = output_summary(output)
    del output
    if not summary["all_finite"]:
        raise RuntimeError(f"Model produced a non-finite output at {size} pixels")
    return summary


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def worker(domain: str, block_index: int, output: Path) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    if domain not in CHECKPOINTS:
        raise ValueError(f"Unknown domain: {domain}")
    if block_index not in SUPPORTED_BLOCKS:
        raise ValueError(f"Block {block_index} is outside the custom-family V1 set")

    checkpoint = CHECKPOINTS[domain]
    run_id = f"{domain}_C3K2_BLOCK_{block_index:02d}"
    run_dir = output / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / f"{run_id}.json"
    scratch_path = run_dir / f".{run_id}_reload_test.pth"
    process = psutil.Process(os.getpid())
    started = time.perf_counter()
    peak_rss = process.memory_info().rss

    def stage(name: str, **details: Any) -> None:
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        print(json.dumps({"run": run_id, "stage": name, **details}, sort_keys=True), flush=True)

    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "status": "FAIL",
        "domain": domain,
        "block_index": block_index,
        "block_path": f"model.{block_index}",
        "family": (
            "C3k2_NONATTENTION_LOGICAL_WIDTH_V1"
            if block_index in NONATTENTION_BLOCKS
            else "C2PSA_HEAD_AWARE_LOGICAL_WIDTH_V1"
            if block_index == 10
            else "C3K2_ATTENTION_HEAD_AWARE_LOGICAL_WIDTH_V1"
        ),
        "method": {
            "requested_logical_fraction": 0.125,
            "importance": "mean-normalized coupled-slice L1 group mean",
            "accuracy_evaluation": False,
            "fine_tuning": False,
            "batchnorm_update": False,
        },
        "inputs": {
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "rule_source": "Pruning_Study/scripts/yolo26n_custom_pruning_rules.py",
            "validator_source": "Pruning_Study/scripts/validate_yolo26n_custom_groups.py",
        },
    }

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        stage("loading_checkpoint")
        checkpoint_hash_before = sha256(checkpoint)
        yolo = YOLO(str(checkpoint), task="detect")
        model = yolo.model.float().cpu().eval()
        block = model.model[block_index]
        if block_index in NONATTENTION_BLOCKS:
            invariants_before = validate_nonattention_c3k2_invariants(block)
        elif block_index == 10:
            invariants_before = validate_c2psa_invariants(block)
        else:
            invariants_before = validate_attention_c3k2_invariants(block)
        hidden_before = int(block.c)
        remove_count = hidden_before // 8
        if remove_count < 1 or hidden_before % 8:
            raise RuntimeError("The frozen one-eighth intervention is not integral")

        parameters_before = sum(parameter.numel() for parameter in model.parameters())
        block_parameters_before = sum(parameter.numel() for parameter in block.parameters())
        stage("baseline_structure", parameters=parameters_before, hidden_channels=hidden_before)
        gflops_before = float(get_flops(model, imgsz=FULL_SIZE))
        small_before = forward_summary(model, SMALL_SIZE)
        full_before = forward_summary(model, FULL_SIZE)

        stage("calculating_logical_importance")
        if block_index in NONATTENTION_BLOCKS:
            importance = nonattention_c3k2_logical_importance(block)
            layout = None
            selected_rank_order = [
                int(index) for index in torch.argsort(importance, stable=True)[:remove_count].tolist()
            ]
            selected_indices = sorted(selected_rank_order)
        elif block_index == 10:
            importance, layout = c2psa_head_aware_importance(block)
            selected_indices = select_balanced_attention_units(importance, layout)
            selected_rank_order = sorted(selected_indices, key=lambda index: (float(importance[index]), index))
        else:
            importance, layout = attention_c3k2_head_aware_importance(block)
            selected_indices = select_balanced_attention_units(importance, layout)
            selected_rank_order = sorted(selected_indices, key=lambda index: (float(importance[index]), index))
        all_scores = [float(value) for value in importance.tolist()]
        rank_order = [int(index) for index in torch.argsort(importance, stable=True).tolist()]
        selected_scores = [float(importance[index]) for index in selected_rank_order]

        stage(
            "applying_custom_rule",
            logical_channels=remove_count,
            selected_indices=selected_indices,
            selection_unit="hidden_channel" if layout is None else "paired_attention_unit",
        )
        if block_index in NONATTENTION_BLOCKS:
            pruning = prune_nonattention_c3k2_logical_channels(
                block, selected_indices, module_path=f"model.{block_index}"
            )
        elif block_index == 10:
            pruning = prune_c2psa_head_aware_units(block, selected_indices, module_path=f"model.{block_index}")
        else:
            pruning = prune_attention_c3k2_head_aware_units(
                block, selected_indices, module_path=f"model.{block_index}"
            )
        model.eval()
        model.zero_grad(set_to_none=True)
        if block_index in NONATTENTION_BLOCKS:
            invariants_after = validate_nonattention_c3k2_invariants(block)
        elif block_index == 10:
            invariants_after = validate_c2psa_invariants(block)
        else:
            invariants_after = validate_attention_c3k2_invariants(block)
        parameters_after = sum(parameter.numel() for parameter in model.parameters())
        block_parameters_after = sum(parameter.numel() for parameter in block.parameters())
        gflops_after = float(get_flops(model, imgsz=FULL_SIZE))
        if parameters_after >= parameters_before:
            raise RuntimeError("Custom rule did not reduce physical parameters")
        if gflops_after >= gflops_before:
            raise RuntimeError("Custom rule did not reduce profiled GFLOPs")

        stage("physical_forward_validation")
        small_after = forward_summary(model, SMALL_SIZE)
        full_after = forward_summary(model, FULL_SIZE)
        if small_after["shapes"] != small_before["shapes"]:
            raise RuntimeError("Small-input external output contract changed")
        if full_after["shapes"] != full_before["shapes"]:
            raise RuntimeError("Full-input external output contract changed")

        stage("save_reload_validation")
        torch.save(model, scratch_path)
        del block, model, yolo, importance
        gc.collect()
        restored = torch.load(scratch_path, map_location="cpu", weights_only=False)
        restored.eval()
        restored_block = restored.model[block_index]
        if block_index in NONATTENTION_BLOCKS:
            invariants_reloaded = validate_nonattention_c3k2_invariants(restored_block)
        elif block_index == 10:
            invariants_reloaded = validate_c2psa_invariants(restored_block)
        else:
            invariants_reloaded = validate_attention_c3k2_invariants(restored_block)
        reload_small = forward_summary(restored, SMALL_SIZE)
        reload_full = forward_summary(restored, FULL_SIZE)
        if reload_small["shapes"] != small_before["shapes"]:
            raise RuntimeError("Reloaded small-input output contract changed")
        if reload_full["shapes"] != full_before["shapes"]:
            raise RuntimeError("Reloaded full-input output contract changed")
        if sum(parameter.numel() for parameter in restored.parameters()) != parameters_after:
            raise RuntimeError("Reloaded parameter count differs from the pruned model")
        if sha256(checkpoint) != checkpoint_hash_before:
            raise RuntimeError("Canonical checkpoint changed during validation")

        record.update(
            {
                "status": "PASS",
                "canonical_checkpoint_modified": False,
                "importance": {
                    "logical_channel_count": hidden_before,
                    "selection_unit": "hidden_channel" if layout is None else "paired_attention_unit",
                    "selected_indices": selected_indices,
                    "selection_rank_order": selected_rank_order,
                    "global_rank_order": rank_order,
                    "selected_scores_rank_order": selected_scores,
                    "all_scores": all_scores,
                },
                "pruning": pruning.to_dict(),
                "invariants": {
                    "before": invariants_before,
                    "after": invariants_after,
                    "after_reload": invariants_reloaded,
                },
                "structure": {
                    "hidden_channels_before": hidden_before,
                    "hidden_channels_removed": pruning.hidden_channels_removed,
                    "hidden_channels_after": int(restored_block.c),
                    "actual_logical_fraction": pruning.hidden_channels_removed / hidden_before,
                    "parameters_before": parameters_before,
                    "parameters_after": parameters_after,
                    "parameters_removed": parameters_before - parameters_after,
                    "parameter_reduction_percent": 100.0 * (parameters_before - parameters_after) / parameters_before,
                    "block_parameters_before": block_parameters_before,
                    "block_parameters_after": block_parameters_after,
                    "block_parameters_removed": block_parameters_before - block_parameters_after,
                    "gflops_before": gflops_before,
                    "gflops_after": gflops_after,
                    "gflops_removed": gflops_before - gflops_after,
                    "gflops_reduction_percent": 100.0 * (gflops_before - gflops_after) / gflops_before,
                },
                "forwards": {
                    "small_before": small_before,
                    "small_after": small_after,
                    "full_before": full_before,
                    "full_after": full_after,
                    "reload_small": reload_small,
                    "reload_full": reload_full,
                },
            }
        )
        del restored, restored_block
        gc.collect()
    except Exception as error:
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        if scratch_path.exists():
            scratch_path.unlink()
        peak_rss = max(peak_rss, process.memory_info().rss)
        record["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_observed_process_rss_bytes": peak_rss,
        }
        atomic_json(result_path, record)
        stage("complete", status=record["status"], result=relative(result_path))
        gc.collect()
    return 0 if record["status"] == "PASS" else 2


def write_summary(output: Path) -> None:
    rows: list[dict[str, Any]] = []
    for path in sorted((output / "runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        structure = record.get("structure", {})
        error = record.get("error", {})
        rows.append(
            {
                "run_id": record.get("run_id", ""),
                "domain": record.get("domain", ""),
                "block_index": record.get("block_index", ""),
                "block_path": record.get("block_path", ""),
                "family": record.get("family", ""),
                "status": record.get("status", ""),
                "hidden_channels_before": structure.get("hidden_channels_before", ""),
                "hidden_channels_removed": structure.get("hidden_channels_removed", ""),
                "hidden_channels_after": structure.get("hidden_channels_after", ""),
                "parameters_removed": structure.get("parameters_removed", ""),
                "parameter_reduction_percent": structure.get("parameter_reduction_percent", ""),
                "gflops_removed": structure.get("gflops_removed", ""),
                "gflops_reduction_percent": structure.get("gflops_reduction_percent", ""),
                "selected_indices": ";".join(
                    str(index) for index in record.get("importance", {}).get("selected_indices", [])
                ),
                "embedding_indices": ";".join(
                    str(index) for index in record.get("pruning", {}).get("embedding_indices", [])
                ),
                "elapsed_seconds": record.get("resources", {}).get("elapsed_seconds", ""),
                "peak_observed_process_rss_bytes": record.get("resources", {}).get(
                    "peak_observed_process_rss_bytes", ""
                ),
                "error_type": error.get("type", ""),
                "error_message": error.get("message", ""),
                "record_path": relative(path),
            }
        )
    fields = list(rows[0]) if rows else ["run_id", "status"]
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "PHYSICAL_VALIDATION_SUMMARY.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(
        output / "status.json",
        {
            "schema_version": SCHEMA_VERSION,
            "runs": len(rows),
            "passed": sum(row["status"] == "PASS" for row in rows),
            "failed": sum(row["status"] != "PASS" for row in rows),
            "accuracy_evaluation_performed": False,
            "sealed_generic_catalogue_modified": False,
        },
    )


def orchestrate(domains: list[str], blocks: list[int], output: Path) -> int:
    output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for domain in domains:
        for block_index in blocks:
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--domain",
                domain,
                "--block",
                str(block_index),
                "--output",
                str(output.resolve()),
            ]
            result = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
            if result.returncode:
                failures += 1
    write_summary(output)
    return 0 if failures == 0 else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--domains", nargs="+", choices=sorted(CHECKPOINTS), default=["GEN", "SNOW"])
    parser.add_argument("--blocks", nargs="+", type=int, default=[2, 8])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", choices=sorted(CHECKPOINTS), help=argparse.SUPPRESS)
    parser.add_argument("--block", type=int, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        if args.domain is None or args.block is None:
            raise SystemExit("--worker requires --domain and --block")
        return worker(args.domain, args.block, args.output.resolve())
    invalid = sorted(set(args.blocks) - set(SUPPORTED_BLOCKS))
    if invalid:
        raise SystemExit(f"Unsupported custom-family blocks: {invalid}")
    return orchestrate(args.domains, args.blocks, args.output.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
