"""Run the frozen 9-group by 2-domain custom-rule T1/T2 accuracy sweep."""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import psutil
import yaml

import run_t1_t2 as base
import run_t1_t2_full_sweep as full
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


PROJECT_ROOT = base.PROJECT_ROOT
STUDY_ROOT = base.STUDY_ROOT
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "T1_T2_CUSTOM_SWEEP_FREEZE_V1.json"
EVAL_PATH = STUDY_ROOT / "configs" / "pruning" / "t1_t2_eval_v1.yaml"
CATALOGUE_PATH = STUDY_ROOT / "results" / "depgraph" / "custom_groups_v1" / "CUSTOM_GROUP_CATALOGUE.csv"
OPERATIONS_PATH = STUDY_ROOT / "results" / "depgraph" / "custom_groups_v1" / "CUSTOM_GROUP_OPERATIONS.json"
RULE_PATH = STUDY_ROOT / "scripts" / "yolo26n_custom_pruning_rules.py"
BUILDER_PATH = STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts" / "build_t1_t2_custom_tables.py"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "t1_t2_custom_sweep_v1"
RUN_SCHEMA = "t1_t2_custom_sweep_run_v1"
NONATTENTION_BLOCKS = (2, 4, 6, 8, 13, 16, 19)
SUPPORTED_BLOCKS = (*NONATTENTION_BLOCKS, 10, 22)
VALIDATION_SIZE = 640


def group_ids(freeze: dict[str, Any]) -> list[str]:
    return [str(value) for value in freeze["sweep_groups"]]


def catalogue_rows() -> dict[str, dict[str, str]]:
    return {row["custom_group_id"]: row for row in base.read_csv(CATALOGUE_PATH)}


def resolved_baseline_reference(baseline: dict[str, Any]) -> Path:
    """Resolve the frozen reference after the documented results reorganization."""

    configured = PROJECT_ROOT / baseline["baseline_reference"]
    if configured.is_file():
        return configured
    organized = (
        STUDY_ROOT
        / "results"
        / "pruning"
        / "prune_12_5"
        / "t1_t2_full_sweep_v2"
        / "baselines"
        / configured.name
    )
    if not organized.is_file():
        raise FileNotFoundError(
            f"Baseline reference is missing from configured and organized paths: "
            f"{configured}; {organized}"
        )
    return organized


def preflight(require_cuda: bool = True) -> dict[str, Any]:
    import torch

    generic = full.preflight(require_cuda=require_cuda)
    freeze = base.read_json(FREEZE_PATH)
    if freeze.get("freeze_id") != "T1_T2_CUSTOM_SWEEP_FREEZE_V1" or freeze.get("status") != "FROZEN":
        raise RuntimeError("The expected custom-sweep freeze is not active")

    for relative_path, evidence in freeze["files"].items():
        if "sha256" not in evidence:
            continue
        base.verify_file(PROJECT_ROOT / relative_path, evidence["sha256"], evidence.get("bytes"))
    for domain, baseline in freeze["baseline_models"].items():
        base.verify_file(PROJECT_ROOT / baseline["path"], baseline["sha256"])
        base.verify_file(PROJECT_ROOT / baseline["dataset_yaml"], baseline["dataset_yaml_sha256"])
        reference_path = resolved_baseline_reference(baseline)
        base.verify_file(reference_path, baseline["baseline_reference_sha256"])
        reference = base.read_json(reference_path)
        if reference.get("status") != "PASS" or reference.get("domain") != domain:
            raise RuntimeError(f"Frozen baseline reference is invalid for {domain}")
        if abs(float(reference["metrics"]["map50_95"]) - float(baseline["frozen_map50_95"])) > 1e-10:
            raise RuntimeError(f"Frozen baseline metric disagrees for {domain}")

    groups = group_ids(freeze)
    catalogue = catalogue_rows()
    if len(groups) != 9 or len(set(groups)) != 9 or set(groups) != set(catalogue):
        raise RuntimeError("Custom sweep membership does not exactly match the nine-group catalogue")
    blocks = []
    for group_id in groups:
        row = catalogue[group_id]
        blocks.append(int(row["block_index"]))
        if row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS":
            raise RuntimeError(f"Custom group lacks physical validation: {group_id}")
        if row["accuracy_status"] != "NOT_EVALUATED":
            raise RuntimeError(f"Frozen source catalogue was unexpectedly changed: {group_id}")
        if abs(float(row["actual_logical_fraction"]) - 0.125) > 1e-12:
            raise RuntimeError(f"Custom group fraction is not one eighth: {group_id}")
    if sorted(blocks) != sorted(SUPPORTED_BLOCKS):
        raise RuntimeError("Custom block coverage differs from the physically validated V1 set")

    evaluation = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    expected_eval = {
        "imgsz": 640, "batch": 16, "device": 0, "workers": 2, "split": "val",
        "rect": True, "conf": 0.001, "iou": 0.7, "max_det": 300, "half": False,
        "dnn": False, "augment": False, "agnostic_nms": False, "cache": False,
        "plots": False, "save_json": False, "verbose": False,
    }
    for key, expected in expected_eval.items():
        if evaluation.get(key) != expected:
            raise RuntimeError(f"Frozen evaluation setting changed: {key}")
    return {
        "freeze_id": freeze["freeze_id"],
        "custom_groups": len(groups),
        "expected_domain_group_runs": len(groups) * 2,
        "generic_preflight": generic,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "physical_memory_bytes": psutil.virtual_memory().total,
    }


def evaluate(
    yolo: Any, dataset: Path, domain: str, run_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    evaluation.pop("task", None)
    evaluation.pop("mode", None)
    names = {int(key): str(value) for key, value in yolo.names.items()}
    freeze = base.read_json(FREEZE_PATH)
    expected = freeze["baseline_models"][domain]
    with tempfile.TemporaryDirectory(prefix=f"custom_sweep_{domain.lower()}_") as temporary:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            metrics = yolo.val(
                data=str(dataset), project=temporary, name="validation", exist_ok=True, **evaluation
            )
        overall, per_class = full.expanded_metrics(
            metrics,
            names,
            expected_images=int(expected["validation_images"]),
            expected_instances=int(expected["validation_instances"]),
        )
    print(
        json.dumps(
            {
                "run": run_id,
                "stage": "validation_complete",
                "map50_95": overall["map50_95"],
                "map50": overall["ap50"],
                "map75": overall["ap75"],
                "map95": overall["ap95"],
                "precision": overall["precision"],
                "recall": overall["recall"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return overall, per_class


def _apply_custom_rule(block: Any, block_index: int) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    import torch

    hidden_before = int(block.c)
    remove_count = hidden_before // 8
    if remove_count < 1 or hidden_before % 8:
        raise RuntimeError("The frozen one-eighth intervention is not integral")

    if block_index in NONATTENTION_BLOCKS:
        invariants_before = validate_nonattention_c3k2_invariants(block)
        importance = nonattention_c3k2_logical_importance(block)
        selected_rank_order = [
            int(index) for index in torch.argsort(importance, stable=True)[:remove_count].tolist()
        ]
        selected_indices = sorted(selected_rank_order)
        selection_unit = "hidden_channel"
        result = prune_nonattention_c3k2_logical_channels(
            block, selected_indices, module_path=f"model.{block_index}"
        )
        invariants_after = validate_nonattention_c3k2_invariants(block)
    elif block_index == 10:
        invariants_before = validate_c2psa_invariants(block)
        importance, layout = c2psa_head_aware_importance(block)
        selected_indices = select_balanced_attention_units(importance, layout)
        selected_rank_order = sorted(selected_indices, key=lambda index: (float(importance[index]), index))
        selection_unit = "paired_attention_unit"
        result = prune_c2psa_head_aware_units(
            block, selected_indices, module_path=f"model.{block_index}"
        )
        invariants_after = validate_c2psa_invariants(block)
    elif block_index == 22:
        invariants_before = validate_attention_c3k2_invariants(block)
        importance, layout = attention_c3k2_head_aware_importance(block)
        selected_indices = select_balanced_attention_units(importance, layout)
        selected_rank_order = sorted(selected_indices, key=lambda index: (float(importance[index]), index))
        selection_unit = "paired_attention_unit"
        result = prune_attention_c3k2_head_aware_units(
            block, selected_indices, module_path=f"model.{block_index}"
        )
        invariants_after = validate_attention_c3k2_invariants(block)
    else:
        raise ValueError(f"Unsupported custom block: {block_index}")

    selected_scores = [float(importance[index]) for index in selected_rank_order]
    importance_record = {
        "logical_channel_count": hidden_before,
        "selection_unit": selection_unit,
        "selected_channel_indices": selected_indices,
        "selected_indices": selected_indices,
        "selection_rank_order": selected_rank_order,
        "selected_scores_rank_order": selected_scores,
        "minimum_selected_score": min(selected_scores),
        "maximum_selected_score": max(selected_scores),
    }
    evidence = {
        "result": result.to_dict(),
        "invariants_before": invariants_before,
        "invariants_after": invariants_after,
    }
    return result, importance_record, evidence


def group_worker(domain: str, group_id: str, output: Path) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    started = time.perf_counter()
    freeze = base.read_json(FREEZE_PATH)
    if group_id not in group_ids(freeze):
        raise ValueError(f"Group is outside the custom sweep: {group_id}")
    baseline = freeze["baseline_models"][domain]
    catalogue = catalogue_rows()[group_id]
    block_index = int(catalogue["block_index"])
    checkpoint = PROJECT_ROOT / baseline["path"]
    dataset = PROJECT_ROOT / baseline["dataset_yaml"]
    baseline_reference_path = resolved_baseline_reference(baseline)
    result_path = output.resolve() / "runs" / f"{domain}_{group_id}.json"
    record: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "run_id": f"{domain}_{group_id}",
        "domain": domain,
        "canonical_group_id": group_id,
        "status": "FAIL",
        "group_catalogue": {
            "block_path": catalogue["block_path"],
            "block_index": block_index,
            "rule_family": catalogue["rule_family"],
            "original_deferred_root_count": int(catalogue["original_deferred_root_count"]),
            "original_depgraph_ids": catalogue["original_depgraph_ids"].split(";"),
            "original_root_paths": catalogue["original_root_paths"].split(";"),
            "operation_skeleton_sha256": catalogue["operation_skeleton_sha256"],
        },
        "method": {
            "requested_logical_fraction": 0.125,
            "importance": "mean-normalized coupled-slice L1 group mean",
            "fine_tuning": False,
            "batchnorm_update": False,
            "cross_group_accumulation": False,
        },
        "inputs": {
            "checkpoint": base.relative(checkpoint),
            "checkpoint_sha256": baseline["sha256"],
            "dataset_yaml": base.relative(dataset),
            "dataset_yaml_sha256": baseline["dataset_yaml_sha256"],
            "freeze": base.relative(FREEZE_PATH),
            "evaluation_config": base.relative(EVAL_PATH),
            "baseline_reference": base.relative(baseline_reference_path),
            "custom_rule_source": base.relative(RULE_PATH),
        },
    }

    def stage(name: str, **details: Any) -> None:
        print(json.dumps({"run": record["run_id"], "stage": name, **details}, sort_keys=True), flush=True)

    monitor = base.MemoryMonitor()
    try:
        preflight(require_cuda=True)
        baseline_reference = base.read_json(baseline_reference_path)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with monitor:
            stage("loading_checkpoint")
            checkpoint_hash_before = base.sha256(checkpoint)
            yolo = YOLO(str(checkpoint), task="detect")
            model = yolo.model.float().cpu().eval()
            block = model.model[block_index]
            hidden_before = int(block.c)
            parameters_before = sum(parameter.numel() for parameter in model.parameters())
            block_parameters_before = sum(parameter.numel() for parameter in block.parameters())
            gflops_before = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_before")
            with torch.inference_mode():
                before_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_before = base.output_summary(before_output)
                public_before = base.public_prediction_summary(before_output)
                del before_output
            if not native_before["all_finite"]:
                raise RuntimeError("Unpruned model produced non-finite output")

            stage("applying_custom_rule", block=block_index, logical_channels=hidden_before // 8)
            pruning, importance, custom_evidence = _apply_custom_rule(block, block_index)
            model.eval()
            model.zero_grad(set_to_none=True)
            parameters_after = sum(parameter.numel() for parameter in model.parameters())
            block_parameters_after = sum(parameter.numel() for parameter in block.parameters())
            gflops_after = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_after")
            parameters_removed = parameters_before - parameters_after
            gflops_removed = gflops_before - gflops_after
            if parameters_removed <= 0 or gflops_removed <= 0:
                raise RuntimeError("Custom rule did not reduce physical parameters and GFLOPs")
            expected_parameters = int(catalogue[f"{domain.lower()}_parameters_removed"])
            expected_gflops = float(catalogue[f"{domain.lower()}_gflops_removed"])
            if parameters_removed != expected_parameters:
                raise RuntimeError("Parameter change differs from physical-validation evidence")
            if abs(gflops_removed - expected_gflops) > 1e-10:
                raise RuntimeError("GFLOP change differs from physical-validation evidence")

            with torch.inference_mode():
                after_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_after = base.output_summary(after_output)
                public_after = base.public_prediction_summary(after_output)
                del after_output
            if not native_after["all_finite"]:
                raise RuntimeError("Pruned model produced non-finite output")
            if public_after["shapes"] != public_before["shapes"]:
                raise RuntimeError("Public prediction contract changed")

            operations = []
            for operation in custom_evidence["result"]["operations"]:
                operations.append(
                    {
                        "target_module_path": operation["module_path"],
                        "target_module_type": "custom_rule_target",
                        "handler": operation["operation"],
                        "indices": operation["indices"],
                        "channels_before": operation["channels_before"],
                        "channels_after": operation["channels_after"],
                    }
                )
            if len(operations) != int(catalogue["operation_count"]):
                raise RuntimeError("Custom operation count differs from frozen catalogue")

            yolo.model = model
            stage("validation_started", domain=domain)
            overall, per_class = evaluate(yolo, dataset, domain, record["run_id"])
            reference_metrics = baseline_reference["metrics"]
            changes: dict[str, dict[str, float]] = {}
            keys = [
                "map50_95", *(full.metric_key(iou) for iou in full.IOU_THRESHOLDS),
                "precision", "recall", "mean_class_f1", "f1_from_mean_precision_recall",
            ]
            for key in keys:
                reference = float(reference_metrics[key])
                pruned = float(overall[key])
                signed_drop = reference - pruned
                changes[key] = {
                    "baseline": reference,
                    "pruned": pruned,
                    "signed_drop": signed_drop,
                    "normalized_signed_drop": signed_drop / reference if abs(reference) >= 1e-12 else 0.0,
                }
            if base.sha256(checkpoint) != checkpoint_hash_before or checkpoint_hash_before != baseline["sha256"]:
                raise RuntimeError("Canonical checkpoint changed during the run")

            record.update(
                {
                    "status": "PASS",
                    "canonical_checkpoint_modified": False,
                    "importance": importance,
                    "dependency": {
                        "operation_family_match": True,
                        "operation_count": len(operations),
                        "operations": operations,
                    },
                    "custom_rule_evidence": custom_evidence,
                    "structure": {
                        "root_channels_before": hidden_before,
                        "root_channels_removed": pruning.hidden_channels_removed,
                        "root_channels_after": pruning.hidden_channels_after,
                        "actual_root_fraction": pruning.hidden_channels_removed / hidden_before,
                        "parameters_before": parameters_before,
                        "parameters_after": parameters_after,
                        "parameters_removed": parameters_removed,
                        "parameter_reduction_percent": 100.0 * parameters_removed / parameters_before,
                        "block_parameters_before": block_parameters_before,
                        "block_parameters_after": block_parameters_after,
                        "block_parameters_removed": block_parameters_before - block_parameters_after,
                        "gflops_before": gflops_before,
                        "gflops_after": gflops_after,
                        "gflops_removed": gflops_removed,
                        "gflops_reduction_percent": 100.0 * gflops_removed / gflops_before,
                        "native_output_before": native_before,
                        "native_output_after": native_after,
                        "public_prediction_before": public_before,
                        "public_prediction_after": public_after,
                    },
                    "metrics": overall,
                    "metric_changes": changes,
                    "per_class_metrics": per_class,
                }
            )
    except Exception as error:
        record["error"] = {
            "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()
        }
    finally:
        record["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_memory_bytes": monitor.peak,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        base.atomic_json(result_path, record)
        try:
            del metrics, model, yolo, block
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if record["status"] == "PASS" else 2


def build_tables(output: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--output", str(output)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Custom-sweep table builder failed")


def write_manifest(output: Path, check: dict[str, Any]) -> None:
    manifest = {
        "schema_version": "t1_t2_custom_sweep_manifest_v1",
        "freeze_id": "T1_T2_CUSTOM_SWEEP_FREEZE_V1",
        "created_local_date": time.strftime("%Y-%m-%d"),
        "execution_policy": "one fresh sequential subprocess per domain-group; frozen generic baseline references reused",
        "expected_domain_group_runs": 18,
        "preflight": check,
        "scripts": {
            base.relative(Path(__file__)): base.sha256(Path(__file__)),
            base.relative(BUILDER_PATH): base.sha256(BUILDER_PATH),
            base.relative(RULE_PATH): base.sha256(RULE_PATH),
        },
        "freeze_sha256": base.sha256(FREEZE_PATH),
    }
    base.atomic_json(output / "experiment_manifest.json", manifest)
    base.atomic_text(
        output / "README.md",
        "# T1/T2 Custom Sweep V1\n\n"
        "This folder contains the frozen nine-custom-group by two-domain immediate post-pruning accuracy extension. "
        "Each run starts from the untouched domain checkpoint and uses the physically validated custom C3k2/C3k/C2PSA rule. "
        "No fine-tuning, BatchNorm update or cumulative pruning is performed. `runs/` contains atomic evidence and `tables/` "
        "contains custom-only and combined supervisor-facing CSVs. The completed generic raw sweep is not modified.\n",
    )


def execute_worker(command: list[str], timeout_seconds: int, min_free_gb: float) -> int:
    process = subprocess.Popen(
        command, cwd=str(PROJECT_ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    started = time.monotonic()
    while process.poll() is None:
        if time.monotonic() - started > timeout_seconds:
            base.terminate_tree(process)
            raise TimeoutError(f"Worker exceeded {timeout_seconds} seconds")
        if psutil.virtual_memory().available / (1024**3) < min_free_gb:
            base.terminate_tree(process)
            raise MemoryError(f"Available RAM fell below {min_free_gb:.1f} GiB")
        time.sleep(2)
    return int(process.returncode or 0)


def parent(output: Path, retry_failed: bool, timeout_seconds: int, min_free_gb: float) -> int:
    output = output.resolve()
    allowed = (STUDY_ROOT / "results" / "pruning").resolve()
    if allowed != output and allowed not in output.parents:
        raise RuntimeError(f"Output must stay below {allowed}")
    check = preflight(require_cuda=True)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, check)
    build_tables(output)
    freeze = base.read_json(FREEZE_PATH)
    queue = [(domain, group_id) for group_id in group_ids(freeze) for domain in ("SNOW", "GEN")]
    for position, (domain, group_id) in enumerate(queue, start=1):
        result_path = output / "runs" / f"{domain}_{group_id}.json"
        if result_path.is_file():
            previous = base.read_json(result_path)
            if previous.get("status") == "PASS":
                print(f"[{position}/18] {domain} {group_id}: already PASS", flush=True)
                continue
            if not retry_failed:
                print(f"[{position}/18] {domain} {group_id}: previous FAIL; use --retry-failed", flush=True)
                return 2
        print(f"[{position}/18] {domain} {group_id}: launching", flush=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--group-worker", "--domain", domain,
            "--group", group_id, "--output", str(output),
        ]
        code = execute_worker(command, timeout_seconds, min_free_gb)
        build_tables(output)
        if code != 0:
            print(f"{domain} {group_id} failed; structured evidence retained", flush=True)
            return code
        print(f"[{position}/18] {domain} {group_id}: PASS", flush=True)

    progress = base.read_json(output / "progress.json")
    if not progress.get("complete"):
        raise RuntimeError(f"Custom sweep ended incomplete: {progress}")
    print(json.dumps(progress, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--group-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", choices=("GEN", "SNOW"), help=argparse.SUPPRESS)
    parser.add_argument("--group", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.group_worker:
        if not args.domain or not args.group:
            parser.error("--group-worker requires --domain and --group")
        return group_worker(args.domain, args.group, args.output)
    if args.preflight:
        print(json.dumps(preflight(require_cuda=True), indent=2, sort_keys=True))
        return 0
    return parent(args.output, args.retry_failed, args.timeout_seconds, args.min_free_gb)


if __name__ == "__main__":
    raise SystemExit(main())
