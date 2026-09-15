"""Execute the frozen V2 T1/T2 sweep over all 42 validated generic groups."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import gc
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
import types
from typing import Any

import numpy as np
import psutil
import yaml

import run_t1_t2 as base


PROJECT_ROOT = base.PROJECT_ROOT
STUDY_ROOT = base.STUDY_ROOT
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "T1_T2_FULL_SWEEP_FREEZE_V2.json"
EVAL_PATH = STUDY_ROOT / "configs" / "pruning" / "t1_t2_eval_v1.yaml"
CATALOGUE_PATH = base.CATALOGUE_PATH
OPERATIONS_PATH = base.OPERATIONS_PATH
ENVIRONMENT_PATH = base.ENVIRONMENT_PATH
BUILDER_PATH = STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts" / "build_t1_t2_full_tables.py"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "t1_t2_full_sweep_v2"
RUN_SCHEMA = "t1_t2_full_sweep_run_v2"
BASELINE_SCHEMA = "t1_t2_full_sweep_baseline_v2"
IOU_THRESHOLDS = tuple(round(0.50 + 0.05 * index, 2) for index in range(10))
TRACE_SIZE = 32
VALIDATION_SIZE = 640


def group_ids(freeze: dict[str, Any]) -> list[str]:
    return [str(value) for value in freeze["sweep_groups"]]


def metric_key(iou: float) -> str:
    return f"ap{int(round(iou * 100)):02d}"


def preflight(require_cuda: bool = True) -> dict[str, Any]:
    import torch
    import ultralytics

    freeze = base.read_json(FREEZE_PATH)
    if freeze.get("freeze_id") != "T1_T2_FULL_SWEEP_FREEZE_V2" or freeze.get("status") != "FROZEN":
        raise RuntimeError("The expected V2 full-sweep freeze is not active")
    for rel_path, evidence in freeze["files"].items():
        base.verify_file(PROJECT_ROOT / rel_path, evidence["sha256"], evidence["bytes"])
    for domain, baseline in freeze["baseline_models"].items():
        base.verify_file(PROJECT_ROOT / baseline["path"], baseline["sha256"])
        base.verify_file(PROJECT_ROOT / baseline["dataset_yaml"], baseline["dataset_yaml_sha256"])
        if not 0.0 < float(baseline["frozen_map50_95"]) <= 1.0:
            raise RuntimeError(f"Invalid frozen baseline metric for {domain}")

    environment = base.read_json(ENVIRONMENT_PATH)
    versions = {
        "torch": torch.__version__,
        "torch_pruning": version("torch-pruning"),
        "ultralytics": ultralytics.__version__,
    }
    for name, actual in versions.items():
        if str(environment[name]) != str(actual):
            raise RuntimeError(f"Environment changed for {name}: frozen={environment[name]}, current={actual}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen evaluator")

    catalogue = {row["canonical_group_id"]: row for row in base.read_csv(CATALOGUE_PATH)}
    operations = base.read_json(OPERATIONS_PATH)
    groups = group_ids(freeze)
    if len(groups) != 42 or len(set(groups)) != 42 or set(groups) != set(catalogue):
        raise RuntimeError("V2 sweep group membership does not exactly match the 42-group catalogue")
    for group_id in groups:
        row = catalogue[group_id]
        if row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS":
            raise RuntimeError(f"Group lacks physical validation in both domains: {group_id}")
        if row["operation_signature_sha256"] != operations[group_id]["operation_signature_sha256"]:
            raise RuntimeError(f"Canonical operation evidence disagrees for {group_id}")

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
        "groups": len(groups),
        "expected_domain_group_runs": len(groups) * 2,
        "versions": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "physical_memory_bytes": psutil.virtual_memory().total,
    }


def expanded_metrics(
    metrics: Any,
    names: dict[int, str],
    expected_images: int,
    expected_instances: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    all_ap = np.asarray(metrics.box.all_ap, dtype=np.float64)
    class_ids = [int(value) for value in metrics.box.ap_class_index]
    if all_ap.ndim != 2 or all_ap.shape != (len(class_ids), len(IOU_THRESHOLDS)):
        raise RuntimeError(f"Unexpected AP matrix shape: {all_ap.shape}")
    if not class_ids or not np.isfinite(all_ap).all():
        raise RuntimeError("AP evidence is empty or non-finite")
    if len(set(class_ids)) != len(class_ids):
        raise RuntimeError("Validation returned duplicate AP class IDs")

    threshold_means = all_ap.mean(axis=0)
    overall: dict[str, Any] = {
        "map50_95": base.finite_metric(metrics.box.map, "map50_95"),
        "precision": base.finite_metric(metrics.box.mp, "precision"),
        "recall": base.finite_metric(metrics.box.mr, "recall"),
        "mean_class_f1": base.finite_metric(np.asarray(metrics.box.f1).mean(), "mean_class_f1"),
        "validation_images": int(expected_images),
        "validation_instances": int(np.asarray(metrics.nt_per_class).sum()),
        "speed_ms_per_image": {
            str(key): base.finite_metric(value, f"speed_{key}")
            for key, value in dict(metrics.speed).items()
        },
    }
    denominator = overall["precision"] + overall["recall"]
    overall["f1_from_mean_precision_recall"] = (
        0.0 if denominator == 0 else 2.0 * overall["precision"] * overall["recall"] / denominator
    )
    if overall["validation_instances"] != int(expected_instances):
        raise RuntimeError(
            f"Validation instance support changed: {overall['validation_instances']} vs {expected_instances}"
        )
    for index, iou in enumerate(IOU_THRESHOLDS):
        overall[metric_key(iou)] = float(threshold_means[index])
    if abs(overall["ap50"] - float(metrics.box.map50)) > 1e-10:
        raise RuntimeError("AP50 disagrees with the validator top-level metric")
    if abs(overall["ap75"] - float(metrics.box.map75)) > 1e-10:
        raise RuntimeError("AP75 disagrees with the validator top-level metric")
    if abs(overall["map50_95"] - float(threshold_means.mean())) > 1e-10:
        raise RuntimeError("mAP50-95 disagrees with the complete AP curve")

    names = {int(key): str(value) for key, value in names.items()}
    supports_i = np.asarray(metrics.nt_per_class)
    supports_img = np.asarray(metrics.nt_per_image)
    class_to_row = {class_id: row for row, class_id in enumerate(class_ids)}
    per_class: list[dict[str, Any]] = []
    for class_id in range(len(names)):
        row_index = class_to_row.get(class_id)
        record: dict[str, Any] = {
            "class_id": class_id,
            "class_name": names[class_id],
            "validation_images_with_class": int(supports_img[class_id]),
            "validation_instances": int(supports_i[class_id]),
        }
        if row_index is None:
            if record["validation_instances"] != 0:
                raise RuntimeError(f"Class {class_id} has targets but no AP result")
            record.update({"precision": None, "recall": None, "f1": None, "map50_95": None})
            for iou in IOU_THRESHOLDS:
                record[metric_key(iou)] = None
        else:
            record.update(
                {
                    "precision": base.finite_metric(metrics.box.p[row_index], "class_precision"),
                    "recall": base.finite_metric(metrics.box.r[row_index], "class_recall"),
                    "f1": base.finite_metric(metrics.box.f1[row_index], "class_f1"),
                    "map50_95": float(all_ap[row_index].mean()),
                }
            )
            for index, iou in enumerate(IOU_THRESHOLDS):
                record[metric_key(iou)] = float(all_ap[row_index, index])
        per_class.append(record)
    return overall, per_class


def evaluate(yolo: Any, dataset: Path, domain: str, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    evaluation.pop("task", None)
    evaluation.pop("mode", None)
    names = {int(key): str(value) for key, value in yolo.names.items()}
    with tempfile.TemporaryDirectory(prefix=f"full_sweep_{domain.lower()}_") as temporary:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            metrics = yolo.val(
                data=str(dataset), project=temporary, name="validation", exist_ok=True, **evaluation
            )
        expected = base.read_json(FREEZE_PATH)["baseline_models"][domain]
        overall, per_class = expanded_metrics(
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


def baseline_worker(domain: str, output: Path) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    started = time.perf_counter()
    freeze = base.read_json(FREEZE_PATH)
    baseline = freeze["baseline_models"][domain]
    checkpoint = PROJECT_ROOT / baseline["path"]
    dataset = PROJECT_ROOT / baseline["dataset_yaml"]
    result_path = output.resolve() / "baselines" / f"{domain}.json"
    record: dict[str, Any] = {
        "schema_version": BASELINE_SCHEMA,
        "run_id": f"BASELINE_{domain}",
        "domain": domain,
        "status": "FAIL",
        "inputs": {
            "checkpoint": base.relative(checkpoint),
            "checkpoint_sha256": baseline["sha256"],
            "dataset_yaml": base.relative(dataset),
            "dataset_yaml_sha256": baseline["dataset_yaml_sha256"],
            "freeze": base.relative(FREEZE_PATH),
            "evaluation_config": base.relative(EVAL_PATH),
        },
    }
    monitor = base.MemoryMonitor()
    try:
        preflight(require_cuda=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with monitor:
            print(json.dumps({"run": record["run_id"], "stage": "loading_checkpoint"}), flush=True)
            yolo = YOLO(str(checkpoint), task="detect")
            model = yolo.model.float().cpu().eval()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            gflops = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "baseline_gflops")
            with torch.inference_mode():
                native = base.output_summary(model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE)))
            if not native["all_finite"]:
                raise RuntimeError("Baseline native inference is non-finite")
            yolo.model = model
            overall, per_class = evaluate(yolo, dataset, domain, record["run_id"])
            if abs(overall["map50_95"] - float(baseline["frozen_map50_95"])) > 1e-10:
                raise RuntimeError(
                    f"Expanded reference mAP does not reproduce frozen baseline: "
                    f"{overall['map50_95']} vs {baseline['frozen_map50_95']}"
                )
            if base.sha256(checkpoint) != baseline["sha256"]:
                raise RuntimeError("Canonical baseline checkpoint changed")
            record.update(
                {
                    "status": "PASS",
                    "canonical_checkpoint_modified": False,
                    "structure": {"parameters": parameters, "gflops": gflops, "native_output": native},
                    "metrics": overall,
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
            del metrics, model, yolo
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if record["status"] == "PASS" else 2


def operation_skeleton(records: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    return base.operation_skeleton(records)


def group_worker(domain: str, group_id: str, output: Path) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    started = time.perf_counter()
    freeze = base.read_json(FREEZE_PATH)
    groups = group_ids(freeze)
    if group_id not in groups:
        raise ValueError(f"Group is outside the V2 sweep: {group_id}")
    baseline = freeze["baseline_models"][domain]
    catalogue = {row["canonical_group_id"]: row for row in base.read_csv(CATALOGUE_PATH)}[group_id]
    canonical = base.read_json(OPERATIONS_PATH)[group_id]
    checkpoint = PROJECT_ROOT / baseline["path"]
    dataset = PROJECT_ROOT / baseline["dataset_yaml"]
    baseline_reference_path = output.resolve() / "baselines" / f"{domain}.json"
    result_path = output.resolve() / "runs" / f"{domain}_{group_id}.json"
    record: dict[str, Any] = {
        "schema_version": RUN_SCHEMA,
        "run_id": f"{domain}_{group_id}",
        "domain": domain,
        "canonical_group_id": group_id,
        "status": "FAIL",
        "group_catalogue": {
            "representative_root": catalogue["representative_root"],
            "alias_roots": [item.strip() for item in catalogue["alias_roots"].split(";")],
            "original_group_ids": [item.strip() for item in catalogue["original_group_ids"].split(";")],
            "canonical_operation_signature_sha256": canonical["operation_signature_sha256"],
        },
        "method": {
            "requested_root_fraction": 0.125,
            "importance": "GroupMagnitudeImportance(p=1, group_reduction='mean', normalizer='mean', bias=False)",
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
        },
    }

    def stage(name: str, **details: Any) -> None:
        print(json.dumps({"run": record["run_id"], "stage": name, **details}, sort_keys=True), flush=True)

    monitor = base.MemoryMonitor()
    try:
        preflight(require_cuda=True)
        if not baseline_reference_path.is_file():
            raise FileNotFoundError(f"Expanded baseline reference is missing: {baseline_reference_path}")
        baseline_reference = base.read_json(baseline_reference_path)
        if baseline_reference.get("status") != "PASS":
            raise RuntimeError(f"Expanded baseline reference is not PASS for {domain}")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with monitor:
            stage("loading_checkpoint")
            yolo = YOLO(str(checkpoint), task="detect")
            model = yolo.model.float().cpu().eval()
            checkpoint_hash_before = base.sha256(checkpoint)
            parameters_before = sum(parameter.numel() for parameter in model.parameters())
            gflops_before = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_before")
            with torch.inference_mode():
                before_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_before = base.output_summary(before_output)
                public_before = base.public_prediction_summary(before_output)
                del before_output
            if not native_before["all_finite"]:
                raise RuntimeError("Unpruned model produced non-finite output")

            for parameter in model.parameters():
                parameter.requires_grad_(True)
            named_modules = dict(model.named_modules())
            module_to_path = {id(module): path for path, module in named_modules.items()}
            root = base.find_module(model, catalogue["representative_root"])
            if not isinstance(root, torch.nn.Conv2d):
                raise TypeError("Canonical root is not Conv2d")
            channels_before = int(root.out_channels)
            if channels_before != int(catalogue["root_out_channels"]):
                raise RuntimeError("Root channels differ from canonical catalogue")
            channels_to_remove = channels_before // 8
            if channels_to_remove < 1:
                raise RuntimeError("Frozen fraction removes no root channels")

            head = model.model[-1]
            if not getattr(head, "end2end", False):
                raise RuntimeError("Expected YOLO26 end-to-end Detect head")
            head.forward = types.MethodType(base.trace_detect_forward, head)

            class TraceWrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, images):
                    tensors = tuple(base.flatten_tensors(self.inner(images)))
                    if not tensors or not all(tensor.requires_grad for tensor in tensors):
                        raise RuntimeError("Trace outputs did not retain Autograd dependencies")
                    return tensors

            wrapper = TraceWrapper(model)
            example = torch.zeros(1, 3, TRACE_SIZE, TRACE_SIZE)
            stage("building_dependency_graph")
            dependency_graph = tp.DependencyGraph().build_dependency(wrapper, example_inputs=example)
            importance_group = dependency_graph.get_pruning_group(
                root, tp.prune_conv_out_channels, idxs=list(range(channels_before))
            )
            importance_fn = tp.importance.GroupMagnitudeImportance(
                p=1, group_reduction="mean", normalizer="mean", bias=False
            )
            importance = importance_fn(importance_group)
            if importance is None or importance.numel() != channels_before or not bool(torch.isfinite(importance).all()):
                raise RuntimeError("Invalid group-aware importance vector")
            ranked_indices = [int(value) for value in torch.argsort(importance, stable=True)[:channels_to_remove].tolist()]
            selected_indices = sorted(ranked_indices)
            selected_scores = [float(importance[index]) for index in ranked_indices]
            group = dependency_graph.get_pruning_group(
                root, tp.prune_conv_out_channels, idxs=selected_indices
            )
            if not dependency_graph.check_pruning_group(group):
                raise RuntimeError("DepGraph rejected the selected 12.5% group")
            actual_operations = base.operation_records(group, module_to_path)
            if operation_skeleton(actual_operations) != operation_skeleton(canonical["operations"]):
                raise RuntimeError("Live DepGraph operation family differs from canonical evidence")
            stage("applying_pruning", channels=channels_to_remove, operations=len(group))
            group.prune()
            delattr(head, "forward")
            model.eval()
            model.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            del group, importance_group, dependency_graph, wrapper, example, importance
            gc.collect()

            parameters_after = sum(parameter.numel() for parameter in model.parameters())
            gflops_after = base.finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_after")
            if int(root.out_channels) != channels_before - channels_to_remove:
                raise RuntimeError("Root channel reduction differs from frozen rule")
            if parameters_after >= parameters_before or gflops_after >= gflops_before:
                raise RuntimeError("Physical parameter or GFLOP count did not decrease")
            with torch.inference_mode():
                after_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_after = base.output_summary(after_output)
                public_after = base.public_prediction_summary(after_output)
                del after_output
            if not native_after["all_finite"]:
                raise RuntimeError("Pruned native output is non-finite")
            if public_after["shapes"] != public_before["shapes"]:
                raise RuntimeError("Public prediction contract changed")

            yolo.model = model
            stage("validation_started", domain=domain)
            overall, per_class = evaluate(yolo, dataset, domain, record["run_id"])
            reference_metrics = baseline_reference["metrics"]
            drops: dict[str, dict[str, float]] = {}
            accuracy_keys = ["map50_95", *(metric_key(iou) for iou in IOU_THRESHOLDS), "precision", "recall", "mean_class_f1", "f1_from_mean_precision_recall"]
            for key in accuracy_keys:
                reference = float(reference_metrics[key])
                pruned = float(overall[key])
                signed_drop = reference - pruned
                drops[key] = {
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
                    "importance": {
                        "selected_channel_indices": selected_indices,
                        "selection_rank_order": ranked_indices,
                        "selected_scores_rank_order": selected_scores,
                        "minimum_selected_score": min(selected_scores),
                        "maximum_selected_score": max(selected_scores),
                    },
                    "dependency": {
                        "operation_family_match": True,
                        "operation_count": len(actual_operations),
                        "operations": actual_operations,
                    },
                    "structure": {
                        "root_channels_before": channels_before,
                        "root_channels_removed": channels_to_remove,
                        "root_channels_after": int(root.out_channels),
                        "actual_root_fraction": channels_to_remove / channels_before,
                        "parameters_before": parameters_before,
                        "parameters_after": parameters_after,
                        "parameters_removed": parameters_before - parameters_after,
                        "parameter_reduction_percent": 100.0 * (parameters_before - parameters_after) / parameters_before,
                        "gflops_before": gflops_before,
                        "gflops_after": gflops_after,
                        "gflops_removed": gflops_before - gflops_after,
                        "gflops_reduction_percent": 100.0 * (gflops_before - gflops_after) / gflops_before,
                        "native_output_before": native_before,
                        "native_output_after": native_after,
                        "public_prediction_before": public_before,
                        "public_prediction_after": public_after,
                    },
                    "metrics": overall,
                    "metric_changes": drops,
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
            del metrics, model, yolo
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if record["status"] == "PASS" else 2


def build_tables(output: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--output", str(output)], cwd=str(PROJECT_ROOT), check=False
    )
    if result.returncode != 0:
        raise RuntimeError("Full-sweep table builder failed")


def write_manifest(output: Path, check: dict[str, Any]) -> None:
    freeze = base.read_json(FREEZE_PATH)
    manifest = {
        "schema_version": "t1_t2_full_sweep_manifest_v2",
        "freeze_id": freeze["freeze_id"],
        "created_local_date": time.strftime("%Y-%m-%d"),
        "execution_policy": "baseline references first; then one fresh sequential subprocess per domain-group",
        "expected_baseline_runs": 2,
        "expected_domain_group_runs": 84,
        "preflight": check,
        "scripts": {
            base.relative(Path(__file__)): base.sha256(Path(__file__)),
            base.relative(BUILDER_PATH): base.sha256(BUILDER_PATH),
        },
        "freeze_sha256": base.sha256(FREEZE_PATH),
    }
    base.atomic_json(output / "experiment_manifest.json", manifest)
    base.atomic_text(
        output / "README.md",
        "# T1/T2 Full Sweep V2\n\nThis folder contains the frozen 42-group by two-domain immediate post-pruning sweep. `baselines/` contains expanded unpruned metric references; `runs/` contains atomic group results; and `tables/` contains regenerated supervisor-facing CSVs. No pruned model, prediction dump, or validator directory is retained. The 19 deferred special-family roots are outside this versioned sweep.\n",
    )


def execute_worker(command: list[str], timeout_seconds: int, min_free_gb: float) -> int:
    # Per-run scientific evidence is written atomically to JSON. Suppressing the
    # child console stream avoids thousands of validator progress updates and a
    # Windows broken-pipe failure during long resumable sweeps.
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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

    for position, domain in enumerate(("SNOW", "GEN"), start=1):
        result_path = output / "baselines" / f"{domain}.json"
        if result_path.is_file() and base.read_json(result_path).get("status") == "PASS":
            print(f"[baseline {position}/2] {domain}: already PASS", flush=True)
            continue
        if result_path.is_file() and not retry_failed:
            print(f"[baseline {position}/2] {domain}: previous FAIL; use --retry-failed", flush=True)
            return 2
        print(f"[baseline {position}/2] {domain}: launching", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--baseline-worker", "--domain", domain, "--output", str(output)]
        code = execute_worker(command, timeout_seconds, min_free_gb)
        build_tables(output)
        if code != 0:
            print(f"Baseline {domain} failed; structured evidence retained", flush=True)
            return code

    queue = [(domain, group_id) for group_id in group_ids(freeze) for domain in ("SNOW", "GEN")]
    for position, (domain, group_id) in enumerate(queue, start=1):
        result_path = output / "runs" / f"{domain}_{group_id}.json"
        if result_path.is_file():
            previous = base.read_json(result_path)
            if previous.get("status") == "PASS":
                print(f"[{position}/84] {domain} {group_id}: already PASS", flush=True)
                continue
            if not retry_failed:
                print(f"[{position}/84] {domain} {group_id}: previous FAIL; use --retry-failed", flush=True)
                return 2
        print(f"[{position}/84] {domain} {group_id}: launching", flush=True)
        command = [
            sys.executable, str(Path(__file__).resolve()), "--group-worker", "--domain", domain,
            "--group", group_id, "--output", str(output),
        ]
        code = execute_worker(command, timeout_seconds, min_free_gb)
        build_tables(output)
        if code != 0:
            print(f"{domain} {group_id} failed; structured evidence retained", flush=True)
            return code
        print(f"[{position}/84] {domain} {group_id}: PASS", flush=True)

    progress = base.read_json(output / "progress.json")
    if not progress.get("complete"):
        raise RuntimeError(f"Full sweep ended incomplete: {progress}")
    print(json.dumps(progress, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--baseline-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--group-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", choices=("GEN", "SNOW"), help=argparse.SUPPRESS)
    parser.add_argument("--group", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.baseline_worker:
        if not args.domain:
            parser.error("--baseline-worker requires --domain")
        return baseline_worker(args.domain, args.output)
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
