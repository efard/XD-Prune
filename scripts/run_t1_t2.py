"""Run the frozen six-run YOLO26n T1/T2 pruning pilot safely and resumably.

Each domain/group measurement is executed in a fresh subprocess.  Only compact
JSON evidence is retained; the pruned in-memory model and validator artefacts
are discarded after every run.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import gc
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
from typing import Any, Iterable

import psutil
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "T1_T2_EXPERIMENT_FREEZE_V1.json"
EVAL_PATH = STUDY_ROOT / "configs" / "pruning" / "t1_t2_eval_v1.yaml"
CATALOGUE_PATH = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3" / "canonical_validated_groups.csv"
OPERATIONS_PATH = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3" / "canonical_group_operations.json"
ENVIRONMENT_PATH = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3" / "environment.json"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "pruning" / "t1_t2_pilot_v1"
BUILDER_PATH = STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts" / "build_t1_t2_tables.py"
TRACE_SIZE = 32
VALIDATION_SIZE = 640
SCHEMA_VERSION = "t1_t2_pilot_run_v1"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def flatten_tensors(value: Any) -> Iterable[Any]:
    import torch

    if torch.is_tensor(value):
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


def public_prediction_summary(value: Any) -> dict[str, Any]:
    """Summarize the public NMS prediction tensor, excluding internal feature maps."""

    import torch

    prediction = value[0] if isinstance(value, (list, tuple)) and value else value
    if not torch.is_tensor(prediction):
        raise RuntimeError("YOLO native output does not begin with a public prediction tensor")
    return output_summary(prediction)


def trace_detect_forward(self, features):
    """Trace-only Detect forward that exposes both one-to-many and one-to-one dependencies."""

    one2many = self.forward_head(features, **self.one2many)
    one2one = self.forward_head(features, **self.one2one)
    return {"one2many": one2many, "one2one": one2one}


def find_module(model: Any, path: str) -> Any:
    modules = dict(model.named_modules())
    if path not in modules:
        raise KeyError(f"Model does not contain frozen root {path}")
    return modules[path]


def finite_metric(value: Any, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise RuntimeError(f"Metric {name} is non-finite: {converted}")
    return converted


def verify_file(path: Path, expected_hash: str, expected_bytes: int | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen file is missing: {path}")
    if expected_bytes is not None and path.stat().st_size != int(expected_bytes):
        raise RuntimeError(f"Frozen file size changed: {path}")
    actual = sha256(path)
    if actual != expected_hash.upper():
        raise RuntimeError(f"Frozen file hash changed: {path}\nexpected {expected_hash}\nactual   {actual}")


def preflight(require_cuda: bool = True) -> dict[str, Any]:
    import torch
    import torch_pruning as tp
    import ultralytics

    freeze = read_json(FREEZE_PATH)
    if freeze.get("status") != "FROZEN" or freeze.get("freeze_id") != "T1_T2_EXPERIMENT_FREEZE_V1":
        raise RuntimeError("T1/T2 freeze record is not the expected frozen version")
    for rel_path, evidence in freeze["files"].items():
        verify_file(PROJECT_ROOT / rel_path, evidence["sha256"], evidence["bytes"])
    for domain, baseline in freeze["baseline_models"].items():
        verify_file(PROJECT_ROOT / baseline["path"], baseline["sha256"])
        verify_file(PROJECT_ROOT / baseline["dataset_yaml"], baseline["dataset_yaml_sha256"])
        if not 0.0 < float(baseline["baseline_map50_95"]) <= 1.0:
            raise RuntimeError(f"Invalid frozen baseline mAP for {domain}")

    environment = read_json(ENVIRONMENT_PATH)
    actual_versions = {
        "torch": torch.__version__,
        # torch-pruning 1.6.1 retains an internal __version__ value of 1.6.0;
        # distribution metadata is the installed-package source of truth.
        "torch_pruning": version("torch-pruning"),
        "ultralytics": ultralytics.__version__,
    }
    for key, actual in actual_versions.items():
        if str(environment[key]) != str(actual):
            raise RuntimeError(f"Environment changed for {key}: frozen={environment[key]}, current={actual}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen evaluation configuration but is unavailable")

    catalogue = {row["canonical_group_id"]: row for row in read_csv(CATALOGUE_PATH)}
    operations = read_json(OPERATIONS_PATH)
    if len(catalogue) != int(freeze["group_counts"]["unique_primary_groups"]):
        raise RuntimeError("Canonical catalogue count differs from the freeze record")
    for group_id in freeze["pilot_groups"]:
        if group_id not in catalogue or group_id not in operations:
            raise RuntimeError(f"Pilot group is absent from canonical evidence: {group_id}")
        row = catalogue[group_id]
        if row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS":
            raise RuntimeError(f"Pilot group was not physically validated on both domains: {group_id}")
        if row["operation_signature_sha256"] != operations[group_id]["operation_signature_sha256"]:
            raise RuntimeError(f"Stored operation signatures disagree for {group_id}")

    eval_config = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    required_eval = {
        "imgsz": 640,
        "batch": 16,
        "device": 0,
        "workers": 2,
        "split": "val",
        "rect": True,
        "conf": 0.001,
        "iou": 0.7,
        "max_det": 300,
        "half": False,
        "augment": False,
        "cache": False,
        "plots": False,
        "save_json": False,
    }
    for key, expected in required_eval.items():
        if eval_config.get(key) != expected:
            raise RuntimeError(f"Frozen evaluation setting changed: {key}")
    return {
        "freeze_id": freeze["freeze_id"],
        "pilot_groups": freeze["pilot_groups"],
        "versions": actual_versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "physical_memory_bytes": psutil.virtual_memory().total,
    }


class MemoryMonitor:
    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)

    def _sample(self) -> None:
        processes = [self.process]
        try:
            processes.extend(self.process.children(recursive=True))
        except (psutil.Error, OSError):
            pass
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except (psutil.Error, OSError):
                continue
        self.peak = max(self.peak, total)

    def _sample_loop(self) -> None:
        while not self._stop.wait(0.2):
            self._sample()

    def __enter__(self):
        self._sample()
        self._thread.start()
        return self

    def __exit__(self, *_args: Any) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._sample()


def operation_records(group: Any, module_to_path: dict[int, str]) -> list[dict[str, Any]]:
    records = []
    for dependency, indices in group:
        target = dependency.target.module
        records.append(
            {
                "target_module_path": module_to_path.get(id(target), "<autograd-operation>"),
                "target_module_type": type(target).__name__,
                "handler": getattr(dependency.handler, "__name__", str(dependency.handler)),
                "indices": [int(index) for index in indices],
            }
        )
    return records


def operation_skeleton(records: list[dict[str, Any]]) -> Counter[tuple[str, str, str]]:
    return Counter(
        (record["target_module_path"], record["target_module_type"], record["handler"])
        for record in records
    )


def read_group_inputs(domain: str, group_id: str) -> tuple[dict[str, Any], dict[str, str], dict[str, Any]]:
    freeze = read_json(FREEZE_PATH)
    if domain not in freeze["baseline_models"]:
        raise ValueError(f"Unknown domain: {domain}")
    if group_id not in freeze["pilot_groups"]:
        raise ValueError(f"Group is not in the frozen pilot: {group_id}")
    catalogue = {row["canonical_group_id"]: row for row in read_csv(CATALOGUE_PATH)}
    operations = read_json(OPERATIONS_PATH)
    return freeze["baseline_models"][domain], catalogue[group_id], operations[group_id]


def run_worker(domain: str, group_id: str, output: Path) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    import torch_pruning as tp
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    from baseline_research_metrics import per_class_records

    output = output.resolve()
    runs_dir = output / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    result_path = runs_dir / f"{domain}_{group_id}.json"
    started = time.perf_counter()
    baseline, catalogue, canonical = read_group_inputs(domain, group_id)
    checkpoint = PROJECT_ROOT / baseline["path"]
    dataset = PROJECT_ROOT / baseline["dataset_yaml"]
    record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": f"{domain}_{group_id}",
        "domain": domain,
        "canonical_group_id": group_id,
        "status": "FAIL",
        "method": {
            "intervention": "isolated dependency-safe structured output-channel group pruning",
            "requested_root_fraction": 0.125,
            "importance": "GroupMagnitudeImportance(p=1, group_reduction='mean', normalizer='mean', bias=False)",
            "fine_tuning": False,
            "batchnorm_update": False,
        },
        "inputs": {
            "checkpoint": relative(checkpoint),
            "checkpoint_sha256": baseline["sha256"],
            "dataset_yaml": relative(dataset),
            "dataset_yaml_sha256": baseline["dataset_yaml_sha256"],
            "freeze": relative(FREEZE_PATH),
        },
    }

    def stage(name: str, **details: Any) -> None:
        message = {"run": record["run_id"], "stage": name, **details}
        print(json.dumps(message, sort_keys=True), flush=True)

    monitor = MemoryMonitor()
    try:
        preflight(require_cuda=True)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with monitor:
            stage("loading_checkpoint")
            yolo = YOLO(str(checkpoint), task="detect")
            model = yolo.model.float().cpu().eval()
            names = dict(yolo.names)
            checkpoint_hash_before = sha256(checkpoint)
            parameters_before = sum(parameter.numel() for parameter in model.parameters())
            gflops_before = finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_before")

            stage("native_baseline_forward")
            with torch.inference_mode():
                baseline_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_before = output_summary(baseline_output)
                public_before = public_prediction_summary(baseline_output)
                del baseline_output
            if not native_before["all_finite"]:
                raise RuntimeError("Unpruned model produced a non-finite native output")

            for parameter in model.parameters():
                parameter.requires_grad_(True)
            named_modules = dict(model.named_modules())
            module_to_path = {id(module): path for path, module in named_modules.items()}
            root = find_module(model, catalogue["representative_root"])
            if not isinstance(root, torch.nn.Conv2d):
                raise TypeError("Frozen group root is not Conv2d")
            channels_before = int(root.out_channels)
            if channels_before != int(catalogue["root_out_channels"]):
                raise RuntimeError("Root channel count differs from canonical catalogue")
            channels_to_remove = channels_before // 8
            if channels_to_remove < 1:
                raise RuntimeError("Frozen pruning fraction removes no root channels")

            head = model.model[-1]
            if not getattr(head, "end2end", False):
                raise RuntimeError("Expected the YOLO26 end-to-end Detect head")
            head.forward = types.MethodType(trace_detect_forward, head)

            class TraceWrapper(torch.nn.Module):
                def __init__(self, inner):
                    super().__init__()
                    self.inner = inner

                def forward(self, images):
                    tensors = tuple(flatten_tensors(self.inner(images)))
                    if not tensors or not all(tensor.requires_grad for tensor in tensors):
                        raise RuntimeError("Trace outputs did not retain Autograd dependencies")
                    return tensors

            wrapper = TraceWrapper(model)
            example = torch.zeros(1, 3, TRACE_SIZE, TRACE_SIZE)
            stage("building_dependency_graph", trace_size=TRACE_SIZE)
            dependency_graph = tp.DependencyGraph().build_dependency(wrapper, example_inputs=example)
            all_indices = list(range(channels_before))
            importance_group = dependency_graph.get_pruning_group(
                root, tp.prune_conv_out_channels, idxs=all_indices
            )
            # This all-index group is queried only so GroupMagnitudeImportance
            # can score every root channel. It must not pass the executable-group
            # check because actually removing every root channel would be invalid.
            importance_fn = tp.importance.GroupMagnitudeImportance(
                p=1, group_reduction="mean", normalizer="mean", bias=False
            )
            importance = importance_fn(importance_group)
            if importance is None or importance.numel() != channels_before:
                raise RuntimeError("Group-aware importance did not return one score per root channel")
            if not bool(torch.isfinite(importance).all()):
                raise RuntimeError("Group-aware importance contains non-finite values")
            ranked_indices = torch.argsort(importance, stable=True)[:channels_to_remove].tolist()
            selected_indices = sorted(int(index) for index in ranked_indices)
            selected_scores = [float(importance[index]) for index in ranked_indices]

            group = dependency_graph.get_pruning_group(
                root, tp.prune_conv_out_channels, idxs=selected_indices
            )
            if not dependency_graph.check_pruning_group(group):
                raise RuntimeError("DepGraph rejected the selected pruning group")
            actual_operations = operation_records(group, module_to_path)
            if operation_skeleton(actual_operations) != operation_skeleton(canonical["operations"]):
                raise RuntimeError("Live DepGraph operation family differs from the frozen canonical group")
            stage("applying_pruning", channels=channels_to_remove, operations=len(group))
            group.prune()

            # The trace adapter is only for dependency discovery. Real inference uses native Detect.
            delattr(head, "forward")
            model.eval()
            model.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            del group, importance_group, dependency_graph, wrapper, example, importance
            gc.collect()

            parameters_after = sum(parameter.numel() for parameter in model.parameters())
            gflops_after = finite_metric(get_flops(model, imgsz=VALIDATION_SIZE), "gflops_after")
            if int(root.out_channels) != channels_before - channels_to_remove:
                raise RuntimeError("Root channel count did not decrease by the frozen amount")
            if parameters_after >= parameters_before:
                raise RuntimeError("Physical parameter count did not decrease")
            if gflops_after >= gflops_before:
                raise RuntimeError("Profiled GFLOPs did not decrease")

            stage("native_pruned_forward")
            with torch.inference_mode():
                pruned_output = model(torch.zeros(1, 3, VALIDATION_SIZE, VALIDATION_SIZE))
                native_after = output_summary(pruned_output)
                public_after = public_prediction_summary(pruned_output)
                del pruned_output
            if not native_after["all_finite"]:
                raise RuntimeError("Pruned model produced a non-finite native output")
            if public_after["shapes"] != public_before["shapes"]:
                raise RuntimeError("Public detector prediction shape changed after internal group pruning")

            eval_config = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
            eval_config.pop("task", None)
            eval_config.pop("mode", None)
            stage("validation_started", dataset=domain)
            yolo.model = model
            with tempfile.TemporaryDirectory(prefix=f"t1t2_{domain.lower()}_{group_id.lower()}_") as temporary:
                metrics = yolo.val(
                    data=str(dataset),
                    project=temporary,
                    name="validation",
                    exist_ok=True,
                    **eval_config,
                )
                map50_95 = finite_metric(metrics.box.map, "map50_95")
                map50 = finite_metric(metrics.box.map50, "map50")
                map75 = finite_metric(metrics.box.map75, "map75")
                precision = finite_metric(metrics.box.mp, "precision")
                recall = finite_metric(metrics.box.mr, "recall")
                per_class = per_class_records(metrics, names)

            supported_ap = [row["ap50_95"] for row in per_class if row["ap50_95"] is not None]
            if not supported_ap or abs(sum(supported_ap) / len(supported_ap) - map50_95) > 1e-8:
                raise RuntimeError("Overall mAP50-95 is inconsistent with per-class AP evidence")
            baseline_map = float(baseline["baseline_map50_95"])
            signed_ad = baseline_map - map50_95
            normalized_signed_d = signed_ad / baseline_map
            checkpoint_hash_after = sha256(checkpoint)
            if checkpoint_hash_after != checkpoint_hash_before or checkpoint_hash_after != baseline["sha256"]:
                raise RuntimeError("Canonical checkpoint changed during the run")

            record.update(
                {
                    "status": "PASS",
                    "canonical_checkpoint_modified": False,
                    "importance": {
                        "selected_channel_indices": selected_indices,
                        "selection_rank_order": [int(index) for index in ranked_indices],
                        "selected_scores_rank_order": selected_scores,
                        "minimum": float(min(selected_scores)),
                        "maximum_selected": float(max(selected_scores)),
                    },
                    "dependency": {
                        "canonical_operation_signature_sha256": canonical["operation_signature_sha256"],
                        "operation_count": len(actual_operations),
                        "operation_family_match": True,
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
                        "gflops_before": gflops_before,
                        "gflops_after": gflops_after,
                        "gflops_removed": gflops_before - gflops_after,
                        "native_output_before": native_before,
                        "native_output_after": native_after,
                        "public_prediction_before": public_before,
                        "public_prediction_after": public_after,
                    },
                    "metrics": {
                        "primary_metric": "mAP50-95",
                        "baseline_map50_95": baseline_map,
                        "map50_95": map50_95,
                        "signed_AD": signed_ad,
                        "normalized_signed_D": normalized_signed_d,
                        "map50": map50,
                        "map75": map75,
                        "precision": precision,
                        "recall": recall,
                    },
                    "per_class_metrics": per_class,
                }
            )
            stage("validation_complete", map50_95=map50_95, signed_AD=signed_ad)
    except Exception as error:
        record["status"] = "FAIL"
        record["error"] = {
            "type": type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(),
        }
    finally:
        record["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_memory_bytes": monitor.peak,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        atomic_json(result_path, record)
        try:
            del metrics, model, yolo
        except UnboundLocalError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if record["status"] == "PASS" else 2


def write_experiment_files(output: Path, check: dict[str, Any]) -> None:
    freeze = read_json(FREEZE_PATH)
    queue = [
        {"domain": domain, "canonical_group_id": group_id, "run_id": f"{domain}_{group_id}"}
        for group_id in freeze["pilot_groups"]
        for domain in ("SNOW", "GEN")
    ]
    manifest = {
        "schema_version": "t1_t2_pilot_manifest_v1",
        "freeze_id": freeze["freeze_id"],
        "created_local_date": time.strftime("%Y-%m-%d"),
        "execution_policy": "one fresh sequential subprocess per domain/group; no parallel runs",
        "queue": queue,
        "preflight": check,
        "scripts": {
            relative(Path(__file__)): sha256(Path(__file__)),
            relative(BUILDER_PATH): sha256(BUILDER_PATH),
        },
    }
    atomic_json(output / "experiment_manifest.json", manifest)
    readme = """# T1/T2 Six-Run Pilot\n\nThis folder contains the frozen pilot measurements for three canonical DepGraph groups on GEN and SNOW. Each JSON in `runs/` is an independent immediate post-pruning evaluation: FP32, 12.5% group-aware L1 channel pruning, no fine-tuning, and no BatchNorm update.\n\n`tables/` is regenerated from those atomic JSON records after every completed run. Temporary pruned models and validator files are deliberately not retained. Failed runs, if any, remain explicit JSON evidence and can be retried without repeating successful runs.\n\nThe pilot verifies the measurement pipeline and is not the final 42-group ranking. Custom-rule groups remain outside this frozen pilot.\n"""
    atomic_text(output / "README.md", readme)


def build_tables(output: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(BUILDER_PATH), "--output", str(output)],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError("Pilot table builder failed")


def terminate_tree(process: subprocess.Popen[Any]) -> None:
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs(children + [parent], timeout=10)
        for item in alive:
            item.kill()
    except psutil.Error:
        process.kill()


def run_parent(output: Path, retry_failed: bool, timeout_seconds: int, min_free_gb: float) -> int:
    output = output.resolve()
    allowed_root = (STUDY_ROOT / "results" / "pruning").resolve()
    if allowed_root != output and allowed_root not in output.parents:
        raise RuntimeError(f"Output must stay under {allowed_root}")
    check = preflight(require_cuda=True)
    output.mkdir(parents=True, exist_ok=True)
    write_experiment_files(output, check)
    build_tables(output)
    freeze = read_json(FREEZE_PATH)
    queue = [(domain, group_id) for group_id in freeze["pilot_groups"] for domain in ("SNOW", "GEN")]
    for position, (domain, group_id) in enumerate(queue, start=1):
        result_path = output / "runs" / f"{domain}_{group_id}.json"
        if result_path.is_file():
            previous = read_json(result_path)
            if previous.get("status") == "PASS":
                print(f"[{position}/6] {domain} {group_id}: already PASS, skipping", flush=True)
                continue
            if not retry_failed:
                print(f"[{position}/6] {domain} {group_id}: previous FAIL; use --retry-failed", flush=True)
                return 2
        print(f"[{position}/6] {domain} {group_id}: launching isolated worker", flush=True)
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--domain",
            domain,
            "--group",
            group_id,
            "--output",
            str(output),
        ]
        process = subprocess.Popen(command, cwd=str(PROJECT_ROOT))
        start = time.monotonic()
        while process.poll() is None:
            if time.monotonic() - start > timeout_seconds:
                terminate_tree(process)
                raise TimeoutError(f"Worker exceeded {timeout_seconds}s: {domain} {group_id}")
            free_gb = psutil.virtual_memory().available / (1024**3)
            if free_gb < min_free_gb:
                terminate_tree(process)
                raise MemoryError(f"Stopped {domain} {group_id}: available RAM fell below {min_free_gb:.1f} GiB")
            time.sleep(2)
        build_tables(output)
        if process.returncode != 0:
            print(f"{domain} {group_id} failed; retained structured evidence in {result_path}", flush=True)
            return int(process.returncode or 2)
        print(f"[{position}/6] {domain} {group_id}: PASS", flush=True)
    progress = read_json(output / "progress.json")
    if not progress.get("complete"):
        raise RuntimeError(f"Pilot queue ended without six successful runs: {progress}")
    print(json.dumps(progress, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", choices=("GEN", "SNOW"), help=argparse.SUPPRESS)
    parser.add_argument("--group", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        if not args.domain or not args.group:
            parser.error("--worker requires --domain and --group")
        return run_worker(args.domain, args.group, args.output)
    if args.preflight:
        print(json.dumps(preflight(require_cuda=True), indent=2, sort_keys=True))
        return 0
    return run_parent(args.output, args.retry_failed, args.timeout_seconds, args.min_free_gb)


if __name__ == "__main__":
    raise SystemExit(main())
