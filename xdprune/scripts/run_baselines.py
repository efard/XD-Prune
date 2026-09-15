"""Run the frozen YOLO26n GEN/SNOW baselines with traceable outputs.

Examples (run from any directory):
    python scripts/run_baselines.py --dataset both --check-only
    python scripts/run_baselines.py --dataset both --smoke
    python scripts/run_baselines.py --dataset snow --full
    python scripts/run_baselines.py --dataset gen --full

The script never downloads weights, never overwrites an existing run, and never
modifies the sealed v1 configuration files. A completed training run is followed
by the frozen standalone FP32 validation of its exact best.pt checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# This must be set before importing Ultralytics. It prevents incidental model or
# asset downloads during its update and AMP checks.
os.environ["YOLO_OFFLINE"] = "true"

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import get_flops, intersect_dicts, torch_distributed_zero_first

from baseline_research_metrics import (
    checkpoint_profile,
    create_coco_ground_truth,
    run_coco_evaluation,
    validate_training_csv,
    write_artifact_manifest,
    write_confusion_matrix_outputs,
    write_per_class_outputs,
)
from research_detection_validator import (
    PER_IMAGE_STATS_FILENAME,
    PREDICTIONS_FILENAME,
    ResearchDetectionValidator,
)


STUDY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_ROOT
VALIDATOR = STUDY_ROOT / "scripts/validate_baseline_spec.py"
CHECKPOINT = STUDY_ROOT / "models/yolo26n_coco_pretrained.pt"
ENVIRONMENT = STUDY_ROOT / "configs/baselines/environment_v1.json"
EVAL_CONFIG_V1 = STUDY_ROOT / "configs/baselines/eval_common_v1.yaml"
EVAL_CONFIG = STUDY_ROOT / "configs/baselines/eval_research_v2.yaml"
EXECUTION_FREEZE = STUDY_ROOT / "configs/baselines/BASELINE_EXECUTION_FREEZE_V5.json"
RESEARCH_METRICS_SCRIPT = STUDY_ROOT / "scripts/baseline_research_metrics.py"
RESEARCH_VALIDATOR_SCRIPT = STUDY_ROOT / "scripts/research_detection_validator.py"
LATENCY_SCRIPT = STUDY_ROOT / "scripts/benchmark_inference.py"
MANIFEST_TEMPLATE = STUDY_ROOT / "manifests/baseline_run_manifest_template.csv"
RUN_MANIFEST = STUDY_ROOT / "results/baselines/baseline_runs.csv"
SMOKE_PROJECT = STUDY_ROOT / "results/baselines/smoke"


@dataclass(frozen=True)
class Baseline:
    key: str
    role: str
    dataset_id: str
    config: Path
    dataset_yaml: Path
    split_manifest: Path
    expected_classes: int
    smoke_fraction: float


BASELINES = {
    "gen": Baseline(
        key="gen",
        role="GEN",
        dataset_id="mio_full_s42_v1",
        config=STUDY_ROOT / "configs/baselines/b_gen_mio_s42_v1.yaml",
        dataset_yaml=STUDY_ROOT / "data_views/mio_full_s42_v1/mio_tcd_full.yaml",
        split_manifest=STUDY_ROOT / "manifests/mio_full_s42_v1_split_manifest.csv",
        expected_classes=11,
        smoke_fraction=0.01,
    ),
    "snow": Baseline(
        key="snow",
        role="SNOW",
        dataset_id="acdc_snow_official_v1",
        config=STUDY_ROOT / "configs/baselines/b_snow_acdc_s42_v1.yaml",
        dataset_yaml=STUDY_ROOT / "data_views/acdc_snow_official_v1/acdc_snow.yaml",
        split_manifest=STUDY_ROOT / "manifests/acdc_snow_official_v1_split_manifest.csv",
        expected_classes=8,
        smoke_fraction=0.25,
    ),
}


class StableWindowsDetectionTrainer(DetectionTrainer):
    """Detection trainer that avoids unstable CUDA page-locked loader mapping.

    The project observed ``CUDA error: resource already mapped`` in PyTorch's
    pin-memory thread on Windows before the first optimizer update. Disabling
    pinning changes only the host-to-device transfer path, not samples, model
    operations, optimization or metrics.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer_step_calls = 0
        self.successful_optimizer_updates = 0
        self.amp_skipped_updates = 0
        self.nan_or_inf_events_detected = 0
        self.nan_recovery_events_total = 0
        self.selected_best_epoch = 0

    def optimizer_step(self) -> None:
        """Count real optimizer calls and distinguish AMP-overflow skips."""
        scale_before = float(self.scaler.get_scale())
        super().optimizer_step()
        scale_after = float(self.scaler.get_scale())
        self.optimizer_step_calls += 1
        if scale_after < scale_before:
            self.amp_skipped_updates += 1
        else:
            self.successful_optimizer_updates += 1

    def _handle_nan_recovery(self, epoch: int) -> bool:
        """Preserve lifetime numerical-instability counts across clean epochs."""
        loss_nonfinite = self.loss is not None and not bool(self.loss.isfinite())
        fitness_nonfinite = self.fitness is not None and not np.isfinite(self.fitness)
        if loss_nonfinite or fitness_nonfinite:
            self.nan_or_inf_events_detected += 1
        recovered = bool(super()._handle_nan_recovery(epoch))
        if recovered:
            self.nan_recovery_events_total += 1
        return recovered

    def save_model(self) -> None:
        """Record the exact one-based epoch whose checkpoint is written as best.pt."""
        writes_best = self.best_fitness == self.fitness
        super().save_model()
        if writes_best:
            self.selected_best_epoch = int(self.epoch) + 1

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        if mode not in {"train", "val"}:
            raise ValueError(f"Unsupported dataloader mode: {mode}")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(
            dataset.batch_shapes == dataset.batch_shapes[0]
        ):
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            pin_memory=False,
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def workspace_path(value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (WORKSPACE_ROOT / path).resolve()
    if resolved != WORKSPACE_ROOT and WORKSPACE_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes workspace: {value}")
    return resolved


def relative_to_workspace(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def run_frozen_validator() -> None:
    print("\n[1/4] Validating the frozen baseline specification and hashes...")
    environment = os.environ.copy()
    environment["YOLO_OFFLINE"] = "true"
    subprocess.run(
        [sys.executable, str(VALIDATOR)],
        cwd=WORKSPACE_ROOT,
        env=environment,
        check=True,
    )


def validate_execution_freeze() -> None:
    """Verify the versioned instrumentation layer without changing sealed v1."""
    if not EXECUTION_FREEZE.is_file():
        raise FileNotFoundError(f"Missing execution freeze: {EXECUTION_FREEZE}")
    freeze = json.loads(EXECUTION_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("freeze_id") != "BASELINE_EXECUTION_FREEZE_V5" or freeze.get("status") != "FROZEN":
        raise RuntimeError("Unexpected execution freeze identity or status")
    parent = freeze.get("parent_freeze", {})
    parent_path = workspace_path(parent.get("path", ""))
    if sha256(parent_path) != parent.get("sha256"):
        raise RuntimeError("The parent BASELINE_FREEZE_V1 record does not match execution freeze v5")
    files = freeze.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimeError("Execution freeze contains no files")
    for relative, expected in files.items():
        path = workspace_path(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen execution file: {path}")
        if path.stat().st_size != expected.get("bytes") or sha256(path) != expected.get("sha256"):
            raise RuntimeError(f"Execution file drift: {relative}")

    v1 = load_yaml(EVAL_CONFIG_V1)
    v2 = load_yaml(EVAL_CONFIG)
    differences = {key for key in set(v1) | set(v2) if v1.get(key) != v2.get(key)}
    if differences != {"save_json"} or v1.get("save_json") is not False or v2.get("save_json") is not True:
        raise RuntimeError(f"Research evaluation v2 changes unexpected settings: {sorted(differences)}")
    print(
        f"Execution instrumentation: {freeze['freeze_id']} PASS; "
        f"{len(files)} files; evaluation-only difference=save_json"
    )


def require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is unavailable; the frozen configs require device=0")
    gpu = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free_gib = shutil.disk_usage(STUDY_ROOT).free / (1024**3)
    print(f"CUDA: {gpu} ({memory_gib:.2f} GiB)")
    print(f"Free workspace disk: {free_gib:.1f} GiB")
    if free_gib < 10:
        print("WARNING: less than 10 GiB is free; preserve enough space for checkpoints and plots.")


def git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=WORKSPACE_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""


def manifest_fields() -> list[str]:
    with MANIFEST_TEMPLATE.open("r", newline="", encoding="utf-8") as handle:
        fields = next(csv.reader(handle))
    if not fields:
        raise RuntimeError("The baseline manifest template has no columns")
    return fields


def append_manifest(row: dict[str, Any]) -> None:
    fields = manifest_fields()
    RUN_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    if RUN_MANIFEST.exists():
        with RUN_MANIFEST.open("r", newline="", encoding="utf-8") as handle:
            existing = next(csv.reader(handle), [])
        if existing != fields:
            raise RuntimeError(f"Manifest header drift: {RUN_MANIFEST}")
        write_header = RUN_MANIFEST.stat().st_size == 0
    else:
        write_header = True
    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def base_manifest_row(
    baseline: Baseline,
    config: dict[str, Any],
    run_id: str,
    run_stage: str,
    output_dir: Path,
    command: str,
    transferred_items: str,
) -> dict[str, Any]:
    launcher = Path(__file__).resolve()
    return {
        "run_id": run_id,
        "run_stage": run_stage,
        "date_utc": utc_now(),
        "dataset_role": baseline.role,
        "dataset_id": baseline.dataset_id,
        "dataset_yaml": relative_to_workspace(baseline.dataset_yaml),
        "dataset_yaml_sha256": sha256(baseline.dataset_yaml),
        "split_manifest": relative_to_workspace(baseline.split_manifest),
        "split_manifest_sha256": sha256(baseline.split_manifest),
        "config_path": relative_to_workspace(baseline.config),
        "config_sha256": sha256(baseline.config),
        "environment_path": relative_to_workspace(ENVIRONMENT),
        "environment_sha256": sha256(ENVIRONMENT),
        "git_commit": git_commit(),
        "seed": config["seed"],
        "initial_checkpoint": relative_to_workspace(CHECKPOINT),
        "initial_checkpoint_sha256": sha256(CHECKPOINT),
        "task": config["task"],
        "image_size": config["imgsz"],
        "batch_size": config["batch"],
        "nominal_batch_size": config["nbs"],
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "lrf": config["lrf"],
        "momentum_beta1": config["momentum"],
        "weight_decay": config["weight_decay"],
        "warmup_epochs": config["warmup_epochs"],
        "max_epochs": config["epochs"],
        "patience": config["patience"],
        "selection_metric": "training validation mAP50-95",
        "inference_dtype": "FP32 standalone validation",
        "amp_train": config["amp"],
        "deterministic": config["deterministic"],
        "transferred_weight_items": transferred_items,
        "command": command,
        "output_dir": relative_to_workspace(output_dir),
        "notes": (
            f"launcher={relative_to_workspace(launcher)}; launcher_sha256={sha256(launcher)}; "
            f"execution_freeze={relative_to_workspace(EXECUTION_FREEZE)}; "
            f"execution_freeze_sha256={sha256(EXECUTION_FREEZE)}; "
            f"standalone_eval={relative_to_workspace(EVAL_CONFIG)}; "
            f"standalone_eval_sha256={sha256(EVAL_CONFIG)}; "
            "dataloader_pin_memory=false"
        ),
    }


def expected_output(config: dict[str, Any], action: str, baseline: Baseline, stamp: str) -> tuple[Path, str]:
    if action == "full":
        name = str(config["name"])
        project = workspace_path(str(config["project"]))
    else:
        name = f"smoke_{baseline.key}_yolo26n_s42_v1_{stamp}"
        project = SMOKE_PROJECT.resolve()
    output = (project / name).resolve()
    results_root = (STUDY_ROOT / "results/baselines").resolve()
    if output != results_root and results_root not in output.parents:
        raise ValueError(f"Unsafe output directory: {output}")
    return output, name


def adaptation_audit(source: YOLO, expected_classes: int) -> tuple[str, int]:
    # Deep-copy is essential because model construction mutates the YAML nc value.
    target = DetectionModel(copy.deepcopy(source.model.yaml), nc=expected_classes, ch=3, verbose=False)
    transferred = len(intersect_dicts(source.model.float().state_dict(), target.state_dict()))
    total_items = len(target.state_dict())
    parameters = sum(parameter.numel() for parameter in target.parameters())
    del target
    return f"{transferred}/{total_items}", parameters


def fused_inference_profile(checkpoint: Path, expected_classes: int, imgsz: int) -> dict[str, Any]:
    """Explicitly profile a separate fused inference graph without changing the checkpoint."""
    yolo = YOLO(str(checkpoint), task="detect")
    model = yolo.model.float().cpu().eval()
    if len(yolo.names) != expected_classes:
        raise RuntimeError(f"Fused profile nc={len(yolo.names)}, expected {expected_classes}")
    model.fuse(verbose=False)
    if not model.is_fused():
        raise RuntimeError("Ultralytics did not produce the expected fused inference graph")
    gflops = metric_value(get_flops(model, imgsz=imgsz))
    result = {
        "profiled_graph": "explicitly fused Conv+BN inference graph",
        "model_is_fused": True,
        "fused_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "ultralytics_get_flops_gflops_fused": gflops,
        "derived_gmacs_fused": gflops / 2.0,
        "note": "Canonical model-size reporting uses the unfused checkpoint profile; this is runtime graph complexity.",
    }
    del model, yolo
    return result


def metric_value(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def metric_operating_point(metrics: Any) -> dict[str, Any]:
    """Describe the global max-mean-F1 point used for reported P/R/F1."""
    f1_curve = np.asarray(metrics.box.f1_curve, dtype=np.float64)
    confidence_axis = np.asarray(metrics.box.px, dtype=np.float64)
    if f1_curve.size == 0 or confidence_axis.size == 0:
        return {
            "definition": "Ultralytics global max-mean-F1 confidence operating point",
            "confidence": None,
            "mean_f1": 0.0,
        }
    mean_curve = f1_curve.mean(axis=0)
    index = int(mean_curve.argmax())
    return {
        "definition": "Ultralytics global max-mean-F1 confidence operating point",
        "confidence": float(confidence_axis[index]),
        "mean_f1": float(mean_curve[index]),
        "note": "Reported precision, recall and F1 are not values at conf=0.001; conf=0.001 is the prediction collection threshold.",
    }


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(0)


def peak_cuda_memory() -> dict[str, float]:
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(0)),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(0) / (1024**2),
        "peak_reserved_mib": torch.cuda.max_memory_reserved(0) / (1024**2),
    }


def standalone_fp32_validation(
    best: Path,
    baseline: Baseline,
    training_dir: Path,
) -> tuple[Any, YOLO, Path, dict[str, Any]]:
    print("\n[3/4] Running frozen standalone FP32 validation of best.pt...")
    evaluation = load_yaml(EVAL_CONFIG)
    evaluation.pop("task", None)
    evaluation.pop("mode", None)
    evaluation.update(
        {
            "data": relative_to_workspace(baseline.dataset_yaml),
            "project": str(training_dir),
            "name": "fp32_val",
            "exist_ok": False,
        }
    )
    validation_model = YOLO(str(best), task="detect")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(0)
    synchronize_cuda()
    started = time.perf_counter()
    metrics = validation_model.val(validator=ResearchDetectionValidator, **evaluation)
    synchronize_cuda()
    wall_seconds = time.perf_counter() - started
    validation_dir = Path(metrics.save_dir).resolve()
    predictions = validation_dir / PREDICTIONS_FILENAME
    per_image_stats = validation_dir / PER_IMAGE_STATS_FILENAME
    for required in (predictions, per_image_stats):
        if not required.is_file():
            raise FileNotFoundError(f"Research validation did not produce: {required}")
    redundant_predictions = validation_dir / "predictions.json"
    if redundant_predictions.exists():
        raise RuntimeError(
            f"Redundant Ultralytics prediction export was not suppressed: {redundant_predictions}"
        )
    details = {
        "wall_seconds": wall_seconds,
        "cuda_memory": peak_cuda_memory(),
        "ultralytics_diagnostic_ms_per_image": {
            key: metric_value(value) for key, value in metrics.speed.items()
        },
        "predictions_full_precision": relative_to_workspace(predictions),
        "predictions_full_precision_sha256": sha256(predictions),
        "per_image_metric_stats": relative_to_workspace(per_image_stats),
        "per_image_metric_stats_sha256": sha256(per_image_stats),
        "evaluation_config": relative_to_workspace(EVAL_CONFIG),
        "evaluation_config_sha256": sha256(EVAL_CONFIG),
        "redundant_predictions_json_suppressed": True,
    }
    return metrics, validation_model, validation_dir, details


def run_latency_benchmarks(best: Path, baseline: Baseline, training_dir: Path) -> dict[str, Any]:
    """Run each precision in an isolated process so GPU state is not shared."""
    print("\n[4/4] Running reproducible batch-1 latency benchmarks...")
    output_dir = training_dir / "benchmark"
    reports: dict[str, Any] = {}
    environment = os.environ.copy()
    environment["YOLO_OFFLINE"] = "true"
    for precision in ("fp32", "fp16"):
        command = [
            sys.executable,
            str(LATENCY_SCRIPT),
            "--weights",
            str(best),
            "--data",
            str(baseline.dataset_yaml),
            "--domain",
            baseline.role,
            "--output",
            str(output_dir),
            "--precision",
            precision,
        ]
        try:
            subprocess.run(command, cwd=WORKSPACE_ROOT, env=environment, check=True)
            report = output_dir / f"inference_benchmark_{precision}.json"
            samples = output_dir / f"inference_latency_samples_{precision}.csv"
            if not report.is_file() or not samples.is_file():
                raise FileNotFoundError(f"Benchmark artifacts missing for {precision}")
            reports[precision] = {
                "status": "completed",
                "report": relative_to_workspace(report),
                "report_sha256": sha256(report),
                "raw_samples": relative_to_workspace(samples),
                "raw_samples_sha256": sha256(samples),
                "accuracy_scope": (
                    "Primary strict-FP32 deployment timing"
                    if precision == "fp32"
                    else "Latency-only deployment measurement; FP16 accuracy was not evaluated"
                ),
            }
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            reports[precision] = {
                "status": "failed",
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "recovery_command": subprocess.list2cmdline(command),
            }
            print(
                f"WARNING: {precision.upper()} latency benchmark failed; "
                "the trained checkpoint and FP32 accuracy evidence remain valid.",
                file=sys.stderr,
            )
    return reports


def run_one(baseline: Baseline, action: str, stamp: str, command: str) -> None:
    pipeline_started = time.perf_counter()
    config = load_yaml(baseline.config)
    output_dir, run_name = expected_output(config, action, baseline, stamp)
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing run: {output_dir}\n"
            "Keep it as evidence. Ask for a checked resume procedure if the run was interrupted."
        )

    print(f"\n[2/4] Preparing {baseline.role} {action} run: {run_name}")
    source = YOLO(str(CHECKPOINT), task="detect")
    transferred_items, target_parameters = adaptation_audit(source, baseline.expected_classes)
    print(
        f"Head adaptation audit: nc={baseline.expected_classes}, "
        f"transferred={transferred_items}, target parameters={target_parameters:,}"
    )

    effective_config = dict(config)
    if action == "smoke":
        effective_config.update(
            {
                "epochs": 1,
                "patience": 0,
                "fraction": baseline.smoke_fraction,
                "project": str(SMOKE_PROJECT),
                "name": run_name,
                "exist_ok": False,
                "save_period": -1,
            }
        )

    run_id = run_name
    row = base_manifest_row(
        baseline,
        effective_config,
        run_id,
        f"baseline_{action}",
        output_dir,
        command,
        transferred_items,
    )
    row["status"] = "started"
    run_note = (
        f"Non-publication mechanics check; one epoch; training fraction={baseline.smoke_fraction}; "
        "complete validation split."
        if action == "smoke"
        else "Frozen publication baseline run."
    )
    row["notes"] = f"{row['notes']}; {run_note}"
    append_manifest(row)
    # Ultralytics resolves a relative ``project`` below its configured
    # ``runs/detect`` root, even though this launcher normalizes cwd. Always
    # pass the already-validated absolute destination so training and all
    # post-training evidence land in the path recorded by the manifest.
    train_kwargs: dict[str, Any] = {
        "cfg": str(baseline.config),
        "project": str(output_dir.parent),
        "name": run_name,
        "exist_ok": False,
    }
    if action == "smoke":
        train_kwargs.update({key: effective_config[key] for key in (
            "epochs", "patience", "fraction", "project", "name", "exist_ok", "save_period"
        )})

    observed_optimizer_updates = 0
    try:
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        synchronize_cuda()
        training_started = time.perf_counter()
        source.train(trainer=StableWindowsDetectionTrainer, **train_kwargs)
        synchronize_cuda()
        training_wall_seconds = time.perf_counter() - training_started
        training_cuda_memory = peak_cuda_memory()

        trainer = source.trainer
        training_dir = Path(trainer.save_dir).resolve()
        best = Path(trainer.best).resolve()
        last = Path(trainer.last).resolve()
        if not best.is_file():
            raise FileNotFoundError(f"Training completed without best.pt: {best}")
        if not last.is_file():
            raise FileNotFoundError(f"Training completed without last.pt: {last}")
        if training_dir != output_dir:
            raise RuntimeError(f"Unexpected output directory: {training_dir}, expected {output_dir}")

        completed_epochs = int(trainer.epoch) + 1
        selected_best_epoch = int(trainer.selected_best_epoch)
        if selected_best_epoch < 1:
            raise RuntimeError("Training never recorded an epoch written to best.pt")
        optimizer_step_calls = int(trainer.optimizer_step_calls)
        successful_optimizer_updates = int(trainer.successful_optimizer_updates)
        observed_optimizer_updates = successful_optimizer_updates
        amp_skipped_updates = int(trainer.amp_skipped_updates)
        if optimizer_step_calls != successful_optimizer_updates + amp_skipped_updates:
            raise RuntimeError("Optimizer update accounting is internally inconsistent")
        if optimizer_step_calls < 1 or successful_optimizer_updates < 1:
            raise RuntimeError("Training completed without a successful optimizer update")
        ema_updates = int(trainer.ema.updates) if trainer.ema is not None else 0
        if ema_updates != optimizer_step_calls:
            raise RuntimeError(
                f"EMA update count {ema_updates} does not match optimizer-step calls {optimizer_step_calls}"
            )
        nan_events = int(trainer.nan_or_inf_events_detected)
        nan_recoveries = int(trainer.nan_recovery_events_total)
        if nan_events or nan_recoveries:
            raise RuntimeError(
                f"Numerical instability detected: events={nan_events}, recoveries={nan_recoveries}"
            )

        train_dataset_images = len(trainer.train_loader.dataset)
        train_images = train_dataset_images * completed_epochs
        training_history = validate_training_csv(training_dir / "results.csv", completed_epochs)
        canonical_best_profile = checkpoint_profile(best, baseline.expected_classes, int(config["imgsz"]))
        canonical_last_profile = checkpoint_profile(last, baseline.expected_classes, int(config["imgsz"]))
        fused_profile = fused_inference_profile(best, baseline.expected_classes, int(config["imgsz"]))
        trainer_best_fitness = metric_value(trainer.stopper.best_fitness)
        stopper_best_epoch = int(trainer.stopper.best_epoch)

        # Release the training graph before standalone accuracy and timing work.
        del trainer, source
        gc.collect()
        torch.cuda.empty_cache()

        metrics, validation_model, validation_dir, validation_runtime = standalone_fp32_validation(
            best,
            baseline,
            training_dir,
        )
        names = validation_model.names
        model_classes = len(names)
        if model_classes != baseline.expected_classes:
            raise RuntimeError(f"Trained nc={model_classes}, expected {baseline.expected_classes}")

        per_class_json, per_class_csv = write_per_class_outputs(metrics, names, validation_dir)
        confusion_paths = write_confusion_matrix_outputs(metrics, names, validation_dir)
        ground_truth, support_path = create_coco_ground_truth(
            baseline.dataset_yaml,
            "val",
            validation_dir,
        )
        predictions = validation_dir / PREDICTIONS_FILENAME
        coco_metrics_path = run_coco_evaluation(ground_truth, predictions, validation_dir)
        support = json.loads(support_path.read_text(encoding="utf-8"))
        operating_point = metric_operating_point(metrics)
        primary_metrics = {
            "map50_95": metric_value(metrics.box.map),
            "map50": metric_value(metrics.box.map50),
            "map75": metric_value(metrics.box.map75),
            "mean_precision_at_max_mean_f1": metric_value(metrics.box.mp),
            "mean_recall_at_max_mean_f1": metric_value(metrics.box.mr),
            "mean_f1_at_max_mean_f1": operating_point["mean_f1"],
            "operating_point": operating_point,
            "validation_images": int(support["images"]),
            "validation_instances": int(support["instances"]),
        }

        del validation_model
        gc.collect()
        torch.cuda.empty_cache()
        latency = run_latency_benchmarks(best, baseline, training_dir)

        completed = dict(row)
        completed.update(
            {
                "date_utc": utc_now(),
                "status": "completed",
                "selected_checkpoint": relative_to_workspace(best),
                "selected_checkpoint_sha256": sha256(best),
                "completed_epochs": completed_epochs,
                "best_epoch": selected_best_epoch,
                "optimizer_updates": successful_optimizer_updates,
                "training_images_seen": train_images,
                "nc": model_classes,
                "parameter_count": canonical_best_profile["unfused_parameters"],
                "macs_gflops": canonical_best_profile["ultralytics_get_flops_gflops_unfused"],
                "checkpoint_size_bytes": best.stat().st_size,
                "selection_metric_value": trainer_best_fitness,
                "map50_95": primary_metrics["map50_95"],
                "map50": primary_metrics["map50"],
                "precision": primary_metrics["mean_precision_at_max_mean_f1"],
                "recall": primary_metrics["mean_recall_at_max_mean_f1"],
                "per_class_ap_path": relative_to_workspace(per_class_json),
                "output_dir": relative_to_workspace(training_dir),
                "notes": (
                    row["notes"]
                    + " parameter_count and macs_gflops are canonical unfused checkpoint parameters and "
                    "Ultralytics GFLOPs; GMACs=GFLOPs/2. Test splits were not accessed."
                ),
            }
        )
        summary_path = training_dir / "baseline_run_summary.json"
        research = {
            "schema_version": "baseline-research-evidence-v2",
            "evidence_scope": (
                "mechanics-only smoke test; not publication evidence"
                if action == "smoke"
                else "primary seed-42 training and frozen validation baseline"
            ),
            "dataset_role": baseline.role,
            "dataset_id": baseline.dataset_id,
            "test_policy": {
                "test_split_accessed": False,
                "reason": (
                    "MIO internal test remains locked until pruning-policy freeze; "
                    "ACDC official test labels are withheld."
                ),
            },
            "training": {
                "wall_seconds": training_wall_seconds,
                "ultralytics_results_csv_cumulative_seconds": training_history[
                    "training_loop_cumulative_seconds"
                ],
                "cuda_memory": training_cuda_memory,
                "completed_epochs": completed_epochs,
                "train_dataset_images_per_epoch": train_dataset_images,
                "logical_training_image_exposures": train_images,
                "selected_best_checkpoint_epoch": selected_best_epoch,
                "early_stopper_best_epoch": stopper_best_epoch,
                "selection_metric": "training-validation mAP50-95",
                "selection_metric_value": trainer_best_fitness,
                "optimizer_step_calls": optimizer_step_calls,
                "successful_optimizer_updates": successful_optimizer_updates,
                "amp_overflow_skipped_updates": amp_skipped_updates,
                "ema_updates": ema_updates,
                "nan_or_inf_events_detected": nan_events,
                "nan_recovery_events": nan_recoveries,
                "history_integrity": training_history,
            },
            "checkpoints": {
                "best": canonical_best_profile,
                "last": canonical_last_profile,
                "fused_best_inference_graph": fused_profile,
            },
            "standalone_fp32_validation": {
                **primary_metrics,
                **validation_runtime,
                "per_class_metrics_json": relative_to_workspace(per_class_json),
                "per_class_metrics_csv": relative_to_workspace(per_class_csv),
                "confusion_matrix_outputs": {
                    key: relative_to_workspace(path) for key, path in confusion_paths.items()
                },
                "validation_support": relative_to_workspace(support_path),
                "coco_ground_truth": relative_to_workspace(ground_truth),
                "secondary_coco_style_metrics": relative_to_workspace(coco_metrics_path),
                "primary_metric_note": (
                    "Ultralytics FP32 mAP50-95 is the study primary metric; pycocotools outputs are secondary."
                ),
                "confusion_matrix_note": (
                    "Counts use the installed Ultralytics confusion-matrix defaults (confidence 0.25, match IoU 0.45); "
                    "evaluation config IoU 0.7 is the NMS threshold."
                ),
            },
            "latency": latency,
            "complete_pipeline_wall_seconds_before_manifest": time.perf_counter() - pipeline_started,
            "instrumentation": {
                "execution_freeze": relative_to_workspace(EXECUTION_FREEZE),
                "execution_freeze_sha256": sha256(EXECUTION_FREEZE),
                "launcher_sha256": sha256(Path(__file__).resolve()),
                "research_metrics_script_sha256": sha256(RESEARCH_METRICS_SCRIPT),
                "research_validator_script_sha256": sha256(RESEARCH_VALIDATOR_SCRIPT),
                "latency_script_sha256": sha256(LATENCY_SCRIPT),
            },
        }
        summary_payload = {**completed, "research": research}
        summary_path.write_text(
            json.dumps(summary_payload, indent=2, default=str, allow_nan=False) + "\n",
            encoding="utf-8",
        )

        required_artifacts = [path for path in training_dir.rglob("*") if path.is_file()]
        artifact_manifest = write_artifact_manifest(
            training_dir,
            required_artifacts,
            training_dir / "artifact_manifest.json",
        )
        append_manifest(completed)

        print("\nRun completed and recorded")
        print(f"best.pt: {best}")
        print(f"SHA-256: {completed['selected_checkpoint_sha256']}")
        print(f"FP32 validation mAP50-95: {completed['map50_95']:.6f}")
        print(f"Run summary: {summary_path}")
        print(f"Artifact hashes: {artifact_manifest}")
        print(f"Append-only manifest: {RUN_MANIFEST}")
    except BaseException as exc:
        active_trainer = getattr(source, "trainer", None) if "source" in locals() else None
        optimizer_updates = int(
            getattr(active_trainer, "successful_optimizer_updates", observed_optimizer_updates)
        )
        failed = dict(row)
        failed.update(
            {
                "date_utc": utc_now(),
                "status": "failed" if not isinstance(exc, KeyboardInterrupt) else "interrupted",
                "optimizer_updates": optimizer_updates,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }
        )
        append_manifest(failed)
        print("\nRun did not complete. Existing output was preserved.", file=sys.stderr)
        traceback.print_exc()
        raise


def check_one(baseline: Baseline) -> None:
    config = load_yaml(baseline.config)
    output_dir, _ = expected_output(config, "full", baseline, utc_stamp())
    source = YOLO(str(CHECKPOINT), task="detect")
    transferred_items, target_parameters = adaptation_audit(source, baseline.expected_classes)
    print(
        f"{baseline.role}: config accepted; nc={baseline.expected_classes}; "
        f"transfer={transferred_items}; target parameters={target_parameters:,}; "
        f"full output={output_dir}"
    )


def loader_check_one(baseline: Baseline) -> None:
    """Exercise real image loading and CUDA transfer without model execution."""
    raw = load_yaml(baseline.config)
    raw.update(epochs=1, fraction=baseline.smoke_fraction)
    args = get_cfg(DEFAULT_CFG_DICT, overrides=raw)
    data = check_det_dataset(str(baseline.dataset_yaml), autodownload=False)
    dataset = build_yolo_dataset(
        args,
        data["train"],
        batch=int(raw["batch"]),
        data=data,
        mode="train",
        rect=False,
        stride=32,
    )
    loader = build_dataloader(
        dataset,
        batch=int(raw["batch"]),
        workers=int(raw["workers"]),
        shuffle=True,
        rank=-1,
        pin_memory=False,
    )
    batches_to_check = min(len(loader), 20)
    for index, batch in enumerate(loader):
        images = batch["img"].to("cuda:0", non_blocking=False)
        torch.cuda.synchronize()
        del images, batch
        if index + 1 >= batches_to_check:
            break
    print(
        f"{baseline.role}: {batches_to_check} real batches transferred; "
        f"workers={loader.num_workers}; pin_memory={loader.pin_memory}; PASS"
    )
    del loader, dataset
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("gen", "snow", "both"),
        default="both",
        help="Baseline domain to check or run (default: both).",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true", help="Validate everything without training.")
    action.add_argument(
        "--loader-check",
        action="store_true",
        help="Test real dataloading and CUDA transfer without running the model.",
    )
    action.add_argument("--smoke", action="store_true", help="Run labelled one-epoch mechanics checks.")
    action.add_argument("--full", action="store_true", help="Run the exact frozen full baseline configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.chdir(WORKSPACE_ROOT)
    print(f"Workspace: {WORKSPACE_ROOT}")
    print(f"Ultralytics: {ultralytics_version}; PyTorch: {torch.__version__}; offline mode: true")
    run_frozen_validator()
    validate_execution_freeze()
    require_cuda()

    selected = [BASELINES[args.dataset]] if args.dataset != "both" else [BASELINES["gen"], BASELINES["snow"]]
    if args.check_only:
        print("\n[2/2] Auditing dataset-specific head adaptation without training...")
        for baseline in selected:
            check_one(baseline)
        print("\nCHECK PASSED. No training or result directory was created.")
        return 0

    if args.loader_check:
        print("\n[2/2] Testing real pageable-memory dataloaders without model execution...")
        for baseline in selected:
            loader_check_one(baseline)
        print("\nLOADER CHECK PASSED. No model was run and no result directory was created.")
        return 0

    action = "smoke" if args.smoke else "full"
    stamp = utc_stamp()
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    for baseline in selected:
        run_one(baseline, action, stamp, command)
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
