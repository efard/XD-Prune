"""Reusable exact-architecture training engine for the active T6 experiment."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any


SELF = Path(__file__).resolve()
OUTPUT_ROOT = SELF.parent
STUDY_ROOT = SELF.parents[4]
PROJECT_ROOT = STUDY_ROOT.parent
CORE_SCRIPTS = STUDY_ROOT / "scripts"
if str(CORE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORE_SCRIPTS))

import torch
from torch import nn
from ultralytics import YOLO

import run_baselines as rb
import run_t1_t2 as base
import run_t1_t2_custom_sweep as custom


SCHEMA = "t6_exact_architecture_engine_v1"
EPOCHS = 20
SEED = 42
IMG_SIZE = 640
BATCH_SIZE = 16
WORKERS = 4
WARMUP_EPOCHS = 1.0
CLOSE_MOSAIC_EPOCHS = 0
TRAINING_ROOT = OUTPUT_ROOT / "training"
SMOKE_ROOT = OUTPUT_ROOT / "smoke"
RECORDS_ROOT = OUTPUT_ROOT / "records"
MODELS_ROOT = OUTPUT_ROOT / "models"
TABLES_ROOT = OUTPUT_ROOT / "tables"
T5_ROOT = OUTPUT_ROOT.parent / "T5_replay10"
T5_MANIFEST = T5_ROOT / "experiment_manifest.json"
DOMAIN_CONFIG: dict[str, dict[str, Path]] = {}


class ExactPrunedArchitectureTrainer(rb.StableWindowsDetectionTrainer):
    """Train a provided physically pruned module without rebuilding its YAML."""

    expected_parameters: int | None = None

    def get_model(
        self,
        cfg: str | None = None,
        weights: str | nn.Module | None = None,
        verbose: bool = True,
    ) -> nn.Module:
        if not isinstance(weights, nn.Module):
            raise TypeError(
                "T6 requires the exact in-memory pruned module as trainer weights"
            )
        model = weights.float()
        actual = count_parameters(model)
        if self.expected_parameters is None or actual != self.expected_parameters:
            raise RuntimeError(
                f"Pruned architecture mismatch: expected "
                f"{self.expected_parameters}, observed {actual}"
            )
        return model

    def final_eval(self) -> None:
        """Skip redundant path-based final validation.

        Epoch validation remains enabled. Publication metrics are recomputed
        afterward from locally loaded plain model objects.
        """

        self.metrics = getattr(self.validator, "metrics", None)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def relative(path: Path) -> str:
    return base.relative(path)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def preflight() -> dict[str, Any]:
    raise NotImplementedError("The experiment wrapper must provide preflight()")


def checkpoint_model(path: Path) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(f"Missing local checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(checkpoint, nn.Module):
        model = checkpoint
    elif isinstance(checkpoint, dict):
        model = checkpoint.get("ema")
        if not isinstance(model, nn.Module):
            model = checkpoint.get("model")
    else:
        model = None
    if not isinstance(model, nn.Module):
        raise TypeError(f"Checkpoint contains no PyTorch model: {path}")
    return model.float().cpu()


def export_plain_model(
    checkpoint: Path,
    destination: Path,
    expected_parameters: int,
) -> dict[str, Any]:
    model = checkpoint_model(checkpoint)
    parameters = count_parameters(model)
    if parameters != expected_parameters:
        raise RuntimeError(
            f"Checkpoint architecture changed: expected {expected_parameters}, "
            f"observed {parameters} in {checkpoint}"
        )
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, IMG_SIZE, IMG_SIZE))
        summary = base.public_prediction_summary(output)
    if not summary["all_finite"]:
        raise RuntimeError(f"Non-finite checkpoint output: {checkpoint}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    torch.save(model, temporary)
    temporary.replace(destination)
    reloaded = torch.load(
        destination,
        map_location="cpu",
        weights_only=False,
    ).float().eval()
    if count_parameters(reloaded) != expected_parameters:
        raise RuntimeError("Exported plain model parameter audit failed")
    return {
        "checkpoint": relative(checkpoint),
        "checkpoint_sha256": base.sha256(checkpoint),
        "plain_model": relative(destination),
        "plain_model_sha256": base.sha256(destination),
        "parameters": parameters,
        "public_output": summary,
    }


def completed_record(domain: str, mode: str) -> dict[str, Any] | None:
    path = RECORDS_ROOT / f"{domain}_{mode}_training.json"
    if not path.is_file():
        return None
    record = base.read_json(path)
    return record if record.get("status") == "PASS" else None


def train_domain(domain: str, smoke: bool) -> dict[str, Any]:
    mode = "smoke" if smoke else "full20"
    prior = completed_record(domain, mode)
    if prior is not None:
        print(f"{domain} {mode}: already complete; skipping", flush=True)
        return prior

    config = DOMAIN_CONFIG[domain]
    project = SMOKE_ROOT if smoke else TRAINING_ROOT
    run_name = f"{domain.lower()}_t6_{mode}_s{SEED}_v1"
    run_dir = project / run_name
    if run_dir.exists():
        if smoke:
            expected_parameters = count_parameters(
                torch.load(
                    config["t5_model"],
                    map_location="cpu",
                    weights_only=False,
                )
            )
            best = run_dir / "weights" / "best.pt"
            last = run_dir / "weights" / "last.pt"
            results_csv = run_dir / "results.csv"
            if best.is_file() and last.is_file() and results_csv.is_file():
                exports = {
                    label: export_plain_model(
                        checkpoint,
                        MODELS_ROOT
                        / "smoke"
                        / f"{domain}_T6_{mode}_{label}.pth",
                        expected_parameters,
                    )
                    for label, checkpoint in (("best", best), ("last", last))
                }
                record_path = RECORDS_ROOT / f"{domain}_{mode}_training.json"
                recovered = {
                    "schema": SCHEMA,
                    "status": "PASS",
                    "domain": domain,
                    "mode": mode,
                    "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "completed_epochs": 1,
                    "training_directory": relative(run_dir),
                    "parameters_before": expected_parameters,
                    "parameters_after": expected_parameters,
                    "parameter_count_preserved": True,
                    "exports": exports,
                    "recovered_existing_smoke": True,
                }
                atomic_json(record_path, recovered)
                return recovered
        raise RuntimeError(
            f"Incomplete output already exists: {run_dir}. Preserve it and "
            "use the checkpoint-resume runner."
        )

    input_model = torch.load(
        config["t5_model"],
        map_location="cpu",
        weights_only=False,
    ).float()
    expected_parameters = count_parameters(input_model)
    for parameter in input_model.parameters():
        parameter.requires_grad_(True)

    ExactPrunedArchitectureTrainer.expected_parameters = expected_parameters
    epochs = 1 if smoke else EPOCHS
    fraction = 0.01 if domain == "GEN" and smoke else (
        0.25 if smoke else 1.0
    )
    record_path = RECORDS_ROOT / f"{domain}_{mode}_training.json"
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "domain": domain,
        "mode": mode,
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_model": relative(config["t5_model"]),
        "input_model_sha256": base.sha256(config["t5_model"]),
        "parameters_before": expected_parameters,
        "dataset": relative(config["dataset"]),
        "epochs_requested": epochs,
        "fraction": fraction,
        "discarded_mechanics_check": smoke,
    }
    atomic_json(record_path, record)

    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    torch.cuda.empty_cache()
    started = time.perf_counter()
    trainer = None
    try:
        overrides = {
            "task": "detect",
            "mode": "train",
            "model": str(config["baseline_checkpoint"]),
            "data": str(config["dataset"]),
            "project": str(project),
            "name": run_name,
            "exist_ok": False,
            "epochs": epochs,
            "patience": 0,
            "batch": BATCH_SIZE,
            "imgsz": IMG_SIZE,
            "device": 0,
            "workers": WORKERS,
            "cache": False,
            "seed": SEED,
            "deterministic": True,
            "fraction": fraction,
            "pretrained": False,
            "amp": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 0.0 if smoke else WARMUP_EPOCHS,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.0,
            "cos_lr": False,
            "close_mosaic": 0 if smoke else CLOSE_MOSAIC_EPOCHS,
            "save": True,
            "save_period": -1 if smoke else 5,
            "val": not smoke,
            "plots": not smoke,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "verbose": True,
        }
        trainer = ExactPrunedArchitectureTrainer(overrides=overrides)
        trainer.model = input_model
        trainer.train()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError("Training did not produce best.pt and last.pt")
        completed_epochs = int(trainer.epoch) + 1
        if completed_epochs != epochs:
            raise RuntimeError(
                f"Expected {epochs} completed epochs, observed {completed_epochs}"
            )
        exports = {}
        for label, checkpoint in (("best", best), ("last", last)):
            exports[label] = export_plain_model(
                checkpoint,
                MODELS_ROOT
                / ("smoke" if smoke else "full20")
                / f"{domain}_T6_{mode}_{label}.pth",
                expected_parameters,
            )
        record.update(
            {
                "status": "PASS",
                "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": time.perf_counter() - started,
                "completed_epochs": completed_epochs,
                "training_directory": relative(run_dir),
                "parameters_after": expected_parameters,
                "parameter_count_preserved": True,
                "best_epoch": int(trainer.selected_best_epoch),
                "optimizer_step_calls": int(trainer.optimizer_step_calls),
                "successful_optimizer_updates": int(
                    trainer.successful_optimizer_updates
                ),
                "amp_skipped_updates": int(trainer.amp_skipped_updates),
                "exports": exports,
            }
        )
    except BaseException as exc:
        record.update(
            {
                "status": "FAIL",
                "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": time.perf_counter() - started,
                "error": {
                    "type": type(exc).__name__,
                    "message": str(exc),
                },
            }
        )
        atomic_json(record_path, record)
        raise
    atomic_json(record_path, record)
    del trainer, input_model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def evaluate_plain(
    domain: str,
    checkpoint_label: str,
    model_path: Path,
) -> dict[str, Any]:
    raise NotImplementedError(
        "The experiment wrapper must provide evaluate_plain()"
    )


def evaluate_full_runs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain in ("GEN", "SNOW"):
        training = completed_record(domain, "full20")
        if training is None:
            raise RuntimeError(f"{domain} full20 training is not complete")
        for checkpoint_label in ("best", "last"):
            model_path = PROJECT_ROOT / training["exports"][checkpoint_label][
                "plain_model"
            ]
            record_path = (
                RECORDS_ROOT
                / f"{domain}_full20_{checkpoint_label}_evaluation.json"
            )
            if record_path.is_file():
                evaluation = base.read_json(record_path)
            else:
                evaluation = evaluate_plain(
                    domain,
                    checkpoint_label,
                    model_path,
                )
                atomic_json(record_path, evaluation)
            metrics = evaluation["metrics"]
            comparison = evaluation["comparisons"]["map50_95"]
            rows.append(
                {
                    "experiment": "T6_end_only_finetune",
                    "domain": domain,
                    "checkpoint": checkpoint_label,
                    "fine_tuning_epochs": EPOCHS,
                    "parameters": evaluation["parameters"],
                    "map50": metrics["ap50"],
                    "map75": metrics["ap75"],
                    "map50_95": metrics["map50_95"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "T5_raw_map50_95": comparison["T5_raw"],
                    "absolute_map50_95_recovered_from_T5": comparison[
                        "change_from_T5"
                    ],
                    "baseline_map50_95": comparison["baseline"],
                    "map50_95_retention_vs_baseline_percent": 100.0
                    * comparison["retention_vs_baseline"],
                    "map50_95_loss_vs_baseline_percent": 100.0
                    * comparison["normalized_drop_vs_baseline"],
                    "model": evaluation["model"],
                    "evaluation_record": relative(record_path),
                }
            )
    write_csv(TABLES_ROOT / "T6_FINAL_RESULTS.csv", rows)
    return rows


def write_summary(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    best_rows = [row for row in rows if row["checkpoint"] == "best"]
    lines = [
        "# T6 - End-Only 20-Epoch Fine-Tuning",
        "",
        "T6 reloads the final T5 architecture and fine-tunes it once at the end.",
        "No additional pruning or separate BatchNorm recalibration is performed.",
        "",
        "## Primary best-checkpoint results",
        "",
        "| Domain | mAP50 | mAP75 | mAP50-95 | Retention vs baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_rows:
        lines.append(
            f"| {row['domain']} | {float(row['map50']):.6f} "
            f"| {float(row['map75']):.6f} "
            f"| {float(row['map50_95']):.6f} "
            f"| {float(row['map50_95_retention_vs_baseline_percent']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "Both best and last checkpoints are retained in "
            "`tables/T6_FINAL_RESULTS.csv`.",
            "",
            f"CUDA device: `{evidence['cuda_device']}`.",
            "",
        ]
    )
    base.atomic_text(OUTPUT_ROOT / "README.md", "\n".join(lines))


def run_smoke(domains: list[str]) -> None:
    evidence = preflight()
    atomic_json(RECORDS_ROOT / "PREFLIGHT.json", evidence)
    for domain in domains:
        train_domain(domain, smoke=True)


def run_full(domains: list[str]) -> None:
    evidence = preflight()
    atomic_json(RECORDS_ROOT / "PREFLIGHT.json", evidence)
    for domain in domains:
        train_domain(domain, smoke=False)
    if set(domains) == {"GEN", "SNOW"}:
        rows = evaluate_full_runs()
        write_summary(rows, evidence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument(
        "--domain",
        choices=("GEN", "SNOW", "BOTH"),
        default="BOTH",
    )
    args = parser.parse_args()
    domains = ["GEN", "SNOW"] if args.domain == "BOTH" else [args.domain]
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
    elif args.smoke:
        run_smoke(domains)
    else:
        run_full(domains)
    return 0
