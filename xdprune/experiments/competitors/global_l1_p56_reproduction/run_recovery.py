"""Run matched 20-epoch GEN recovery for unrestricted Global-L1 p56.

The pruning architecture is generated separately by generate_global_l1_p56.py.
This runner keeps the same data, optimizer, image size, batch size, seed and
evaluation path as the verified unrestricted p10 reproduction. A separate
continuation runner can extend this same run to 40 total epochs if the curve
is still rising at epoch 20.
"""

from __future__ import annotations

import csv
import gc
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
from ultralytics import YOLO


SELF = Path(__file__).resolve()
OUTPUT_ROOT = SELF.parent
STUDY_ROOT = SELF.parents[3]
PROJECT_ROOT = STUDY_ROOT
T6_ROOT = STUDY_ROOT / "experiments" / "recovery" / "recovery_10pct" / "T6"
if str(T6_ROOT) not in sys.path:
    sys.path.insert(0, str(T6_ROOT))

import t6_engine as recovery


SCHEMA = "global_l1_p56_unrestricted_recovery_20epochs_v1"
RAW_MODEL = OUTPUT_ROOT / "raw" / "global_l1_gen_p56_unrestricted_raw.pth"
PRUNING_MANIFEST = OUTPUT_ROOT / "experiment_manifest.json"
BASELINE = (
    STUDY_ROOT
    / "results"
    / "baselines"
    / "b_gen_mio_yolo26n_s42_v1"
    / "weights"
    / "best.pt"
)
BASELINE_SUMMARY = (
    STUDY_ROOT
    / "results"
    / "baselines"
    / "b_gen_mio_yolo26n_s42_v1"
    / "baseline_run_summary.json"
)
DATASET = STUDY_ROOT / "data_views" / "mio_full_s42_v1" / "mio_tcd_full.yaml"
TRAINING_ROOT = OUTPUT_ROOT / "recovery" / "training"
RECORDS_ROOT = OUTPUT_ROOT / "recovery" / "records"
MODELS_ROOT = OUTPUT_ROOT / "recovery" / "models"
TABLES_ROOT = OUTPUT_ROOT / "recovery" / "tables"
RUN_NAME = "gen_global_l1_p56_full20_s42_v1"
EPOCHS = 20
EXPECTED_PARAMETERS = 1_083_192
BASELINE_PARAMETERS = 2_508_090
DOMAIN = "GEN"


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def preflight() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for matched recovery")
    for path in (RAW_MODEL, PRUNING_MANIFEST, BASELINE, BASELINE_SUMMARY, DATASET):
        if not path.is_file():
            raise FileNotFoundError(path)
    pruning = read_json(PRUNING_MANIFEST)
    if pruning.get("status") != "PASS":
        raise RuntimeError("Global L1 physical-pruning manifest is not complete")
    manifest_parameters = int(pruning["final_parameters"])
    model = torch.load(RAW_MODEL, map_location="cpu", weights_only=False).float().eval()
    parameters = recovery.count_parameters(model)
    if parameters != manifest_parameters:
        raise RuntimeError(
            f"Raw Global L1 p56 architecture mismatch: observed {parameters}, "
            f"manifest {manifest_parameters}"
        )
    # Group pruning can overshoot the nominal target by one legal channel.
    # Use the manifest's physically observed count for all subsequent checks.
    global EXPECTED_PARAMETERS
    EXPECTED_PARAMETERS = manifest_parameters
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 640, 640))
    public = recovery.base.public_prediction_summary(output)
    if not public["all_finite"]:
        raise RuntimeError("Raw Global L1 p56 output is not finite")
    del model, output
    gc.collect()
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "cuda_device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "raw_model": relative(RAW_MODEL),
        "raw_model_sha256": recovery.base.sha256(RAW_MODEL),
        "pruning_manifest": relative(PRUNING_MANIFEST),
        "pruning_manifest_sha256": recovery.base.sha256(PRUNING_MANIFEST),
        "baseline": relative(BASELINE),
        "baseline_sha256": recovery.base.sha256(BASELINE),
        "dataset": relative(DATASET),
        "dataset_sha256": recovery.base.sha256(DATASET),
        "parameters": parameters,
        "nominal_target_parameters": 1_083_192,
        "epochs": EPOCHS,
        "seed": 42,
        "imgsz": 640,
        "batch": 16,
        "workers": 4,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "public_output": public,
        "test_data_used": False,
    }


def train() -> dict[str, Any]:
    record_path = RECORDS_ROOT / f"{DOMAIN}_full20_training.json"
    if record_path.is_file():
        record = read_json(record_path)
        if record.get("status") == "PASS":
            return record
        raise RuntimeError("A non-complete recovery record already exists; preserve and inspect it")
    run_dir = TRAINING_ROOT / RUN_NAME
    if run_dir.exists():
        raise RuntimeError("An incomplete recovery directory already exists; preserve and inspect it")

    model = torch.load(RAW_MODEL, map_location="cpu", weights_only=False).float()
    if recovery.count_parameters(model) != EXPECTED_PARAMETERS:
        raise RuntimeError("Raw model parameter count changed before training")
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    recovery.ExactPrunedArchitectureTrainer.expected_parameters = EXPECTED_PARAMETERS
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_model": relative(RAW_MODEL),
        "input_model_sha256": recovery.base.sha256(RAW_MODEL),
        "parameters_before": EXPECTED_PARAMETERS,
        "epochs_requested": EPOCHS,
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
            "model": str(BASELINE),
            "data": str(DATASET),
            "project": str(TRAINING_ROOT),
            "name": RUN_NAME,
            "exist_ok": False,
            "epochs": EPOCHS,
            "patience": 0,
            "batch": 16,
            "imgsz": 640,
            "device": 0,
            "workers": 4,
            "cache": False,
            "seed": 42,
            "deterministic": True,
            "fraction": 1.0,
            "pretrained": False,
            "amp": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": 1.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.0,
            "cos_lr": False,
            "close_mosaic": 0,
            "save": True,
            "save_period": 5,
            "val": True,
            "plots": True,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "verbose": True,
        }
        trainer = recovery.ExactPrunedArchitectureTrainer(overrides=overrides)
        trainer.model = model
        trainer.train()
        best = run_dir / "weights" / "best.pt"
        last = run_dir / "weights" / "last.pt"
        if not best.is_file() or not last.is_file():
            raise FileNotFoundError("Recovery did not produce best.pt and last.pt")
        completed_epochs = int(trainer.epoch) + 1
        if completed_epochs != EPOCHS:
            raise RuntimeError(f"Expected {EPOCHS} epochs, observed {completed_epochs}")
        exports = {
            label: recovery.export_plain_model(
                checkpoint,
                MODELS_ROOT / f"{DOMAIN}_GlobalL1_p56_full20_{label}.pth",
                EXPECTED_PARAMETERS,
            )
            for label, checkpoint in (("best", best), ("last", last))
        }
        record.update(
            {
                "status": "PASS",
                "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": time.perf_counter() - started,
                "completed_epochs": completed_epochs,
                "training_directory": relative(run_dir),
                "parameters_after": EXPECTED_PARAMETERS,
                "best_epoch": int(trainer.selected_best_epoch),
                "optimizer_step_calls": int(trainer.optimizer_step_calls),
                "successful_optimizer_updates": int(trainer.successful_optimizer_updates),
                "amp_skipped_updates": int(trainer.amp_skipped_updates),
                "exports": exports,
            }
        )
    except BaseException as error:
        record.update(
            {
                "status": "FAIL",
                "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": time.perf_counter() - started,
                "error": {"type": type(error).__name__, "message": str(error)},
            }
        )
        atomic_json(record_path, record)
        raise
    atomic_json(record_path, record)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def evaluate(training: dict[str, Any]) -> list[dict[str, Any]]:
    baseline = read_json(BASELINE_SUMMARY)
    rows: list[dict[str, Any]] = []
    for label in ("best", "last"):
        model_path = PROJECT_ROOT / training["exports"][label]["plain_model"]
        record_path = RECORDS_ROOT / f"{DOMAIN}_full20_{label}_evaluation.json"
        if record_path.is_file():
            result = read_json(record_path)
        else:
            model = torch.load(model_path, map_location="cpu", weights_only=False).float().eval()
            yolo = YOLO(str(BASELINE), task="detect")
            yolo.model = model
            overall, per_class = recovery.custom.evaluate(
                yolo,
                DATASET,
                DOMAIN,
                f"GlobalL1_{DOMAIN}_p56_{label}",
            )
            result = {
                "schema": SCHEMA,
                "status": "PASS",
                "checkpoint": label,
                "model": relative(model_path),
                "model_sha256": recovery.base.sha256(model_path),
                "parameters": EXPECTED_PARAMETERS,
                "metrics": overall,
                "per_class_metrics": per_class,
                "baseline_map50_95": float(baseline["map50_95"]),
                "map50_95_retention_percent": 100.0
                * float(overall["map50_95"])
                / float(baseline["map50_95"]),
                "test_data_used": False,
            }
            atomic_json(record_path, result)
            del model, yolo
            gc.collect()
        metrics = result["metrics"]
        rows.append(
            {
                "method": "Global_L1_unrestricted",
                "domain": DOMAIN,
                "checkpoint": label,
                "fine_tuning_epochs": EPOCHS,
                "parameters": EXPECTED_PARAMETERS,
                "parameter_reduction_percent": 100.0 * (1.0 - EXPECTED_PARAMETERS / BASELINE_PARAMETERS),
                "map50": metrics["ap50"],
                "map75": metrics["ap75"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "baseline_map50_95": result["baseline_map50_95"],
                "map50_95_retention_percent": result["map50_95_retention_percent"],
                "model": result["model"],
                "evaluation_record": relative(record_path),
            }
        )
    write_csv(TABLES_ROOT / "GLOBAL_L1_P56_FINAL_RESULTS.csv", rows)
    return rows


def main() -> int:
    evidence = preflight()
    atomic_json(RECORDS_ROOT / "PREFLIGHT.json", evidence)
    training = train()
    rows = evaluate(training)
    best = next(row for row in rows if row["checkpoint"] == "best")
    print(json.dumps({"preflight": evidence, "best_result": best}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
