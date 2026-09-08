"""Run the frozen T7 pruning path on SNOW with a 5-5-5-5-20 schedule.

Stages 1-4 replay the frozen accepted group sequence and perform five recovery
epochs after each pruning stage. Stage 5 makes no additional architecture
change and performs a final 20-epoch recovery. Intermediate warm-up is
disabled; the final stage uses the single T6-style warm-up.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SELF = Path(__file__).resolve()
OUTPUT_ROOT = SELF.parent
RECOVERY_ROOT = OUTPUT_ROOT.parent
SOURCE_T7_ROOT = RECOVERY_ROOT / "T7"
T6_ENGINE_PATH = RECOVERY_ROOT / "T6" / "t6_engine.py"
STUDY_ROOT = SELF.parents[4]
PROJECT_ROOT = STUDY_ROOT.parent
CORE_SCRIPTS = STUDY_ROOT / "scripts"
PRUNE25_ROOT = STUDY_ROOT / "results" / "pruning" / "prune_25"
SEQUENTIAL_SCRIPTS = (
    STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts"
)
for module_root in (CORE_SCRIPTS, PRUNE25_ROOT, SEQUENTIAL_SCRIPTS):
    if str(module_root) not in sys.path:
        sys.path.insert(0, str(module_root))


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trainlib = load_module("t7_40_trainlib", T6_ENGINE_PATH)
base = importlib.import_module("run_t1_t2")
ratio_engine = importlib.import_module("run_t1_t2_ratio_sweep")
prune_engine = importlib.import_module("run_sequential_gen_protected_v3")

SCHEMA = "t7_snow_frozen_path_5_5_5_5_20_v1"
DOMAIN = "SNOW"
SEED = 42
STAGE_EPOCHS = (5, 5, 5, 5, 20)
STAGE_WARMUP_EPOCHS = (0.0, 0.0, 0.0, 0.0, 1.0)
TOTAL_EPOCHS = sum(STAGE_EPOCHS)

# This is the accepted order used by the completed T5/T7 10.25% model.
# CDG005 jumps from the strict 7.57% frontier to about 10.25%, so stage 4
# crosses both the nominal 8% and 10% targets. Stage 5 therefore performs
# recovery only and makes no further architecture change.
STAGE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("DG025", "DG026"),
    ("DG004", "DG040", "DG019", "DG034", "DG008"),
    (
        "DG009",
        "DG014",
        "DG039",
        "DG033",
        "DG007",
        "DG018",
        "DG012",
        "DG041",
        "DG029",
        "DG013",
        "DG042",
    ),
    ("DG015", "CDG005"),
    (),
)
FROZEN_GROUP_SEQUENCE = tuple(
    group for stage_groups in STAGE_GROUPS for group in stage_groups
)

BASELINE_CHECKPOINT = (
    STUDY_ROOT
    / "results"
    / "baselines"
    / "b_snow_acdc_yolo26n_s42_v1"
    / "weights"
    / "best.pt"
)
DATASET_YAML = (
    STUDY_ROOT
    / "data_views"
    / "acdc_snow_official_v1"
    / "acdc_snow.yaml"
)
SOURCE_MANIFEST = SOURCE_T7_ROOT / "experiment_manifest.json"
T4_PATH = PRUNE25_ROOT / "tables" / "T4_GEN_PROTECTED_25PCT.csv"
BASELINE_METRICS = (
    RECOVERY_ROOT.parent
    / "baseline_validation"
    / "records"
    / "SNOW_baseline_native_metrics.json"
)
MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
FINAL_TABLE = OUTPUT_ROOT / "tables" / "T7_40EPOCH_SNOW_FINAL_RESULTS.csv"
PROTECTED_GROUPS = {"DG001", "DG020"}


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def selected_queue() -> list[dict[str, Any]]:
    rows = base.read_csv(T4_PATH)
    rows.sort(key=lambda row: int(row["rank_by_P_GP_descending"]))
    selected = []
    for row in rows:
        if row["group_id"] in PROTECTED_GROUPS:
            continue
        selected.append(
            {
                **row,
                "T4_rank_25pct": row["rank_by_P_GP_descending"],
                "rank_by_P_GP_desc": row["rank_by_P_GP_descending"],
                "architecture_zone": "UNCLASSIFIED",
                "guard_classification": "FROZEN_T7_REPLAY",
                "local_ratio_percent": 25.0,
            }
        )
    if len(selected) != 49:
        raise RuntimeError("Expected 49 unprotected pruning groups")
    return selected


def apply_group(model: Any, group_id: str) -> dict[str, Any]:
    fraction = __import__("fractions").Fraction(1, 4)
    if group_id.startswith("CDG"):
        result = ratio_engine.apply_custom(model, group_id, fraction)
    else:
        result = ratio_engine.apply_generic(model, group_id, fraction)
    result["guarded_local_ratio_percent"] = 25.0
    return result


def configure_workers(selected: list[dict[str, Any]]) -> None:
    def worker_preflight(
        require_cuda: bool,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Run current checks without obsolete legacy-document dependencies."""

        evidence = preflight(require_cuda=require_cuda)
        generic_ids = {
            row["canonical_group_id"]
            for row in prune_engine.base.read_csv(prune_engine.base.CATALOGUE_PATH)
        }
        custom_ids = set(prune_engine.custom_catalogue())
        unknown = [
            row["group_id"]
            for row in candidates
            if row["group_id"] not in generic_ids | custom_ids
        ]
        if unknown:
            raise RuntimeError(
                f"Groups are absent from validated dependency catalogues: {unknown}"
            )
        return evidence

    prune_engine.SCHEMA = SCHEMA
    prune_engine.RANKING_PATH = T4_PATH
    prune_engine.OUTPUT_ROOT = OUTPUT_ROOT
    prune_engine.ranked_candidates = lambda max_candidates=None: (
        selected if max_candidates is None else selected[:max_candidates]
    )
    # The historical engine's preflight also freezes two prose specification
    # files that were intentionally removed during project cleanup. Those files
    # do not affect pruning. Keep all executable-input and catalogue checks here.
    prune_engine.preflight = worker_preflight
    prune_engine.apply_group = apply_group


def baseline_record() -> dict[str, Any]:
    return base.read_json(BASELINE_METRICS)


def queue_index() -> dict[str, int]:
    return {
        row["group_id"]: index
        for index, row in enumerate(selected_queue(), start=1)
    }


def model_parameters(path: Path) -> int:
    model = trainlib.torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(model, dict):
        model = model.get("ema") if model.get("ema") is not None else model.get("model")
    if model is None or not hasattr(model, "parameters"):
        raise TypeError(f"Cannot locate a model module in {path}")
    value = trainlib.count_parameters(model)
    del model
    return value


def preflight(require_cuda: bool) -> dict[str, Any]:
    for path in (
        T6_ENGINE_PATH,
        SOURCE_MANIFEST,
        T4_PATH,
        BASELINE_METRICS,
        BASELINE_CHECKPOINT,
        DATASET_YAML,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = base.read_json(SOURCE_MANIFEST)
    if source_manifest.get("status") != "PASS_WITH_DOCUMENTED_TARGET_OVERRIDE":
        raise RuntimeError("The source T7 result is not frozen and complete")
    if tuple(source_manifest.get("accepted_groups", ())) != FROZEN_GROUP_SEQUENCE:
        raise RuntimeError("The frozen T7 accepted group order has changed")

    indexes = queue_index()
    missing = sorted(set(FROZEN_GROUP_SEQUENCE) - set(indexes))
    if missing:
        raise RuntimeError(f"Groups missing from the T4 queue: {missing}")

    if require_cuda and not trainlib.torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for training")

    baseline_parameters = model_parameters(BASELINE_CHECKPOINT)
    expected = int(
        source_manifest["final_results"][DOMAIN]["parameters_before"]
    )
    if baseline_parameters != expected:
        raise RuntimeError(
            f"SNOW baseline mismatch: {baseline_parameters} != {expected}"
        )

    return {
        "schema": SCHEMA,
        "domain": DOMAIN,
        "seed": SEED,
        "source_T7_manifest": base.relative(SOURCE_MANIFEST),
        "source_T7_manifest_sha256": base.sha256(SOURCE_MANIFEST),
        "baseline_checkpoint": base.relative(BASELINE_CHECKPOINT),
        "baseline_checkpoint_sha256": base.sha256(
            BASELINE_CHECKPOINT
        ),
        "dataset": base.relative(DATASET_YAML),
        "dataset_sha256": base.sha256(DATASET_YAML),
        "baseline_parameters": baseline_parameters,
        "frozen_group_sequence": list(FROZEN_GROUP_SEQUENCE),
        "stage_groups": [list(groups) for groups in STAGE_GROUPS],
        "stage_epochs": list(STAGE_EPOCHS),
        "stage_warmup_epochs": list(STAGE_WARMUP_EPOCHS),
        "total_epochs": TOTAL_EPOCHS,
        "cuda_available": trainlib.torch.cuda.is_available(),
        "cuda_device": (
            trainlib.torch.cuda.get_device_name(0)
            if trainlib.torch.cuda.is_available()
            else None
        ),
        "torch": trainlib.torch.__version__,
    }


def worker_prune(
    candidate_index: int,
    group_id: str,
    input_model: Path | None,
    output_root: Path,
) -> int:
    selected = selected_queue()
    configure_workers(selected)
    return prune_engine.worker_prune(
        DOMAIN,
        candidate_index,
        group_id,
        input_model,
        output_root,
        selected,
    )


def run_prune_worker(
    candidate_index: int,
    group_id: str,
    input_model: Path | None,
    output_root: Path,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        str(SELF),
        "--worker-prune",
        "--candidate-index",
        str(candidate_index),
        "--group",
        group_id,
        "--output-root",
        str(output_root),
    ]
    if input_model is not None:
        command.extend(["--input-model", str(input_model)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if result.returncode:
        raise RuntimeError(
            f"Pruning {group_id} failed; inspect {log_path}"
        )


def train_stage(
    stage: int,
    input_model_path: Path,
    epochs: int,
    warmup_epochs: float,
) -> dict[str, Any]:
    stage_root = OUTPUT_ROOT / f"stage_{stage}"
    training_root = stage_root / "training"
    # This tested recovery helper is also reused by later domain-specific T7
    # protocols. Derive artifact names from DOMAIN so GEN runs cannot be
    # mislabeled as SNOW; the original SNOW naming remains unchanged.
    run_name = f"{DOMAIN.lower()}_stage{stage}_{epochs}ep_s{SEED}_v1"
    run_dir = training_root / run_name
    record_path = stage_root / "training_record.json"

    if record_path.is_file():
        record = base.read_json(record_path)
        if record.get("status") == "PASS":
            return record
    if run_dir.exists():
        raise RuntimeError(
            f"Incomplete stage-{stage} training output exists at {run_dir}. "
            "Preserve it before retrying."
        )

    input_model = trainlib.torch.load(
        input_model_path,
        map_location="cpu",
        weights_only=False,
    ).float()
    expected_parameters = trainlib.count_parameters(input_model)
    for parameter in input_model.parameters():
        parameter.requires_grad_(True)

    trainlib.ExactPrunedArchitectureTrainer.expected_parameters = (
        expected_parameters
    )
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "stage": stage,
        "domain": DOMAIN,
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "input_model": base.relative(input_model_path),
        "input_model_sha256": base.sha256(input_model_path),
        "parameters": expected_parameters,
        "epochs_requested": epochs,
        "warmup_epochs": warmup_epochs,
        "mosaic": 1.0,
        "close_mosaic": 0,
    }
    atomic_json(record_path, record)

    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    trainlib.torch.cuda.empty_cache()
    trainer = None
    started = time.perf_counter()
    try:
        overrides = {
            "task": "detect",
            "mode": "train",
            "model": str(BASELINE_CHECKPOINT),
            "data": str(DATASET_YAML),
            "project": str(training_root),
            "name": run_name,
            "exist_ok": False,
            "epochs": epochs,
            "patience": 0,
            "batch": 16,
            "imgsz": 640,
            "device": 0,
            "workers": 4,
            "cache": False,
            "seed": SEED,
            "deterministic": True,
            "fraction": 1.0,
            "pretrained": False,
            "amp": True,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.01,
            "momentum": 0.9,
            "weight_decay": 0.0005,
            "warmup_epochs": warmup_epochs,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.0,
            "cos_lr": False,
            "mosaic": 1.0,
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
        trainer = trainlib.ExactPrunedArchitectureTrainer(overrides=overrides)
        trainer.model = input_model
        trainer.train()
        completed_epochs = int(trainer.epoch) + 1
        if completed_epochs != epochs:
            raise RuntimeError(
                f"Stage {stage}: expected {epochs} epochs, observed "
                f"{completed_epochs}"
            )

        exports: dict[str, Any] = {}
        model_root = stage_root / "models"
        for label in ("best", "last"):
            checkpoint = run_dir / "weights" / f"{label}.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            exports[label] = trainlib.export_plain_model(
                checkpoint,
                model_root / f"{DOMAIN}_stage{stage}_{epochs}ep_{label}.pth",
                expected_parameters,
            )

        record.update(
            {
                "status": "PASS",
                "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
                "duration_seconds": time.perf_counter() - started,
                "completed_epochs": completed_epochs,
                "best_epoch": int(trainer.selected_best_epoch),
                "optimizer_step_calls": int(trainer.optimizer_step_calls),
                "successful_optimizer_updates": int(
                    trainer.successful_optimizer_updates
                ),
                "amp_skipped_updates": int(trainer.amp_skipped_updates),
                "exports": exports,
            }
        )
        atomic_json(record_path, record)
        return record
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
    finally:
        del trainer
        gc.collect()
        trainlib.torch.cuda.empty_cache()


def evaluate_model(model_path: Path, label: str) -> dict[str, Any]:
    model = trainlib.torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    ).float()
    yolo = trainlib.YOLO(str(BASELINE_CHECKPOINT), task="detect")
    yolo.model = model.eval()
    overall, per_class = trainlib.custom.evaluate(
        yolo,
        DATASET_YAML,
        DOMAIN,
        label,
    )
    del model
    del yolo
    gc.collect()
    trainlib.torch.cuda.empty_cache()
    return {"metrics": overall, "per_class_metrics": per_class}


def stage_output_model(record: dict[str, Any], label: str = "last") -> Path:
    path = PROJECT_ROOT / record["exports"][label]["plain_model"]
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def orchestrate(resume: bool) -> int:
    evidence = preflight(require_cuda=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if MANIFEST_PATH.is_file():
        if not resume:
            raise RuntimeError(
                "Experiment output already exists. Use --resume to skip "
                "fully completed stages."
            )
        manifest = base.read_json(MANIFEST_PATH)
        if (
            manifest["source_T7_manifest_sha256"]
            != evidence["source_T7_manifest_sha256"]
        ):
            raise RuntimeError("The frozen source T7 manifest changed")
    else:
        manifest = {
            **evidence,
            "status": "RUNNING",
            "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_stages": [],
            "policy": {
                "snow_only_development_run": True,
                "same_group_order_as_frozen_T7": True,
                "same_25pct_local_pruning_ratio": True,
                "recompute_channel_importance_on_current_recovered_model": True,
                "intermediate_warmup_disabled": True,
                "final_stage_T6_style_warmup": True,
                "batchnorm_updated_during_training": True,
                "separate_batchnorm_recalibration": False,
                "test_data_used": False,
            },
        }
        atomic_json(MANIFEST_PATH, manifest)

    indexes = queue_index()
    completed = set(int(value) for value in manifest["completed_stages"])
    current_model: Path | None = None
    sequence_step = 0

    for stage, (groups, epochs, warmup) in enumerate(
        zip(STAGE_GROUPS, STAGE_EPOCHS, STAGE_WARMUP_EPOCHS),
        start=1,
    ):
        if stage in completed:
            record = base.read_json(
                OUTPUT_ROOT / f"stage_{stage}" / "training_record.json"
            )
            current_model = stage_output_model(record, "last")
            sequence_step += len(groups)
            continue

        pruning_root = OUTPUT_ROOT / f"stage_{stage}" / "pruning"
        for group_id in groups:
            sequence_step += 1
            candidate_index = indexes[group_id]
            run_prune_worker(
                candidate_index,
                group_id,
                current_model,
                pruning_root,
                OUTPUT_ROOT
                / "logs"
                / f"stage{stage}_{sequence_step:02d}_{group_id}_prune.log",
            )
            structure_path = (
                pruning_root
                / "records"
                / f"{DOMAIN}_candidate{candidate_index:02d}_structure.json"
            )
            structure = base.read_json(structure_path)
            if structure.get("status") != "PASS":
                raise RuntimeError(f"Pruning failed: {structure_path}")
            current_model = PROJECT_ROOT / structure["output_model"]
            if not current_model.is_file():
                raise FileNotFoundError(current_model)

        if current_model is None:
            raise RuntimeError("No current model is available for recovery")

        training_record = train_stage(
            stage,
            current_model,
            epochs,
            warmup,
        )
        current_model = stage_output_model(training_record, "last")
        stage_evaluation = evaluate_model(
            current_model,
            f"T7_40_SNOW_stage{stage}_last",
        )
        parameters = model_parameters(current_model)
        reduction_percent = 100.0 * (
            1.0 - parameters / evidence["baseline_parameters"]
        )
        stage_record = {
            "schema": SCHEMA,
            "status": "PASS",
            "stage": stage,
            "groups_added": list(groups),
            "epochs": epochs,
            "warmup_epochs": warmup,
            "architecture_changed": bool(groups),
            "model": base.relative(current_model),
            "model_sha256": base.sha256(current_model),
            "parameters": parameters,
            "parameter_reduction_percent": reduction_percent,
            **stage_evaluation,
        }
        atomic_json(
            OUTPUT_ROOT / f"stage_{stage}" / "stage_evaluation.json",
            stage_record,
        )

        completed.add(stage)
        manifest["completed_stages"] = sorted(completed)
        manifest["last_completed_stage"] = stage
        manifest["current_model"] = base.relative(current_model)
        manifest.setdefault("stage_results", {})[str(stage)] = stage_record
        atomic_json(MANIFEST_PATH, manifest)
        print(
            f"Stage {stage}/5 complete: {epochs} epochs, "
            f"{reduction_percent:.3f}% parameters removed, "
            f"mAP50-95={stage_evaluation['metrics']['map50_95']:.6f}",
            flush=True,
        )

    if current_model is None:
        raise RuntimeError("No final model was produced")

    final_training = base.read_json(
        OUTPUT_ROOT / "stage_5" / "training_record.json"
    )
    baseline = baseline_record()["metrics"]
    final_rows: list[dict[str, Any]] = []
    final_records: dict[str, Any] = {}
    final_root = OUTPUT_ROOT / "models" / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    for checkpoint_label in ("best", "last"):
        model_path = stage_output_model(final_training, checkpoint_label)
        evaluated = evaluate_model(
            model_path,
            f"T7_40_SNOW_final_{checkpoint_label}",
        )
        metrics = evaluated["metrics"]
        parameters = model_parameters(model_path)
        reduction_percent = 100.0 * (
            1.0 - parameters / evidence["baseline_parameters"]
        )
        final_model = (
            final_root / f"SNOW_T7_40epoch_{checkpoint_label}.pth"
        )
        shutil.copy2(model_path, final_model)
        final_record = {
            "schema": SCHEMA,
            "status": "PASS",
            "domain": DOMAIN,
            "checkpoint": checkpoint_label,
            "model": base.relative(final_model),
            "model_sha256": base.sha256(final_model),
            "parameters": parameters,
            "parameter_reduction_percent": reduction_percent,
            "total_fine_tuning_epochs": TOTAL_EPOCHS,
            **evaluated,
        }
        final_records[checkpoint_label] = final_record
        atomic_json(
            OUTPUT_ROOT
            / "records"
            / f"SNOW_final_{checkpoint_label}_evaluation.json",
            final_record,
        )
        final_rows.append(
            {
                "experiment": "T7_SNOW_5_5_5_5_20",
                "domain": DOMAIN,
                "checkpoint": checkpoint_label,
                "accepted_group_count": len(FROZEN_GROUP_SEQUENCE),
                "parameters_before": evidence["baseline_parameters"],
                "parameters_after": parameters,
                "parameter_reduction_percent": reduction_percent,
                "stage_schedule": "5-5-5-5-20",
                "total_fine_tuning_epochs": TOTAL_EPOCHS,
                "map50": metrics["ap50"],
                "map75": metrics["ap75"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "baseline_map50_95": baseline["map50_95"],
                "map50_95_retention_vs_baseline_percent": (
                    100.0
                    * float(metrics["map50_95"])
                    / float(baseline["map50_95"])
                ),
                "model": base.relative(final_model),
                "model_sha256": base.sha256(final_model),
            }
        )

    write_csv(FINAL_TABLE, final_rows)
    manifest.update(
        {
            "status": "PASS",
            "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_results": final_records,
            "final_table": base.relative(FINAL_TABLE),
        }
    )
    atomic_json(MANIFEST_PATH, manifest)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-prune", action="store_true")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()

    if args.preflight:
        print(json.dumps(preflight(require_cuda=False), indent=2))
        return 0
    if args.worker_prune:
        if (
            args.candidate_index is None
            or not args.group
            or args.output_root is None
        ):
            parser.error(
                "worker mode requires --candidate-index, --group and "
                "--output-root"
            )
        return worker_prune(
            args.candidate_index,
            args.group,
            args.input_model,
            args.output_root,
        )
    if args.run:
        return orchestrate(resume=args.resume)
    parser.error("choose --preflight or --run")


if __name__ == "__main__":
    raise SystemExit(main())
