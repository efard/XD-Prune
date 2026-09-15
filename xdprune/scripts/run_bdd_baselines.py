"""Train traceable BDD100K GEN-2 and NGN-2 YOLO26n baselines.

This launcher is separate from the sealed historical GEN/SNOW runner.  It
preserves the same training and standalone FP32 validation settings while
recording new BDD-specific evidence below ``results/baselines``.

Examples (from the workspace root):
    python scripts/run_bdd_baselines.py --dataset gen2 --check-only
    python scripts/run_bdd_baselines.py --dataset gen2 --full
    python scripts/run_bdd_baselines.py --dataset ngn2 --full
    python scripts/run_bdd_baselines.py --dataset gen2 --finalize-existing
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["YOLO_OFFLINE"] = "true"

import torch
import yaml
from ultralytics import YOLO, __version__ as ultralytics_version
from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg

from baseline_research_metrics import (
    checkpoint_profile,
    create_coco_ground_truth,
    run_coco_evaluation,
    validate_training_csv,
    write_artifact_manifest,
    write_confusion_matrix_outputs,
    write_per_class_outputs,
)
from run_baselines import (
    StableWindowsDetectionTrainer,
    adaptation_audit,
    fused_inference_profile,
    metric_operating_point,
    metric_value,
    require_cuda,
    standalone_fp32_validation,
)
from research_detection_validator import PREDICTIONS_FILENAME


STUDY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_ROOT
CHECKPOINT = STUDY_ROOT / "models" / "yolo26n_coco_pretrained.pt"
REFERENCE_CONFIG = STUDY_ROOT / "configs" / "baselines" / "b_gen_mio_s42_v1.yaml"
ENVIRONMENT = STUDY_ROOT / "configs" / "baselines" / "environment_v1.json"
RESULTS_ROOT = STUDY_ROOT / "results" / "baselines"
RUN_MANIFEST = RESULTS_ROOT / "bdd_baseline_runs.csv"

MANIFEST_FIELDS = [
    "run_id", "run_stage", "date_utc", "status", "dataset_role", "dataset_id",
    "dataset_yaml", "dataset_yaml_sha256", "split_manifest", "split_manifest_sha256",
    "config_path", "config_sha256", "seed", "initial_checkpoint",
    "initial_checkpoint_sha256", "selected_checkpoint", "selected_checkpoint_sha256",
    "completed_epochs", "best_epoch", "parameter_count", "macs_gflops",
    "checkpoint_size_bytes", "map50_95", "map50", "precision", "recall",
    "validation_images", "validation_instances", "output_dir", "command", "notes",
    "failure_reason",
]


@dataclass(frozen=True)
class BddBaseline:
    key: str
    role: str
    dataset_id: str
    output_name: str
    config: Path
    dataset_yaml: Path
    split_manifest: Path
    expected_classes: int = 6


BASELINES = {
    "gen2": BddBaseline(
        key="gen2",
        role="GEN-2",
        dataset_id="bdd100k_clear_adverse_alltime_gen_v1",
        output_name="b_gen2_bdd_clear_alltime_yolo26n_s42_v1",
        config=STUDY_ROOT / "configs" / "baselines" / "b_gen2_bdd_clear_alltime_s42_v1.yaml",
        dataset_yaml=STUDY_ROOT / "data_views" / "bdd100k_clear_adverse_alltime" / "bdd_gen" / "dataset.yaml",
        split_manifest=STUDY_ROOT / "data_views" / "bdd100k_clear_adverse_alltime" / "bdd_gen" / "manifest.csv",
    ),
    "ngn2": BddBaseline(
        key="ngn2",
        role="NGN-2",
        dataset_id="bdd100k_clear_adverse_alltime_ngn_v1",
        output_name="b_ngn2_bdd_adverse_alltime_yolo26n_s42_v1",
        config=STUDY_ROOT / "configs" / "baselines" / "b_ngn2_bdd_adverse_alltime_s42_v1.yaml",
        dataset_yaml=STUDY_ROOT / "data_views" / "bdd100k_clear_adverse_alltime" / "bdd_ngn" / "dataset.yaml",
        split_manifest=STUDY_ROOT / "data_views" / "bdd100k_clear_adverse_alltime" / "bdd_ngn" / "manifest.csv",
    ),
}


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


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(WORKSPACE_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected YAML mapping: {path}")
    return value


def append_manifest(row: dict[str, Any]) -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_header = not RUN_MANIFEST.exists()
    if not write_header:
        with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
            if next(csv.reader(handle), []) != MANIFEST_FIELDS:
                raise RuntimeError(f"Manifest header drift: {RUN_MANIFEST}")
    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS, extrasaction="raise")
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in MANIFEST_FIELDS})


def validate_baseline(baseline: BddBaseline) -> dict[str, Any]:
    for path in (CHECKPOINT, ENVIRONMENT, REFERENCE_CONFIG, baseline.config, baseline.dataset_yaml, baseline.split_manifest):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = load_yaml(REFERENCE_CONFIG)
    config = load_yaml(baseline.config)
    get_cfg(DEFAULT_CFG_DICT, overrides=config)
    differences = {key for key in set(reference) | set(config) if reference.get(key) != config.get(key)}
    if differences != {"data", "name"}:
        raise RuntimeError(f"{baseline.config.name} differs from the frozen baseline settings: {sorted(differences)}")
    if config.get("data") != relative(baseline.dataset_yaml) or config.get("name") != baseline.output_name:
        raise RuntimeError(f"Unexpected BDD config identity: {baseline.config}")
    if config.get("project") != "results/baselines" or config.get("exist_ok") is not False:
        raise RuntimeError("BDD baseline output policy must be non-overwriting and use results/baselines")
    data = load_yaml(baseline.dataset_yaml)
    names = data.get("names")
    if not isinstance(names, dict) or list(names) != list(range(baseline.expected_classes)):
        raise RuntimeError(f"Unexpected BDD class mapping: {baseline.dataset_yaml}")
    with baseline.split_manifest.open(newline="", encoding="utf-8") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if not manifest_rows or set().union(*(set(row) for row in manifest_rows)) != {
        "split", "image", "source_image", "source_annotation", "weather", "timeofday",
        "retained_box_count", "retained_categories", "image_materialization",
    }:
        raise RuntimeError(f"Unexpected BDD split manifest schema: {baseline.split_manifest}")
    for split in ("train", "val"):
        split_path = (baseline.dataset_yaml.parent / str(data.get(split, ""))).resolve()
        if not split_path.is_dir():
            raise FileNotFoundError(f"Missing configured split: {split_path}")
        manifest_images = {row["image"] for row in manifest_rows if row.get("split") == split}
        if not manifest_images or len(manifest_images) != sum(row.get("split") == split for row in manifest_rows):
            raise RuntimeError(f"{baseline.role}/{split} manifest is empty or has duplicate image identities")
        image_names = {path.name for path in split_path.glob("*.jpg")}
        label_names = {path.stem for path in (baseline.dataset_yaml.parent / "labels" / split).glob("*.txt")}
        expected_labels = {Path(name).stem for name in manifest_images}
        if image_names != manifest_images or label_names != expected_labels:
            raise RuntimeError(
                f"{baseline.role}/{split} mismatch: manifest={len(manifest_images)}, "
                f"images={len(image_names)}, labels={len(label_names)}"
            )
    return config


def base_row(baseline: BddBaseline, config: dict[str, Any], run_id: str, stage: str, output: Path, command: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_stage": stage,
        "date_utc": utc_now(),
        "dataset_role": baseline.role,
        "dataset_id": baseline.dataset_id,
        "dataset_yaml": relative(baseline.dataset_yaml),
        "dataset_yaml_sha256": sha256(baseline.dataset_yaml),
        "split_manifest": relative(baseline.split_manifest),
        "split_manifest_sha256": sha256(baseline.split_manifest),
        "config_path": relative(baseline.config),
        "config_sha256": sha256(baseline.config),
        "seed": config["seed"],
        "initial_checkpoint": relative(CHECKPOINT),
        "initial_checkpoint_sha256": sha256(CHECKPOINT),
        "output_dir": relative(output),
        "command": command,
        "notes": (
            f"launcher={relative(Path(__file__))}; launcher_sha256={sha256(Path(__file__))}; "
            f"environment={relative(ENVIRONMENT)}; environment_sha256={sha256(ENVIRONMENT)}; "
            "historical GEN/SNOW execution freeze remains sealed and does not claim this new BDD study; "
            "dataloader_pin_memory=false; test split not accessed"
        ),
    }


def primary_metrics_from_saved_per_class(path: Path, expected_classes: int) -> dict[str, float]:
    """Recover overall metrics exactly as means of the saved class-wise values."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing saved per-class validation metrics: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) != expected_classes:
        raise RuntimeError(f"Expected {expected_classes} saved per-class records, got {type(records).__name__} length={len(records) if isinstance(records, list) else 'n/a'}")
    expected_ids = list(range(expected_classes))
    if sorted(int(record.get("class_id", -1)) for record in records if isinstance(record, dict)) != expected_ids:
        raise RuntimeError("Saved per-class records do not cover the expected contiguous class IDs")
    source_to_target = {
        "ap50_95": "map50_95",
        "ap50": "map50",
        "precision": "precision",
        "recall": "recall",
    }
    primary: dict[str, float] = {}
    for source, target in source_to_target.items():
        values = [float(record[source]) for record in records]
        if not all(value >= 0.0 and value <= 1.0 for value in values):
            raise RuntimeError(f"Saved per-class {source} values are out of range")
        primary[target] = sum(values) / len(values)
    return primary


def finalize_existing(baseline: BddBaseline, command: str) -> None:
    """Complete evidence export after a post-training export-only failure.

    This never calls ``train`` and refuses to overwrite a completed summary.
    It is intentionally limited to a run whose normal validation artifacts are
    already present under the frozen BDD baseline output name.
    """
    config = validate_baseline(baseline)
    training_dir = (RESULTS_ROOT / str(config["name"])).resolve()
    validation_dir = training_dir / "fp32_val"
    best = training_dir / "weights" / "best.pt"
    last = training_dir / "weights" / "last.pt"
    summary_path = training_dir / "bdd_baseline_run_summary.json"
    if summary_path.exists():
        raise FileExistsError(f"Existing baseline evidence is already finalized: {summary_path}")
    for required in (best, last, training_dir / "results.csv", validation_dir / PREDICTIONS_FILENAME, validation_dir / "validation_ground_truth_coco.json", validation_dir / "validation_support.json", validation_dir / "per_class_metrics.json"):
        if not required.is_file():
            raise FileNotFoundError(f"Cannot finalize missing training evidence: {required}")

    row = base_row(baseline, config, str(config["name"]), "bdd_baseline_finalize_existing", training_dir, command)
    row["status"] = "started"
    append_manifest(row)
    try:
        history_rows = sum(1 for _ in csv.DictReader((training_dir / "results.csv").open(newline="", encoding="utf-8-sig")))
        history = validate_training_csv(training_dir / "results.csv", history_rows)
        profile = checkpoint_profile(best, baseline.expected_classes, int(config["imgsz"]))
        fused = fused_inference_profile(best, baseline.expected_classes, int(config["imgsz"]))
        primary = primary_metrics_from_saved_per_class(validation_dir / "per_class_metrics.json", baseline.expected_classes)
        ground_truth = validation_dir / "validation_ground_truth_coco.json"
        coco_metrics = run_coco_evaluation(ground_truth, validation_dir / PREDICTIONS_FILENAME, validation_dir)
        support = json.loads((validation_dir / "validation_support.json").read_text(encoding="utf-8"))
        summary = {
            "schema_version": "bdd_baseline_evidence_v1",
            "status": "completed_after_post_training_export_recovery",
            "dataset_role": baseline.role,
            "dataset_id": baseline.dataset_id,
            "test_split_accessed": False,
            "recovery_note": (
                "Training and standalone FP32 validation completed before the secondary COCO export rejected "
                "edge-clipped zero-area predictions. This summary reuses those saved validation artifacts; "
                "no training or validation inference was repeated."
            ),
            "training": {
                "wall_seconds": None,
                "wall_seconds_note": "Not recoverable after the original post-training failure; results.csv cumulative time is retained in history.",
                "completed_epochs": history_rows,
                "best_epoch": history["best_epoch_from_results_csv"],
                "history": history,
            },
            "checkpoints": {"best": profile, "last_path": relative(last), "fused_best_inference_graph": fused},
            "standalone_fp32_validation": {
                **primary,
                "overall_metric_recovery": "Arithmetic mean of the saved six per-class FP32 validation metrics.",
                "validation_images": int(support["images"]),
                "validation_instances": int(support["instances"]),
                "per_class_metrics": relative(validation_dir / "per_class_metrics.json"),
                "secondary_coco_metrics": relative(coco_metrics),
            },
        }
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifact_manifest = write_artifact_manifest(training_dir, [path for path in training_dir.rglob("*") if path.is_file()], training_dir / "artifact_manifest.json")
        completed = dict(row)
        completed.update({
            "date_utc": utc_now(), "status": "completed", "selected_checkpoint": relative(best),
            "selected_checkpoint_sha256": sha256(best), "completed_epochs": history_rows,
            "best_epoch": history["best_epoch_from_results_csv"], "parameter_count": profile["unfused_parameters"],
            "macs_gflops": profile["ultralytics_get_flops_gflops_unfused"],
            "checkpoint_size_bytes": best.stat().st_size, **primary,
            "validation_images": int(support["images"]), "validation_instances": int(support["instances"]),
            "notes": row["notes"] + f"; recovered_post_training_evidence=true; artifact_manifest={relative(artifact_manifest)}",
        })
        append_manifest(completed)
        print(f"{baseline.role} evidence finalization PASS: mAP50-95={primary['map50_95']:.6f}; output={training_dir}")
    except BaseException as exc:
        failed = dict(row)
        failed.update({"date_utc": utc_now(), "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
        append_manifest(failed)
        traceback.print_exc()
        raise


def run_one(baseline: BddBaseline, action: str, stamp: str, command: str) -> None:
    config = validate_baseline(baseline)
    name = str(config["name"]) if action == "full" else f"smoke_{baseline.key}_bdd_s42_v1_{stamp}"
    output = (RESULTS_ROOT if action == "full" else RESULTS_ROOT / "smoke") / name
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {output}")
    effective = dict(config)
    if action == "smoke":
        effective.update(epochs=1, patience=0, fraction=0.01)
    row = base_row(baseline, effective, name, f"bdd_baseline_{action}", output, command)
    row["status"] = "started"
    append_manifest(row)
    source: YOLO | None = None
    try:
        source = YOLO(str(CHECKPOINT), task="detect")
        transferred, target_parameters = adaptation_audit(source, baseline.expected_classes)
        print(f"{baseline.role}: nc={baseline.expected_classes}; transferred={transferred}; target parameters={target_parameters:,}")
        torch.cuda.empty_cache()
        started = time.perf_counter()
        source.train(
            trainer=StableWindowsDetectionTrainer,
            cfg=str(baseline.config),
            project=str(output.parent),
            name=name,
            exist_ok=False,
            **({"epochs": 1, "patience": 0, "fraction": 0.01} if action == "smoke" else {}),
        )
        training_seconds = time.perf_counter() - started
        trainer = source.trainer
        training_dir = Path(trainer.save_dir).resolve()
        best, last = Path(trainer.best).resolve(), Path(trainer.last).resolve()
        if training_dir != output or not best.is_file() or not last.is_file():
            raise RuntimeError(f"Missing expected training output under {output}")
        completed_epochs = int(trainer.epoch) + 1
        best_epoch = int(trainer.selected_best_epoch)
        if best_epoch < 1:
            raise RuntimeError("No best checkpoint epoch was recorded")
        history = validate_training_csv(training_dir / "results.csv", completed_epochs)
        profile = checkpoint_profile(best, baseline.expected_classes, int(config["imgsz"]))
        fused = fused_inference_profile(best, baseline.expected_classes, int(config["imgsz"]))
        del trainer, source
        source = None
        gc.collect()
        torch.cuda.empty_cache()
        metrics, validation_model, validation_dir, validation_runtime = standalone_fp32_validation(best, baseline, training_dir)
        per_class_json, _ = write_per_class_outputs(metrics, validation_model.names, validation_dir)
        if len(validation_model.names) != baseline.expected_classes:
            raise RuntimeError(f"Trained nc={len(validation_model.names)}, expected {baseline.expected_classes}")
        confusion = write_confusion_matrix_outputs(metrics, validation_model.names, validation_dir)
        ground_truth, support_path = create_coco_ground_truth(baseline.dataset_yaml, "val", validation_dir)
        coco_metrics = run_coco_evaluation(ground_truth, validation_dir / PREDICTIONS_FILENAME, validation_dir)
        support = json.loads(support_path.read_text(encoding="utf-8"))
        primary = {
            "map50_95": metric_value(metrics.box.map),
            "map50": metric_value(metrics.box.map50),
            "precision": metric_value(metrics.box.mp),
            "recall": metric_value(metrics.box.mr),
        }
        summary = {
            "schema_version": "bdd_baseline_evidence_v1",
            "status": "completed",
            "dataset_role": baseline.role,
            "dataset_id": baseline.dataset_id,
            "test_split_accessed": False,
            "training": {"wall_seconds": training_seconds, "completed_epochs": completed_epochs, "best_epoch": best_epoch, "history": history},
            "checkpoints": {"best": profile, "last_path": relative(last), "fused_best_inference_graph": fused},
            "standalone_fp32_validation": {
                **primary,
                "validation_images": int(support["images"]),
                "validation_instances": int(support["instances"]),
                "operating_point": metric_operating_point(metrics),
                "per_class_metrics": relative(per_class_json),
                "confusion_matrix": {key: relative(value) for key, value in confusion.items()},
                "secondary_coco_metrics": relative(coco_metrics),
                **validation_runtime,
            },
        }
        (training_dir / "bdd_baseline_run_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        artifact_manifest = write_artifact_manifest(training_dir, [path for path in training_dir.rglob("*") if path.is_file()], training_dir / "artifact_manifest.json")
        completed = dict(row)
        completed.update({
            "date_utc": utc_now(), "status": "completed", "selected_checkpoint": relative(best),
            "selected_checkpoint_sha256": sha256(best), "completed_epochs": completed_epochs,
            "best_epoch": best_epoch, "parameter_count": profile["unfused_parameters"],
            "macs_gflops": profile["ultralytics_get_flops_gflops_unfused"],
            "checkpoint_size_bytes": best.stat().st_size, **primary,
            "validation_images": int(support["images"]), "validation_instances": int(support["instances"]),
            "notes": row["notes"] + f"; artifact_manifest={relative(artifact_manifest)}",
        })
        append_manifest(completed)
        print(f"{baseline.role} PASS: mAP50-95={primary['map50_95']:.6f}; output={training_dir}")
    except BaseException as exc:
        failed = dict(row)
        failed.update({"date_utc": utc_now(), "status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "failure_reason": f"{type(exc).__name__}: {exc}"})
        append_manifest(failed)
        traceback.print_exc()
        raise
    finally:
        if source is not None:
            del source
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gen2", "ngn2"), required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--smoke", action="store_true")
    action.add_argument("--full", action="store_true")
    action.add_argument("--finalize-existing", action="store_true")
    args = parser.parse_args()
    os.chdir(WORKSPACE_ROOT)
    baseline = BASELINES[args.dataset]
    config = validate_baseline(baseline)
    print(f"BDD baseline specification PASS: {baseline.role}; Ultralytics={ultralytics_version}; config={baseline.config}")
    if args.check_only:
        return 0
    command = subprocess.list2cmdline([sys.executable, *sys.argv])
    if args.finalize_existing:
        finalize_existing(baseline, command)
        return 0
    require_cuda()
    action_name = "smoke" if args.smoke else "full"
    run_one(baseline, action_name, utc_stamp(), command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
