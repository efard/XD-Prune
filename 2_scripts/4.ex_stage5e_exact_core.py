#!/usr/bin/env python3
"""
Stage 5E: matched 20-epoch recovery for one raw Global L1 model.

Once for GEN and once for SNOW.

The exact physically pruned model object is injected into a custom DetectionTrainer.
This prevents Ultralytics from rebuilding the original unpruned architecture from YAML.

Matched recovery protocol
- epochs: 20 continuous epochs
- optimizer: AdamW
- lr0: 0.001
- lrf: 0.01
- momentum: 0.9
- weight decay: 0.0005
- warmup epochs: 1
- batch: 16
- image size: 640
- workers: 4
- AMP: enabled
- deterministic: true
- pretrained: false
- rect during training: false
- mosaic: 1.0
- close_mosaic: 0

Final evaluation
- full validation split
- FP32
- rect: true
- batch: 16
- conf: 0.001
- IoU: 0.70
- max_det: 300
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer
import ultralytics


EXPECTED_PARAMETERS = {
    "GEN": 2_250_880,
    "SNOW": 2_250_272,
}

FROZEN_BASELINES = {
    "GEN": {
        "map50_95": 0.6414207461,
        "map50": 0.8300363459,
    },
    "SNOW": {
        "map50_95": 0.1899135424,
        "map50": 0.3241117518,
    },
}

RAW_METRICS = {
    "GEN": {
        "map50_95": 0.0,
        "map50": 0.0,
    },
    "SNOW": {
        "map50_95": 0.0,
        "map50": 0.0,
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []

    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(item))

    return tensors


def architecture_rows(model: nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for name, module in model.named_modules():
        row: dict[str, Any] = {
            "module_name": name or "<root>",
            "module_type": module.__class__.__name__,
            "direct_parameter_count": sum(
                parameter.numel()
                for parameter in module.parameters(recurse=False)
            ),
        }

        for attribute in (
            "in_channels",
            "out_channels",
            "groups",
            "in_features",
            "out_features",
            "num_features",
            "nc",
            "reg_max",
        ):
            value = getattr(module, attribute, "")
            row[attribute] = (
                value
                if isinstance(value, (str, int, float, bool))
                else ""
            )

        rows.append(row)

    return rows


def architecture_sha256(model: nn.Module) -> str:
    payload = json.dumps(
        architecture_rows(model),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class ExactStructureDetectionTrainer(DetectionTrainer):
    """Return the already-loaded raw pruned model instead of rebuilding YAML."""

    external_exact_model: nn.Module | None = None

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ) -> nn.Module:
        model = type(self).external_exact_model

        if model is None:
            raise RuntimeError(
                "Exact pruned model was not supplied to the trainer."
            )

        type(self).external_exact_model = None
        return model


def parse_results_csv(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Training results not found: {path}")

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Training results CSV has no rows.")

    numeric_rows: list[dict[str, float]] = []
    nonfinite: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        parsed_row: dict[str, float] = {}

        for key, value in row.items():
            if value in (None, ""):
                continue

            try:
                parsed = float(value)
            except ValueError:
                continue

            parsed_row[key] = parsed

            if not math.isfinite(parsed):
                nonfinite.append(
                    {
                        "row": row_index,
                        "column": key,
                        "value": value,
                    }
                )

        numeric_rows.append(parsed_row)

    last = numeric_rows[-1]
    last_losses = {
        key: value
        for key, value in last.items()
        if "loss" in key.lower()
    }

    if not last_losses:
        raise RuntimeError("No loss values found in results.csv.")

    best_training_map = None
    training_map_values = [
        row.get("metrics/mAP50-95(B)")
        for row in numeric_rows
        if row.get("metrics/mAP50-95(B)") is not None
    ]

    if training_map_values:
        best_training_map = max(training_map_values)

    return {
        "rows": len(rows),
        "expected_rows": 20,
        "has_20_rows": len(rows) == 20,
        "all_numeric_values_finite": not nonfinite,
        "nonfinite_values": nonfinite,
        "last_numeric_values": last,
        "last_loss_values": last_losses,
        "best_training_map50_95": best_training_map,
    }


def inspect_checkpoint(
    checkpoint: Path,
    imgsz: int,
    device: torch.device,
) -> dict[str, Any]:
    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(device).eval()

    parameters = count_parameters(model)
    architecture_hash = architecture_sha256(model)

    with torch.no_grad():
        output = model(
            torch.zeros(
                1,
                3,
                imgsz,
                imgsz,
                device=device,
            )
        )

    tensors = flatten_tensors(output)

    if not tensors:
        raise RuntimeError(
            f"{checkpoint.name}: forward output contained no tensors."
        )

    finite = all(
        bool(torch.isfinite(tensor).all().item())
        for tensor in tensors
    )

    result = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "parameters": parameters,
        "architecture_sha256": architecture_hash,
        "forward_pass_ok": True,
        "output_tensor_count": len(tensors),
        "all_output_tensors_finite": finite,
    }

    del model, yolo

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


def metric_value(metrics: Any, path: str) -> float | None:
    value = metrics

    for component in path.split("."):
        value = getattr(value, component, None)

        if value is None:
            return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--domain",
        choices=["GEN", "SNOW"],
        required=True,
    )
    parser.add_argument("--raw-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    domain = args.domain
    expected_parameters = EXPECTED_PARAMETERS[domain]

    for label, path in (
        ("raw model", args.raw_model),
        ("dataset YAML", args.data),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    if args.epochs != 20:
        raise SystemExit(
            "This matched comparison requires exactly 20 epochs."
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(args.seed)

    device = torch.device(
        "cuda:0"
        if args.device not in {"cpu", "mps"}
        and torch.cuda.is_available()
        else args.device
    )

    source_yolo = YOLO(str(args.raw_model.resolve()))
    source_model = source_yolo.model

    source_parameters = count_parameters(source_model)
    source_architecture_hash = architecture_sha256(source_model)

    if source_parameters != expected_parameters:
        raise RuntimeError(
            f"{domain}: source parameters {source_parameters} != "
            f"expected {expected_parameters}"
        )

    source_inspection = inspect_checkpoint(
        args.raw_model.resolve(),
        args.imgsz,
        device,
    )

    training_overrides = {
        "model": str(args.raw_model.resolve()),
        "data": str(args.data.resolve()),
        "task": "detect",
        "mode": "train",
        "epochs": 20,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "pretrained": False,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "rect": False,
        "mosaic": 1.0,
        "close_mosaic": 0,
        "val": True,
        "save": True,
        "save_period": -1,
        "plots": False,
        "verbose": True,
        "cache": False,
        "resume": False,
        "project": str(output_dir),
        "name": "train_run",
        "exist_ok": True,
        "patience": 100,
    }

    protocol = {
        "domain": domain,
        "source_raw_model": str(args.raw_model.resolve()),
        "source_parameters": source_parameters,
        "source_architecture_sha256": source_architecture_hash,
        "dataset_yaml": str(args.data.resolve()),
        "training_overrides": training_overrides,
        "final_validation": {
            "split": "val",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "workers": args.workers,
            "half": False,
            "rect": True,
            "conf": 0.001,
            "iou": 0.70,
            "max_det": 300,
            "augment": False,
        },
        "frozen_baseline_metrics": FROZEN_BASELINES[domain],
        "raw_metrics": RAW_METRICS[domain],
        "exact_structure_trainer": (
            "ExactStructureDetectionTrainer"
        ),
        "ultralytics_version": getattr(
            ultralytics,
            "__version__",
            "unknown",
        ),
        "torch_version": torch.__version__,
    }

    (output_dir / "recovery_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    ExactStructureDetectionTrainer.external_exact_model = source_model

    trainer = ExactStructureDetectionTrainer(
        overrides=training_overrides,
    )

    training_started = time.time()
    trainer.train()
    training_elapsed = time.time() - training_started

    best_path = Path(trainer.best)
    last_path = Path(trainer.last)
    results_csv = Path(trainer.csv)

    if not best_path.is_file():
        raise RuntimeError(f"best.pt not found: {best_path}")

    if not last_path.is_file():
        raise RuntimeError(f"last.pt not found: {last_path}")

    training_results = parse_results_csv(results_csv)

    trained_parameters = count_parameters(trainer.model)
    trained_architecture_hash = architecture_sha256(
        trainer.model
    )

    if trained_parameters != expected_parameters:
        raise RuntimeError(
            f"{domain}: trained model parameters changed to "
            f"{trained_parameters}"
        )

    if trained_architecture_hash != source_architecture_hash:
        raise RuntimeError(
            f"{domain}: trained model architecture changed."
        )

    best_inspection = inspect_checkpoint(
        best_path,
        args.imgsz,
        device,
    )
    last_inspection = inspect_checkpoint(
        last_path,
        args.imgsz,
        device,
    )

    recovered_best = (
        output_dir
        / f"{domain}_global_L1_42root_recovered_best.pt"
    )
    recovered_last = (
        output_dir
        / f"{domain}_global_L1_42root_recovered_last.pt"
    )

    shutil.copy2(best_path, recovered_best)
    shutil.copy2(last_path, recovered_last)

    copied_best_inspection = inspect_checkpoint(
        recovered_best,
        args.imgsz,
        device,
    )
    copied_last_inspection = inspect_checkpoint(
        recovered_last,
        args.imgsz,
        device,
    )

    print(f"\n{domain}: starting final explicit validation")

    validation_yolo = YOLO(str(recovered_best))

    metrics = validation_yolo.val(
        data=str(args.data.resolve()),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        half=False,
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        augment=False,
        plots=False,
        save_json=False,
        verbose=True,
        seed=args.seed,
    )

    recovered_map = metric_value(metrics, "box.map")
    recovered_map50 = metric_value(metrics, "box.map50")
    recovered_map75 = metric_value(metrics, "box.map75")
    recovered_precision = metric_value(metrics, "box.mp")
    recovered_recall = metric_value(metrics, "box.mr")

    if recovered_map is None or recovered_map50 is None:
        raise RuntimeError(
            f"{domain}: final validation metrics could not be extracted."
        )

    baseline = FROZEN_BASELINES[domain]
    raw = RAW_METRICS[domain]

    checks = {
        "training_completed": True,
        "exactly_20_epoch_rows": training_results["has_20_rows"],
        "all_training_values_finite": (
            training_results["all_numeric_values_finite"]
        ),
        "trained_parameters_preserved": (
            trained_parameters == expected_parameters
        ),
        "trained_architecture_preserved": (
            trained_architecture_hash
            == source_architecture_hash
        ),
        "best_parameters_preserved": (
            best_inspection["parameters"]
            == expected_parameters
        ),
        "last_parameters_preserved": (
            last_inspection["parameters"]
            == expected_parameters
        ),
        "best_architecture_preserved": (
            best_inspection["architecture_sha256"]
            == source_architecture_hash
        ),
        "last_architecture_preserved": (
            last_inspection["architecture_sha256"]
            == source_architecture_hash
        ),
        "best_forward_outputs_finite": (
            best_inspection["all_output_tensors_finite"]
        ),
        "last_forward_outputs_finite": (
            last_inspection["all_output_tensors_finite"]
        ),
        "copied_best_matches_original_best": (
            copied_best_inspection["checkpoint_sha256"]
            == best_inspection["checkpoint_sha256"]
        ),
        "copied_last_matches_original_last": (
            copied_last_inspection["checkpoint_sha256"]
            == last_inspection["checkpoint_sha256"]
        ),
        "final_validation_completed": True,
    }

    result = {
        "domain": domain,
        "status": (
            "PASSED_FULL_20E_RECOVERY"
            if all(checks.values())
            else "FAILED_FULL_20E_RECOVERY"
        ),
        "checks": checks,
        "source": source_inspection,
        "trained_model": {
            "parameters": trained_parameters,
            "architecture_sha256": trained_architecture_hash,
        },
        "best": best_inspection,
        "last": last_inspection,
        "recovered_best_copy": copied_best_inspection,
        "recovered_last_copy": copied_last_inspection,
        "training_results": training_results,
        "final_metrics": {
            "baseline_map50_95": baseline["map50_95"],
            "raw_map50_95": raw["map50_95"],
            "recovered_map50_95": recovered_map,
            "signed_accuracy_drop_map50_95": (
                baseline["map50_95"] - recovered_map
            ),
            "accuracy_retention_percent": (
                100.0
                * recovered_map
                / baseline["map50_95"]
            ),
            "recovery_gain_map50_95": (
                recovered_map - raw["map50_95"]
            ),
            "baseline_map50": baseline["map50"],
            "raw_map50": raw["map50"],
            "recovered_map50": recovered_map50,
            "signed_accuracy_drop_map50": (
                baseline["map50"] - recovered_map50
            ),
            "recovery_gain_map50": (
                recovered_map50 - raw["map50"]
            ),
            "recovered_map75": recovered_map75,
            "recovered_precision": recovered_precision,
            "recovered_recall": recovered_recall,
        },
        "training_elapsed_seconds": training_elapsed,
        "output_dir": str(output_dir),
    }

    (output_dir / "domain_recovery_summary.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    write_csv(
        output_dir / "domain_recovery_result.csv",
        [
            {
                "domain": domain,
                "status": result["status"],
                "parameters": expected_parameters,
                "baseline_map50_95": baseline["map50_95"],
                "raw_map50_95": raw["map50_95"],
                "recovered_map50_95": recovered_map,
                "signed_accuracy_drop_map50_95": (
                    baseline["map50_95"] - recovered_map
                ),
                "accuracy_retention_percent": (
                    100.0
                    * recovered_map
                    / baseline["map50_95"]
                ),
                "baseline_map50": baseline["map50"],
                "raw_map50": raw["map50"],
                "recovered_map50": recovered_map50,
                "recovered_map75": recovered_map75,
                "precision": recovered_precision,
                "recall": recovered_recall,
                "best_checkpoint": str(recovered_best),
                "last_checkpoint": str(recovered_last),
            }
        ],
    )

    print(f"\n===== {domain} STAGE 5E COMPLETE =====")
    print(json.dumps(result, indent=2))
    print(f"\nOutput: {output_dir}")


if __name__ == "__main__":
    main()
