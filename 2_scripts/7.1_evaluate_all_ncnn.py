#!/usr/bin/env python3
"""
Evaluate every NCNN detection model under one folder

For each folder containing both:
    model.ncnn.param
    model.ncnn.bin

the script:
1. reads metadata.yaml
2. detects whether the model belongs to GEN or SNOW from its class names
3. uses the model's own exported input size from metadata.yaml
4. evaluates the actual NCNN model with Ultralytics
5. records mAP50-95, mAP50, mAP75, precision and recall
6. records per-class AP metrics
7. continues to the next model if one model fails
8. writes CSV/JSON results and creates a compact ZIP

The official COCO model is intentionally reported as SKIPPED
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path
from typing import Any

import yaml
from ultralytics import YOLO


DEFAULT_GEN_DATA = Path(
    "/project/ko/afm176/datasets/1_data/reproduction_exact_v1/"
    "GEN_MIO_TCD_exact/dataset_GEN_local.yaml"
)

DEFAULT_SNOW_DATA = Path(
    "/project/ko/afm176/datasets/1_data/reproduction_exact_v1/"
    "SNOW_ACDC_exact/dataset_SNOW_local.yaml"
)

GEN_NAMES = [
    "articulated_truck",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "non_motorized_vehicle",
    "pedestrian",
    "pickup_truck",
    "single_unit_truck",
    "work_van",
]

SNOW_NAMES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_names(value: Any) -> list[str]:
    if isinstance(value, dict):
        def sort_key(item):
            key = item[0]
            try:
                return (0, int(key))
            except Exception:
                return (1, str(key))

        return [
            str(class_name)
            for _, class_name in sorted(
                value.items(),
                key=sort_key,
            )
        ]

    if isinstance(value, list):
        return [str(item) for item in value]

    return []


def read_metadata(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "metadata.yaml"

    if not path.is_file():
        raise FileNotFoundError(
            f"metadata.yaml missing from {model_dir}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = yaml.safe_load(handle)

    if not isinstance(metadata, dict):
        raise RuntimeError(
            f"Invalid metadata.yaml in {model_dir}"
        )

    return metadata


def determine_domain(
    model_names: list[str],
) -> str | None:
    if model_names == GEN_NAMES:
        return "GEN"

    if model_names == SNOW_NAMES:
        return "SNOW"

    return None


def metadata_imgsz(
    metadata: dict[str, Any],
) -> int:
    value = metadata.get("imgsz", 640)

    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"Unsupported imgsz metadata: {value}"
            )

        height = int(value[0])
        width = int(value[1])

        if height != width:
            raise ValueError(
                "This evaluator expects square exported "
                f"input sizes; got {value}"
            )

        return height

    return int(value)


def metric_value(
    metrics: Any,
    path: str,
) -> float | None:
    value = metrics

    for component in path.split("."):
        value = getattr(value, component, None)

        if value is None:
            return None

    try:
        number = float(value)
    except Exception:
        return None

    return number if math.isfinite(number) else None


def safe_vector(value: Any) -> list[float]:
    if value is None:
        return []

    try:
        if hasattr(value, "tolist"):
            value = value.tolist()
    except Exception:
        return []

    if not isinstance(value, list):
        return []

    output = []

    for item in value:
        try:
            number = float(item)
        except Exception:
            output.append(float("nan"))
            continue

        output.append(number)

    return output


def discover_ncnn_models(
    root: Path,
) -> list[Path]:
    model_dirs = []

    for param_file in root.rglob(
        "model.ncnn.param"
    ):
        model_dir = param_file.parent

        if (
            (model_dir / "model.ncnn.bin").is_file()
            and (model_dir / "metadata.yaml").is_file()
        ):
            model_dirs.append(
                model_dir.resolve()
            )

    return sorted(
        set(model_dirs),
        key=lambda path: str(path),
    )


def classify_variant(
    relative_path: str,
    metadata: dict[str, Any],
) -> str:
    lower = relative_path.lower()

    if "int8" in lower:
        return "INT8"

    if (
        "fp16" in lower
        or metadata.get("args", {}).get(
            "quantize"
        ) == 16
    ):
        return "FP16"

    return "FP32"


def make_ultralytics_ncnn_alias(model_dir: Path, alias_root: Path) -> Path:
    """Create a temporary *_ncnn_model symlink without renaming originals."""
    alias_root.mkdir(parents=True, exist_ok=True)
    safe = str(model_dir).replace('/', '__').replace('\\', '__').replace(' ', '_').strip('_')
    alias = alias_root / f"{safe}_ncnn_model"
    if alias.is_symlink() or alias.exists():
        if alias.is_dir() and not alias.is_symlink():
            shutil.rmtree(alias)
        else:
            alias.unlink()
    alias.symlink_to(model_dir, target_is_directory=True)
    return alias


def evaluate_one(
    *,
    model_dir: Path,
    relative_path: str,
    metadata: dict[str, Any],
    data_yaml: Path,
    domain: str,
    output_root: Path,
    workers: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
]:
    imgsz = metadata_imgsz(metadata)
    names = normalize_names(
        metadata.get("names")
    )

    variant = classify_variant(
        relative_path,
        metadata,
    )

    run_name = (
        relative_path.replace("/", "__")
        .replace("\\", "__")
        .replace(" ", "_")
    )

    model_size_bytes = (
        (model_dir / "model.ncnn.param")
        .stat().st_size
        + (model_dir / "model.ncnn.bin")
        .stat().st_size
    )

    started = time.time()

    # Ultralytics recognizes NCNN from a directory name ending in _ncnn_model.
    # Use a temporary symlink alias so the original model folder is untouched.
    alias_dir = make_ultralytics_ncnn_alias(
        model_dir,
        output_root / "_ncnn_aliases",
    )
    model = YOLO(
        str(alias_dir),
        task="detect",
    )

    metrics = model.val(
        data=str(data_yaml.resolve()),
        split="val",
        imgsz=imgsz,
        batch=1,        # Export metadata uses batch=1.
        device="cpu",   # NCNN executes on CPU here.
        workers=workers,
        half=False,     # Precision is defined by the exported NCNN model.
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        augment=False,
        plots=False,
        save_json=False,
        verbose=True,
        project=str(
            output_root / "ultralytics_runs"
        ),
        name=run_name,
        exist_ok=True,
    )

    elapsed = time.time() - started

    speed = getattr(
        metrics,
        "speed",
        {},
    )

    result = {
        "model": relative_path,
        "domain": domain,
        "variant": variant,
        "imgsz": imgsz,
        "classes": len(names),
        "status": "ok",
        "error": "",
        "map50_95": metric_value(
            metrics,
            "box.map",
        ),
        "map50": metric_value(
            metrics,
            "box.map50",
        ),
        "map75": metric_value(
            metrics,
            "box.map75",
        ),
        "precision": metric_value(
            metrics,
            "box.mp",
        ),
        "recall": metric_value(
            metrics,
            "box.mr",
        ),
        "preprocess_ms_per_image": (
            speed.get("preprocess")
            if isinstance(speed, dict)
            else None
        ),
        "inference_ms_per_image": (
            speed.get("inference")
            if isinstance(speed, dict)
            else None
        ),
        "postprocess_ms_per_image": (
            speed.get("postprocess")
            if isinstance(speed, dict)
            else None
        ),
        "wall_time_seconds": elapsed,
        "ncnn_size_bytes": model_size_bytes,
        "ncnn_size_mib": (
            model_size_bytes
            / (1024 * 1024)
        ),
        "param_sha256": sha256_file(
            model_dir / "model.ncnn.param"
        ),
        "bin_sha256": sha256_file(
            model_dir / "model.ncnn.bin"
        ),
        "dataset_yaml": str(
            data_yaml.resolve()
        ),
    }

    # Ultralytics BoxMetrics exposes AP per class in box.maps.
    per_class_maps = safe_vector(
        getattr(
            getattr(metrics, "box", None),
            "maps",
            None,
        )
    )

    # Other vectors are useful when the installed version exposes them.
    per_class_precision = safe_vector(
        getattr(
            getattr(metrics, "box", None),
            "p",
            None,
        )
    )
    per_class_recall = safe_vector(
        getattr(
            getattr(metrics, "box", None),
            "r",
            None,
        )
    )

    per_class_rows = []

    for class_id, class_name in enumerate(
        names
    ):
        per_class_rows.append(
            {
                "model": relative_path,
                "domain": domain,
                "variant": variant,
                "imgsz": imgsz,
                "class_id": class_id,
                "class_name": class_name,
                "map50_95": (
                    per_class_maps[class_id]
                    if class_id
                    < len(per_class_maps)
                    else None
                ),
                "precision": (
                    per_class_precision[class_id]
                    if class_id
                    < len(
                        per_class_precision
                    )
                    else None
                ),
                "recall": (
                    per_class_recall[class_id]
                    if class_id
                    < len(
                        per_class_recall
                    )
                    else None
                ),
            }
        )

    return result, per_class_rows


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-root",
        type=Path,
        required=True,
        help=(
            "Folder containing the extracted "
            "0_yolo26n_official, 1_baseline_light, "
            "2_baseline_full, ... folders."
        ),
    )

    parser.add_argument(
        "--gen-data",
        type=Path,
        default=DEFAULT_GEN_DATA,
    )

    parser.add_argument(
        "--snow-data",
        type=Path,
        default=DEFAULT_SNOW_DATA,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )

    args = parser.parse_args()

    root = args.model_root.resolve()
    output = args.output_dir.resolve()

    if not root.is_dir():
        raise SystemExit(
            f"Model root not found: {root}"
        )

    if not args.gen_data.is_file():
        raise SystemExit(
            f"GEN YAML not found: "
            f"{args.gen_data}"
        )

    if not args.snow_data.is_file():
        raise SystemExit(
            f"SNOW YAML not found: "
            f"{args.snow_data}"
        )

    output.mkdir(
        parents=True,
        exist_ok=True,
    )

    models = discover_ncnn_models(root)

    if not models:
        raise RuntimeError(
            f"No NCNN model folders found under "
            f"{root}"
        )

    print(
        f"Discovered {len(models)} NCNN models."
    )

    aggregate_rows = []
    per_class_rows = []

    for index, model_dir in enumerate(
        models,
        start=1,
    ):
        relative = str(
            model_dir.relative_to(root)
        )

        print()
        print(
            "=" * 78
        )
        print(
            f"[{index}/{len(models)}] {relative}"
        )
        print(
            "=" * 78
        )

        try:
            metadata = read_metadata(
                model_dir
            )

            names = normalize_names(
                metadata.get("names")
            )

            domain = determine_domain(
                names
            )

            if domain is None:
                print(
                    "SKIPPED: class list is not "
                    "compatible with GEN or SNOW."
                )

                aggregate_rows.append(
                    {
                        "model": relative,
                        "domain": "INCOMPATIBLE",
                        "variant": classify_variant(
                            relative,
                            metadata,
                        ),
                        "imgsz": metadata_imgsz(
                            metadata
                        ),
                        "classes": len(names),
                        "status": "skipped",
                        "error": (
                            "Class list does not match "
                            "GEN or SNOW label space."
                        ),
                    }
                )

                continue

            data_yaml = (
                args.gen_data
                if domain == "GEN"
                else args.snow_data
            )

            print(
                f"Domain: {domain}"
            )
            print(
                f"Input size: "
                f"{metadata_imgsz(metadata)}"
            )
            print(
                f"Dataset: {data_yaml}"
            )

            result, class_rows = (
                evaluate_one(
                    model_dir=model_dir,
                    relative_path=relative,
                    metadata=metadata,
                    data_yaml=data_yaml,
                    domain=domain,
                    output_root=output,
                    workers=args.workers,
                )
            )

            aggregate_rows.append(
                result
            )
            per_class_rows.extend(
                class_rows
            )

            print(
                "RESULT: "
                f"mAP50-95="
                f"{result['map50_95']:.6f}, "
                f"mAP50="
                f"{result['map50']:.6f}, "
                f"P="
                f"{result['precision']:.6f}, "
                f"R="
                f"{result['recall']:.6f}"
            )

        except Exception as exc:
            print(
                "FAILED:",
                repr(exc),
            )
            traceback.print_exc()

            aggregate_rows.append(
                {
                    "model": relative,
                    "domain": "",
                    "variant": "",
                    "imgsz": "",
                    "classes": "",
                    "status": "failed",
                    "error": repr(exc),
                }
            )

        # Save progress after EVERY model so an interrupted run still preserves all completed measurements.
        write_csv(
            output
            / "ncnn_accuracy_results.csv",
            aggregate_rows,
            [
                "model",
                "domain",
                "variant",
                "imgsz",
                "classes",
                "status",
                "error",
                "map50_95",
                "map50",
                "map75",
                "precision",
                "recall",
                "preprocess_ms_per_image",
                "inference_ms_per_image",
                "postprocess_ms_per_image",
                "wall_time_seconds",
                "ncnn_size_bytes",
                "ncnn_size_mib",
                "param_sha256",
                "bin_sha256",
                "dataset_yaml",
            ],
        )

        write_csv(
            output
            / "ncnn_accuracy_per_class.csv",
            per_class_rows,
            [
                "model",
                "domain",
                "variant",
                "imgsz",
                "class_id",
                "class_name",
                "map50_95",
                "precision",
                "recall",
            ],
        )

    successful = [
        row
        for row in aggregate_rows
        if row.get("status") == "ok"
    ]

    failed = [
        row
        for row in aggregate_rows
        if row.get("status") == "failed"
    ]

    skipped = [
        row
        for row in aggregate_rows
        if row.get("status") == "skipped"
    ]

    ranked = sorted(
        successful,
        key=lambda row: (
            -float(
                row.get("map50_95")
                or -1
            ),
            str(row.get("model")),
        ),
    )

    write_csv(
        output
        / "ncnn_accuracy_ranked_by_map50_95.csv",
        ranked,
        [
            "model",
            "domain",
            "variant",
            "imgsz",
            "map50_95",
            "map50",
            "map75",
            "precision",
            "recall",
            "inference_ms_per_image",
            "ncnn_size_mib",
        ],
    )

    summary = {
        "status": (
            "PASSED_NCNN_ACCURACY_SWEEP"
            if successful
            else "FAILED_NCNN_ACCURACY_SWEEP"
        ),
        "model_root": str(root),
        "models_discovered": len(
            models
        ),
        "models_successful": len(
            successful
        ),
        "models_failed": len(
            failed
        ),
        "models_skipped_incompatible": len(
            skipped
        ),
        "gen_dataset": str(
            args.gen_data.resolve()
        ),
        "snow_dataset": str(
            args.snow_data.resolve()
        ),
        "evaluation": {
            "split": "val",
            "batch": 1,
            "device": "cpu",
            "confidence_threshold": 0.001,
            "nms_iou_threshold": 0.70,
            "max_detections": 300,
            "rectangular_batches": True,
            "test_time_augmentation": False,
            "input_size": (
                "read independently from each "
                "NCNN metadata.yaml"
            ),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
        },
        "best_models_by_map50_95": [
            {
                "model": row["model"],
                "domain": row["domain"],
                "variant": row["variant"],
                "imgsz": row["imgsz"],
                "map50_95": row[
                    "map50_95"
                ],
                "map50": row["map50"],
            }
            for row in ranked[:10]
        ],
    }

    summary_path = (
        output
        / "ncnn_accuracy_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = output / "README_RESULTS.txt"

    readme.write_text(
        (
            "NCNN ACCURACY SWEEP RESULTS\n"
            "===========================\n\n"
            "Main report file:\n"
            "  ncnn_accuracy_results.csv\n\n"
            "Sorted comparison:\n"
            "  ncnn_accuracy_ranked_by_map50_95.csv\n\n"
            "Per-class results:\n"
            "  ncnn_accuracy_per_class.csv\n\n"
            "Summary:\n"
            "  ncnn_accuracy_summary.json\n\n"
            "Accuracy metrics:\n"
            "- mAP50-95: main overall detection accuracy metric\n"
            "- mAP50: AP at IoU=0.50\n"
            "- mAP75: stricter localization accuracy\n"
            "- Precision: fraction of detections that are correct\n"
            "- Recall: fraction of ground-truth objects detected\n\n"
            "The input size is read from each NCNN model's metadata.yaml.\n"
            "GEN models are evaluated on the fixed 11,000-image GEN val split.\n"
            "SNOW models are evaluated on the fixed 100-image SNOW val split.\n"
            "Models with incompatible class definitions (e.g. COCO 80-class\n"
            "official YOLO26n) are reported as skipped rather than producing\n"
            "a misleading accuracy number.\n"
        ),
        encoding="utf-8",
    )

    bundle = (
        output
        / "NCNN_accuracy_results_bundle.zip"
    )

    with zipfile.ZipFile(
        bundle,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for file in [
            output
            / "ncnn_accuracy_results.csv",
            output
            / "ncnn_accuracy_ranked_by_map50_95.csv",
            output
            / "ncnn_accuracy_per_class.csv",
            summary_path,
            readme,
        ]:
            if file.is_file():
                archive.write(
                    file,
                    arcname=file.name,
                )

    print()
    print(
        "===== COMPLETE ====="
    )
    print(
        json.dumps(
            summary,
            indent=2,
        )
    )
    print()
    print(
        "Download bundle:"
    )
    print(bundle)


if __name__ == "__main__":
    main()
