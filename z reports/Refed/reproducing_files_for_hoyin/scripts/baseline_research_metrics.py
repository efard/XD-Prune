"""Research-output helpers for the frozen YOLO26n baseline launcher.

These functions do not choose checkpoints or alter training. They validate and
export evidence from an already selected checkpoint and fixed validation split.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from contextlib import redirect_stdout
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from ultralytics import YOLO


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}

REQUIRED_RESULTS_COLUMNS = {
    "epoch",
    "time",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def json_write(path: Path, value: Any) -> Path:
    """Write strict JSON atomically so interrupted exports do not look complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, indent=2, default=str, allow_nan=False) + "\n"
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(serialized, encoding="utf-8")
    temporary.replace(path)
    return path


def metric_value(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


def validate_training_csv(results_csv: Path, completed_epochs: int) -> dict[str, Any]:
    """Reject incomplete, non-finite or malformed training histories."""
    if completed_epochs < 1:
        raise ValueError(f"completed_epochs must be positive, got {completed_epochs}")
    if not results_csv.is_file():
        raise FileNotFoundError(f"Missing training history: {results_csv}")
    with results_csv.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise RuntimeError("Training history has no header")
        if any(not column for column in fieldnames):
            raise RuntimeError("Training history contains an empty header field")
        if len(fieldnames) != len(set(fieldnames)):
            raise RuntimeError("Training history contains duplicate header fields")
        missing = sorted(REQUIRED_RESULTS_COLUMNS - set(fieldnames))
        if missing:
            raise RuntimeError(f"Training history is missing required columns: {missing}")
        rows = list(reader)
    if len(rows) != completed_epochs:
        raise RuntimeError(
            f"Training history has {len(rows)} rows but trainer completed {completed_epochs} epochs"
        )
    if not rows:
        raise RuntimeError("Training history is empty")

    numeric_columns = list(fieldnames)
    epochs: list[int] = []
    cumulative_times: list[float] = []
    for row_number, row in enumerate(rows, start=1):
        if None in row:
            raise RuntimeError(f"Training history row {row_number} has more fields than the header")
        parsed: dict[str, float] = {}
        for column in numeric_columns:
            try:
                value = float(row[column])
            except (TypeError, ValueError) as exc:
                raise RuntimeError(f"Non-numeric {column!r} at results row {row_number}") from exc
            if not math.isfinite(value):
                raise RuntimeError(f"Non-finite {column!r} at results row {row_number}: {value}")
            parsed[column] = value
        if not parsed["epoch"].is_integer():
            raise RuntimeError(f"Non-integral epoch at results row {row_number}: {parsed['epoch']}")
        epochs.append(int(parsed["epoch"]))
        cumulative_times.append(parsed["time"])
        if parsed["time"] < 0:
            raise RuntimeError(f"Negative cumulative time at results row {row_number}")
        for column in ("train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"):
            if parsed[column] < 0:
                raise RuntimeError(f"Negative loss {column!r} at results row {row_number}")
        for column in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
            if not 0.0 <= parsed[column] <= 1.0:
                raise RuntimeError(f"Out-of-range metric {column!r} at results row {row_number}: {parsed[column]}")
        for column in fieldnames:
            if column.startswith("lr/") and parsed[column] < 0:
                raise RuntimeError(f"Negative learning rate {column!r} at results row {row_number}")

    expected_epochs = list(range(1, completed_epochs + 1))
    if epochs != expected_epochs:
        raise RuntimeError(f"Training epochs are not the expected consecutive sequence: {epochs}")
    if any(current < previous for previous, current in zip(cumulative_times, cumulative_times[1:])):
        raise RuntimeError("Training cumulative time decreases between epochs")

    metric_key = "metrics/mAP50-95(B)"
    best_row = max(rows, key=lambda row: float(row[metric_key]))
    final = rows[-1]
    return {
        "status": "PASS",
        "epoch_rows": len(rows),
        "first_epoch": int(float(rows[0]["epoch"])),
        "last_epoch": int(float(final["epoch"])),
        "best_epoch_from_results_csv": int(float(best_row["epoch"])),
        "best_training_validation_map50_95": float(best_row[metric_key]),
        "training_loop_cumulative_seconds": float(final["time"]),
        "final_train_losses": {
            "box": float(final["train/box_loss"]),
            "cls": float(final["train/cls_loss"]),
            "dfl": float(final["train/dfl_loss"]),
        },
        "final_validation_losses": {
            "box": float(final["val/box_loss"]),
            "cls": float(final["val/cls_loss"]),
            "dfl": float(final["val/dfl_loss"]),
        },
    }


def checkpoint_profile(checkpoint: Path, expected_classes: int, imgsz: int) -> dict[str, Any]:
    """Profile the saved, unfused checkpoint before any validation-time fusion."""
    from ultralytics.utils.torch_utils import get_flops

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    yolo = YOLO(str(checkpoint), task="detect")
    model = yolo.model.float().cpu().eval()
    if model.is_fused():
        raise RuntimeError("Expected a fresh unfused training checkpoint, but the loaded model is already fused")
    classes = len(yolo.names)
    if classes != expected_classes:
        raise RuntimeError(f"Checkpoint nc={classes}, expected {expected_classes}")
    state = model.state_dict()
    non_finite_by_tensor: dict[str, int] = {}
    checked_floating_values = 0
    for name, tensor in state.items():
        if tensor.is_floating_point() or tensor.is_complex():
            checked_floating_values += tensor.numel()
            count = int((~torch.isfinite(tensor)).sum().item())
            if count:
                non_finite_by_tensor[name] = count
    if non_finite_by_tensor:
        raise RuntimeError(f"Checkpoint contains non-finite state values: {non_finite_by_tensor}")

    gflops = metric_value(get_flops(model, imgsz=imgsz))
    if not math.isfinite(gflops) or gflops <= 0:
        raise RuntimeError(f"Ultralytics get_flops() did not produce a positive finite result: {gflops}")
    try:
        thop_version = version("ultralytics-thop")
    except PackageNotFoundError as exc:
        raise RuntimeError("ultralytics-thop is required for traceable GFLOPs profiling") from exc

    parameters = sum(parameter.numel() for parameter in model.parameters())
    profile = {
        "classes": classes,
        "profiled_graph": "unfused training/checkpoint graph",
        "model_is_fused": False,
        "unfused_parameters": parameters,
        "state_dict_items": len(state),
        "checked_floating_state_values": checked_floating_values,
        "non_finite_state_values": 0,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256(checkpoint),
        "strides": [int(value) for value in model.stride.tolist()],
        "ultralytics_get_flops_gflops_unfused": gflops,
        "derived_gmacs_unfused": gflops / 2.0,
        "flops_profiler": "ultralytics.utils.torch_utils.get_flops (THOP MAC count multiplied by 2)",
        "ultralytics_thop_version": thop_version,
    }
    del yolo, model, state
    return profile


def _normalise_names(names: dict[int, str] | list[str]) -> dict[int, str]:
    if isinstance(names, dict):
        normalised = {int(class_id): str(name) for class_id, name in names.items()}
    elif isinstance(names, list):
        normalised = {class_id: str(name) for class_id, name in enumerate(names)}
    else:
        raise TypeError(f"Unsupported class-name mapping: {type(names).__name__}")
    if list(sorted(normalised)) != list(range(len(normalised))):
        raise RuntimeError(f"Class IDs are not contiguous from zero: {sorted(normalised)}")
    if len(set(normalised.values())) != len(normalised):
        raise RuntimeError("Class names are not unique")
    return normalised


def per_class_records(metrics: Any, names: dict[int, str] | list[str]) -> list[dict[str, Any]]:
    names_by_id = _normalise_names(names)
    class_to_row = {int(class_id): row for row, class_id in enumerate(metrics.box.ap_class_index)}
    if len(class_to_row) != len(metrics.box.ap_class_index):
        raise RuntimeError("Duplicate class IDs in validation AP output")
    if any(class_id not in names_by_id for class_id in class_to_row):
        raise RuntimeError(f"Validation AP output has unknown classes: {sorted(class_to_row)}")
    target_instances = np.asarray(metrics.nt_per_class)
    target_images = np.asarray(metrics.nt_per_image)
    if target_instances.shape != (len(names_by_id),) or target_images.shape != (len(names_by_id),):
        raise RuntimeError(
            f"Unexpected support shapes: instances={target_instances.shape}, images={target_images.shape}"
        )
    if np.any(target_instances < 0) or np.any(target_images < 0):
        raise RuntimeError("Validation support contains negative counts")
    all_ap_matrix = np.asarray(metrics.box.all_ap)
    if all_ap_matrix.ndim != 2 or all_ap_matrix.shape[1] != 10:
        raise RuntimeError(f"Expected AP values at 10 IoU thresholds, got {all_ap_matrix.shape}")
    records: list[dict[str, Any]] = []
    for class_id in range(len(names_by_id)):
        row = class_to_row.get(class_id)
        class_name = names_by_id[class_id]
        if row is None:
            if int(target_instances[class_id]) != 0:
                raise RuntimeError(f"Class {class_id} has targets but no AP row")
            precision = recall = f1 = ap50_95 = ap50 = ap75 = None
        else:
            all_ap = all_ap_matrix[row]
            precision = metric_value(metrics.box.p[row])
            recall = metric_value(metrics.box.r[row])
            f1 = metric_value(metrics.box.f1[row])
            ap50_95 = metric_value(all_ap.mean())
            ap50 = metric_value(all_ap[0])
            ap75 = metric_value(all_ap[5])
            values = {"precision": precision, "recall": recall, "f1": f1, "ap50_95": ap50_95, "ap50": ap50, "ap75": ap75}
            if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values.values()):
                raise RuntimeError(f"Invalid per-class metrics for class {class_id}: {values}")
        records.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "validation_images_with_class": int(target_images[class_id]),
                "validation_instances": int(target_instances[class_id]),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "ap50_95": ap50_95,
                "ap50": ap50,
                "ap75": ap75,
            }
        )
    return records


def write_per_class_outputs(metrics: Any, names: dict[int, str] | list[str], output_dir: Path) -> tuple[Path, Path]:
    records = per_class_records(metrics, names)
    if not records:
        raise RuntimeError("Cannot export per-class metrics without classes")
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = json_write(output_dir / "per_class_metrics.json", records)
    csv_path = output_dir / "per_class_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return json_path, csv_path


def _image_id(path: Path) -> int | str:
    return int(path.stem) if path.stem.isnumeric() else path.stem


def create_coco_ground_truth(dataset_yaml: Path, split: str, output_dir: Path) -> tuple[Path, Path]:
    """Create a traceable COCO-format copy of the frozen YOLO validation labels."""
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"Missing dataset YAML: {dataset_yaml}")
    dataset = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict):
        raise RuntimeError(f"Dataset YAML is not a mapping: {dataset_yaml}")
    names = _normalise_names(dataset["names"])
    split_value = dataset.get(split)
    if not isinstance(split_value, str):
        raise RuntimeError(f"Dataset split {split!r} must be one directory path")
    configured_root = dataset.get("path")
    if configured_root is None:
        data_root = dataset_yaml.parent.resolve()
    else:
        configured_root_path = Path(str(configured_root))
        data_root = (
            configured_root_path.resolve()
            if configured_root_path.is_absolute()
            else (dataset_yaml.parent / configured_root_path).resolve()
        )
    split_path = Path(split_value)
    images_root = split_path.resolve() if split_path.is_absolute() else (data_root / split_path).resolve()
    if not images_root.is_dir():
        raise FileNotFoundError(f"Missing image split directory: {images_root}")
    try:
        images_relative = images_root.relative_to(data_root / "images")
    except ValueError as exc:
        raise RuntimeError(
            f"Expected split images below {data_root / 'images'}, got {images_root}"
        ) from exc
    labels_root = data_root / "labels" / images_relative
    images = sorted(path for path in images_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise RuntimeError(f"No images found for {split}: {images_root}")
    identifiers = [_image_id(path) for path in images]
    if len(identifiers) != len(set(identifiers)):
        raise RuntimeError("Image stems are not unique; raw prediction image IDs would collide")

    coco_images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    instance_counts = [0] * len(names)
    image_counts = [0] * len(names)
    size_counts = {"small": 0, "medium": 0, "large": 0}
    missing_label_files = 0
    empty_label_files = 0
    annotation_id = 1
    for image_path, image_id in zip(images, identifiers):
        relative = image_path.relative_to(images_root)
        label_path = (labels_root / relative).with_suffix(".txt")
        with Image.open(image_path) as image:
            width, height = image.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid image dimensions for {image_path}: {width}x{height}")
        coco_images.append(
            {
                "id": image_id,
                "file_name": relative.as_posix(),
                "width": width,
                "height": height,
            }
        )
        present: set[int] = set()
        if label_path.is_file():
            label_lines = label_path.read_text(encoding="utf-8-sig").splitlines()
            if not any(line.strip() for line in label_lines):
                empty_label_files += 1
        else:
            # A missing YOLO label is a valid background image. Record it explicitly.
            label_lines = []
            missing_label_files += 1
        for line_number, line in enumerate(label_lines, start=1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise RuntimeError(f"Expected five YOLO fields: {label_path}:{line_number}")
            try:
                class_value = float(fields[0])
                xc, yc, box_width, box_height = map(float, fields[1:])
            except ValueError as exc:
                raise RuntimeError(f"Non-numeric YOLO label: {label_path}:{line_number}") from exc
            if not class_value.is_integer():
                raise RuntimeError(f"Non-integral class ID: {label_path}:{line_number}")
            class_id = int(class_value)
            if class_id not in names:
                raise RuntimeError(f"Invalid class {class_id}: {label_path}:{line_number}")
            coordinates = (xc, yc, box_width, box_height)
            if any(not math.isfinite(value) for value in coordinates):
                raise RuntimeError(f"Non-finite box coordinates: {label_path}:{line_number}")
            if not 0.0 <= xc <= 1.0 or not 0.0 <= yc <= 1.0:
                raise RuntimeError(f"Out-of-range box centre: {label_path}:{line_number}")
            if not 0.0 < box_width <= 1.0 or not 0.0 < box_height <= 1.0:
                raise RuntimeError(f"Invalid normalized box size: {label_path}:{line_number}")
            pixel_width = box_width * width
            pixel_height = box_height * height
            x = xc * width - pixel_width / 2
            y = yc * height - pixel_height / 2
            area = pixel_width * pixel_height
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id + 1,
                    "bbox": [x, y, pixel_width, pixel_height],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            annotation_id += 1
            instance_counts[class_id] += 1
            present.add(class_id)
            if area < 32**2:
                size_counts["small"] += 1
            elif area < 96**2:
                size_counts["medium"] += 1
            else:
                size_counts["large"] += 1
        for class_id in present:
            image_counts[class_id] += 1

    ground_truth = {
        "info": {
            "description": "Generated from the frozen YOLO validation view for secondary COCO-style evaluation",
            "dataset_yaml": str(dataset_yaml),
            "dataset_yaml_sha256": sha256(dataset_yaml),
            "split": split,
        },
        "images": coco_images,
        "annotations": annotations,
        "categories": [
            {"id": class_id + 1, "name": names[class_id], "supercategory": "object"}
            for class_id in range(len(names))
        ],
    }
    ground_truth_path = json_write(output_dir / "validation_ground_truth_coco.json", ground_truth)
    support = {
        "split": split,
        "images": len(images),
        "instances": len(annotations),
        "missing_label_files_treated_as_background": missing_label_files,
        "empty_label_files": empty_label_files,
        "coco_area_thresholds_pixels_squared": {"small_max_exclusive": 1024, "medium_max_exclusive": 9216},
        "instances_by_size": size_counts,
        "per_class": [
            {
                "class_id": class_id,
                "class_name": names[class_id],
                "images_with_class": image_counts[class_id],
                "instances": instance_counts[class_id],
            }
            for class_id in range(len(names))
        ],
    }
    support_path = json_write(output_dir / "validation_support.json", support)
    return ground_truth_path, support_path


def run_coco_evaluation(ground_truth: Path, predictions: Path, output_dir: Path) -> Path:
    """Run secondary COCO-style area and recall metrics at maxDets=300."""
    if not ground_truth.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"Missing COCO input: ground_truth={ground_truth}, predictions={predictions}")
    ground_truth_data = json.loads(ground_truth.read_text(encoding="utf-8"))
    prediction_data = json.loads(predictions.read_text(encoding="utf-8"))
    if not isinstance(prediction_data, list):
        raise RuntimeError("COCO predictions must be a JSON list")
    valid_image_ids = {image["id"] for image in ground_truth_data.get("images", [])}
    categories = {category["id"]: category["name"] for category in ground_truth_data.get("categories", [])}
    if not valid_image_ids or not categories:
        raise RuntimeError("COCO ground truth has no images or categories")
    for index, prediction in enumerate(prediction_data):
        if not isinstance(prediction, dict):
            raise RuntimeError(f"Prediction record {index} is not a mapping")
        missing = {"image_id", "category_id", "bbox", "score"} - set(prediction)
        if missing:
            raise RuntimeError(f"Prediction record {index} is missing fields: {sorted(missing)}")
        if prediction["image_id"] not in valid_image_ids:
            raise RuntimeError(f"Prediction record {index} has unknown image_id={prediction['image_id']!r}")
        if prediction["category_id"] not in categories:
            raise RuntimeError(f"Prediction record {index} has unknown category_id={prediction['category_id']!r}")
        bbox = prediction["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise RuntimeError(f"Prediction record {index} has an invalid bbox")
        numeric = [*bbox, prediction["score"]]
        if any(not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in numeric):
            raise RuntimeError(f"Prediction record {index} contains non-finite numeric values")
        if float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
            raise RuntimeError(f"Prediction record {index} has a non-positive box size")
        if not 0.0 <= float(prediction["score"]) <= 1.0:
            raise RuntimeError(f"Prediction record {index} has an invalid confidence score")
    if not prediction_data:
        result = {
            "status": "NO_PREDICTIONS",
            "note": "No predictions were available for secondary COCO-style evaluation.",
            "predictions": str(predictions),
            "predictions_sha256": sha256(predictions),
            "prediction_records": 0,
            "ground_truth": str(ground_truth),
            "ground_truth_sha256": sha256(ground_truth),
        }
        return json_write(output_dir / "coco_style_metrics.json", result)

    log = io.StringIO()
    with redirect_stdout(log):
        gt = COCO(str(ground_truth))
        dt = gt.loadRes(prediction_data)
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.maxDets = [1, 10, 300]
        evaluator.evaluate()
        evaluator.accumulate()

    precision = evaluator.eval["precision"]  # IoU, recall, category, area, maxDet
    recall = evaluator.eval["recall"]  # IoU, category, area, maxDet
    area_indices = {label: evaluator.params.areaRngLbl.index(label) for label in ("all", "small", "medium", "large")}
    area_all = area_indices["all"]
    max_det_index = len(evaluator.params.maxDets) - 1
    iou50 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.50))[0])
    iou75 = int(np.flatnonzero(np.isclose(evaluator.params.iouThrs, 0.75))[0])

    def valid_mean(values: np.ndarray) -> float | None:
        valid = np.asarray(values, dtype=np.float64)
        valid = valid[np.isfinite(valid) & (valid >= 0)]
        return float(valid.mean()) if valid.size else None

    def aggregate_ap(area: str, iou_index: int | None = None) -> float | None:
        values = precision[:, :, :, area_indices[area], max_det_index]
        if iou_index is not None:
            values = values[iou_index]
        return valid_mean(values)

    def aggregate_ar(area: str, max_detections_index: int) -> float | None:
        return valid_mean(recall[:, :, area_indices[area], max_detections_index])

    per_category = []
    for category_index, category_id in enumerate(evaluator.params.catIds):
        category_precision = precision[:, :, category_index, area_all, max_det_index]
        category_recall = recall[:, category_index, area_all, max_det_index]
        per_category.append(
            {
                "category_id": int(category_id),
                "class_id": int(category_id) - 1,
                "class_name": categories[category_id],
                "AP50_95": valid_mean(category_precision),
                "AP50": valid_mean(category_precision[iou50]),
                "AP75": valid_mean(category_precision[iou75]),
                "AR_maxDet_300": valid_mean(category_recall),
            }
        )
    result = {
        "status": "PASS",
        "definition": (
            "Secondary pycocotools bbox metrics computed from COCOeval evaluate/accumulate tensors using "
            "COCO area thresholds and maxDets=[1,10,300]"
        ),
        "summary_note": (
            "Metrics are computed directly from accumulated tensors because pycocotools COCOeval.summarize() "
            "hardcodes its first overall AP display to maxDets=100. All AP values here use maxDets=300."
        ),
        "primary_study_metric_remains": "Ultralytics standalone FP32 mAP50-95",
        "AP50_95": aggregate_ap("all"),
        "AP50": aggregate_ap("all", iou50),
        "AP75": aggregate_ap("all", iou75),
        "AP_small": aggregate_ap("small"),
        "AP_medium": aggregate_ap("medium"),
        "AP_large": aggregate_ap("large"),
        "AR_maxDet_1": aggregate_ar("all", 0),
        "AR_maxDet_10": aggregate_ar("all", 1),
        "AR_maxDet_300": aggregate_ar("all", 2),
        "AR_small": aggregate_ar("small", max_det_index),
        "AR_medium": aggregate_ar("medium", max_det_index),
        "AR_large": aggregate_ar("large", max_det_index),
        "per_class": per_category,
        "predictions": str(predictions),
        "predictions_sha256": sha256(predictions),
        "prediction_records": len(prediction_data),
        "ground_truth": str(ground_truth),
        "ground_truth_sha256": sha256(ground_truth),
        "pycocotools_evaluate_accumulate_log": log.getvalue(),
    }
    return json_write(output_dir / "coco_style_metrics.json", result)


def write_confusion_matrix_outputs(
    metrics: Any,
    names: dict[int, str] | list[str],
    output_dir: Path,
) -> dict[str, Path]:
    """Export raw and true-class-normalized Ultralytics confusion matrices."""
    names_by_id = _normalise_names(names)
    matrix = np.asarray(metrics.confusion_matrix.matrix, dtype=np.float64)
    expected_shape = (len(names_by_id) + 1, len(names_by_id) + 1)
    if matrix.shape != expected_shape:
        raise RuntimeError(f"Unexpected confusion-matrix shape {matrix.shape}, expected {expected_shape}")
    if not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise RuntimeError("Confusion matrix contains invalid values")
    rounded = np.rint(matrix)
    if not np.allclose(matrix, rounded, atol=1e-9, rtol=0):
        raise RuntimeError("Confusion matrix contains non-integral counts")
    counts = rounded.astype(np.int64)
    labels = [names_by_id[class_id] for class_id in range(len(names_by_id))] + ["background"]
    denominators = counts.sum(axis=0, keepdims=True)
    normalized = np.divide(
        counts.astype(np.float64),
        denominators,
        out=np.zeros_like(counts, dtype=np.float64),
        where=denominators != 0,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    def write_csv(path: Path, values: np.ndarray) -> Path:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["predicted\\true", *labels])
            for label, row in zip(labels, values.tolist()):
                writer.writerow([label, *row])
        return path

    common = {
        "orientation": "rows=predicted class; columns=true class",
        "labels": labels,
        "threshold_note": (
            "Generated by Ultralytics 8.4.21 detection validation. With its default val conf=0.001, "
            "the confusion-matrix implementation uses conf=0.25 and IoU=0.45."
        ),
    }
    paths = {
        "counts_json": json_write(
            output_dir / "confusion_matrix_counts.json",
            {**common, "normalization": "none", "matrix": counts.tolist()},
        ),
        "counts_csv": write_csv(output_dir / "confusion_matrix_counts.csv", counts),
        "normalized_json": json_write(
            output_dir / "confusion_matrix_normalized_by_true.json",
            {**common, "normalization": "each true-class column sums to one when nonempty", "matrix": normalized.tolist()},
        ),
        "normalized_csv": write_csv(output_dir / "confusion_matrix_normalized_by_true.csv", normalized),
    }
    return paths


def write_artifact_manifest(run_dir: Path, paths: list[Path], output: Path) -> Path:
    run_root = run_dir.resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Missing run directory: {run_root}")
    output_resolved = output.resolve()
    if output_resolved != run_root and run_root not in output_resolved.parents:
        raise RuntimeError(f"Artifact manifest must be inside the run directory: {output}")
    resolved_paths = [Path(path).resolve() for path in paths]
    if not resolved_paths:
        raise RuntimeError("Artifact manifest received no files")
    missing = [str(path) for path in resolved_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Artifact manifest inputs are missing: {missing}")
    if output_resolved in resolved_paths:
        raise RuntimeError("Artifact manifest cannot hash itself")
    unique = sorted(set(resolved_paths), key=lambda path: str(path).casefold())
    records = []
    for path in unique:
        try:
            relative = path.relative_to(run_root).as_posix()
        except ValueError:
            relative = str(path)
        records.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    return json_write(
        output,
        {
            "schema": "baseline-artifact-manifest-v1",
            "run_directory": str(run_root),
            "file_count": len(records),
            "files": records,
        },
    )
