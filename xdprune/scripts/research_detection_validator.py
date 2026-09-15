"""Research-grade detection exports for the frozen YOLO26n baseline.

This module is intentionally separate from the baseline launcher and frozen
configuration files.  It extends the Ultralytics 8.4.21 detection validator
with two lossless-enough-for-recomputation artifacts:

``predictions_full_precision.json``
    Post-NMS detections in original-image pixels without the rounding applied
    by Ultralytics' default JSON exporter.

``per_image_metric_stats.npz``
    Compressed, non-pickled NumPy arrays retaining per-image metric inputs,
    including the exact boolean TP matrix used at the ten COCO IoU thresholds.

Intended standalone use::

    from ultralytics import YOLO
    from Pruning_Study.scripts.research_detection_validator import (
        ResearchDetectionValidator,
    )

    model = YOLO("path/to/best.pt", task="detect")
    metrics = model.val(
        validator=ResearchDetectionValidator,
        data="path/to/dataset.yaml",
        split="val",
        imgsz=640,
        batch=16,
        device=0,
        half=False,
        conf=0.001,
        iou=0.7,
        max_det=300,
        plots=True,
    )

Both artifacts are written inside ``metrics.save_dir``.  The validator is
single-process by design: distributed validation is rejected so that partial
per-rank exports cannot be mistaken for a complete evaluation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import __version__ as ULTRALYTICS_VERSION
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import ops


EXPECTED_ULTRALYTICS_VERSION = "8.4.21"
PREDICTIONS_FILENAME = "predictions_full_precision.json"
PER_IMAGE_STATS_FILENAME = "per_image_metric_stats.npz"
SCHEMA_VERSION = "research_detection_validator_v1"


class ResearchDetectionValidator(DetectionValidator):
    """Detection validator retaining per-image data needed for paired analysis.

    The implementation mirrors ``DetectionValidator.update_metrics`` in
    Ultralytics 8.4.21, with export capture added before aggregate metric stats
    are cleared.  Prediction offsets index all prediction-aligned arrays;
    target offsets index ``target_class``.  For image ``i``, the corresponding
    slices are ``prediction_offsets[i]:prediction_offsets[i + 1]`` and
    ``target_offsets[i]:target_offsets[i + 1]``.
    """

    def __init__(
        self,
        dataloader=None,
        save_dir=None,
        args=None,
        _callbacks=None,
        *,
        predictions_filename: str = PREDICTIONS_FILENAME,
        per_image_stats_filename: str = PER_IMAGE_STATS_FILENAME,
    ) -> None:
        if ULTRALYTICS_VERSION != EXPECTED_ULTRALYTICS_VERSION:
            raise RuntimeError(
                "ResearchDetectionValidator was validated against "
                f"Ultralytics {EXPECTED_ULTRALYTICS_VERSION}, but the active "
                f"version is {ULTRALYTICS_VERSION}. Audit source compatibility "
                "before using it for research results."
            )
        super().__init__(dataloader=dataloader, save_dir=save_dir, args=args, _callbacks=_callbacks)
        self.predictions_filename = self._safe_filename(predictions_filename, ".json")
        self.per_image_stats_filename = self._safe_filename(per_image_stats_filename, ".npz")
        self._reset_research_buffers()

    @staticmethod
    def _safe_filename(value: str, required_suffix: str) -> str:
        """Require an output basename, preventing writes outside ``save_dir``."""
        path = Path(value)
        if path.name != value or path.suffix.lower() != required_suffix:
            raise ValueError(f"Expected a {required_suffix} output basename, received: {value!r}")
        return value

    def _reset_research_buffers(self) -> None:
        self._image_ids: list[str] = []
        self._image_paths: list[str] = []
        self._original_shapes_hw: list[tuple[int, int]] = []
        self._seen_image_ids: set[str] = set()
        self._prediction_offsets: list[int] = [0]
        self._target_offsets: list[int] = [0]
        self._tp_iou_chunks: list[np.ndarray] = []
        self._confidence_chunks: list[np.ndarray] = []
        self._predicted_class_chunks: list[np.ndarray] = []
        self._target_class_chunks: list[np.ndarray] = []
        self._prediction_bbox_xywh_chunks: list[np.ndarray] = []
        self._artifacts_written = False

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize Ultralytics metrics and a fresh set of research buffers."""
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            if torch.distributed.get_world_size() > 1:
                raise RuntimeError(
                    "ResearchDetectionValidator requires single-process validation; "
                    "distributed per-image artifact gathering is intentionally disabled."
                )
        super().init_metrics(model)
        self._reset_research_buffers()

    @staticmethod
    def _image_stem(image_path: str | Path) -> str:
        """Return the stable string identifier retained in the NPZ artifact."""
        image_stem = Path(image_path).stem
        if not image_stem:
            raise ValueError(f"Unable to derive image stem from path: {image_path!r}")
        return image_stem

    @classmethod
    def _json_image_id(cls, image_path: str | Path) -> int | str:
        """Match Ultralytics/COCO image-ID semantics for JSON interoperability."""
        image_stem = cls._image_stem(image_path)
        return int(image_stem) if image_stem.isnumeric() else image_stem

    def _register_image(self, pbatch: dict[str, Any]) -> str:
        """Register one image and fail rather than emit ambiguous duplicate IDs."""
        image_path = Path(pbatch["im_file"]).resolve()
        image_id = self._image_stem(image_path)
        if image_id in self._seen_image_ids:
            raise ValueError(
                f"Duplicate image stem {image_id!r} encountered during validation. "
                "Image-stem IDs must be unique for paired research exports."
            )
        self._seen_image_ids.add(image_id)
        self._image_ids.append(image_id)
        self._image_paths.append(str(image_path))
        shape = tuple(int(x) for x in pbatch["ori_shape"])
        if len(shape) != 2:
            raise ValueError(f"Expected original image shape (height, width), received: {shape}")
        self._original_shapes_hw.append((shape[0], shape[1]))
        return image_id

    @staticmethod
    def _xyxy_to_top_left_xywh(boxes: torch.Tensor) -> torch.Tensor:
        """Convert XYXY boxes to top-left XYWH without numeric rounding."""
        xywh = ops.xyxy2xywh(boxes.detach().clone())
        xywh[:, :2] -= xywh[:, 2:] / 2
        return xywh

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update standard metrics while retaining exact per-image inputs.

        This method deliberately mirrors Ultralytics 8.4.21 detection validation
        logic.  JSON serialization is performed for every non-empty prediction
        set even when the caller's ordinary ``save_json`` option is false.
        """
        for sample_index, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(sample_index, batch)
            predn = self._prepare_pred(pred)
            self._register_image(pbatch)

            target_cls_metric = pbatch["cls"].detach().cpu().numpy()
            no_pred = predn["cls"].shape[0] == 0
            confidence_metric = (
                np.zeros(0, dtype=np.float32)
                if no_pred
                else predn["conf"].detach().cpu().numpy()
            )
            predicted_cls_metric = (
                np.zeros(0, dtype=np.float32)
                if no_pred
                else predn["cls"].detach().cpu().numpy()
            )
            metric_stat = {
                **self._process_batch(predn, pbatch),
                "target_cls": target_cls_metric,
                "target_img": np.unique(target_cls_metric),
                "conf": confidence_metric,
                "pred_cls": predicted_cls_metric,
            }
            self.metrics.update_stats(metric_stat)

            tp_iou = np.ascontiguousarray(metric_stat["tp"], dtype=np.bool_)
            confidence = np.ascontiguousarray(confidence_metric.reshape(-1))
            predicted_class = np.ascontiguousarray(predicted_cls_metric.reshape(-1), dtype=np.int64)
            target_class = np.ascontiguousarray(target_cls_metric.reshape(-1), dtype=np.int64)

            if no_pred:
                prediction_bbox_xywh = np.empty((0, 4), dtype=confidence.dtype)
                predn_scaled = None
            else:
                predn_scaled = self.scale_preds(predn, pbatch)
                prediction_bbox_xywh = np.ascontiguousarray(
                    self._xyxy_to_top_left_xywh(predn_scaled["bboxes"]).detach().cpu().numpy()
                )

            prediction_count = confidence.shape[0]
            if tp_iou.shape != (prediction_count, self.niou):
                raise RuntimeError(
                    "Unexpected TP matrix shape: "
                    f"{tp_iou.shape}; expected {(prediction_count, self.niou)}"
                )
            if predicted_class.shape[0] != prediction_count or prediction_bbox_xywh.shape[0] != prediction_count:
                raise RuntimeError("Prediction-aligned research arrays have inconsistent lengths.")

            self._tp_iou_chunks.append(tp_iou)
            self._confidence_chunks.append(confidence)
            self._predicted_class_chunks.append(predicted_class)
            self._target_class_chunks.append(target_class)
            self._prediction_bbox_xywh_chunks.append(prediction_bbox_xywh)
            self._prediction_offsets.append(self._prediction_offsets[-1] + prediction_count)
            self._target_offsets.append(self._target_offsets[-1] + target_class.shape[0])

            # Preserve the stock confusion-matrix behavior exactly.
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(batch["img"][sample_index], pbatch["im_file"], self.save_dir)

            if no_pred:
                continue

            # Always retain full-precision post-NMS detections. Ordinary TXT
            # output remains governed by the caller's standard argument.
            assert predn_scaled is not None
            self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Append unrounded, original-image post-NMS detections to ``jdict``."""
        path = Path(pbatch["im_file"])
        image_id = self._json_image_id(path)
        boxes = self._xyxy_to_top_left_xywh(predn["bboxes"]).detach().cpu().tolist()
        scores = predn["conf"].detach().cpu().tolist()
        classes = predn["cls"].detach().cpu().tolist()
        for box, score, class_value in zip(boxes, scores, classes):
            class_index = int(class_value)
            if self.is_coco or self.is_lvis:
                category_id = int(self.class_map[class_index])
            else:
                category_id = class_index + 1
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": category_id,
                    "bbox": [float(value) for value in box],
                    "score": float(score),
                }
            )

    @staticmethod
    def _concatenate(chunks: list[np.ndarray], empty_shape: tuple[int, ...], dtype: np.dtype) -> np.ndarray:
        """Concatenate chunks while retaining a valid typed empty result."""
        if not chunks:
            return np.empty(empty_shape, dtype=dtype)
        return np.ascontiguousarray(np.concatenate(chunks, axis=0))

    @staticmethod
    def _unicode_array(values: list[str]) -> np.ndarray:
        """Create a non-object string array that loads with ``allow_pickle=False``."""
        return np.asarray(values, dtype=np.str_) if values else np.empty((0,), dtype="<U1")

    def _assembled_arrays(self) -> dict[str, np.ndarray]:
        """Assemble and validate the compressed per-image artifact payload."""
        confidence_dtype = self._confidence_chunks[0].dtype if self._confidence_chunks else np.dtype(np.float32)
        bbox_dtype = (
            self._prediction_bbox_xywh_chunks[0].dtype
            if self._prediction_bbox_xywh_chunks
            else confidence_dtype
        )
        tp_iou = self._concatenate(self._tp_iou_chunks, (0, self.niou), np.dtype(np.bool_))
        confidence = self._concatenate(self._confidence_chunks, (0,), confidence_dtype)
        predicted_class = self._concatenate(self._predicted_class_chunks, (0,), np.dtype(np.int64))
        target_class = self._concatenate(self._target_class_chunks, (0,), np.dtype(np.int64))
        prediction_bbox_xywh = self._concatenate(
            self._prediction_bbox_xywh_chunks,
            (0, 4),
            bbox_dtype,
        )
        prediction_offsets = np.asarray(self._prediction_offsets, dtype=np.int64)
        target_offsets = np.asarray(self._target_offsets, dtype=np.int64)
        image_ids = self._unicode_array(self._image_ids)
        image_paths = self._unicode_array(self._image_paths)
        original_shapes_hw = (
            np.asarray(self._original_shapes_hw, dtype=np.int64).reshape(-1, 2)
            if self._original_shapes_hw
            else np.empty((0, 2), dtype=np.int64)
        )

        image_count = len(self._image_ids)
        prediction_count = confidence.shape[0]
        target_count = target_class.shape[0]
        if len(set(self._image_ids)) != image_count:
            raise RuntimeError("Image IDs are not unique.")
        if prediction_offsets.shape != (image_count + 1,) or target_offsets.shape != (image_count + 1,):
            raise RuntimeError("Per-image offset arrays have invalid lengths.")
        if prediction_offsets[-1] != prediction_count or target_offsets[-1] != target_count:
            raise RuntimeError("Final per-image offsets do not match flattened array lengths.")
        if np.any(np.diff(prediction_offsets) < 0) or np.any(np.diff(target_offsets) < 0):
            raise RuntimeError("Per-image offsets must be monotonic.")
        if tp_iou.shape != (prediction_count, self.niou):
            raise RuntimeError("Flattened TP array does not align with predictions.")
        if predicted_class.shape[0] != prediction_count or prediction_bbox_xywh.shape != (prediction_count, 4):
            raise RuntimeError("Flattened prediction arrays do not align.")
        if len(self.jdict) != prediction_count:
            raise RuntimeError(
                f"JSON detection count {len(self.jdict)} does not match flattened predictions {prediction_count}."
            )

        return {
            "schema_version": np.asarray(SCHEMA_VERSION),
            "ultralytics_version": np.asarray(ULTRALYTICS_VERSION),
            "image_ids": image_ids,
            "image_paths": image_paths,
            "original_shapes_hw": original_shapes_hw,
            "prediction_offsets": prediction_offsets,
            "target_offsets": target_offsets,
            "tp_iou": tp_iou,
            "iou_thresholds": self.iouv.detach().cpu().numpy(),
            "confidence": confidence,
            "predicted_class": predicted_class,
            "target_class": target_class,
            "prediction_bbox_xywh": prediction_bbox_xywh,
            "prediction_bbox_format": np.asarray("top_left_xywh_original_pixels"),
            "custom_category_id_base": np.asarray(1, dtype=np.int64),
            "num_images": np.asarray(image_count, dtype=np.int64),
            "num_predictions": np.asarray(prediction_count, dtype=np.int64),
            "num_targets": np.asarray(target_count, dtype=np.int64),
        }

    @staticmethod
    def _atomic_json_write(destination: Path, payload: list[dict[str, Any]]) -> None:
        """Atomically write compact JSON, rejecting NaN and Infinity."""
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
                handle.write("\n")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_npz_write(destination: Path, payload: dict[str, np.ndarray]) -> None:
        """Atomically write a compressed NPZ without object arrays."""
        temporary = destination.with_name(destination.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_research_artifacts(self) -> tuple[Path, Path]:
        """Write complete JSON and NPZ artifacts, including valid empties."""
        if self._artifacts_written:
            return (
                Path(self.save_dir) / self.predictions_filename,
                Path(self.save_dir) / self.per_image_stats_filename,
            )
        save_dir = Path(self.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        predictions_path = save_dir / self.predictions_filename
        stats_path = save_dir / self.per_image_stats_filename
        arrays = self._assembled_arrays()
        self._atomic_json_write(predictions_path, self.jdict)
        self._atomic_npz_write(stats_path, arrays)
        self._artifacts_written = True
        return predictions_path, stats_path

    def get_stats(self) -> dict[str, Any]:
        """Persist per-image inputs immediately before aggregate stats clear."""
        self._write_research_artifacts()
        # The research JSON above is the lossless canonical prediction export.
        # Prevent BaseValidator.__call__ from writing a second, redundant
        # ``predictions.json`` after this method returns. At conf=0.001 that
        # duplicate exceeded 600 MB for the 11,000-image MIO validation split.
        # This toggle occurs only after prediction collection and metric
        # calculation, so it cannot affect accuracy values.
        self.args.save_json = False
        return super().get_stats()


__all__ = [
    "PREDICTIONS_FILENAME",
    "PER_IMAGE_STATS_FILENAME",
    "ResearchDetectionValidator",
]
