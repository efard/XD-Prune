#!/usr/bin/env python3
"""
Train YOLO26n with INT8 quantization-aware training (QAT) at a fixed image size.

This script uses NVIDIA ModelOpt to insert fake INT8 quantizers before the
Ultralytics optimizer is created. During training, weights remain floating-point
parameters, while each forward pass simulates INT8 rounding/clipping so the model
can learn to compensate for quantization error.

The first implementation intentionally supports one GPU only. Multi-GPU QAT needs
explicit quantizer-state synchronization and should not be enabled implicitly.
"""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOCAL_RANK, LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class QATDetectionTrainer(DetectionTrainer):
    """Ultralytics detection trainer that inserts ModelOpt INT8 fake quantizers."""

    calibration_batches: int = 32

    def _build_train_pipeline(self) -> None:
        """
        Build dataloaders, calibrate INT8 ranges, then create the optimizer.

        Quantization must happen before optimizer creation. Otherwise, replacing
        ordinary layers with quantized wrappers after optimizer construction can
        leave the optimizer attached to an outdated parameter/module structure.
        """
        if self.world_size > 1:
            raise RuntimeError(
                "This QAT script supports one GPU only. "
                "Use --device 0 and request one GPU in SLURM."
            )

        batch_size = self.batch_size // max(self.world_size, 1)

        # Build the normal Ultralytics training and validation dataloaders first.
        # The training loader supplies representative real images for calibration.
        self.train_loader = self.get_dataloader(
            self.data["train"],
            batch_size=batch_size,
            rank=LOCAL_RANK,
            mode="train",
        )
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task in {"obb", "semantic"} else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )

        if self.calibration_batches < 1:
            raise ValueError("--calib-batches must be at least 1.")

        LOGGER.info(
            f"Preparing ModelOpt INT8 QAT with {self.calibration_batches} "
            "representative calibration batches."
        )

        # Calibration collects activation ranges from real preprocessed images.
        # eval() avoids updating BatchNorm statistics during this short range pass.
        self.model.eval()
        calibrated_batches = 0

        def calibration_loop(model: torch.nn.Module) -> None:
            """Run representative images through the model to determine INT8 scales."""
            nonlocal calibrated_batches

            with torch.no_grad():
                for batch in self.train_loader:
                    batch = self.preprocess_batch(batch)
                    model(batch["img"])
                    calibrated_batches += 1

                    if calibrated_batches >= self.calibration_batches:
                        break

        # INT8_DEFAULT_CFG is ModelOpt's CNN-oriented W8A8 configuration:
        # per-channel weight quantization and per-tensor activation quantization.
        self.model = mtq.quantize(
            self.model,
            deepcopy(mtq.INT8_DEFAULT_CFG),
            forward_loop=calibration_loop,
        )

        if calibrated_batches < self.calibration_batches:
            raise RuntimeError(
                f"Calibration requested {self.calibration_batches} batches, "
                f"but only {calibrated_batches} were available."
            )

        # QAT now proceeds through the ordinary Ultralytics training loop.
        # Fake-quantization modules remain active during each forward pass.
        self.model.train()
        mtq.print_quant_summary(self.model)

        # Reproduce Ultralytics' optimizer/scheduler construction after QAT modules
        # have been inserted, so all trainable parameters are tracked correctly.
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        weight_decay = (
            self.args.weight_decay
            * self.batch_size
            * self.accumulate
            / self.args.nbs
        )
        iterations = (
            math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs))
            * self.epochs
        )
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self._setup_scheduler()

    def save_model(self) -> bool:
        """Save normal Ultralytics checkpoints plus restorable ModelOpt QAT checkpoints."""
        saved = super().save_model()

        # Ultralytics validates and scores the EMA model, so the same EMA state is
        # saved for hardware/software handoff rather than the noisier live weights.
        model_to_save = self.ema.ema if self.ema is not None else self.model
        model_to_save = unwrap_model(model_to_save)

        # Preserve the best-scoring QAT state as soon as it is observed. ModelOpt's
        # save format stores both weights and architecture/quantizer modifications.
        if self.best_fitness == self.fitness:
            mto.save(model_to_save, str(self.wdir / "qat_int8_modelopt_best.pth"))

        # Preserve the final state separately for reproducibility and comparisons.
        if self.epoch + 1 >= self.epochs:
            mto.save(model_to_save, str(self.wdir / "qat_int8_modelopt_last.pth"))

        return saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Single-GPU YOLO26n INT8 QAT training with NVIDIA ModelOpt."
    )
    parser.add_argument("--model", required=True, help="FP32 baseline best.pt checkpoint.")
    parser.add_argument("--data", required=True, help="Ultralytics dataset YAML file.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--calib-batches", type=int, default=32)
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--lrf", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--project", default="runs/qat")
    parser.add_argument("--name", default="yolo26n_qat_int8_640_e100")
    parser.add_argument(
        "--cache",
        choices=("false", "ram", "disk"),
        default="false",
        help="Dataset caching mode. 'false' avoids extra storage use.",
    )
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    data_path = Path(args.data).expanduser().resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {model_path}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data_path}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this QAT run.")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1.")
    if args.batch < 1:
        raise ValueError("--batch must be a fixed positive integer.")

    QATDetectionTrainer.calibration_batches = args.calib_batches

    cache_value: bool | str = False if args.cache == "false" else args.cache

    trainer = QATDetectionTrainer(
        overrides={
            "model": str(model_path),
            "data": str(data_path),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "device": args.device,
            "optimizer": args.optimizer,
            "lr0": args.lr0,
            "lrf": args.lrf,
            "seed": args.seed,
            "deterministic": True,
            "project": args.project,
            "name": args.name,
            "cache": cache_value,
            "amp": False,
            "compile": False,
            "patience": 0,
            "save": True,
            "save_period": 10,
            "val": True,
            "plots": True,
            "exist_ok": False,
            "resume": False,
        }
    )
    trainer.train()

    # Ultralytics checkpoints and ModelOpt best/last checkpoints were saved during
    # training. Write a compact metadata record for the later hardware handoff.
    modelopt_best_checkpoint = trainer.wdir / "qat_int8_modelopt_best.pth"
    modelopt_last_checkpoint = trainer.wdir / "qat_int8_modelopt_last.pth"

    metadata_path = trainer.save_dir / "qat_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_fp32_checkpoint": str(model_path),
                "dataset_yaml": str(data_path),
                "image_size": args.imgsz,
                "epochs": args.epochs,
                "batch_size": args.batch,
                "calibration_batches": args.calib_batches,
                "quantization": "ModelOpt INT8_DEFAULT_CFG (W8A8 fake quantization)",
                "amp": False,
                "modelopt_best_checkpoint": str(modelopt_best_checkpoint),
                "modelopt_last_checkpoint": str(modelopt_last_checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(f"Best ModelOpt QAT checkpoint: {modelopt_best_checkpoint}")
    LOGGER.info(f"Last ModelOpt QAT checkpoint: {modelopt_last_checkpoint}")
    LOGGER.info(f"QAT metadata saved to: {metadata_path}")
