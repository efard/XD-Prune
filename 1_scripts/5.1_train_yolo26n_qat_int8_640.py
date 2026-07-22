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
        """
        Save final QAT checkpoints as .pt files using ModelOpt format.

        The .pt extension meets the project handoff requirement, while ModelOpt
        stores architecture modifications and tensor weights without pickling the
        runtime-generated QuantConv2d classes. These files must be restored with
        mto.restore(); they are not standard Ultralytics full-model checkpoints.

        ModelOpt creates dynamic quantized module classes such as QuantConv2d.
        Ultralytics' normal checkpoint writer serializes the complete EMA model
        object, which Python pickle cannot resolve for those runtime-generated
        classes. ModelOpt's checkpoint format stores the ModelOpt state together
        with tensor weights and is therefore the correct format for this model.
        """
        self.wdir.mkdir(parents=True, exist_ok=True)

        # Ultralytics validates using EMA weights. Saving the EMA model keeps the
        # checkpoint aligned with the mAP values reported at the end of each epoch.
        model_to_save = self.ema.ema if self.ema is not None else self.model
        model_to_save = unwrap_model(model_to_save)

        last_path = self.wdir / "last.pt"
        best_path = self.wdir / "best.pt"

        # Save the latest restorable QAT model after every epoch. The model is small
        # enough that the added checkpoint write is preferable to losing a long run.
        mto.save(model_to_save, str(last_path))

        # validate() updates best_fitness before save_model() is called, so equality
        # means that this epoch produced the best validation fitness seen so far.
        if self.best_fitness == self.fitness:
            mto.save(model_to_save, str(best_path))
            LOGGER.info(f"Saved best ModelOpt QAT checkpoint: {best_path}")

        # Preserve explicit periodic snapshots when --save-period is enabled.
        if self.save_period > 0 and (self.epoch + 1) % self.save_period == 0:
            epoch_path = self.wdir / f"epoch{self.epoch + 1}.pt"
            mto.save(model_to_save, str(epoch_path))
            LOGGER.info(f"Saved periodic ModelOpt QAT checkpoint: {epoch_path}")

        LOGGER.info(f"Saved latest ModelOpt QAT checkpoint: {last_path}")
        return True

    def final_eval(self) -> None:
        """
        Skip Ultralytics' best.pt reload because QAT checkpoints use ModelOpt format.

        Validation already runs at the end of every epoch, including the final one.
        The best epoch remains recorded in results.csv and in
        best.pt. A separate ModelOpt-aware validation/export
        script should restore that checkpoint for the final hardware handoff.
        """
        LOGGER.info(
            "Skipping Ultralytics best.pt final reload. "
            "Use weights/best.pt for ModelOpt-aware final validation."
        )


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

    # ModelOpt best/last checkpoints were saved during training. Standard
    # Ultralytics best.pt/last.pt files are intentionally not created because their
    # full-model pickle format is incompatible with ModelOpt's dynamic QAT classes.
    modelopt_best_checkpoint = trainer.wdir / "best.pt"
    modelopt_last_checkpoint = trainer.wdir / "last.pt"

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
                "final_qat_best_pt": str(modelopt_best_checkpoint),
                "final_qat_last_pt": str(modelopt_last_checkpoint),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    LOGGER.info(f"Final QAT INT8 .pt model: {modelopt_best_checkpoint}")
    LOGGER.info(f"Last ModelOpt QAT checkpoint: {modelopt_last_checkpoint}")
    LOGGER.info(f"QAT metadata saved to: {metadata_path}")
