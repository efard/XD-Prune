"""Export a checkpoint to a traceable INT8 NCNN deployment folder.

The script performs post-training quantization (PTQ). It does not train or
fine-tune the model. Calibration images must come from a training split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{description} not found or empty: {path}")
    return path


def require_tool(folder: Path, name: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return require_file(folder / f"{name}{suffix}", f"NCNN tool {name}")


def run(command: list[str]) -> None:
    print("\n$ " + subprocess.list2cmdline(command), flush=True)
    subprocess.run(command, check=True)


def cpp_round_positive(value: float) -> int:
    return int(math.floor(value + 0.5))


def collect_images(folder: Path, limit: int) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Calibration directory not found: {folder}")
    images = sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No calibration images found under: {folder}")
    return images[:limit] if limit > 0 else images


def write_letterboxed_image(source: Path, destination: Path, imgsz: int) -> None:
    import cv2

    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"OpenCV failed to read: {source}")

    height, width = image.shape[:2]
    scale = min(imgsz / width, imgsz / height)
    resized_width = cpp_round_positive(width * scale)
    resized_height = cpp_round_positive(height * scale)
    horizontal_padding = imgsz - resized_width
    vertical_padding = imgsz - resized_height
    left = cpp_round_positive(horizontal_padding / 2.0 - 0.1)
    right = cpp_round_positive(horizontal_padding / 2.0 + 0.1)
    top = cpp_round_positive(vertical_padding / 2.0 - 0.1)
    bottom = cpp_round_positive(vertical_padding / 2.0 + 0.1)

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )
    letterboxed = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    if letterboxed.shape[:2] != (imgsz, imgsz):
        raise RuntimeError(
            f"Unexpected letterbox size for {source}: "
            f"{letterboxed.shape[1]}x{letterboxed.shape[0]}"
        )
    if not cv2.imwrite(str(destination), letterboxed):
        raise RuntimeError(f"Failed to write calibration image: {destination}")


def valid_name(value: str) -> bool:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
    return bool(value) and all(character in allowed for character in value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--ncnn-tools", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-calib", type=int, default=10000)
    parser.add_argument("--ncnn-release", default="20260526")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if not valid_name(args.name):
        parser.error("--name may contain only letters, numbers, dot, underscore, and hyphen")
    if args.imgsz <= 0 or args.threads <= 0 or args.max_calib < 0:
        parser.error("imgsz and threads must be positive; max-calib cannot be negative")

    model_path = require_file(args.model.resolve(), "Source checkpoint")
    calibration_root = args.calib.resolve()
    tools_root = args.ncnn_tools.resolve()
    output_root = args.output_root.resolve()
    final_folder = output_root / args.name
    ncnn2table = require_tool(tools_root, "ncnn2table")
    ncnn2int8 = require_tool(tools_root, "ncnn2int8")
    calibration_images = collect_images(calibration_root, args.max_calib)

    try:
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("OpenCV, PyTorch, and Ultralytics are required") from error

    print(f"Checkpoint: {model_path}")
    print(f"Calibration root: {calibration_root}")
    print(f"Calibration images: {len(calibration_images)}")
    print(f"Image size: {args.imgsz}x{args.imgsz}")
    print(f"NCNN tools: {tools_root}")
    print(f"Output: {final_folder}")
    if args.preflight_only:
        print("Preflight passed; no model was exported or quantized.")
        return 0

    if final_folder.exists() and not args.force:
        parser.error(f"output already exists: {final_folder}; use --force intentionally")

    with tempfile.TemporaryDirectory(prefix="ncnn_int8_") as temporary:
        temporary_root = Path(temporary)
        temporary_checkpoint = temporary_root / model_path.name
        shutil.copy2(model_path, temporary_checkpoint)

        print("\n[1/4] Exporting FP32 NCNN at the requested input size...")
        exported_folder = Path(
            YOLO(str(temporary_checkpoint)).export(
                format="ncnn",
                imgsz=args.imgsz,
                batch=1,
                device="cpu",
                half=False,
                int8=False,
                end2end=False,
            )
        ).resolve()
        fp32_param = require_file(exported_folder / "model.ncnn.param", "FP32 NCNN param")
        fp32_bin = require_file(exported_folder / "model.ncnn.bin", "FP32 NCNN bin")
        metadata = require_file(exported_folder / "metadata.yaml", "NCNN metadata")

        print("\n[2/4] Reproducing board letterbox preprocessing for calibration...")
        prepared_folder = temporary_root / "calibration_320"
        prepared_folder.mkdir()
        prepared_paths: list[Path] = []
        for index, source in enumerate(calibration_images, start=1):
            destination = prepared_folder / f"{index:06d}.png"
            write_letterboxed_image(source, destination, args.imgsz)
            prepared_paths.append(destination)
            if index == 1 or index % 100 == 0 or index == len(calibration_images):
                print(f"Prepared {index}/{len(calibration_images)}", flush=True)

        temporary_list = temporary_root / "calibration_prepared.txt"
        temporary_list.write_text(
            "\n".join(str(path) for path in prepared_paths) + "\n",
            encoding="utf-8",
        )
        calibration_table = temporary_root / "model.int8.table"

        print("\n[3/4] Computing KL activation calibration scales...")
        run(
            [
                str(ncnn2table),
                str(fp32_param),
                str(fp32_bin),
                str(temporary_list),
                str(calibration_table),
                "mean=[0,0,0]",
                "norm=[0.003921568627,0.003921568627,0.003921568627]",
                f"shape=[{args.imgsz},{args.imgsz},3]",
                "pixel=RGB",
                f"thread={args.threads}",
                "method=kl",
            ]
        )
        require_file(calibration_table, "INT8 calibration table")

        int8_param = temporary_root / "model.ncnn.param"
        int8_bin = temporary_root / "model.ncnn.bin"
        print("\n[4/4] Converting FP32 NCNN weights to INT8...")
        run(
            [
                str(ncnn2int8),
                str(fp32_param),
                str(fp32_bin),
                str(int8_param),
                str(int8_bin),
                str(calibration_table),
            ]
        )
        require_file(int8_param, "INT8 NCNN param")
        require_file(int8_bin, "INT8 NCNN bin")

        staging = temporary_root / "final"
        staging.mkdir()
        for source in (int8_param, int8_bin, metadata, calibration_table):
            shutil.copy2(source, staging / source.name)
        relative_sources = [
            str(path.relative_to(calibration_root)) for path in calibration_images
        ]
        (staging / "calibration_sources.txt").write_text(
            "\n".join(relative_sources) + "\n", encoding="utf-8"
        )

        output_files = [
            staging / "model.ncnn.param",
            staging / "model.ncnn.bin",
            staging / "metadata.yaml",
            staging / "model.int8.table",
            staging / "calibration_sources.txt",
        ]
        manifest = {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": args.name,
            "source_checkpoint": str(model_path),
            "source_checkpoint_sha256": sha256(model_path),
            "format": "ncnn",
            "precision": "int8_post_training_quantization",
            "input_size": args.imgsz,
            "calibration_root": str(calibration_root),
            "calibration_image_count": len(calibration_images),
            "calibration_selection": "recursive deterministic sort, first N",
            "calibration_method": "kl",
            "preprocessing": "BGR read; aspect resize; centered 114 letterbox; RGB; 1/255",
            "ncnn_release": args.ncnn_release,
            "ncnn_tools": {
                ncnn2table.name: sha256(ncnn2table),
                ncnn2int8.name: sha256(ncnn2int8),
            },
            "python": sys.version,
            "platform": platform.platform(),
            "torch_version": torch.__version__,
            "ultralytics_version": ultralytics.__version__,
            "opencv_version": cv2.__version__,
            "files": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in output_files
            },
        }
        (staging / "deployment_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

        output_root.mkdir(parents=True, exist_ok=True)
        if final_folder.exists():
            shutil.rmtree(final_folder)
        shutil.copytree(staging, final_folder)

    print(f"\nINT8 NCNN deployment created: {final_folder}")
    print("Board loading and smoke inference remain required.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
