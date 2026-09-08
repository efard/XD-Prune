"""Export one verified Ultralytics checkpoint to a traceable NCNN folder."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_export_files(folder: Path) -> dict[str, Path]:
    files = {
        "param": folder / "model.ncnn.param",
        "bin": folder / "model.ncnn.bin",
        "metadata": folder / "metadata.yaml",
    }
    missing = [str(path) for path in files.values() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("NCNN export is incomplete: " + ", ".join(missing))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Source .pt checkpoint")
    parser.add_argument("--name", required=True, help="Portable board model folder name")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--adapter-module",
        help="Optional module that must be imported before loading a custom whole-layer checkpoint",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate paths and imports without loading or exporting the model",
    )
    args = parser.parse_args()

    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")
    if not args.name or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in args.name):
        parser.error("--name may contain only letters, numbers, dot, underscore, and hyphen")

    model_path = args.model.resolve()
    if not model_path.is_file():
        parser.error(f"checkpoint does not exist: {model_path}")
    output_root = args.output_root.resolve()
    final_dir = output_root / args.name
    if final_dir.exists() and not args.force:
        parser.error(f"output already exists: {final_dir}; use --force only for an intentional replacement")

    if args.adapter_module:
        importlib.import_module(args.adapter_module)

    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("PyTorch and Ultralytics are required for NCNN export") from error

    if args.preflight_only:
        print(f"Preflight passed: {model_path}")
        return 0

    model = YOLO(str(model_path))
    exported = Path(
        model.export(
            format="ncnn",
            imgsz=args.imgsz,
            device="cpu",
            half=args.precision == "fp16",
            int8=False,
            end2end=False,
        )
    ).resolve()
    source_files = require_export_files(exported)

    output_root.mkdir(parents=True, exist_ok=True)
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir()
    for path in source_files.values():
        shutil.copy2(path, final_dir / path.name)
    final_files = require_export_files(final_dir)

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": args.name,
        "source_checkpoint": str(model_path),
        "source_checkpoint_sha256": sha256(model_path),
        "input_size": args.imgsz,
        "precision": args.precision,
        "format": "ncnn",
        "export_end2end": False,
        "adapter_module": args.adapter_module,
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "ultralytics_version": ultralytics.__version__,
        "files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in final_files.values()
        },
    }
    (final_dir / "deployment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"NCNN deployment created: {final_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
