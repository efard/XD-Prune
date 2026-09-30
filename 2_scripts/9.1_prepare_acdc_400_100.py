#!/usr/bin/env python3
"""
1. use ACDC snow dataset if it has 400 / 100.
2. Otherwise, create a 400 / 100 split (seed 42).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

KNOWN_YAMLS = [
    Path(
        "/project/ko/afm176/datasets/1_data/reproduction_exact_v1/"
        "SNOW_ACDC_exact/dataset_SNOW_local.yaml"
    ),
    Path(
        "/project/ko/afm176/datasets/1_data/acdc_snow_yolo/"
        "acdc_snow_mio.yaml"
    ),
]


def resolve_entry(yaml_path: Path, config: dict[str, Any], split: str) -> Path:
    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    value = config[split]
    if isinstance(value, list):
        if len(value) != 1:
            raise RuntimeError(
                f"{yaml_path}: expected one directory for split={split}, got {value}"
            )
        value = value[0]

    path = Path(value)
    if not path.is_absolute():
        path = (root / path).resolve()
    return path


def list_images(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        item.resolve()
        for item in path.rglob("*")
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
    )


def inspect_yaml(yaml_path: Path) -> dict[str, Any] | None:
    if not yaml_path.is_file():
        return None

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "train" not in config or "val" not in config:
        return None

    train_dir = resolve_entry(yaml_path, config, "train")
    val_dir = resolve_entry(yaml_path, config, "val")
    train_images = list_images(train_dir)
    val_images = list_images(val_dir)

    return {
        "yaml": yaml_path.resolve(),
        "config": config,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "train_images": train_images,
        "val_images": val_images,
    }


def label_path_for_image(image: Path, image_root: Path, label_root: Path) -> Path:
    relative = image.relative_to(image_root)
    return label_root / relative.with_suffix(".txt")


def create_split(source: dict[str, Any], output_root: Path, seed: int) -> Path:
    yaml_path: Path = source["yaml"]
    config: dict[str, Any] = source["config"]

    train_dir: Path = source["train_dir"]
    val_dir: Path = source["val_dir"]

    root = Path(config.get("path", yaml_path.parent))
    if not root.is_absolute():
        root = (yaml_path.parent / root).resolve()

    # This experiment only uses an existing YOLO-format source. The common
    # layout is <root>/images/{train,val} and <root>/labels/{train,val}.
    train_label_root = root / "labels/train"
    val_label_root = root / "labels/val"
    if not train_label_root.is_dir() or not val_label_root.is_dir():
        raise RuntimeError(
            "Could not create a new split because the source is not in the "
            "expected YOLO images/{train,val}, labels/{train,val} layout."
        )

    samples: list[tuple[Path, Path | None]] = []
    for image, image_root, label_root in [
        *( (img, train_dir, train_label_root) for img in source["train_images"] ),
        *( (img, val_dir, val_label_root) for img in source["val_images"] ),
    ]:
        label = label_path_for_image(image, image_root, label_root)
        if not label.is_file():
            # Background image: YOLO permits no label file. Keep it and use
            # None via an empty placeholder generated in the new split.
            samples.append((image, None))
        else:
            samples.append((image, label.resolve()))

    # De-duplicate by resolved image path.
    unique: dict[str, tuple[Path, Path | None]] = {}
    for image, label in samples:
        unique[str(image)] = (image, label)
    samples = list(unique.values())

    if len(samples) < 500:
        raise RuntimeError(
            f"Need at least 500 YOLO ACDC samples to create 400/100 split, "
            f"found {len(samples)}."
        )

    rng = random.Random(seed)
    chosen = rng.sample(samples, 500)
    train_samples = chosen[:400]
    val_samples = chosen[400:]

    if output_root.exists():
        shutil.rmtree(output_root)

    for split, split_samples in [("train", train_samples), ("val", val_samples)]:
        image_out = output_root / f"images/{split}"
        label_out = output_root / f"labels/{split}"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        manifest_lines: list[str] = []
        for index, (image, label) in enumerate(split_samples):
            # Prefix with index to avoid filename collisions from different
            # original subdirectories.
            destination_name = f"{index:04d}_{image.name}"
            destination_image = image_out / destination_name
            destination_image.symlink_to(image)

            destination_label = label_out / Path(destination_name).with_suffix(".txt")
            if label is not None:
                destination_label.symlink_to(label)
            else:
                destination_label.write_text("", encoding="utf-8")

            manifest_lines.append(str(image))

        (output_root / f"{split}_manifest.txt").write_text(
            "\n".join(manifest_lines) + "\n",
            encoding="utf-8",
        )

    output_yaml = output_root / "dataset_ACDC_400_100_seed42.yaml"
    output_config = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "nc": int(config["nc"]),
        "names": config["names"],
    }
    output_yaml.write_text(
        yaml.safe_dump(output_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_yaml


def write_existing_manifests(dataset: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        images = dataset[f"{split}_images"]
        (output_dir / f"{split}_manifest.txt").write_text(
            "\n".join(str(path) for path in images) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path(
            "/project/ko/afm176/datasets/1_data/"
            "ACDC_snow_400_100_seed42_pure_global_l1"
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    inspected = [item for item in (inspect_yaml(path) for path in KNOWN_YAMLS) if item]

    exact = next(
        (
            item
            for item in inspected
            if len(item["train_images"]) == 400
            and len(item["val_images"]) == 100
        ),
        None,
    )

    if exact is not None:
        selected_yaml = exact["yaml"]
        mode = "reused_existing_400_100"
        train_count = 400
        val_count = 100
        write_existing_manifests(exact, args.output_dir)
    else:
        source = next(
            (
                item
                for item in inspected
                if len(item["train_images"]) + len(item["val_images"]) >= 500
            ),
            None,
        )
        if source is None:
            raise RuntimeError(
                "No usable YOLO-format ACDC dataset was found. Checked:\n"
                + "\n".join(str(path) for path in KNOWN_YAMLS)
            )

        selected_yaml = create_split(source, args.split_root.resolve(), args.seed)
        mode = "created_seed42_400_100_split"
        train_count = 400
        val_count = 100
        generated = inspect_yaml(selected_yaml)
        if generated is None:
            raise RuntimeError("Generated dataset YAML could not be re-opened.")
        write_existing_manifests(generated, args.output_dir)

    report = {
        "status": "PASSED_ACDC_400_100_DATASET_PREPARATION",
        "mode": mode,
        "dataset_yaml": str(Path(selected_yaml).resolve()),
        "train_images": train_count,
        "val_images": val_count,
        "seed_if_resplit": args.seed,
    }
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selected_dataset_yaml.txt").write_text(
        str(Path(selected_yaml).resolve()) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
