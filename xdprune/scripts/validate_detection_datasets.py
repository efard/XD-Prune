"""Validate prepared YOLO dataset views for the GEN/SNOW pruning study.

Checks include image/label pairing, YOLO coordinate ranges, class IDs, split
leakage, exact duplicate placement, source hard-link identity, image readability,
manifest consistency, and a small Ultralytics dataloader smoke test.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VALIDATION_VERSION = 2


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def parse_label(path: Path, class_count: int) -> Counter:
    counts = Counter()
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return counts
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields, found {len(parts)}")
        class_id = int(parts[0])
        values = list(map(float, parts[1:]))
        if class_id < 0 or class_id >= class_count:
            raise ValueError(f"{path}:{line_number}: class ID {class_id} outside [0,{class_count - 1}]")
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"{path}:{line_number}: non-finite coordinate")
        xc, yc, width, height = values
        if not (0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            raise ValueError(f"{path}:{line_number}: invalid normalized box {values}")
        tolerance = 2e-7
        if xc - width / 2 < -tolerance or xc + width / 2 > 1 + tolerance:
            raise ValueError(f"{path}:{line_number}: horizontal box boundary outside image")
        if yc - height / 2 < -tolerance or yc + height / 2 > 1 + tolerance:
            raise ValueError(f"{path}:{line_number}: vertical box boundary outside image")
        counts[class_id] += 1
    return counts


def inspect_image(path: Path) -> tuple[str, int, int, str]:
    with Image.open(path) as image:
        width, height = image.size
        image_format = image.format or "unknown"
        image.verify()
    return str(path), width, height, image_format


def validate_hardlink(view_path: Path, source_path: Path) -> None:
    if not view_path.is_file():
        raise FileNotFoundError(view_path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not os.path.samefile(view_path, source_path):
        raise ValueError(f"Image view is not a hard link to its source: {view_path} -> {source_path}")


def inspect_image_and_hardlink(paths: tuple[Path, Path]) -> tuple[str, int, int, str]:
    """Verify one image and its source identity in a single worker task."""
    view_path, source_path = paths
    result = inspect_image(view_path)
    validate_hardlink(view_path, source_path)
    return result


def sha256_path(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return str(path), digest.hexdigest()


def validation_fingerprint(paths: Iterable[Path], settings: object) -> str:
    """Fingerprint the manifests/configuration used by a resumable validation."""
    digest = hashlib.sha256()
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for path in paths:
        digest.update(str(path.resolve()).encode("utf-8"))
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def write_checkpoint(
    checkpoint_path: Path,
    fingerprint: str,
    view_root: Path,
    expected_splits: dict[str, int],
    split_reports: dict[str, object],
) -> None:
    payload = {
        "validation_version": VALIDATION_VERSION,
        "fingerprint": fingerprint,
        "view_root": str(view_root),
        "status": "complete" if len(split_reports) == len(expected_splits) else "in_progress",
        "completed_splits": split_reports,
    }
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(checkpoint_path)


def validate_view(
    view_root: Path,
    source_root: Path,
    yaml_name: str,
    expected_splits: dict[str, int],
    test_has_labels: bool,
    workers: int,
    checkpoint_path: Path,
    resume: bool,
    image_check_limit: int,
    sample_seed: int,
) -> dict[str, object]:
    yaml_path = view_root / yaml_name
    manifest_path = view_root / "split_manifest.csv"
    summary_path = view_root / "preparation_summary.json"
    for required in (yaml_path, manifest_path, summary_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    dataset = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = dataset["names"]
    class_names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
    manifest = read_csv(manifest_path)
    manifest_by_split: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in manifest:
        manifest_by_split[row["split"]].append(row)

    fingerprint = validation_fingerprint(
        (yaml_path, manifest_path, summary_path),
        {
            "validation_version": VALIDATION_VERSION,
            "expected_splits": expected_splits,
            "test_has_labels": test_has_labels,
        },
    )
    resumed_reports: dict[str, object] = {}
    if resume and checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if (
            checkpoint.get("validation_version") == VALIDATION_VERSION
            and checkpoint.get("fingerprint") == fingerprint
            and checkpoint.get("view_root") == str(view_root)
        ):
            resumed_reports = checkpoint.get("completed_splits", {})
            print(f"Resuming {view_root.name} from completed splits: {sorted(resumed_reports)}")
        else:
            print(f"Ignoring stale validation checkpoint: {checkpoint_path}")

    split_reports: dict[str, object] = {}
    all_view_paths: set[str] = set()
    class_totals = Counter()

    for split, expected_count in expected_splits.items():
        image_root = view_root / "images" / split
        label_root = view_root / "labels" / split
        images = sorted(p for p in image_root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_root.rglob("*.txt")) if label_root.exists() else []
        expected_label_count = expected_count if (split != "test" or test_has_labels) else 0
        if len(images) != expected_count:
            raise ValueError(f"{view_root.name}/{split}: expected {expected_count} images, found {len(images)}")
        if len(labels) != expected_label_count:
            raise ValueError(
                f"{view_root.name}/{split}: expected {expected_label_count} labels, found {len(labels)}"
            )
        if len(manifest_by_split[split]) != expected_count:
            raise ValueError(
                f"{view_root.name}/{split}: manifest has {len(manifest_by_split[split])}, expected {expected_count}"
            )

        manifest_view_paths = {row["view_relative_path"].replace("\\", "/") for row in manifest_by_split[split]}
        disk_view_paths = {str(path.relative_to(view_root)).replace("\\", "/") for path in images}
        if manifest_view_paths != disk_view_paths:
            missing = sorted(manifest_view_paths - disk_view_paths)
            extra = sorted(disk_view_paths - manifest_view_paths)
            raise ValueError(
                f"{view_root.name}/{split}: manifest/disk mismatch; missing={missing[:3]}, extra={extra[:3]}"
            )
        overlap = all_view_paths & disk_view_paths
        if overlap:
            raise ValueError(f"Paths appear in more than one split: {sorted(overlap)[:3]}")
        all_view_paths.update(disk_view_paths)

        required_image_checks = (
            len(images)
            if image_check_limit <= 0
            else min(image_check_limit, len(images))
        )
        if split in resumed_reports:
            cached_report = dict(resumed_reports[split])
            # Version-2 checkpoints created before sampled validation always
            # checked every image, so their split image count is a safe fallback.
            cached_image_checks = int(
                cached_report.get("images_integrity_checked", cached_report["images"])
            )
            if cached_image_checks >= required_image_checks:
                cached_report.setdefault("images_integrity_checked", cached_image_checks)
                cached_report.setdefault("hardlinks_checked", cached_image_checks)
                cached_report.setdefault("image_check_policy", "exhaustive")
                split_reports[split] = cached_report
                for class_id, class_name in enumerate(class_names):
                    class_totals[class_id] += int(cached_report["objects_by_class"][class_name])
                print(f"Reused completed validation for {view_root.name}/{split}")
                continue

        object_counts = Counter()
        negative_images = 0
        if expected_label_count:
            for image_path in images:
                label_path = view_root / str(image_path.relative_to(view_root)).replace(
                    f"images{os.sep}", f"labels{os.sep}", 1
                )
                label_path = label_path.with_suffix(".txt")
                if not label_path.is_file():
                    raise FileNotFoundError(label_path)
                counts = parse_label(label_path, len(class_names))
                object_counts.update(counts)
                negative_images += int(sum(counts.values()) == 0)

        rows_by_view_path = {
            row["view_relative_path"].replace("\\", "/"): row
            for row in manifest_by_split[split]
        }
        image_source_pairs = [
            (
                image_path,
                source_root / rows_by_view_path[
                    str(image_path.relative_to(view_root)).replace("\\", "/")
                ]["source_relative_path"],
            )
            for image_path in images
        ]
        if required_image_checks < len(image_source_pairs):
            rng = random.Random(f"{sample_seed}:{view_root.name}:{split}")
            selected_indices = sorted(rng.sample(range(len(image_source_pairs)), required_image_checks))
            checked_pairs = [image_source_pairs[index] for index in selected_indices]
            image_check_policy = f"fixed-seed sample ({required_image_checks:,}/{len(images):,})"
        else:
            checked_pairs = image_source_pairs
            image_check_policy = "exhaustive"
        print(
            f"Checking image integrity and hardlink identity for "
            f"{len(checked_pairs):,}/{len(images):,} {view_root.name}/{split} images..."
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, _ in enumerate(
                pool.map(inspect_image_and_hardlink, checked_pairs, chunksize=128),
                start=1,
            ):
                if index % 10000 == 0 or index == len(checked_pairs):
                    print(
                        f"  validated {index:,}/{len(checked_pairs):,} checked "
                        f"{view_root.name}/{split} images"
                    )

        class_totals.update(object_counts)
        split_reports[split] = {
            "images": len(images),
            "labels": len(labels),
            "objects": sum(object_counts.values()),
            "negative_images": negative_images,
            "images_integrity_checked": len(checked_pairs),
            "hardlinks_checked": len(checked_pairs),
            "image_check_policy": image_check_policy,
            "objects_by_class": {
                class_names[class_id]: object_counts[class_id]
                for class_id in range(len(class_names))
            },
        }
        write_checkpoint(checkpoint_path, fingerprint, view_root, expected_splits, split_reports)

    report: dict[str, object] = {
        "dataset_id": json.loads(summary_path.read_text(encoding="utf-8"))["dataset_id"],
        "view_root": str(view_root),
        "yaml": str(yaml_path),
        "classes": class_names,
        "splits": split_reports,
        "total_objects_in_labelled_splits": sum(class_totals.values()),
        "hardlink_identity": "passed for every checked image; see per-split check policy",
        "image_readability": "passed for every checked image; see per-split check policy",
        "label_validation": "exhaustive; passed for every public label file",
        "manifest_and_split_validation": "exhaustive; passed",
    }

    if view_root.name.startswith("mio_"):
        sha_splits: dict[str, set[str]] = defaultdict(set)
        for row in manifest:
            sha_splits[row["sha256"]].add(row["split"])
        crossing = {digest: splits for digest, splits in sha_splits.items() if len(splits) > 1}
        if crossing:
            example = next(iter(crossing.items()))
            raise ValueError(f"Exact MIO duplicate crosses splits: {example}")
        val_test_duplicates = [
            row for row in manifest
            if row["split"] in {"val", "test"} and int(row["exact_hash_group_size"]) != 1
        ]
        if val_test_duplicates:
            raise ValueError("MIO validation/test includes a non-singleton exact-hash group.")
        report["exact_duplicate_split_leakage"] = "none"
        report["validation_and_test_hashes"] = "singleton only"
    else:
        print(f"Hashing {len(manifest):,} {view_root.name} images for cross-split duplicate checking...")
        path_to_split = {
            str(view_root / row["view_relative_path"]): row["split"]
            for row in manifest
        }
        digest_splits: dict[str, set[str]] = defaultdict(set)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for path_string, digest in pool.map(
                sha256_path,
                [Path(path_string) for path_string in path_to_split],
                chunksize=32,
            ):
                digest_splits[digest].add(path_to_split[path_string])
        crossing = {digest: splits for digest, splits in digest_splits.items() if len(splits) > 1}
        if crossing:
            example = next(iter(crossing.items()))
            raise ValueError(f"Exact {view_root.name} duplicate crosses official splits: {example}")
        report["exact_duplicate_split_leakage"] = "none"
        report["unique_image_hashes"] = len(digest_splits)
    return report


def smoke_test(yaml_path: Path) -> dict[str, object]:
    """Load a tiny validation fraction through the installed Ultralytics pipeline."""
    from ultralytics.cfg import DEFAULT_CFG
    from ultralytics.data.build import build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(yaml_path), autodownload=False)
    cfg = deepcopy(DEFAULT_CFG)
    cfg.imgsz = 640
    cfg.cache = False
    cfg.single_cls = False
    cfg.classes = None
    cfg.fraction = 0.02
    cfg.rect = False
    dataset = build_yolo_dataset(cfg, data["val"], batch=2, data=data, mode="val", rect=False, stride=32)
    if len(dataset) < 2:
        raise ValueError(f"Ultralytics smoke dataset is unexpectedly small: {len(dataset)}")
    first = dataset[0]
    second = dataset[1]
    batch = dataset.collate_fn([first, second])
    image_shape = list(batch["img"].shape)
    if image_shape[0] != 2 or image_shape[1] != 3:
        raise ValueError(f"Unexpected smoke batch shape: {image_shape}")
    return {
        "status": "passed",
        "dataset_length_after_fraction": len(dataset),
        "batch_image_shape": image_shape,
        "batch_boxes": int(batch["bboxes"].shape[0]),
        "batch_classes": int(batch["cls"].shape[0]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mio-view",
        type=Path,
        default=PROJECT / "data_views" / "mio_full_s42_v1",
    )
    parser.add_argument(
        "--acdc-view",
        type=Path,
        default=PROJECT / "data_views" / "acdc_snow_official_v1",
    )
    parser.add_argument(
        "--mio-source",
        type=Path,
        default=PROJECT / "MIO-TCD-Localization" / "MIO-TCD-Localization",
    )
    parser.add_argument("--acdc-source", type=Path, default=PROJECT / "rgb_anon")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "results" / "dataset_prep" / "dataset_validation_report.json",
    )
    parser.add_argument("--workers", type=int, default=min(16, max(1, os.cpu_count() or 1)))
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed splits from matching validation checkpoints.",
    )
    parser.add_argument(
        "--mio-image-check-limit",
        type=int,
        default=2000,
        help="Image-integrity/hardlink sample per MIO split; 0 checks every image.",
    )
    parser.add_argument(
        "--acdc-image-check-limit",
        type=int,
        default=0,
        help="Image-integrity/hardlink sample per ACDC split; 0 checks every image.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_root = args.output.parent / "checkpoints"
    mio = validate_view(
        args.mio_view.resolve(),
        args.mio_source.resolve(),
        "mio_tcd_full.yaml",
        {"train": 88000, "val": 11000, "test": 11000},
        test_has_labels=True,
        workers=args.workers,
        checkpoint_path=checkpoint_root / f"{args.mio_view.name}_validation.json",
        resume=args.resume,
        image_check_limit=args.mio_image_check_limit,
        sample_seed=args.sample_seed,
    )
    acdc = validate_view(
        args.acdc_view.resolve(),
        args.acdc_source.resolve(),
        "acdc_snow.yaml",
        {"train": 400, "val": 100, "test": 500},
        test_has_labels=False,
        workers=args.workers,
        checkpoint_path=checkpoint_root / f"{args.acdc_view.name}_validation.json",
        resume=args.resume,
        image_check_limit=args.acdc_image_check_limit,
        sample_seed=args.sample_seed,
    )
    if not args.skip_smoke:
        print("Running Ultralytics MIO dataloader smoke test...")
        mio["dataloader_smoke"] = smoke_test(args.mio_view / "mio_tcd_full.yaml")
        print("Running Ultralytics ACDC dataloader smoke test...")
        acdc["dataloader_smoke"] = smoke_test(args.acdc_view / "acdc_snow.yaml")

    report = {"status": "passed", "mio": mio, "acdc_snow": acdc}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Validation report: {args.output.resolve()}")


if __name__ == "__main__":
    main()
