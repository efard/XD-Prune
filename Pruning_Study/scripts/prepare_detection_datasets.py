"""Prepare reproducible YOLO dataset views for the GEN/SNOW pruning study.

The downloaded MIO-TCD and ACDC sources are treated as immutable. This script
creates generated YOLO labels, split manifests, dataset YAML files, and NTFS
hard links to the original images. Hard links do not duplicate image bytes.

MIO protocol
------------
* Source: all 110,000 labelled training images.
* Output: 88,000 train / 11,000 validation / 11,000 internal test images.
* Exact duplicate images are forced into train. Validation and test therefore
  contain only singleton image hashes and cannot leak exact training images.
* Remaining singleton images are split with deterministic multilabel
  stratification (seed 42 by default).
* The official 27,743-image challenge test set remains label-free and is linked
  only for optional final inference/submission.

ACDC protocol
-------------
* Preserve the official snow train/val/test split (400/100/500 images).
* COCO ``iscrowd=1`` annotations are excluded, matching Ultralytics' standard
  COCO-to-YOLO conversion behaviour.
* Images with no ordinary annotations are retained with empty label files.
* Test images remain label-free because their ground truth is withheld.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable

import yaml
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]

MIO_CLASSES = [
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

ACDC_CLASSES = [
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motorcycle",
    "bicycle",
]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def normalize_mio_class(value: str) -> str:
    """Normalize the two spellings used for MIO's non-motorized class."""
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def stable_value(seed: int, *parts: str) -> int:
    payload = "|".join([str(seed), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def sha256_file(path: Path) -> tuple[str, str]:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return path.name, digest.hexdigest()


def hash_files(paths: list[Path], workers: int) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for name, digest in pool.map(sha256_file, paths, chunksize=256):
            hashes[name] = digest
    return hashes


def iterative_multilabel_split(
    labels_by_file: dict[str, set[int]],
    target_sizes: dict[str, int],
    seed: int,
) -> dict[str, str]:
    """Deterministically split multilabel samples while approximating label ratios.

    This is a compact iterative-stratification implementation. It repeatedly
    processes the rarest remaining label and assigns each associated sample to
    the split with the greatest remaining need for that label, then for the
    sample's other labels, and finally for total capacity.
    """
    if sum(target_sizes.values()) != len(labels_by_file):
        raise ValueError("Target sizes must equal the number of samples to split.")

    split_order = list(target_sizes)
    capacities = dict(target_sizes)
    total_samples = len(labels_by_file)
    all_labels = sorted({label for labels in labels_by_file.values() for label in labels})
    label_totals = Counter(label for labels in labels_by_file.values() for label in labels)
    desired = {
        split: {
            label: label_totals[label] * target_sizes[split] / total_samples
            for label in all_labels
        }
        for split in split_order
    }

    label_to_files: dict[int, set[str]] = {label: set() for label in all_labels}
    for file_name, labels in labels_by_file.items():
        for label in labels:
            label_to_files[label].add(file_name)

    remaining = set(labels_by_file)
    assignment: dict[str, str] = {}

    while remaining:
        remaining_counts = {
            label: len(files & remaining)
            for label, files in label_to_files.items()
            if files & remaining
        }
        if not remaining_counts:
            # This handles any images left with no valid annotations.
            for file_name in sorted(remaining, key=lambda x: stable_value(seed, x)):
                choices = [s for s in split_order if capacities[s] > 0]
                chosen = max(choices, key=lambda s: (capacities[s], -split_order.index(s)))
                assignment[file_name] = chosen
                capacities[chosen] -= 1
            remaining.clear()
            break

        rare_label = min(remaining_counts, key=lambda label: (remaining_counts[label], label))
        candidates = list(label_to_files[rare_label] & remaining)
        candidates.sort(
            key=lambda name: (
                -len(labels_by_file[name]),
                stable_value(seed, name),
            )
        )

        for file_name in candidates:
            if file_name not in remaining:
                continue
            labels = labels_by_file[file_name]
            choices = [s for s in split_order if capacities[s] > 0]
            if not choices:
                raise RuntimeError("No split capacity remains during stratification.")

            def score(split: str) -> tuple[float, float, float, int]:
                primary_need = desired[split].get(rare_label, 0.0)
                total_need = sum(max(0.0, desired[split].get(label, 0.0)) for label in labels)
                capacity_fraction = capacities[split] / max(1, target_sizes[split])
                tie = -stable_value(seed, file_name, split)
                return primary_need, total_need, capacity_fraction, tie

            chosen = max(choices, key=score)
            assignment[file_name] = chosen
            capacities[chosen] -= 1
            for label in labels:
                desired[chosen][label] = desired[chosen].get(label, 0.0) - 1.0
            remaining.remove(file_name)

    if any(capacities.values()):
        raise RuntimeError(f"Stratification ended with unfilled capacities: {capacities}")
    return assignment


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def hardlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    os.link(src, dst)


def yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> str:
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(
            f"Box ({x1}, {y1}, {x2}, {y2}) is outside image dimensions ({width}, {height})."
        )
    xc = ((x1 + x2) / 2.0) / width
    yc = ((y1 + y2) / 2.0) / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{class_id} {xc:.8f} {yc:.8f} {bw:.8f} {bh:.8f}"


def split_statistics(
    assignments: dict[str, str],
    annotations: dict[str, list[tuple[int, float, float, float, float]]],
    class_names: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        files = [name for name, assigned in assignments.items() if assigned == split]
        object_counts = Counter()
        image_counts = Counter()
        for name in files:
            labels = {ann[0] for ann in annotations[name]}
            for label in labels:
                image_counts[label] += 1
            for ann in annotations[name]:
                object_counts[ann[0]] += 1
        for class_id, class_name in enumerate(class_names):
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "images_with_class": image_counts[class_id],
                    "object_count": object_counts[class_id],
                    "split_images": len(files),
                }
            )
    return rows


def prepare_mio(args: argparse.Namespace) -> dict[str, object]:
    source = args.mio_root
    image_dir = source / "train"
    official_test_dir = source / "test"
    annotation_csv = source / "gt_train.csv"
    for required in (image_dir, official_test_dir, annotation_csv):
        if not required.exists():
            raise FileNotFoundError(required)

    class_to_id = {name: idx for idx, name in enumerate(MIO_CLASSES)}
    annotations: dict[str, list[tuple[int, float, float, float, float]]] = defaultdict(list)
    invalid_rows: list[dict[str, object]] = []
    raw_annotation_signatures: dict[str, list[tuple[str, float, float, float, float]]] = defaultdict(list)

    with annotation_csv.open(newline="", encoding="utf-8") as f:
        for row_number, row in enumerate(csv.reader(f), start=1):
            if len(row) != 6:
                raise ValueError(f"MIO CSV row {row_number} has {len(row)} fields, expected 6.")
            image_id, raw_class, x1, y1, x2, y2 = row
            file_name = f"{int(image_id):08d}.jpg"
            class_name = normalize_mio_class(raw_class)
            if class_name not in class_to_id:
                raise ValueError(f"Unknown MIO class at row {row_number}: {raw_class!r}")
            coords = tuple(map(float, (x1, y1, x2, y2)))
            raw_annotation_signatures[file_name].append((class_name, *coords))
            if coords[2] <= coords[0] or coords[3] <= coords[1]:
                invalid_rows.append(
                    {
                        "row_number": row_number,
                        "file_name": file_name,
                        "class_name": class_name,
                        "x1": coords[0],
                        "y1": coords[1],
                        "x2": coords[2],
                        "y2": coords[3],
                        "reason": "non-positive width or height",
                    }
                )
                continue
            annotations[file_name].append((class_to_id[class_name], *coords))

    image_paths = sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    official_test_paths = sorted(
        p for p in official_test_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    actual_names = {p.name for p in image_paths}
    annotated_names = set(raw_annotation_signatures)
    if actual_names != annotated_names:
        missing = sorted(annotated_names - actual_names)
        unannotated = sorted(actual_names - annotated_names)
        raise ValueError(
            f"MIO image/annotation mismatch: missing={len(missing)}, unannotated={len(unannotated)}; "
            f"examples missing={missing[:3]}, unannotated={unannotated[:3]}"
        )
    if len(image_paths) != args.mio_total:
        raise ValueError(f"Expected {args.mio_total} MIO images, found {len(image_paths)}.")
    if args.mio_train + args.mio_val + args.mio_test != args.mio_total:
        raise ValueError("MIO train/val/test counts must sum to the total.")

    print(f"Hashing {len(image_paths):,} MIO images for exact-duplicate grouping...")
    hashes = hash_files(image_paths, args.workers)
    hash_groups: dict[str, list[str]] = defaultdict(list)
    for file_name, digest in hashes.items():
        hash_groups[digest].append(file_name)
    for names in hash_groups.values():
        names.sort()

    duplicate_groups = {digest: names for digest, names in hash_groups.items() if len(names) > 1}
    duplicate_names = {name for names in duplicate_groups.values() for name in names}
    singleton_names = sorted(actual_names - duplicate_names)
    forced_train = len(duplicate_names)
    singleton_targets = {
        "train": args.mio_train - forced_train,
        "val": args.mio_val,
        "test": args.mio_test,
    }
    if singleton_targets["train"] < 0 or sum(singleton_targets.values()) != len(singleton_names):
        raise ValueError(
            f"Duplicate-aware target mismatch: duplicate files={forced_train}, "
            f"singletons={len(singleton_names)}, singleton targets={singleton_targets}"
        )

    singleton_labels = {
        name: {ann[0] for ann in annotations[name]}
        for name in singleton_names
    }
    singleton_assignment = iterative_multilabel_split(singleton_labels, singleton_targets, args.seed)
    assignments = {name: "train" for name in duplicate_names}
    assignments.update(singleton_assignment)

    split_counts = Counter(assignments.values())
    expected_counts = {"train": args.mio_train, "val": args.mio_val, "test": args.mio_test}
    if dict(split_counts) != expected_counts:
        raise RuntimeError(f"Unexpected MIO split counts: {split_counts}, expected {expected_counts}")
    for digest, names in duplicate_groups.items():
        if {assignments[name] for name in names} != {"train"}:
            raise RuntimeError(f"Duplicate hash group crossed splits: {digest}")

    inconsistent_duplicate_groups = 0
    duplicate_manifest_rows: list[dict[str, object]] = []
    for group_index, (digest, names) in enumerate(sorted(duplicate_groups.items()), start=1):
        signatures = {
            tuple(sorted(raw_annotation_signatures[name]))
            for name in names
        }
        consistent = len(signatures) == 1
        inconsistent_duplicate_groups += int(not consistent)
        group_id = f"dup_{group_index:05d}"
        for name in names:
            duplicate_manifest_rows.append(
                {
                    "duplicate_group_id": group_id,
                    "sha256": digest,
                    "group_size": len(names),
                    "file_name": name,
                    "split": assignments[name],
                    "annotation_count": len(raw_annotation_signatures[name]),
                    "annotations_identical_within_group": consistent,
                }
            )

    summary = {
        "dataset_id": f"mio_full_s{args.seed}_v1",
        "source": str(source),
        "seed": args.seed,
        "classes": MIO_CLASSES,
        "source_labelled_images": len(image_paths),
        "source_official_test_images": len(official_test_paths),
        "valid_boxes": sum(len(v) for v in annotations.values()),
        "excluded_invalid_boxes": len(invalid_rows),
        "unique_image_hashes": len(hash_groups),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_files_forced_to_train": forced_train,
        "duplicate_groups_with_nonidentical_annotations": inconsistent_duplicate_groups,
        "split_counts": expected_counts,
        "split_rule": "all exact duplicate groups in train; singleton images multilabel-stratified",
    }

    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return summary

    final_root = args.output_root / summary["dataset_id"]
    staging_root = args.output_root / f".{summary['dataset_id']}.building"
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing MIO output: {final_root} or {staging_root}")
    staging_root.mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    image_by_name = {p.name: p for p in image_paths}
    hash_group_sizes = {digest: len(names) for digest, names in hash_groups.items()}

    print("Writing MIO hard links and YOLO labels...")
    for index, file_name in enumerate(sorted(assignments), start=1):
        split = assignments[file_name]
        src = image_by_name[file_name]
        dst_image = staging_root / "images" / split / file_name
        dst_label = staging_root / "labels" / split / f"{Path(file_name).stem}.txt"
        hardlink(src, dst_image)
        with Image.open(src) as image:
            width, height = image.size
        label_lines = [
            yolo_line(class_id, x1, y1, x2, y2, width, height)
            for class_id, x1, y1, x2, y2 in annotations[file_name]
        ]
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        dst_label.write_text(("\n".join(label_lines) + "\n") if label_lines else "", encoding="utf-8")
        digest = hashes[file_name]
        manifest_rows.append(
            {
                "dataset_id": summary["dataset_id"],
                "split": split,
                "file_name": file_name,
                "source_relative_path": f"train/{file_name}",
                "view_relative_path": f"images/{split}/{file_name}",
                "sha256": digest,
                "exact_hash_group_size": hash_group_sizes[digest],
                "object_count": len(label_lines),
                "class_ids_present": ";".join(map(str, sorted({a[0] for a in annotations[file_name]}))),
            }
        )
        if index % 20000 == 0:
            print(f"  prepared {index:,}/{len(assignments):,} labelled images")

    print("Linking the label-free official MIO challenge test images...")
    for src in official_test_paths:
        hardlink(src, staging_root / "images" / "official_test" / src.name)

    yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(MIO_CLASSES)},
    }
    (staging_root / "mio_tcd_full.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8"
    )
    for split in ("train", "val", "test"):
        lines = [
            f"./images/{split}/{row['file_name']}"
            for row in manifest_rows
            if row["split"] == split
        ]
        (staging_root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (staging_root / "official_test.txt").write_text(
        "\n".join(f"./images/official_test/{p.name}" for p in official_test_paths) + "\n",
        encoding="utf-8",
    )

    write_csv(
        staging_root / "split_manifest.csv",
        [
            "dataset_id",
            "split",
            "file_name",
            "source_relative_path",
            "view_relative_path",
            "sha256",
            "exact_hash_group_size",
            "object_count",
            "class_ids_present",
        ],
        manifest_rows,
    )
    write_csv(
        staging_root / "duplicate_groups.csv",
        [
            "duplicate_group_id",
            "sha256",
            "group_size",
            "file_name",
            "split",
            "annotation_count",
            "annotations_identical_within_group",
        ],
        duplicate_manifest_rows,
    )
    write_csv(
        staging_root / "excluded_invalid_boxes.csv",
        ["row_number", "file_name", "class_name", "x1", "y1", "x2", "y2", "reason"],
        invalid_rows,
    )
    class_rows = split_statistics(assignments, annotations, MIO_CLASSES)
    write_csv(
        staging_root / "class_distribution.csv",
        ["split", "class_id", "class_name", "images_with_class", "object_count", "split_images"],
        class_rows,
    )
    write_json(staging_root / "preparation_summary.json", summary)

    staging_root.replace(final_root)
    args.manifests_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_root / "split_manifest.csv", args.manifests_root / f"{summary['dataset_id']}_split_manifest.csv")
    shutil.copy2(final_root / "duplicate_groups.csv", args.manifests_root / f"{summary['dataset_id']}_duplicate_groups.csv")
    shutil.copy2(final_root / "class_distribution.csv", args.results_root / f"{summary['dataset_id']}_class_distribution.csv")
    shutil.copy2(final_root / "preparation_summary.json", args.results_root / f"{summary['dataset_id']}_summary.json")
    print(f"Prepared MIO dataset view: {final_root}")
    return summary


def load_acdc_split(source: Path, split: str) -> tuple[dict, Path]:
    suffix = "image_info" if split == "test" else "gt_detection"
    json_path = source / "gt_detection" / "snow" / f"instancesonly_snow_{split}_{suffix}.json"
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    return json.loads(json_path.read_text(encoding="utf-8")), json_path


def prepare_acdc(args: argparse.Namespace) -> dict[str, object]:
    source = args.acdc_root
    class_to_id = {name: idx for idx, name in enumerate(ACDC_CLASSES)}
    split_payloads: dict[str, dict] = {}
    crowd_counts: dict[str, int] = {}
    ordinary_counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}

    for split in ("train", "val", "test"):
        payload, _ = load_acdc_split(source, split)
        categories = [category["name"] for category in payload["categories"]]
        if categories != ACDC_CLASSES:
            raise ValueError(f"Unexpected ACDC category order for {split}: {categories}")
        image_ids = {image["id"] for image in payload["images"]}
        if len(image_ids) != len(payload["images"]):
            raise ValueError(f"Duplicate ACDC image IDs in {split}.")
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        crowd = 0
        for ann in payload.get("annotations", []):
            if ann["image_id"] not in image_ids:
                raise ValueError(f"ACDC annotation references missing image ID {ann['image_id']}.")
            if ann.get("iscrowd", 0):
                crowd += 1
                continue
            annotations_by_image[ann["image_id"]].append(ann)
        crowd_counts[split] = crowd
        ordinary_counts[split] = sum(len(v) for v in annotations_by_image.values())
        negative_counts[split] = sum(image["id"] not in annotations_by_image for image in payload["images"])
        split_payloads[split] = {
            "payload": payload,
            "annotations_by_image": annotations_by_image,
        }

    expected_images = {"train": 400, "val": 100, "test": 500}
    for split, expected in expected_images.items():
        found = len(split_payloads[split]["payload"]["images"])
        if found != expected:
            raise ValueError(f"Expected {expected} ACDC snow {split} images, found {found}.")

    summary = {
        "dataset_id": "acdc_snow_official_v1",
        "source": str(source),
        "classes": ACDC_CLASSES,
        "split_counts": expected_images,
        "ordinary_annotations": ordinary_counts,
        "excluded_iscrowd_annotations": crowd_counts,
        "images_without_ordinary_annotations": negative_counts,
        "split_rule": "official ACDC snow train/val/test split preserved",
        "test_ground_truth": "withheld; official evaluation server only",
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return summary

    final_root = args.output_root / summary["dataset_id"]
    staging_root = args.output_root / f".{summary['dataset_id']}.building"
    if final_root.exists() or staging_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing ACDC output: {final_root} or {staging_root}")
    staging_root.mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    for split in ("train", "val", "test"):
        payload = split_payloads[split]["payload"]
        annotations_by_image = split_payloads[split]["annotations_by_image"]
        object_counts = Counter()
        image_class_counts = Counter()

        for image_info in sorted(payload["images"], key=lambda item: item["file_name"]):
            source_image = source / Path(image_info["file_name"])
            if not source_image.is_file():
                raise FileNotFoundError(source_image)
            relative_tail = Path(*Path(image_info["file_name"]).parts[2:])
            if not relative_tail.parts:
                raise ValueError(f"Unexpected ACDC file path: {image_info['file_name']}")
            view_image = staging_root / "images" / split / relative_tail
            hardlink(source_image, view_image)

            label_lines: list[str] = []
            labels_present: set[int] = set()
            for ann in annotations_by_image.get(image_info["id"], []):
                category_name = next(
                    category["name"]
                    for category in payload["categories"]
                    if category["id"] == ann["category_id"]
                )
                class_id = class_to_id[category_name]
                x, y, width, height = map(float, ann["bbox"])
                if width <= 0 or height <= 0:
                    raise ValueError(f"Invalid ACDC bbox in {split}: {ann['bbox']}")
                label_lines.append(
                    yolo_line(
                        class_id,
                        x,
                        y,
                        x + width,
                        y + height,
                        int(image_info["width"]),
                        int(image_info["height"]),
                    )
                )
                object_counts[class_id] += 1
                labels_present.add(class_id)
            for class_id in labels_present:
                image_class_counts[class_id] += 1

            if split != "test":
                label_path = staging_root / "labels" / split / relative_tail.with_suffix(".txt")
                label_path.parent.mkdir(parents=True, exist_ok=True)
                label_path.write_text(
                    ("\n".join(label_lines) + "\n") if label_lines else "",
                    encoding="utf-8",
                )

            manifest_rows.append(
                {
                    "dataset_id": summary["dataset_id"],
                    "split": split,
                    "image_id": image_info["id"],
                    "file_name": image_info["file_name"],
                    "source_relative_path": image_info["file_name"],
                    "view_relative_path": str(Path("images") / split / relative_tail).replace("\\", "/"),
                    "width": image_info["width"],
                    "height": image_info["height"],
                    "ordinary_object_count": len(label_lines),
                    "has_public_label": split != "test",
                    "class_ids_present": ";".join(map(str, sorted(labels_present))),
                }
            )

        for class_id, class_name in enumerate(ACDC_CLASSES):
            class_rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "images_with_class": image_class_counts[class_id],
                    "object_count": object_counts[class_id],
                    "split_images": expected_images[split],
                }
            )

    yaml_data = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {idx: name for idx, name in enumerate(ACDC_CLASSES)},
    }
    (staging_root / "acdc_snow.yaml").write_text(
        yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8"
    )
    write_csv(
        staging_root / "split_manifest.csv",
        [
            "dataset_id",
            "split",
            "image_id",
            "file_name",
            "source_relative_path",
            "view_relative_path",
            "width",
            "height",
            "ordinary_object_count",
            "has_public_label",
            "class_ids_present",
        ],
        manifest_rows,
    )
    write_csv(
        staging_root / "class_distribution.csv",
        ["split", "class_id", "class_name", "images_with_class", "object_count", "split_images"],
        class_rows,
    )
    write_json(staging_root / "preparation_summary.json", summary)

    staging_root.replace(final_root)
    args.manifests_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    shutil.copy2(final_root / "split_manifest.csv", args.manifests_root / f"{summary['dataset_id']}_split_manifest.csv")
    shutil.copy2(final_root / "class_distribution.csv", args.results_root / f"{summary['dataset_id']}_class_distribution.csv")
    shutil.copy2(final_root / "preparation_summary.json", args.results_root / f"{summary['dataset_id']}_summary.json")
    print(f"Prepared ACDC dataset view: {final_root}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("both", "mio", "acdc"), default="both")
    parser.add_argument(
        "--mio-root",
        type=Path,
        default=PROJECT / "MIO-TCD-Localization" / "MIO-TCD-Localization",
    )
    parser.add_argument("--acdc-root", type=Path, default=PROJECT / "rgb_anon")
    parser.add_argument("--output-root", type=Path, default=PROJECT / "data_views")
    parser.add_argument("--manifests-root", type=Path, default=PROJECT / "manifests")
    parser.add_argument("--results-root", type=Path, default=PROJECT / "results" / "dataset_prep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mio-total", type=int, default=110000)
    parser.add_argument("--mio-train", type=int, default=88000)
    parser.add_argument("--mio-val", type=int, default=11000)
    parser.add_argument("--mio-test", type=int, default=11000)
    parser.add_argument("--workers", type=int, default=min(16, max(1, os.cpu_count() or 1)))
    parser.add_argument("--dry-run", action="store_true", help="Audit and compute splits without writing outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.mio_root = args.mio_root.resolve()
    args.acdc_root = args.acdc_root.resolve()
    args.output_root = args.output_root.resolve()
    args.manifests_root = args.manifests_root.resolve()
    args.results_root = args.results_root.resolve()

    summaries: dict[str, object] = {}
    if args.dataset in ("both", "mio"):
        summaries["mio"] = prepare_mio(args)
    if args.dataset in ("both", "acdc"):
        summaries["acdc"] = prepare_acdc(args)
    print(json.dumps({"status": "dry-run" if args.dry_run else "prepared", "datasets": summaries}, indent=2))


if __name__ == "__main__":
    main()
