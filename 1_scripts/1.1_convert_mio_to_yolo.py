"""
Convert MIO-TCD Localization annotations to YOLO detection format.

Expected MIO-TCD Localization annotation CSV format, based on the public gt_train.csv example:
    image_id, class_name, x1, y1, x2, y2
without a header row, where image_id 0 maps to 00000000.jpg.

This script creates the report split:
    11000 total images = 7700 train + 1650 val + 1650 test

The script is intentionally strict: if required files, classes, or bounding boxes are invalid,
it stops instead of silently changing the experiment logic.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
import yaml
from PIL import Image
from tqdm import tqdm


parser = argparse.ArgumentParser(description="Convert MIO-TCD Localization to YOLO format.")
parser.add_argument("--image-dir", required=True, type=Path, help="Folder containing MIO images such as 00000000.jpg.")
parser.add_argument("--ann-csv", required=True, type=Path, help="MIO annotation CSV, usually gt_train.csv.")
parser.add_argument("--out", required=True, type=Path, help="Output YOLO dataset folder.")
parser.add_argument("--total-images", type=int, default=11000, help="Total selected annotated images.")
parser.add_argument("--train-images", type=int, default=7700, help="Number of train images.")
parser.add_argument("--val-images", type=int, default=1650, help="Number of validation images.")
parser.add_argument("--test-images", type=int, default=1650, help="Number of test images.")
parser.add_argument("--seed", type=int, default=42, help="Fixed seed for reproducible sampling and splitting.")
parser.add_argument(
    "--copy-mode",
    choices=["copy", "symlink"],
    default="symlink",
    help="Use symlink to save space, or copy to make a standalone dataset.",
)
args = parser.parse_args()

# These are the 11 MIO-TCD Localization labels. The order becomes the YOLO class id.
# Keeping this list fixed is important because changing class order changes the meaning of labels.
class_names = [
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
class_to_id = {name: idx for idx, name in enumerate(class_names)}

if args.total_images != args.train_images + args.val_images + args.test_images:
    raise ValueError("total-images must equal train-images + val-images + test-images.")

if not args.image_dir.is_dir():
    raise FileNotFoundError(f"Image directory not found: {args.image_dir}")

if not args.ann_csv.is_file():
    raise FileNotFoundError(f"Annotation CSV not found: {args.ann_csv}")

# MIO gt_train.csv has no header in the public example.
# image_id is converted to an 8-digit jpg filename, e.g., 0 -> 00000000.jpg.
ann = pd.read_csv(
    args.ann_csv,
    header=None,
    names=["image_id", "class_name", "x1", "y1", "x2", "y2"],
)

# Normalize class strings so that labels such as "pickup truck" and "pickup_truck"
# map to the same fixed YOLO class name.
ann["class_name"] = (
    ann["class_name"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace("-", "_", regex=False)
    .str.replace(" ", "_", regex=False)
)

unknown_classes = sorted(set(ann["class_name"]) - set(class_names))
if unknown_classes:
    raise ValueError(f"Unknown class names found in CSV: {unknown_classes}")

# Bounding boxes must be valid pixel coordinates before conversion to YOLO normalized format.
# Stopping here prevents invalid labels from silently entering training.
for col in ["image_id", "x1", "y1", "x2", "y2"]:
    ann[col] = pd.to_numeric(ann[col], errors="raise")

invalid_box_mask = (ann["x2"] <= ann["x1"]) | (ann["y2"] <= ann["y1"])
invalid_box_count = int(invalid_box_mask.sum())

if invalid_box_count > 0:
    # Invalid boxes have zero or negative width/height, so they cannot be converted
    # into YOLO format. They are recorded for traceability, then removed from the
    # annotation table before image sampling and label generation.
    invalid_boxes_out = args.out.parent / "invalid_mio_boxes.csv"
    invalid_boxes_out.parent.mkdir(parents=True, exist_ok=True)
    ann.loc[invalid_box_mask].to_csv(invalid_boxes_out, index=False)

    print(
        f"Warning: removed {invalid_box_count} invalid bounding boxes "
        f"where x2<=x1 or y2<=y1. Details saved to: {invalid_boxes_out}"
    )

    ann = ann.loc[~invalid_box_mask].copy()

ann["image_id"] = ann["image_id"].astype(int)
ann["file_name"] = ann["image_id"].map(lambda x: f"{x:08d}.jpg")

# Check that every annotated image exists before sampling. This avoids producing a split
# that later fails during training.
unique_file_names = sorted(ann["file_name"].unique())
missing_images = [name for name in unique_file_names if not (args.image_dir / name).is_file()]
if missing_images:
    example = ", ".join(missing_images[:10])
    raise FileNotFoundError(f"Missing {len(missing_images)} images in {args.image_dir}. Examples: {example}")

# Each image can contain several objects. For stratified image sampling, the image is assigned
# to its most frequent object class. This keeps the image split approximately class-balanced
# while still writing all bounding boxes for that image.
primary_class_by_file = {}
for file_name, group in ann.groupby("file_name"):
    primary_class_by_file[file_name] = Counter(group["class_name"]).most_common(1)[0][0]

files_by_primary_class = defaultdict(list)
for file_name, primary_class in primary_class_by_file.items():
    files_by_primary_class[primary_class].append(file_name)

random.seed(args.seed)
for file_list in files_by_primary_class.values():
    random.shuffle(file_list)

available_total = sum(len(v) for v in files_by_primary_class.values())
if available_total < args.total_images:
    raise ValueError(f"Only {available_total} annotated images available, but {args.total_images} requested.")

# Proportional quota with largest-remainder correction gives an exact total image count.
# This avoids using a simple random split that may accidentally over/under represent classes.
raw_selection_quota = []
for class_name in class_names:
    available = len(files_by_primary_class[class_name])
    exact = available / available_total * args.total_images
    raw_selection_quota.append([class_name, int(exact), exact - int(exact), available])

selected_count = sum(row[1] for row in raw_selection_quota)
for row in sorted(raw_selection_quota, key=lambda x: x[2], reverse=True):
    if selected_count >= args.total_images:
        break
    if row[1] < row[3]:
        row[1] += 1
        selected_count += 1

selected_files = []
for class_name, quota, _fraction, available in raw_selection_quota:
    if quota > available:
        raise ValueError(f"Requested quota {quota} exceeds available images {available} for class {class_name}.")
    selected_files.extend(files_by_primary_class[class_name][:quota])

if len(selected_files) != args.total_images:
    raise RuntimeError(f"Internal split error: selected {len(selected_files)} images instead of {args.total_images}.")

# Split the selected images per primary class to keep train/val/test distributions close.
selected_by_primary_class = defaultdict(list)
for file_name in selected_files:
    selected_by_primary_class[primary_class_by_file[file_name]].append(file_name)

split_targets = {"train": args.train_images, "val": args.val_images, "test": args.test_images}
split_files = {"train": [], "val": [], "test": []}

for class_name in class_names:
    file_list = selected_by_primary_class[class_name]
    random.shuffle(file_list)
    n = len(file_list)
    train_n = int(n * args.train_images / args.total_images)
    val_n = int(n * args.val_images / args.total_images)
    test_n = n - train_n - val_n
    split_files["train"].extend(file_list[:train_n])
    split_files["val"].extend(file_list[train_n : train_n + val_n])
    split_files["test"].extend(file_list[train_n + val_n : train_n + val_n + test_n])

# Correct exact split sizes by moving files between splits. The moves are deterministic
# because all lists were shuffled using the fixed seed above.
for split_name in split_files:
    random.shuffle(split_files[split_name])

for split_name, target in split_targets.items():
    while len(split_files[split_name]) > target:
        moved_file = split_files[split_name].pop()
        for receiver_name, receiver_target in split_targets.items():
            if len(split_files[receiver_name]) < receiver_target:
                split_files[receiver_name].append(moved_file)
                break

for split_name, target in split_targets.items():
    if len(split_files[split_name]) != target:
        raise RuntimeError(f"Split {split_name} has {len(split_files[split_name])} images, expected {target}.")

args.out.mkdir(parents=True, exist_ok=False)
for split_name in ["train", "val", "test"]:
    (args.out / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / split_name).mkdir(parents=True, exist_ok=True)

# Pre-group annotations for faster label writing.
ann_by_file = {file_name: group.copy() for file_name, group in ann.groupby("file_name")}

manifest_rows = []
class_counter_by_split = {"train": Counter(), "val": Counter(), "test": Counter()}

for split_name in ["train", "val", "test"]:
    for file_name in tqdm(split_files[split_name], desc=f"Writing {split_name}"):
        src_image = args.image_dir / file_name
        dst_image = args.out / "images" / split_name / file_name
        dst_label = args.out / "labels" / split_name / f"{Path(file_name).stem}.txt"

        if args.copy_mode == "copy":
            shutil.copy2(src_image, dst_image)
        else:
            os.symlink(src_image.resolve(), dst_image)

        with Image.open(src_image) as img:
            img_w, img_h = img.size

        label_lines = []
        for row in ann_by_file[file_name].itertuples(index=False):
            class_id = class_to_id[row.class_name]
            x1 = float(row.x1)
            y1 = float(row.y1)
            x2 = float(row.x2)
            y2 = float(row.y2)

            if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                raise ValueError(
                    f"Bounding box outside image boundary for {file_name}: "
                    f"box=({x1}, {y1}, {x2}, {y2}), image=({img_w}, {img_h})"
                )

            # YOLO format: class_id x_center y_center width height, all normalized to [0, 1].
            x_center = ((x1 + x2) / 2.0) / img_w
            y_center = ((y1 + y2) / 2.0) / img_h
            box_w = (x2 - x1) / img_w
            box_h = (y2 - y1) / img_h
            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")
            class_counter_by_split[split_name][row.class_name] += 1

        dst_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
        manifest_rows.append(
            {
                "split": split_name,
                "file_name": file_name,
                "primary_class": primary_class_by_file[file_name],
                "object_count": len(label_lines),
            }
        )

# Ultralytics dataset YAML. Paths are relative to this YAML file, so the folder can move together.
yaml_data = {
    "path": str(args.out.resolve()),
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": {idx: name for idx, name in enumerate(class_names)},
}
(args.out / "mio_tcd.yaml").write_text(yaml.safe_dump(yaml_data, sort_keys=False), encoding="utf-8")

with (args.out / "split_manifest.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["split", "file_name", "primary_class", "object_count"])
    writer.writeheader()
    writer.writerows(manifest_rows)

with (args.out / "class_balance.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class_name", "train_objects", "val_objects", "test_objects", "total_objects"])
    for class_name in class_names:
        train_count = class_counter_by_split["train"][class_name]
        val_count = class_counter_by_split["val"][class_name]
        test_count = class_counter_by_split["test"][class_name]
        writer.writerow([class_name, train_count, val_count, test_count, train_count + val_count + test_count])

# Write image-level primary class balance.
# This checks whether the stratified image split kept the primary-class image distribution balanced.
# It is different from class_balance.csv, which counts object boxes instead of images.
with (args.out / "primary_class_balance.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["class_name", "train_images", "val_images", "test_images", "total_images"])

    for class_name in class_names:
        train_count = sum(
            1 for file_name in split_files["train"]
            if primary_class_by_file[file_name] == class_name
        )
        val_count = sum(
            1 for file_name in split_files["val"]
            if primary_class_by_file[file_name] == class_name
        )
        test_count = sum(
            1 for file_name in split_files["test"]
            if primary_class_by_file[file_name] == class_name
        )
        writer.writerow([class_name, train_count, val_count, test_count, train_count + val_count + test_count])

# Write a compact split summary for the report.
# This makes it easy to verify that the generated dataset matches the 7700/1650/1650 report setting.
with (args.out / "split_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["split", "image_count", "object_count"])

    for split_name in ["train", "val", "test"]:
        image_count = len(split_files[split_name])
        object_count = sum(
            class_counter_by_split[split_name][class_name]
            for class_name in class_names
        )
        writer.writerow([split_name, image_count, object_count])

print("Conversion completed.")
print(f"YOLO dataset: {args.out.resolve()}")
print(f"YAML file: {(args.out / 'mio_tcd.yaml').resolve()}")
print(f"Split counts: train={len(split_files['train'])}, val={len(split_files['val'])}, test={len(split_files['test'])}")
