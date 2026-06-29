from pathlib import Path
import csv
import random
import shutil
from PIL import Image
from tqdm import tqdm

# Class order must stay consistent with the generated YOLO label IDs and data.yaml.
# Example: "car" has the same index here, in every .txt label file, and in mio_tcd.yaml.
CLASS_NAMES = [
    "articulated_truck",
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "motorized_vehicle",
    "non-motorized_vehicle",
    "pedestrian",
    "pickup_truck",
    "single_unit_truck",
    "work_van",
]

# Convert class names from the MIO-TCD CSV into integer YOLO class IDs.
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}

# Raw MIO-TCD inputs and converted YOLO output folder.
# The output folder will contain images/train, images/val, labels/train, labels/val, and mio_tcd.yaml.
RAW_IMAGE_DIR = Path("data/raw/MIO-TCD-Localization/train")
CSV_PATH = Path("data/raw/MIO-TCD-Localization/gt_train.csv")
OUTPUT_DIR = Path("data/mio_tcd_yolo")

# Use a fixed validation ratio and random seed so the train/val split is reproducible.
VAL_RATIO = 0.1
RANDOM_SEED = 42

# check path
if not RAW_IMAGE_DIR.exists():
    raise FileNotFoundError(f"Image folder not found: {RAW_IMAGE_DIR}")

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

# image bounding boxes: annotations["00000000"] = [(class_name, xmin, ymin, xmax, ymax), ...]
annotations = {}
invalid_rows = []

# Read each CSV row as one object annotation:
# image_id, class_name, xmin, ymin, xmax, ymax.
# Invalid rows are rejected early to avoid producing incorrect YOLO labels.
with CSV_PATH.open("r", newline="", encoding="utf-8") as csv_file:
    reader = csv.reader(csv_file)

    for row_number, row in enumerate(reader, start=1):
        if len(row) != 6:
            raise ValueError(f"Row {row_number} should have 6 columns, but got {len(row)}: {row}")

        image_id = row[0]
        class_name = row[1]

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"Unknown class '{class_name}' at row {row_number}. "
                f"Please update CLASS_NAMES to match your CSV."
            )

        xmin = int(row[2])
        ymin = int(row[3])
        xmax = int(row[4])
        ymax = int(row[5])

        # YOLO training requires valid boxes with positive width and height.
        # A zero-area or negative-area box cannot be converted into a meaningful YOLO bbox.
        if xmax <= xmin or ymax <= ymin:
            invalid_rows.append({
                "row_number": row_number,
                "image_id": image_id,
                "class_name": class_name,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "reason": "zero_or_negative_bbox_size",
            })
            continue

        # Group all object boxes by image ID.
        # One image can contain multiple objects, so each image_id maps to a list of bounding boxes.
        annotations.setdefault(image_id, []).append((class_name, xmin, ymin, xmax, ymax))

if invalid_rows:
    invalid_log_path = OUTPUT_DIR / "invalid_bboxes.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with invalid_log_path.open("w", newline="", encoding="utf-8") as log_file:
        fieldnames = [
            "row_number",
            "image_id",
            "class_name",
            "xmin",
            "ymin",
            "xmax",
            "ymax",
            "reason",
        ]

        writer = csv.DictWriter(log_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(invalid_rows)

    print(f"Skipped invalid bounding boxes: {len(invalid_rows)}")
    print(f"Invalid bbox log saved to: {invalid_log_path}")

image_ids = sorted(annotations.keys())

# ---
# confirm both side of spilting contain every classes
from collections import Counter, defaultdict

image_ids = sorted(annotations.keys())

# Build per-image class counters so the validation split can keep a similar class distribution.
# This is important because MIO-TCD is class-imbalanced; rare classes should still appear in validation.
image_class_counts = {}

for image_id in image_ids:
    class_counter = Counter()

    for class_name, xmin, ymin, xmax, ymax in annotations[image_id]:
        class_id = CLASS_TO_ID[class_name]
        class_counter[class_id] += 1

    image_class_counts[image_id] = class_counter

# Count total objects per class across the whole dataset.
# These counts are used to estimate the target number of validation objects per class.
total_class_counts = Counter()

for class_counter in image_class_counts.values():
    total_class_counts.update(class_counter)

# for each class: validation set's target object amount
# if VAL_RATIO = 0.1: validation set contain around 10% objects of each class
target_val_class_counts = {
    class_id: max(1, int(total_count * VAL_RATIO))
    for class_id, total_count in total_class_counts.items()
}

target_val_image_count = int(len(image_ids) * VAL_RATIO)

# class_id -> image_ids: put rare classes to validation set first
images_by_class = defaultdict(list)

for image_id, class_counter in image_class_counts.items():
    for class_id in class_counter:
        images_by_class[class_id].append(image_id)

random.seed(RANDOM_SEED)

# Select validation images class-by-class, starting from rare classes.
# This reduces the chance that rare classes disappear from the validation set.
class_order = sorted(total_class_counts.keys(), key=lambda class_id: total_class_counts[class_id])

val_ids = set()
val_class_counts = Counter()

for class_id in class_order:
    candidate_image_ids = images_by_class[class_id][:]
    random.shuffle(candidate_image_ids)

    for image_id in candidate_image_ids:
        if len(val_ids) >= target_val_image_count:
            break

        if val_class_counts[class_id] >= target_val_class_counts[class_id]:
            break

        if image_id in val_ids:
            continue

        val_ids.add(image_id)
        val_class_counts.update(image_class_counts[image_id])

# If the stratified selection does not reach the target validation image count,
# fill the remaining validation set with randomly selected unused images.
remaining_image_ids = [image_id for image_id in image_ids if image_id not in val_ids]
random.shuffle(remaining_image_ids)

for image_id in remaining_image_ids:
    if len(val_ids) >= target_val_image_count:
        break

    val_ids.add(image_id)
    val_class_counts.update(image_class_counts[image_id])

train_ids = set(image_ids) - val_ids

print("\nStratified split summary")
print(f"Train images: {len(train_ids)}")
print(f"Val images: {len(val_ids)}")

for class_id, class_name in enumerate(CLASS_NAMES):
    print(
        f"{class_id:2d} | {class_name:25s} | "
        f"val objects: {val_class_counts[class_id]:8d} | "
        f"target: {target_val_class_counts.get(class_id, 0):8d}"
    )
# ---

for split_name in ["train", "val"]:
    (OUTPUT_DIR / "images" / split_name).mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels" / split_name).mkdir(parents=True, exist_ok=True)

for image_id in tqdm(image_ids, desc="Converting MIO-TCD to YOLO"):
    split_name = "val" if image_id in val_ids else "train"

    source_image_path = RAW_IMAGE_DIR / f"{image_id}.jpg"
    target_image_path = OUTPUT_DIR / "images" / split_name / f"{image_id}.jpg"
    target_label_path = OUTPUT_DIR / "labels" / split_name / f"{image_id}.txt"

    if not source_image_path.exists():
        raise FileNotFoundError(f"Image not found for CSV entry: {source_image_path}")

    # read image size, YOLO label need normalized bbox
    with Image.open(source_image_path) as image:
        image_width, image_height = image.size

    yolo_lines = []

    for class_name, xmin, ymin, xmax, ymax in annotations[image_id]:
        class_id = CLASS_TO_ID[class_name]

        # Convert MIO-TCD bbox format to YOLO bbox format.
        # MIO-TCD uses pixel xyxy: xmin, ymin, xmax, ymax.
        # YOLO uses normalized xywh: x_center, y_center, width, height, all divided by image size.
        x_center = ((xmin + xmax) / 2) / image_width
        y_center = ((ymin + ymax) / 2) / image_height
        box_width = (xmax - xmin) / image_width
        box_height = (ymax - ymin) / image_height

        yolo_lines.append(
            f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
        )

    # Copy the original image into the YOLO folder structure.
    # Ultralytics matches each image with a label file using the same filename.
    shutil.copy2(source_image_path, target_image_path)

    # Write one YOLO label file per image.
    # Each line represents one object: class_id x_center y_center width height.
    target_label_path.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

# Generate the dataset YAML required by Ultralytics.
# It tells YOLO where the train/val images are and how class IDs map to class names.
yaml_lines = [
    f"path: {OUTPUT_DIR.as_posix()}",
    "train: images/train",
    "val: images/val",
    "names:",
]

for class_id, class_name in enumerate(CLASS_NAMES):
    yaml_lines.append(f"  {class_id}: {class_name}")

(OUTPUT_DIR / "mio_tcd.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

print(f"Done. Train images: {len(train_ids)}, Val images: {len(val_ids)}")
print(f"Dataset YAML: {OUTPUT_DIR / 'mio_tcd.yaml'}")

'''
python scripts/1.1_convert_mio_to_yolo.py
'''