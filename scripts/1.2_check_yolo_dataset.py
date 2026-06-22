from pathlib import Path
from collections import Counter

# = mio_tcd.yaml class order
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

DATASET_DIR = Path("data/mio_tcd_yolo")
TRAIN_LABEL_DIR = DATASET_DIR / "labels" / "train"
VAL_LABEL_DIR = DATASET_DIR / "labels" / "val"


# check exists of label folder
if not TRAIN_LABEL_DIR.exists():
    raise FileNotFoundError(f"Train label folder not found: {TRAIN_LABEL_DIR}")

if not VAL_LABEL_DIR.exists():
    raise FileNotFoundError(f"Val label folder not found: {VAL_LABEL_DIR}")


for split_name, label_dir in [("train", TRAIN_LABEL_DIR), ("val", VAL_LABEL_DIR)]:
    class_counter = Counter()
    label_files = sorted(label_dir.glob("*.txt"))

    for label_file in label_files:
        lines = label_file.read_text(encoding="utf-8").strip().splitlines()

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            # YOLO detection label: class_id, x_center, y_center, width, height
            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO label format in {label_file}, "
                    f"line {line_number}: {line}"
                )

            class_id = int(parts[0])

            # check class_id vs CLASS_NAMES range: dont want to learn wrong label
            if class_id < 0 or class_id >= len(CLASS_NAMES):
                raise ValueError(
                    f"Invalid class_id {class_id} in {label_file}, "
                    f"line {line_number}"
                )

            class_counter[class_id] += 1

    print(f"\n=== {split_name.upper()} SET ===")
    print(f"Label files: {len(label_files)}")

    total_objects = sum(class_counter.values())
    print(f"Total objects: {total_objects}")

    for class_id, class_name in enumerate(CLASS_NAMES):
        count = class_counter[class_id]
        percentage = count / total_objects * 100 if total_objects > 0 else 0

        print(f"{class_id:2d} | {class_name:25s} | {count:8d} | {percentage:6.2f}%")

'''
python scripts/1.2_check_yolo_dataset.py
'''