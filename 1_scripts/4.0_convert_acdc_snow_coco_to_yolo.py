from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


MIO_NAMES = [
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

# ACDC COCO category name -> current MIO/YOLO class id.
# Only overlapping traffic classes are mapped.
ACDC_TO_MIO = {
    "bicycle": 1,
    "bus": 2,
    "car": 3,
    "motorcycle": 4,
    "person": 7,
    "pedestrian": 7,
    "truck": 9,
}


def find_image(acdc_root: Path, file_name: str, split: str) -> Path:
    basename = Path(file_name).name

    # First search likely snow image folders.
    matches = [
        p for p in acdc_root.rglob(basename)
        if "/snow/" in str(p) and f"/{split}/" in str(p)
    ]

    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return matches[0]

    # Fallback: search by basename anywhere.
    matches = list(acdc_root.rglob(basename))
    if len(matches) == 1:
        return matches[0]

    if len(matches) > 1:
        return matches[0]

    raise FileNotFoundError(f"Image not found for COCO file_name: {file_name}")


def convert_split(
    acdc_root: Path,
    json_path: Path,
    out_root: Path,
    split: str,
    copy_images: bool,
) -> dict[str, int]:
    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    category_id_to_name = {
        int(cat["id"]): str(cat["name"]).lower()
        for cat in coco["categories"]
    }

    image_id_to_info = {
        int(img["id"]): img
        for img in coco["images"]
    }

    anns_by_image: dict[int, list[dict]] = {}
    for ann in coco["annotations"]:
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    image_out_dir = out_root / "images" / split
    label_out_dir = out_root / "labels" / split
    image_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    label_count = 0
    skipped_class_count = 0
    missing_image_count = 0

    for image_id, info in image_id_to_info.items():
        file_name = info["file_name"]
        width = float(info["width"])
        height = float(info["height"])

        try:
            src_img = find_image(acdc_root, file_name, split)
        except FileNotFoundError:
            missing_image_count += 1
            continue

        dst_img = image_out_dir / src_img.name

        if not dst_img.exists():
            if copy_images:
                shutil.copy2(src_img, dst_img)
            else:
                dst_img.symlink_to(src_img.resolve())

        label_lines = []

        for ann in anns_by_image.get(image_id, []):
            category_name = category_id_to_name[int(ann["category_id"])]

            if category_name not in ACDC_TO_MIO:
                skipped_class_count += 1
                continue

            cls = ACDC_TO_MIO[category_name]

            # COCO bbox = [x_min, y_min, width, height]
            x, y, w, h = map(float, ann["bbox"])

            if w <= 0 or h <= 0:
                continue

            x_center = (x + w / 2.0) / width
            y_center = (y + h / 2.0) / height
            box_w = w / width
            box_h = h / height

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            box_w = min(max(box_w, 0.0), 1.0)
            box_h = min(max(box_h, 0.0), 1.0)

            label_lines.append(
                f"{cls} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}"
            )

        label_path = label_out_dir / f"{src_img.stem}.txt"
        label_path.write_text(
            "\n".join(label_lines) + ("\n" if label_lines else ""),
            encoding="utf-8",
        )

        image_count += 1
        label_count += len(label_lines)

    return {
        "split": split,
        "images": image_count,
        "labels": label_count,
        "skipped_classes": skipped_class_count,
        "missing_images": missing_image_count,
    }


parser = argparse.ArgumentParser(description="Convert ACDC SNOW COCO detection data to YOLO format.")
parser.add_argument("--acdc-root", required=True, type=Path)
parser.add_argument("--json-root", required=True, type=Path)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--copy-images", action="store_true")
args = parser.parse_args()

train_json = args.json_root / "instancesonly_snow_train_gt_detection.json"
val_json = args.json_root / "instancesonly_snow_val_gt_detection.json"

if not train_json.is_file():
    raise FileNotFoundError(f"Missing train JSON: {train_json}")
if not val_json.is_file():
    raise FileNotFoundError(f"Missing val JSON: {val_json}")

args.out.mkdir(parents=True, exist_ok=True)

train_stats = convert_split(
    acdc_root=args.acdc_root,
    json_path=train_json,
    out_root=args.out,
    split="train",
    copy_images=args.copy_images,
)

val_stats = convert_split(
    acdc_root=args.acdc_root,
    json_path=val_json,
    out_root=args.out,
    split="val",
    copy_images=args.copy_images,
)

yaml_path = args.out / "acdc_snow_mio.yaml"

yaml_text = f"""path: {args.out.resolve()}
train: images/train
val: images/val

nc: {len(MIO_NAMES)}
names:
"""

for i, name in enumerate(MIO_NAMES):
    yaml_text += f"  {i}: {name}\n"

yaml_path.write_text(yaml_text, encoding="utf-8")

print("ACDC SNOW conversion completed.")
print(f"Train JSON: {train_json}")
print(f"Val JSON: {val_json}")
print(f"Output dataset: {args.out.resolve()}")
print(f"YAML: {yaml_path.resolve()}")
print(f"Train stats: {train_stats}")
print(f"Val stats: {val_stats}")
print("Class mapping:")
for acdc_name, mio_id in ACDC_TO_MIO.items():
    print(f"  {acdc_name} -> {mio_id} {MIO_NAMES[mio_id]}")
