import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/mio_tcd_yolo")
    parser.add_argument("--source_yaml", default="data/mio_tcd_yolo/mio_tcd.yaml")
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--out_yaml", default="data/mio_tcd_yolo/mio_tcd_profile_500.yaml")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    source_yaml = Path(args.source_yaml)
    out_yaml = Path(args.out_yaml)

    val_image_dir = dataset_dir / "images" / "val"
    val_txt = dataset_dir / f"profile_val_{args.n}.txt"

    image_paths = sorted(val_image_dir.glob("*.jpg"))

    if len(image_paths) < args.n:
        raise ValueError(f"Requested {args.n} images, but only found {len(image_paths)} validation images.")

    selected_paths = image_paths[:args.n]

    # Reason:
    # Ultralytics can read a txt file containing image paths.
    # Absolute paths reduce path confusion when the script is launched from different folders.
    val_txt.write_text(
        "\n".join(path.resolve().as_posix() for path in selected_paths) + "\n",
        encoding="utf-8",
    )

    yaml_lines = []

    for line in source_yaml.read_text(encoding="utf-8").splitlines():
        if line.startswith("val:"):
            yaml_lines.append(f"val: {val_txt.resolve().as_posix()}")
        else:
            yaml_lines.append(line)

    out_yaml.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Validation subset txt saved to: {val_txt}")
    print(f"New YAML saved to: {out_yaml}")


if __name__ == "__main__":
    main()


'''
python scripts/2.0_make_profile_val_yaml.py --n 500
'''