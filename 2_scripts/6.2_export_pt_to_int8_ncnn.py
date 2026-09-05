from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
from ultralytics import YOLO

import layer_replacement_adapter

MODEL_PATH = Path(r"C:\a USask\yolo_project\1_models\4_layer_replacement\LR_gen_p10_320x.pt")
CALIBRATION_DIR = Path(r"C:\a USask\yolo_project\1_data\raw\MIO-TCD-Localization\train")
NCNN_TOOLS_DIR = Path(r"C:\a USask\yolo_project\1_models\for_int8")

# 0 = use every discovered calibration image.
MAX_CALIBRATION_IMAGES = 10000
IMAGE_SIZE = 320
CALIBRATION_THREADS = 8

# Set to None to automatically create:
#   <MODEL_NAME>_int8_ncnn_model
# beside the original .pt model.
OUTPUT_DIR: Path | None = None

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def cpp_round_positive(value: float) -> int:
    """Match C++ std::round() for the non-negative values used here."""
    return int(math.floor(value + 0.5))


def require_file(path: Path, description: str) -> None:
    """Stop early instead of silently continuing with a wrong/missing file."""
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def require_executable(tools_dir: Path, name: str) -> Path:
    """Resolve one NCNN CLI tool without guessing a different tool location."""
    suffix = ".exe" if os.name == "nt" else ""
    path = tools_dir / f"{name}{suffix}"
    require_file(path, f"NCNN tool {name}")
    return path


def run_command(command: list[str]) -> None:
    """Run one conversion step and fail immediately if NCNN reports an error."""
    print("\n$ " + " ".join(command))
    subprocess.run(command, check=True)


def collect_calibration_images(directory: Path, max_images: int) -> list[Path]:
    """
    Collect calibration images deterministically.

    A stable sort makes repeated quantization runs use the same image order,
    which is useful when comparing models experimentally.
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"Calibration directory not found: {directory}")

    images = sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(f"No calibration images found under: {directory}")

    if max_images > 0:
        images = images[:max_images]

    return images


def create_letterboxed_calibration_image(
    source_path: Path,
    output_path: Path,
    imgsz: int,
) -> None:
    """
    Reproduce the current PYNQ runner's image geometry exactly.

    The runner:
      1. reads BGR with OpenCV,
      2. preserves aspect ratio,
      3. applies centered letterbox padding with value 114,
      4. converts BGR -> RGB inside NCNN,
      5. scales pixels by 1/255.

    This function performs only steps 1-3 and stores a lossless PNG.
    ncnn2table performs steps 4-5 during calibration, matching deployment.
    """
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise RuntimeError(f"OpenCV failed to read calibration image: {source_path}")

    original_height, original_width = image.shape[:2]
    scale = min(imgsz / original_width, imgsz / original_height)

    # Use C++-style rounding to match yolo26_ncnn_runner.cpp rather than
    # Python's bankers-rounding behavior at exact .5 boundaries.
    resized_width = cpp_round_positive(original_width * scale)
    resized_height = cpp_round_positive(original_height * scale)

    horizontal_padding = imgsz - resized_width
    vertical_padding = imgsz - resized_height

    pad_left = cpp_round_positive(horizontal_padding / 2.0 - 0.1)
    pad_right = cpp_round_positive(horizontal_padding / 2.0 + 0.1)
    pad_top = cpp_round_positive(vertical_padding / 2.0 - 0.1)
    pad_bottom = cpp_round_positive(vertical_padding / 2.0 + 0.1)

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_LINEAR,
    )

    letterboxed = cv2.copyMakeBorder(
        resized,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )

    if letterboxed.shape[0] != imgsz or letterboxed.shape[1] != imgsz:
        raise RuntimeError(
            "Letterbox preprocessing produced an unexpected size for "
            f"{source_path}: {letterboxed.shape[1]}x{letterboxed.shape[0]}"
        )

    # PNG is used because lossy JPEG compression would change the pixel values
    # seen during calibration and make preprocessing differ from deployment.
    if not cv2.imwrite(str(output_path), letterboxed):
        raise RuntimeError(f"Failed to write calibration image: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export an Ultralytics FP32 .pt model to FP32 NCNN, then apply "
            "NCNN post-training INT8 quantization."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_PATH,
        help="Input FP32 .pt model; overrides MODEL_PATH",
    )
    parser.add_argument(
        "--calib",
        type=Path,
        default=CALIBRATION_DIR,
        help="Calibration image directory; overrides CALIBRATION_DIR",
    )
    parser.add_argument(
        "--ncnn-tools",
        type=Path,
        default=NCNN_TOOLS_DIR,
        help="Directory containing ncnn2table/ncnn2int8; overrides NCNN_TOOLS_DIR",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
        help="Final INT8 NCNN directory; overrides OUTPUT_DIR",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help="YOLO input size; overrides IMAGE_SIZE",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=CALIBRATION_THREADS,
        help="Calibration CPU threads; overrides CALIBRATION_THREADS",
    )
    parser.add_argument(
        "--max-calib",
        type=int,
        default=MAX_CALIBRATION_IMAGES,
        help="Maximum calibration images; overrides MAX_CALIBRATION_IMAGES",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    calibration_dir = args.calib.expanduser().resolve()
    tools_dir = args.ncnn_tools.expanduser().resolve()

    require_file(model_path, "Input .pt model")

    if args.imgsz <= 0:
        raise ValueError("--imgsz must be greater than zero")
    if args.threads <= 0:
        raise ValueError("--threads must be greater than zero")
    if args.max_calib < 0:
        raise ValueError("--max-calib cannot be negative")

    ncnn2table = require_executable(tools_dir, "ncnn2table")
    ncnn2int8 = require_executable(tools_dir, "ncnn2int8")

    output_dir = (
        args.output.expanduser().resolve()
        if args.output is not None
        else model_path.parent / f"{model_path.stem}_int8_ncnn_model"
    )

    # Refuse to mix a new quantized model with an existing non-empty folder.
    # This prevents accidentally testing stale param/bin files on the PYNQ.
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration_images = collect_calibration_images(
        calibration_dir,
        args.max_calib,
    )

    print("============================================================")
    print("FP32 .pt -> INT8 NCNN")
    print("============================================================")
    print(f"Model:              {model_path}")
    print(f"Calibration folder: {calibration_dir}")
    print(f"Calibration images: {len(calibration_images)}")
    print(f"Input size:          {args.imgsz}x{args.imgsz}")
    print(f"INT8 output:         {output_dir}")

    # ------------------------------------------------------------------
    # Step 1: Export the trained PyTorch checkpoint to FP32 NCNN.
    # Ultralytics currently exports NCNN through PNNX. NCNN's own PTQ guide
    # says models converted by PNNX should skip a separate ncnnoptimize step.
    # ------------------------------------------------------------------
    print("\n[1/4] Exporting FP32 .pt -> FP32 NCNN...")
    model = YOLO(str(model_path))
    exported = Path(
        model.export(
            format="ncnn",
            imgsz=args.imgsz,
            batch=1,
            device="cpu",
            end2end=False,
        )
    ).resolve()

    fp32_param = exported / "model.ncnn.param"
    fp32_bin = exported / "model.ncnn.bin"
    metadata = exported / "metadata.yaml"

    require_file(fp32_param, "Exported FP32 NCNN param")
    require_file(fp32_bin, "Exported FP32 NCNN bin")
    require_file(metadata, "Exported Ultralytics metadata")

    with tempfile.TemporaryDirectory(prefix="yolo_ncnn_int8_") as temporary:
        temp_dir = Path(temporary)
        preprocessed_dir = temp_dir / "calibration_png"
        preprocessed_dir.mkdir()
        calibration_list = temp_dir / "calibration_images.txt"
        calibration_table = temp_dir / "model.table"
        int8_param_temp = temp_dir / "model-int8.param"
        int8_bin_temp = temp_dir / "model-int8.bin"

        # --------------------------------------------------------------
        # Step 2: Reproduce the PYNQ runner's letterbox preprocessing.
        # The resulting images remain BGR PNGs. ncnn2table will perform the
        # same BGR->RGB conversion and 1/255 normalization as the runner.
        # --------------------------------------------------------------
        print("\n[2/4] Preparing calibration images...")
        preprocessed_paths: list[Path] = []

        for index, source in enumerate(calibration_images, start=1):
            destination = preprocessed_dir / f"{index:06d}.png"
            create_letterboxed_calibration_image(
                source,
                destination,
                args.imgsz,
            )
            preprocessed_paths.append(destination)

            if index == 1 or index % 100 == 0 or index == len(calibration_images):
                print(
                    f"  Prepared {index}/{len(calibration_images)} calibration images",
                    end="\r" if index != len(calibration_images) else "\n",
                )

        calibration_list.write_text(
            "\n".join(str(path) for path in preprocessed_paths) + "\n",
            encoding="utf-8",
        )

        # --------------------------------------------------------------
        # Step 3: Measure activation ranges and produce an INT8 scale table.
        # mean=0 and norm=1/255 match input.substract_mean_normalize(nullptr,
        # normalization) in the current yolo26_ncnn_runner.cpp.
        # pixel=RGB makes ncnn2table convert OpenCV's BGR pixels to RGB.
        # --------------------------------------------------------------
        print("\n[3/4] Creating NCNN INT8 calibration table...")
        run_command(
            [
                str(ncnn2table),
                str(fp32_param),
                str(fp32_bin),
                str(calibration_list),
                str(calibration_table),
                "mean=[0,0,0]",
                "norm=[0.003921568627,0.003921568627,0.003921568627]",
                f"shape=[{args.imgsz},{args.imgsz},3]",
                "pixel=RGB",
                f"thread={args.threads}",
                "method=kl",
            ]
        )
        require_file(calibration_table, "Generated INT8 calibration table")

        # --------------------------------------------------------------
        # Step 4: Convert the FP32 NCNN weights/graph using the scale table.
        # --------------------------------------------------------------
        print("\n[4/4] Quantizing FP32 NCNN -> INT8 NCNN...")
        run_command(
            [
                str(ncnn2int8),
                str(fp32_param),
                str(fp32_bin),
                str(int8_param_temp),
                str(int8_bin_temp),
                str(calibration_table),
            ]
        )

        require_file(int8_param_temp, "Generated INT8 NCNN param")
        require_file(int8_bin_temp, "Generated INT8 NCNN bin")

        # Keep the filenames expected by the existing PYNQ runner, so no C++
        # model-loading change is required for the INT8 test.
        shutil.copy2(int8_param_temp, output_dir / "model.ncnn.param")
        shutil.copy2(int8_bin_temp, output_dir / "model.ncnn.bin")
        shutil.copy2(metadata, output_dir / "metadata.yaml")
        shutil.copy2(calibration_table, output_dir / "model.int8.table")

    final_param = output_dir / "model.ncnn.param"
    final_bin = output_dir / "model.ncnn.bin"

    print("\n============================================================")
    print("INT8 NCNN conversion completed")
    print("============================================================")
    print(f"Final directory: {output_dir}")
    print(f"Param:           {final_param}")
    print(f"Bin:             {final_bin}")
    print(f"Metadata:        {output_dir / 'metadata.yaml'}")
    print(f"Calibration:     {output_dir / 'model.int8.table'}")
    print()
    print("Model size:")
    print(f"  FP32 NCNN bin: {fp32_bin.stat().st_size / (1024 * 1024):.3f} MiB")
    print(f"  INT8 NCNN bin: {final_bin.stat().st_size / (1024 * 1024):.3f} MiB")
    print()
    print("The final folder uses model.ncnn.param/model.ncnn.bin so your")
    print("current PYNQ runner can load it without changing its filenames.")


if __name__ == "__main__":
    main()
