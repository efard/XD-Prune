"""
Generate an ASAP YOLO26n original baseline package.

This script is intentionally simple because the ASAP goal is to unblock the
hardware-side work first. It prepares the original YOLO26n model, ONNX export,
model summary, and layer table.

Generated files:
- models/yolo26n_original.pt
- models/yolo26n_original.onnx
- results/yolo26n_model_summary.txt
- results/yolo26n_layer_table.csv
- results/package_manifest.txt
- README_ASAP_reproduce.md

Important:
- This package is for the original YOLO26n baseline.
- Do not use pruned, quantized, error-injected, or manually modified models here.
"""

from pathlib import Path
from contextlib import redirect_stdout
import argparse
import csv
import shutil

from ultralytics import YOLO


parser = argparse.ArgumentParser(
    description="Generate ASAP YOLO26n original baseline package."
)

parser.add_argument(
    "--model",
    type=str,
    default="yolo26n.pt",
    help="Original YOLO26n model name or path. Default: yolo26n.pt"
)

parser.add_argument(
    "--imgsz",
    type=int,
    default=640,
    help="Image size used for ONNX export. Default: 640"
)

parser.add_argument(
    "--out-dir",
    type=str,
    default="YOLO26n_ASAP_baseline_package",
    help="Output folder for the ASAP package."
)

args = parser.parse_args()

out_dir = Path(args.out_dir)
models_dir = out_dir / "models"
results_dir = out_dir / "results"

# Create package folders.
# These folders only store generated output files and do not change model logic.
models_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

# Load original YOLO26n.
# If yolo26n.pt is not already present, Ultralytics will download it.
model = YOLO(args.model)

source_pt = Path(args.model)

# Stop immediately if the expected original PT file cannot be found.
# This avoids silently packaging the wrong model.
if not source_pt.exists():
    raise FileNotFoundError(
        f"Cannot find the model file after loading: {source_pt}. "
        "Check whether the model name/path is correct."
    )

# Copy the original YOLO26n PT file into the package.
# This makes the package self-contained for the hardware-side collaborator.
pt_out = models_dir / "yolo26n_original.pt"
shutil.copy2(source_pt, pt_out)

# Reload from the copied package file.
# This ensures the ONNX export is produced from the exact PT file being shared.
model = YOLO(str(pt_out))

# Export ONNX for hardware-side inspection and profiling.
# ONNX is useful for checking operators, layer structure, and deployment flow.
onnx_generated = Path(
    model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        simplify=False
    )
)

onnx_out = models_dir / "yolo26n_original.onnx"

# Move or rename the exported ONNX into the expected package name.
# This keeps the package file names clear and consistent.
if onnx_generated.resolve() != onnx_out.resolve():
    shutil.move(str(onnx_generated), str(onnx_out))

# Save model summary.
# This helps the hardware side quickly inspect the full YOLO26n structure.
summary_path = results_dir / "yolo26n_model_summary.txt"

with summary_path.open("w", encoding="utf-8") as f:
    with redirect_stdout(f):
        print("YOLO26n Original Model Summary")
        print("=" * 40)
        print("")
        print(model.model)
        print("")
        model.info(verbose=True)

# Save layer table.
# Layer ID and layer type are needed before selecting C3k2, Detect-head,
# or other hardware bottleneck blocks.
layer_table_path = results_dir / "yolo26n_layer_table.csv"

with layer_table_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["layer_id", "layer_class", "module_text"])

    for layer_id, layer in enumerate(model.model.model):
        writer.writerow([
            layer_id,
            layer.__class__.__name__,
            str(layer).replace("\n", " ")
        ])

# Save a short README.
# This gives him enough instruction to reproduce the ASAP package.
readme_path = out_dir / "README_ASAP_reproduce.md"

with readme_path.open("w", encoding="utf-8") as f:
    f.write("# YOLO26n ASAP Original Baseline Package\n\n")
    f.write("This package contains the original YOLO26n baseline files needed to start hardware-side inspection.\n\n")

    f.write("## Generated files\n\n")
    f.write("- `models/yolo26n_original.pt`\n")
    f.write("- `models/yolo26n_original.onnx`\n")
    f.write("- `results/yolo26n_model_summary.txt`\n")
    f.write("- `results/yolo26n_layer_table.csv`\n")
    f.write("- `results/package_manifest.txt`\n\n")

    f.write("## Reproduce this package\n\n")
    f.write("```bash\n")
    f.write("python generate_yolo26n_asap_package.py --model yolo26n.pt --imgsz 640\n")
    f.write("```\n\n")

    f.write("## Important note\n\n")
    f.write("These files are for the original YOLO26n baseline only. ")
    f.write("They are not pruned, quantized, error-injected, or manually modified.\n")

# Save manifest.
# The manifest is a quick checklist of what was generated.
manifest_path = results_dir / "package_manifest.txt"

with manifest_path.open("w", encoding="utf-8") as f:
    f.write("YOLO26n ASAP Baseline Package Manifest\n")
    f.write("=" * 44 + "\n\n")
    f.write(f"Input model: {args.model}\n")
    f.write(f"Image size: {args.imgsz}\n\n")
    f.write(f"PT file: {pt_out}\n")
    f.write(f"ONNX file: {onnx_out}\n")
    f.write(f"Model summary: {summary_path}\n")
    f.write(f"Layer table: {layer_table_path}\n")
    f.write(f"README: {readme_path}\n")

print(f"ASAP package generated at: {out_dir}")
print(f"Share this folder first: {out_dir}")