"""Export the completed BDD T7 Isomorphic-Taylor models as FP32 NCNN packages.

This is a standalone, immutable extension to the completed primary BDD T7
bundle.  It exports only the two controlled Isomorphic-Taylor adaptations
(GEN2 and NGN2) and does not modify the original bundle or source checkpoints.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SELF = Path(__file__).resolve()
PC_ROOT = SELF.parent
DEPLOYMENT_ROOT = SELF.parents[1]
STUDY_ROOT = SELF.parents[3]
PROJECT_ROOT = STUDY_ROOT.parent
EXPORTER = PC_ROOT / "export_ncnn.py"
DEFAULT_OUTPUT_ROOT = (
    DEPLOYMENT_ROOT / "exported_models" / "bdd_t7_56pct_isomorphic_taylor_fp32_v1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def model_specs() -> list[dict[str, object]]:
    baseline_root = STUDY_ROOT / "results" / "baselines"
    recovery_root = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct" / "bdd_competitors"
    return [
        {
            "model_id": "bdd_gen2_isomorphic_taylor_t7_56pct",
            "domain": "GEN2",
            "method": "Controlled Isomorphic-Taylor adaptation (T7, 50 Taylor batches, 56% target)",
            "source": recovery_root
            / "T7_bdd_gen2_isomorphic_taylor_taylor50b_70ep_v2"
            / "models"
            / "final"
            / "GEN2_T7_70epoch_best.pth",
            "baseline": baseline_root
            / "b_gen2_bdd_clear_alltime_yolo26n_s42_v1"
            / "weights"
            / "best.pt",
            "board_model_name": "bdd_gen2_isomorphic_taylor_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.16677440206852,
        },
        {
            "model_id": "bdd_ngn2_isomorphic_taylor_t7_56pct",
            "domain": "NGN2",
            "method": "Controlled Isomorphic-Taylor adaptation (T7, 50 Taylor batches, 56% target)",
            "source": recovery_root
            / "T7_bdd_ngn2_isomorphic_taylor_taylor50b_70ep_v2"
            / "models"
            / "final"
            / "NGN2_T7_70epoch_best.pth",
            "baseline": baseline_root
            / "b_ngn2_bdd_adverse_alltime_yolo26n_s42_v1"
            / "weights"
            / "best.pt",
            "board_model_name": "bdd_ngn2_isomorphic_taylor_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.16677440206852,
        },
    ]


def package_whole_model(source: Path, baseline: Path, destination: Path, imgsz: int) -> dict[str, object]:
    """Package a verified pruned whole-model .pth as an Ultralytics .pt file."""
    import torch
    from ultralytics import YOLO, __version__ as ultralytics_version

    raw_model = torch.load(source, map_location="cpu", weights_only=False).float().eval()
    parameters = sum(parameter.numel() for parameter in raw_model.parameters())
    input_tensor = torch.zeros(1, 3, imgsz, imgsz)
    with torch.inference_mode():
        raw_output = raw_model(input_tensor)
    public_output = raw_output[0] if isinstance(raw_output, (tuple, list)) else raw_output
    if not isinstance(public_output, torch.Tensor) or not torch.isfinite(public_output).all():
        raise RuntimeError(f"Source model failed finite-output audit: {source}")

    baseline_checkpoint = torch.load(baseline, map_location="cpu", weights_only=False)
    if not isinstance(baseline_checkpoint, dict):
        raise TypeError(f"Baseline is not an Ultralytics checkpoint dictionary: {baseline}")
    checkpoint = dict(baseline_checkpoint)
    checkpoint.update(
        {
            "date": datetime.now(timezone.utc).isoformat(),
            "version": ultralytics_version,
            "epoch": -1,
            "best_fitness": None,
            "model": raw_model,
            "ema": None,
            "updates": None,
            "optimizer": None,
            "scaler": None,
            "train_metrics": {},
            "train_results": {},
            "train_args": dict(getattr(raw_model, "args", {})),
        }
    )
    temporary = destination.with_name(destination.name + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)

    loaded = YOLO(str(destination), task="detect")
    loaded_parameters = sum(parameter.numel() for parameter in loaded.model.parameters())
    if loaded_parameters != parameters:
        raise RuntimeError(
            f"Packaging changed parameter count for {source.name}: {parameters} -> {loaded_parameters}"
        )
    with torch.inference_mode():
        loaded_output = loaded.model(input_tensor)
    loaded_public = loaded_output[0] if isinstance(loaded_output, (tuple, list)) else loaded_output
    if not isinstance(loaded_public, torch.Tensor) or not torch.isfinite(loaded_public).all():
        raise RuntimeError(f"Packaged checkpoint failed finite-output audit: {destination}")
    return {"parameters": parameters, "public_output_shape": list(public_output.shape)}


def export_one(spec: dict[str, object], output_root: Path, package_root: Path, imgsz: int) -> dict[str, object]:
    source = Path(spec["source"])
    baseline = Path(spec["baseline"])
    board_name = str(spec["board_model_name"])
    output = output_root / board_name
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing package: {output}")
    if not source.is_file() or not baseline.is_file():
        raise FileNotFoundError(f"Missing source or baseline: source={source}, baseline={baseline}")

    packaged = package_root / f"{board_name}.pt"
    audit = package_whole_model(source, baseline, packaged, imgsz)
    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--model",
            str(packaged),
            "--name",
            board_name,
            "--output-root",
            str(output_root),
            "--imgsz",
            str(imgsz),
            "--precision",
            "fp32",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )
    required = [
        output / "model.ncnn.param",
        output / "model.ncnn.bin",
        output / "metadata.yaml",
        output / "deployment_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("NCNN export is incomplete: " + ", ".join(missing))

    staging_dir = packaged.parent / f"{packaged.stem}_ncnn_model"
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir)
    lineage = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "PYNQ-Z2 PS-side NCNN latency measurement for a controlled Isomorphic-Taylor BDD T7 comparison",
        "measurement_scope": "ARM PS-side NCNN inference only; this package does not use FPGA programmable logic",
        "model_id": spec["model_id"],
        "domain": spec["domain"],
        "method": spec["method"],
        "input_size": imgsz,
        "precision": "fp32",
        "source_checkpoint": relative(source),
        "source_checkpoint_sha256": sha256(source),
        "reference_baseline": relative(baseline),
        "reference_baseline_sha256": sha256(baseline),
        "packaged_ultralytics_checkpoint": relative(packaged),
        "packaged_ultralytics_checkpoint_sha256": sha256(packaged),
        "parameter_reduction_percent": spec["parameter_reduction_percent"],
        "package_audit": audit,
        "ncnn_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required[:3]
        },
    }
    write_json(output / "pruning_lineage.json", lineage)
    return {
        "model_id": spec["model_id"],
        "domain": spec["domain"],
        "method": spec["method"],
        "board_model_name": board_name,
        "source_checkpoint": relative(source),
        "source_sha256": sha256(source),
        "parameter_reduction_percent": spec["parameter_reduction_percent"],
        "output_folder": relative(output),
        "status": "exported",
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

    output_root = args.output_root.resolve()
    specs = model_specs()
    missing = [
        str(path)
        for spec in specs
        for path in (Path(spec["source"]), Path(spec["baseline"]))
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Required checkpoints are missing: " + ", ".join(sorted(set(missing))))
    if not EXPORTER.is_file():
        raise FileNotFoundError(EXPORTER)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output bundle already exists and will not be replaced: {output_root}")
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "models": [item["board_model_name"] for item in specs]}, indent=2))
        return 0

    output_root.mkdir(parents=True)
    package_root = output_root / "packaged_checkpoints"
    package_root.mkdir()
    rows = [export_one(spec, output_root, package_root, args.imgsz) for spec in specs]
    with (output_root / "model_registry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_root / "README.md").write_text(
        "# BDD T7 Isomorphic-Taylor FP32 NCNN bundle\n\n"
        "This immutable bundle contains the completed GEN2 and NGN2 controlled "
        "Isomorphic-Taylor 70-epoch T7 models. It is for PYNQ-Z2 ARM PS-side NCNN "
        "latency measurements only; it does not use programmable logic.\n\n"
        "Install the two model folders into `/home/xilinx/yolo26_ps/1_models/`, then "
        "run the shared preflight and frozen full benchmark. Retain every result archive.\n",
        encoding="utf-8",
    )
    write_json(
        output_root / "bundle_manifest.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Controlled Isomorphic-Taylor BDD T7 56% PYNQ PS-side NCNN latency comparison",
            "software": {"python": sys.version, "platform": platform.platform()},
            "models": rows,
        },
    )
    print(json.dumps({"status": "PASS", "output_root": str(output_root), "models": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
