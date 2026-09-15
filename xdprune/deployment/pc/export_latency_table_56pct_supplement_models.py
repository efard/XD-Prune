"""Export the missing local 56%-table models as traceable FP32 NCNN packages.

This creates a new immutable bundle.  It deliberately leaves the completed
primary BDD and Isomorphic--Taylor bundles unchanged.  The bundle fills the
remaining local deployment gaps: Formula R4 for GEN2/NGN2, unrestricted GEN
Global-L1 at the 56% target, and SNOW baseline/T7 checkpoints.
"""

from __future__ import annotations

import argparse
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
PROJECT_ROOT = STUDY_ROOT
EXPORTER = PC_ROOT / "export_ncnn.py"
DEFAULT_OUTPUT_ROOT = (
    DEPLOYMENT_ROOT / "exported_models" / "latency_table_56pct_supplement_fp32_v1"
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
    baselines = STUDY_ROOT / "results" / "baselines"
    recovery = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct"
    return [
        {
            "model_id": "gen_global_l1_unrestricted_p56",
            "domain": "GEN",
            "method": "Unrestricted Global L1, 20-epoch recovery",
            "source": STUDY_ROOT
            / "experiments"
            / "competitors"
            / "global_l1_p56_reproduction"
            / "recovery"
            / "models"
            / "GEN_GlobalL1_p56_full20_best.pth",
            "baseline": baselines / "b_gen_mio_yolo26n_s42_v1" / "weights" / "best.pt",
            "board_model_name": "gen_global_l1_unrestricted_p56_640_fp32",
            "parameter_reduction_percent": 56.80996295986189,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "snow_baseline",
            "domain": "SNOW",
            "method": "Unpruned baseline",
            "source": baselines / "b_snow_acdc_yolo26n_s42_v1" / "weights" / "best.pt",
            "baseline": baselines / "b_snow_acdc_yolo26n_s42_v1" / "weights" / "best.pt",
            "board_model_name": "snow_baseline_640_fp32",
            "parameter_reduction_percent": 0.0,
            "checkpoint_kind": "ultralytics_pt",
        },
        {
            "model_id": "snow_xdprune_t7_56pct",
            "domain": "SNOW",
            "method": "XD-Prune T7 recovery",
            "source": recovery
            / "T7_snow_50ep"
            / "models"
            / "final"
            / "SNOW_T7_70epoch_best.pth",
            "baseline": baselines / "b_snow_acdc_yolo26n_s42_v1" / "weights" / "best.pt",
            "board_model_name": "snow_xdprune_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.83859078071897,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_gen2_r4_signed_t7_56pct",
            "domain": "GEN2",
            "method": "Formula R4 signed, T7 recovery",
            "source": recovery
            / "T7_bdd_gen2_r4_signed_70ep"
            / "models"
            / "final"
            / "GEN2_T7_70epoch_best.pth",
            "baseline": baselines
            / "b_gen2_bdd_clear_alltime_yolo26n_s42_v1"
            / "weights"
            / "best.pt",
            "board_model_name": "bdd_gen2_r4_signed_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.107639637051406,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_ngn2_r4_signed_t7_56pct",
            "domain": "NGN2",
            "method": "Formula R4 signed, T7 recovery",
            "source": recovery
            / "T7_bdd_ngn2_r4_signed_70ep"
            / "models"
            / "final"
            / "NGN2_T7_70epoch_best.pth",
            "baseline": baselines
            / "b_ngn2_bdd_adverse_alltime_yolo26n_s42_v1"
            / "weights"
            / "best.pt",
            "board_model_name": "bdd_ngn2_r4_signed_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.107639637051406,
            "checkpoint_kind": "whole_model_pth",
        },
    ]


def package_whole_model(source: Path, baseline: Path, destination: Path, imgsz: int) -> dict[str, object]:
    """Package a verified structured ``.pth`` model as an Ultralytics checkpoint."""
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
    name = str(spec["board_model_name"])
    final_dir = output_root / name
    if final_dir.exists():
        raise FileExistsError(f"Refusing to replace existing board package: {final_dir}")

    packaged: Path | None = None
    package_audit: dict[str, object] = {}
    export_source = source
    if spec["checkpoint_kind"] == "whole_model_pth":
        packaged = package_root / f"{name}.pt"
        package_audit = package_whole_model(source, baseline, packaged, imgsz)
        export_source = packaged

    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--model",
            str(export_source),
            "--name",
            name,
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
        final_dir / "model.ncnn.param",
        final_dir / "model.ncnn.bin",
        final_dir / "metadata.yaml",
        final_dir / "deployment_manifest.json",
    ]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError("NCNN export is incomplete: " + ", ".join(missing))

    staging_dir = export_source.parent / f"{export_source.stem}_ncnn_model"
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir)

    lineage = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Supplementary locally traceable 56% latency-table package",
        "measurement_scope": "PYNQ-Z2 ARM PS-side NCNN inference only; this package does not use FPGA programmable logic",
        "model_id": spec["model_id"],
        "domain": spec["domain"],
        "method": spec["method"],
        "input_size": imgsz,
        "precision": "fp32",
        "source_checkpoint": relative(source),
        "source_checkpoint_sha256": sha256(source),
        "reference_baseline": relative(baseline),
        "reference_baseline_sha256": sha256(baseline),
        "packaged_ultralytics_checkpoint": relative(packaged) if packaged else None,
        "packaged_ultralytics_checkpoint_sha256": sha256(packaged) if packaged else None,
        "parameter_reduction_percent": spec["parameter_reduction_percent"],
        "package_audit": package_audit,
        "ncnn_files": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in required[:3]
        },
    }
    write_json(final_dir / "pruning_lineage.json", lineage)
    return {
        "model_id": spec["model_id"],
        "domain": spec["domain"],
        "method": spec["method"],
        "board_model_name": name,
        "source_checkpoint": relative(source),
        "source_sha256": sha256(source),
        "parameter_reduction_percent": spec["parameter_reduction_percent"],
        "output_folder": relative(final_dir),
        "status": "exported",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

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
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output bundle already exists and will not be replaced: {output_root}")
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "models": [spec["board_model_name"] for spec in specs]}, indent=2))
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
        "# Supplementary 56% PYNQ NCNN model bundle\n\n"
        "This immutable bundle contains the locally traceable models missing from the "
        "existing primary and Isomorphic--Taylor BDD bundles: Formula R4 for GEN2/NGN2, "
        "unrestricted GEN Global-L1 p56, and SNOW baseline/T7.\n\n"
        "The packages are for PYNQ-Z2 ARM processing-system NCNN latency measurements "
        "only and do not use programmable logic.\n",
        encoding="utf-8",
    )
    write_json(
        output_root / "bundle_manifest.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Supplementary local 56% PYNQ PS-side NCNN latency package",
            "software": {"python": sys.version, "platform": platform.platform()},
            "models": rows,
        },
    )
    print(json.dumps({"status": "PASS", "output_root": str(output_root), "models": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
