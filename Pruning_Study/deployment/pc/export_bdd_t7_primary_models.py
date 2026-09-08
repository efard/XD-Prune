"""Export the primary BDD T7 models to traceable FP32 NCNN board packages.

This preserves each original checkpoint and writes all generated artifacts to a
new versioned deployment directory.  The recovered structured-pruning models
are stored as whole-model ``.pth`` files, so they are first packaged into a
standard Ultralytics checkpoint before using the maintained NCNN exporter.

The primary hardware comparison is intentionally limited to the matching
baseline, proposed dependency-aware method, Global-L1, and FPGM for GEN2 and
NGN2. Formula-R4 is an ablation and is not included in the primary board table.
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
PROJECT_ROOT = STUDY_ROOT.parent
DEFAULT_OUTPUT_ROOT = (
    DEPLOYMENT_ROOT / "exported_models" / "bdd_t7_56pct_primary_fp32_v1"
)
EXPORTER = PC_ROOT / "export_ncnn.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def model_specs() -> list[dict[str, object]]:
    baseline_gen2 = (
        STUDY_ROOT
        / "results"
        / "baselines"
        / "b_gen2_bdd_clear_alltime_yolo26n_s42_v1"
        / "weights"
        / "best.pt"
    )
    baseline_ngn2 = (
        STUDY_ROOT
        / "results"
        / "baselines"
        / "b_ngn2_bdd_adverse_alltime_yolo26n_s42_v1"
        / "weights"
        / "best.pt"
    )
    recovery = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct"
    competitors = recovery / "bdd_competitors"
    return [
        {
            "model_id": "bdd_gen2_baseline",
            "domain": "GEN2",
            "method": "Unpruned baseline",
            "source": baseline_gen2,
            "baseline": baseline_gen2,
            "board_model_name": "bdd_gen2_baseline_640_fp32",
            "parameter_reduction_percent": 0.0,
            "checkpoint_kind": "ultralytics_pt",
        },
        {
            "model_id": "bdd_gen2_proposed_t7_56pct",
            "domain": "GEN2",
            "method": "Proposed dependency-aware pruning (T7, 56% target)",
            "source": recovery / "T7_bdd_gen2_70ep" / "models" / "final" / "GEN2_T7_70epoch_best.pth",
            "baseline": baseline_gen2,
            "board_model_name": "bdd_gen2_proposed_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.315449256625726,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_gen2_global_l1_t7_56pct",
            "domain": "GEN2",
            "method": "Global L1 with DepGraph propagation (T7, 56% target)",
            "source": competitors / "T7_bdd_gen2_global_l1_70ep" / "models" / "final" / "GEN2_T7_70epoch_best.pth",
            "baseline": baseline_gen2,
            "board_model_name": "bdd_gen2_global_l1_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.315449256625726,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_gen2_fpgm_t7_56pct",
            "domain": "GEN2",
            "method": "FPGM with DepGraph propagation (T7, 56% target)",
            "source": competitors / "T7_bdd_gen2_fpgm_70ep" / "models" / "final" / "GEN2_T7_70epoch_best.pth",
            "baseline": baseline_gen2,
            "board_model_name": "bdd_gen2_fpgm_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.315449256625726,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_ngn2_baseline",
            "domain": "NGN2",
            "method": "Unpruned baseline",
            "source": baseline_ngn2,
            "baseline": baseline_ngn2,
            "board_model_name": "bdd_ngn2_baseline_640_fp32",
            "parameter_reduction_percent": 0.0,
            "checkpoint_kind": "ultralytics_pt",
        },
        {
            "model_id": "bdd_ngn2_proposed_t7_56pct",
            "domain": "NGN2",
            "method": "Proposed dependency-aware pruning (T7, 56% target)",
            "source": recovery / "T7_bdd_ngn2_70ep" / "models" / "final" / "NGN2_T7_70epoch_best.pth",
            "baseline": baseline_ngn2,
            "board_model_name": "bdd_ngn2_proposed_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.315449256625726,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_ngn2_global_l1_t7_56pct",
            "domain": "NGN2",
            "method": "Global L1 with DepGraph propagation (T7, 56% target)",
            "source": competitors / "T7_bdd_ngn2_global_l1_70ep" / "models" / "final" / "NGN2_T7_70epoch_best.pth",
            "baseline": baseline_ngn2,
            "board_model_name": "bdd_ngn2_global_l1_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.315449256625726,
            "checkpoint_kind": "whole_model_pth",
        },
        {
            "model_id": "bdd_ngn2_fpgm_t7_56pct",
            "domain": "NGN2",
            "method": "FPGM with DepGraph propagation (T7, 56% target)",
            "source": competitors / "T7_bdd_ngn2_fpgm_70ep" / "models" / "final" / "NGN2_T7_70epoch_best.pth",
            "baseline": baseline_ngn2,
            "board_model_name": "bdd_ngn2_fpgm_t7_56pct_640_fp32",
            "parameter_reduction_percent": 56.315449256625726,
            "checkpoint_kind": "whole_model_pth",
        },
    ]


def package_whole_model(source: Path, baseline: Path, destination: Path, imgsz: int) -> dict[str, object]:
    """Package a pruned whole-model .pth as a standard Ultralytics checkpoint."""
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
    return {
        "parameters": parameters,
        "public_output_shape": list(public_output.shape),
        "serialized_as": relative(destination),
    }


def export_one(spec: dict[str, object], output_root: Path, package_root: Path, imgsz: int) -> dict[str, object]:
    source = Path(spec["source"])
    baseline = Path(spec["baseline"])
    name = str(spec["board_model_name"])
    final_dir = output_root / name
    if final_dir.exists():
        raise FileExistsError(f"Refusing to replace existing board package: {final_dir}")
    if not source.is_file() or not baseline.is_file():
        raise FileNotFoundError(f"Missing checkpoint for {name}: source={source}, baseline={baseline}")

    packaged_checkpoint: Path | None = None
    audit: dict[str, object] = {}
    exporter_source = source
    if spec["checkpoint_kind"] == "whole_model_pth":
        packaged_checkpoint = package_root / f"{name}.pt"
        audit = package_whole_model(source, baseline, packaged_checkpoint, imgsz)
        exporter_source = packaged_checkpoint

    subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--model",
            str(exporter_source),
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

    # Ultralytics writes a PNNX/NCNN staging folder beside its input checkpoint.
    # The three final NCNN files have already been copied into ``final_dir`` by
    # the maintained exporter, so keeping the staging graph would only pollute
    # the immutable result/checkpoint folders.  Remove only the exact staging
    # directory derived from this invocation; never touch an output bundle.
    staging_dir = exporter_source.parent / f"{exporter_source.stem}_ncnn_model"
    if staging_dir.is_dir():
        shutil.rmtree(staging_dir)

    lineage = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "BDD-100K T7 56% pruning hardware latency comparison on PYNQ-Z2 PS-side NCNN",
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
        "packaged_ultralytics_checkpoint": relative(packaged_checkpoint) if packaged_checkpoint else None,
        "packaged_ultralytics_checkpoint_sha256": sha256(packaged_checkpoint) if packaged_checkpoint else None,
        "checkpoint_kind": spec["checkpoint_kind"],
        "parameter_reduction_percent": spec["parameter_reduction_percent"],
        "package_audit": audit,
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


def write_bundle_readme(output_root: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# BDD T7 56% primary PYNQ NCNN model bundle",
        "",
        "This bundle contains FP32 NCNN exports for the primary BDD-100K hardware comparison:",
        "matching GEN2/NGN2 baselines, the proposed method, Global-L1, and FPGM.",
        "Formula-R4 is intentionally omitted because it is a formula ablation rather than a primary method comparison.",
        "",
        "These packages are for **PYNQ-Z2 ARM processing-system (PS) NCNN inference**. They do not measure FPGA programmable-logic acceleration.",
        "",
        "## Board installation",
        "",
        "Copy each named folder to `/home/xilinx/yolo26_ps/1_models/` and retain its `deployment_manifest.json` and `pruning_lineage.json`.",
        "Then run the maintained board scripts from `Pruning_Study/deployment/pynq_ps_ncnn/board`.",
        "",
        "```bash",
        "cd /home/xilinx/yolo26_ps",
        "bash 1_scripts/preflight_model.sh <board_model_name>",
        "bash 1_scripts/run_research_benchmark.sh <board_model_name> benchmark 1",
        "bash 1_scripts/run_research_benchmark.sh <board_model_name> all 0",
        "```",
        "",
        "The one-image invocation is a smoke test only. `all 0` is the frozen full measurement.",
        "",
        "## Included models",
        "",
        "| Domain | Method | Board model name | Parameter reduction |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['domain']} | {row['method']} | `{row['board_model_name']}` | {float(row['parameter_reduction_percent']):.3f}% |"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.imgsz <= 0:
        parser.error("--imgsz must be positive")

    output_root = args.output_root.resolve()
    package_root = output_root / "packaged_checkpoints"
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output bundle already exists and will not be replaced: {output_root}\n"
            "Use a new versioned --output-root for a fresh export."
        )
    if not EXPORTER.is_file():
        raise FileNotFoundError(EXPORTER)
    specs = model_specs()
    missing = [
        str(path)
        for spec in specs
        for path in (Path(spec["source"]), Path(spec["baseline"]))
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Required input checkpoints are missing: " + ", ".join(sorted(set(missing))))
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "models": [str(spec["board_model_name"]) for spec in specs]}, indent=2))
        return 0

    output_root.mkdir(parents=True)
    package_root.mkdir()
    rows = [export_one(spec, output_root, package_root, args.imgsz) for spec in specs]
    fieldnames = list(rows[0])
    with (output_root / "model_registry.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_bundle_readme(output_root, rows)
    write_json(
        output_root / "bundle_manifest.json",
        {
            "schema_version": 1,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "purpose": "Primary BDD T7 56% PS-side NCNN latency comparison",
            "software": {"python": sys.version, "platform": platform.platform()},
            "models": rows,
        },
    )
    print(json.dumps({"status": "PASS", "output_root": str(output_root), "models": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
