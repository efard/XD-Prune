"""Build the public bundle of final PyTorch and NCNN paper models.

The source is the locally verified ``Latency_Results_Models/models`` tree.
Absolute workstation paths from the original deployment manifests are never
copied into the public bundle. Each output contains the standardized PyTorch
checkpoint, NCNN parameter and weight files, sanitized metadata, and a new
hash-verified publication manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import yaml


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
DEFAULT_SOURCE = STUDY_ROOT
DEFAULT_DESTINATION = STUDY_ROOT / "models" / "paper_models"

MODEL_SPECS = (
    ("baseline_gen_640_fp32", "GEN", "Unpruned baseline", "baseline"),
    ("7_t7_gen_p56_25_640_fp32", "GEN", "XD-Prune T7", "proposed"),
    ("gen_global_l1_unrestricted_p56_640_fp32", "GEN", "Global L1 unrestricted", "comparator"),
    ("snow_baseline_640_fp32", "SNOW", "Unpruned baseline", "baseline"),
    ("snow_xdprune_t7_56pct_640_fp32", "SNOW", "XD-Prune T7", "proposed"),
    ("bdd_gen2_baseline_640_fp32", "GEN2", "Unpruned baseline", "baseline"),
    ("bdd_gen2_proposed_t7_56pct_640_fp32", "GEN2", "XD-Prune", "proposed"),
    ("bdd_gen2_r4_signed_t7_56pct_640_fp32", "GEN2", "Formula R4 signed", "ablation"),
    ("bdd_gen2_global_l1_t7_56pct_640_fp32", "GEN2", "Global L1 + DepGraph", "comparator"),
    ("bdd_gen2_fpgm_t7_56pct_640_fp32", "GEN2", "FPGM + DepGraph", "comparator"),
    ("bdd_gen2_isomorphic_taylor_t7_56pct_640_fp32", "GEN2", "Isomorphic Pruning + Taylor", "comparator"),
    ("bdd_ngn2_baseline_640_fp32", "NGN2", "Unpruned baseline", "baseline"),
    ("bdd_ngn2_proposed_t7_56pct_640_fp32", "NGN2", "XD-Prune", "proposed"),
    ("bdd_ngn2_r4_signed_t7_56pct_640_fp32", "NGN2", "Formula R4 signed", "ablation"),
    ("bdd_ngn2_global_l1_t7_56pct_640_fp32", "NGN2", "Global L1 + DepGraph", "comparator"),
    ("bdd_ngn2_fpgm_t7_56pct_640_fp32", "NGN2", "FPGM + DepGraph", "comparator"),
    ("bdd_ngn2_isomorphic_taylor_t7_56pct_640_fp32", "NGN2", "Isomorphic Pruning + Taylor", "comparator"),
)

# Completed pruned-model recovery runs reported by the paper. Baselines are
# intentionally absent because they are evaluated directly, not recovered.
RECOVERY_SPECS = (
    "recovery_56pct/T7_gen_70ep",
    "recovery_56pct/T7_snow_50ep",
    "recovery_56pct/T7_bdd_gen2_70ep",
    "recovery_56pct/T7_bdd_ngn2_70ep",
    "recovery_56pct/T7_bdd_gen2_r4_signed_70ep",
    "recovery_56pct/T7_bdd_ngn2_r4_signed_70ep",
    "recovery_56pct/bdd_competitors/T7_bdd_gen2_global_l1_70ep",
    "recovery_56pct/bdd_competitors/T7_bdd_gen2_fpgm_70ep",
    "recovery_56pct/bdd_competitors/T7_bdd_gen2_isomorphic_taylor_taylor50b_70ep_v2",
    "recovery_56pct/bdd_competitors/T7_bdd_ngn2_global_l1_70ep",
    "recovery_56pct/bdd_competitors/T7_bdd_ngn2_fpgm_70ep",
    "recovery_56pct/bdd_competitors/T7_bdd_ngn2_isomorphic_taylor_taylor50b_70ep_v2",
)

RECOVERY_CHECKPOINT_RUNS = {
    "7_t7_gen_p56_25_640_fp32": "recovery_56pct/T7_gen_70ep",
    "snow_xdprune_t7_56pct_640_fp32": "recovery_56pct/T7_snow_50ep",
    "bdd_gen2_proposed_t7_56pct_640_fp32": "recovery_56pct/T7_bdd_gen2_70ep",
    "bdd_ngn2_proposed_t7_56pct_640_fp32": "recovery_56pct/T7_bdd_ngn2_70ep",
    "bdd_gen2_r4_signed_t7_56pct_640_fp32": "recovery_56pct/T7_bdd_gen2_r4_signed_70ep",
    "bdd_ngn2_r4_signed_t7_56pct_640_fp32": "recovery_56pct/T7_bdd_ngn2_r4_signed_70ep",
    "bdd_gen2_global_l1_t7_56pct_640_fp32": "recovery_56pct/bdd_competitors/T7_bdd_gen2_global_l1_70ep",
    "bdd_gen2_fpgm_t7_56pct_640_fp32": "recovery_56pct/bdd_competitors/T7_bdd_gen2_fpgm_70ep",
    "bdd_gen2_isomorphic_taylor_t7_56pct_640_fp32": "recovery_56pct/bdd_competitors/T7_bdd_gen2_isomorphic_taylor_taylor50b_70ep_v2",
    "bdd_ngn2_global_l1_t7_56pct_640_fp32": "recovery_56pct/bdd_competitors/T7_bdd_ngn2_global_l1_70ep",
    "bdd_ngn2_fpgm_t7_56pct_640_fp32": "recovery_56pct/bdd_competitors/T7_bdd_ngn2_fpgm_70ep",
    "bdd_ngn2_isomorphic_taylor_t7_56pct_640_fp32": "recovery_56pct/bdd_competitors/T7_bdd_ngn2_isomorphic_taylor_taylor50b_70ep_v2",
}

DIRECT_FINAL_CHECKPOINTS = {
    "gen_global_l1_unrestricted_p56_640_fp32": (
        "experiments/competitors/global_l1_p56_reproduction/recovery/models/GEN_GlobalL1_p56_full20_best.pth",
        "experiments/competitors/global_l1_p56_reproduction/recovery/records/GEN_full20_best_evaluation.json",
    ),
}

RECOVERY_FILENAMES = {
    "experiment_manifest.json",
    "T7_STAGE_RESULTS.csv",
    "T7_GEN_70EP_RESULTS.csv",
    "T7_SNOW_70EP_RESULTS.csv",
    "T7_GEN2_70EP_RESULTS.csv",
    "T7_NGN2_70EP_RESULTS.csv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build(source_study: Path, destination: Path) -> None:
    source_packages = source_study / "deployment" / "Latency_Results_Models" / "models"
    if not source_packages.is_dir():
        raise FileNotFoundError(f"Missing verified deployment packages: {source_packages}")
    destination.mkdir(parents=True, exist_ok=True)
    allowed_entries = {"README.md", "MODEL_INDEX.csv", *(spec[0] for spec in MODEL_SPECS)}
    unexpected = [entry for entry in destination.iterdir() if entry.name not in allowed_entries]
    if unexpected:
        raise RuntimeError(f"Destination contains unrecognized entries: {unexpected}")

    index: list[dict[str, Any]] = []
    for model_id, domain, method, role in MODEL_SPECS:
        source_dir = source_packages / model_id
        original_manifest_path = source_dir / "deployment_manifest.json"
        original_manifest = json.loads(original_manifest_path.read_text(encoding="utf-8"))
        ncnn_source_checkpoint = Path(original_manifest["source_checkpoint"])
        expected_ncnn_source_hash = str(original_manifest["source_checkpoint_sha256"]).lower()
        if not ncnn_source_checkpoint.is_file():
            raise FileNotFoundError(f"Missing NCNN source checkpoint for {model_id}: {ncnn_source_checkpoint}")
        if sha256(ncnn_source_checkpoint) != expected_ncnn_source_hash:
            raise RuntimeError(f"Source checkpoint hash mismatch for {model_id}")

        checkpoint = ncnn_source_checkpoint
        if model_id in RECOVERY_CHECKPOINT_RUNS:
            recovery_run = source_study / "experiments" / "recovery" / RECOVERY_CHECKPOINT_RUNS[model_id]
            recovery_manifest = json.loads((recovery_run / "experiment_manifest.json").read_text(encoding="utf-8"))
            final_checkpoints = list((recovery_run / "models" / "final").glob("*best.pth"))
            if len(final_checkpoints) != 1:
                raise RuntimeError(f"Expected one final-best checkpoint in {recovery_run}, found {len(final_checkpoints)}")
            checkpoint = final_checkpoints[0]
            expected_final_hash = str(recovery_manifest["final_results"]["best"]["model_sha256"]).lower()
            if sha256(checkpoint) != expected_final_hash:
                raise RuntimeError(f"Final evaluation checkpoint hash mismatch for {model_id}")
        elif model_id in DIRECT_FINAL_CHECKPOINTS:
            checkpoint_relative, evaluation_relative = DIRECT_FINAL_CHECKPOINTS[model_id]
            checkpoint = source_study / checkpoint_relative
            evaluation = json.loads((source_study / evaluation_relative).read_text(encoding="utf-8"))
            expected_final_hash = str(evaluation["model_sha256"]).lower()
            if sha256(checkpoint) != expected_final_hash:
                raise RuntimeError(f"Final evaluation checkpoint hash mismatch for {model_id}")

        output_dir = destination / model_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_checkpoint = output_dir / "model.pt"
        output_param = output_dir / "model.ncnn.param"
        output_bin = output_dir / "model.ncnn.bin"
        output_metadata = output_dir / "metadata.yaml"
        shutil.copy2(checkpoint, output_checkpoint)
        shutil.copy2(source_dir / "model.ncnn.param", output_param)
        shutil.copy2(source_dir / "model.ncnn.bin", output_bin)

        metadata = yaml.safe_load((source_dir / "metadata.yaml").read_text(encoding="utf-8"))
        metadata["description"] = f"YOLO26n paper artifact: {domain} / {method}"
        output_metadata.write_text(
            yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        files = {
            "model.pt": file_record(output_checkpoint),
            "model.ncnn.param": file_record(output_param),
            "model.ncnn.bin": file_record(output_bin),
            "metadata.yaml": file_record(output_metadata),
        }
        public_manifest = {
            "schema": "xdprune_public_paper_model_v1",
            "model_id": model_id,
            "domain": domain,
            "method": method,
            "paper_role": role,
            "input_size": int(original_manifest["input_size"]),
            "precision": original_manifest["precision"],
            "ncnn_export": {
                "end2end": bool(original_manifest["export_end2end"]),
                "torch_version": original_manifest["torch_version"],
                "ultralytics_version": original_manifest["ultralytics_version"],
                "source_checkpoint_sha256": expected_ncnn_source_hash,
            },
            "files": files,
        }
        write_json(output_dir / "artifact_manifest.json", public_manifest)
        index.append({
            "model_id": model_id,
            "domain": domain,
            "method": method,
            "paper_role": role,
            "checkpoint_bytes": files["model.pt"]["bytes"],
            "checkpoint_sha256": files["model.pt"]["sha256"],
            "ncnn_param_bytes": files["model.ncnn.param"]["bytes"],
            "ncnn_param_sha256": files["model.ncnn.param"]["sha256"],
            "ncnn_bin_bytes": files["model.ncnn.bin"]["bytes"],
            "ncnn_bin_sha256": files["model.ncnn.bin"]["sha256"],
        })

    index_path = destination / "MODEL_INDEX.csv"
    with index_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    print(json.dumps({"status": "PASS", "models": len(index), "destination": str(destination)}, indent=2))


def collect_recovery_evidence(source_study: Path) -> None:
    source_root = source_study / "experiments" / "recovery"
    destination_root = STUDY_ROOT / "experiments" / "recovery"
    copied = 0
    for relative_run in RECOVERY_SPECS:
        source_run = source_root / relative_run
        if not source_run.is_dir():
            raise FileNotFoundError(f"Missing recovery run: {source_run}")
        selected = [
            path
            for path in source_run.rglob("*")
            if path.is_file()
            and (path.name in RECOVERY_FILENAMES or path.name.endswith("_final_best_evaluation.json"))
        ]
        if len(selected) != 4:
            raise RuntimeError(f"Expected four publication evidence files in {source_run}, found {len(selected)}")
        for source_path in selected:
            relative_file = source_path.relative_to(source_run)
            destination_path = destination_root / relative_run / relative_file
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
            copied += 1
    print(json.dumps({"status": "PASS", "recovery_runs": len(RECOVERY_SPECS), "evidence_files": copied}, indent=2))


def collect_global_l1_evidence(source_study: Path) -> None:
    relative_root = Path("experiments/competitors/global_l1_p56_reproduction")
    source_root = source_study / relative_root
    destination_root = STUDY_ROOT / relative_root
    selected = (
        "experiment_manifest.json",
        "generate_global_l1_p56.py",
        "run_recovery.py",
        "raw/selection_sequence.csv",
        "recovery/records/GEN_full20_best_evaluation.json",
        "recovery/tables/GLOBAL_L1_P56_FINAL_RESULTS.csv",
    )
    for relative_file in selected:
        source_path = source_root / relative_file
        if not source_path.is_file():
            raise FileNotFoundError(f"Missing Global-L1 publication evidence: {source_path}")
        destination_path = destination_root / relative_file
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
    readme = """# Unrestricted Global L1 at approximately 56% reduction

This is the final GEN Global-L1 comparison reported by the paper. It uses the
42 eligible stock DepGraph roots, ranks channels globally by raw L1 filter
magnitude, rebuilds the dependency graph after every channel removal, applies
no per-root 50% cap, and does not use GEN/SNOW accuracy or domain information
during selection.

The compact public evidence includes the frozen experiment manifest, exact
selection sequence, generation and recovery code, final-best evaluation, and
final results table. The final checkpoint and matching NCNN export are in
`../../../models/paper_models/gen_global_l1_unrestricted_p56_640_fp32/`.

Intermediate training checkpoints, optimizer state, plots, and duplicate model
copies are intentionally excluded.
"""
    (destination_root / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"status": "PASS", "global_l1_evidence_files": len(selected) + 1}, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-study-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()
    build(args.source_study_root.resolve(), args.destination.resolve())
    collect_recovery_evidence(args.source_study_root.resolve())
    collect_global_l1_evidence(args.source_study_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
