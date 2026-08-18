#!/usr/bin/env python3
"""
GEN ~56% Global L1 Stage-5 pipeline.

Search:
- exact Stage 5B V2 incremental Global L1 logic
- exact eval_all_outputs DepGraph tracing
- exact protected-root checks
- same 42 GENERIC roots / 9 CUSTOM exclusions

Raw replay:
- Stage 5C deterministic replay and save/reload validation.

Recovery:
- Stage 5E exact-structure 20-epoch trainer.

Target:
- 56.4956% parameter reduction by default, matching the completed
  efficiency-greedy whole-layer-replacement experiment.

Safety:
- first try the original 50% per-root pruning limit
- if that scope cannot reach 56.4956%, automatically retry from the untouched GEN baseline
  with a 90% per-root limit while retaining at least 4 channels
- record which cap was required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
import zipfile
from pathlib import Path

import torch

import stage5b_exact_core as s5b
import stage5c_exact_core as s5c
import stage5e_exact_core as s5e


DEFAULT_PROJECT_ROOT = Path("/home/afm176/yolo_project")

DEFAULT_GEN_MODEL = (
    DEFAULT_PROJECT_ROOT
    / "5_reproduction/reproducing_files_for_hoyin/models/GEN_baseline_best.pt"
)

DEFAULT_GEN_DATA = Path(
    "/project/ko/afm176/datasets/1_data/reproduction_exact_v1/"
    "GEN_MIO_TCD_exact/dataset_GEN_local.yaml"
)

DEFAULT_T4 = (
    DEFAULT_PROJECT_ROOT
    / "5_reproduction/reproducing_files_for_hoyin/tables/"
    "T4_25pct_group_ranking.csv"
)

DEFAULT_PROTECTED = (
    DEFAULT_PROJECT_ROOT
    / "5_reproduction/reproducing_files_for_hoyin/"
    "group_definitions/protected_root_manifest.csv"
)

BASELINE_PARAMS = 2_508_090
TARGET_REDUCTION_PERCENT = 56.4956

BASELINE_MAP50_95 = 0.6414207461
BASELINE_MAP50 = 0.8300363459


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def target_parameters_for_reduction(
    baseline_params: int,
    target_reduction_percent: float,
) -> int:
    # Stage 5B searches around a parameter target and selects the closer
    # step on either side of the crossing.
    return int(
        round(
            baseline_params
            * (1.0 - target_reduction_percent / 100.0)
        )
    )


def write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )


def run_search_attempt(
    *,
    attempt_dir: Path,
    gen_model: Path,
    generic_rows,
    protected_rows,
    device: torch.device,
    imgsz: int,
    seed: int,
    target_params: int,
    maximum_root_pruning_fraction: float,
    maximum_steps: int,
):
    attempt_dir.mkdir(parents=True, exist_ok=True)

    # The exact Stage 5B core reads this module-level dictionary.
    old_target = s5b.TARGET_PARAMS["GEN"]
    s5b.TARGET_PARAMS["GEN"] = int(target_params)

    try:
        summary = s5b.run_domain(
            domain="GEN",
            checkpoint=gen_model.resolve(),
            generic_rows=generic_rows,
            protected_rows=protected_rows,
            output_dir=attempt_dir,
            imgsz=imgsz,
            device=device,
            seed=seed,
            maximum_root_pruning_fraction=(
                maximum_root_pruning_fraction
            ),
            absolute_minimum_channels=4,
            maximum_steps=maximum_steps,
        )
    finally:
        s5b.TARGET_PARAMS["GEN"] = old_target

    summary["requested_target_parameters"] = int(target_params)
    summary["requested_target_reduction_percent"] = (
        100.0
        * (BASELINE_PARAMS - target_params)
        / BASELINE_PARAMS
    )
    summary["maximum_root_pruning_fraction_used"] = (
        maximum_root_pruning_fraction
    )

    write_json(
        attempt_dir / "GEN" / "summary_with_target.json",
        summary,
    )

    return summary


def call_stage5e_main(
    *,
    raw_model: Path,
    data: Path,
    output_dir: Path,
    expected_parameters: int,
    raw_map50_95: float,
    raw_map50: float,
    imgsz: int,
    batch: int,
    workers: int,
    seed: int,
    device: str,
):
    old_expected = s5e.EXPECTED_PARAMETERS["GEN"]
    old_baseline = dict(s5e.FROZEN_BASELINES["GEN"])
    old_raw = dict(s5e.RAW_METRICS["GEN"])
    old_argv = list(sys.argv)

    s5e.EXPECTED_PARAMETERS["GEN"] = int(expected_parameters)
    s5e.FROZEN_BASELINES["GEN"] = {
        "map50_95": BASELINE_MAP50_95,
        "map50": BASELINE_MAP50,
    }
    s5e.RAW_METRICS["GEN"] = {
        "map50_95": float(raw_map50_95),
        "map50": float(raw_map50),
    }

    sys.argv = [
        "stage5e_exact_core.py",
        "--domain",
        "GEN",
        "--raw-model",
        str(raw_model.resolve()),
        "--data",
        str(data.resolve()),
        "--output-dir",
        str(output_dir.resolve()),
        "--epochs",
        "20",
        "--imgsz",
        str(imgsz),
        "--batch",
        str(batch),
        "--workers",
        str(workers),
        "--seed",
        str(seed),
        "--device",
        str(device),
    ]

    try:
        s5e.main()
    finally:
        sys.argv = old_argv
        s5e.EXPECTED_PARAMETERS["GEN"] = old_expected
        s5e.FROZEN_BASELINES["GEN"] = old_baseline
        s5e.RAW_METRICS["GEN"] = old_raw


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
    )
    parser.add_argument(
        "--gen-model",
        type=Path,
        default=DEFAULT_GEN_MODEL,
    )
    parser.add_argument(
        "--gen-data",
        type=Path,
        default=DEFAULT_GEN_DATA,
    )
    parser.add_argument(
        "--t4",
        type=Path,
        default=DEFAULT_T4,
    )
    parser.add_argument(
        "--protected-manifest",
        type=Path,
        default=DEFAULT_PROTECTED,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--target-percent",
        type=float,
        default=TARGET_REDUCTION_PERCENT,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--device",
        default="0",
    )
    args = parser.parse_args()

    required = [
        ("GEN baseline", args.gen_model),
        ("GEN data", args.gen_data),
        ("T4 table", args.t4),
        ("protected manifest", args.protected_manifest),
    ]
    for label, path in required:
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda:0"
        if args.device != "cpu" and torch.cuda.is_available()
        else "cpu"
    )

    target_params = target_parameters_for_reduction(
        BASELINE_PARAMS,
        args.target_percent,
    )

    print("===== GEN ~56% GLOBAL L1 EXACT PIPELINE =====")
    print(f"Baseline params:          {BASELINE_PARAMS:,}")
    print(f"Requested reduction:     {args.target_percent:.6f}%")
    print(f"Search target params:    {target_params:,}")
    print()

    t4_rows = s5b.read_csv(args.t4)
    protected_rows = s5b.read_csv(args.protected_manifest)

    generic_rows = [
        row for row in t4_rows
        if row["group_kind"] == "GENERIC"
    ]
    custom_rows = [
        row for row in t4_rows
        if row["group_kind"] == "CUSTOM"
    ]

    if len(generic_rows) != 42 or len(custom_rows) != 9:
        raise RuntimeError(
            f"Unexpected T4 scope: generic={len(generic_rows)}, "
            f"custom={len(custom_rows)}"
        )

    protocol = {
        "method": (
            "Exact Stage 5B V2 dependency-aware incremental Global L1 "
            "channel-magnitude structured pruning"
        ),
        "target_reduction_percent": args.target_percent,
        "target_parameters": target_params,
        "generic_roots_included": 42,
        "custom_roots_excluded": 9,
        "protected_manifest_rows": len(protected_rows),
        "trace_mode": (
            "exact successful Stage5B V2: model.eval(), "
            "requires_grad input, all_tensor_output_transform"
        ),
        "root_cap_policy": (
            "try 50% per-root first; if target is unreachable, "
            "automatically restart from baseline with 90% per-root cap"
        ),
        "absolute_minimum_channels": 4,
        "recovery_epochs": 20,
    }
    write_json(output_dir / "protocol.json", protocol)

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    print("\n===== SEARCH ATTEMPT 1: ORIGINAL 50% PER-ROOT CAP =====")

    search50 = output_dir / "search_cap50"
    summary50 = run_search_attempt(
        attempt_dir=search50,
        gen_model=args.gen_model,
        generic_rows=generic_rows,
        protected_rows=protected_rows,
        device=device,
        imgsz=args.imgsz,
        seed=args.seed,
        target_params=target_params,
        maximum_root_pruning_fraction=0.50,
        maximum_steps=2500,
    )

    if summary50["target_crossed_during_search"]:
        selected_search_dir = search50
        selected_search_summary = summary50
        cap_used = 0.50
        print(
            "\n50% per-root cap reached the requested target. "
            "No aggressive fallback needed."
        )
    else:
        print(
            "\n50% per-root cap could not cross the requested target."
        )
        print(
            f"Closest reduction under 50% cap: "
            f"{summary50['chosen_parameter_reduction_percent']:.6f}%"
        )
        print(
            "\n===== SEARCH ATTEMPT 2: 90% PER-ROOT FALLBACK ====="
        )

        search90 = output_dir / "search_cap90"
        summary90 = run_search_attempt(
            attempt_dir=search90,
            gen_model=args.gen_model,
            generic_rows=generic_rows,
            protected_rows=protected_rows,
            device=device,
            imgsz=args.imgsz,
            seed=args.seed,
            target_params=target_params,
            maximum_root_pruning_fraction=0.90,
            maximum_steps=5000,
        )

        if not summary90["target_crossed_during_search"]:
            raise RuntimeError(
                "Even the 90% per-root fallback could not reach the "
                f"{args.target_percent:.4f}% target. Closest achieved: "
                f"{summary90['chosen_parameter_reduction_percent']:.4f}%."
            )

        selected_search_dir = search90
        selected_search_summary = summary90
        cap_used = 0.90

    selected_plan = (
        selected_search_dir
        / "GEN/chosen_replay_plan.csv"
    )

    if not selected_plan.is_file():
        raise RuntimeError(
            f"Selected replay plan not found: {selected_plan}"
        )

    shutil.copy2(
        selected_plan,
        output_dir / "chosen_replay_plan.csv",
    )

    selected_search_result = {
        "cap_used": cap_used,
        **selected_search_summary,
    }
    write_json(
        output_dir / "selected_search_summary.json",
        selected_search_result,
    )

    print("\n===== SELECTED GLOBAL L1 SEARCH RESULT =====")
    print(f"Per-root cap used:        {cap_used * 100:.1f}%")
    print(
        f"Chosen reduction:        "
        f"{selected_search_summary['chosen_parameter_reduction_percent']:.6f}%"
    )
    print(
        f"Chosen remaining params: "
        f"{selected_search_summary['chosen_remaining_parameters']:,}"
    )
    print(
        f"Replay steps:            "
        f"{selected_search_summary['chosen_replay_steps']}"
    )

    # ------------------------------------------------------------------
    # STAGE 5C REPLAY / RAW MODEL / RAW MAP
    # ------------------------------------------------------------------

    print("\n===== EXACT STAGE 5C REPLAY + RAW GEN VALIDATION =====")

    raw_dir = output_dir / "raw_stage5c"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Stage5C only needs the GEN summary fields from Stage5B.
    raw_result = s5c.replay_and_validate_domain(
        domain="GEN",
        checkpoint=args.gen_model.resolve(),
        plan_path=selected_plan.resolve(),
        data_yaml=args.gen_data.resolve(),
        generic_rows=generic_rows,
        protected_rows=protected_rows,
        search_summary=selected_search_summary,
        output_dir=raw_dir,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=device,
        val_device=args.device,
        seed=args.seed,
        score_tolerance=1e-5,
        forward_check_interval=25,
    )

    if raw_result["status"] != "ok":
        raise RuntimeError(
            f"Stage5C raw replay failed: {raw_result}"
        )

    raw_checkpoint = Path(
        raw_result["raw_checkpoint"]
    )

    if not raw_checkpoint.is_file():
        raise RuntimeError(
            f"Raw checkpoint missing: {raw_checkpoint}"
        )

    raw_map = float(raw_result["raw_map50_95"])
    raw_map50 = float(raw_result["raw_map50"])
    raw_params = int(raw_result["raw_parameters"])

    shutil.copy2(
        raw_checkpoint,
        output_dir / "GEN_global_L1_56pct_raw.pt",
    )

    write_json(
        output_dir / "raw_result.json",
        raw_result,
    )

    # ------------------------------------------------------------------
    # STAGE 5E 20-EPOCH RECOVERY
    # ------------------------------------------------------------------

    print("\n===== EXACT STAGE 5E 20-EPOCH RECOVERY =====")

    recovery_dir = output_dir / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    call_stage5e_main(
        raw_model=raw_checkpoint,
        data=args.gen_data,
        output_dir=recovery_dir,
        expected_parameters=raw_params,
        raw_map50_95=raw_map,
        raw_map50=raw_map50,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
    )

    recovery_summary_path = (
        recovery_dir / "domain_recovery_summary.json"
    )
    if not recovery_summary_path.is_file():
        raise RuntimeError(
            "Stage5E recovery summary was not produced."
        )

    recovery_summary = json.loads(
        recovery_summary_path.read_text(
            encoding="utf-8"
        )
    )

    if (
        recovery_summary.get("status")
        != "PASSED_FULL_20E_RECOVERY"
    ):
        raise RuntimeError(
            "Stage5E recovery did not pass: "
            f"{recovery_summary.get('status')}"
        )

    stage5e_best = (
        recovery_dir
        / "GEN_global_L1_42root_recovered_best.pt"
    )
    stage5e_last = (
        recovery_dir
        / "GEN_global_L1_42root_recovered_last.pt"
    )

    final_best = (
        output_dir
        / "GEN_global_L1_56pct_recovered_best.pt"
    )
    final_last = (
        output_dir
        / "GEN_global_L1_56pct_recovered_last.pt"
    )

    shutil.copy2(stage5e_best, final_best)
    shutil.copy2(stage5e_last, final_last)

    final_metrics = recovery_summary["final_metrics"]

    # ------------------------------------------------------------------
    # REPORT TABLE / FINAL SUMMARY / ZIP
    # ------------------------------------------------------------------

    report_csv = output_dir / "report_table.csv"
    with report_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "parameters",
                "parameter_reduction_percent",
                "map50_95",
                "map50",
                "accuracy_retention_percent",
            ],
        )
        writer.writeheader()

        writer.writerow(
            {
                "stage": "baseline",
                "parameters": BASELINE_PARAMS,
                "parameter_reduction_percent": 0.0,
                "map50_95": BASELINE_MAP50_95,
                "map50": BASELINE_MAP50,
                "accuracy_retention_percent": 100.0,
            }
        )

        writer.writerow(
            {
                "stage": "raw_global_l1_56pct",
                "parameters": raw_params,
                "parameter_reduction_percent": (
                    raw_result[
                        "parameter_reduction_percent"
                    ]
                ),
                "map50_95": raw_map,
                "map50": raw_map50,
                "accuracy_retention_percent": (
                    100.0
                    * raw_map
                    / BASELINE_MAP50_95
                ),
            }
        )

        writer.writerow(
            {
                "stage": "recovered_global_l1_56pct",
                "parameters": raw_params,
                "parameter_reduction_percent": (
                    raw_result[
                        "parameter_reduction_percent"
                    ]
                ),
                "map50_95": (
                    final_metrics[
                        "recovered_map50_95"
                    ]
                ),
                "map50": (
                    final_metrics[
                        "recovered_map50"
                    ]
                ),
                "accuracy_retention_percent": (
                    final_metrics[
                        "accuracy_retention_percent"
                    ]
                ),
            }
        )

    final_summary = {
        "status": "PASSED_GEN56_GLOBAL_L1_EXACT_PIPELINE",
        "method": protocol["method"],
        "target_reduction_percent": args.target_percent,
        "target_parameters": target_params,
        "per_root_cap_used": cap_used,
        "generic_roots": 42,
        "custom_roots_excluded": 9,
        "search": selected_search_result,
        "raw_result": raw_result,
        "recovery_status": recovery_summary["status"],
        "final_metrics": final_metrics,
        "files": {
            "raw_model": str(
                (
                    output_dir
                    / "GEN_global_L1_56pct_raw.pt"
                ).resolve()
            ),
            "recovered_best_model": str(
                final_best.resolve()
            ),
            "recovered_last_model": str(
                final_last.resolve()
            ),
            "report_table": str(
                report_csv.resolve()
            ),
            "chosen_replay_plan": str(
                (
                    output_dir
                    / "chosen_replay_plan.csv"
                ).resolve()
            ),
        },
    }

    final_summary_path = (
        output_dir / "summary.json"
    )
    write_json(
        final_summary_path,
        final_summary,
    )

    readme = output_dir / "README.txt"
    readme.write_text(
        (
            "GEN ~56% GLOBAL L1 EXACT PIPELINE\n"
            "================================\n\n"
            f"Target reduction: {args.target_percent:.6f}%\n"
            f"Actual reduction: "
            f"{raw_result['parameter_reduction_percent']:.6f}%\n"
            f"Per-root cap used: {cap_used * 100:.1f}%\n"
            f"Remaining parameters: {raw_params}\n"
            f"Raw mAP50-95: {raw_map:.9f}\n"
            f"Recovered mAP50-95: "
            f"{final_metrics['recovered_map50_95']:.9f}\n\n"
            "This run uses the exact previously successful Stage 5B V2\n"
            "search implementation, Stage 5C deterministic replay, and\n"
            "Stage 5E exact-structure recovery implementation.\n"
        ),
        encoding="utf-8",
    )

    bundle = (
        output_dir
        / "GEN_global_L1_56pct_exact_full_bundle.zip"
    )

    with zipfile.ZipFile(
        bundle,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        important_files = [
            output_dir / "GEN_global_L1_56pct_raw.pt",
            final_best,
            final_last,
            final_summary_path,
            report_csv,
            output_dir / "chosen_replay_plan.csv",
            output_dir / "selected_search_summary.json",
            output_dir / "raw_result.json",
            readme,
            recovery_dir / "train_run/results.csv",
            raw_dir / "GEN/replay_progress.csv",
            raw_dir / "GEN/final_42_root_inventory.csv",
            raw_dir / "GEN/protected_output_verification.csv",
        ]

        for file in important_files:
            if file.is_file():
                archive.write(
                    file,
                    arcname=file.name,
                )

    print("\n===== COMPLETE =====")
    print(
        json.dumps(
            final_summary,
            indent=2,
        )
    )
    print("\nDownload ZIP:")
    print(bundle)


if __name__ == "__main__":
    main()
