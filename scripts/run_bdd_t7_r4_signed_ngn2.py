"""Run the Formula R4 signed 56%-target NGN2 T7 recovery ablation.

This runner uses the Formula R4 signed/non-clipped 37.5% ranking only to
construct a new cumulative raw-T5 architecture.  It then replays that exact
frozen architecture through the established five-stage 6-6-8-10-40 epoch
recovery protocol (70 total epochs).  It writes to a new labelled experiment
folder and never overwrites the completed current-formula T7 result.

Prerequisite: ``prepare_bdd_raw56_r4_signed.py --run`` must have completed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
REFERENCE_RUNNER = STUDY_ROOT / "scripts" / "run_bdd_t7_recovery.py"
RAW_ROOT = (
    STUDY_ROOT / "results" / "pruning"
    / "bdd_gen2_ngn2_r4_signed_37_5pct_raw56_v2"
)
RAW_MANIFEST = RAW_ROOT / "experiment_manifest.json"
FROZEN_PLAN = RAW_ROOT / "frozen_pruning_plan.json"
FROZEN_PLAN_CSV = RAW_ROOT / "tables" / "FROZEN_PRUNING_PLAN.csv"
T4_PATH = (
    STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_formula_r4_ablation_v1"
    / "r4_signed_alpha2of3_param_only" / "37_5pct" / "tables"
    / "T4_R4_SIGNED_PRUNABILITY_37_5PCT.csv"
)
NATIVE_WORKER = STUDY_ROOT / "scripts" / "prepare_bdd_raw56_r4_signed.py"
RESULT_ROOT = (
    STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct"
    / "T7_bdd_ngn2_r4_signed_70ep"
)
FORMULA_ID = "BDD_R4_SIGNED_WEIGHTED_PARAM_ONLY_V1"
SCHEMA = "bdd_t7_ngn2_r4_signed_frozen_masks_6_6_8_10_40_raw56_v2"
DOMAIN = "NGN2"
STAGE_EPOCHS = (6, 6, 8, 10, 40)
STAGE_WARMUP_EPOCHS = (0.0, 0.0, 0.0, 0.0, 1.0)
# The prior current-formula T7 stage endpoints corresponded to these first
# four observed cumulative parameter reductions.  Formula R4 reaches the final
# target with a different number of groups, so its stage boundaries are mapped
# to the same resource milestones rather than an arbitrary fixed group count.
REFERENCE_STAGE_REDUCTION_MILESTONES = (
    11.990710814240224,
    26.395652278005223,
    37.211328976034864,
    44.918480212597856,
)
BASELINE_PARAMETERS = 2_506_140


def load_module(name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def frozen_schedule() -> tuple[tuple[tuple[str, ...], ...], tuple[float, ...], int]:
    """Map Formula R4's frozen groups to matched T7 resource milestones."""

    if not FROZEN_PLAN.is_file() or not FROZEN_PLAN_CSV.is_file():
        raise FileNotFoundError(
            "Formula R4 raw-T5 plan is missing. Run prepare_bdd_raw56_r4_signed.py --run first."
        )
    plan = json.loads(FROZEN_PLAN.read_text(encoding="utf-8"))
    if plan.get("formula_id") != FORMULA_ID:
        raise RuntimeError(f"Unexpected frozen-plan formula: {plan.get('formula_id')!r}")
    groups = tuple(str(entry["group_id"]) for entry in plan.get("entries", ()))
    import csv

    with FROZEN_PLAN_CSV.open("r", encoding="utf-8", newline="") as stream:
        rows = {row["group_id"]: row for row in csv.DictReader(stream)}
    if set(groups) != set(rows):
        raise RuntimeError("Frozen plan JSON and CSV contain different Formula R4 groups")

    reductions_by_group = {
        group: 100.0 * (1.0 - int(float(rows[group][f"{DOMAIN}_parameters_after"])) / BASELINE_PARAMETERS)
        for group in groups
    }
    stages: list[tuple[str, ...]] = []
    start = 0
    for target in REFERENCE_STAGE_REDUCTION_MILESTONES:
        endpoint = next(
            (
                index
                for index in range(start, len(groups))
                if reductions_by_group[groups[index]] >= target
            ),
            None,
        )
        if endpoint is None:
            raise RuntimeError(
                f"Formula R4 never reaches the required {target:.6f}% T7 milestone"
            )
        stages.append(groups[start:endpoint + 1])
        start = endpoint + 1
    if start >= len(groups):
        raise RuntimeError("Formula R4 has no remaining groups for final 40-epoch recovery stage")
    stages.append(groups[start:])

    reductions: list[float] = []
    for stage in stages:
        parameters_after = int(float(rows[stage[-1]][f"{DOMAIN}_parameters_after"]))
        reductions.append(100.0 * (1.0 - parameters_after / BASELINE_PARAMETERS))
    final_parameters = int(float(rows[stages[-1][-1]][f"{DOMAIN}_parameters_after"]))
    return tuple(stages), tuple(reductions), final_parameters


STAGE_GROUPS, STAGE_TARGET_REDUCTIONS, EXPECTED_PARAMETERS = frozen_schedule()
FROZEN_GROUP_SEQUENCE = tuple(group for stage in STAGE_GROUPS for group in stage)
STAGE_SIZES = tuple(len(stage) for stage in STAGE_GROUPS)

runner = load_module("_bdd_t7_r4_reference", REFERENCE_RUNNER)
runner.SELF = SELF
runner.RAW_ROOT = RAW_ROOT
runner.RAW_MANIFEST = RAW_MANIFEST
runner.FROZEN_PLAN = FROZEN_PLAN
runner.FROZEN_PLAN_CSV = FROZEN_PLAN_CSV
runner.T4_PATH = T4_PATH
runner.BDD_NATIVE_WORKER_PATH = NATIVE_WORKER
runner.STAGE_GROUPS = STAGE_GROUPS
runner.FROZEN_GROUP_SEQUENCE = FROZEN_GROUP_SEQUENCE
runner.STAGE_TARGET_REDUCTIONS = STAGE_TARGET_REDUCTIONS
runner.EXPECTED_PARAMETERS = EXPECTED_PARAMETERS
runner.EXPECTED_BASELINE_PARAMETERS = BASELINE_PARAMETERS
runner.EXPECTED_GROUP_COUNT = None
runner.EXPECTED_RAW_FORMULA_IDS = frozenset({FORMULA_ID})
runner.SCHEMA = SCHEMA
runner.DOMAIN = DOMAIN
runner.STAGE_EPOCHS = STAGE_EPOCHS
runner.STAGE_WARMUP_EPOCHS = STAGE_WARMUP_EPOCHS
runner.TOTAL_EPOCHS = sum(STAGE_EPOCHS)
runner.output_root = lambda domain: RESULT_ROOT

reference_build_runtime = runner.build_runtime


def build_runtime(domain: str) -> dict[str, Any]:
    if domain != DOMAIN:
        raise ValueError(f"This Formula R4 ablation is NGN2-only, not {domain}")
    context = reference_build_runtime(domain)
    reference_preflight = context["preflight"]

    def preflight(require_cuda: bool) -> dict[str, Any]:
        evidence = reference_preflight(require_cuda)
        evidence["schema"] = SCHEMA
        evidence["formula_id"] = FORMULA_ID
        evidence["formula_variant"] = "signed_non_clipped"
        evidence["formula_t4"] = str(T4_PATH)
        evidence["formula_t4_sha256"] = sha256(T4_PATH)
        evidence["stage_partition"] = list(STAGE_SIZES)
        evidence["stage_boundary_policy"] = (
            "First R4 cumulative state meeting each matched current-formula "
            "parameter-reduction milestone; final stage contains remaining groups."
        )
        return evidence

    context["schema"] = SCHEMA
    context["preflight"] = preflight
    return context


runner.build_runtime = build_runtime
reference_main = runner.main


def finalize_metadata() -> None:
    manifest_path = RESULT_ROOT / "experiment_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema"] = SCHEMA
    manifest["formula_id"] = FORMULA_ID
    manifest["formula_variant"] = "signed_non_clipped"
    manifest["formula_t4"] = str(T4_PATH.relative_to(PROJECT_ROOT))
    manifest["formula_t4_sha256"] = sha256(T4_PATH)
    manifest["raw_plan"] = str(FROZEN_PLAN.relative_to(PROJECT_ROOT))
    manifest["raw_plan_sha256"] = sha256(FROZEN_PLAN)
    atomic_json(manifest_path, manifest)

    readme = RESULT_ROOT / "README.md"
    if readme.is_file():
        readme.write_text(
            readme.read_text(encoding="utf-8")
            + "\nFormula arm: **R4 signed/non-clipped**, with sensitivity "
              "`(2/3) NAD_GEN2 + (1/3) NAD_NGN2` and parameter-only resource term. "
              "The exact 37.5% R4 cumulative masks are frozen in the linked raw plan.\n",
            encoding="utf-8",
        )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    result = reference_main()
    finalize_metadata()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
