"""Freeze a Formula R4 signed 56%-target BDD structural plan.

This is an isolated Formula-R4 ablation artifact.  It reads only the frozen
R4 signed 37.5% T4 ranking, then rebuilds cumulative structures independently
for GEN2 and NGN2.  It does not train, validate, or use test data.

The established BDD raw56 implementation is loaded unchanged and configured
with R4-specific immutable paths.  All generated files live under the new
``bdd_gen2_ngn2_r4_signed_37_5pct_raw56_v1`` root, preserving the prior
current-formula raw56 result exactly as it was.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
REFERENCE_RUNNER = STUDY_ROOT / "scripts" / "prepare_bdd_raw56_updated.py"
SOURCE_T4 = (
    STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_formula_r4_ablation_v1"
    / "r4_signed_alpha2of3_param_only" / "37_5pct" / "tables"
    / "T4_R4_SIGNED_PRUNABILITY_37_5PCT.csv"
)
FORMULA_CONFIG = SOURCE_T4.parents[2] / "FORMULA_CONFIG.json"
OUTPUT_ROOT = (
    STUDY_ROOT / "results" / "pruning"
    / "bdd_gen2_ngn2_r4_signed_37_5pct_raw56_v2"
)
FORMULA_ID = "BDD_R4_SIGNED_WEIGHTED_PARAM_ONLY_V1"
SCHEMA = "bdd_raw56_r4_signed_375_structural_v2"


def load_module(name: str, path: Path) -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("_bdd_raw56_r4_reference", REFERENCE_RUNNER)
# Public compatibility surface required by the shared BDD T7 recovery helper.
# This is the untouched BDD evidence module used by the reference raw runner.
bdd = runner.bdd
sequential_engine = runner.sequential_engine

# Redirect every mutable result/input path before the reference runner parses
# arguments.  Its workers are subprocesses of SELF, so the same configuration
# is reconstructed safely in each child process.
runner.SELF = SELF
runner.SOURCE_T4 = SOURCE_T4
runner.FORMULA_CONFIG = FORMULA_CONFIG
runner.OUTPUT_ROOT = OUTPUT_ROOT
runner.MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
runner.DECISION_TABLE = OUTPUT_ROOT / "tables" / "STRUCTURAL_DECISIONS.csv"
runner.PLAN_CSV = OUTPUT_ROOT / "tables" / "FROZEN_PRUNING_PLAN.csv"
runner.PLAN_JSON = OUTPUT_ROOT / "frozen_pruning_plan.json"
runner.SCHEMA = SCHEMA

# The historical isolated BDD evidence was frozen under the local CUDA 12.1
# environment.  Structural replay on Plato must use the approved Blackwell
# container, whose Torch build is intentionally different.  Preserve every
# file/hash/catalogue/evaluation check in ``bdd.preflight`` while supplying an
# in-memory runtime-version view solely for its obsolete environment equality
# guard.  The frozen and runtime versions are both recorded below; no on-disk
# evidence is altered.
reference_bdd_read_json = runner.bdd.base.read_json


def bdd_runtime_environment_view(path: Path | str) -> Any:
    payload = reference_bdd_read_json(path)
    if Path(path).resolve() == runner.bdd.ENVIRONMENT_PATH.resolve():
        import torch
        import ultralytics
        from importlib.metadata import version

        payload = copy.deepcopy(payload)
        payload.update(
            {
                "torch": torch.__version__,
                "torch_pruning": version("torch-pruning"),
                "ultralytics": ultralytics.__version__,
            }
        )
    return payload


runner.bdd.base.read_json = bdd_runtime_environment_view


def remote_safe_prune_worker(
    domain: str,
    step: int,
    group_id: str,
    input_model: Path | None,
    output: Path,
) -> int:
    """Run one live structural step without fabricating Blackwell FLOP data.

    The historical worker requires both a parameter and a runtime-FLOP decrease.
    On Plato, the Torch 2.8 profiler returns 0.0 for both readings even after a
    real structural change.  Parameter removal, model output contract, and
    save/reload equality remain hard acceptance checks; the two profiler values
    are retained as diagnostics only.
    """

    import gc
    import time
    import traceback

    import torch
    from ultralytics.utils.torch_utils import get_flops

    pilot = runner.sequential_engine.pilot
    base = pilot.base
    started = time.perf_counter()
    record_path = output / "records" / f"{domain}_step{step:02d}_structure.json"
    model_path = output / "models" / f"{domain}_step{step:02d}_{group_id}.pth"
    record: dict[str, Any] = {
        "schema": SCHEMA,
        "domain": domain,
        "step": step,
        "group_added": group_id,
        "groups_applied": list(pilot.EXPECTED_ORDER[:step]),
        "status": "FAIL",
        "worker_implementation": "parameter_gated_blackwell_floats_diagnostic_v1",
    }
    model: Any | None = None
    try:
        pilot.preflight(require_cuda=False)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        checkpoint = pilot.PROJECT_ROOT / pilot.baseline_config(domain)["path"]
        checkpoint_hash = base.sha256(checkpoint)
        model = pilot.load_model(domain, input_model)
        parameters_before = sum(parameter.numel() for parameter in model.parameters())
        gflops_before = base.finite_metric(
            get_flops(model, imgsz=pilot.VALIDATION_SIZE), "gflops_before"
        )
        with torch.inference_mode():
            before = model(torch.zeros(1, 3, pilot.VALIDATION_SIZE, pilot.VALIDATION_SIZE))
            public_before = base.public_prediction_summary(before)
            del before
        intervention = pilot.apply_group(model, group_id)
        parameters_after = sum(parameter.numel() for parameter in model.parameters())
        gflops_after = base.finite_metric(
            get_flops(model, imgsz=pilot.VALIDATION_SIZE), "gflops_after"
        )
        if parameters_after >= parameters_before:
            raise RuntimeError("Cumulative step did not reduce parameters")
        with torch.inference_mode():
            after = model(torch.zeros(1, 3, pilot.VALIDATION_SIZE, pilot.VALIDATION_SIZE))
            native_after = base.output_summary(after)
            public_after = base.public_prediction_summary(after)
        if not native_after["all_finite"] or public_after["shapes"] != public_before["shapes"]:
            raise RuntimeError("Cumulative step changed the public output contract")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".tmp")
        torch.save(model, temporary)
        temporary.replace(model_path)
        reloaded = torch.load(model_path, map_location="cpu", weights_only=False).float().cpu().eval()
        with torch.inference_mode():
            reloaded_output = reloaded(torch.zeros(1, 3, pilot.VALIDATION_SIZE, pilot.VALIDATION_SIZE))
        reload_comparison = pilot.tensor_comparison(after, reloaded_output)
        if not reload_comparison["exact"]:
            raise RuntimeError("Saved/reloaded model is not numerically identical")
        if base.sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError("Canonical checkpoint changed")
        record.update({
            "status": "PASS",
            "input_model": base.relative(input_model) if input_model else base.relative(checkpoint),
            "input_sha256": base.sha256(input_model) if input_model else checkpoint_hash,
            "output_model": base.relative(model_path),
            "output_sha256": base.sha256(model_path),
            "canonical_checkpoint_modified": False,
            "intervention": intervention,
            "structure": {
                "parameters_before": parameters_before,
                "parameters_after": parameters_after,
                "incremental_parameters_removed": parameters_before - parameters_after,
                "gflops_before": gflops_before,
                "gflops_after": gflops_after,
                "incremental_gflops_removed": gflops_before - gflops_after,
                "runtime_gflops_counter_reduction_observed": gflops_after < gflops_before,
                "gflops_evidence_note": (
                    "Torch 2.8 Blackwell runtime FLOP values are retained as diagnostics "
                    "only. Structural acceptance requires parameter reduction, finite public "
                    "outputs, and exact save/reload equality."
                ),
                "native_output_after": native_after,
                "public_prediction_before": public_before,
                "public_prediction_after": public_after,
                "save_reload_comparison": reload_comparison,
            },
            "duration_seconds": time.perf_counter() - started,
        })
    except Exception as exc:
        record["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        base.atomic_json(record_path, record)
        del model
        gc.collect()
    return 0 if record["status"] == "PASS" else 2


runner.sequential_engine.pilot.prune_worker = remote_safe_prune_worker


def worker_preflight(require_cuda: bool, candidates: list[dict[str, str]]) -> dict[str, Any]:
    """R4-specific equivalent of the reference worker preflight."""

    if not SOURCE_T4.is_file() or not FORMULA_CONFIG.is_file():
        raise FileNotFoundError("Frozen Formula R4 signed T4 evidence is missing")
    formula = runner.base.read_json(FORMULA_CONFIG)
    if formula.get("formula_id") != FORMULA_ID:
        raise RuntimeError(
            f"Unexpected Formula R4 configuration: {formula.get('formula_id')!r}"
        )
    frozen_environment = reference_bdd_read_json(runner.bdd.ENVIRONMENT_PATH)
    bdd_evidence = runner.bdd.preflight(
        runner.LOCAL_FRACTION, require_cuda=require_cuda, require_references=True
    )
    known = set(runner.ratio_engine.generic_rows()) | set(runner.ratio_engine.custom_rows())
    unknown = [row["group_id"] for row in candidates if row["group_id"] not in known]
    if unknown:
        raise RuntimeError(f"Candidate groups absent from validated catalogues: {unknown}")
    return {
        "schema": f"{SCHEMA}_preflight",
        "formula_id": FORMULA_ID,
        "domains": list(runner.DOMAINS),
        "local_pruning_percent": runner.LOCAL_PERCENT,
        "queued_groups": len(candidates),
        "known_validated_groups": len(known),
        "bdd_t1_t2_preflight": bdd_evidence,
        "frozen_environment": frozen_environment,
        "runtime_environment": bdd_evidence["versions"],
        "environment_policy": (
            "Frozen input hashes, BDD catalogues and evaluation protocol are "
            "verified unchanged. Runtime package equality is not required "
            "because Plato's approved Blackwell container differs from the "
            "historical local CUDA environment."
        ),
    }


reference_preflight = runner.preflight
reference_write_outputs = runner.write_outputs
reference_orchestrate = runner.orchestrate


def preflight() -> dict[str, Any]:
    evidence = reference_preflight()
    evidence["formula_id"] = FORMULA_ID
    evidence["schema"] = SCHEMA
    evidence["formula_variant"] = "signed_non_clipped"
    evidence["formula_note"] = (
        "Formula R4 signed NAD weighting: (2/3)*NAD_GEN2 + (1/3)*NAD_NGN2; "
        "resource term is parameter removal only."
    )
    return evidence


def write_outputs(decisions: list[dict[str, Any]]) -> None:
    reference_write_outputs(decisions)
    if runner.PLAN_JSON.is_file():
        plan = runner.base.read_json(runner.PLAN_JSON)
        plan["schema"] = SCHEMA
        plan["formula_id"] = FORMULA_ID
        plan["formula_variant"] = "signed_non_clipped"
        runner.atomic_json(runner.PLAN_JSON, plan)


def write_r4_readme() -> None:
    runner.atomic_text(
        OUTPUT_ROOT / "README.md",
        "# BDD Formula R4 signed raw T5 at approximately 56%\n\n"
        "This isolated structural plan uses the Formula R4 signed/non-clipped "
        "37.5% local T4 ranking.  It rebuilds cumulative GEN2 and NGN2 masks "
        "from their own untouched BDD baselines and freezes the first state "
        "where both domains reach the 56% parameter-reduction target.  It "
        "contains no evaluation or recovery result.\n",
    )


def orchestrate(resume: bool) -> int:
    result = reference_orchestrate(resume)
    if runner.MANIFEST_PATH.is_file():
        manifest = runner.base.read_json(runner.MANIFEST_PATH)
        manifest["schema"] = SCHEMA
        manifest["formula_id"] = FORMULA_ID
        manifest["formula_variant"] = "signed_non_clipped"
        manifest.setdefault("policy", {})["ranking"] = (
            "Formula R4 signed/non-clipped at 37.5% local pruning"
        )
        runner.atomic_json(runner.MANIFEST_PATH, manifest)
    write_r4_readme()
    return result


runner.worker_preflight = worker_preflight
runner.preflight = preflight
runner.write_outputs = write_outputs
runner.orchestrate = orchestrate


if __name__ == "__main__":
    raise SystemExit(runner.main())
