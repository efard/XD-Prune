"""Run an auditable 70-epoch BDD T7 recovery for a frozen pruning competitor.

Supported methods are ``global_l1`` (DepGraph + global L1) and ``fpgm``
(DepGraph-constrained FPGM).  The required raw-T5 run has already selected the
channels and groups; this runner freezes that evidence and replays it exactly
through five 6-6-8-10-40 stages.  It never re-ranks or re-selects channels.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
REFERENCE_RUNNER = STUDY_ROOT / "scripts" / "run_bdd_t7_recovery.py"
RAW_HELPER = STUDY_ROOT / "scripts" / "prepare_bdd_competitor_raw56.py"
NATIVE_WORKER = STUDY_ROOT / "scripts" / "prepare_bdd_raw56_updated.py"
OUTPUT_BASE = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_competitors_raw56_v1"
RECOVERY_BASE = STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct" / "bdd_competitors"
DOMAINS = ("GEN2", "NGN2")
METHODS = ("global_l1", "fpgm")
BASELINE_PARAMETERS = 2_506_140
STAGE_EPOCHS = (6, 6, 8, 10, 40)
STAGE_WARMUP_EPOCHS = (0.0, 0.0, 0.0, 0.0, 1.0)
MILESTONES = (11.990710814240224, 26.395652278005223, 37.211328976034864, 44.918480212597856)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def raw_root(method: str) -> Path:
    return OUTPUT_BASE / f"{method}_37_5pct_raw56"


def paths(method: str) -> dict[str, Path]:
    root = raw_root(method)
    return {
        "raw_root": root,
        "raw_manifest": root / "experiment_manifest.json",
        "plan": root / "frozen_pruning_plan.json",
        "plan_csv": root / "tables" / "FROZEN_PRUNING_PLAN.csv",
    }


def result_root(method: str, domain: str) -> Path:
    return RECOVERY_BASE / f"T7_bdd_{domain.lower()}_{method}_70ep"


def load_plan(method: str) -> dict[str, Any]:
    plan_path = paths(method)["plan"]
    if not plan_path.is_file():
        raise FileNotFoundError(
            f"Missing frozen plan: {plan_path}. Run freeze_bdd_competitor_raw56_plan.py first."
        )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "bdd_competitor_raw56_frozen_plan_v1" or plan.get("method") != method:
        raise RuntimeError(f"Unexpected competitor frozen plan: {plan_path}")
    return plan


def schedule(method: str, domain: str) -> tuple[tuple[tuple[str, ...], ...], tuple[float, ...], int]:
    plan = load_plan(method)
    entries = list(plan.get("entries", ()))
    groups = tuple(str(entry["group_id"]) for entry in entries)
    if not groups or len(groups) != len(set(groups)):
        raise RuntimeError("Frozen competitor plan has no usable unique group sequence")
    reductions = [
        float(entry["domains"][domain]["cumulative_parameter_reduction_percent"])
        for entry in entries
    ]
    stages: list[tuple[str, ...]] = []
    start = 0
    for target in MILESTONES:
        endpoint = next((i for i in range(start, len(groups)) if reductions[i] >= target), None)
        if endpoint is None:
            raise RuntimeError(f"{method} {domain} never reaches the {target:.6f}% stage milestone")
        stages.append(groups[start : endpoint + 1])
        start = endpoint + 1
    if start >= len(groups):
        raise RuntimeError(f"{method} {domain} has no groups left for the final 40-epoch stage")
    stages.append(groups[start:])
    stage_reductions = tuple(reductions[sum(len(stage) for stage in stages[:i + 1]) - 1] for i in range(5))
    final_parameters = int(entries[-1]["domains"][domain]["parameters_after"])
    return tuple(stages), stage_reductions, final_parameters


def configure(method: str, domain: str) -> Any:
    if method not in METHODS or domain not in DOMAINS:
        raise ValueError("Unsupported method or domain")
    raw = paths(method)
    stages, reductions, expected_parameters = schedule(method, domain)
    groups = tuple(group for stage in stages for group in stage)
    schema = f"bdd_t7_{domain.lower()}_{method}_frozen_masks_6_6_8_10_40_raw56_v1"

    runner = load_module(f"_bdd_t7_competitor_reference_{method}_{domain}", REFERENCE_RUNNER)
    runner.SELF = SELF
    runner.RAW_ROOT = raw["raw_root"]
    runner.RAW_MANIFEST = raw["raw_manifest"]
    runner.FROZEN_PLAN = raw["plan"]
    runner.FROZEN_PLAN_CSV = raw["plan_csv"]
    runner.BDD_NATIVE_WORKER_PATH = NATIVE_WORKER
    runner.STAGE_GROUPS = stages
    runner.FROZEN_GROUP_SEQUENCE = groups
    runner.STAGE_TARGET_REDUCTIONS = reductions
    runner.EXPECTED_PARAMETERS = expected_parameters
    runner.EXPECTED_BASELINE_PARAMETERS = BASELINE_PARAMETERS
    runner.EXPECTED_GROUP_COUNT = None
    runner.EXPECTED_RAW_FORMULA_IDS = frozenset()
    runner.SCHEMA = schema
    runner.DOMAIN = domain
    runner.STAGE_EPOCHS = STAGE_EPOCHS
    runner.STAGE_WARMUP_EPOCHS = STAGE_WARMUP_EPOCHS
    runner.TOTAL_EPOCHS = sum(STAGE_EPOCHS)
    runner.output_root = lambda requested_domain: result_root(method, requested_domain)
    return runner


def build_runtime(method: str, domain: str) -> tuple[Any, dict[str, Any]]:
    runner = configure(method, domain)
    raw = paths(method)
    plan = load_plan(method)
    stages, reductions, expected_parameters = schedule(method, domain)
    groups = tuple(group for stage in stages for group in stage)
    schema = runner.SCHEMA
    root = result_root(method, domain)
    raw_final = PROJECT_ROOT / plan["final_models"][domain]["model"]
    raw_helper = load_module(f"_bdd_t7_competitor_raw_{method}_{domain}", RAW_HELPER)
    reference_context = runner.build_runtime(domain)
    base = reference_context["base"]
    legacy = reference_context["legacy"]
    template = reference_context["template"]
    bdd_native = reference_context["bdd_native"]
    config = runner.domain_inputs(base, domain)
    baseline_checkpoint = PROJECT_ROOT / config["checkpoint"]
    dataset_yaml = PROJECT_ROOT / config["dataset_yaml"]
    baseline_metrics = (
        STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t1_t2_v1"
        / "baselines" / f"{domain}.json"
    )

    def selected_queue() -> list[dict[str, Any]]:
        return [
            {"group_id": entry["group_id"], "rank_by_prunability_descending": entry["candidate_index"]}
            for entry in plan["entries"]
        ]

    def frozen_intervention(group_id: str) -> dict[str, Any]:
        entry = next((item for item in plan["entries"] if item["group_id"] == group_id), None)
        if entry is None:
            raise KeyError(group_id)
        evidence = entry["domains"][domain]
        record_path = PROJECT_ROOT / evidence["structure_record"]
        if not record_path.is_file() or sha256(record_path) != evidence["structure_record_sha256"]:
            raise RuntimeError(f"Frozen raw evidence changed for {group_id}")
        return {
            "entry": entry,
            "domain": evidence,
            "source_record": base.read_json(record_path),
            "selected": [int(value) for value in evidence["selected_indices"]],
        }

    def apply_group(model: Any, group_id: str) -> dict[str, Any]:
        frozen = frozen_intervention(group_id)
        selected = frozen["selected"]
        expected = frozen["domain"]
        if group_id.startswith("CDG"):
            intervention = raw_helper.apply_custom(model, group_id, selected)
        else:
            intervention = raw_helper.apply_generic(model, group_id, selected)
        for field in ("representative_root", "selection_unit", "root_channels_before", "root_channels_removed", "root_channels_after"):
            if str(intervention[field]) != str(expected[field]):
                raise RuntimeError(f"{group_id}: replayed {field} differs from frozen raw evidence")
        if [int(value) for value in intervention["selected_indices"]] != selected:
            raise RuntimeError(f"{group_id}: replayed selected indices differ from frozen raw evidence")
        intervention["frozen_source_record"] = expected["structure_record"]
        intervention["replay_policy"] = "exact raw-T5 frozen selected indices"
        return intervention

    def bdd_baseline_config(requested_domain: str) -> dict[str, Any]:
        if requested_domain != domain:
            raise KeyError(requested_domain)
        metrics = base.read_json(baseline_metrics)
        return {
            "path": config["checkpoint"], "dataset_yaml": config["dataset_yaml"],
            "baseline_reference": base.relative(baseline_metrics),
            "sha256": base.sha256(baseline_checkpoint),
            "dataset_yaml_sha256": base.sha256(dataset_yaml),
            "baseline_reference_sha256": base.sha256(baseline_metrics),
            "frozen_map50_95": metrics["metrics"]["map50_95"],
        }

    def preflight(require_cuda: bool) -> dict[str, Any]:
        required = (raw["raw_manifest"], raw["plan"], raw["plan_csv"], raw_final, baseline_checkpoint, dataset_yaml, baseline_metrics)
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        if require_cuda and not legacy.trainlib.torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 is required for T7 training")
        raw_manifest = base.read_json(raw["raw_manifest"])
        if raw_manifest.get("status") != "PASS" or not raw_manifest.get("target_reached") or raw_manifest.get("method") != method:
            raise RuntimeError("Completed raw competitor evidence is unavailable or mismatched")
        if tuple(raw_manifest.get("accepted_groups", ())) != groups:
            raise RuntimeError("Raw accepted-group sequence differs from frozen plan")
        if len(plan["entries"]) != len(groups):
            raise RuntimeError("Frozen plan group count is inconsistent")
        final = raw_manifest["final_models"][domain]
        if base.sha256(raw_final) != final["sha256"]:
            raise RuntimeError("Final raw competitor checkpoint hash changed")
        if legacy.model_parameters(baseline_checkpoint) != BASELINE_PARAMETERS:
            raise RuntimeError("BDD baseline parameter count changed")
        if legacy.model_parameters(raw_final) != expected_parameters:
            raise RuntimeError("Raw competitor final parameter count changed")
        for entry in plan["entries"]:
            evidence = entry["domains"][domain]
            source = PROJECT_ROOT / evidence["structure_record"]
            if not evidence["selected_indices"] or not source.is_file() or base.sha256(source) != evidence["structure_record_sha256"]:
                raise RuntimeError(f"Frozen structure evidence changed for {entry['group_id']}")
        return {
            "schema": schema, "method": method, "domain": domain, "seed": runner.SEED,
            "source_raw_manifest": base.relative(raw["raw_manifest"]), "source_raw_manifest_sha256": base.sha256(raw["raw_manifest"]),
            "frozen_plan": base.relative(raw["plan"]), "frozen_plan_sha256": base.sha256(raw["plan"]),
            "baseline_checkpoint": base.relative(baseline_checkpoint), "baseline_checkpoint_sha256": base.sha256(baseline_checkpoint),
            "dataset": base.relative(dataset_yaml), "dataset_sha256": base.sha256(dataset_yaml),
            "baseline_parameters": BASELINE_PARAMETERS, "final_parameters": expected_parameters,
            "final_parameter_reduction_percent": reductions[-1], "frozen_group_sequence": list(groups),
            "frozen_group_count": len(groups), "stage_groups": [list(stage) for stage in stages],
            "stage_partition": [len(stage) for stage in stages], "stage_target_reductions": list(reductions),
            "stage_boundary_policy": "First cumulative raw-T5 state meeting each prior T7 parameter-reduction milestone; remaining groups form the final stage.",
            "stage_epochs": list(STAGE_EPOCHS), "stage_warmup_epochs": list(STAGE_WARMUP_EPOCHS), "total_epochs": sum(STAGE_EPOCHS),
            "cuda_available": legacy.trainlib.torch.cuda.is_available(),
            "cuda_device": legacy.trainlib.torch.cuda.get_device_name(0) if legacy.trainlib.torch.cuda.is_available() else None,
            "t7_runtime_torch": legacy.trainlib.torch.__version__,
            "runtime_environment_note": "Raw T5 was created locally under its recorded environment. T7 runs in the Blackwell container and revalidates immutable masks, record hashes, and structural parameter counts.",
            "replay_policy": "Exact raw-T5 per-domain masks; no ranking or channel selection is performed in T7.",
        }

    def worker_prune(candidate_index: int, group_id: str, input_model: Path | None, output: Path) -> int:
        import torch
        from ultralytics import YOLO

        if not 1 <= candidate_index <= len(groups) or groups[candidate_index - 1] != group_id:
            raise ValueError("Worker candidate index and frozen group do not agree")
        record_path = output / "records" / f"{domain}_candidate{candidate_index:02d}_structure.json"
        model_path = output / "models" / "trials" / f"{domain}_candidate{candidate_index:02d}_{group_id}.pth"
        record: dict[str, Any] = {
            "schema": schema, "method": method, "domain": domain, "candidate_index": candidate_index,
            "group_added": group_id, "groups_applied": list(groups[:candidate_index]), "status": "FAIL",
            "worker_implementation": "competitor_exact_frozen_mask_replay_v1",
        }
        model: Any | None = None
        try:
            preflight(False)
            torch.set_num_threads(1)
            if input_model is None:
                model, input_path = YOLO(str(baseline_checkpoint), task="detect").model.float().cpu().eval(), baseline_checkpoint
            else:
                if not input_model.is_file():
                    raise FileNotFoundError(input_model)
                model, input_path = torch.load(input_model, map_location="cpu", weights_only=False).float().cpu().eval(), input_model
            parameters_before = sum(parameter.numel() for parameter in model.parameters())
            with torch.inference_mode():
                before = model(torch.zeros(1, 3, 640, 640)); public_before = base.public_prediction_summary(before)
            intervention = apply_group(model, group_id)
            parameters_after = sum(parameter.numel() for parameter in model.parameters())
            frozen = frozen_intervention(group_id)["source_record"]["structure"]
            if parameters_before != int(frozen["parameters_before"]) or parameters_after != int(frozen["parameters_after"]):
                raise RuntimeError(f"{group_id}: replayed structural parameter counts differ from frozen raw evidence")
            with torch.inference_mode():
                after = model(torch.zeros(1, 3, 640, 640)); native_after = base.output_summary(after); public_after = base.public_prediction_summary(after)
            if not native_after["all_finite"] or public_after["shapes"] != public_before["shapes"]:
                raise RuntimeError("Frozen intervention changed the public output contract")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_suffix(".tmp"); torch.save(model, temporary); temporary.replace(model_path)
            reloaded = torch.load(model_path, map_location="cpu", weights_only=False).float().cpu().eval()
            with torch.inference_mode():
                reloaded_output = reloaded(torch.zeros(1, 3, 640, 640))
            comparison = bdd_native.sequential_engine.pilot.tensor_comparison(after, reloaded_output)
            if not comparison["exact"]:
                raise RuntimeError("Saved/reloaded frozen-intervention model is not numerically identical")
            record.update({
                "status": "PASS", "input_model": base.relative(input_path), "input_sha256": base.sha256(input_path),
                "output_model": base.relative(model_path), "output_sha256": base.sha256(model_path), "intervention": intervention,
                "structure": {"parameters_before": parameters_before, "parameters_after": parameters_after,
                    "incremental_parameters_removed": parameters_before - parameters_after,
                    "frozen_cumulative_parameter_reduction_percent": float(frozen["cumulative_parameter_reduction_percent"]),
                    "native_output_after": native_after, "public_prediction_before": public_before,
                    "public_prediction_after": public_after, "save_reload_comparison": comparison},
            })
        except Exception as exc:
            record["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
            print(record["error"]["traceback"], file=sys.stderr, flush=True)
        finally:
            runner.atomic_json(base, record_path, record)
            del model
            gc.collect()
        return 0 if record["status"] == "PASS" else 2

    # The original template expects its historical raw records to have a
    # different operation format.  Reuse only its validated recovery/training
    # engine, while routing each physical pruning operation through the frozen
    # competitor evidence above.
    template.apply_group = apply_group
    template.frozen_intervention = frozen_intervention
    template.selected_queue = selected_queue
    template.RAW_FINAL_MODEL = raw_final
    legacy.selected_queue = selected_queue
    legacy.apply_group = apply_group
    legacy.baseline_config = bdd_baseline_config
    if hasattr(legacy, "prune_engine"):
        legacy.prune_engine.baseline_config = bdd_baseline_config
        if hasattr(legacy.prune_engine, "pilot"):
            legacy.prune_engine.pilot.baseline_config = bdd_baseline_config
    legacy.preflight = preflight
    context = {**reference_context, "root": root, "schema": schema, "preflight": preflight, "worker_prune": worker_prune,
               "template": template, "legacy": legacy, "base": base, "bdd_native": bdd_native,
               "baseline_checkpoint": baseline_checkpoint, "baseline_metrics": baseline_metrics}
    return runner, context


def finalize_metadata(method: str, domain: str) -> None:
    root = result_root(method, domain)
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stages, reductions, _ = schedule(method, domain)
    manifest.update({"method": method, "competitor": "DepGraph + Global L1" if method == "global_l1" else "DepGraph-constrained FPGM", "stage_partition": [len(stage) for stage in stages], "stage_target_reductions": list(reductions)})
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, default=os.environ.get("BDD_COMPETITOR_METHOD"))
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-prune", action="store_true")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.method not in METHODS:
        parser.error("--method is required (or set BDD_COMPETITOR_METHOD)")
    runner, ctx = build_runtime(args.method, args.domain)
    if args.preflight:
        print(json.dumps(ctx["preflight"](False), indent=2, sort_keys=True)); return 0
    if args.worker_prune:
        if args.candidate_index is None or not args.group or args.output_root is None:
            parser.error("worker mode requires --candidate-index, --group, and --output-root")
        return ctx["worker_prune"](args.candidate_index, args.group, args.input_model, args.output_root)
    if args.run and args.resume:
        parser.error("Choose --run or --resume, not both")
    if not args.run and not args.resume:
        parser.error("Choose --preflight, --run, or --resume")
    result = runner.orchestrate(ctx, resume=args.resume)
    finalize_metadata(args.method, args.domain)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
