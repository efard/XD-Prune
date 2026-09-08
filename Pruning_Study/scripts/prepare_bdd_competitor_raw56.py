"""Build a structural BDD raw-T5 plan for the finalized L1 or FPGM competitor.

The V2 ranking freezes the group order.  At each cumulative step, this runner
recomputes the method's channel/unit choice on the live model, then applies the
selection through DepGraph (generic groups) or the validated custom rule.
This is necessary because earlier structural operations may change later live
tensor widths.  No validation, fine-tuning, BatchNorm update, or test data is
used here.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
import types
from typing import Any

SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT.parent
RANKER_PATH = STUDY_ROOT / "scripts" / "rank_bdd_l1_fpgm_competitors.py"
OUTPUT_BASE = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_competitors_raw56_v1"
DOMAINS = ("GEN2", "NGN2")
METHODS = ("global_l1", "fpgm")
BASELINE_PARAMETERS = 2_506_140
TARGET_REDUCTION = 0.56


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ranker = load_module("_bdd_competitor_ranker", RANKER_PATH)
base = ranker.base
engine = ranker.engine


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    return base.relative(path)


def ranking_root(method: str) -> Path:
    return ranker.output_root(method)


def output_root(method: str) -> Path:
    return OUTPUT_BASE / f"{method}_37_5pct_raw56"


def queue(method: str) -> list[dict[str, Any]]:
    path = ranking_root(method) / "shared_cumulative_queue.json"
    manifest = ranking_root(method) / "ranking_manifest.json"
    if not path.is_file() or not manifest.is_file():
        raise FileNotFoundError(f"Finalized V2 ranking is missing for {method}")
    if base.read_json(manifest).get("status") != "PASS":
        raise RuntimeError(f"Finalized V2 ranking did not pass for {method}")
    entries = list(base.read_json(path).get("entries", ()))
    if len(entries) != 49 or len({entry["group_id"] for entry in entries}) != 49:
        raise RuntimeError(f"{method}: expected 49 unique unprotected ranking entries")
    entries.sort(key=lambda entry: int(entry["cumulative_queue_index"]))
    return entries


def preflight(method: str) -> dict[str, Any]:
    evidence = ranker.preflight()
    candidates = queue(method)
    return {
        "schema": "bdd_competitor_raw56_preflight_v1",
        "method": method,
        "ranking_root": relative(ranking_root(method)),
        "ranking_manifest_sha256": base.sha256(ranking_root(method) / "ranking_manifest.json"),
        "ranking_queue_sha256": base.sha256(ranking_root(method) / "shared_cumulative_queue.json"),
        "candidate_queue": [entry["group_id"] for entry in candidates],
        "candidate_count": len(candidates),
        "target_parameter_reduction_percent": 100.0 * TARGET_REDUCTION,
        "ranking_preflight": evidence,
    }


def apply_generic(model: Any, group_id: str, selected: list[int]) -> dict[str, Any]:
    import torch
    import torch_pruning as tp

    row = engine.generic_rows()[group_id]
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: representative root is not Conv2d")
    width = int(root.out_channels)
    if not selected or min(selected) < 0 or max(selected) >= width:
        raise RuntimeError(f"{group_id}: live selected output channels are invalid")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    head = model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("Expected a YOLO26 end-to-end Detect head")
    head.forward = types.MethodType(base.trace_detect_forward, head)

    class TraceWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__(); self.inner = inner
        def forward(self, images: Any) -> tuple[Any, ...]:
            values = tuple(base.flatten_tensors(self.inner(images)))
            if not values or not all(value.requires_grad for value in values):
                raise RuntimeError("Trace outputs lost Autograd dependencies")
            return values
    try:
        graph = tp.DependencyGraph().build_dependency(TraceWrapper(model), example_inputs=torch.zeros(1, 3, engine.TRACE_SIZE, engine.TRACE_SIZE))
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=selected)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"DepGraph rejected {group_id} at the live cumulative width")
        group.prune()
    finally:
        if "forward" in head.__dict__: delattr(head, "forward")
    model.eval(); model.zero_grad(set_to_none=True)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    return {"rule_family": "GENERIC_DEPGRAPH", "representative_root": row["representative_root"], "selection_unit": "output_channel", "root_channels_before": width, "root_channels_removed": len(selected), "root_channels_after": int(root.out_channels), "selected_indices": selected}


def apply_custom(model: Any, group_id: str, selected: list[int]) -> dict[str, Any]:
    row = engine.custom_rows()[group_id]
    block_index = int(row["block_index"]); block = model.model[block_index]; before = int(block.c)
    if block_index in engine.custom.NONATTENTION_BLOCKS:
        engine.validate_nonattention_c3k2_invariants(block)
        result = engine.prune_nonattention_c3k2_logical_channels(block, selected, module_path=f"model.{block_index}")
        unit = "hidden_channel"
    elif block_index == 10:
        engine.validate_c2psa_invariants(block)
        result = engine.prune_c2psa_head_aware_units(block, selected, module_path=f"model.{block_index}")
        unit = "paired_attention_unit"
    elif block_index == 22:
        engine.validate_attention_c3k2_invariants(block)
        result = engine.prune_attention_c3k2_head_aware_units(block, selected, module_path=f"model.{block_index}")
        unit = "paired_attention_unit"
    else:
        raise ValueError(f"Unsupported custom group: {group_id}")
    return {"rule_family": row["rule_family"], "representative_root": row["block_path"], "selection_unit": unit, "root_channels_before": before, "root_channels_removed": result.hidden_channels_removed, "root_channels_after": result.hidden_channels_after, "selected_indices": selected, "operations": result.to_dict()["operations"]}


def run_step(method: str, domain: str, index: int, group_id: str, input_model: Path | None, root: Path) -> dict[str, Any]:
    import torch
    from ultralytics import YOLO

    record_path = root / "records" / f"{domain}_step{index:02d}_{group_id}.json"
    model_path = root / "models" / f"{domain}_step{index:02d}_{group_id}.pth"
    config = ranker.bdd.domain_config(domain); checkpoint = PROJECT_ROOT / config["path"]
    record: dict[str, Any] = {"schema": "bdd_competitor_raw56_step_v1", "method": method, "domain": domain, "step": index, "group_id": group_id, "status": "FAIL", "selection_policy": "live cumulative re-selection from frozen method queue"}
    model: Any | None = None
    try:
        # ``set_num_interop_threads`` may only be called once per process;
        # this runner handles many paired steps in one process.
        torch.set_num_threads(1)
        if input_model is None:
            model, input_path = YOLO(str(checkpoint), task="detect").model.float().cpu().eval(), checkpoint
        else:
            model, input_path = torch.load(input_model, map_location="cpu", weights_only=False).float().cpu().eval(), input_model
        params_before = sum(parameter.numel() for parameter in model.parameters())
        with torch.inference_mode(): before = model(torch.zeros(1, 3, 640, 640)); public_before = base.public_prediction_summary(before)
        selection = ranker.score_group(model, group_id, method)
        selected = [int(value) for value in selection["selected_indices"]]
        intervention = apply_custom(model, group_id, selected) if group_id.startswith("CDG") else apply_generic(model, group_id, selected)
        params_after = sum(parameter.numel() for parameter in model.parameters())
        if params_after >= params_before: raise RuntimeError("Cumulative step did not reduce parameters")
        with torch.inference_mode(): after = model(torch.zeros(1, 3, 640, 640)); native = base.output_summary(after); public_after = base.public_prediction_summary(after)
        if not native["all_finite"] or public_after["shapes"] != public_before["shapes"]: raise RuntimeError("Public output contract changed")
        model_path.parent.mkdir(parents=True, exist_ok=True); temporary = model_path.with_suffix(".tmp"); torch.save(model, temporary); temporary.replace(model_path)
        reloaded = torch.load(model_path, map_location="cpu", weights_only=False).float().cpu().eval()
        with torch.inference_mode(): reloaded_output = reloaded(torch.zeros(1, 3, 640, 640))
        comparison = engine.tensor_comparison(after, reloaded_output)
        if not comparison["exact"]: raise RuntimeError("Saved/reloaded model differs")
        record.update({"status": "PASS", "input_model": relative(input_path), "input_sha256": base.sha256(input_path), "output_model": relative(model_path), "output_sha256": base.sha256(model_path), "selection": selection, "intervention": intervention, "structure": {"parameters_before": params_before, "parameters_after": params_after, "incremental_parameters_removed": params_before - params_after, "cumulative_parameter_reduction_percent": 100.0 * (1.0 - params_after / BASELINE_PARAMETERS), "native_output_after": native, "save_reload_comparison": comparison}})
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        atomic_json(record_path, record)
        try: del model
        except UnboundLocalError: pass
    if record["status"] != "PASS": raise RuntimeError(f"{method} {domain} {group_id} failed; inspect {record_path}")
    return record


def restore(root: Path, candidates: list[dict[str, Any]]) -> tuple[int, dict[str, Path | None], dict[str, int]]:
    current: dict[str, Path | None] = {domain: None for domain in DOMAINS}; parameters = {domain: BASELINE_PARAMETERS for domain in DOMAINS}
    for index, candidate in enumerate(candidates, start=1):
        records = [root / "records" / f"{domain}_step{index:02d}_{candidate['group_id']}.json" for domain in DOMAINS]
        if not any(path.is_file() for path in records): return index, current, parameters
        if not all(path.is_file() for path in records): raise RuntimeError(f"Incomplete paired step {index}; do not resume it automatically")
        for domain, path in zip(DOMAINS, records):
            record = base.read_json(path)
            if record.get("status") != "PASS": raise RuntimeError(f"Failed existing step: {path}")
            current[domain] = PROJECT_ROOT / record["output_model"]; parameters[domain] = int(record["structure"]["parameters_after"])
    return len(candidates) + 1, current, parameters


def run(method: str, resume: bool, max_steps: int | None) -> int:
    root = output_root(method); manifest_path = root / "experiment_manifest.json"; evidence = preflight(method); candidates = queue(method)
    if max_steps is not None and not 1 <= max_steps <= len(candidates):
        raise ValueError(f"--max-steps must be between 1 and {len(candidates)}")
    if manifest_path.is_file() and not resume: raise RuntimeError(f"Output exists: {root}. Use --resume; no evidence is overwritten.")
    if manifest_path.is_file(): manifest = base.read_json(manifest_path)
    else:
        manifest = {**evidence, "schema": "bdd_competitor_raw56_manifest_v1", "status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": relative(SELF), "script_sha256": base.sha256(SELF), "policy": {"ranking_queue_frozen": True, "live_channel_selection_recomputed_each_cumulative_step": True, "validation_data_used": False, "fine_tuning": False, "batchnorm_update": False, "structural_acceptance": "parameter decrease, finite output, public output shape, exact save/reload"}}
        atomic_json(manifest_path, manifest)
    start, current, parameters = restore(root, candidates)
    for index, candidate in enumerate(candidates[start-1:], start=start):
        if max_steps is not None and index > max_steps:
            manifest.update({"status": "SMOKE_PASS", "target_reached": False, "completed_steps": index - 1})
            atomic_json(manifest_path, manifest)
            print(f"Stopped after requested smoke limit of {max_steps} paired steps.")
            return 0
        group_id = candidate["group_id"]; print(f"[{index}/49] {method} {group_id}", flush=True)
        for domain in DOMAINS:
            record = run_step(method, domain, index, group_id, current[domain], root)
            current[domain] = PROJECT_ROOT / record["output_model"]; parameters[domain] = int(record["structure"]["parameters_after"])
        manifest["completed_steps"] = index; manifest["current_parameter_reduction_percent"] = {domain: 100.0 * (1.0 - parameters[domain] / BASELINE_PARAMETERS) for domain in DOMAINS}; atomic_json(manifest_path, manifest)
        if all(1.0 - parameters[domain] / BASELINE_PARAMETERS >= TARGET_REDUCTION for domain in DOMAINS):
            final = {}
            for domain in DOMAINS:
                target = root / "T5" / "models" / "final" / f"{domain}_{method}_raw_ranked_56pct.pth"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(current[domain], target)
                final[domain] = {"model": relative(target), "sha256": base.sha256(target), "parameters_before": BASELINE_PARAMETERS, "parameters_after": parameters[domain], "parameter_reduction_percent": 100.0 * (1.0 - parameters[domain] / BASELINE_PARAMETERS)}
            manifest.update({"status": "PASS", "target_reached": True, "accepted_groups": [item["group_id"] for item in candidates[:index]], "final_models": final, "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}); atomic_json(manifest_path, manifest); print(json.dumps(manifest["current_parameter_reduction_percent"], sort_keys=True)); return 0
    manifest.update({"status": "NOT_REACHED", "target_reached": False}); atomic_json(manifest_path, manifest); return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--method", choices=METHODS, required=True); parser.add_argument("--max-steps", type=int, help="Stop after this many paired cumulative steps; use 1 for a structural smoke test."); action = parser.add_mutually_exclusive_group(required=True); action.add_argument("--preflight", action="store_true"); action.add_argument("--run", action="store_true"); action.add_argument("--resume", action="store_true"); args = parser.parse_args()
    if args.preflight: print(json.dumps(preflight(args.method), indent=2, sort_keys=True)); return 0
    return run(args.method, resume=args.resume, max_steps=args.max_steps)


if __name__ == "__main__": raise SystemExit(main())
