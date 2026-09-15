"""Replay frozen Isomorphic-Taylor masks into a paired BDD 56% raw-T5 model.

The ranking stage is deliberately one-shot, as in the Isomorphic Pruning
paper: this runner never recalculates gradients or changes masks. It only
replays the frozen, audited masks through the project's validated DepGraph and
custom C3k2/C2PSA structural operations, checking every intermediate model.
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
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
RANKER_PATH = STUDY_ROOT / "scripts" / "rank_bdd_isomorphic_taylor.py"
STRUCTURAL_PATH = STUDY_ROOT / "scripts" / "prepare_bdd_competitor_raw56.py"
# v1 derives from the two-batch integration smoke ranking and is retained as
# diagnostic evidence only.  v2 consumes the immutable 50-batch ranking.
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_isomorphic_taylor_raw56_taylor50b_v2"
DOMAINS = ("GEN2", "NGN2")
METHOD = "isomorphic_taylor"
BASELINE_PARAMETERS = 2_506_140
TARGET_REDUCTION = 56.0


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ranker = load_module("_bdd_isomorphic_taylor_ranker", RANKER_PATH)
structural = load_module("_bdd_isomorphic_taylor_structural", STRUCTURAL_PATH)
base = ranker.base
apply_generic = structural.apply_generic
apply_custom = structural.apply_custom


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty frozen-plan CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def relative(path: Path) -> str:
    return base.relative(path)


def ranking_files() -> tuple[Path, Path, dict[str, Path]]:
    manifest = ranker.OUTPUT_ROOT / "ranking_manifest.json"
    queue = ranker.OUTPUT_ROOT / "shared_cumulative_queue.json"
    masks = {domain: ranker.OUTPUT_ROOT / "masks" / f"{domain}_isomorphic_taylor_37_5pct_masks.json" for domain in DOMAINS}
    return manifest, queue, masks


def queue_and_masks() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest_path, queue_path, mask_paths = ranking_files()
    if not manifest_path.is_file() or not queue_path.is_file() or any(not path.is_file() for path in mask_paths.values()):
        raise FileNotFoundError("Completed Isomorphic-Taylor ranking/masks are required before raw replay")
    manifest = base.read_json(manifest_path)
    if manifest.get("status") != "PASS" or manifest.get("method") != METHOD:
        raise RuntimeError("Isomorphic-Taylor ranking is incomplete or has the wrong identity")
    entries = list(base.read_json(queue_path).get("entries", ()))
    entries.sort(key=lambda item: int(item["cumulative_queue_index"]))
    if len(entries) != 49 or len({item["group_id"] for item in entries}) != 49:
        raise RuntimeError("Expected 49 unique unprotected Isomorphic-Taylor candidates")
    masks = {domain: dict(base.read_json(path).get("masks", {})) for domain, path in mask_paths.items()}
    for domain, value in masks.items():
        if set(value) != set(ranker.all_group_ids()):
            raise RuntimeError(f"{domain} frozen mask set does not match the legal group catalogue")
    return entries, masks


def preflight() -> dict[str, Any]:
    entries, _ = queue_and_masks()
    manifest, queue, masks = ranking_files()
    return {
        "schema": "bdd_isomorphic_taylor_raw56_preflight_v2",
        "method": METHOD,
        "ranking_manifest": relative(manifest),
        "ranking_manifest_sha256": base.sha256(manifest),
        "ranking_queue": relative(queue),
        "ranking_queue_sha256": base.sha256(queue),
        "mask_files": {domain: {"path": relative(path), "sha256": base.sha256(path)} for domain, path in masks.items()},
        "candidate_count": len(entries),
        "target_parameter_reduction_percent": TARGET_REDUCTION,
        "one_shot_policy": "Frozen baseline Taylor masks are replayed exactly; no live gradient rescoring is performed.",
    }


def load_model(domain: str, source: Path | None) -> tuple[Any, Path]:
    import torch
    from ultralytics import YOLO

    if source is None:
        checkpoint = PROJECT_ROOT / ranker.bdd.domain_config(domain)["path"]
        return YOLO(str(checkpoint), task="detect").model.float().cpu().eval(), checkpoint
    return torch.load(source, map_location="cpu", weights_only=False).float().cpu().eval(), source


def run_step(domain: str, index: int, group_id: str, mask: dict[str, Any], source: Path | None) -> dict[str, Any]:
    import torch

    root = OUTPUT_ROOT
    record_path = root / "records" / f"{domain}_step{index:02d}_{group_id}.json"
    model_path = root / "models" / "trials" / f"{domain}_step{index:02d}_{group_id}.pth"
    record: dict[str, Any] = {"schema": "bdd_isomorphic_taylor_raw56_step_v2", "method": METHOD, "domain": domain, "step": index, "group_id": group_id, "status": "FAIL", "selection_policy": "exact one-shot frozen isomorphic-Taylor mask"}
    model: Any | None = None
    try:
        torch.set_num_threads(1)
        model, input_path = load_model(domain, source)
        parameters_before = sum(parameter.numel() for parameter in model.parameters())
        with torch.inference_mode():
            before = model(torch.zeros(1, 3, 640, 640))
            public_before = base.public_prediction_summary(before)
        selected = [int(value) for value in mask["selected_indices"]]
        intervention = structural.apply_custom(model, group_id, selected) if group_id.startswith("CDG") else structural.apply_generic(model, group_id, selected)
        parameters_after = sum(parameter.numel() for parameter in model.parameters())
        if parameters_after >= parameters_before:
            raise RuntimeError("Cumulative structural operation did not reduce parameters")
        with torch.inference_mode():
            after = model(torch.zeros(1, 3, 640, 640))
            native = base.output_summary(after)
            public_after = base.public_prediction_summary(after)
        if not native["all_finite"] or public_after["shapes"] != public_before["shapes"]:
            raise RuntimeError("Structural operation changed the public output contract")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = model_path.with_suffix(".tmp")
        torch.save(model, temporary)
        temporary.replace(model_path)
        reloaded = torch.load(model_path, map_location="cpu", weights_only=False).float().cpu().eval()
        with torch.inference_mode():
            comparison = structural.engine.tensor_comparison(after, reloaded(torch.zeros(1, 3, 640, 640)))
        if not comparison["exact"]:
            raise RuntimeError("Saved/reloaded structural model differs")
        record.update({
            "status": "PASS", "input_model": relative(input_path), "input_sha256": base.sha256(input_path),
            "output_model": relative(model_path), "output_sha256": base.sha256(model_path),
            "selection": mask, "intervention": intervention,
            "structure": {"parameters_before": parameters_before, "parameters_after": parameters_after,
                          "incremental_parameters_removed": parameters_before - parameters_after,
                          "cumulative_parameter_reduction_percent": 100.0 * (1.0 - parameters_after / BASELINE_PARAMETERS),
                          "native_output_after": native, "public_prediction_before": public_before,
                          "public_prediction_after": public_after, "save_reload_comparison": comparison},
        })
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        atomic_json(record_path, record)
    if record["status"] != "PASS":
        raise RuntimeError(f"{domain} {group_id} failed; inspect {record_path}")
    return record


def restore(entries: list[dict[str, Any]]) -> tuple[int, dict[str, Path | None], dict[str, int]]:
    current: dict[str, Path | None] = {domain: None for domain in DOMAINS}
    parameters = {domain: BASELINE_PARAMETERS for domain in DOMAINS}
    for index, item in enumerate(entries, start=1):
        paths = {domain: OUTPUT_ROOT / "records" / f"{domain}_step{index:02d}_{item['group_id']}.json" for domain in DOMAINS}
        if not any(path.is_file() for path in paths.values()):
            return index, current, parameters
        if not all(path.is_file() for path in paths.values()):
            raise RuntimeError(f"Incomplete paired step {index}; do not resume automatically")
        for domain, path in paths.items():
            record = base.read_json(path)
            if record.get("status") != "PASS":
                raise RuntimeError(f"Existing failure at {path}")
            current[domain] = PROJECT_ROOT / record["output_model"]
            parameters[domain] = int(record["structure"]["parameters_after"])
    return len(entries) + 1, current, parameters


def run(resume: bool, max_steps: int | None) -> int:
    entries, masks = queue_and_masks()
    manifest_path = OUTPUT_ROOT / "experiment_manifest.json"
    if manifest_path.is_file() and not resume:
        raise RuntimeError(f"Output exists: {OUTPUT_ROOT}. Use --resume; existing evidence is immutable.")
    manifest = base.read_json(manifest_path) if manifest_path.is_file() else {**preflight(), "schema": "bdd_isomorphic_taylor_raw56_manifest_v2", "status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": relative(SELF), "script_sha256": base.sha256(SELF)}
    atomic_json(manifest_path, manifest)
    start, current, parameters = restore(entries)
    for index, item in enumerate(entries[start - 1:], start=start):
        if max_steps is not None and index > max_steps:
            manifest.update({"status": "SMOKE_PASS", "completed_steps": index - 1, "target_reached": False})
            atomic_json(manifest_path, manifest)
            return 0
        group_id = str(item["group_id"])
        print(f"[{index}/49] Isomorphic-Taylor {group_id}", flush=True)
        records = {domain: run_step(domain, index, group_id, masks[domain][group_id], current[domain]) for domain in DOMAINS}
        for domain, record in records.items():
            current[domain] = PROJECT_ROOT / record["output_model"]
            parameters[domain] = int(record["structure"]["parameters_after"])
        manifest.update({"completed_steps": index, "current_parameter_reduction_percent": {domain: 100.0 * (1.0 - parameters[domain] / BASELINE_PARAMETERS) for domain in DOMAINS}})
        atomic_json(manifest_path, manifest)
        if all(100.0 * (1.0 - parameters[domain] / BASELINE_PARAMETERS) >= TARGET_REDUCTION for domain in DOMAINS):
            entries_frozen = []
            final_models = {}
            for step, candidate in enumerate(entries[:index], start=1):
                group = str(candidate["group_id"])
                domain_evidence = {}
                for domain in DOMAINS:
                    path = OUTPUT_ROOT / "records" / f"{domain}_step{step:02d}_{group}.json"
                    record = base.read_json(path)
                    domain_evidence[domain] = {"structure_record": relative(path), "structure_record_sha256": base.sha256(path), "output_model": record["output_model"], "output_sha256": record["output_sha256"], "selected_indices": record["selection"]["selected_indices"], "representative_root": record["intervention"]["representative_root"], "selection_unit": record["intervention"]["selection_unit"], "root_channels_before": record["intervention"]["root_channels_before"], "root_channels_removed": record["intervention"]["root_channels_removed"], "root_channels_after": record["intervention"]["root_channels_after"], "parameters_before": record["structure"]["parameters_before"], "parameters_after": record["structure"]["parameters_after"], "cumulative_parameter_reduction_percent": record["structure"]["cumulative_parameter_reduction_percent"]}
                entries_frozen.append({"sequence_step": step, "candidate_index": step, "group_id": group, "domains": domain_evidence})
            for domain in DOMAINS:
                target = OUTPUT_ROOT / "T5" / "models" / "final" / f"{domain}_isomorphic_taylor_raw_ranked_56pct.pth"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(current[domain], target)
                final_models[domain] = {"model": relative(target), "sha256": base.sha256(target), "parameters_before": BASELINE_PARAMETERS, "parameters_after": parameters[domain], "parameter_reduction_percent": 100.0 * (1.0 - parameters[domain] / BASELINE_PARAMETERS)}
            plan = {"schema": "bdd_isomorphic_taylor_raw56_frozen_plan_v2", "method": METHOD, "replay_policy": "exact one-shot baseline masks generated by the controlled 50-batch Isomorphic-Taylor ranking", "entries": entries_frozen, "final_models": final_models}
            atomic_json(OUTPUT_ROOT / "frozen_pruning_plan.json", plan)
            csv_rows: list[dict[str, Any]] = []
            for entry in entries_frozen:
                row: dict[str, Any] = {"sequence_step": entry["sequence_step"], "candidate_index": entry["candidate_index"], "group_id": entry["group_id"]}
                for domain in DOMAINS:
                    evidence = entry["domains"][domain]
                    row.update({f"{domain}_parameters_after": evidence["parameters_after"], f"{domain}_reduction_percent": evidence["cumulative_parameter_reduction_percent"], f"{domain}_selected_indices": ";".join(map(str, evidence["selected_indices"])), f"{domain}_structure_record": evidence["structure_record"]})
                csv_rows.append(row)
            atomic_csv(OUTPUT_ROOT / "tables" / "FROZEN_PRUNING_PLAN.csv", csv_rows)
            manifest.update({"status": "PASS", "target_reached": True, "accepted_groups": [str(entry["group_id"]) for entry in entries[:index]], "final_models": final_models, "frozen_plan": relative(OUTPUT_ROOT / "frozen_pruning_plan.json"), "frozen_plan_sha256": base.sha256(OUTPUT_ROOT / "frozen_pruning_plan.json"), "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            atomic_json(manifest_path, manifest)
            print(json.dumps(manifest["current_parameter_reduction_percent"], sort_keys=True))
            return 0
    manifest.update({"status": "NOT_REACHED", "target_reached": False})
    atomic_json(manifest_path, manifest)
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, help="Use 1 for the mandatory structural smoke test.")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    return run(args.resume, args.max_steps)


if __name__ == "__main__":
    raise SystemExit(main())
