"""Construct BDD GEN-2/NGN-2 raw T5 models at >=56% parameter reduction.

This is the BDD continuation of the established direct 56% workflow.  It uses
the revised 37.5%-local BDD T4 sequence, not the historical MIO/ACDC sequence.
Each candidate is re-pruned on the current cumulative model independently for
GEN2 and NGN2. The first state with both models at >=56% reduction is frozen
with exact selected indices for later T5 evaluation and T6/T7 recovery.

This runner performs structural preparation only: no validation evaluation,
fine-tuning, BatchNorm recalibration, or test-data access occurs here.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
SOURCE_T4 = (
    STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t3_t4_updated_v1"
    / "37_5pct" / "tables" / "T4_UPDATED_PRUNABILITY_37_5PCT.csv"
)
FORMULA_CONFIG = SOURCE_T4.parents[2] / "FORMULA_CONFIG.json"
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_raw56_updated_v1"
SCHEMA = "bdd_raw56_updated_local375_structural_v1"
LOCAL_FRACTION = Fraction(3, 8)
LOCAL_PERCENT = 37.5
TARGET_FRACTION = 0.56
DOMAINS = ("GEN2", "NGN2")
BASELINE_PARAMETERS = {"GEN2": 2_506_140, "NGN2": 2_506_140}
PROTECTED_GROUPS = {
    "DG001": "stem remains protected",
    "DG020": "excluded by the completed cumulative-collapse audit",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The historical workers use local sibling imports rather than package imports.
for import_root in (
    STUDY_ROOT / "scripts",
    STUDY_ROOT / "results" / "pruning" / "prune_37_5",
    STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


# The BDD runner adapts the established, validated 37.5% DepGraph/custom-rule
# engine to the BDD checkpoints and frozen 640x640 evaluator.
bdd = load_module("_bdd_raw56_runner", STUDY_ROOT / "scripts" / "run_bdd_t1_t2_sweep.py")
base = bdd.base
ratio_engine = bdd.engine
sequential_engine = load_module(
    "_bdd_raw56_structural_worker",
    STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts" / "run_sequential_gen_protected_v3.py",
)

MANIFEST_PATH = OUTPUT_ROOT / "experiment_manifest.json"
DECISION_TABLE = OUTPUT_ROOT / "tables" / "STRUCTURAL_DECISIONS.csv"
PLAN_CSV = OUTPUT_ROOT / "tables" / "FROZEN_PRUNING_PLAN.csv"
PLAN_JSON = OUTPUT_ROOT / "frozen_pruning_plan.json"


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def relative(path: Path) -> str:
    return base.relative(path)


def queue() -> list[dict[str, str]]:
    rows = base.read_csv(SOURCE_T4)
    if len(rows) != 51 or len({row["group_id"] for row in rows}) != 51:
        raise RuntimeError("The BDD 37.5% T4 table must contain 51 unique groups")
    rows.sort(key=lambda row: int(row["rank_by_prunability_descending"]))
    selected = [row for row in rows if row["group_id"] not in PROTECTED_GROUPS]
    if len(selected) != 49 or len({row["group_id"] for row in selected}) != 49:
        raise RuntimeError("Expected 49 BDD candidates after protected-group exclusions")
    return selected


def baseline_config(domain: str) -> dict[str, Any]:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown BDD domain: {domain}")
    return bdd.domain_config(domain)


def apply_group(model: Any, group_id: str) -> dict[str, Any]:
    if group_id.startswith("CDG"):
        result = ratio_engine.apply_custom(model, group_id, LOCAL_FRACTION)
    else:
        result = ratio_engine.apply_generic(model, group_id, LOCAL_FRACTION)
    result["guarded_local_ratio_percent"] = LOCAL_PERCENT
    return result


def worker_preflight(require_cuda: bool, candidates: list[dict[str, str]]) -> dict[str, Any]:
    if not SOURCE_T4.is_file() or not FORMULA_CONFIG.is_file():
        raise FileNotFoundError("The frozen revised BDD T4 evidence is missing")
    formula = base.read_json(FORMULA_CONFIG)
    if formula.get("formula_id") != "BDD_UPDATED_GEN_PROTECTED_V1":
        raise RuntimeError("Unexpected BDD T4 formula configuration")
    bdd_evidence = bdd.preflight(LOCAL_FRACTION, require_cuda=require_cuda, require_references=True)
    known = set(ratio_engine.generic_rows()) | set(ratio_engine.custom_rows())
    unknown = [row["group_id"] for row in candidates if row["group_id"] not in known]
    if unknown:
        raise RuntimeError(f"Candidate groups absent from validated catalogues: {unknown}")
    return {
        "schema": f"{SCHEMA}_preflight",
        "formula_id": formula["formula_id"],
        "domains": list(DOMAINS),
        "local_pruning_percent": LOCAL_PERCENT,
        "queued_groups": len(candidates),
        "known_validated_groups": len(known),
        "bdd_t1_t2_preflight": bdd_evidence,
    }


def configure_workers(candidates: list[dict[str, str]]) -> None:
    # This patches only the worker module loaded in this process. It reuses its
    # structural/save-reload checks but routes baseline lookup to BDD GEN2/NGN2.
    sequential_engine.SCHEMA = SCHEMA
    sequential_engine.RANKING_PATH = SOURCE_T4
    sequential_engine.OUTPUT_ROOT = OUTPUT_ROOT
    sequential_engine.baseline_config = baseline_config
    sequential_engine.ranked_candidates = lambda max_candidates=None: (
        candidates if max_candidates is None else candidates[:max_candidates]
    )
    sequential_engine.apply_group = apply_group
    sequential_engine.preflight = lambda require_cuda, queued: worker_preflight(require_cuda, queued)


def preflight() -> dict[str, Any]:
    candidates = queue()
    configure_workers(candidates)
    evidence = worker_preflight(False, candidates)
    isolated_sum = sum(float(row["parameters_removed"]) for row in candidates) / BASELINE_PARAMETERS["GEN2"]
    return {
        "schema": SCHEMA,
        "formula_id": "BDD_UPDATED_GEN_PROTECTED_V1",
        "T4_source": relative(SOURCE_T4),
        "T4_sha256": base.sha256(SOURCE_T4),
        "formula_config": relative(FORMULA_CONFIG),
        "formula_config_sha256": base.sha256(FORMULA_CONFIG),
        "local_pruning_fraction": float(LOCAL_FRACTION),
        "local_pruning_percent": LOCAL_PERCENT,
        "target_global_parameter_reduction_fraction": TARGET_FRACTION,
        "candidate_count": len(candidates),
        "candidate_queue": [row["group_id"] for row in candidates],
        "protected_groups": PROTECTED_GROUPS,
        "baseline_parameters": BASELINE_PARAMETERS,
        "theoretical_isolated_parameter_reduction_sum_fraction": isolated_sum,
        "warning": "Isolated resource removals are not additive; live cumulative structures are measured after every accepted group.",
        "worker_preflight": evidence,
    }


def structure_path(domain: str, candidate_index: int) -> Path:
    return OUTPUT_ROOT / "records" / f"{domain}_candidate{candidate_index:02d}_structure.json"


def decision_path(candidate_index: int) -> Path:
    return OUTPUT_ROOT / "decisions" / f"decision_{candidate_index:02d}.json"


def load_structure(domain: str, candidate_index: int) -> dict[str, Any] | None:
    path = structure_path(domain, candidate_index)
    return base.read_json(path) if path.is_file() else None


def reduction(domain: str, parameters_after: int) -> float:
    return 1.0 - parameters_after / BASELINE_PARAMETERS[domain]


def worker_command(domain: str, candidate_index: int, group_id: str, input_model: Path | None) -> list[str]:
    command = [sys.executable, str(SELF), "--worker", "prune", "--domain", domain, "--candidate-index", str(candidate_index), "--group", group_id]
    if input_model is not None:
        command.extend(["--input-model", str(input_model)])
    return command


def launch_worker(domain: str, candidate_index: int, group_id: str, input_model: Path | None) -> int:
    log = OUTPUT_ROOT / "logs" / f"{domain}_rank{candidate_index:02d}_{group_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.run(worker_command(domain, candidate_index, group_id, input_model), stdout=stream, stderr=subprocess.STDOUT)
    return int(process.returncode)


def normalize_record(record: dict[str, Any], path: Path, accepted_before: list[str], group_id: str, candidate_index: int) -> dict[str, Any]:
    record["candidate_index"] = candidate_index
    record["accepted_groups_before"] = list(accepted_before)
    record["groups_applied"] = [*accepted_before, group_id]
    record["local_pruning_percent"] = LOCAL_PERCENT
    atomic_json(path, record)
    return record


def all_decisions() -> list[dict[str, Any]]:
    records = []
    for path in sorted((OUTPUT_ROOT / "decisions").glob("decision_*.json")):
        record = base.read_json(path)
        if record.get("status") != "PASS":
            raise RuntimeError(f"Incomplete decision record: {path}")
        records.append(record)
    return records


def write_outputs(decisions: list[dict[str, Any]]) -> None:
    rows = [record["row"] for record in decisions]
    atomic_csv(DECISION_TABLE, rows)
    entries = []
    for record in decisions:
        row = record["row"]
        if row["decision"] != "ACCEPTED_STRUCTURALLY":
            continue
        domain_entries = {}
        for domain in DOMAINS:
            path = structure_path(domain, int(row["candidate_index"]))
            evidence = base.read_json(path)
            intervention = evidence["intervention"]
            domain_entries[domain] = {
                "structure_record": relative(path),
                "structure_record_sha256": base.sha256(path),
                "input_model": evidence["input_model"],
                "input_sha256": evidence["input_sha256"],
                "output_model": evidence["output_model"],
                "output_sha256": evidence["output_sha256"],
                "selection_unit": intervention.get("selection_unit"),
                "representative_root": intervention.get("representative_root", intervention.get("root")),
                "root_channels_before": intervention["root_channels_before"],
                "root_channels_removed": intervention["root_channels_removed"],
                "root_channels_after": intervention["root_channels_after"],
                "selected_indices": intervention.get("selected_indices", []),
                "operation_count": intervention.get("operation_count"),
            }
        entries.append({
            "sequence_step": len(entries) + 1,
            "candidate_index": int(row["candidate_index"]),
            "T4_rank": int(row["T4_rank"]),
            "group_id": row["group_id"],
            "local_pruning_percent": LOCAL_PERCENT,
            "domains": domain_entries,
        })
    plan_rows = []
    rows_by_index = {int(row["candidate_index"]): row for row in rows}
    for entry in entries:
        row = rows_by_index[entry["candidate_index"]]
        plan_rows.append({
            "sequence_step": entry["sequence_step"], "candidate_index": entry["candidate_index"], "T4_rank": entry["T4_rank"], "group_id": entry["group_id"], "local_pruning_percent": LOCAL_PERCENT,
            "GEN2_selected_indices": ";".join(map(str, entry["domains"]["GEN2"]["selected_indices"])),
            "NGN2_selected_indices": ";".join(map(str, entry["domains"]["NGN2"]["selected_indices"])),
            "GEN2_parameters_after": row["GEN2_parameters_after"], "NGN2_parameters_after": row["NGN2_parameters_after"],
            "GEN2_reduction_percent": row["GEN2_reduction_percent"], "NGN2_reduction_percent": row["NGN2_reduction_percent"],
            "GEN2_structure_record": entry["domains"]["GEN2"]["structure_record"], "NGN2_structure_record": entry["domains"]["NGN2"]["structure_record"],
        })
    atomic_csv(PLAN_CSV, plan_rows)
    atomic_json(PLAN_JSON, {
        "schema": SCHEMA,
        "purpose": "freeze exact BDD GEN2/NGN2 architecture/channel masks for T5 evaluation and T6/T7 recovery",
        "formula_id": "BDD_UPDATED_GEN_PROTECTED_V1",
        "T4_source": relative(SOURCE_T4), "T4_sha256": base.sha256(SOURCE_T4),
        "local_pruning_percent": LOCAL_PERCENT,
        "target_global_parameter_reduction_percent": 100.0 * TARGET_FRACTION,
        "protected_groups": PROTECTED_GROUPS,
        "entries": entries,
    })


def restore_state(decisions: list[dict[str, Any]]) -> tuple[list[str], dict[str, Path | None], dict[str, int]]:
    accepted: list[str] = []
    current: dict[str, Path | None] = {domain: None for domain in DOMAINS}
    parameters = dict(BASELINE_PARAMETERS)
    for decision in decisions:
        row = decision["row"]
        if row["decision"] != "ACCEPTED_STRUCTURALLY":
            continue
        accepted.append(row["group_id"])
        for domain in DOMAINS:
            structure = base.read_json(structure_path(domain, int(row["candidate_index"])))
            current[domain] = PROJECT_ROOT / structure["output_model"]
            parameters[domain] = int(structure["structure"]["parameters_after"])
    return accepted, current, parameters


def freeze_final_models(current: dict[str, Path | None], parameters: dict[str, int], target_reached: bool) -> dict[str, Any]:
    label = "raw_ranked_56pct" if target_reached else "max_reached_not_56pct"
    output = {}
    for domain in DOMAINS:
        source = current[domain]
        if source is None:
            raise RuntimeError(f"No structurally accepted {domain} model exists")
        destination = OUTPUT_ROOT / "T5" / "models" / "final" / f"{domain}_{label}.pth"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        output[domain] = {
            "model": relative(destination), "sha256": base.sha256(destination),
            "parameters_before": BASELINE_PARAMETERS[domain], "parameters_after": parameters[domain],
            "parameter_reduction_percent": 100.0 * reduction(domain, parameters[domain]),
        }
    return output


def orchestrate(resume: bool) -> int:
    evidence = preflight()
    candidates = queue()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        if not resume:
            raise RuntimeError("This BDD raw56 run already exists. Use --resume; evidence will not be overwritten.")
        manifest = base.read_json(MANIFEST_PATH)
        if manifest.get("T4_sha256") != evidence["T4_sha256"] or manifest.get("formula_config_sha256") != evidence["formula_config_sha256"]:
            raise RuntimeError("Frozen ranking/formula evidence changed; resume is unsafe")
        if manifest.get("target_reached"):
            print("Target was already reached; no work is required.")
            return 0
    else:
        manifest = {**evidence, "status": "RUNNING", "started_local": time.strftime("%Y-%m-%d %H:%M:%S"), "target_reached": False, "accepted_groups": [], "policy": {
            "ranking": "BDD_UPDATED_GEN_PROTECTED_V1 at 37.5% local pruning", "same_group_sequence_both_domains": True,
            "local_channel_importance": "live L1 importance; generic/custom DepGraph rules rebuilt after each accepted step",
            "structural_acceptance_requires_both_domains": True, "accuracy_gate": False, "fine_tuning": False,
            "batchnorm_recalibration": False, "test_data_used": False,
            "stop_rule": "first accepted state where GEN2 and NGN2 both reach >=56% parameter reduction",
        }}
        atomic_json(MANIFEST_PATH, manifest)

    decisions = all_decisions()
    accepted, current, parameters = restore_state(decisions)
    completed = {int(record["row"]["candidate_index"]) for record in decisions}
    for candidate_index, candidate in enumerate(candidates, start=1):
        if candidate_index in completed:
            continue
        group_id = candidate["group_id"]
        accepted_before = list(accepted)
        return_codes, structures = {}, {}
        for domain in DOMAINS:
            return_codes[domain] = launch_worker(domain, candidate_index, group_id, current[domain])
            path = structure_path(domain, candidate_index)
            record = load_structure(domain, candidate_index)
            if record is not None:
                record = normalize_record(record, path, accepted_before, group_id, candidate_index)
            structures[domain] = record
        valid = all(return_codes[domain] == 0 and structures[domain] is not None and structures[domain].get("status") == "PASS" for domain in DOMAINS)
        if valid:
            decision_name, reason = "ACCEPTED_STRUCTURALLY", "both BDD domain workers passed all structural checks"
            accepted.append(group_id)
            for domain in DOMAINS:
                record = structures[domain]
                assert record is not None
                current[domain] = PROJECT_ROOT / record["output_model"]
                parameters[domain] = int(record["structure"]["parameters_after"])
        else:
            decision_name = "REJECTED_STRUCTURALLY"
            reason = "; ".join(f"{domain}:exit={return_codes[domain]},status={None if structures[domain] is None else structures[domain].get('status')}" for domain in DOMAINS)
        row = {
            "candidate_index": candidate_index, "T4_rank": int(candidate["rank_by_prunability_descending"]), "group_id": group_id,
            "decision": decision_name, "reason": reason, "accepted_group_count_after": len(accepted), "accepted_groups_after": "+".join(accepted),
            **{f"{domain}_parameters_after": parameters[domain] for domain in DOMAINS},
            **{f"{domain}_reduction_percent": 100.0 * reduction(domain, parameters[domain]) for domain in DOMAINS},
            **{f"{domain}_worker_exit_code": return_codes[domain] for domain in DOMAINS},
        }
        decision = {"schema": SCHEMA, "status": "PASS", "row": row, "completed_local": time.strftime("%Y-%m-%d %H:%M:%S")}
        atomic_json(decision_path(candidate_index), decision)
        decisions.append(decision)
        write_outputs(decisions)
        manifest.update({"accepted_groups": accepted, "latest_parameters": parameters, "latest_reduction_percent": {domain: 100.0 * reduction(domain, parameters[domain]) for domain in DOMAINS}})
        atomic_json(MANIFEST_PATH, manifest)
        print(f"[{candidate_index:02d}/{len(candidates)}] {group_id} {decision_name}; GEN2={row['GEN2_reduction_percent']:.3f}%, NGN2={row['NGN2_reduction_percent']:.3f}%", flush=True)
        if all(reduction(domain, parameters[domain]) >= TARGET_FRACTION for domain in DOMAINS):
            manifest["target_reached"] = True
            break

    target_reached = bool(manifest.get("target_reached"))
    final_models = freeze_final_models(current, parameters, target_reached)
    manifest.update({"status": "PASS" if target_reached else "TARGET_NOT_REACHED", "target_reached": target_reached, "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"), "accepted_groups": accepted, "accepted_group_count": len(accepted), "final_models": final_models, "final_reduction_percent": {domain: 100.0 * reduction(domain, parameters[domain]) for domain in DOMAINS}})
    atomic_json(MANIFEST_PATH, manifest)
    write_outputs(decisions)
    atomic_text(OUTPUT_ROOT / "README.md", "# BDD raw T5 at approximately 56%\n\nThis folder contains the BDD GEN2/NGN2 direct cumulative structural preparation using the revised 37.5% T4 order. It contains no T5 accuracy evaluation or recovery result. Exact masks are frozen only after both domains reach the target.\n")
    print(json.dumps(final_models, indent=2), flush=True)
    return 0 if target_reached else 2


def worker_mode(args: argparse.Namespace) -> int:
    candidates = queue()
    configure_workers(candidates)
    return sequential_engine.worker_prune(args.domain, args.candidate_index, args.group, args.input_model, OUTPUT_ROOT, candidates)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker", choices=("prune",))
    parser.add_argument("--domain", choices=DOMAINS)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.domain is None or args.candidate_index is None or not args.group:
            parser.error("worker mode requires --domain, --candidate-index, and --group")
        return worker_mode(args)
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    if args.run and args.resume:
        parser.error("Choose --run or --resume, not both")
    if not args.run and not args.resume:
        parser.error("Choose --preflight, --run, or --resume")
    return orchestrate(resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
