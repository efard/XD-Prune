"""Run an approval-gated sequential GEN-protected pruning sweep.

This runner uses the frozen Custom Experiment-3 ranking only as a candidate
queue. It does not assume isolated pruning effects are additive. Each candidate
is applied independently to copies of the current accepted GEN and SNOW models,
then validated on both datasets. A candidate is accepted only if *both* domains
remain inside explicit cumulative-loss budgets supplied at launch.

No default accuracy budget exists intentionally. The approved values must be
passed on the command line, so this script cannot silently begin the real sweep
with an unapproved policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

CORE_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
if str(CORE_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(CORE_SCRIPTS))

import run_t1_t2 as base
import run_t1_t2_custom_sweep as custom
import run_cumulative_balanced_v2_top3_pilot as pilot


PROJECT_ROOT = base.PROJECT_ROOT
STUDY_ROOT = base.STUDY_ROOT
RANKING_PATH = STUDY_ROOT / "Rafed's Custom experiments" / "experiment_3_gen_protected_v3" / "T4_GEN_PROTECTED_PRUNABILITY.csv"
PROTOCOL_PATH = STUDY_ROOT / "configs" / "pruning" / "sequential_gen_protected_v3_pending_limits.json"
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "sequential_gen_protected_v3"
SCHEMA = "sequential_gen_protected_v3_v1"


def ranked_candidates(max_candidates: int | None) -> list[dict[str, str]]:
    """Return the frozen feasible positive-score queue without changing its order."""

    rows = base.read_csv(RANKING_PATH)
    rows = [
        row for row in rows
        if row["isolated_accuracy_feasible"] == "True"
        and float(row["gen_protected_prunability_P_GP"]) > 0.0
    ]
    rows.sort(key=lambda row: int(row["rank_by_P_GP_desc"]))
    if not rows:
        raise RuntimeError("The GEN-protected ranking has no feasible positive-score candidates")
    if max_candidates is not None:
        if max_candidates < 1:
            raise ValueError("--max-candidates must be at least 1")
        rows = rows[:max_candidates]
    if len({row["group_id"] for row in rows}) != len(rows):
        raise RuntimeError("Candidate queue contains duplicate group identifiers")
    return rows


def custom_catalogue() -> dict[str, dict[str, str]]:
    return {row["custom_group_id"]: row for row in base.read_csv(custom.CATALOGUE_PATH)}


def baseline_config(domain: str) -> dict[str, Any]:
    config = dict(base.read_json(custom.FREEZE_PATH)["baseline_models"][domain])
    config["baseline_reference"] = base.relative(custom.resolved_baseline_reference(config))
    return config


def preflight(require_cuda: bool, candidates: list[dict[str, str]]) -> dict[str, Any]:
    """Verify frozen inputs and that every queued group has a validated path."""

    import torch

    custom.preflight(require_cuda=require_cuda)
    generic_ids = {row["canonical_group_id"] for row in base.read_csv(base.CATALOGUE_PATH)}
    custom_ids = set(custom_catalogue())
    unknown = [row["group_id"] for row in candidates if row["group_id"] not in generic_ids | custom_ids]
    if unknown:
        raise RuntimeError(f"Candidate groups are absent from validated catalogues: {unknown}")
    for domain in ("GEN", "SNOW"):
        config = baseline_config(domain)
        base.verify_file(PROJECT_ROOT / config["path"], config["sha256"])
        base.verify_file(PROJECT_ROOT / config["dataset_yaml"], config["dataset_yaml_sha256"])
        base.verify_file(PROJECT_ROOT / config["baseline_reference"], config["baseline_reference_sha256"])
    return {
        "schema": SCHEMA,
        "ranking_source": base.relative(RANKING_PATH),
        "protocol": base.relative(PROTOCOL_PATH),
        "candidate_queue": [row["group_id"] for row in candidates],
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def apply_group(model: Any, group_id: str) -> dict[str, Any]:
    """Apply one live generic or validated custom dependency intervention."""

    catalogue = custom_catalogue()
    if group_id in catalogue:
        block_index = int(catalogue[group_id]["block_index"])
        result, importance, evidence = custom._apply_custom_rule(model.model[block_index], block_index)
        model.eval()
        model.zero_grad(set_to_none=True)
        return {
            "method": "validated custom logical-channel rule",
            "root": catalogue[group_id]["block_path"],
            "root_channels_before": result.hidden_channels_before,
            "root_channels_removed": result.hidden_channels_removed,
            "root_channels_after": result.hidden_channels_after,
            "importance": importance,
            "operations": evidence["result"]["operations"],
            "operation_count": len(evidence["result"]["operations"]),
            "invariants_before": evidence["invariants_before"],
            "invariants_after": evidence["invariants_after"],
        }
    return pilot.apply_generic(model, group_id)


def configure_shared_workers(candidates: list[dict[str, str]]) -> None:
    """Reuse the independently tested structural and evaluation workers safely."""

    group_ids = tuple(row["group_id"] for row in candidates)
    pilot.SCHEMA = SCHEMA
    pilot.EXPECTED_ORDER = group_ids
    pilot.RANKING_PATH = RANKING_PATH
    pilot.baseline_config = baseline_config
    pilot.preflight = lambda require_cuda: preflight(require_cuda, candidates)
    pilot.apply_group = apply_group


def run_subprocess(arguments: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), *arguments], stdout=stream, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"Worker failed ({process.returncode}); inspect {log_path}")


def update_worker_record(record_path: Path, accepted_before: list[str], group_id: str, candidate_index: int) -> dict[str, Any]:
    """Correct inherited pilot metadata for skipped candidates and preserve it."""

    record = base.read_json(record_path)
    if record.get("status") != "PASS":
        raise RuntimeError(f"Worker did not pass: {record_path}")
    record["candidate_index"] = candidate_index
    record["accepted_groups_before"] = accepted_before
    record["candidate_group"] = group_id
    record["trial_groups"] = [*accepted_before, group_id]
    base.atomic_json(record_path, record)
    return record


def cumulative_loss(metrics_record: dict[str, Any]) -> float:
    """Return non-negative cumulative mAP50-95 loss against the frozen baseline."""

    value = float(metrics_record["changes_from_unpruned_baseline"]["map50_95"]["normalized_signed_drop"])
    return max(0.0, value)


def decision_row(
    candidate_index: int,
    rank: int,
    group_id: str,
    accepted_before: list[str],
    structures: dict[str, dict[str, Any]],
    metrics: dict[str, dict[str, Any]],
    max_gen_loss: float,
    max_snow_loss: float,
) -> dict[str, Any]:
    gen_loss = cumulative_loss(metrics["GEN"])
    snow_loss = cumulative_loss(metrics["SNOW"])
    accepted = gen_loss <= max_gen_loss and snow_loss <= max_snow_loss
    return {
        "candidate_index": candidate_index,
        "rank_from_frozen_T4": rank,
        "group_id": group_id,
        "accepted_groups_before": "+".join(accepted_before) or "NONE",
        "decision": "ACCEPTED" if accepted else "REJECTED",
        "reason": "both cumulative domain budgets passed" if accepted else "one or both cumulative domain budgets exceeded",
        "max_gen_normalized_loss": max_gen_loss,
        "max_snow_normalized_loss": max_snow_loss,
        "gen_normalized_loss": gen_loss,
        "snow_normalized_loss": snow_loss,
        "gen_map50_95": metrics["GEN"]["metrics"]["map50_95"],
        "snow_map50_95": metrics["SNOW"]["metrics"]["map50_95"],
        "gen_parameters_after": structures["GEN"]["structure"]["parameters_after"],
        "snow_parameters_after": structures["SNOW"]["structure"]["parameters_after"],
        "gen_gflops_after": structures["GEN"]["structure"]["gflops_after"],
        "snow_gflops_after": structures["SNOW"]["structure"]["gflops_after"],
        "gen_structure_record": base.relative(structures["GEN"]["_path"]),
        "snow_structure_record": base.relative(structures["SNOW"]["_path"]),
        "gen_metrics_record": base.relative(metrics["GEN"]["_path"]),
        "snow_metrics_record": base.relative(metrics["SNOW"]["_path"]),
    }


def write_decision_table(output: Path) -> None:
    records = []
    for path in sorted((output / "decisions").glob("decision_*.json")):
        record = base.read_json(path)
        if record.get("status") == "PASS":
            records.append(record["row"])
    if not records:
        return
    table = output / "tables" / "SEQUENTIAL_DECISIONS.csv"
    table.parent.mkdir(parents=True, exist_ok=True)
    temporary = table.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    try:
        temporary.replace(table)
    except PermissionError:
        # Excel commonly locks the canonical CSV while a long experiment is
        # running. Preserve the experiment rather than failing after a valid
        # candidate decision; a separate live copy remains fully traceable.
        live = table.with_name("SEQUENTIAL_DECISIONS_LIVE.csv")
        temporary.replace(live)
        print(f"WARNING: {table.name} is locked; wrote {live.name} instead", flush=True)


def initial_manifest(evidence: dict[str, Any], output: Path, max_gen_loss: float, max_snow_loss: float) -> dict[str, Any]:
    return {
        **evidence,
        "status": "RUNNING",
        "output": base.relative(output),
        "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy": {
            "same_candidate_applied_independently_to_each_domain": True,
            "acceptance_requires_both_domains": True,
            "max_gen_normalized_map50_95_loss": max_gen_loss,
            "max_snow_normalized_map50_95_loss": max_snow_loss,
            "fraction_per_live_root": 0.125,
            "fine_tuning": False,
            "batchnorm_update": False,
            "test_data_used": False,
            "rejected_trials_retained": True,
            "generic_dependencies_rebuilt_live_after_each_accepted_step": True,
        },
        "accepted_groups": [],
    }


def existing_decision(output: Path, candidate_index: int) -> dict[str, Any] | None:
    path = output / "decisions" / f"decision_{candidate_index:02d}.json"
    if not path.is_file():
        return None
    value = base.read_json(path)
    return value if value.get("status") == "PASS" else None


def orchestrate(output: Path, max_gen_loss: float, max_snow_loss: float, max_candidates: int | None, resume: bool) -> int:
    if not 0.0 <= max_gen_loss < 1.0 or not 0.0 <= max_snow_loss < 1.0:
        raise ValueError("Cumulative normalized-loss budgets must be in [0, 1)")
    candidates = ranked_candidates(max_candidates)
    evidence = preflight(require_cuda=True, candidates=candidates)
    configure_shared_workers(candidates)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment_manifest.json"
    if resume and manifest_path.is_file():
        manifest = base.read_json(manifest_path)
        previous = manifest.get("policy", {})
        if (previous.get("max_gen_normalized_map50_95_loss") != max_gen_loss
                or previous.get("max_snow_normalized_map50_95_loss") != max_snow_loss
                or manifest.get("candidate_queue") != evidence["candidate_queue"]):
            raise RuntimeError("Cannot resume with a changed candidate queue or acceptance policy")
    else:
        manifest = initial_manifest(evidence, output, max_gen_loss, max_snow_loss)
        base.atomic_json(manifest_path, manifest)

    accepted_groups: list[str] = []
    accepted_models: dict[str, Path | None] = {"GEN": None, "SNOW": None}
    for candidate_index, candidate in enumerate(candidates, start=1):
        prior = existing_decision(output, candidate_index) if resume else None
        if prior is not None:
            if prior["row"]["decision"] == "ACCEPTED":
                accepted_groups.append(candidate["group_id"])
                for domain in ("GEN", "SNOW"):
                    accepted_models[domain] = PROJECT_ROOT / base.read_json(
                        output / "records" / f"{domain}_candidate{candidate_index:02d}_structure.json"
                    )["output_model"]
            continue

        group_id = candidate["group_id"]
        structures: dict[str, dict[str, Any]] = {}
        metrics: dict[str, dict[str, Any]] = {}
        for domain in ("GEN", "SNOW"):
            structure_path = output / "records" / f"{domain}_candidate{candidate_index:02d}_structure.json"
            metrics_path = output / "records" / f"{domain}_candidate{candidate_index:02d}_metrics.json"
            model_path = output / "models" / "trials" / f"{domain}_candidate{candidate_index:02d}_{group_id}.pth"
            worker_args = ["--worker", "prune", "--domain", domain, "--candidate-index", str(candidate_index), "--group", group_id, "--output", str(output)]
            if accepted_models[domain] is not None:
                worker_args.extend(["--input-model", str(accepted_models[domain])])
            run_subprocess(worker_args, output / "logs" / f"{domain}_candidate{candidate_index:02d}_prune.log")
            structures[domain] = update_worker_record(structure_path, accepted_groups, group_id, candidate_index)
            structures[domain]["_path"] = structure_path
            run_subprocess(
                ["--worker", "evaluate", "--domain", domain, "--candidate-index", str(candidate_index), "--group", group_id,
                 "--input-model", str(model_path), "--output", str(output)],
                output / "logs" / f"{domain}_candidate{candidate_index:02d}_evaluate.log",
            )
            metrics[domain] = update_worker_record(metrics_path, accepted_groups, group_id, candidate_index)
            metrics[domain]["_path"] = metrics_path

        row = decision_row(candidate_index, int(candidate["rank_by_P_GP_desc"]), group_id, accepted_groups, structures, metrics, max_gen_loss, max_snow_loss)
        decision = {"schema": SCHEMA, "status": "PASS", "row": row, "completed_local": time.strftime("%Y-%m-%d %H:%M:%S")}
        base.atomic_json(output / "decisions" / f"decision_{candidate_index:02d}.json", decision)
        if row["decision"] == "ACCEPTED":
            accepted_groups.append(group_id)
            for domain in ("GEN", "SNOW"):
                accepted_models[domain] = PROJECT_ROOT / structures[domain]["output_model"]
        manifest["accepted_groups"] = accepted_groups
        base.atomic_json(manifest_path, manifest)
        write_decision_table(output)

    manifest.update({"status": "PASS", "completed_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    base.atomic_json(manifest_path, manifest)
    write_decision_table(output)
    base.atomic_text(output / "README.md", """# Sequential GEN-Protected Pruning Sweep

This folder is created only when an explicitly approved cumulative GEN and SNOW accuracy budget is supplied at launch. The runner uses the frozen GEN-protected ranking as a candidate queue, not an additive prediction. Each trial is structurally validated and evaluated independently on both domains. A candidate is accepted only when both domain copies pass the approved cumulative limits. No fine-tuning, BatchNorm update, or test data is used in this raw-screening stage.

`tables/SEQUENTIAL_DECISIONS.csv` is the supervisor-facing decision log. `records/` contains structural and metric evidence for every accepted and rejected trial; rejected trial models are intentionally retained for traceability.
""")
    return 0


def worker_prune(domain: str, candidate_index: int, group_id: str, input_model: Path | None, output: Path, candidates: list[dict[str, str]]) -> int:
    configure_shared_workers(candidates)
    # The established worker names output models by step; temporarily adapt its paths after completion.
    status = pilot.prune_worker(domain, candidate_index, group_id, input_model, output)
    old_structure = output / "records" / f"{domain}_step{candidate_index:02d}_structure.json"
    new_structure = output / "records" / f"{domain}_candidate{candidate_index:02d}_structure.json"
    old_model = output / "models" / f"{domain}_step{candidate_index:02d}_{group_id}.pth"
    new_model = output / "models" / "trials" / f"{domain}_candidate{candidate_index:02d}_{group_id}.pth"
    if old_structure.is_file():
        new_structure.parent.mkdir(parents=True, exist_ok=True)
        old_structure.replace(new_structure)
    if old_model.is_file():
        new_model.parent.mkdir(parents=True, exist_ok=True)
        old_model.replace(new_model)
        record = base.read_json(new_structure)
        if record.get("status") == "PASS":
            record["output_model"] = base.relative(new_model)
            record["output_sha256"] = base.sha256(new_model)
            base.atomic_json(new_structure, record)
    return status


def worker_evaluate(domain: str, candidate_index: int, group_id: str, input_model: Path, output: Path, candidates: list[dict[str, str]]) -> int:
    configure_shared_workers(candidates)
    status = pilot.evaluation_worker(domain, candidate_index, group_id, input_model, output)
    old_metrics = output / "records" / f"{domain}_step{candidate_index:02d}_metrics.json"
    new_metrics = output / "records" / f"{domain}_candidate{candidate_index:02d}_metrics.json"
    if old_metrics.is_file():
        old_metrics.replace(new_metrics)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--max-gen-loss", type=float, help="Approved cumulative normalized mAP50-95 loss limit for GEN")
    parser.add_argument("--max-snow-loss", type=float, help="Approved cumulative normalized mAP50-95 loss limit for SNOW")
    parser.add_argument("--max-candidates", type=int, help="Optional cap on the frozen candidate queue")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight", action="store_true", help="Check frozen inputs only; does not prune or evaluate")
    parser.add_argument("--worker", choices=("prune", "evaluate"))
    parser.add_argument("--domain", choices=("GEN", "SNOW"))
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    args = parser.parse_args()
    candidates = ranked_candidates(args.max_candidates)
    if args.preflight:
        print(json.dumps(preflight(require_cuda=False, candidates=candidates), indent=2, sort_keys=True))
        return 0
    if args.worker == "prune":
        if args.domain is None or args.candidate_index is None or args.group is None:
            parser.error("worker pruning requires --domain, --candidate-index and --group")
        return worker_prune(args.domain, args.candidate_index, args.group, args.input_model, args.output.resolve(), candidates)
    if args.worker == "evaluate":
        if args.domain is None or args.candidate_index is None or args.group is None or args.input_model is None:
            parser.error("worker evaluation requires --domain, --candidate-index, --group and --input-model")
        return worker_evaluate(args.domain, args.candidate_index, args.group, args.input_model, args.output.resolve(), candidates)
    if args.max_gen_loss is None or args.max_snow_loss is None:
        parser.error("--max-gen-loss and --max-snow-loss are required for a pruning sweep; use --preflight for preparation only")
    return orchestrate(args.output.resolve(), args.max_gen_loss, args.max_snow_loss, args.max_candidates, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
