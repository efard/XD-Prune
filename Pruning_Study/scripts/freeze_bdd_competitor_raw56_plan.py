"""Freeze completed BDD competitor raw-T5 evidence into replayable T7 plans.

This script is deliberately read-only with respect to raw models and records.
It verifies the completed raw manifest and every accepted per-step record, then
writes deterministic frozen-plan JSON/CSV evidence beneath the corresponding
raw result root.  T7 recovery consumes these files to replay exactly the raw
architecture instead of re-ranking or re-selecting channels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT.parent
OUTPUT_BASE = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_competitors_raw56_v1"
DOMAINS = ("GEN2", "NGN2")
METHODS = ("global_l1", "fpgm")
BASELINE_PARAMETERS = 2_506_140
SCHEMA = "bdd_competitor_raw56_frozen_plan_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty frozen-plan CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def raw_root(method: str) -> Path:
    return OUTPUT_BASE / f"{method}_37_5pct_raw56"


def compact_domain_record(record: dict[str, Any], path: Path) -> dict[str, Any]:
    intervention = dict(record["intervention"])
    structure = dict(record["structure"])
    selected = [int(value) for value in intervention["selected_indices"]]
    if not selected:
        raise RuntimeError(f"Empty frozen selection: {path}")
    evidence = {
        "structure_record": relative(path),
        "structure_record_sha256": sha256(path),
        "input_model": str(record["input_model"]),
        "input_sha256": str(record["input_sha256"]),
        "output_model": str(record["output_model"]),
        "output_sha256": str(record["output_sha256"]),
        "representative_root": str(intervention["representative_root"]),
        "rule_family": str(intervention["rule_family"]),
        "selection_unit": str(intervention["selection_unit"]),
        "root_channels_before": int(intervention["root_channels_before"]),
        "root_channels_removed": int(intervention["root_channels_removed"]),
        "root_channels_after": int(intervention["root_channels_after"]),
        "selected_indices": selected,
        "parameters_before": int(structure["parameters_before"]),
        "parameters_after": int(structure["parameters_after"]),
        "cumulative_parameter_reduction_percent": float(
            structure["cumulative_parameter_reduction_percent"]
        ),
    }
    if "operations" in intervention:
        evidence["operation_count"] = len(intervention["operations"])
    return evidence


def build(method: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = raw_root(method)
    manifest_path = root / "experiment_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("status") != "PASS" or not manifest.get("target_reached"):
        raise RuntimeError(f"{method}: raw T5 is not a completed target-reaching run")
    if manifest.get("method") != method:
        raise RuntimeError(f"{method}: raw manifest names {manifest.get('method')!r}")
    groups = [str(group) for group in manifest.get("accepted_groups", ())]
    if not groups or len(groups) != len(set(groups)):
        raise RuntimeError(f"{method}: accepted group sequence is empty or non-unique")

    entries: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    for step, group_id in enumerate(groups, start=1):
        domains: dict[str, Any] = {}
        for domain in DOMAINS:
            record_path = root / "records" / f"{domain}_step{step:02d}_{group_id}.json"
            if not record_path.is_file():
                raise FileNotFoundError(record_path)
            record = read_json(record_path)
            if (
                record.get("status") != "PASS"
                or record.get("method") != method
                or record.get("domain") != domain
                or int(record.get("step", -1)) != step
                or record.get("group_id") != group_id
            ):
                raise RuntimeError(f"{method}: invalid raw evidence {record_path}")
            domains[domain] = compact_domain_record(record, record_path)
        entry = {
            "sequence_step": step,
            "candidate_index": step,
            "group_id": group_id,
            "local_pruning_percent": 37.5,
            "domains": domains,
        }
        entries.append(entry)
        csv_rows.append(
            {
                "sequence_step": step,
                "candidate_index": step,
                "group_id": group_id,
                "local_pruning_percent": 37.5,
                "GEN2_parameters_after": domains["GEN2"]["parameters_after"],
                "NGN2_parameters_after": domains["NGN2"]["parameters_after"],
                "GEN2_reduction_percent": domains["GEN2"]["cumulative_parameter_reduction_percent"],
                "NGN2_reduction_percent": domains["NGN2"]["cumulative_parameter_reduction_percent"],
                "GEN2_selected_indices": ";".join(map(str, domains["GEN2"]["selected_indices"])),
                "NGN2_selected_indices": ";".join(map(str, domains["NGN2"]["selected_indices"])),
                "GEN2_structure_record": domains["GEN2"]["structure_record"],
                "NGN2_structure_record": domains["NGN2"]["structure_record"],
            }
        )

    finals: dict[str, Any] = {}
    for domain in DOMAINS:
        final = manifest["final_models"][domain]
        model_path = PROJECT_ROOT / final["model"]
        if not model_path.is_file() or sha256(model_path) != str(final["sha256"]):
            raise RuntimeError(f"{method}: final {domain} raw model is missing or changed")
        if int(final["parameters_after"]) != entries[-1]["domains"][domain]["parameters_after"]:
            raise RuntimeError(f"{method}: final {domain} parameter count disagrees with raw steps")
        finals[domain] = {
            "model": str(final["model"]),
            "sha256": str(final["sha256"]),
            "parameters_before": int(final["parameters_before"]),
            "parameters_after": int(final["parameters_after"]),
            "parameter_reduction_percent": float(final["parameter_reduction_percent"]),
        }

    plan = {
        "schema": SCHEMA,
        "method": method,
        "source_raw_manifest": relative(manifest_path),
        "source_raw_manifest_sha256": sha256(manifest_path),
        "baseline_parameters": BASELINE_PARAMETERS,
        "accepted_group_count": len(entries),
        "replay_policy": "exact per-domain raw-T5 selected indices; no re-ranking or live re-selection during T7",
        "entries": entries,
        "final_models": finals,
    }
    return plan, csv_rows


def write(method: str) -> dict[str, Any]:
    root = raw_root(method)
    plan, rows = build(method)
    plan_path = root / "frozen_pruning_plan.json"
    csv_path = root / "tables" / "FROZEN_PRUNING_PLAN.csv"
    # Plans are generated solely from immutable completed evidence.  Refuse to
    # silently replace a different plan if future scripts produce a mismatch.
    serialized = json.dumps(plan, indent=2, sort_keys=True) + "\n"
    if plan_path.is_file() and plan_path.read_text(encoding="utf-8") != serialized:
        raise RuntimeError(f"Refusing to overwrite non-identical frozen plan: {plan_path}")
    if not plan_path.is_file():
        atomic_json(plan_path, plan)
    atomic_csv(csv_path, rows)
    return {
        "method": method,
        "frozen_plan": relative(plan_path),
        "frozen_plan_sha256": sha256(plan_path),
        "frozen_plan_csv": relative(csv_path),
        "accepted_group_count": plan["accepted_group_count"],
        "final_parameter_reduction_percent": {
            domain: plan["final_models"][domain]["parameter_reduction_percent"]
            for domain in DOMAINS
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    args = parser.parse_args()
    print(json.dumps(write(args.method), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
