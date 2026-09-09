"""Build canonical DepGraph catalogues for the frozen T1/T2 sensitivity sweep.

This script is intentionally model-free.  It reads only the completed
``week1_safe_v3`` manifests and streamed dependency records, validates their
internal consistency, and writes deterministic CSV/JSON/Markdown catalogues.
It does not load a checkpoint, prune a model, or evaluate accuracy.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
RUN_ROOT = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"Invalid JSON at {path}:{line_number}") from error
    return records


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def group_number(group_id: str) -> int:
    if not group_id.startswith("G"):
        raise RuntimeError(f"Unexpected DepGraph group ID: {group_id}")
    return int(group_id[1:])


def operation_signature(record: dict[str, Any]) -> tuple[tuple[Any, ...], ...]:
    """Return a root-independent physical dependency signature.

    Exact representative index mappings are included.  Two root entries are
    aliases only when they request the same handler, target, type, and channel
    mapping across the entire generated dependency group.
    """

    operations = []
    for operation in record["operations"]:
        operations.append(
            (
                operation["target_module_path"],
                operation["target_module_type"],
                operation["handler"],
                tuple(int(index) for index in operation["indices"]),
            )
        )
    return tuple(sorted(operations))


def signature_hash(signature: tuple[tuple[Any, ...], ...]) -> str:
    encoded = json.dumps(signature, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("; ", "<br>")


def custom_family(reason: str) -> str:
    if reason.startswith("C2PSA"):
        return "C2PSA paired split/attention"
    if reason.startswith("C3k2"):
        return "C3k2/C3k split/residual"
    return "Manual structural rule"


def protected_family(reason: str) -> str:
    if "attention" in reason:
        return "Packed attention projection"
    if "Detect" in reason:
        return "Detect-head internal root"
    return "Protected structural root"


def main() -> int:
    required = [
        RUN_ROOT / "validated_group_manifest.csv",
        RUN_ROOT / "protected_root_manifest.csv",
        RUN_ROOT / "gen" / "groups_streamed.jsonl",
        RUN_ROOT / "snow" / "groups_streamed.jsonl",
        RUN_ROOT / "gen" / "module_inventory.csv",
        RUN_ROOT / "snow" / "module_inventory.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing canonical Week 1 inputs: " + ", ".join(missing))

    manifest_rows = read_csv(RUN_ROOT / "validated_group_manifest.csv")
    protected_rows = read_csv(RUN_ROOT / "protected_root_manifest.csv")
    gen_inventory = read_csv(RUN_ROOT / "gen" / "module_inventory.csv")
    snow_inventory_rows = read_csv(RUN_ROOT / "snow" / "module_inventory.csv")
    gen_records_rows = read_jsonl(RUN_ROOT / "gen" / "groups_streamed.jsonl")
    snow_records_rows = read_jsonl(RUN_ROOT / "snow" / "groups_streamed.jsonl")

    if len(manifest_rows) != 72:
        raise RuntimeError(f"Expected 72 candidate-root rows, found {len(manifest_rows)}")
    if len(protected_rows) != 54:
        raise RuntimeError(f"Expected 54 protected-root rows, found {len(protected_rows)}")
    if len(gen_inventory) != 126 or len(snow_inventory_rows) != 126:
        raise RuntimeError("Expected 126 Conv2d inventory rows in each domain")

    def unique_by(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            value = str(row[key])
            if value in result:
                raise RuntimeError(f"Duplicate {key}: {value}")
            result[value] = row
        return result

    manifest = unique_by(manifest_rows, "root_module_path")
    protected = unique_by(protected_rows, "module_path")
    snow_inventory = unique_by(snow_inventory_rows, "module_path")
    gen_records = unique_by(gen_records_rows, "root_module_path")
    snow_records = unique_by(snow_records_rows, "root_module_path")

    gen_paths = [row["module_path"] for row in gen_inventory]
    if set(gen_paths) != set(snow_inventory):
        raise RuntimeError("GEN and SNOW Conv2d inventories do not contain the same paths")
    if set(manifest) != set(gen_records) or set(manifest) != set(snow_records):
        raise RuntimeError("Candidate manifest and streamed group paths do not match")
    if set(manifest).intersection(protected):
        raise RuntimeError("A root cannot be both a candidate and protected")
    if set(manifest).union(protected) != set(gen_paths):
        raise RuntimeError("The candidate/protected split does not cover all 126 roots")

    for root in manifest:
        if operation_signature(gen_records[root]) != operation_signature(snow_records[root]):
            raise RuntimeError(f"GEN/SNOW dependency signature mismatch at {root}")

    validated = [row for row in manifest_rows if row["final_status"] == "VALIDATED_COMMON_GROUP"]
    custom = [row for row in manifest_rows if row["final_status"] == "CUSTOM_RULE_REQUIRED"]
    incomplete = [row for row in manifest_rows if row["final_status"] not in {"VALIDATED_COMMON_GROUP", "CUSTOM_RULE_REQUIRED"}]
    if len(validated) != 53 or len(custom) != 19 or incomplete:
        raise RuntimeError(
            f"Unexpected physical status counts: validated={len(validated)}, "
            f"custom={len(custom)}, incomplete={len(incomplete)}"
        )

    signatures: dict[tuple[tuple[Any, ...], ...], list[dict[str, str]]] = defaultdict(list)
    for row in validated:
        signatures[operation_signature(gen_records[row["root_module_path"]])].append(row)
    ordered_signature_groups = sorted(
        signatures.items(),
        key=lambda item: min(group_number(row["group_id"]) for row in item[1]),
    )
    if len(ordered_signature_groups) != 42:
        raise RuntimeError(f"Expected 42 unique validated groups, found {len(ordered_signature_groups)}")

    canonical_rows: list[dict[str, Any]] = []
    canonical_operations: dict[str, Any] = {}
    root_to_canonical: dict[str, str] = {}
    representative_by_group: dict[str, str] = {}

    for index, (signature, aliases) in enumerate(ordered_signature_groups, start=1):
        canonical_id = f"DG{index:03d}"
        aliases = sorted(aliases, key=lambda row: group_number(row["group_id"]))
        representative = aliases[0]
        roots = [row["root_module_path"] for row in aliases]
        records = [gen_records[root] for root in roots]
        widths = {int(record["root_out_channels"]) for record in records}
        probe_sets = {tuple(int(value) for value in record["probe_indices"]) for record in records}
        removed_gen = {int(row["gen_parameters_removed"]) for row in aliases}
        removed_snow = {int(row["snow_parameters_removed"]) for row in aliases}
        if len(widths) != 1 or len(probe_sets) != 1:
            raise RuntimeError(f"Alias roots disagree on width or probes for {canonical_id}")
        if removed_gen != removed_snow or len(removed_gen) != 1:
            raise RuntimeError(f"Alias/domain pilot removals disagree for {canonical_id}")
        width = next(iter(widths))
        if width % 8:
            raise RuntimeError(f"Validated root width is not divisible by eight: {canonical_id}")
        dependent_paths = sorted(
            {
                operation[0]
                for operation in signature
                if operation[0] != "<autograd-operation>"
            }
        )
        signature_digest = signature_hash(signature)
        for root in roots:
            root_to_canonical[root] = canonical_id
        representative_by_group[canonical_id] = representative["root_module_path"]
        canonical_rows.append(
            {
                "canonical_group_id": canonical_id,
                "representative_original_group_id": representative["group_id"],
                "representative_root": representative["root_module_path"],
                "alias_count": len(roots),
                "original_group_ids": "; ".join(row["group_id"] for row in aliases),
                "alias_roots": "; ".join(roots),
                "root_out_channels": width,
                "probe_indices": ";".join(str(value) for value in next(iter(probe_sets))),
                "dependency_operation_count": len(signature),
                "dependent_module_paths": "; ".join(dependent_paths),
                "operation_signature_sha256": signature_digest,
                "pilot_parameters_removed_for_3_indices": next(iter(removed_gen)),
                "gen_physical_status": "PASS",
                "snow_physical_status": "PASS",
                "t1_t2_status": "PRIMARY_SWEEP",
            }
        )
        canonical_operations[canonical_id] = {
            "canonical_group_id": canonical_id,
            "representative_root": representative["root_module_path"],
            "original_group_ids": [row["group_id"] for row in aliases],
            "alias_roots": roots,
            "root_out_channels": width,
            "representative_probe_indices": list(next(iter(probe_sets))),
            "operation_signature_sha256": signature_digest,
            "operations": [
                {
                    "target_module_path": operation[0],
                    "target_module_type": operation[1],
                    "handler": operation[2],
                    "representative_indices": list(operation[3]),
                }
                for operation in signature
            ],
        }

    canonical_fields = [
        "canonical_group_id",
        "representative_original_group_id",
        "representative_root",
        "alias_count",
        "original_group_ids",
        "alias_roots",
        "root_out_channels",
        "probe_indices",
        "dependency_operation_count",
        "dependent_module_paths",
        "operation_signature_sha256",
        "pilot_parameters_removed_for_3_indices",
        "gen_physical_status",
        "snow_physical_status",
        "t1_t2_status",
    ]
    canonical_path = RUN_ROOT / "canonical_validated_groups.csv"
    operations_path = RUN_ROOT / "canonical_group_operations.json"
    write_csv(canonical_path, canonical_fields, canonical_rows)
    write_json(operations_path, canonical_operations)

    crosswalk_rows: list[dict[str, Any]] = []
    for index, inventory_row in enumerate(gen_inventory, start=1):
        root = inventory_row["module_path"]
        snow_row = snow_inventory[root]
        if root in manifest:
            decision = manifest[root]
            original_group_id = decision["group_id"]
            if decision["final_status"] == "VALIDATED_COMMON_GROUP":
                canonical_id = root_to_canonical[root]
                representative = representative_by_group[canonical_id]
                role = "REPRESENTATIVE" if root == representative else "ALIAS"
                family = "DepGraph-validated structured group"
                disposition = f"INCLUDED_ONCE_AS_{canonical_id}"
                reason = decision["decision_reason"]
            else:
                canonical_id = ""
                representative = ""
                role = "CUSTOM_ROOT_ENTRY"
                family = custom_family(decision["decision_reason"])
                disposition = "DEFERRED_UNTIL_CUSTOM_RULE_VALIDATION"
                reason = decision["decision_reason"]
            category = decision["final_status"]
        else:
            decision = protected[root]
            original_group_id = ""
            canonical_id = ""
            representative = ""
            role = "PROTECTED_ROOT"
            family = protected_family(decision["reason"])
            disposition = "EXCLUDED_AS_INDEPENDENT_T1_T2_ROOT"
            reason = decision["reason"]
            category = decision["status"]
        crosswalk_rows.append(
            {
                "conv_root_id": f"R{index:03d}",
                "module_path": root,
                "module_type": inventory_row["module_type"],
                "in_channels": inventory_row["in_channels"],
                "gen_out_channels": inventory_row["out_channels"],
                "snow_out_channels": snow_row["out_channels"],
                "original_depgraph_id": original_group_id,
                "category": category,
                "canonical_group_id": canonical_id,
                "canonical_representative_root": representative,
                "root_role": role,
                "rule_or_block_family": family,
                "t1_t2_disposition": disposition,
                "reason": reason,
            }
        )

    crosswalk_fields = [
        "conv_root_id",
        "module_path",
        "module_type",
        "in_channels",
        "gen_out_channels",
        "snow_out_channels",
        "original_depgraph_id",
        "category",
        "canonical_group_id",
        "canonical_representative_root",
        "root_role",
        "rule_or_block_family",
        "t1_t2_disposition",
        "reason",
    ]
    crosswalk_path = RUN_ROOT / "full_root_crosswalk.csv"
    write_csv(crosswalk_path, crosswalk_fields, crosswalk_rows)

    custom_rows = sorted(custom, key=lambda row: group_number(row["group_id"]))
    custom_components: dict[tuple[tuple[Any, ...], ...], list[str]] = defaultdict(list)
    for row in custom_rows:
        # Ignore the representative probe mapping only for describing overlapping
        # connected structures.  These are not treated as canonical prune groups.
        operation_set = tuple(
            sorted(
                (
                    operation["target_module_path"],
                    operation["target_module_type"],
                    operation["handler"],
                )
                for operation in gen_records[row["root_module_path"]]["operations"]
            )
        )
        custom_components[operation_set].append(row["root_module_path"])

    catalogue_lines = [
        "# YOLO26n Dependency-Group Catalogue v1",
        "",
        "Status: **COMPLETE for the generic DepGraph audit; frozen for the initial T1/T2 sweep on 2026-07-19.**",
        "",
        "This catalogue converts the completed GEN/SNOW dependency audit into unique experimental units. A pruning root is only an entry point into the graph; several roots can describe the same physical channel-removal group. Accuracy has not been measured for these groups yet.",
        "",
        "## Coverage",
        "",
        "Table 1 accounts for every Conv2d location in each YOLO26n baseline and separates physical groups from raw root entries.",
        "",
        "| Item | Count | Interpretation |",
        "|---|---:|---|",
        "| Conv2d locations per model | 126 | Complete root-level inventory |",
        "| Protected independent roots | 54 | 48 Detect-head and 6 packed-attention roots |",
        "| Candidate root entries generated by DepGraph | 72 | Roots checked on the shared GEN/SNOW topology |",
        "| Root entries passing physical validation | 53 | Includes aliases of the same physical group |",
        "| Unique validated physical groups | 42 | Primary T1/T2 experimental units |",
        "| Duplicate validated root aliases removed | 11 | Retained in the crosswalk, not re-evaluated |",
        "| Root entries requiring custom rules | 19 | 17 C3k2/C3k and 2 C2PSA entries |",
        f"| Overlapping custom dependency components | {len(custom_components)} | Descriptive only; final groups require custom-rule validation |",
        "",
        "Physical validation means that representative channel removal reduced parameters, completed a 640 x 640 forward pass with finite outputs, and passed save/reload plus a second forward pass on scratch GEN and SNOW models. It does not establish accuracy retention.",
        "",
        "## Validated groups entering T1 and T2",
        "",
        "Table 2 lists the 42 unique dependency groups. Alias paths are retained so that every successful root remains traceable without repeating the same experiment.",
        "",
        "| Group | Representative root | Root aliases | Channels | Dependency operations |",
        "|---|---|---|---:|---:|",
    ]
    for row in canonical_rows:
        alias_roots = row["alias_roots"].split("; ")
        non_representative = [root for root in alias_roots if root != row["representative_root"]]
        aliases_text = "; ".join(non_representative) if non_representative else "-"
        catalogue_lines.append(
            "| {group} | `{root}` | {aliases} | {channels} | {operations} |".format(
                group=row["canonical_group_id"],
                root=row["representative_root"],
                aliases=markdown_cell("; ".join(f"`{value}`" for value in non_representative) if non_representative else aliases_text),
                channels=row["root_out_channels"],
                operations=row["dependency_operation_count"],
            )
        )

    catalogue_lines.extend(
        [
            "",
            "## Entries requiring custom pruning rules",
            "",
            "Table 3 retains every structurally constrained root. These entries are not assigned an accuracy drop until a reusable block rule passes the same physical tests as the generic groups.",
            "",
            "| Original ID | Root | Rule family | Operations | Current decision |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in custom_rows:
        catalogue_lines.append(
            f"| {row['group_id']} | `{row['root_module_path']}` | {custom_family(row['decision_reason'])} | {row['gen_depgraph_operations']} | Deferred pending custom-rule validation |"
        )

    catalogue_lines.extend(
        [
            "",
            "## Protected roots",
            "",
            "Table 4 summarizes roots that are not legal independent T1/T2 entry points. They can still appear as dependent operations when an accepted upstream group is pruned.",
            "",
            "| Protected family | Roots | Reason |",
            "|---|---:|---|",
            "| Detect-head internals | 48 | Class-dependent and one-to-many/one-to-one output structure is protected during the first sweep |",
            "| Packed attention projections | 6 | Independent removal requires a head-aware rule |",
            "",
            "The row-level `full_root_crosswalk.csv` is the complete 126-root appendix. `canonical_group_operations.json` contains the exact handler, target and representative index mapping for every validated physical group.",
            "",
            "## Experimental decision",
            "",
            "T1 and T2 will initially contain one row for each of the 42 validated physical groups. The 19 custom-rule entries are reported as deferred rather than assigned artificial zero sensitivity. Protected roots remain excluded as independent pruning decisions. A later validated custom-rule extension may add new block-level groups without changing the initial catalogue.",
            "",
        ]
    )
    supervisor_path = RUN_ROOT / "SUPERVISOR_GROUP_CATALOGUE.md"
    write_text(supervisor_path, "\n".join(catalogue_lines))

    summary = {
        "catalogue_id": "YOLO26N_DEPGRAPH_CATALOGUE_V1",
        "status": "COMPLETE_FOR_GENERIC_T1_T2_SWEEP",
        "frozen_local_date": "2026-07-19",
        "conv2d_roots_per_model": 126,
        "protected_root_entries": 54,
        "depgraph_candidate_root_entries": 72,
        "physically_validated_root_entries": 53,
        "unique_validated_physical_groups": 42,
        "duplicate_validated_alias_entries": 11,
        "custom_rule_root_entries": 19,
        "custom_overlapping_operation_components": len(custom_components),
        "initial_t1_t2_groups": 42,
        "accuracy_evaluation_performed": False,
        "source_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
            for path in required
        },
        "outputs_sha256": {
            str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): sha256(path)
            for path in [canonical_path, operations_path, crosswalk_path, supervisor_path]
        },
    }
    summary_path = RUN_ROOT / "canonical_catalogue_summary.json"
    write_json(summary_path, summary)

    print(json.dumps({key: summary[key] for key in [
        "conv2d_roots_per_model",
        "physically_validated_root_entries",
        "unique_validated_physical_groups",
        "duplicate_validated_alias_entries",
        "custom_rule_root_entries",
        "protected_root_entries",
    ]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
