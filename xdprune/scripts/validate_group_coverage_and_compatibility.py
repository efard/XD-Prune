"""Validate YOLO26n root coverage and statically screen all 51 group pairs."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import re
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
SOURCE_ROOT = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3"
CUSTOM_ROOT = STUDY_ROOT / "results" / "depgraph" / "custom_groups_v1"
GENERIC_SWEEP = STUDY_ROOT / "results" / "pruning" / "t1_t2_full_sweep_v2"
CUSTOM_SWEEP = STUDY_ROOT / "results" / "pruning" / "t1_t2_custom_sweep_v1"
DEFAULT_OUTPUT = STUDY_ROOT / "results" / "depgraph" / "group_coverage_validation_v1"
SPEC_PATH = STUDY_ROOT / "GROUP_COVERAGE_VALIDATION_SPEC_V1.md"

CROSSWALK_PATH = SOURCE_ROOT / "full_root_crosswalk.csv"
GENERIC_CATALOGUE_PATH = SOURCE_ROOT / "canonical_validated_groups.csv"
GENERIC_OPERATIONS_PATH = SOURCE_ROOT / "canonical_group_operations.json"
CUSTOM_CATALOGUE_PATH = CUSTOM_ROOT / "CUSTOM_GROUP_CATALOGUE.csv"
CUSTOM_OPERATIONS_PATH = CUSTOM_ROOT / "CUSTOM_GROUP_OPERATIONS.json"
COMBINED_SUPERVISOR_PATH = GENERIC_SWEEP / "tables" / "SUPERVISOR_FOCUSED_T1_T2.csv"

CHECKPOINTS = {
    "GEN": STUDY_ROOT / "results" / "baselines" / "b_gen_mio_yolo26n_s42_v1" / "weights" / "best.pt",
    "SNOW": STUDY_ROOT / "results" / "baselines" / "b_snow_acdc_yolo26n_s42_v1" / "weights" / "best.pt",
}
EXPECTED_CHECKPOINT_HASHES = {
    "GEN": "D0250E9FFD459C2C46A01AC81421D95E9E1D0BB35799B5DEDE30C92E7B94D250",
    "SNOW": "5C9BCEBA2872E98EE1BEB967E10E4CA10E17BEA6A8F8200E306BCDAE067BDB68",
}

FIXED_PREDICTION_PATTERN = re.compile(
    r"^model\.23\.(?:one2one_)?cv[23]\.\d+\.2$"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def split_values(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def group_sort_key(group_id: str) -> tuple[int, int]:
    if group_id.startswith("DG"):
        return (0, int(group_id[2:]))
    if group_id.startswith("CDG"):
        return (1, int(group_id[3:]))
    raise ValueError(f"Unexpected group ID: {group_id}")


def live_conv_inventory(checkpoint: Path) -> list[dict[str, Any]]:
    import torch
    from ultralytics import YOLO

    os.environ.setdefault("YOLO_OFFLINE", "true")
    yolo = YOLO(str(checkpoint), task="detect")
    model = yolo.model.float().cpu().eval()
    rows = []
    for path, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            rows.append(
                {
                    "module_path": path,
                    "module_type": type(module).__name__,
                    "in_channels": int(module.in_channels),
                    "out_channels": int(module.out_channels),
                    "groups": int(module.groups),
                }
            )
    del model, yolo
    gc.collect()
    return rows


def operation_axes(handler: str) -> tuple[str, ...]:
    if handler == "prune_out_channels":
        return ("OUT",)
    if handler == "prune_in_channels":
        return ("IN",)
    if handler == "prune_depthwise_channels":
        return ("IN", "OUT")
    raise RuntimeError(f"Unknown pruning handler: {handler}")


def build_footprints(
    generic_operations: dict[str, Any], custom_operations: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, set[tuple[str, str]]], dict[str, set[str]]]:
    rows: list[dict[str, Any]] = []
    axes_by_group: dict[str, set[tuple[str, str]]] = defaultdict(set)
    paths_by_group: dict[str, set[str]] = defaultdict(set)

    for group_id, group in generic_operations.items():
        for operation_index, operation in enumerate(group["operations"], start=1):
            module_path = str(operation["target_module_path"])
            handler = str(operation["handler"])
            real_module = not module_path.startswith("<")
            axes = operation_axes(handler)
            for axis in axes:
                rows.append(
                    {
                        "group_id": group_id,
                        "group_source": "GENERIC_DEPGRAPH",
                        "operation_index": operation_index,
                        "module_path": module_path,
                        "module_type": operation["target_module_type"],
                        "handler": handler,
                        "channel_axis": axis if real_module else "GRAPH_OPERATION",
                        "real_module_path": real_module,
                        "channels_before": "",
                        "channels_after": "",
                        "indices_removed_in_physical_probe": len(operation.get("representative_indices", [])),
                    }
                )
                if real_module:
                    axes_by_group[group_id].add((module_path, axis))
                    paths_by_group[group_id].add(module_path)

    for group_id, group in custom_operations.items():
        for operation_index, operation in enumerate(group["operation_skeleton"], start=1):
            module_path = str(operation["module_path"])
            handler = str(operation["operation"])
            for axis in operation_axes(handler):
                rows.append(
                    {
                        "group_id": group_id,
                        "group_source": "CUSTOM_BLOCK_RULE",
                        "operation_index": operation_index,
                        "module_path": module_path,
                        "module_type": "custom_rule_target",
                        "handler": handler,
                        "channel_axis": axis,
                        "real_module_path": True,
                        "channels_before": operation["channels_before"],
                        "channels_after": operation["channels_after"],
                        "indices_removed_in_physical_probe": operation["indices_removed"],
                    }
                )
                axes_by_group[group_id].add((module_path, axis))
                paths_by_group[group_id].add(module_path)
    return rows, axes_by_group, paths_by_group


def audit_accuracy_runs(
    folder: Path, groups: list[str], expected_schema: str
) -> dict[str, Any]:
    records = []
    for path in sorted(folder.glob("*.json")):
        record = read_json(path)
        if record.get("schema_version") != expected_schema:
            raise RuntimeError(f"Unexpected schema in {path}")
        records.append(record)
    expected = {(domain, group_id) for group_id in groups for domain in ("GEN", "SNOW")}
    observed = {(record["domain"], record["canonical_group_id"]) for record in records}
    if len(records) != len(expected) or len(observed) != len(expected) or observed != expected:
        raise RuntimeError(f"Accuracy-run membership mismatch in {folder}")
    failures = [record["run_id"] for record in records if record.get("status") != "PASS"]
    if failures:
        raise RuntimeError(f"Non-PASS accuracy records: {failures}")
    if any(record.get("canonical_checkpoint_modified") is not False for record in records):
        raise RuntimeError(f"Checkpoint-integrity evidence failed in {folder}")
    return {"records": len(records), "unique_pairs": len(observed), "failed": 0}


def pairwise_compatibility(
    groups: list[str],
    axes_by_group: dict[str, set[tuple[str, str]]],
    paths_by_group: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str], Counter[tuple[str, str]]]:
    matrix_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    source_counts: Counter[tuple[str, str]] = Counter()
    classification_by_pair: dict[tuple[str, str], str] = {}

    for left_index, group_a in enumerate(groups):
        for group_b in groups[left_index + 1 :]:
            shared_paths = sorted(paths_by_group[group_a] & paths_by_group[group_b])
            same_axis = sorted(axes_by_group[group_a] & axes_by_group[group_b])
            if same_axis:
                classification = "SHARED_TENSOR_AXIS"
                interpretation = "Potential direct overlap; cumulative runtime validation is required"
            elif shared_paths:
                classification = "SHARED_MODULE_DIFFERENT_AXIS"
                interpretation = "Order-dependent connection; recalculate the second group's importance and indices"
            else:
                classification = "NO_RECORDED_OVERLAP"
                interpretation = "No overlap in isolated operation records; cumulative validity is not yet proven"
            classification_by_pair[(group_a, group_b)] = classification
            classification_by_pair[(group_b, group_a)] = classification
            counts[classification] += 1
            pair_family = (
                "GENERIC_GENERIC" if group_a.startswith("DG") and group_b.startswith("DG")
                else "CUSTOM_CUSTOM" if group_a.startswith("CDG") and group_b.startswith("CDG")
                else "GENERIC_CUSTOM"
            )
            source_counts[(pair_family, classification)] += 1
            detail_rows.append(
                {
                    "group_a": group_a,
                    "group_b": group_b,
                    "pair_family": pair_family,
                    "classification": classification,
                    "shared_real_module_count": len(shared_paths),
                    "shared_real_modules": ";".join(shared_paths),
                    "shared_tensor_axis_count": len(same_axis),
                    "shared_tensor_axes": ";".join(f"{path}:{axis}" for path, axis in same_axis),
                    "priority_runtime_check_due_to_recorded_overlap": classification != "NO_RECORDED_OVERLAP",
                    "interpretation": interpretation,
                }
            )

    for group_a in groups:
        row: dict[str, Any] = {"group_id": group_a}
        for group_b in groups:
            row[group_b] = "SELF" if group_a == group_b else classification_by_pair[(group_a, group_b)]
        matrix_rows.append(row)
    return matrix_rows, detail_rows, counts, source_counts


def status_markdown(summary: dict[str, Any]) -> str:
    roots = summary["root_coverage"]
    pairs = summary["pairwise_compatibility"]
    return "\n".join(
        [
            "# YOLO26n Group-Coverage Validation V1",
            "",
            f"Overall status: **{summary['overall_status']}**",
            "",
            "## Root inventory",
            "",
            "| Classification | Count |",
            "|---|---:|",
            f"| Live Conv2d locations | {roots['live_conv2d_locations']} |",
            f"| Generic root entries | {roots['generic_root_entries']} |",
            f"| Generic canonical groups | {roots['generic_groups']} |",
            f"| Custom logical-root entries | {roots['custom_logical_root_entries']} |",
            f"| Custom canonical groups | {roots['custom_groups']} |",
            f"| Protected attention roots covered dependently | {roots['custom_dependent_attention_roots']} |",
            f"| Fixed box/class prediction outputs | {roots['fixed_prediction_outputs']} |",
            f"| Intermediate Detect roots deferred by scope | {roots['deferred_detect_intermediate_roots']} |",
            f"| Unaccounted roots | {roots['unaccounted_roots']} |",
            "",
            "The 51-group catalogue completely covers the frozen 72-root candidate scope. It is not an exhaustive catalogue of every conceivable YOLO26n pruning opportunity because 36 intermediate Detect-head roots remain scope-deferred.",
            "",
            "## Static pair screening",
            "",
            "| Classification | Unique pairs |",
            "|---|---:|",
            f"| No recorded overlap | {pairs['NO_RECORDED_OVERLAP']} |",
            f"| Shared module, different axes | {pairs['SHARED_MODULE_DIFFERENT_AXIS']} |",
            f"| Shared tensor axis | {pairs['SHARED_TENSOR_AXIS']} |",
            f"| Total | {pairs['total_unique_pairs']} |",
            "",
            "Static labels screen combinations; they do not replace cumulative physical validation.",
            "",
        ]
    )


def build(output: Path) -> dict[str, Any]:
    output = output.resolve()
    allowed = (STUDY_ROOT / "results" / "depgraph").resolve()
    if allowed != output and allowed not in output.parents:
        raise RuntimeError(f"Output must remain below {allowed}")
    output.mkdir(parents=True, exist_ok=True)

    inputs = [
        Path(__file__).resolve(),
        SPEC_PATH,
        CROSSWALK_PATH,
        GENERIC_CATALOGUE_PATH,
        GENERIC_OPERATIONS_PATH,
        CUSTOM_CATALOGUE_PATH,
        CUSTOM_OPERATIONS_PATH,
        COMBINED_SUPERVISOR_PATH,
        *CHECKPOINTS.values(),
    ]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    for domain, checkpoint in CHECKPOINTS.items():
        actual_hash = sha256(checkpoint)
        if actual_hash != EXPECTED_CHECKPOINT_HASHES[domain]:
            raise RuntimeError(f"Frozen {domain} checkpoint hash changed")

    crosswalk = read_csv(CROSSWALK_PATH)
    generic_catalogue = read_csv(GENERIC_CATALOGUE_PATH)
    custom_catalogue = read_csv(CUSTOM_CATALOGUE_PATH)
    generic_operations = read_json(GENERIC_OPERATIONS_PATH)
    custom_operations = read_json(CUSTOM_OPERATIONS_PATH)
    if len(crosswalk) != 126 or len({row["module_path"] for row in crosswalk}) != 126:
        raise RuntimeError("The frozen Conv2d crosswalk is not 126 unique paths")

    generic_groups = sorted((row["canonical_group_id"] for row in generic_catalogue), key=group_sort_key)
    custom_groups = sorted((row["custom_group_id"] for row in custom_catalogue), key=group_sort_key)
    groups = [*generic_groups, *custom_groups]
    if len(generic_groups) != 42 or len(custom_groups) != 9 or len(set(groups)) != 51:
        raise RuntimeError("Expected 42 generic plus nine custom groups")
    if set(generic_operations) != set(generic_groups) or set(custom_operations) != set(custom_groups):
        raise RuntimeError("Operation-record membership differs from the catalogues")
    if any(row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS" for row in generic_catalogue):
        raise RuntimeError("A generic group lacks two-domain physical validation")
    if any(row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS" for row in custom_catalogue):
        raise RuntimeError("A custom group lacks two-domain physical validation")

    live = {domain: live_conv_inventory(path) for domain, path in CHECKPOINTS.items()}
    live_by_domain = {
        domain: {row["module_path"]: row for row in rows} for domain, rows in live.items()
    }
    frozen_paths = {row["module_path"] for row in crosswalk}
    for domain in ("GEN", "SNOW"):
        if len(live[domain]) != 126 or set(live_by_domain[domain]) != frozen_paths:
            raise RuntimeError(f"Live {domain} Conv2d inventory differs from the 126-root crosswalk")
    if set(live_by_domain["GEN"]) != set(live_by_domain["SNOW"]):
        raise RuntimeError("GEN and SNOW Conv2d paths differ")
    for row in crosswalk:
        path = row["module_path"]
        gen = live_by_domain["GEN"][path]
        snow = live_by_domain["SNOW"][path]
        for key in ("module_path", "module_type", "in_channels", "groups"):
            if gen[key] != snow[key]:
                raise RuntimeError(f"GEN/SNOW live convolution property {key} differs at {path}")
        if int(row["in_channels"]) != gen["in_channels"]:
            raise RuntimeError(f"Crosswalk input width differs at {path}")
        if int(row["gen_out_channels"]) != gen["out_channels"] or int(row["snow_out_channels"]) != snow["out_channels"]:
            raise RuntimeError(f"Crosswalk output width differs at {path}")

    footprint_rows, axes_by_group, paths_by_group = build_footprints(
        generic_operations, custom_operations
    )
    custom_root_to_group: dict[str, str] = {}
    custom_representative_root: dict[str, str] = {}
    for row in custom_catalogue:
        roots = split_values(row["original_root_paths"])
        if len(roots) != int(row["original_deferred_root_count"]):
            raise RuntimeError(f"Custom root count differs for {row['custom_group_id']}")
        for root in roots:
            if root in custom_root_to_group:
                raise RuntimeError(f"Custom root is assigned twice: {root}")
            custom_root_to_group[root] = row["custom_group_id"]
        custom_representative_root[row["custom_group_id"]] = roots[0]
    if len(custom_root_to_group) != 19:
        raise RuntimeError("The nine custom groups do not map exactly 19 logical roots")

    custom_path_to_groups: dict[str, list[str]] = defaultdict(list)
    for group_id in custom_groups:
        for module_path in paths_by_group[group_id]:
            custom_path_to_groups[module_path].append(group_id)

    coverage_rows: list[dict[str, Any]] = []
    unresolved = []
    for row in crosswalk:
        path = row["module_path"]
        category = row["category"]
        current_class = ""
        group_id = ""
        dependent_groups = ""
        independent = False
        rationale = ""
        if category == "VALIDATED_COMMON_GROUP":
            group_id = row["canonical_group_id"]
            independent = row["root_role"] == "REPRESENTATIVE"
            current_class = (
                "GENERIC_GROUP_REPRESENTATIVE" if independent else "GENERIC_DEPENDENCY_EQUIVALENT_ALIAS"
            )
            rationale = "Physically validated generic DepGraph root mapped to one canonical group"
        elif category == "CUSTOM_RULE_REQUIRED":
            group_id = custom_root_to_group.get(path, "")
            if not group_id:
                unresolved.append(path)
            independent = bool(group_id) and path == custom_representative_root[group_id]
            current_class = "CUSTOM_LOGICAL_ROOT_ENTRY"
            rationale = "Original deferred root is represented by a physically validated block-aware logical group"
        elif category == "PROTECTED_NOT_A_T1_T2_ROOT" and "packed attention" in row["reason"]:
            mapped = sorted(custom_path_to_groups.get(path, []), key=group_sort_key)
            dependent_groups = ";".join(mapped)
            if not mapped:
                unresolved.append(path)
            current_class = "CUSTOM_DEPENDENT_ATTENTION_ROOT"
            rationale = "Not an independent root; its channel axes are modified inside the head-aware custom rule"
        elif category == "PROTECTED_NOT_A_T1_T2_ROOT" and path.startswith("model.23."):
            if FIXED_PREDICTION_PATTERN.match(path):
                current_class = "FIXED_PREDICTION_OUTPUT"
                rationale = "Box-distribution or class-logit output width is semantically fixed"
            else:
                current_class = "DEFERRED_DETECT_INTERMEDIATE_ROOT"
                rationale = "Originally protected by study scope; potentially prunable only after a dedicated Detect-head audit"
        else:
            current_class = "UNACCOUNTED"
            rationale = "No current coverage rule"
            unresolved.append(path)
        coverage_rows.append(
            {
                "conv_root_id": row["conv_root_id"],
                "module_path": path,
                "module_type": row["module_type"],
                "in_channels": int(row["in_channels"]),
                "gen_out_channels": int(row["gen_out_channels"]),
                "snow_out_channels": int(row["snow_out_channels"]),
                "groups": live_by_domain["GEN"][path]["groups"],
                "original_category": category,
                "current_coverage_class": current_class,
                "canonical_group_id": group_id,
                "covered_as_dependency_by_custom_groups": dependent_groups,
                "independent_root_representative": independent,
                "included_in_frozen_51_group_catalogue": bool(group_id),
                "requires_future_dedicated_rule": current_class == "DEFERRED_DETECT_INTERMEDIATE_ROOT",
                "rationale": rationale,
            }
        )
    if unresolved:
        raise RuntimeError(f"Unaccounted root paths remain: {sorted(set(unresolved))}")

    class_counts = Counter(row["current_coverage_class"] for row in coverage_rows)
    expected_class_counts = {
        "GENERIC_GROUP_REPRESENTATIVE": 42,
        "GENERIC_DEPENDENCY_EQUIVALENT_ALIAS": 11,
        "CUSTOM_LOGICAL_ROOT_ENTRY": 19,
        "CUSTOM_DEPENDENT_ATTENTION_ROOT": 6,
        "FIXED_PREDICTION_OUTPUT": 12,
        "DEFERRED_DETECT_INTERMEDIATE_ROOT": 36,
    }
    if dict(class_counts) != expected_class_counts:
        raise RuntimeError(f"Coverage classification counts differ: {dict(class_counts)}")

    generic_run_audit = audit_accuracy_runs(
        GENERIC_SWEEP / "runs", generic_groups, "t1_t2_full_sweep_run_v2"
    )
    custom_run_audit = audit_accuracy_runs(
        CUSTOM_SWEEP / "runs", custom_groups, "t1_t2_custom_sweep_run_v1"
    )
    combined_rows = read_csv(COMBINED_SUPERVISOR_PATH)
    combined_ids = [row["canonical_group_id"] for row in combined_rows]
    if len(combined_rows) != 51 or len(set(combined_ids)) != 51 or set(combined_ids) != set(groups):
        raise RuntimeError("Combined supervisor table is not the expected 51 unique groups")
    if any(any(value == "" for value in row.values()) for row in combined_rows):
        raise RuntimeError("Combined supervisor table contains blank cells")

    matrix_rows, detail_rows, pair_counts, pair_source_counts = pairwise_compatibility(
        groups, axes_by_group, paths_by_group
    )
    if len(detail_rows) != 1275 or sum(pair_counts.values()) != 1275:
        raise RuntimeError("Pairwise compatibility audit did not classify all 1,275 unique pairs")

    root_fields = [
        "conv_root_id", "module_path", "module_type", "in_channels", "gen_out_channels",
        "snow_out_channels", "groups",
        "original_category", "current_coverage_class", "canonical_group_id",
        "covered_as_dependency_by_custom_groups", "independent_root_representative",
        "included_in_frozen_51_group_catalogue", "requires_future_dedicated_rule", "rationale",
    ]
    footprint_fields = [
        "group_id", "group_source", "operation_index", "module_path", "module_type", "handler",
        "channel_axis", "real_module_path", "channels_before", "channels_after",
        "indices_removed_in_physical_probe",
    ]
    detail_fields = [
        "group_a", "group_b", "pair_family", "classification", "shared_real_module_count",
        "shared_real_modules", "shared_tensor_axis_count", "shared_tensor_axes",
        "priority_runtime_check_due_to_recorded_overlap", "interpretation",
    ]
    atomic_csv(output / "ROOT_COVERAGE.csv", root_fields, coverage_rows)
    atomic_csv(output / "GROUP_OPERATION_FOOTPRINTS.csv", footprint_fields, footprint_rows)
    atomic_csv(output / "GROUP_COMPATIBILITY_MATRIX.csv", ["group_id", *groups], matrix_rows)
    atomic_csv(output / "GROUP_COMPATIBILITY_DETAILS.csv", detail_fields, detail_rows)
    atomic_csv(
        output / "GROUP_COMPATIBILITY_SUMMARY.csv",
        ["classification", "unique_pair_count", "interpretation"],
        [
            {
                "classification": key,
                "unique_pair_count": pair_counts[key],
                "interpretation": {
                    "NO_RECORDED_OVERLAP": "No common real module path in isolated operation records",
                    "SHARED_MODULE_DIFFERENT_AXIS": "Common module but different channel axes; order-dependent",
                    "SHARED_TENSOR_AXIS": "Same module channel axis; cumulative runtime check required",
                }[key],
            }
            for key in ("NO_RECORDED_OVERLAP", "SHARED_MODULE_DIFFERENT_AXIS", "SHARED_TENSOR_AXIS")
        ],
    )
    atomic_csv(
        output / "GROUP_COMPATIBILITY_SOURCE_SUMMARY.csv",
        ["pair_family", "classification", "unique_pair_count"],
        [
            {
                "pair_family": family,
                "classification": classification,
                "unique_pair_count": pair_source_counts[(family, classification)],
            }
            for family in ("GENERIC_GENERIC", "GENERIC_CUSTOM", "CUSTOM_CUSTOM")
            for classification in (
                "NO_RECORDED_OVERLAP", "SHARED_MODULE_DIFFERENT_AXIS", "SHARED_TENSOR_AXIS"
            )
        ],
    )

    manifest = {
        relative(path): {"bytes": path.stat().st_size, "sha256": sha256(path)} for path in inputs
    }
    summary = {
        "schema_version": "yolo26n_group_coverage_validation_v1",
        "overall_status": "PASS_WITH_DECLARED_DETECT_SCOPE_BOUNDARY",
        "inventory_fully_accounted": True,
        "frozen_72_root_candidate_scope_complete": True,
        "exhaustive_all_potential_yolo26n_pruning_roots": False,
        "root_coverage": {
            "live_conv2d_locations": 126,
            "generic_root_entries": 53,
            "generic_groups": 42,
            "generic_alias_roots": 11,
            "custom_logical_root_entries": 19,
            "custom_groups": 9,
            "custom_dependent_attention_roots": 6,
            "fixed_prediction_outputs": 12,
            "deferred_detect_intermediate_roots": 36,
            "unaccounted_roots": 0,
        },
        "accuracy_evidence": {
            "generic": generic_run_audit,
            "custom": custom_run_audit,
            "total_passed_domain_group_runs": 102,
            "combined_supervisor_groups": 51,
            "combined_supervisor_blank_cells": 0,
        },
        "pairwise_compatibility": {
            "groups": 51,
            "total_unique_pairs": 1275,
            **{key: pair_counts[key] for key in (
                "NO_RECORDED_OVERLAP", "SHARED_MODULE_DIFFERENT_AXIS", "SHARED_TENSOR_AXIS"
            )},
            "by_pair_family": {
                family: {
                    classification: pair_source_counts[(family, classification)]
                    for classification in (
                        "NO_RECORDED_OVERLAP", "SHARED_MODULE_DIFFERENT_AXIS", "SHARED_TENSOR_AXIS"
                    )
                }
                for family in ("GENERIC_GENERIC", "GENERIC_CUSTOM", "CUSTOM_CUSTOM")
            },
            "claim_boundary": "Static operation-overlap screening only; cumulative physical validity is pending",
        },
        "input_manifest": manifest,
    }
    atomic_json(output / "VALIDATION_SUMMARY.json", summary)
    atomic_text(output / "VALIDATION_STATUS.md", status_markdown(summary))
    atomic_text(
        output / "README.md",
        "# Group Coverage Validation V1\n\n"
        "This folder is a derived, read-only audit of the frozen YOLO26n pruning evidence. "
        "`ROOT_COVERAGE.csv` accounts for all 126 live Conv2d locations. "
        "`GROUP_COMPATIBILITY_MATRIX.csv` and `GROUP_COMPATIBILITY_DETAILS.csv` statically screen all 1,275 unique pairs among the 51 isolated groups. "
        "No model was pruned and no accuracy result was changed by this audit.\n",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = build(args.output)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
