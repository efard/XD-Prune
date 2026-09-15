"""Build the canonical catalogue for physically validated YOLO26 custom groups."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from depgraph_yolo26n_safe_audit import PROJECT_ROOT, atomic_json, sha256


STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
RUN_ROOT = STUDY_ROOT / "results" / "depgraph" / "custom_groups_v1"
SOURCE_CROSSWALK = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3" / "full_root_crosswalk.csv"
BLOCKS = (2, 4, 6, 8, 10, 13, 16, 19, 22)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")


def top_block(module_path: str) -> int:
    parts = module_path.split(".")
    if len(parts) < 2 or parts[0] != "model":
        raise ValueError(f"Unexpected YOLO module path: {module_path}")
    return int(parts[1])


def operation_skeleton(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "module_path": operation["module_path"],
            "operation": operation["operation"],
            "channels_before": operation["channels_before"],
            "channels_after": operation["channels_after"],
            "indices_removed": len(operation["indices"]),
        }
        for operation in record["pruning"]["operations"]
    ]


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def main() -> int:
    deferred = [
        row
        for row in read_csv(SOURCE_CROSSWALK)
        if row["t1_t2_disposition"] == "DEFERRED_UNTIL_CUSTOM_RULE_VALIDATION"
    ]
    if len(deferred) != 19:
        raise RuntimeError(f"Expected 19 deferred roots, found {len(deferred)}")
    by_block: dict[int, list[dict[str, str]]] = {block: [] for block in BLOCKS}
    for row in deferred:
        block = top_block(row["module_path"])
        if block not in by_block:
            raise RuntimeError(f"Deferred root maps outside the custom block set: {row['module_path']}")
        by_block[block].append(row)
    if sum(len(rows) for rows in by_block.values()) != 19 or any(not rows for rows in by_block.values()):
        raise RuntimeError("The deferred-root to custom-block mapping is incomplete")

    catalogue_rows: list[dict[str, Any]] = []
    operations_output: dict[str, Any] = {}
    for ordinal, block in enumerate(BLOCKS, start=1):
        custom_id = f"CDG{ordinal:03d}"
        gen_path = RUN_ROOT / "runs" / f"GEN_C3K2_BLOCK_{block:02d}.json"
        snow_path = RUN_ROOT / "runs" / f"SNOW_C3K2_BLOCK_{block:02d}.json"
        gen = read_json(gen_path)
        snow = read_json(snow_path)
        if gen["status"] != "PASS" or snow["status"] != "PASS":
            raise RuntimeError(f"{custom_id}: GEN/SNOW physical validation is not complete")
        if gen["family"] != snow["family"]:
            raise RuntimeError(f"{custom_id}: GEN/SNOW family mismatch")
        gen_skeleton = operation_skeleton(gen)
        snow_skeleton = operation_skeleton(snow)
        if gen_skeleton != snow_skeleton:
            raise RuntimeError(f"{custom_id}: GEN/SNOW operation skeleton mismatch")
        for key in (
            "hidden_channels_before",
            "hidden_channels_removed",
            "hidden_channels_after",
            "parameters_removed",
        ):
            if gen["structure"][key] != snow["structure"][key]:
                raise RuntimeError(f"{custom_id}: GEN/SNOW structural mismatch for {key}")

        roots = sorted(by_block[block], key=lambda row: row["original_depgraph_id"])
        original_ids = [row["original_depgraph_id"] for row in roots]
        original_paths = [row["module_path"] for row in roots]
        skeleton_sha = digest(gen_skeleton)
        catalogue_rows.append(
            {
                "custom_group_id": custom_id,
                "block_path": f"model.{block}",
                "block_index": block,
                "rule_family": gen["family"],
                "original_deferred_root_count": len(roots),
                "original_depgraph_ids": ";".join(original_ids),
                "original_root_paths": ";".join(original_paths),
                "operation_count": len(gen_skeleton),
                "operation_skeleton_sha256": skeleton_sha,
                "hidden_channels_before": gen["structure"]["hidden_channels_before"],
                "hidden_channels_removed": gen["structure"]["hidden_channels_removed"],
                "hidden_channels_after": gen["structure"]["hidden_channels_after"],
                "actual_logical_fraction": gen["structure"]["actual_logical_fraction"],
                "gen_parameters_removed": gen["structure"]["parameters_removed"],
                "snow_parameters_removed": snow["structure"]["parameters_removed"],
                "gen_gflops_removed": gen["structure"]["gflops_removed"],
                "snow_gflops_removed": snow["structure"]["gflops_removed"],
                "gen_selected_units": ";".join(str(value) for value in gen["importance"]["selected_indices"]),
                "snow_selected_units": ";".join(str(value) for value in snow["importance"]["selected_indices"]),
                "gen_physical_status": gen["status"],
                "snow_physical_status": snow["status"],
                "accuracy_status": "NOT_EVALUATED",
                "t1_t2_extension_disposition": "ELIGIBLE_FOR_VERSIONED_ACCURACY_EXTENSION",
                "gen_record": relative(gen_path),
                "snow_record": relative(snow_path),
            }
        )
        operations_output[custom_id] = {
            "block_path": f"model.{block}",
            "family": gen["family"],
            "source_deferred_roots": [
                {
                    "original_depgraph_id": row["original_depgraph_id"],
                    "module_path": row["module_path"],
                    "reason": row["reason"],
                }
                for row in roots
            ],
            "operation_skeleton_sha256": skeleton_sha,
            "operation_skeleton": gen_skeleton,
            "gen_selected_indices": gen["importance"]["selected_indices"],
            "snow_selected_indices": snow["importance"]["selected_indices"],
            "gen_pruning_mapping": gen["pruning"],
            "snow_pruning_mapping": snow["pruning"],
        }

    catalogue_path = RUN_ROOT / "CUSTOM_GROUP_CATALOGUE.csv"
    with catalogue_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(catalogue_rows[0]))
        writer.writeheader()
        writer.writerows(catalogue_rows)
    atomic_json(RUN_ROOT / "CUSTOM_GROUP_OPERATIONS.json", operations_output)

    table_lines = [
        "# Custom Group Physical-Validation Status",
        "",
        "The 19 deferred convolution-root entries have been canonicalized into nine",
        "logical block-width candidates. All nine passed physical validation on the",
        "frozen GEN and SNOW checkpoints. No detection-accuracy evaluation, T1/T2",
        "accuracy drop, fine-tuning or cumulative pruning is claimed here.",
        "",
        "| ID | Block | Family | Deferred roots | Hidden width | Parameters removed | GEN GFLOPs removed |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in catalogue_rows:
        table_lines.append(
            f"| {row['custom_group_id']} | `{row['block_path']}` | {row['rule_family']} | "
            f"{row['original_deferred_root_count']} | {row['hidden_channels_before']} -> "
            f"{row['hidden_channels_after']} | {row['gen_parameters_removed']:,} | "
            f"{float(row['gen_gflops_removed']):.6f} |"
        )
    table_lines.extend(
        [
            "",
            "## Proven",
            "",
            "- paired split, concatenation and residual mappings pass full-model forwards;",
            "- C2PSA and attention-C3k2 packed Q/K/V mappings retain valid head shapes;",
            "- every candidate reduces dense parameters and profiled GFLOPs;",
            "- 64-pixel and 640-pixel forwards pass before and after save/reload;",
            "- GEN/SNOW operation skeletons match; and",
            "- the canonical baseline checkpoints were not modified.",
            "",
            "## Not yet proven",
            "",
            "- detection accuracy or accuracy drop for any custom group;",
            "- superiority over generic or prior custom pruning methods;",
            "- cumulative compatibility when several custom groups are pruned together;",
            "- recovery fine-tuning performance;",
            "- device latency or FPGA benefit; or",
            "- a first-in-literature novelty claim.",
            "",
        ]
    )
    status_md = RUN_ROOT / "CUSTOM_GROUP_STATUS.md"
    status_md.write_text("\n".join(table_lines), encoding="utf-8")

    status = read_json(RUN_ROOT / "status.json")
    status.update(
        {
            "deferred_root_entries_accounted_for": 19,
            "canonical_custom_groups": len(catalogue_rows),
            "physically_validated_custom_groups_gen": len(catalogue_rows),
            "physically_validated_custom_groups_snow": len(catalogue_rows),
            "catalogue_sha256": sha256(catalogue_path),
            "operations_sha256": sha256(RUN_ROOT / "CUSTOM_GROUP_OPERATIONS.json"),
            "status_markdown_sha256": sha256(status_md),
            "accuracy_evaluation_performed": False,
        }
    )
    atomic_json(RUN_ROOT / "status.json", status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
