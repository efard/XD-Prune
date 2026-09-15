"""Run GEN-only staged T7 recovery for the frozen 56.81% architecture.

The exact group order and GEN channel masks produced by the completed raw T5
run are replayed from the untouched GEN baseline. Recovery is performed at
five global-parameter milestones using 6-6-8-10-40 epochs (70 total). The last
checkpoint feeds the next stage; best and last are both retained and evaluated
at the final stage.
"""

from __future__ import annotations

import argparse
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types
from typing import Any


SELF = Path(__file__).resolve()
ROOT = SELF.parent
RECOVERY_56_ROOT = ROOT.parent
STUDY_ROOT = SELF.parents[4]
PROJECT_ROOT = STUDY_ROOT
SOURCE_T7_RUNNER = (
    STUDY_ROOT / "experiments" / "recovery" / "recovery_10pct"
    / "T7_40epoch_snow" / "run_t7_40epoch_snow.py"
)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module("t7_56_shared_engine", SOURCE_T7_RUNNER)
base = legacy.base
ratio_engine = legacy.ratio_engine
prune_engine = legacy.prune_engine
trainlib = legacy.trainlib

SCHEMA = "t7_gen_frozen_masks_6_6_8_10_40_raw56_v1"
DOMAIN = "GEN"
SEED = 42
ORIGINAL_STAGE_EPOCHS = (6, 6, 8, 10, 20)
STAGE_EPOCHS = (6, 6, 8, 10, 40)
STAGE_WARMUP_EPOCHS = (0.0, 0.0, 0.0, 0.0, 1.0)
TOTAL_EPOCHS = sum(STAGE_EPOCHS)
EXPECTED_BASELINE_PARAMETERS = 2_508_090
EXPECTED_FINAL_PARAMETERS = 1_083_192
EXPECTED_FINAL_REDUCTION = 56.81207612167027

STAGE_GROUPS: tuple[tuple[str, ...], ...] = (
    (
        "DG026", "DG040", "DG041", "DG004", "DG009", "DG034",
        "DG019", "DG039", "DG014", "DG033", "DG007", "DG018",
        "DG008", "DG035", "DG012", "DG013", "DG025", "DG038",
        "DG022", "DG042",
    ),
    ("DG015", "DG036", "DG005", "DG003", "DG006", "DG010"),
    ("DG023", "CDG004", "DG032", "CDG005", "CDG009"),
    ("CDG006", "DG024", "DG029", "DG011", "DG021", "CDG003", "DG031"),
    ("DG017", "DG002", "DG037", "CDG007", "CDG008", "DG030", "DG028", "DG016"),
)
FROZEN_GROUP_SEQUENCE = tuple(
    group for stage_groups in STAGE_GROUPS for group in stage_groups
)
STAGE_TARGET_REDUCTIONS = (
    12.4777818977788,
    23.0123320933459,
    37.5831010848893,
    45.7263495329115,
    56.8120761216703,
)

RAW_MANIFEST = RECOVERY_56_ROOT / "experiment_manifest.json"
FROZEN_PLAN = RECOVERY_56_ROOT / "frozen_pruning_plan.json"
FROZEN_PLAN_CSV = (
    RECOVERY_56_ROOT / "tables" / "FROZEN_PRUNING_PLAN.csv"
)
RAW_FINAL_MODEL = (
    RECOVERY_56_ROOT / "T5" / "models" / "final"
    / "GEN_raw_ranked_56pct.pth"
)
T4_PATH = (
    STUDY_ROOT / "results" / "pruning" / "prune_37_5"
    / "tables" / "T4_GEN_PROTECTED_37_5PCT.csv"
)
BASELINE_CHECKPOINT = (
    STUDY_ROOT / "results" / "baselines"
    / "b_gen_mio_yolo26n_s42_v1" / "weights" / "best.pt"
)
DATASET_YAML = (
    STUDY_ROOT / "data_views" / "mio_full_s42_v1" / "mio_tcd_full.yaml"
)
BASELINE_METRICS = (
    STUDY_ROOT / "experiments" / "recovery" / "baseline_validation"
    / "records" / "GEN_baseline_native_metrics.json"
)
MANIFEST_PATH = ROOT / "experiment_manifest.json"
FINAL_TABLE = ROOT / "tables" / "T7_GEN_70EP_RESULTS.csv"
STAGE_TABLE = ROOT / "tables" / "T7_STAGE_RESULTS.csv"
PROTECTED_GROUPS = {"DG001", "DG020"}

# Redirect the tested shared training/evaluation helpers into this new folder.
legacy.SELF = SELF
legacy.OUTPUT_ROOT = ROOT
legacy.SCHEMA = SCHEMA
legacy.DOMAIN = DOMAIN
legacy.SEED = SEED
legacy.STAGE_EPOCHS = STAGE_EPOCHS
legacy.STAGE_WARMUP_EPOCHS = STAGE_WARMUP_EPOCHS
legacy.TOTAL_EPOCHS = TOTAL_EPOCHS
legacy.STAGE_GROUPS = STAGE_GROUPS
legacy.FROZEN_GROUP_SEQUENCE = FROZEN_GROUP_SEQUENCE
legacy.BASELINE_CHECKPOINT = BASELINE_CHECKPOINT
legacy.DATASET_YAML = DATASET_YAML
legacy.BASELINE_METRICS = BASELINE_METRICS
legacy.T4_PATH = T4_PATH
legacy.MANIFEST_PATH = MANIFEST_PATH
legacy.FINAL_TABLE = FINAL_TABLE


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plan_entries() -> list[dict[str, Any]]:
    return base.read_json(FROZEN_PLAN)["entries"]


def plan_by_group() -> dict[str, dict[str, Any]]:
    entries = plan_entries()
    result = {entry["group_id"]: entry for entry in entries}
    if len(result) != len(entries):
        raise RuntimeError("Frozen plan contains duplicate group identifiers")
    return result


def selected_queue() -> list[dict[str, Any]]:
    rows = base.read_csv(T4_PATH)
    rows.sort(key=lambda row: int(row["rank_by_P_GP_descending"]))
    selected = []
    for row in rows:
        if row["group_id"] in PROTECTED_GROUPS:
            continue
        selected.append(
            {
                **row,
                "rank_by_P_GP_desc": row["rank_by_P_GP_descending"],
                "guard_classification": "FROZEN_MASK_T7_REPLAY",
                "local_ratio_percent": 37.5,
            }
        )
    if len(selected) != 49:
        raise RuntimeError("Expected 49 unprotected T4 candidates")
    return selected


def frozen_intervention(group_id: str) -> dict[str, Any]:
    entry = plan_by_group()[group_id]
    domain = entry["domains"][DOMAIN]
    record_path = PROJECT_ROOT / domain["structure_record"]
    if not record_path.is_file():
        raise FileNotFoundError(record_path)
    if base.sha256(record_path) != domain["structure_record_sha256"]:
        raise RuntimeError(f"Frozen source record changed for {group_id}")
    record = base.read_json(record_path)
    return {
        "entry": entry,
        "domain": domain,
        "source_record": record,
        "selected": [int(value) for value in domain["selected_indices"]],
    }


def apply_frozen_generic(model: Any, group_id: str) -> dict[str, Any]:
    import torch
    import torch_pruning as tp

    frozen = frozen_intervention(group_id)
    expected = frozen["domain"]
    selected = frozen["selected"]
    row = ratio_engine.generic_rows()[group_id]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    modules = dict(model.named_modules())
    module_to_path = {id(module): path for path, module in modules.items()}
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: frozen root is not Conv2d")
    channels_before = int(root.out_channels)
    if channels_before != int(expected["root_channels_before"]):
        raise RuntimeError(
            f"{group_id}: live width {channels_before} differs from frozen "
            f"width {expected['root_channels_before']}"
        )
    if len(selected) != int(expected["root_channels_removed"]):
        raise RuntimeError(f"{group_id}: frozen mask length changed")
    if not selected or min(selected) < 0 or max(selected) >= channels_before:
        raise RuntimeError(f"{group_id}: frozen indices are outside the live root")

    head = model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("Expected YOLO26 end-to-end Detect head")
    head.forward = types.MethodType(base.trace_detect_forward, head)

    class TraceWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, images: Any) -> tuple[Any, ...]:
            tensors = tuple(base.flatten_tensors(self.inner(images)))
            if not tensors or not all(tensor.requires_grad for tensor in tensors):
                raise RuntimeError("Trace outputs lost Autograd dependencies")
            return tensors

    try:
        graph = tp.DependencyGraph().build_dependency(
            TraceWrapper(model),
            example_inputs=torch.zeros(
                1, 3, ratio_engine.TRACE_SIZE, ratio_engine.TRACE_SIZE
            ),
        )
        group = graph.get_pruning_group(
            root, tp.prune_conv_out_channels, idxs=selected
        )
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"DepGraph rejected frozen group {group_id}")
        operations = base.operation_records(group, module_to_path)
        original_operations = frozen["source_record"]["intervention"]["operations"]
        if base.operation_skeleton(operations) != base.operation_skeleton(
            original_operations
        ):
            raise RuntimeError(
                f"{group_id}: live dependency family differs from frozen T5"
            )
        group.prune()
    finally:
        if "forward" in head.__dict__:
            delattr(head, "forward")
    model.eval()
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "method": "frozen-mask live DepGraph generic group",
        "representative_root": row["representative_root"],
        "rule_family": "GENERIC_DEPGRAPH",
        "selection_unit": "output_channel",
        "root_channels_before": channels_before,
        "root_channels_removed": len(selected),
        "root_channels_after": int(root.out_channels),
        "actual_root_fraction": len(selected) / channels_before,
        "selected_indices": selected,
        "operations": operations,
        "operation_count": len(operations),
        "operation_family_match": True,
        "frozen_source_record": frozen["domain"]["structure_record"],
    }


def apply_frozen_custom(model: Any, group_id: str) -> dict[str, Any]:
    frozen = frozen_intervention(group_id)
    expected = frozen["domain"]
    selected = frozen["selected"]
    row = ratio_engine.custom_rows()[group_id]
    block_index = int(row["block_index"])
    block = model.model[block_index]
    hidden_before = int(block.c)
    if hidden_before != int(expected["root_channels_before"]):
        raise RuntimeError(
            f"{group_id}: live custom width differs from the frozen width"
        )
    selection_unit = str(expected["selection_unit"])
    channels_per_selected_unit = (
        2 if selection_unit == "paired_attention_unit" else 1
    )
    expected_removed = int(expected["root_channels_removed"])
    observed_removed = len(selected) * channels_per_selected_unit
    if observed_removed != expected_removed:
        raise RuntimeError(
            f"{group_id}: frozen custom mask represents {observed_removed} "
            f"removed channels, expected {expected_removed}"
        )
    selectable_units = hidden_before // channels_per_selected_unit
    if not selected or min(selected) < 0 or max(selected) >= selectable_units:
        raise RuntimeError(f"{group_id}: frozen custom indices are outside the live root")

    if block_index in ratio_engine.custom.NONATTENTION_BLOCKS:
        before = ratio_engine.validate_nonattention_c3k2_invariants(block)
        result = ratio_engine.prune_nonattention_c3k2_logical_channels(
            block, selected, module_path=f"model.{block_index}"
        )
        after = ratio_engine.validate_nonattention_c3k2_invariants(block)
        selection_unit = "hidden_channel"
    elif block_index == 10:
        before = ratio_engine.validate_c2psa_invariants(block)
        result = ratio_engine.prune_c2psa_head_aware_units(
            block, selected, module_path=f"model.{block_index}"
        )
        after = ratio_engine.validate_c2psa_invariants(block)
        selection_unit = "paired_attention_unit"
    elif block_index == 22:
        before = ratio_engine.validate_attention_c3k2_invariants(block)
        result = ratio_engine.prune_attention_c3k2_head_aware_units(
            block, selected, module_path=f"model.{block_index}"
        )
        after = ratio_engine.validate_attention_c3k2_invariants(block)
        selection_unit = "paired_attention_unit"
    else:
        raise ValueError(f"Unsupported custom block index: {block_index}")
    return {
        "method": "frozen-mask validated custom block rule",
        "representative_root": row["block_path"],
        "rule_family": row["rule_family"],
        "selection_unit": selection_unit,
        "root_channels_before": hidden_before,
        "root_channels_removed": result.hidden_channels_removed,
        "root_channels_after": result.hidden_channels_after,
        "actual_root_fraction": result.hidden_channels_removed / hidden_before,
        "selected_indices": selected,
        "operations": result.to_dict()["operations"],
        "operation_count": len(result.operations),
        "invariants_before": before,
        "invariants_after": after,
        "frozen_source_record": frozen["domain"]["structure_record"],
    }


def apply_group(model: Any, group_id: str) -> dict[str, Any]:
    return (
        apply_frozen_custom(model, group_id)
        if group_id.startswith("CDG")
        else apply_frozen_generic(model, group_id)
    )


def preflight(require_cuda: bool) -> dict[str, Any]:
    for path in (
        SOURCE_T7_RUNNER,
        RAW_MANIFEST,
        FROZEN_PLAN,
        FROZEN_PLAN_CSV,
        RAW_FINAL_MODEL,
        T4_PATH,
        BASELINE_CHECKPOINT,
        DATASET_YAML,
        BASELINE_METRICS,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if require_cuda and not trainlib.torch.cuda.is_available():
        raise RuntimeError("CUDA device 0 is required for T7 training")

    raw_manifest = base.read_json(RAW_MANIFEST)
    if raw_manifest.get("status") != "PASS" or not raw_manifest.get(
        "target_reached"
    ):
        raise RuntimeError("The raw 56% experiment is not complete")
    if tuple(raw_manifest.get("accepted_groups", ())) != FROZEN_GROUP_SEQUENCE:
        raise RuntimeError("The raw accepted-group sequence changed")
    plan = base.read_json(FROZEN_PLAN)
    if tuple(entry["group_id"] for entry in plan["entries"]) != FROZEN_GROUP_SEQUENCE:
        raise RuntimeError("The frozen plan order differs from the raw manifest")
    if len(FROZEN_GROUP_SEQUENCE) != 46:
        raise RuntimeError("Expected 46 frozen groups")

    queue_ids = {row["group_id"] for row in selected_queue()}
    missing = sorted(set(FROZEN_GROUP_SEQUENCE) - queue_ids)
    if missing:
        raise RuntimeError(f"Frozen groups are absent from T4: {missing}")

    baseline_parameters = legacy.model_parameters(BASELINE_CHECKPOINT)
    final_parameters = legacy.model_parameters(RAW_FINAL_MODEL)
    if baseline_parameters != EXPECTED_BASELINE_PARAMETERS:
        raise RuntimeError("GEN baseline parameter count changed")
    if final_parameters != EXPECTED_FINAL_PARAMETERS:
        raise RuntimeError("Raw final parameter count changed")
    final_manifest = raw_manifest["final_models"][DOMAIN]
    if base.sha256(RAW_FINAL_MODEL) != final_manifest["sha256"]:
        raise RuntimeError("Raw final model hash changed")

    for entry in plan["entries"]:
        domain = entry["domains"][DOMAIN]
        if not domain["selected_indices"]:
            raise RuntimeError(f"Missing frozen mask for {entry['group_id']}")
        source = PROJECT_ROOT / domain["structure_record"]
        if base.sha256(source) != domain["structure_record_sha256"]:
            raise RuntimeError(
                f"Frozen structure evidence changed for {entry['group_id']}"
            )

    return {
        "schema": SCHEMA,
        "domain": DOMAIN,
        "seed": SEED,
        "source_raw_manifest": base.relative(RAW_MANIFEST),
        "source_raw_manifest_sha256": base.sha256(RAW_MANIFEST),
        "frozen_plan": base.relative(FROZEN_PLAN),
        "frozen_plan_sha256": base.sha256(FROZEN_PLAN),
        "baseline_checkpoint": base.relative(BASELINE_CHECKPOINT),
        "baseline_checkpoint_sha256": base.sha256(BASELINE_CHECKPOINT),
        "dataset": base.relative(DATASET_YAML),
        "dataset_sha256": base.sha256(DATASET_YAML),
        "baseline_parameters": baseline_parameters,
        "final_parameters": final_parameters,
        "final_parameter_reduction_percent": EXPECTED_FINAL_REDUCTION,
        "frozen_group_sequence": list(FROZEN_GROUP_SEQUENCE),
        "stage_groups": [list(groups) for groups in STAGE_GROUPS],
        "stage_target_reductions": list(STAGE_TARGET_REDUCTIONS),
        "stage_epochs": list(STAGE_EPOCHS),
        "stage_warmup_epochs": list(STAGE_WARMUP_EPOCHS),
        "total_epochs": TOTAL_EPOCHS,
        "cuda_available": trainlib.torch.cuda.is_available(),
        "cuda_device": (
            trainlib.torch.cuda.get_device_name(0)
            if trainlib.torch.cuda.is_available()
            else None
        ),
        "torch": trainlib.torch.__version__,
        "comparison_warning": (
            "T7 uses 70 epochs while the current quick T6 uses 30; accuracy "
            "differences combine schedule and training-budget effects."
        ),
    }


# Patch only runtime hooks in the tested shared pruning/training helpers.
legacy.selected_queue = selected_queue
legacy.apply_group = apply_group
legacy.preflight = preflight


def run_prune_worker(
    candidate_index: int,
    group_id: str,
    input_model: Path | None,
    output_root: Path,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        str(SELF),
        "--worker-prune",
        "--candidate-index",
        str(candidate_index),
        "--group",
        group_id,
        "--output-root",
        str(output_root),
    ]
    if input_model is not None:
        command.extend(["--input-model", str(input_model)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Pruning {group_id} failed; inspect {log_path}")


def worker_prune(
    candidate_index: int,
    group_id: str,
    input_model: Path | None,
    output_root: Path,
) -> int:
    return legacy.worker_prune(
        candidate_index, group_id, input_model, output_root
    )


def expected_stage_parameters(stage: int) -> int:
    last_group = STAGE_GROUPS[stage - 1][-1]
    rows = {row["group_id"]: row for row in base.read_csv(FROZEN_PLAN_CSV)}
    return int(float(rows[last_group]["GEN_parameters_after"]))


def verify_prune_record(
    record: dict[str, Any], group_id: str, stage: int
) -> None:
    if record.get("status") != "PASS":
        raise RuntimeError(f"Stage {stage} pruning failed for {group_id}")
    observed = [int(value) for value in record["intervention"]["selected_indices"]]
    expected = frozen_intervention(group_id)["selected"]
    if observed != expected:
        raise RuntimeError(f"{group_id}: worker did not replay the frozen mask")


def write_readme(rows: list[dict[str, Any]], evidence: dict[str, Any]) -> None:
    lines = [
        "# T7 GEN-only 70-epoch staged recovery",
        "",
        "The frozen 56.81%-reduced architecture was reconstructed using the ",
        "exact T5 group order and GEN channel masks. Recovery used a ",
        "6-6-8-10-40 schedule across five parameter milestones.",
        "",
        "| Checkpoint | mAP50 | mAP75 | mAP50-95 | Baseline retention |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['checkpoint']} | {float(row['map50']):.6f} "
            f"| {float(row['map75']):.6f} "
            f"| {float(row['map50_95']):.6f} "
            f"| {float(row['map50_95_retention_vs_baseline_percent']):.3f}% |"
        )
    lines.extend(
        [
            "",
            "The last checkpoint of each intermediate stage feeds the next ",
            "stage. Best-checkpoint selection is used only for final reporting.",
            "",
            f"**Comparison caution:** {evidence['comparison_warning']}",
            "",
        ]
    )
    base.atomic_text(ROOT / "README.md", "\n".join(lines))


def orchestrate(resume: bool) -> int:
    evidence = preflight(require_cuda=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    if MANIFEST_PATH.is_file():
        if not resume:
            raise RuntimeError(
                "T7 output already exists. Use -Resume to skip fully completed stages."
            )
        manifest = base.read_json(MANIFEST_PATH)
        if manifest["frozen_plan_sha256"] != evidence["frozen_plan_sha256"]:
            raise RuntimeError("The frozen mask plan changed; resume is unsafe")
        if manifest.get("status") == "PASS":
            print("T7 is already complete")
            return 0
        saved_epochs = tuple(int(value) for value in manifest["stage_epochs"])
        if saved_epochs != STAGE_EPOCHS:
            completed_before_amendment = {
                int(value) for value in manifest.get("completed_stages", [])
            }
            stage5_training = ROOT / "stage_5" / "training_record.json"
            if (
                saved_epochs != ORIGINAL_STAGE_EPOCHS
                or completed_before_amendment != {1, 2, 3, 4}
                or stage5_training.exists()
            ):
                raise RuntimeError(
                    "The recovery schedule changed in an unsafe run state"
                )
            manifest["protocol_amendment"] = {
                "reason": (
                    "User-requested extension after Stage 4 completed and "
                    "before Stage 5 training began"
                ),
                "original_stage_epochs": list(ORIGINAL_STAGE_EPOCHS),
                "amended_stage_epochs": list(STAGE_EPOCHS),
                "original_total_epochs": sum(ORIGINAL_STAGE_EPOCHS),
                "amended_total_epochs": TOTAL_EPOCHS,
                "completed_stages_preserved": [1, 2, 3, 4],
            }
        manifest.update(
            {
                "schema": SCHEMA,
                "stage_epochs": list(STAGE_EPOCHS),
                "total_epochs": TOTAL_EPOCHS,
                "comparison_warning": evidence["comparison_warning"],
            }
        )
        atomic_json(MANIFEST_PATH, manifest)
    else:
        manifest = {
            **evidence,
            "status": "RUNNING",
            "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_stages": [],
            "policy": {
                "gen_only_exploratory_run": True,
                "exact_frozen_group_order": True,
                "exact_frozen_channel_masks": True,
                "same_final_architecture_as_T5_and_T6": True,
                "last_checkpoint_feeds_next_stage": True,
                "best_checkpoint_used_only_for_final_reporting": True,
                "batchnorm_updated_during_training": True,
                "separate_batchnorm_recalibration": False,
                "intermediate_warmup_disabled": True,
                "final_stage_warmup_epochs": 1.0,
                "test_data_used": False,
            },
        }
        atomic_json(MANIFEST_PATH, manifest)

    indexes = legacy.queue_index()
    completed = set(int(value) for value in manifest["completed_stages"])
    current_model: Path | None = None
    sequence_step = 0
    stage_rows: list[dict[str, Any]] = []

    for stage, (groups, epochs, warmup) in enumerate(
        zip(STAGE_GROUPS, STAGE_EPOCHS, STAGE_WARMUP_EPOCHS), start=1
    ):
        if stage in completed:
            record = base.read_json(ROOT / f"stage_{stage}" / "training_record.json")
            current_model = legacy.stage_output_model(record, "last")
            sequence_step += len(groups)
            evaluation = base.read_json(
                ROOT / f"stage_{stage}" / "stage_evaluation.json"
            )
            stage_rows.append(
                {
                    "stage": stage,
                    "groups_added": "+".join(groups),
                    "group_count": len(groups),
                    "epochs": epochs,
                    "cumulative_epochs": sum(STAGE_EPOCHS[:stage]),
                    "parameters": evaluation["parameters"],
                    "parameter_reduction_percent": evaluation[
                        "parameter_reduction_percent"
                    ],
                    "map50": evaluation["metrics"]["ap50"],
                    "map50_95": evaluation["metrics"]["map50_95"],
                }
            )
            continue

        pruning_root = ROOT / f"stage_{stage}" / "pruning"
        for group_id in groups:
            sequence_step += 1
            candidate_index = indexes[group_id]
            log = ROOT / "logs" / f"stage{stage}_{sequence_step:02d}_{group_id}.log"
            run_prune_worker(
                candidate_index, group_id, current_model, pruning_root, log
            )
            structure_path = (
                pruning_root / "records"
                / f"{DOMAIN}_candidate{candidate_index:02d}_structure.json"
            )
            structure = base.read_json(structure_path)
            verify_prune_record(structure, group_id, stage)
            current_model = PROJECT_ROOT / structure["output_model"]
            if not current_model.is_file():
                raise FileNotFoundError(current_model)

        if current_model is None:
            raise RuntimeError("No current model exists for staged recovery")
        observed_parameters = legacy.model_parameters(current_model)
        planned_parameters = expected_stage_parameters(stage)
        if observed_parameters != planned_parameters:
            raise RuntimeError(
                f"Stage {stage} architecture mismatch: {observed_parameters} "
                f"!= {planned_parameters}"
            )

        training = legacy.train_stage(stage, current_model, epochs, warmup)
        current_model = legacy.stage_output_model(training, "last")
        evaluated = legacy.evaluate_model(
            current_model, f"T7_70_GEN_stage{stage}_last"
        )
        parameters = legacy.model_parameters(current_model)
        reduction = 100.0 * (1.0 - parameters / EXPECTED_BASELINE_PARAMETERS)
        stage_record = {
            "schema": SCHEMA,
            "status": "PASS",
            "stage": stage,
            "groups_added": list(groups),
            "epochs": epochs,
            "cumulative_epochs": sum(STAGE_EPOCHS[:stage]),
            "warmup_epochs": warmup,
            "model": base.relative(current_model),
            "model_sha256": base.sha256(current_model),
            "parameters": parameters,
            "parameter_reduction_percent": reduction,
            **evaluated,
        }
        atomic_json(ROOT / f"stage_{stage}" / "stage_evaluation.json", stage_record)
        stage_rows.append(
            {
                "stage": stage,
                "groups_added": "+".join(groups),
                "group_count": len(groups),
                "epochs": epochs,
                "cumulative_epochs": sum(STAGE_EPOCHS[:stage]),
                "parameters": parameters,
                "parameter_reduction_percent": reduction,
                "map50": evaluated["metrics"]["ap50"],
                "map50_95": evaluated["metrics"]["map50_95"],
            }
        )
        write_csv(STAGE_TABLE, stage_rows)

        completed.add(stage)
        manifest["completed_stages"] = sorted(completed)
        manifest["last_completed_stage"] = stage
        manifest["current_model"] = base.relative(current_model)
        manifest.setdefault("stage_results", {})[str(stage)] = stage_record
        atomic_json(MANIFEST_PATH, manifest)
        print(
            f"Stage {stage}/5 complete: reduction={reduction:.3f}%, "
            f"epochs={epochs}, mAP50-95={evaluated['metrics']['map50_95']:.6f}",
            flush=True,
        )

    final_training = base.read_json(ROOT / "stage_5" / "training_record.json")
    baseline = base.read_json(BASELINE_METRICS)["metrics"]
    final_root = ROOT / "models" / "final"
    final_root.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, Any]] = []
    final_records: dict[str, Any] = {}
    for label in ("best", "last"):
        model_path = legacy.stage_output_model(final_training, label)
        evaluated = legacy.evaluate_model(model_path, f"T7_70_GEN_final_{label}")
        metrics = evaluated["metrics"]
        parameters = legacy.model_parameters(model_path)
        if parameters != EXPECTED_FINAL_PARAMETERS:
            raise RuntimeError("Final T7 architecture differs from T5/T6")
        destination = final_root / f"GEN_T7_70epoch_{label}.pth"
        shutil.copy2(model_path, destination)
        reduction = 100.0 * (1.0 - parameters / EXPECTED_BASELINE_PARAMETERS)
        record = {
            "schema": SCHEMA,
            "status": "PASS",
            "domain": DOMAIN,
            "checkpoint": label,
            "model": base.relative(destination),
            "model_sha256": base.sha256(destination),
            "parameters": parameters,
            "parameter_reduction_percent": reduction,
            "total_fine_tuning_epochs": TOTAL_EPOCHS,
            **evaluated,
        }
        final_records[label] = record
        atomic_json(ROOT / "records" / f"GEN_final_{label}_evaluation.json", record)
        final_rows.append(
            {
                "experiment": "T7_GEN_6_6_8_10_40",
                "domain": DOMAIN,
                "checkpoint": label,
                "accepted_group_count": len(FROZEN_GROUP_SEQUENCE),
                "parameters_before": EXPECTED_BASELINE_PARAMETERS,
                "parameters_after": parameters,
                "parameter_reduction_percent": reduction,
                "stage_schedule": "6-6-8-10-40",
                "total_fine_tuning_epochs": TOTAL_EPOCHS,
                "map50": metrics["ap50"],
                "map75": metrics["ap75"],
                "map50_95": metrics["map50_95"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "baseline_map50_95": baseline["map50_95"],
                "map50_95_retention_vs_baseline_percent": (
                    100.0 * float(metrics["map50_95"])
                    / float(baseline["map50_95"])
                ),
                "model": base.relative(destination),
                "model_sha256": base.sha256(destination),
            }
        )

    write_csv(FINAL_TABLE, final_rows)
    write_readme(final_rows, evidence)
    manifest.update(
        {
            "status": "PASS",
            "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"),
            "final_results": final_records,
            "final_table": base.relative(FINAL_TABLE),
            "stage_table": base.relative(STAGE_TABLE),
        }
    )
    atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps(final_rows, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-prune", action="store_true")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(require_cuda=False), indent=2, sort_keys=True))
        return 0
    if args.worker_prune:
        if args.candidate_index is None or not args.group or args.output_root is None:
            parser.error("worker mode requires candidate index, group, and output root")
        return worker_prune(
            args.candidate_index, args.group, args.input_model, args.output_root
        )
    if args.run:
        return orchestrate(resume=args.resume)
    parser.error("choose --preflight or --run")


if __name__ == "__main__":
    raise SystemExit(main())
