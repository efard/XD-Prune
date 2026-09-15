"""Run one BDD GEN2 or NGN2 staged T7 recovery from frozen raw-T5 masks.

The BDD raw-T5 architecture is replayed from its untouched domain baseline,
using the exact per-domain masks frozen by the completed direct 56% run.  Five
stages use the established 6-6-8-10-40 schedule (70 total epochs); the final
stage receives one warm-up epoch and the last checkpoint feeds each next stage.

Run this script once with --domain GEN2 and once with --domain NGN2.  The two
processes have independent inputs and outputs, so they may run concurrently
only when the remote GPU allocation has sufficient free memory.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import csv
import gc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import yaml


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
RAW_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_raw56_updated_v1"
RAW_MANIFEST = RAW_ROOT / "experiment_manifest.json"
FROZEN_PLAN = RAW_ROOT / "frozen_pruning_plan.json"
FROZEN_PLAN_CSV = RAW_ROOT / "tables" / "FROZEN_PRUNING_PLAN.csv"
T4_PATH = (
    STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t3_t4_updated_v1"
    / "37_5pct" / "tables" / "T4_UPDATED_PRUNABILITY_37_5PCT.csv"
)
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "BDD_T1_T2_EXPERIMENT_FREEZE_V1.json"
TEMPLATE_PATH = (
    STUDY_ROOT / "experiments" / "recovery" / "recovery_56pct"
    / "T7_gen_70ep" / "run_t7_gen_70ep.py"
)
BDD_NATIVE_WORKER_PATH = STUDY_ROOT / "scripts" / "prepare_bdd_raw56_updated.py"
# Optional wrapper overrides.  Historical runs leave these unset and retain
# the original paths; competitor wrappers can bind an explicitly named raw
# checkpoint and the shared BDD baseline metrics without copying either file.
RAW_FINAL_MODEL: Path | None = None
BASELINE_METRICS: Path | None = None
DOMAINS = ("GEN2", "NGN2")
SEED = 42
STAGE_EPOCHS = (6, 6, 8, 10, 40)
STAGE_WARMUP_EPOCHS = (0.0, 0.0, 0.0, 0.0, 1.0)
TOTAL_EPOCHS = sum(STAGE_EPOCHS)
STAGE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("DG026", "DG025", "DG040", "DG039", "CDG009", "DG034", "DG004"),
    ("DG038", "DG041", "CDG005", "DG042", "DG036"),
    ("DG023", "DG003", "DG037", "DG033", "DG024", "DG035"),
    ("DG031", "CDG004", "DG007", "DG019", "DG018", "DG009", "DG005", "DG010"),
    ("DG028", "CDG006", "DG030", "DG016", "DG008", "DG015", "DG006", "DG029", "DG021", "CDG003", "DG017", "DG032", "DG014", "CDG008"),
)
FROZEN_GROUP_SEQUENCE = tuple(group for groups in STAGE_GROUPS for group in groups)
STAGE_TARGET_REDUCTIONS = (11.990710814240224, 26.395652278005223, 37.211328976034864, 44.918480212597856, 56.315449256625726)
EXPECTED_PARAMETERS = 1_094_796
EXPECTED_BASELINE_PARAMETERS = 2_506_140
# Default invariant for the completed current-formula raw56 plan.  Formula
# ablation wrappers may set this to ``None`` only when they explicitly record a
# different stage-boundary policy.
EXPECTED_GROUP_COUNT: int | None = 40
# The current experiment accepts only its own formula by default.  A separate
# formula-ablation wrapper must deliberately replace this immutable allow-list.
EXPECTED_RAW_FORMULA_IDS = frozenset({"BDD_UPDATED_GEN_PROTECTED_V1"})


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def atomic_json(base: Any, path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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


def domain_inputs(base: Any, domain: str) -> dict[str, Any]:
    freeze = base.read_json(FREEZE_PATH)
    if domain not in DOMAINS or domain not in freeze["domains"]:
        raise ValueError(f"Unknown BDD domain: {domain}")
    return dict(freeze["domains"][domain])


def output_root(domain: str) -> Path:
    return RAW_ROOT / f"T7_{domain}_70ep"


def stage_expected_parameters(base: Any, stage: int, domain: str) -> int:
    final_group = STAGE_GROUPS[stage - 1][-1]
    rows = {row["group_id"]: row for row in base.read_csv(FROZEN_PLAN_CSV)}
    return int(float(rows[final_group][f"{domain}_parameters_after"]))


def build_runtime(domain: str) -> dict[str, Any]:
    template = load_module(f"_bdd_t7_template_{domain}", TEMPLATE_PATH)
    bdd_native = load_module(
        f"_bdd_t7_native_worker_{domain}", BDD_NATIVE_WORKER_PATH
    )
    base = template.base
    legacy = template.legacy
    config = domain_inputs(base, domain)
    root = output_root(domain)
    raw_model = RAW_FINAL_MODEL or (RAW_ROOT / "T5" / "models" / "final" / f"{domain}_raw_ranked_56pct.pth")
    baseline_checkpoint = PROJECT_ROOT / config["checkpoint"]
    dataset_yaml = PROJECT_ROOT / config["dataset_yaml"]
    baseline_metrics = BASELINE_METRICS or (RAW_ROOT.parent / "bdd_gen2_ngn2_t1_t2_v1" / "baselines" / f"{domain}.json")
    schema = f"bdd_t7_{domain.lower()}_frozen_masks_6_6_8_10_40_raw56_v1"

    def bdd_baseline_config(requested_domain: str) -> dict[str, Any]:
        """Adapt the legacy GEN/SNOW helper to the active BDD domain.

        The shared pruning worker calls ``baseline_config`` internally.  Its
        historical implementation only knows the old GEN/SNOW keys, so the
        BDD wrapper must provide the equivalent path/hash/metric record for
        GEN2 or NGN2 explicitly.
        """

        if requested_domain != domain:
            raise KeyError(requested_domain)
        metrics = base.read_json(baseline_metrics)
        return {
            "path": config["checkpoint"],
            "dataset_yaml": config["dataset_yaml"],
            "baseline_reference": base.relative(baseline_metrics),
            "sha256": base.sha256(baseline_checkpoint),
            "dataset_yaml_sha256": base.sha256(dataset_yaml),
            "baseline_reference_sha256": base.sha256(baseline_metrics),
            "frozen_map50_95": metrics["metrics"]["map50_95"],
        }

    # Reconfigure the previously validated recovery helpers for this BDD domain.
    for target in (template, legacy):
        target.SELF = SELF
        target.OUTPUT_ROOT = root
        target.SCHEMA = schema
        target.DOMAIN = domain
        target.SEED = SEED
        target.STAGE_EPOCHS = STAGE_EPOCHS
        target.STAGE_WARMUP_EPOCHS = STAGE_WARMUP_EPOCHS
        target.TOTAL_EPOCHS = TOTAL_EPOCHS
        target.STAGE_GROUPS = STAGE_GROUPS
        target.FROZEN_GROUP_SEQUENCE = FROZEN_GROUP_SEQUENCE
        target.BASELINE_CHECKPOINT = baseline_checkpoint
        target.DATASET_YAML = dataset_yaml
        target.BASELINE_METRICS = baseline_metrics
        target.T4_PATH = T4_PATH
        target.MANIFEST_PATH = root / "experiment_manifest.json"
        target.FINAL_TABLE = root / "tables" / f"T7_{domain}_70EP_RESULTS.csv"

    template.ROOT = root
    template.RECOVERY_56_ROOT = RAW_ROOT
    template.RAW_MANIFEST = RAW_MANIFEST
    template.FROZEN_PLAN = FROZEN_PLAN
    template.FROZEN_PLAN_CSV = FROZEN_PLAN_CSV
    template.RAW_FINAL_MODEL = raw_model
    template.EXPECTED_BASELINE_PARAMETERS = EXPECTED_BASELINE_PARAMETERS
    template.EXPECTED_FINAL_PARAMETERS = EXPECTED_PARAMETERS
    template.EXPECTED_FINAL_REDUCTION = STAGE_TARGET_REDUCTIONS[-1]
    template.STAGE_TARGET_REDUCTIONS = STAGE_TARGET_REDUCTIONS
    template.PROTECTED_GROUPS = {"DG001", "DG020"}

    def selected_queue() -> list[dict[str, str]]:
        rows = base.read_csv(T4_PATH)
        rows.sort(key=lambda row: int(row["rank_by_prunability_descending"]))
        result = [row for row in rows if row["group_id"] not in {"DG001", "DG020"}]
        if len(result) != 49:
            raise RuntimeError("Expected 49 unprotected groups in the revised BDD T4 table")
        return result

    def preflight(require_cuda: bool) -> dict[str, Any]:
        required = (RAW_MANIFEST, FROZEN_PLAN, FROZEN_PLAN_CSV, raw_model, T4_PATH, FREEZE_PATH, baseline_checkpoint, dataset_yaml, baseline_metrics)
        for path in required:
            if not path.is_file():
                raise FileNotFoundError(path)
        if require_cuda and not legacy.trainlib.torch.cuda.is_available():
            raise RuntimeError("CUDA device 0 is required for T7 training")
        raw_manifest = base.read_json(RAW_MANIFEST)
        if raw_manifest.get("status") != "PASS" or not raw_manifest.get("target_reached"):
            raise RuntimeError("BDD raw T5 is not complete")
        if tuple(raw_manifest.get("accepted_groups", ())) != FROZEN_GROUP_SEQUENCE:
            raise RuntimeError("BDD raw accepted-group sequence changed")
        if raw_manifest.get("formula_id") not in EXPECTED_RAW_FORMULA_IDS:
            raise RuntimeError(
                f"Unexpected BDD raw-T5 formula identifier: {raw_manifest.get('formula_id')!r}"
            )
        plan = base.read_json(FROZEN_PLAN)
        if tuple(entry["group_id"] for entry in plan["entries"]) != FROZEN_GROUP_SEQUENCE:
            raise RuntimeError("Frozen BDD plan order differs from raw T5")
        if EXPECTED_GROUP_COUNT is not None and len(FROZEN_GROUP_SEQUENCE) != EXPECTED_GROUP_COUNT:
            raise RuntimeError(f"Expected {EXPECTED_GROUP_COUNT} frozen BDD groups")
        if set(FROZEN_GROUP_SEQUENCE) - {row["group_id"] for row in selected_queue()}:
            raise RuntimeError("A frozen BDD group is absent from T4")
        frozen_environment_path = bdd_native.bdd.ENVIRONMENT_PATH
        frozen_environment = base.read_json(frozen_environment_path)
        runtime_torch = legacy.trainlib.torch.__version__
        baseline_parameters = legacy.model_parameters(baseline_checkpoint)
        final_parameters = legacy.model_parameters(raw_model)
        if baseline_parameters != EXPECTED_BASELINE_PARAMETERS or final_parameters != EXPECTED_PARAMETERS:
            raise RuntimeError("BDD baseline or raw-T5 parameter count changed")
        if base.sha256(raw_model) != raw_manifest["final_models"][domain]["sha256"]:
            raise RuntimeError("Frozen raw-T5 checkpoint hash changed")
        for entry in plan["entries"]:
            evidence = entry["domains"][domain]
            source = PROJECT_ROOT / evidence["structure_record"]
            if not evidence["selected_indices"] or base.sha256(source) != evidence["structure_record_sha256"]:
                raise RuntimeError(f"Frozen BDD mask/evidence changed for {entry['group_id']}")
        return {
            "schema": schema, "domain": domain, "seed": SEED,
            "source_raw_manifest": base.relative(RAW_MANIFEST), "source_raw_manifest_sha256": base.sha256(RAW_MANIFEST),
            "frozen_plan": base.relative(FROZEN_PLAN), "frozen_plan_sha256": base.sha256(FROZEN_PLAN),
            "baseline_checkpoint": base.relative(baseline_checkpoint), "baseline_checkpoint_sha256": base.sha256(baseline_checkpoint),
            "dataset": base.relative(dataset_yaml), "dataset_sha256": base.sha256(dataset_yaml),
            "baseline_parameters": baseline_parameters, "final_parameters": final_parameters,
            "final_parameter_reduction_percent": STAGE_TARGET_REDUCTIONS[-1],
            "frozen_group_sequence": list(FROZEN_GROUP_SEQUENCE), "frozen_group_count": len(FROZEN_GROUP_SEQUENCE), "stage_groups": [list(groups) for groups in STAGE_GROUPS],
            "stage_target_reductions": list(STAGE_TARGET_REDUCTIONS), "stage_epochs": list(STAGE_EPOCHS),
            "stage_warmup_epochs": list(STAGE_WARMUP_EPOCHS), "total_epochs": TOTAL_EPOCHS,
            "cuda_available": legacy.trainlib.torch.cuda.is_available(),
            "cuda_device": legacy.trainlib.torch.cuda.get_device_name(0) if legacy.trainlib.torch.cuda.is_available() else None,
            "raw_t5_frozen_torch": str(frozen_environment.get("torch")),
            "t7_runtime_torch": runtime_torch,
            "runtime_environment_note": "Raw-T5 was frozen under its recorded local environment. T7 executes in the Blackwell-compatible container; immutable inputs, raw-model hashes, frozen masks, and structural parameter counts are revalidated before recovery.",
            "schedule_note": "Same historical T7 total budget: 6-6-8-10-40 epochs (70 total). BDD stage boundaries follow its observed raw-T5 cumulative parameter milestones.",
        }

    def worker_prune(
        candidate_index: int,
        group_id: str,
        input_model: Path | None,
        output: Path,
    ) -> int:
        """Replay one frozen BDD structural intervention without legacy hooks."""
        import torch
        from ultralytics import YOLO
        from ultralytics.utils.torch_utils import get_flops

        if not 1 <= candidate_index <= len(FROZEN_GROUP_SEQUENCE):
            raise ValueError(f"Invalid frozen candidate index: {candidate_index}")
        if FROZEN_GROUP_SEQUENCE[candidate_index - 1] != group_id:
            raise ValueError(f"Candidate {candidate_index} is not frozen group {group_id}")

        record_path = output / "records" / f"{domain}_candidate{candidate_index:02d}_structure.json"
        model_path = output / "models" / "trials" / f"{domain}_candidate{candidate_index:02d}_{group_id}.pth"
        if record_path.is_file():
            previous = base.read_json(record_path)
            if previous.get("status") != "PASS":
                atomic_json(
                    base,
                    output / "diagnostics" / f"{domain}_candidate{candidate_index:02d}_before_direct_bdd_worker.json",
                    previous,
                )
        record: dict[str, Any] = {
            "schema": schema,
            "domain": domain,
            "candidate_index": candidate_index,
            "group_added": group_id,
            "groups_applied": list(FROZEN_GROUP_SEQUENCE[:candidate_index]),
            "status": "FAIL",
            "worker_implementation": "direct_bdd_frozen_mask_replay_v1",
        }
        model: Any | None = None
        try:
            preflight(require_cuda=False)
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            if input_model is None:
                model = YOLO(str(baseline_checkpoint), task="detect").model.float().cpu().eval()
                input_path = baseline_checkpoint
            else:
                if not input_model.is_file():
                    raise FileNotFoundError(input_model)
                model = torch.load(input_model, map_location="cpu", weights_only=False).float().cpu().eval()
                input_path = input_model
            input_hash = base.sha256(input_path)
            parameters_before = sum(parameter.numel() for parameter in model.parameters())
            gflops_before = base.finite_metric(get_flops(model, imgsz=640), "gflops_before")
            with torch.inference_mode():
                before = model(torch.zeros(1, 3, 640, 640))
                public_before = base.public_prediction_summary(before)
                del before
            intervention = template.apply_group(model, group_id)
            parameters_after = sum(parameter.numel() for parameter in model.parameters())
            gflops_after = base.finite_metric(get_flops(model, imgsz=640), "gflops_after")
            frozen = template.frozen_intervention(group_id)
            frozen_structure = frozen["source_record"]["structure"]
            expected_parameters_before = int(frozen_structure["parameters_before"])
            expected_parameters_after = int(frozen_structure["parameters_after"])
            if parameters_before != expected_parameters_before:
                raise RuntimeError(
                    f"Live input has {parameters_before} parameters; frozen evidence expects "
                    f"{expected_parameters_before} before {group_id}"
                )
            if parameters_after != expected_parameters_after:
                raise RuntimeError(
                    f"Live output has {parameters_after} parameters; frozen evidence expects "
                    f"{expected_parameters_after} after {group_id}"
                )
            with torch.inference_mode():
                after = model(torch.zeros(1, 3, 640, 640))
                native_after = base.output_summary(after)
                public_after = base.public_prediction_summary(after)
            if not native_after["all_finite"] or public_after["shapes"] != public_before["shapes"]:
                raise RuntimeError("Frozen intervention changed the public output contract")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_suffix(".tmp")
            torch.save(model, temporary)
            temporary.replace(model_path)
            reloaded = torch.load(model_path, map_location="cpu", weights_only=False).float().cpu().eval()
            with torch.inference_mode():
                reloaded_output = reloaded(torch.zeros(1, 3, 640, 640))
            comparison = bdd_native.sequential_engine.pilot.tensor_comparison(after, reloaded_output)
            if not comparison["exact"]:
                raise RuntimeError("Saved/reloaded frozen-intervention model is not numerically identical")
            record.update({
                "status": "PASS",
                "input_model": base.relative(input_path),
                "input_sha256": input_hash,
                "output_model": base.relative(model_path),
                "output_sha256": base.sha256(model_path),
                "intervention": intervention,
                "structure": {
                    "parameters_before": parameters_before,
                    "parameters_after": parameters_after,
                    "incremental_parameters_removed": parameters_before - parameters_after,
                    "gflops_before": gflops_before,
                    "gflops_after": gflops_after,
                    "incremental_gflops_removed": gflops_before - gflops_after,
                    "remote_gflops_counter_reduction_observed": gflops_after < gflops_before,
                    "frozen_gflops_before": float(frozen_structure["gflops_before"]),
                    "frozen_gflops_after": float(frozen_structure["gflops_after"]),
                    "gflops_evidence_note": "Exact structural acceptance uses frozen per-step parameter counts because the Torch 2.8 Blackwell runtime profiler may not reproduce the Torch 2.4.1 FLOP reading. Both runtime and frozen FLOP values are retained.",
                    "native_output_after": native_after,
                    "public_prediction_before": public_before,
                    "public_prediction_after": public_after,
                    "save_reload_comparison": comparison,
                },
            })
        except Exception as exc:
            record["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            print(record["error"]["traceback"], file=sys.stderr, flush=True)
        finally:
            atomic_json(base, record_path, record)
            del model
            gc.collect()
        return 0 if record["status"] == "PASS" else 2

    def evaluate_bdd(yolo: Any, dataset: Path, requested_domain: str, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Evaluate a recovered model with the frozen BDD 640x640 protocol."""
        if requested_domain != domain:
            raise ValueError(f"Evaluator for {domain} received {requested_domain}")
        if dataset.resolve() != dataset_yaml.resolve():
            raise RuntimeError(f"Evaluator dataset changed: {dataset}")
        evaluation_path = STUDY_ROOT / "configs" / "pruning" / "t1_t2_eval_v1.yaml"
        evaluation = yaml.safe_load(evaluation_path.read_text(encoding="utf-8"))
        evaluation.pop("task", None)
        evaluation.pop("mode", None)
        names = {int(key): str(value) for key, value in yolo.names.items()}
        with tempfile.TemporaryDirectory(prefix=f"bdd_t7_{domain.lower()}_") as temporary:
            with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
                metrics = yolo.val(
                    data=str(dataset),
                    project=temporary,
                    name="validation",
                    exist_ok=True,
                    **evaluation,
                )
        overall, per_class = bdd_native.bdd.full.expanded_metrics(
            metrics,
            names,
            expected_images=int(config["validation_images"]),
            expected_instances=int(config["validation_instances"]),
        )
        print(json.dumps({
            "run": run_id,
            "stage": "validation_complete",
            "domain": domain,
            "images": int(config["validation_images"]),
            "instances": int(config["validation_instances"]),
            "map50_95": overall["map50_95"],
            "map50": overall["ap50"],
            "map75": overall["ap75"],
            "precision": overall["precision"],
            "recall": overall["recall"],
        }, sort_keys=True), flush=True)
        return overall, per_class

    template.selected_queue = selected_queue
    template.preflight = preflight
    legacy.selected_queue = selected_queue
    legacy.apply_group = template.apply_group
    # The 40-epoch recovery helper delegates its physical pruning work to the
    # sequential GEN/SNOW engine.  Patch that engine's lookup directly before
    # it configures its downstream pilot worker.
    legacy.baseline_config = bdd_baseline_config
    if hasattr(legacy, "prune_engine"):
        legacy.prune_engine.baseline_config = bdd_baseline_config
        if hasattr(legacy.prune_engine, "pilot"):
            legacy.prune_engine.pilot.baseline_config = bdd_baseline_config
    legacy.preflight = preflight
    legacy.trainlib.custom.evaluate = evaluate_bdd
    return {"template": template, "bdd_native": bdd_native, "base": base, "legacy": legacy, "domain": domain, "root": root, "schema": schema, "baseline_checkpoint": baseline_checkpoint, "baseline_metrics": baseline_metrics, "preflight": preflight, "worker_prune": worker_prune}


def run_prune_worker(ctx: dict[str, Any], candidate_index: int, group_id: str, input_model: Path | None, pruning_root: Path, log_path: Path) -> None:
    command = [sys.executable, str(SELF), "--worker-prune", "--domain", ctx["domain"], "--candidate-index", str(candidate_index), "--group", group_id, "--output-root", str(pruning_root)]
    if input_model is not None:
        command.extend(["--input-model", str(input_model)])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Pruning {group_id} failed; inspect {log_path}")


def verify_prune_record(ctx: dict[str, Any], record: dict[str, Any], group_id: str, stage: int) -> None:
    if record.get("status") != "PASS":
        raise RuntimeError(f"Stage {stage} pruning failed for {group_id}")
    expected = ctx["template"].frozen_intervention(group_id)["selected"]
    observed = [int(value) for value in record["intervention"]["selected_indices"]]
    if observed != expected:
        raise RuntimeError(f"{group_id}: worker did not replay the frozen BDD mask")


def write_readme(ctx: dict[str, Any], final_rows: list[dict[str, Any]]) -> None:
    domain = ctx["domain"]
    lines = [f"# BDD {domain} T7 staged recovery", "", "Exact raw-T5 masks were replayed from the untouched BDD domain checkpoint. The schedule is 6-6-8-10-40 epochs (70 total).", "", "| Checkpoint | mAP50 | mAP75 | mAP50-95 | Baseline retention |", "|---|---:|---:|---:|---:|"]
    for row in final_rows:
        lines.append(f"| {row['checkpoint']} | {float(row['map50']):.6f} | {float(row['map75']):.6f} | {float(row['map50_95']):.6f} | {float(row['map50_95_retention_vs_baseline_percent']):.3f}% |")
    lines.extend(["", "Each intermediate stage begins from the previous stage's `last` checkpoint. Best and last checkpoints are both reported only at the final stage.", ""])
    atomic_text(ctx["root"] / "README.md", "\n".join(lines))


def orchestrate(ctx: dict[str, Any], resume: bool) -> int:
    base, legacy, domain, root = ctx["base"], ctx["legacy"], ctx["domain"], ctx["root"]
    evidence = ctx["preflight"](True)
    manifest_path = root / "experiment_manifest.json"
    root.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        if not resume:
            raise RuntimeError("T7 output already exists. Use --resume to continue it safely.")
        manifest = base.read_json(manifest_path)
        if manifest.get("frozen_plan_sha256") != evidence["frozen_plan_sha256"] or manifest.get("stage_epochs") != list(STAGE_EPOCHS):
            raise RuntimeError("Frozen plan or stage schedule changed; resume is unsafe")
        if manifest.get("status") == "PASS":
            print(f"{domain} T7 is already complete")
            return 0
    else:
        manifest = {**evidence, "status": "RUNNING", "started_local": time.strftime("%Y-%m-%d %H:%M:%S"), "completed_stages": [], "policy": {
            "exact_frozen_group_order": True, "exact_frozen_channel_masks": True,
            "same_final_architecture_as_T5": True, "last_checkpoint_feeds_next_stage": True,
            "best_checkpoint_used_only_for_final_reporting": True, "batchnorm_updated_during_training": True,
            "separate_batchnorm_recalibration": False, "intermediate_warmup_disabled": True,
            "final_stage_warmup_epochs": 1.0, "test_data_used": False,
        }}
        atomic_json(base, manifest_path, manifest)

    indexes = legacy.queue_index()
    completed = {int(value) for value in manifest.get("completed_stages", [])}
    current_model: Path | None = None
    stage_rows: list[dict[str, Any]] = []
    sequence_step = 0
    for stage, (groups, epochs, warmup) in enumerate(zip(STAGE_GROUPS, STAGE_EPOCHS, STAGE_WARMUP_EPOCHS), start=1):
        if stage in completed:
            training = base.read_json(root / f"stage_{stage}" / "training_record.json")
            current_model = legacy.stage_output_model(training, "last")
            evaluation = base.read_json(root / f"stage_{stage}" / "stage_evaluation.json")
            sequence_step += len(groups)
            stage_rows.append({"stage": stage, "groups_added": "+".join(groups), "group_count": len(groups), "epochs": epochs, "cumulative_epochs": sum(STAGE_EPOCHS[:stage]), "parameters": evaluation["parameters"], "parameter_reduction_percent": evaluation["parameter_reduction_percent"], "map50": evaluation["metrics"]["ap50"], "map50_95": evaluation["metrics"]["map50_95"]})
            continue
        stage_root = root / f"stage_{stage}"
        training_record_path = stage_root / "training_record.json"
        prior_training = base.read_json(training_record_path) if training_record_path.is_file() else None
        if prior_training is not None and prior_training.get("status") == "PASS":
            # Training finished before a later evaluation/export failure.  Reuse
            # the verified checkpoint so --resume never repeats completed epochs.
            training = prior_training
            current_model = legacy.stage_output_model(training, "last")
            sequence_step += len(groups)
        else:
            pruning_root = stage_root / "pruning"
            for group_id in groups:
                sequence_step += 1
                index = indexes[group_id]
                run_prune_worker(ctx, index, group_id, current_model, pruning_root, root / "logs" / f"stage{stage}_{sequence_step:02d}_{group_id}.log")
                structure = base.read_json(pruning_root / "records" / f"{domain}_candidate{index:02d}_structure.json")
                verify_prune_record(ctx, structure, group_id, stage)
                current_model = PROJECT_ROOT / structure["output_model"]
                if not current_model.is_file():
                    raise FileNotFoundError(current_model)
            if current_model is None:
                raise RuntimeError("No current model exists for recovery")
            observed = legacy.model_parameters(current_model)
            planned = stage_expected_parameters(base, stage, domain)
            if observed != planned:
                raise RuntimeError(f"Stage {stage} architecture mismatch: {observed} != {planned}")
            training = legacy.train_stage(stage, current_model, epochs, warmup)
            current_model = legacy.stage_output_model(training, "last")
        evaluated = legacy.evaluate_model(current_model, f"BDD_{domain}_T7_stage{stage}_last")
        parameters = legacy.model_parameters(current_model)
        reduction = 100.0 * (1.0 - parameters / EXPECTED_BASELINE_PARAMETERS)
        stage_record = {"schema": ctx["schema"], "status": "PASS", "stage": stage, "groups_added": list(groups), "epochs": epochs, "cumulative_epochs": sum(STAGE_EPOCHS[:stage]), "warmup_epochs": warmup, "model": base.relative(current_model), "model_sha256": base.sha256(current_model), "parameters": parameters, "parameter_reduction_percent": reduction, **evaluated}
        atomic_json(base, root / f"stage_{stage}" / "stage_evaluation.json", stage_record)
        stage_rows.append({"stage": stage, "groups_added": "+".join(groups), "group_count": len(groups), "epochs": epochs, "cumulative_epochs": sum(STAGE_EPOCHS[:stage]), "parameters": parameters, "parameter_reduction_percent": reduction, "map50": evaluated["metrics"]["ap50"], "map50_95": evaluated["metrics"]["map50_95"]})
        atomic_csv(root / "tables" / "T7_STAGE_RESULTS.csv", stage_rows)
        completed.add(stage)
        manifest.update({"completed_stages": sorted(completed), "last_completed_stage": stage, "current_model": base.relative(current_model)})
        manifest.setdefault("stage_results", {})[str(stage)] = stage_record
        atomic_json(base, manifest_path, manifest)
        print(f"{domain} stage {stage}/5 complete: reduction={reduction:.3f}%, epochs={epochs}, mAP50-95={evaluated['metrics']['map50_95']:.6f}", flush=True)

    final_training = base.read_json(root / "stage_5" / "training_record.json")
    baseline = base.read_json(ctx["baseline_metrics"])["metrics"]
    final_rows, final_records = [], {}
    for label in ("best", "last"):
        model = legacy.stage_output_model(final_training, label)
        evaluated = legacy.evaluate_model(model, f"BDD_{domain}_T7_final_{label}")
        parameters = legacy.model_parameters(model)
        if parameters != EXPECTED_PARAMETERS:
            raise RuntimeError("Final T7 architecture differs from frozen BDD T5")
        destination = root / "models" / "final" / f"{domain}_T7_70epoch_{label}.pth"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model, destination)
        metrics = evaluated["metrics"]
        reduction = 100.0 * (1.0 - parameters / EXPECTED_BASELINE_PARAMETERS)
        record = {"schema": ctx["schema"], "status": "PASS", "domain": domain, "checkpoint": label, "model": base.relative(destination), "model_sha256": base.sha256(destination), "parameters": parameters, "parameter_reduction_percent": reduction, "total_fine_tuning_epochs": TOTAL_EPOCHS, **evaluated}
        final_records[label] = record
        atomic_json(base, root / "records" / f"{domain}_final_{label}_evaluation.json", record)
        final_rows.append({"experiment": f"T7_{domain}_6_6_8_10_40", "domain": domain, "checkpoint": label, "accepted_group_count": len(FROZEN_GROUP_SEQUENCE), "parameters_before": EXPECTED_BASELINE_PARAMETERS, "parameters_after": parameters, "parameter_reduction_percent": reduction, "stage_schedule": "6-6-8-10-40", "total_fine_tuning_epochs": TOTAL_EPOCHS, "map50": metrics["ap50"], "map75": metrics["ap75"], "map50_95": metrics["map50_95"], "precision": metrics["precision"], "recall": metrics["recall"], "baseline_map50_95": baseline["map50_95"], "map50_95_retention_vs_baseline_percent": 100.0 * float(metrics["map50_95"]) / float(baseline["map50_95"]), "model": base.relative(destination), "model_sha256": base.sha256(destination)})
    final_table = root / "tables" / f"T7_{domain}_70EP_RESULTS.csv"
    atomic_csv(final_table, final_rows)
    write_readme(ctx, final_rows)
    manifest.update({"status": "PASS", "completed_local": time.strftime("%Y-%m-%d %H:%M:%S"), "final_results": final_records, "final_table": base.relative(final_table), "stage_table": base.relative(root / "tables" / "T7_STAGE_RESULTS.csv")})
    atomic_json(base, manifest_path, manifest)
    print(json.dumps(final_rows, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True, choices=DOMAINS)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-prune", action="store_true")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--group")
    parser.add_argument("--input-model", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    ctx = build_runtime(args.domain)
    if args.preflight:
        print(json.dumps(ctx["preflight"](False), indent=2, sort_keys=True))
        return 0
    if args.worker_prune:
        if args.candidate_index is None or not args.group or args.output_root is None:
            parser.error("worker mode requires --candidate-index, --group, and --output-root")
        return ctx["worker_prune"](args.candidate_index, args.group, args.input_model, args.output_root)
    if args.run and args.resume:
        parser.error("Choose --run or --resume, not both")
    if not args.run and not args.resume:
        parser.error("Choose --preflight, --run, or --resume")
    return orchestrate(ctx, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
