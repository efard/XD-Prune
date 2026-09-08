"""Reconstruct and bootstrap the seven signed negative BDD T1/T2 AD cases.

The script is deliberately narrower than the T1/T2 screening sweep.  It uses
only the frozen negative records, loads a fresh original domain checkpoint per
case, reapplies the *saved* selected channel indices (never recomputed
importance), and exports the same full-precision per-image metric inputs used
by the research validator.  It then runs a paired, image-level bootstrap on
the matching baseline/pruned artifacts.

Run the complete study:
    python .\Pruning_Study\scripts\run_bdd_negative_ad_bootstrap.py --phase all

The export phase requires CUDA.  The bootstrap-only phase is CPU-only and can
be resumed after an interruption.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import types
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "BDD_NEGATIVE_AD_BOOTSTRAP_FREEZE_V1.json"
BDD_RUNNER_PATH = STUDY_ROOT / "scripts" / "run_bdd_t1_t2_sweep.py"
VALIDATOR_PATH = STUDY_ROOT / "scripts" / "research_detection_validator.py"
REPRODUCTION_TOLERANCE = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


if str(STUDY_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(STUDY_ROOT / "scripts"))


def load_bdd_runner() -> Any:
    if not BDD_RUNNER_PATH.is_file():
        raise FileNotFoundError(f"Missing BDD T1/T2 runner: {BDD_RUNNER_PATH}")
    spec = importlib.util.spec_from_file_location("_bdd_t1_t2_runner", BDD_RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load BDD T1/T2 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bdd = load_bdd_runner()
engine = bdd.engine
base = bdd.base
full = bdd.full

from research_detection_validator import PER_IMAGE_STATS_FILENAME, ResearchDetectionValidator  # noqa: E402


def read_freeze() -> dict[str, Any]:
    freeze = read_json(FREEZE_PATH)
    if freeze.get("freeze_id") != "BDD_NEGATIVE_AD_BOOTSTRAP_FREEZE_V1" or freeze.get("status") != "FROZEN":
        raise RuntimeError("Unexpected bootstrap freeze")
    if len(freeze.get("negative_cases", [])) != 7:
        raise RuntimeError("Bootstrap freeze must contain exactly seven negative cases")
    return freeze


def output_root(freeze: dict[str, Any]) -> Path:
    output = PROJECT_ROOT / freeze["output_root"]
    allowed = (STUDY_ROOT / "results" / "pruning").resolve()
    if allowed not in output.resolve().parents:
        raise RuntimeError(f"Output must stay below {allowed}: {output}")
    return output


def ratio_fraction(percent: float) -> Fraction:
    mapping = {12.5: Fraction(1, 8), 25.0: Fraction(1, 4), 37.5: Fraction(3, 8)}
    try:
        return mapping[float(percent)]
    except KeyError as error:
        raise ValueError(f"Unsupported frozen local ratio: {percent}") from error


def case_id(case: dict[str, Any]) -> str:
    ratio = str(case["ratio_percent"]).replace(".", "_")
    return f"{case['domain']}_{case['group_id']}_{ratio}pct"


def baseline_artifact(domain: str) -> Path:
    config = bdd.domain_config(domain)
    evidence = PROJECT_ROOT / config["baseline_evidence"]
    artifact = evidence.parent / "fp32_val" / PER_IMAGE_STATS_FILENAME
    if not artifact.is_file():
        raise FileNotFoundError(f"Missing frozen baseline per-image artifact: {artifact}")
    return artifact


def source_record(case: dict[str, Any]) -> dict[str, Any]:
    path = PROJECT_ROOT / case["source_record"]
    if sha256(path) != case["source_record_sha256"]:
        raise RuntimeError(f"Frozen source record hash changed: {path}")
    record = read_json(path)
    if record.get("status") != "PASS":
        raise RuntimeError(f"Source record is not PASS: {path}")
    for key in ("domain", "group_id", "group_kind"):
        if record.get(key) != case[key]:
            raise RuntimeError(f"Source record mismatch for {key}: {path}")
    if abs(float(record["requested_percent"]) - float(case["ratio_percent"])) > 1e-12:
        raise RuntimeError(f"Source record ratio mismatch: {path}")
    source_ad = float(record["metric_changes"]["map50_95"]["signed_drop"])
    if abs(source_ad - float(case["source_signed_AD_map50_95"])) > REPRODUCTION_TOLERANCE or source_ad >= 0.0:
        raise RuntimeError(f"Source record no longer matches frozen negative AD: {path}")
    return record


def preflight(require_cuda: bool) -> dict[str, Any]:
    import torch
    import ultralytics
    from importlib.metadata import version

    freeze = read_freeze()
    required = {
        PROJECT_ROOT / freeze["source_t1_t2_freeze"]: freeze["source_t1_t2_freeze_sha256"],
        PROJECT_ROOT / freeze["evaluation_config"]: freeze["evaluation_config_sha256"],
        PROJECT_ROOT / freeze["research_validator"]: freeze["research_validator_sha256"],
    }
    for path, expected in required.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Frozen input hash changed: {path}")
    bdd_freeze = bdd.read_freeze()
    bdd.verify_frozen_inputs(bdd_freeze)
    cases = [source_record(case) for case in freeze["negative_cases"]]
    discovered = []
    sweep_root = PROJECT_ROOT / freeze["source_sweep_root"]
    for path in sweep_root.glob("*pct/runs/*.json"):
        record = read_json(path)
        if record.get("status") == "PASS" and float(record["metric_changes"]["map50_95"]["signed_drop"]) < 0.0:
            discovered.append(relative(path))
    frozen_paths = sorted(case["source_record"] for case in freeze["negative_cases"])
    if sorted(discovered) != frozen_paths:
        raise RuntimeError("Frozen negative-case list no longer equals the source sweep negative records")
    for domain in sorted({case["domain"] for case in freeze["negative_cases"]}):
        artifact = baseline_artifact(domain)
        load_artifact(artifact)
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for reconstruction and prediction export")
    return {
        "schema": "bdd_negative_ad_bootstrap_preflight_v1",
        "freeze_id": freeze["freeze_id"],
        "negative_cases": len(cases),
        "negative_case_source_records": frozen_paths,
        "bootstrap": freeze["bootstrap"],
        "versions": {"torch": torch.__version__, "torch_pruning": version("torch-pruning"), "ultralytics": ultralytics.__version__},
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def load_artifact(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        artifact = {name: archive[name] for name in archive.files}
    expected = {"image_ids", "prediction_offsets", "target_offsets", "tp_iou", "confidence", "predicted_class", "target_class"}
    missing = expected - set(artifact)
    if missing:
        raise RuntimeError(f"{path}: missing per-image metric arrays: {sorted(missing)}")
    image_count = len(artifact["image_ids"])
    if len(artifact["prediction_offsets"]) != image_count + 1 or len(artifact["target_offsets"]) != image_count + 1:
        raise RuntimeError(f"{path}: invalid image offset lengths")
    artifact["_prediction_image_index"] = np.repeat(np.arange(image_count, dtype=np.int64), np.diff(artifact["prediction_offsets"]))
    artifact["_target_image_index"] = np.repeat(np.arange(image_count, dtype=np.int64), np.diff(artifact["target_offsets"]))
    return artifact


def metric_from_arrays(tp: np.ndarray, confidence: np.ndarray, predicted_class: np.ndarray, target_class: np.ndarray) -> tuple[float, float]:
    from ultralytics.utils.metrics import ap_per_class

    ap = ap_per_class(tp, confidence, predicted_class, target_class, plot=False)[5]
    return float(ap.mean()), float(ap[:, 0].mean())


def point_metric(artifact: dict[str, np.ndarray]) -> tuple[float, float]:
    return metric_from_arrays(artifact["tp_iou"], artifact["confidence"], artifact["predicted_class"], artifact["target_class"])


def resampled_metric(artifact: dict[str, np.ndarray], indices: np.ndarray) -> tuple[float, float]:
    counts = np.bincount(indices, minlength=len(artifact["image_ids"]))
    prediction_repeats = counts[artifact["_prediction_image_index"]]
    target_repeats = counts[artifact["_target_image_index"]]
    return metric_from_arrays(
        np.repeat(artifact["tp_iou"], prediction_repeats, axis=0),
        np.repeat(artifact["confidence"], prediction_repeats, axis=0),
        np.repeat(artifact["predicted_class"], prediction_repeats, axis=0),
        np.repeat(artifact["target_class"], target_repeats, axis=0),
    )


def class_preserving_samples(artifact: dict[str, np.ndarray], iterations: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    image_count = len(artifact["image_ids"])
    target_offsets = artifact["target_offsets"]
    all_classes = set(int(value) for value in np.unique(artifact["target_class"]))
    samples: list[np.ndarray] = []
    attempts = 0
    while len(samples) < iterations:
        attempts += 1
        if attempts > iterations * 50:
            raise RuntimeError("Unable to draw class-preserving bootstrap samples")
        indices = rng.integers(0, image_count, image_count, dtype=np.int64)
        present: set[int] = set()
        for index in indices:
            start, end = int(target_offsets[index]), int(target_offsets[index + 1])
            present.update(int(value) for value in artifact["target_class"][start:end])
            if present == all_classes:
                break
        if present == all_classes:
            samples.append(indices)
    return np.stack(samples, axis=0)


def load_fresh_model(domain: str) -> tuple[Any, Any, dict[str, Any], Path]:
    from ultralytics import YOLO

    config = bdd.domain_config(domain)
    checkpoint = PROJECT_ROOT / config["path"]
    dataset = PROJECT_ROOT / config["dataset_yaml"]
    if sha256(checkpoint) != config["sha256"] or sha256(dataset) != config["dataset_yaml_sha256"]:
        raise RuntimeError(f"{domain}: frozen checkpoint or dataset hash changed")
    yolo = YOLO(str(checkpoint), task="detect")
    return yolo, yolo.model.float().cpu().eval(), config, dataset


def apply_generic_saved(model: Any, source: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torch_pruning as tp

    group_id = source["group_id"]
    row = engine.generic_rows()[group_id]
    selected = [int(index) for index in source["intervention"]["selected_indices"]]
    if selected != sorted(set(selected)):
        raise RuntimeError(f"{group_id}: frozen selected indices are not sorted unique")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    modules = dict(model.named_modules())
    module_to_path = {id(module): path for path, module in modules.items()}
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: generic root is not Conv2d")
    channels_before = int(root.out_channels)
    fraction = ratio_fraction(float(source["requested_percent"]))
    expected_count = engine.exact_count(channels_before, fraction, f"{group_id} root")
    if len(selected) != expected_count or min(selected) < 0 or max(selected) >= channels_before:
        raise RuntimeError(f"{group_id}: saved channel selection is invalid")
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
                raise RuntimeError("Trace outputs did not retain Autograd dependencies")
            return tensors

    try:
        graph = tp.DependencyGraph().build_dependency(TraceWrapper(model), example_inputs=torch.zeros(1, 3, engine.TRACE_SIZE, engine.TRACE_SIZE))
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=selected)
        if not graph.check_pruning_group(group):
            raise RuntimeError(f"DepGraph rejected frozen selection for {group_id}")
        operations = base.operation_records(group, module_to_path)
        source_operations = source["intervention"]["operations"]
        canonical = base.read_json(engine.GENERIC_OPERATIONS)[group_id]["operations"]
        if base.operation_skeleton(operations) != base.operation_skeleton(source_operations):
            raise RuntimeError(f"{group_id}: live operation skeleton differs from source record")
        if base.operation_skeleton(operations) != base.operation_skeleton(canonical):
            raise RuntimeError(f"{group_id}: live operation skeleton differs from canonical evidence")
        group.prune()
    finally:
        if "forward" in head.__dict__:
            delattr(head, "forward")
    model.eval()
    model.zero_grad(set_to_none=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {
        "method": "live DepGraph group with frozen source selected indices",
        "rule_family": "GENERIC_DEPGRAPH",
        "representative_root": row["representative_root"],
        "selection_unit": "output_channel",
        "selected_indices": selected,
        "root_channels_before": channels_before,
        "root_channels_removed": len(selected),
        "root_channels_after": int(root.out_channels),
        "actual_root_fraction": len(selected) / channels_before,
        "operations": operations,
        "operation_count": len(operations),
    }


def apply_custom_saved(model: Any, source: dict[str, Any]) -> dict[str, Any]:
    group_id = source["group_id"]
    row = engine.custom_rows()[group_id]
    block_index = int(row["block_index"])
    block = model.model[block_index]
    selected = [int(index) for index in source["intervention"]["selected_indices"]]
    if selected != sorted(set(selected)):
        raise RuntimeError(f"{group_id}: frozen selected indices are not sorted unique")
    hidden_before = int(block.c)
    fraction = ratio_fraction(float(source["requested_percent"]))
    expected_count = engine.exact_count(hidden_before, fraction, f"{group_id} hidden width")
    if len(selected) != expected_count or min(selected) < 0 or max(selected) >= hidden_before:
        raise RuntimeError(f"{group_id}: saved custom selection count is invalid")
    if block_index in engine.custom.NONATTENTION_BLOCKS:
        before = engine.validate_nonattention_c3k2_invariants(block)
        result = engine.prune_nonattention_c3k2_logical_channels(block, selected, module_path=f"model.{block_index}")
        after = engine.validate_nonattention_c3k2_invariants(block)
        selection_unit = "hidden_channel"
    elif block_index == 10:
        before = engine.validate_c2psa_invariants(block)
        result = engine.prune_c2psa_head_aware_units(block, selected, module_path=f"model.{block_index}")
        after = engine.validate_c2psa_invariants(block)
        selection_unit = "paired_attention_unit"
    elif block_index == 22:
        before = engine.validate_attention_c3k2_invariants(block)
        result = engine.prune_attention_c3k2_head_aware_units(block, selected, module_path=f"model.{block_index}")
        after = engine.validate_attention_c3k2_invariants(block)
        selection_unit = "paired_attention_unit"
    else:
        raise ValueError(f"Unsupported custom block index: {block_index}")
    operations = result.to_dict()["operations"]
    if json.dumps(operations, sort_keys=True) != json.dumps(source["intervention"]["operations"], sort_keys=True):
        raise RuntimeError(f"{group_id}: custom operations differ from the frozen source record")
    if result.hidden_channels_removed != expected_count or result.hidden_channels_after != hidden_before - expected_count:
        raise RuntimeError(f"{group_id}: custom intervention width invariant failed")
    model.eval()
    return {
        "method": "validated custom rule with frozen source selected indices",
        "rule_family": row["rule_family"],
        "representative_root": row["block_path"],
        "selection_unit": selection_unit,
        "selected_indices": selected,
        "root_channels_before": hidden_before,
        "root_channels_removed": result.hidden_channels_removed,
        "root_channels_after": result.hidden_channels_after,
        "actual_root_fraction": result.hidden_channels_removed / hidden_before,
        "operations": operations,
        "operation_count": len(operations),
        "invariants_before": before,
        "invariants_after": after,
    }


def apply_saved_intervention(model: Any, source: dict[str, Any]) -> dict[str, Any]:
    if source["group_kind"] == "GENERIC":
        return apply_generic_saved(model, source)
    if source["group_kind"] == "CUSTOM":
        return apply_custom_saved(model, source)
    raise ValueError(f"Unsupported group kind: {source['group_kind']}")


def source_operations_match(source: dict[str, Any], intervention: dict[str, Any]) -> bool:
    """Compare operation evidence using the schema of the intervention family."""

    expected = source["intervention"]["operations"]
    actual = intervention["operations"]
    if source["group_kind"] == "GENERIC":
        return base.operation_skeleton(actual) == base.operation_skeleton(expected)
    if source["group_kind"] == "CUSTOM":
        # Custom primitive operations use module_path/operation rather than the
        # DepGraph target_module_path/handler schema.  Compare the full
        # JSON-normalized operation list, including indices and widths.
        return json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    raise ValueError(f"Unsupported group kind: {source['group_kind']}")


def evaluation_args() -> dict[str, Any]:
    config = yaml.safe_load((STUDY_ROOT / "configs" / "pruning" / "t1_t2_eval_v1.yaml").read_text(encoding="utf-8"))
    config.pop("task", None)
    config.pop("mode", None)
    config.update({"plots": False, "save_json": False, "verbose": False})
    return config


def validate_model(yolo: Any, domain: str, dataset: Path, destination: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    names = {int(key): str(value) for key, value in yolo.names.items()}
    metrics = yolo.val(
        validator=ResearchDetectionValidator,
        data=str(dataset),
        project=str(destination.parent),
        name=destination.name,
        exist_ok=True,
        **evaluation_args(),
    )
    config = bdd.domain_config(domain)
    overall, per_class = full.expanded_metrics(
        metrics,
        names,
        expected_images=int(config["validation_images"]),
        expected_instances=int(config["validation_instances"]),
    )
    artifact = destination / PER_IMAGE_STATS_FILENAME
    if not artifact.is_file():
        raise FileNotFoundError(f"Research validator did not create {artifact}")
    return overall, per_class


def export_case(case: dict[str, Any], root: Path, force: bool) -> None:
    source = source_record(case)
    name = case_id(case)
    case_root = root / "exports" / name
    record_path = case_root / "record.json"
    if record_path.is_file() and not force:
        existing = read_json(record_path)
        if existing.get("status") == "PASS" and (case_root / PER_IMAGE_STATS_FILENAME).is_file():
            print(f"SKIP {name}: completed PASS export", flush=True)
            return
    record: dict[str, Any] = {
        "schema": "bdd_negative_ad_export_v1",
        "case": name,
        "status": "FAIL",
        "source_record": case["source_record"],
        "source_record_sha256": case["source_record_sha256"],
        "policy": read_freeze()["policy"],
    }
    started = time.perf_counter()
    try:
        import torch

        torch.cuda.empty_cache()
        yolo, model, config, dataset = load_fresh_model(case["domain"])
        parameters_before = sum(parameter.numel() for parameter in model.parameters())
        intervention = apply_saved_intervention(model, source)
        selected_match = intervention["selected_indices"] == source["intervention"]["selected_indices"]
        operation_match = source_operations_match(source, intervention)
        if not selected_match or not operation_match:
            raise RuntimeError("Frozen selection or dependency operation verification failed")
        yolo.model = model
        overall, per_class = validate_model(yolo, case["domain"], dataset, case_root)
        expected = float(source["metrics"]["map50_95"])
        difference = float(overall["map50_95"]) - expected
        if abs(difference) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(f"Reproduced mAP50-95 differs from source by {difference:.3g}")
        artifact = load_artifact(case_root / PER_IMAGE_STATS_FILENAME)
        artifact_map, artifact_map50 = point_metric(artifact)
        if abs(artifact_map - float(overall["map50_95"])) > REPRODUCTION_TOLERANCE:
            raise RuntimeError("Per-image metric artifact does not reproduce validator mAP50-95")
        baseline = load_artifact(baseline_artifact(case["domain"]))
        if not np.array_equal(artifact["image_ids"], baseline["image_ids"]):
            raise RuntimeError("Pruned and frozen baseline image identifiers/order differ")
        if sha256(PROJECT_ROOT / config["path"]) != config["sha256"]:
            raise RuntimeError("Canonical checkpoint changed during export")
        record.update({
            "status": "PASS",
            "domain": case["domain"],
            "group_id": case["group_id"],
            "group_kind": case["group_kind"],
            "ratio_percent": case["ratio_percent"],
            "checkpoint": config["path"],
            "checkpoint_sha256": config["sha256"],
            "dataset_yaml": config["dataset_yaml"],
            "dataset_yaml_sha256": config["dataset_yaml_sha256"],
            "intervention": intervention,
            "source_selected_indices_match": selected_match,
            "source_operation_skeleton_match": operation_match,
            "parameters_before": parameters_before,
            "parameters_after": sum(parameter.numel() for parameter in model.parameters()),
            "metrics": overall,
            "per_class_metrics": per_class,
            "source_pruned_map50_95": expected,
            "reproduction_difference_map50_95": difference,
            "per_image_artifact": relative(case_root / PER_IMAGE_STATS_FILENAME),
            "per_image_artifact_sha256": sha256(case_root / PER_IMAGE_STATS_FILENAME),
            "artifact_recomputed_map50_95": artifact_map,
            "artifact_recomputed_map50": artifact_map50,
        })
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        record["elapsed_seconds"] = time.perf_counter() - started
        atomic_json(record_path, record)
        try:
            del model, yolo
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    if record["status"] != "PASS":
        raise RuntimeError(f"{name} export failed: {record['error']['message']}")
    print(f"PASS {name}: reproduced mAP50-95={record['metrics']['map50_95']:.9f}", flush=True)


def export_all(root: Path, force: bool) -> None:
    preflight(require_cuda=True)
    for case in read_freeze()["negative_cases"]:
        export_case(case, root, force)


def bootstrap_all(root: Path) -> dict[str, Any]:
    freeze = read_freeze()
    preflight(require_cuda=False)
    iterations = int(freeze["bootstrap"]["iterations"])
    seed = int(freeze["bootstrap"]["seed"])
    cases = freeze["negative_cases"]
    rows: list[dict[str, Any]] = []
    baseline_cache: dict[str, tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, float]] = {}
    for domain in sorted({case["domain"] for case in cases}):
        artifact = load_artifact(baseline_artifact(domain))
        point_map, point_map50 = point_metric(artifact)
        expected_map = float(bdd.domain_config(domain)["frozen_map50_95"])
        if abs(point_map - expected_map) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(f"{domain}: frozen baseline per-image artifact does not reproduce mAP50-95")
        samples = class_preserving_samples(artifact, iterations, seed)
        maps = np.empty(iterations, dtype=np.float64)
        maps50 = np.empty(iterations, dtype=np.float64)
        for index, sample in enumerate(samples):
            maps[index], maps50[index] = resampled_metric(artifact, sample)
        baseline_cache[domain] = (artifact, samples, maps, maps50, point_map50)
        print(f"BOOTSTRAP {domain} baseline: {iterations}/{iterations}", flush=True)

    for position, case in enumerate(cases, start=1):
        name = case_id(case)
        export_record = read_json(root / "exports" / name / "record.json")
        if export_record.get("status") != "PASS":
            raise RuntimeError(f"Cannot bootstrap non-PASS export: {name}")
        artifact = load_artifact(root / "exports" / name / PER_IMAGE_STATS_FILENAME)
        baseline, samples, baseline_map, baseline_map50, baseline_point_map50 = baseline_cache[case["domain"]]
        if not np.array_equal(artifact["image_ids"], baseline["image_ids"]):
            raise RuntimeError(f"{name}: pruned/baseline image IDs differ")
        pruned_map = np.empty(iterations, dtype=np.float64)
        pruned_map50 = np.empty(iterations, dtype=np.float64)
        for index, sample in enumerate(samples):
            pruned_map[index], pruned_map50[index] = resampled_metric(artifact, sample)
        ad = baseline_map - pruned_map
        ad50 = baseline_map50 - pruned_map50
        lower, upper = np.percentile(ad, [2.5, 97.5])
        lower50, upper50 = np.percentile(ad50, [2.5, 97.5])
        pruned_point_map, pruned_point_map50 = point_metric(artifact)
        if abs(pruned_point_map - float(export_record["metrics"]["map50_95"])) > REPRODUCTION_TOLERANCE:
            raise RuntimeError(f"{name}: per-image artifact no longer reproduces export mAP50-95")
        point_ad = float(bdd.domain_config(case["domain"])["frozen_map50_95"]) - float(export_record["metrics"]["map50_95"])
        rows.append({
            "case": name,
            "domain": case["domain"],
            "group_id": case["group_id"],
            "group_kind": case["group_kind"],
            "ratio_percent": case["ratio_percent"],
            "validation_images": len(artifact["image_ids"]),
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
            "bootstrap_design": freeze["bootstrap"]["design"],
            "point_AD_map50_95": point_ad,
            "bootstrap_mean_AD_map50_95": float(ad.mean()),
            "bootstrap_SE_AD_map50_95": float(ad.std(ddof=1)),
            "CI95_lower_AD_map50_95": float(lower),
            "CI95_upper_AD_map50_95": float(upper),
            "CI_excludes_zero": bool(lower > 0.0 or upper < 0.0),
            "supported_improvement": bool(upper < 0.0),
            "bootstrap_probability_AD_below_zero": float(np.mean(ad < 0.0)),
            "point_AD_map50": baseline_point_map50 - pruned_point_map50,
            "CI95_lower_AD_map50": float(lower50),
            "CI95_upper_AD_map50": float(upper50),
            "export_record": relative(root / "exports" / name / "record.json"),
        })
        print(f"[{position}/{len(cases)}] BOOTSTRAP {name}: CI=[{lower:+.9f}, {upper:+.9f}]", flush=True)
    atomic_csv(root / "tables" / "PAIRED_BOOTSTRAP_RESULTS.csv", rows)
    supported = [row["case"] for row in rows if row["supported_improvement"]]
    summary = {
        "schema": "bdd_negative_ad_bootstrap_summary_v1",
        "status": "PASS",
        "freeze_id": freeze["freeze_id"],
        "negative_cases": len(rows),
        "bootstrap": freeze["bootstrap"],
        "exact_reproductions_within_tolerance": sum(abs(float(read_json(root / "exports" / row["case"] / "record.json")["reproduction_difference_map50_95"])) <= REPRODUCTION_TOLERANCE for row in rows),
        "supported_improvements": len(supported),
        "supported_improvement_cases": supported,
        "results_csv": relative(root / "tables" / "PAIRED_BOOTSTRAP_RESULTS.csv"),
        "interpretation": freeze["bootstrap"]["decision"],
    }
    atomic_json(root / "BOOTSTRAP_SUMMARY.json", summary)
    return summary


def write_manifest(root: Path, evidence: dict[str, Any]) -> None:
    freeze = read_freeze()
    manifest_path = root / "experiment_manifest.json"
    existing = read_json(manifest_path) if manifest_path.is_file() else {}
    manifest = {
        "schema": "bdd_negative_ad_bootstrap_manifest_v1",
        "status": existing.get("status", "RUNNING"),
        "freeze_id": freeze["freeze_id"],
        "source_sweep_root": freeze["source_sweep_root"],
        "preflight": evidence,
        "scripts": {relative(Path(__file__)): sha256(Path(__file__)), relative(BDD_RUNNER_PATH): sha256(BDD_RUNNER_PATH), relative(VALIDATOR_PATH): sha256(VALIDATOR_PATH)},
        "freeze_sha256": sha256(FREEZE_PATH),
        "started_local": existing.get("started_local", time.strftime("%Y-%m-%d %H:%M:%S")),
        "completed_local": existing.get("completed_local"),
    }
    atomic_json(manifest_path, manifest)
    (root / "README.md").write_text(
        "# BDD negative-AD paired bootstrap\n\n"
        "This frozen follow-up reconstructs only the seven signed negative BDD T1/T2 records using their saved channel indices. "
        "It contains no fine-tuning, BatchNorm update, cumulative pruning, test data, or independent-seed training. "
        "The paired bootstrap resamples matching validation images for baseline and pruned artifacts.\n",
        encoding="utf-8",
    )


def complete_manifest(root: Path, status: str) -> None:
    path = root / "experiment_manifest.json"
    manifest = read_json(path)
    manifest["status"] = status
    manifest["completed_local"] = time.strftime("%Y-%m-%d %H:%M:%S")
    atomic_json(path, manifest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("export", "bootstrap", "all"), default="all")
    parser.add_argument("--force-export", action="store_true", help="Replace completed case exports; does not change the frozen case list.")
    parser.add_argument("--preflight-only", action="store_true", help="Validate frozen inputs and case discovery without exporting or bootstrapping.")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    freeze = read_freeze()
    root = output_root(freeze)
    root.mkdir(parents=True, exist_ok=True)
    evidence = preflight(require_cuda=args.phase in {"export", "all"} and not args.preflight_only)
    write_manifest(root, evidence)
    if args.preflight_only:
        print(json.dumps(evidence, indent=2))
        return 0
    try:
        if args.phase in {"export", "all"}:
            export_all(root, args.force_export)
        if args.phase in {"bootstrap", "all"}:
            summary = bootstrap_all(root)
            print(json.dumps(summary, indent=2))
        complete_manifest(root, "PASS" if args.phase in {"bootstrap", "all"} else "EXPORTS_PASS_BOOTSTRAP_PENDING")
    except Exception:
        complete_manifest(root, "FAIL")
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
