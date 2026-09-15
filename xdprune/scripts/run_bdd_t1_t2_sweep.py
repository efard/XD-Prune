"""Run the frozen BDD GEN-2/NGN-2 immediate post-pruning T1/T2 sweep.

The runner deliberately reuses the successful historical 25/37.5% ratio engine
without modifying it.  It changes only the frozen domain checkpoints, datasets,
and output namespace.  Every worker loads a fresh checkpoint, prunes one of the
same 51 validated groups, validates at 640x640 FP32, writes compact atomic
evidence, and exits.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
import gc
import importlib.util
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any

import psutil
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STUDY_ROOT = PROJECT_ROOT / "Pruning_Study"
FREEZE_PATH = STUDY_ROOT / "configs" / "pruning" / "BDD_T1_T2_EXPERIMENT_FREEZE_V1.json"
EVAL_PATH = STUDY_ROOT / "configs" / "pruning" / "t1_t2_eval_v1.yaml"
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_t1_t2_v1"
HISTORICAL_ENGINE = STUDY_ROOT / "results" / "pruning" / "prune_25" / "run_t1_t2_ratio_sweep.py"
ENVIRONMENT_PATH = STUDY_ROOT / "results" / "depgraph" / "week1_safe_v3" / "environment.json"
DOMAINS = ("GEN2", "NGN2")
RATIOS = {
    "12.5": Fraction(1, 8),
    "25": Fraction(1, 4),
    "25.0": Fraction(1, 4),
    "37.5": Fraction(3, 8),
}


def load_engine() -> Any:
    if not HISTORICAL_ENGINE.is_file():
        raise FileNotFoundError(f"Historical validated ratio engine is missing: {HISTORICAL_ENGINE}")
    spec = importlib.util.spec_from_file_location("_bdd_historical_ratio_engine", HISTORICAL_ENGINE)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the historical ratio engine")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


engine = load_engine()
base = engine.base
full = engine.full


def ratio_slug(fraction: Fraction) -> str:
    mapping = {
        Fraction(1, 8): "12_5pct",
        Fraction(1, 4): "25pct",
        Fraction(3, 8): "37_5pct",
    }
    try:
        return mapping[fraction]
    except KeyError as error:
        raise ValueError(f"Unsupported frozen ratio: {fraction}") from error


engine.ratio_slug = ratio_slug


def read_freeze() -> dict[str, Any]:
    freeze = base.read_json(FREEZE_PATH)
    if freeze.get("freeze_id") != "BDD_T1_T2_EXPERIMENT_FREEZE_V1" or freeze.get("status") != "FROZEN":
        raise RuntimeError("BDD T1/T2 freeze is not the expected frozen version")
    return freeze


def ratio_output(fraction: Fraction) -> Path:
    return OUTPUT_ROOT / ratio_slug(fraction)


def baseline_reference_path(domain: str) -> Path:
    return OUTPUT_ROOT / "baselines" / f"{domain}.json"


def domain_config(domain: str) -> dict[str, Any]:
    freeze = read_freeze()
    if domain not in freeze["domains"]:
        raise ValueError(f"Unknown BDD domain: {domain}")
    config = dict(freeze["domains"][domain])
    reference = baseline_reference_path(domain)
    config.update(
        {
            "path": config["checkpoint"],
            "sha256": config["checkpoint_sha256"],
            "dataset_yaml_sha256": config["dataset_yaml_sha256"],
            "baseline_reference": base.relative(reference),
            "baseline_reference_sha256": base.sha256(reference) if reference.is_file() else "PENDING",
        }
    )
    return config


def generic_rows() -> dict[str, dict[str, str]]:
    return engine.generic_rows()


def custom_rows() -> dict[str, dict[str, str]]:
    return engine.custom_rows()


def all_group_ids() -> list[str]:
    groups = engine.all_group_ids()
    if len(groups) != 51:
        raise RuntimeError("Expected exactly 51 historical validated groups")
    return groups


def verify_frozen_inputs(freeze: dict[str, Any]) -> None:
    for relative_path, expected_hash in freeze["frozen_input_hashes"].items():
        base.verify_file(PROJECT_ROOT / relative_path, expected_hash)
    for config in freeze["domains"].values():
        checkpoint = PROJECT_ROOT / config["checkpoint"]
        dataset = PROJECT_ROOT / config["dataset_yaml"]
        evidence = PROJECT_ROOT / config["baseline_evidence"]
        base.verify_file(checkpoint, config["checkpoint_sha256"])
        base.verify_file(dataset, config["dataset_yaml_sha256"])
        base.verify_file(evidence, config["baseline_evidence_sha256"])
        summary = base.read_json(evidence)
        observed = float(summary["standalone_fp32_validation"]["map50_95"])
        if summary.get("status") != "completed" or abs(observed - float(config["frozen_map50_95"])) > 1e-12:
            raise RuntimeError(f"Frozen baseline evidence disagrees for {config['role']}")


def preflight(fraction: Fraction, require_cuda: bool = True, require_references: bool = True) -> dict[str, Any]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    freeze = read_freeze()
    if fraction not in set(RATIOS.values()):
        raise ValueError(f"Ratio is outside the frozen set: {fraction}")
    verify_frozen_inputs(freeze)

    environment = base.read_json(ENVIRONMENT_PATH)
    actual_versions = {
        "torch": torch.__version__,
        "torch_pruning": version("torch-pruning"),
        "ultralytics": ultralytics.__version__,
    }
    for name, actual in actual_versions.items():
        if str(environment[name]) != str(actual):
            raise RuntimeError(f"Environment changed for {name}: frozen={environment[name]}, current={actual}")
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen BDD evaluator")

    generic = generic_rows()
    custom = custom_rows()
    generic_ops = base.read_json(PROJECT_ROOT / freeze["groups"]["generic_operations"])
    custom_ops = base.read_json(PROJECT_ROOT / freeze["groups"]["custom_operations"])
    if len(generic) != 42 or len(custom) != 9 or len(all_group_ids()) != 51:
        raise RuntimeError("Historical group catalogue cardinality changed")
    for group_id, row in generic.items():
        if row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS":
            raise RuntimeError(f"Generic group lacks historical physical validation: {group_id}")
        if group_id not in generic_ops or row["operation_signature_sha256"] != generic_ops[group_id]["operation_signature_sha256"]:
            raise RuntimeError(f"Generic operation evidence mismatch: {group_id}")
        engine.exact_count(int(row["root_out_channels"]), fraction, f"{group_id} root")
    for group_id, row in custom.items():
        if row["gen_physical_status"] != "PASS" or row["snow_physical_status"] != "PASS":
            raise RuntimeError(f"Custom group lacks historical physical validation: {group_id}")
        if group_id not in custom_ops:
            raise RuntimeError(f"Custom operation evidence is missing: {group_id}")
        engine.exact_count(int(row["hidden_channels_before"]), fraction, f"{group_id} logical width")

    evaluation = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    required_eval = {
        "imgsz": 640, "batch": 16, "device": 0, "workers": 2, "split": "val",
        "rect": True, "conf": 0.001, "iou": 0.7, "max_det": 300, "half": False,
        "dnn": False, "augment": False, "agnostic_nms": False, "cache": False,
        "plots": False, "save_json": False, "verbose": False,
    }
    for key, expected in required_eval.items():
        if evaluation.get(key) != expected:
            raise RuntimeError(f"Frozen evaluation setting changed: {key}")

    # The two attention-aware groups require an equal integral removal count per head.
    checkpoint = PROJECT_ROOT / domain_config("GEN2")["path"]
    model = YOLO(str(checkpoint), task="detect").model.float().cpu().eval()
    _, c2_layout = engine.c2psa_head_aware_importance(model.model[10])
    _, c3_layout = engine.attention_c3k2_head_aware_importance(model.model[22])
    engine.exact_count(c2_layout.key_dim, fraction, "C2PSA units per head")
    engine.exact_count(c3_layout.key_dim, fraction, "attention-C3k2 units per head")
    del model
    gc.collect()

    reference_hashes: dict[str, str] = {}
    if require_references:
        for domain in DOMAINS:
            reference = baseline_reference_path(domain)
            if not reference.is_file():
                raise FileNotFoundError(f"Expanded BDD baseline reference is missing: {reference}")
            record = base.read_json(reference)
            if record.get("status") != "PASS" or record.get("domain") != domain:
                raise RuntimeError(f"Expanded BDD baseline reference is invalid: {reference}")
            reference_hashes[domain] = base.sha256(reference)

    return {
        "schema": "bdd_t1_t2_preflight_v1",
        "freeze_id": freeze["freeze_id"],
        "ratio_fraction": str(fraction),
        "ratio_percent": engine.ratio_percent(fraction),
        "groups": 51,
        "generic_groups": 42,
        "custom_groups": 9,
        "domains": list(DOMAINS),
        "expected_group_domain_runs": 102,
        "exact_group_width_arithmetic": True,
        "exact_attention_head_arithmetic": True,
        "baseline_reference_hashes": reference_hashes,
        "versions": actual_versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "physical_memory_bytes": psutil.virtual_memory().total,
    }


def evaluate_bdd(yolo: Any, dataset: Path, domain: str, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    evaluation = yaml.safe_load(EVAL_PATH.read_text(encoding="utf-8"))
    evaluation.pop("task", None)
    evaluation.pop("mode", None)
    names = {int(key): str(value) for key, value in yolo.names.items()}
    config = domain_config(domain)
    with tempfile.TemporaryDirectory(prefix=f"bdd_t1_t2_{domain.lower()}_") as temporary:
        with open(os.devnull, "w", encoding="utf-8") as sink, redirect_stdout(sink), redirect_stderr(sink):
            metrics = yolo.val(data=str(dataset), project=temporary, name="validation", exist_ok=True, **evaluation)
        overall, per_class = full.expanded_metrics(
            metrics,
            names,
            expected_images=int(config["validation_images"]),
            expected_instances=int(config["validation_instances"]),
        )
    print(json.dumps({"run": run_id, "stage": "validation_complete", "map50_95": overall["map50_95"]}, sort_keys=True), flush=True)
    return overall, per_class


full.evaluate = evaluate_bdd
engine.baseline_config = domain_config
engine.preflight = lambda fraction, require_cuda=True: preflight(fraction, require_cuda=require_cuda, require_references=True)


def baseline_worker(domain: str) -> int:
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("PIN_MEMORY", "false")
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    config = domain_config(domain)
    checkpoint = PROJECT_ROOT / config["path"]
    dataset = PROJECT_ROOT / config["dataset_yaml"]
    result_path = baseline_reference_path(domain)
    record: dict[str, Any] = {
        "schema": "bdd_t1_t2_expanded_baseline_v1",
        "domain": domain,
        "role": config["role"],
        "status": "FAIL",
        "inputs": {
            "checkpoint": base.relative(checkpoint),
            "checkpoint_sha256": config["sha256"],
            "dataset_yaml": base.relative(dataset),
            "dataset_yaml_sha256": config["dataset_yaml_sha256"],
            "freeze": base.relative(FREEZE_PATH),
            "evaluation_config": base.relative(EVAL_PATH),
        },
    }
    started = time.perf_counter()
    monitor = base.MemoryMonitor()
    try:
        preflight(Fraction(1, 8), require_cuda=True, require_references=False)
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        with monitor:
            yolo = YOLO(str(checkpoint), task="detect")
            model = yolo.model.float().cpu().eval()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            gflops = base.finite_metric(get_flops(model, imgsz=640), "baseline_gflops")
            with torch.inference_mode():
                output = model(torch.zeros(1, 3, 640, 640))
                native = base.output_summary(output)
            if not native["all_finite"]:
                raise RuntimeError("Unpruned BDD model produced non-finite output")
            yolo.model = model
            overall, per_class = evaluate_bdd(yolo, dataset, domain, f"BASELINE_{domain}")
            if abs(overall["map50_95"] - float(config["frozen_map50_95"])) > 1e-10:
                raise RuntimeError(
                    f"Expanded BDD baseline does not reproduce frozen primary mAP: "
                    f"{overall['map50_95']} vs {config['frozen_map50_95']}"
                )
            if base.sha256(checkpoint) != config["sha256"]:
                raise RuntimeError("Canonical BDD checkpoint changed during baseline validation")
            record.update({
                "status": "PASS",
                "canonical_checkpoint_modified": False,
                "structure": {"parameters": parameters, "gflops": gflops, "native_output": native},
                "metrics": overall,
                "per_class_metrics": per_class,
            })
    except Exception as error:
        record["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
    finally:
        record["resources"] = {
            "elapsed_seconds": time.perf_counter() - started,
            "peak_cpu_memory_bytes": monitor.peak,
            "peak_cuda_memory_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
        }
        base.atomic_json(result_path, record)
        try:
            del model, yolo
        except (NameError, UnboundLocalError):
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0 if record["status"] == "PASS" else 2


def table_row(record: dict[str, Any], output: Path) -> dict[str, Any]:
    return engine.table_row(record, output)


def build_tables(output: Path, fraction: Fraction) -> dict[str, Any]:
    expected_schema = f"t1_t2_ratio_{ratio_slug(fraction)}_run_v1"
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for path in sorted((output / "runs").glob("*.json")):
        record = base.read_json(path)
        if record.get("schema") != expected_schema:
            raise RuntimeError(f"Unexpected record schema: {path}")
        (records if record.get("status") == "PASS" else failures).append(record)
    expected = {(domain, group_id) for group_id in all_group_ids() for domain in DOMAINS}
    actual = {(record["domain"], record["group_id"]) for record in [*records, *failures]}
    if len(actual) != len(records) + len(failures) or not actual <= expected:
        raise RuntimeError("BDD sweep contains duplicate or unknown records")
    rows = [table_row(record, output) for record in records]
    label = ratio_slug(fraction).upper()
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain in DOMAINS:
        selected = [row for row in rows if row["domain"] == domain]
        selected.sort(key=lambda row: (float(row["AD_map50_95"]), row["group_id"]))
        for rank, row in enumerate(selected, start=1):
            row["rank_by_AD_ascending"] = rank
        by_domain[domain] = selected
    engine.atomic_csv(output / "tables" / f"T1_GEN2_{label}.csv", engine.TABLE_FIELDS, by_domain["GEN2"])
    engine.atomic_csv(output / "tables" / f"T2_NGN2_{label}.csv", engine.TABLE_FIELDS, by_domain["NGN2"])
    engine.atomic_csv(output / "tables" / f"ALL_RESULTS_{label}.csv", engine.TABLE_FIELDS, sorted(rows, key=lambda row: (row["group_id"], row["domain"])))

    per_class: list[dict[str, Any]] = []
    for record in records:
        for metric in record["per_class_metrics"]:
            per_class.append({"ratio_percent": record["requested_percent"], "domain": record["domain"], "group_id": record["group_id"], "group_kind": record["group_kind"], **metric})
    per_class.sort(key=lambda row: (row["group_id"], row["domain"], int(row["class_id"])))
    engine.atomic_csv(output / "tables" / f"PER_CLASS_RESULTS_{label}.csv", engine.PER_CLASS_FIELDS, per_class)

    keyed = {(row["domain"], row["group_id"]): row for row in rows}
    paired: list[dict[str, Any]] = []
    for group_id in all_group_ids():
        gen, ngn = keyed.get(("GEN2", group_id)), keyed.get(("NGN2", group_id))
        if gen and ngn:
            paired.append({
                "ratio_percent": engine.ratio_percent(fraction), "group_id": group_id, "group_kind": gen["group_kind"],
                "representative_root": gen["representative_root"], "parameters_removed": gen["parameters_removed"], "gflops_removed": gen["gflops_removed"],
                "gen2_pruned_map50_95": gen["pruned_map50_95"], "gen2_AD_map50_95": gen["AD_map50_95"], "gen2_NAD_map50_95": gen["NAD_map50_95"],
                "ngn2_pruned_map50_95": ngn["pruned_map50_95"], "ngn2_AD_map50_95": ngn["AD_map50_95"], "ngn2_NAD_map50_95": ngn["NAD_map50_95"],
                "directional_NAD_difference_GEN2_minus_NGN2": float(gen["NAD_map50_95"]) - float(ngn["NAD_map50_95"]),
            })
    paired_fields = list(paired[0]) if paired else ["ratio_percent", "group_id", "group_kind", "representative_root", "parameters_removed", "gflops_removed", "gen2_pruned_map50_95", "gen2_AD_map50_95", "gen2_NAD_map50_95", "ngn2_pruned_map50_95", "ngn2_AD_map50_95", "ngn2_NAD_map50_95", "directional_NAD_difference_GEN2_minus_NGN2"]
    engine.atomic_csv(output / "tables" / f"PAIRED_GEN2_NGN2_{label}.csv", paired_fields, paired)
    progress = {
        "schema": f"bdd_t1_t2_{ratio_slug(fraction)}_progress_v1", "ratio_percent": engine.ratio_percent(fraction),
        "expected_runs": 102, "successful_runs": len(records), "failed_runs": len(failures),
        "remaining_runs": 102 - len(records) - len(failures), "complete_pairs": len(paired),
        "complete": len(records) == 102 and not failures and len(paired) == 51,
    }
    base.atomic_json(output / "progress.json", progress)
    return progress


def write_manifest(output: Path, fraction: Fraction, preflight_record: dict[str, Any]) -> None:
    manifest = {
        "schema": "bdd_t1_t2_ratio_manifest_v1", "status": "RUNNING", "freeze_id": read_freeze()["freeze_id"],
        "ratio_fraction": str(fraction), "ratio_percent": engine.ratio_percent(fraction), "groups": 51,
        "domains": list(DOMAINS), "expected_runs": 102, "preflight": preflight_record,
        "policy": {"fresh_checkpoint_per_run": True, "isolated_group": True, "fine_tuning": False, "batchnorm_update": False, "cumulative_pruning": False, "test_data_used": False, "exact_ratio_required": True},
        "scripts": {base.relative(Path(__file__)): base.sha256(Path(__file__)), base.relative(HISTORICAL_ENGINE): base.sha256(HISTORICAL_ENGINE)},
        "freeze_sha256": base.sha256(FREEZE_PATH), "started_local": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    base.atomic_json(output / "experiment_manifest.json", manifest)
    base.atomic_text(output / "README.md", f"# BDD GEN-2 / NGN-2 {engine.ratio_percent(fraction):g}% T1/T2 Sweep\n\nThis versioned screening sweep evaluates the same 51 validated YOLO26n dependency groups independently on BDD GEN-2 and NGN-2. Every worker starts from an untouched domain checkpoint, applies exactly one structural intervention, and uses the frozen FP32 validation protocol. No fine-tuning, BatchNorm recalibration, cumulative pruning, or test data is used.\n")


def execute(command: list[str], log_path: Path, timeout_seconds: int, min_free_gb: float) -> int:
    return engine.execute_worker(command, log_path, timeout_seconds, min_free_gb)


def ensure_baselines(timeout_seconds: int, min_free_gb: float, retry_failed: bool) -> None:
    for domain in DOMAINS:
        path = baseline_reference_path(domain)
        if path.is_file() and base.read_json(path).get("status") == "PASS":
            continue
        if path.is_file() and not retry_failed:
            raise RuntimeError(f"Existing BDD baseline reference failed: {path}; rerun with --retry-failed")
        command = [sys.executable, str(Path(__file__).resolve()), "--baseline-worker", "--domain", domain]
        code = execute(command, OUTPUT_ROOT / "logs" / f"BASELINE_{domain}.log", timeout_seconds, min_free_gb)
        if code != 0:
            raise RuntimeError(f"BDD expanded baseline failed: {domain}; inspect {path}")


def run_ratio(fraction: Fraction, retry_failed: bool, timeout_seconds: int, min_free_gb: float) -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_baselines(timeout_seconds, min_free_gb, retry_failed)
    output = ratio_output(fraction).resolve()
    allowed = (STUDY_ROOT / "results" / "pruning").resolve()
    if allowed != output and allowed not in output.parents:
        raise RuntimeError(f"Output must stay below {allowed}")
    evidence = preflight(fraction, require_cuda=True, require_references=True)
    output.mkdir(parents=True, exist_ok=True)
    write_manifest(output, fraction, evidence)
    build_tables(output, fraction)
    queue = [(domain, group_id) for group_id in all_group_ids() for domain in DOMAINS]
    for position, (domain, group_id) in enumerate(queue, start=1):
        path = output / "runs" / f"{domain}_{group_id}.json"
        if path.is_file():
            existing = base.read_json(path)
            if existing.get("status") == "PASS":
                print(f"[{position}/102] {domain} {group_id}: already PASS", flush=True)
                continue
            if not retry_failed:
                print(f"[{position}/102] {domain} {group_id}: previous FAIL; rerun with --retry-failed", flush=True)
                return 2
        print(f"[{position}/102] {domain} {group_id}: launching", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--group-worker", "--ratio", ratio_argument(fraction), "--domain", domain, "--group", group_id]
        code = execute(command, output / "logs" / f"{domain}_{group_id}.log", timeout_seconds, min_free_gb)
        progress = build_tables(output, fraction)
        if code != 0:
            print(f"{domain} {group_id}: FAIL; evidence retained in {path}", flush=True)
            return code
        print(f"[{position}/102] {domain} {group_id}: PASS", flush=True)
    progress = build_tables(output, fraction)
    if not progress["complete"]:
        raise RuntimeError(f"BDD ratio sweep ended incomplete: {progress}")
    manifest = base.read_json(output / "experiment_manifest.json")
    manifest.update({"status": "PASS", "completed_local": time.strftime("%Y-%m-%d %H:%M:%S")})
    base.atomic_json(output / "experiment_manifest.json", manifest)
    print(json.dumps(progress, sort_keys=True), flush=True)
    return 0


def parse_ratio(value: str) -> Fraction:
    if value not in RATIOS:
        raise argparse.ArgumentTypeError("ratio must be one of 12.5, 25, or 37.5")
    return RATIOS[value]


def ratio_argument(fraction: Fraction) -> str:
    return {Fraction(1, 8): "12.5", Fraction(1, 4): "25", Fraction(3, 8): "37.5"}[fraction]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratio", type=parse_ratio, default=Fraction(1, 8), metavar="{12.5,25,37.5}")
    parser.add_argument("--all-ratios", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--min-free-gb", type=float, default=4.0)
    parser.add_argument("--baseline-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--group-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--domain", choices=DOMAINS, help=argparse.SUPPRESS)
    parser.add_argument("--group", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.baseline_worker:
        if args.domain is None:
            parser.error("--baseline-worker requires --domain")
        return baseline_worker(args.domain)
    if args.group_worker:
        if args.domain is None or args.group is None:
            parser.error("--group-worker requires --domain and --group")
        if args.group not in all_group_ids():
            parser.error(f"Unknown group: {args.group}")
        return engine.group_worker(args.domain, args.group, ratio_output(args.ratio), args.ratio)
    if args.preflight:
        print(json.dumps(preflight(args.ratio, require_cuda=False, require_references=False), indent=2, sort_keys=True))
        return 0
    fractions = [Fraction(1, 8), Fraction(1, 4), Fraction(3, 8)] if args.all_ratios else [args.ratio]
    for fraction in fractions:
        code = run_ratio(fraction, args.retry_failed, args.timeout_seconds, args.min_free_gb)
        if code != 0:
            return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
