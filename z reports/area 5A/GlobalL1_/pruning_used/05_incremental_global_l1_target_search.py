#!/usr/bin/env python3
"""
Stage 5B V2: deterministic incremental Global L1 parameter-target search.

Why V1 is not used
------------------
The high-level one-shot global-ratio search was non-monotonic because pure
global ranking concentrated many selected channels in one root. When a group
crossed Torch-Pruning's maximum-pruning safety test, that whole group could be
skipped. A larger requested ratio could therefore remove fewer parameters.

V2 directly implements the requested criterion:
- calculate current Conv2d output-filter L1 magnitudes;
- rank all eligible channels globally;
- choose the smallest eligible L1 channel;
- obtain its complete dependency group;
- physically prune that group;
- rebuild DepGraph and repeat until the parameter target is reached.

No accuracy, sensitivity, or dataset metric is used for channel selection.

Scope
-----
- 42 T4 GENERIC representative roots only.
- 9 CUSTOM C3k2/C2PSA roots excluded because their custom code is absent.
- 54 protected roots may receive required input-channel updates, but their
  output channels must never be removed.

Safety rule
-----------
Each eligible root must retain at least:
    max(4 channels, ceil(50% of its original output channels))

This prevents pure Global L1 from collapsing almost an entire layer while
remaining a global L1-magnitude method.

This search saves CSV/JSON plans only. It does not save a pruned .pt model,
train, or evaluate mAP.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
import ultralytics


EXPECTED_BASELINE_PARAMS = {
    "GEN": 2_508_090,
    "SNOW": 2_506_920,
}

TARGET_PARAMS = {
    "GEN": 2_250_986,
    "SNOW": 2_249_816,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, dict):
        for item in value.values():
            tensors.extend(flatten_tensors(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(flatten_tensors(item))
    return tensors


def all_tensor_output_transform(output: Any) -> tuple[torch.Tensor, ...]:
    tensors = flatten_tensors(output)
    if not tensors:
        raise RuntimeError("Model output contained no tensors.")
    return tuple(tensors)


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def build_dependency_graph(
    model: nn.Module,
    imgsz: int,
    device: torch.device,
) -> tp.DependencyGraph:
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    # A constant input keeps tracing identical across every pruning step.
    example = torch.zeros(
        1,
        3,
        imgsz,
        imgsz,
        device=device,
        requires_grad=True,
    )

    with torch.enable_grad():
        graph = tp.DependencyGraph().build_dependency(
            model,
            example_inputs=example,
            output_transform=all_tensor_output_transform,
        )

    return graph


def flatten_indices(values: Any) -> list[int]:
    result: list[int] = []

    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().flatten().tolist()

    if isinstance(values, (list, tuple, set)):
        for item in values:
            result.extend(flatten_indices(item))
    else:
        try:
            result.append(int(values))
        except (TypeError, ValueError):
            pass

    return result


def group_safety_check(
    *,
    graph: tp.DependencyGraph,
    group: Any,
    names_by_id: dict[int, str],
    allowed_roots: set[str],
    protected_roots: set[str],
    minimum_remaining: dict[str, int],
) -> tuple[bool, str, list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []

    for dependency, indices in group:
        module = dependency.target.module
        module_name = names_by_id.get(id(module), "<unresolved>")
        unique_indices = sorted(set(flatten_indices(indices)))
        handler_name = getattr(
            dependency.handler,
            "__name__",
            str(dependency.handler),
        )

        is_out_prune = graph.is_out_channel_pruning_fn(
            dependency.handler
        )

        operation = {
            "module_name": module_name,
            "module_type": module.__class__.__name__,
            "handler": handler_name,
            "is_out_channel_prune": is_out_prune,
            "index_count": len(unique_indices),
            "indices": ";".join(map(str, unique_indices)),
        }

        if is_out_prune and isinstance(module, nn.Conv2d):
            operation["current_out_channels"] = module.out_channels
            operation["projected_out_channels"] = (
                module.out_channels - len(unique_indices)
            )

            if module_name in protected_roots:
                return (
                    False,
                    f"would prune protected output root {module_name}",
                    operations + [operation],
                )

            if module_name in allowed_roots:
                projected = module.out_channels - len(unique_indices)
                required = minimum_remaining[module_name]

                if projected < required:
                    return (
                        False,
                        (
                            f"would reduce {module_name} to {projected}, "
                            f"below minimum {required}"
                        ),
                        operations + [operation],
                    )

        operations.append(operation)

    if not graph.check_pruning_group(group):
        return False, "DependencyGraph rejected pruning group", operations

    return True, "", operations


def current_root_inventory(
    model: nn.Module,
    allowed_roots: set[str],
    original_out_channels: dict[str, int],
    minimum_remaining: dict[str, int],
) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []

    for name in sorted(allowed_roots):
        module = modules[name]
        current = module.out_channels
        original = original_out_channels[name]
        rows.append(
            {
                "root_module": name,
                "original_out_channels": original,
                "current_out_channels": current,
                "out_channels_removed": original - current,
                "root_pruning_fraction": (
                    (original - current) / original
                ),
                "minimum_remaining_out_channels": minimum_remaining[name],
                "at_safety_limit": current <= minimum_remaining[name],
            }
        )

    return rows


def global_l1_candidates(
    model: nn.Module,
    allowed_roots: set[str],
    minimum_remaining: dict[str, int],
    group_id_by_root: dict[str, str],
) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    candidates: list[dict[str, Any]] = []

    for root_name in sorted(allowed_roots):
        module = modules[root_name]
        if module.out_channels <= minimum_remaining[root_name]:
            continue

        weight = module.weight.detach().float()
        scores = weight.abs().sum(
            dim=tuple(range(1, weight.ndim))
        ).cpu()

        for channel_index, score in enumerate(scores.tolist()):
            candidates.append(
                {
                    "group_id": group_id_by_root[root_name],
                    "root_module": root_name,
                    "current_channel_index": channel_index,
                    "l1_score": float(score),
                    "current_root_out_channels": module.out_channels,
                    "minimum_remaining_out_channels": (
                        minimum_remaining[root_name]
                    ),
                }
            )

    candidates.sort(
        key=lambda row: (
            row["l1_score"],
            row["group_id"],
            row["current_channel_index"],
        )
    )
    return candidates


def verify_protected_outputs(
    model: nn.Module,
    protected_out_channels: dict[str, int],
) -> tuple[bool, list[dict[str, Any]]]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []
    ok = True

    for name, expected_out in sorted(protected_out_channels.items()):
        module = modules.get(name)
        if not isinstance(module, nn.Conv2d):
            rows.append(
                {
                    "module_path": name,
                    "status": "MISSING_OR_WRONG_TYPE",
                    "expected_out_channels": expected_out,
                    "actual_out_channels": "",
                }
            )
            ok = False
            continue

        same = module.out_channels == expected_out
        rows.append(
            {
                "module_path": name,
                "status": "OK" if same else "OUTPUT_CHANGED",
                "expected_out_channels": expected_out,
                "actual_out_channels": module.out_channels,
                "current_in_channels": module.in_channels,
            }
        )
        ok = ok and same

    return ok, rows


def run_domain(
    *,
    domain: str,
    checkpoint: Path,
    generic_rows: list[dict[str, str]],
    protected_rows: list[dict[str, str]],
    output_dir: Path,
    imgsz: int,
    device: torch.device,
    seed: int,
    maximum_root_pruning_fraction: float,
    absolute_minimum_channels: int,
    maximum_steps: int,
) -> dict[str, Any]:
    started = time.time()
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(seed)

    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(device).eval()

    baseline_parameters = count_parameters(model)
    expected_parameters = EXPECTED_BASELINE_PARAMS[domain]
    target_parameters = TARGET_PARAMS[domain]

    if baseline_parameters != expected_parameters:
        raise RuntimeError(
            f"{domain}: baseline parameters {baseline_parameters} "
            f"!= expected {expected_parameters}"
        )

    modules = dict(model.named_modules())
    allowed_roots = {
        row["representative_root"] for row in generic_rows
    }
    group_id_by_root = {
        row["representative_root"]: row["group_id"]
        for row in generic_rows
    }
    protected_roots = {
        row["module_path"] for row in protected_rows
    }

    missing_allowed = sorted(allowed_roots - set(modules))
    if missing_allowed:
        raise RuntimeError(
            f"{domain}: missing allowed roots: {missing_allowed}"
        )

    original_out_channels: dict[str, int] = {}
    minimum_remaining: dict[str, int] = {}

    for name in allowed_roots:
        module = modules[name]
        if not isinstance(module, nn.Conv2d):
            raise RuntimeError(
                f"{domain}: allowed root is not Conv2d: {name}"
            )

        original = module.out_channels
        original_out_channels[name] = original

        minimum_from_fraction = math.ceil(
            original * (1.0 - maximum_root_pruning_fraction)
        )
        minimum_remaining[name] = max(
            absolute_minimum_channels,
            minimum_from_fraction,
        )

    protected_out_channels: dict[str, int] = {}
    for row in protected_rows:
        name = row["module_path"]
        module = modules.get(name)
        if isinstance(module, nn.Conv2d):
            protected_out_channels[name] = module.out_channels

    # Baseline forward verification.
    with torch.no_grad():
        output = model(
            torch.zeros(1, 3, imgsz, imgsz, device=device)
        )
        if not flatten_tensors(output):
            raise RuntimeError(
                f"{domain}: baseline forward produced no tensors"
            )

    step_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []

    initial_inventory = current_root_inventory(
        model,
        allowed_roots,
        original_out_channels,
        minimum_remaining,
    )
    write_csv(
        domain_dir / "initial_42_root_inventory.csv",
        initial_inventory,
    )

    previous_parameters = baseline_parameters
    chosen_step_count: int | None = None
    target_crossed = False

    for step in range(1, maximum_steps + 1):
        graph = build_dependency_graph(model, imgsz, device)
        modules = dict(model.named_modules())
        names_by_id = {
            id(module): name for name, module in model.named_modules()
        }

        candidates = global_l1_candidates(
            model,
            allowed_roots,
            minimum_remaining,
            group_id_by_root,
        )

        if not candidates:
            break

        selected: dict[str, Any] | None = None
        selected_group = None
        selected_operations: list[dict[str, Any]] = []

        for global_rank, candidate in enumerate(candidates, start=1):
            root_name = candidate["root_module"]
            module = modules[root_name]
            channel_index = candidate["current_channel_index"]

            try:
                group = graph.get_pruning_group(
                    module,
                    tp.prune_conv_out_channels,
                    idxs=[channel_index],
                )

                safe, reason, operations = group_safety_check(
                    graph=graph,
                    group=group,
                    names_by_id=names_by_id,
                    allowed_roots=allowed_roots,
                    protected_roots=protected_roots,
                    minimum_remaining=minimum_remaining,
                )
            except Exception as exc:
                safe = False
                reason = repr(exc)
                operations = []
                group = None

            if safe:
                selected = dict(candidate)
                selected["global_candidate_rank_this_step"] = global_rank
                selected_group = group
                selected_operations = operations
                break

            rejection_rows.append(
                {
                    "step": step,
                    "global_candidate_rank_this_step": global_rank,
                    **candidate,
                    "rejection_reason": reason,
                }
            )

        if selected is None or selected_group is None:
            break

        before_parameters = count_parameters(model)
        before_root_out = modules[
            selected["root_module"]
        ].out_channels

        # Group must be used immediately and sequentially.
        selected_group.prune()

        after_parameters = count_parameters(model)
        modules_after = dict(model.named_modules())
        after_root_out = modules_after[
            selected["root_module"]
        ].out_channels

        with torch.no_grad():
            output = model(
                torch.zeros(1, 3, imgsz, imgsz, device=device)
            )
            if not flatten_tensors(output):
                raise RuntimeError(
                    f"{domain}: step {step} forward returned no tensors"
                )

        protected_ok, protected_check_rows = (
            verify_protected_outputs(
                model,
                protected_out_channels,
            )
        )
        if not protected_ok:
            write_csv(
                domain_dir
                / f"FAILED_step_{step:04d}_protected_outputs.csv",
                protected_check_rows,
            )
            raise RuntimeError(
                f"{domain}: protected output changed at step {step}"
            )

        current_inventory = current_root_inventory(
            model,
            allowed_roots,
            original_out_channels,
            minimum_remaining,
        )
        minimum_current = min(
            row["current_out_channels"]
            for row in current_inventory
        )

        row = {
            "step": step,
            "domain": domain,
            "group_id": selected["group_id"],
            "root_module": selected["root_module"],
            "current_channel_index_pruned": (
                selected["current_channel_index"]
            ),
            "l1_score_at_selection": selected["l1_score"],
            "global_candidate_rank_this_step": (
                selected["global_candidate_rank_this_step"]
            ),
            "root_out_channels_before": before_root_out,
            "root_out_channels_after": after_root_out,
            "parameters_before": before_parameters,
            "parameters_after": after_parameters,
            "parameters_removed_this_step": (
                before_parameters - after_parameters
            ),
            "cumulative_parameters_removed": (
                baseline_parameters - after_parameters
            ),
            "cumulative_parameter_reduction_percent": (
                100.0
                * (baseline_parameters - after_parameters)
                / baseline_parameters
            ),
            "target_parameters": target_parameters,
            "target_error_after": (
                after_parameters - target_parameters
            ),
            "absolute_target_error_after": abs(
                after_parameters - target_parameters
            ),
            "minimum_current_generic_root_out_channels": (
                minimum_current
            ),
            "protected_outputs_unchanged": True,
            "forward_pass_ok": True,
        }
        step_rows.append(row)

        for operation_index, operation in enumerate(
            selected_operations,
            start=1,
        ):
            operation_rows.append(
                {
                    "step": step,
                    "operation_index": operation_index,
                    "selected_group_id": selected["group_id"],
                    "selected_root_module": selected["root_module"],
                    **operation,
                }
            )

        # Continuously persist progress.
        write_csv(
            domain_dir / "incremental_pruning_steps.csv",
            step_rows,
        )
        write_csv(
            domain_dir / "dependency_operations.csv",
            operation_rows,
        )
        write_csv(
            domain_dir / "rejected_candidates.csv",
            rejection_rows,
        )

        print(
            f"{domain} step={step:04d} "
            f"root={selected['root_module']} "
            f"idx={selected['current_channel_index']} "
            f"L1={selected['l1_score']:.8g} "
            f"params={after_parameters:,} "
            f"error={after_parameters - target_parameters:+,}"
        )

        if after_parameters <= target_parameters:
            previous_error = abs(
                before_parameters - target_parameters
            )
            current_error = abs(
                after_parameters - target_parameters
            )

            chosen_step_count = (
                step if current_error <= previous_error else step - 1
            )
            target_crossed = True
            break

        previous_parameters = after_parameters

    if not step_rows:
        raise RuntimeError(
            f"{domain}: no channel could be safely pruned"
        )

    if chosen_step_count is None:
        closest_row = min(
            step_rows,
            key=lambda item: item["absolute_target_error_after"],
        )
        chosen_step_count = int(closest_row["step"])

    chosen_plan = [
        row for row in step_rows
        if int(row["step"]) <= chosen_step_count
    ]

    if chosen_step_count == 0:
        chosen_remaining = baseline_parameters
        chosen_reduction = 0
        chosen_reduction_percent = 0.0
    else:
        chosen_last = chosen_plan[-1]
        chosen_remaining = int(chosen_last["parameters_after"])
        chosen_reduction = (
            baseline_parameters - chosen_remaining
        )
        chosen_reduction_percent = (
            100.0 * chosen_reduction / baseline_parameters
        )

    write_csv(
        domain_dir / "chosen_replay_plan.csv",
        chosen_plan,
    )

    final_inventory = current_root_inventory(
        model,
        allowed_roots,
        original_out_channels,
        minimum_remaining,
    )
    write_csv(
        domain_dir / "search_end_42_root_inventory.csv",
        final_inventory,
    )

    protected_ok, protected_final_rows = verify_protected_outputs(
        model,
        protected_out_channels,
    )
    write_csv(
        domain_dir / "search_end_protected_output_check.csv",
        protected_final_rows,
    )

    summary = {
        "domain": domain,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "baseline_parameters": baseline_parameters,
        "target_parameters": target_parameters,
        "target_crossed_during_search": target_crossed,
        "total_search_steps_executed": len(step_rows),
        "chosen_replay_steps": chosen_step_count,
        "chosen_remaining_parameters": chosen_remaining,
        "chosen_parameters_removed": chosen_reduction,
        "chosen_parameter_reduction_percent": (
            chosen_reduction_percent
        ),
        "chosen_target_error_parameters": (
            chosen_remaining - target_parameters
        ),
        "chosen_absolute_target_error_parameters": abs(
            chosen_remaining - target_parameters
        ),
        "maximum_root_pruning_fraction": (
            maximum_root_pruning_fraction
        ),
        "absolute_minimum_channels": absolute_minimum_channels,
        "protected_outputs_unchanged_at_search_end": protected_ok,
        "search_end_forward_pass_ok": True,
        "trial_model_saved": False,
        "training_performed": False,
        "accuracy_evaluated": False,
        "chosen_plan_file": "chosen_replay_plan.csv",
        "step_log_file": "incremental_pruning_steps.csv",
    }

    (domain_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    del model, yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary["elapsed_seconds"] = time.time() - started
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-model", type=Path, required=True)
    parser.add_argument("--snow-model", type=Path, required=True)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument(
        "--protected-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--maximum-root-pruning-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--absolute-minimum-channels",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--maximum-steps",
        type=int,
        default=2000,
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if not 0.0 < args.maximum_root_pruning_fraction < 1.0:
        raise SystemExit(
            "--maximum-root-pruning-fraction must be in (0, 1)"
        )

    for label, path in (
        ("GEN model", args.gen_model),
        ("SNOW model", args.snow_model),
        ("T4", args.t4),
        ("protected manifest", args.protected_manifest),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t4_rows = read_csv(args.t4)
    protected_rows = read_csv(args.protected_manifest)

    generic_rows = [
        row for row in t4_rows if row["group_kind"] == "GENERIC"
    ]
    custom_rows = [
        row for row in t4_rows if row["group_kind"] == "CUSTOM"
    ]

    if len(generic_rows) != 42 or len(custom_rows) != 9:
        raise RuntimeError(
            f"Unexpected T4 scope: generic={len(generic_rows)}, "
            f"custom={len(custom_rows)}"
        )

    protocol = {
        "experiment": (
            "Deterministic incremental Global L1 "
            "dependency-aware structured channel pruning"
        ),
        "stage": "5B V2 parameter-target search",
        "selection_rule": (
            "At every step, rank all currently eligible output filters "
            "from the 42 generic roots by raw L1 weight magnitude and "
            "prune the smallest safe dependency group."
        ),
        "accuracy_or_sensitivity_used_for_selection": False,
        "target_parameters": TARGET_PARAMS,
        "scope": {
            "generic_roots_included": 42,
            "custom_roots_excluded": 9,
            "protected_root_rows": len(protected_rows),
        },
        "trace_mode": "eval_all_outputs",
        "maximum_root_pruning_fraction": (
            args.maximum_root_pruning_fraction
        ),
        "absolute_minimum_channels": (
            args.absolute_minimum_channels
        ),
        "seed": args.seed,
        "no_model_saved": True,
        "no_training": True,
        "no_accuracy_evaluation": True,
    }
    (output_dir / "stage5b_v2_protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    environment = {
        "generated_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": getattr(
            ultralytics,
            "__version__",
            "unknown",
        ),
        "torch_pruning": getattr(tp, "__version__", "unknown"),
        "device": args.device,
        "cuda_available": torch.cuda.is_available(),
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    device = torch.device(args.device)

    summaries = [
        run_domain(
            domain="GEN",
            checkpoint=args.gen_model.resolve(),
            generic_rows=generic_rows,
            protected_rows=protected_rows,
            output_dir=output_dir,
            imgsz=args.imgsz,
            device=device,
            seed=args.seed,
            maximum_root_pruning_fraction=(
                args.maximum_root_pruning_fraction
            ),
            absolute_minimum_channels=(
                args.absolute_minimum_channels
            ),
            maximum_steps=args.maximum_steps,
        ),
        run_domain(
            domain="SNOW",
            checkpoint=args.snow_model.resolve(),
            generic_rows=generic_rows,
            protected_rows=protected_rows,
            output_dir=output_dir,
            imgsz=args.imgsz,
            device=device,
            seed=args.seed,
            maximum_root_pruning_fraction=(
                args.maximum_root_pruning_fraction
            ),
            absolute_minimum_channels=(
                args.absolute_minimum_channels
            ),
            maximum_steps=args.maximum_steps,
        ),
    ]

    checks = {
        "both_crossed_target": all(
            summary["target_crossed_during_search"]
            for summary in summaries
        ),
        "both_have_nonempty_replay_plan": all(
            summary["chosen_replay_steps"] > 0
            for summary in summaries
        ),
        "both_keep_protected_outputs": all(
            summary[
                "protected_outputs_unchanged_at_search_end"
            ]
            for summary in summaries
        ),
        "no_model_saved": True,
        "no_training": True,
        "no_accuracy_evaluation": True,
    }

    status = (
        "PASSED_INCREMENTAL_TARGET_SEARCH"
        if all(checks.values())
        else "FAILED_INCREMENTAL_TARGET_SEARCH"
    )

    final = {
        "status": status,
        "checks": checks,
        "domains": summaries,
        "methodological_limit": (
            "The 9 custom C3k2/C2PSA groups remain excluded because "
            "their pruning implementation was not included."
        ),
        "safe_next_step": (
            "Replay each chosen plan on a fresh checkpoint, save the "
            "raw structurally pruned model, reload it, verify its "
            "architecture, and run raw GEN/SNOW validation."
        ),
    }

    (output_dir / "stage5b_v2_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )

    print("\n===== STAGE 5B V2 COMPLETE =====")
    print(json.dumps(final, indent=2))
    print(f"\nOutput: {output_dir}")
    print("No pruned model checkpoint was saved.")


if __name__ == "__main__":
    main()
