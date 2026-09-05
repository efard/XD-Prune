#!/usr/bin/env python3
"""
Stage 5A V3: trace-mode diagnostic and corrected T4-root audit.

1. eval_all_outputs:
   Evaluation mode, but flatten every tensor in the complete model output,
   including the raw one-to-many and one-to-one dictionaries when available.
2. train_all_outputs:
   Training mode so the raw one-to-many branch is returned; BatchNorm layers
   remain in evaluation mode to avoid changing their running statistics.
3. train_default:
   Training mode with the unmodified output.

For every mode, AutoGrad is enabled and all model parameters are temporarily
set to requires_grad=True inside the in-memory audit model.

selects the mode that makes the largest number of the 42 authoritative T4
GENERIC roots available as valid DepGraph pruning groups.

This script does not prune, train, or save a modified checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
import ultralytics


EXPECTED_PARAMS = {"GEN": 2_508_090, "SNOW": 2_506_920}
EXPECTED_GENERIC = 42
EXPECTED_CUSTOM = 9
EXPECTED_TOTAL = 51
EXPECTED_PROTECTED = 54


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
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        raise RuntimeError("Model output did not contain any tensors.")
    return tuple(tensors)


def configure_model_for_trace(model: nn.Module, mode: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(True)

    if mode.startswith("train"):
        model.train()
        # Keep the detection head in training mode while preventing BatchNorm
        # running-stat changes during the trace.
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
    elif mode.startswith("eval"):
        model.eval()
    else:
        raise ValueError(f"Unknown trace mode: {mode}")


def build_dependency_graph(
    model: nn.Module,
    example: torch.Tensor,
    mode: str,
) -> tp.DependencyGraph:
    configure_model_for_trace(model, mode)
    model.zero_grad(set_to_none=True)

    kwargs: dict[str, Any] = {"example_inputs": example}

    signature = inspect.signature(tp.DependencyGraph.build_dependency)
    if mode.endswith("all_outputs"):
        if "output_transform" not in signature.parameters:
            raise RuntimeError(
                "Installed Torch-Pruning build_dependency() does not support "
                "output_transform."
            )
        kwargs["output_transform"] = all_tensor_output_transform

    with torch.enable_grad():
        return tp.DependencyGraph().build_dependency(model, **kwargs)


def probe_roots(
    dg: tp.DependencyGraph,
    model: nn.Module,
    generic_rows: list[dict[str, str]],
    validated_by_root: dict[str, dict[str, str]],
    domain: str,
) -> tuple[list[dict[str, Any]], int]:
    modules = dict(model.named_modules())
    rows: list[dict[str, Any]] = []

    for item in generic_rows:
        group_id = item["group_id"]
        root_path = item["representative_root"]
        module = modules.get(root_path)
        expected = validated_by_root.get(root_path, {}).get(
            "gen_depgraph_operations"
            if domain == "GEN"
            else "snow_depgraph_operations",
            "",
        )

        if module is None:
            rows.append(
                {
                    "group_id": group_id,
                    "root_module": root_path,
                    "status": "MISSING_MODULE",
                    "expected_operations": expected,
                    "actual_operations": "",
                    "error": "Root not found in model.named_modules().",
                }
            )
            continue

        if not isinstance(module, nn.Conv2d):
            rows.append(
                {
                    "group_id": group_id,
                    "root_module": root_path,
                    "status": "WRONG_TYPE",
                    "expected_operations": expected,
                    "actual_operations": "",
                    "error": f"Expected Conv2d, got {type(module).__name__}.",
                }
            )
            continue

        try:
            group = dg.get_pruning_group(
                module,
                tp.prune_conv_out_channels,
                idxs=[0],
            )
            valid = bool(dg.check_pruning_group(group))
            actual = len(group)
            status = "OK" if valid else "INVALID_GROUP"
            error = ""
        except Exception as exc:
            actual = ""
            status = "DEPGRAPH_ERROR"
            error = repr(exc)

        rows.append(
            {
                "group_id": group_id,
                "root_module": root_path,
                "status": status,
                "expected_operations": expected,
                "actual_operations": actual,
                "operation_count_matches": (
                    str(expected) == str(actual)
                    if expected != "" and actual != ""
                    else ""
                ),
                "error": error,
            }
        )

    ok_count = sum(row["status"] == "OK" for row in rows)
    return rows, ok_count


def audit_domain(
    domain: str,
    checkpoint: Path,
    generic_rows: list[dict[str, str]],
    validated_by_root: dict[str, dict[str, str]],
    output_dir: Path,
    imgsz: int,
    device: torch.device,
) -> dict[str, Any]:
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)

    # Independent baseline check.
    baseline_yolo = YOLO(str(checkpoint))
    baseline_model = baseline_yolo.model.to(device).eval()
    parameters = count_parameters(baseline_model)

    with torch.no_grad():
        _ = baseline_model(torch.randn(1, 3, imgsz, imgsz, device=device))

    del baseline_model, baseline_yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    trace_modes = [
        "eval_all_outputs",
        "train_all_outputs",
        "train_default",
    ]

    mode_summaries: list[dict[str, Any]] = []
    mode_rows: dict[str, list[dict[str, Any]]] = {}

    for mode in trace_modes:
        print(f"\n{domain}: testing trace mode {mode}")

        yolo = YOLO(str(checkpoint))
        model = yolo.model.to(device)
        example = torch.randn(
            1,
            3,
            imgsz,
            imgsz,
            device=device,
            requires_grad=True,
        )

        try:
            dg = build_dependency_graph(model, example, mode)
            rows, ok_count = probe_roots(
                dg,
                model,
                generic_rows,
                validated_by_root,
                domain,
            )
            error = ""
        except Exception as exc:
            rows = [
                {
                    "group_id": item["group_id"],
                    "root_module": item["representative_root"],
                    "status": "GRAPH_BUILD_ERROR",
                    "expected_operations": "",
                    "actual_operations": "",
                    "operation_count_matches": "",
                    "error": repr(exc),
                }
                for item in generic_rows
            ]
            ok_count = 0
            error = repr(exc)

        mode_rows[mode] = rows
        write_csv(domain_dir / f"trace_{mode}_root_audit.csv", rows)

        mode_summaries.append(
            {
                "domain": domain,
                "trace_mode": mode,
                "generic_roots_ok": ok_count,
                "generic_roots_requested": len(generic_rows),
                "graph_or_mode_error": error,
                "status_counts": dict(Counter(row["status"] for row in rows)),
            }
        )

        del model, yolo, example
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_csv(domain_dir / "trace_mode_comparison.csv", mode_summaries)

    selected = max(
        mode_summaries,
        key=lambda item: (
            item["generic_roots_ok"],
            item["trace_mode"] == "eval_all_outputs",
            item["trace_mode"] == "train_all_outputs",
        ),
    )
    selected_mode = selected["trace_mode"]

    # Rebuild the selected graph once and generate the final detailed audit and
    # the actual L1 channel ranking.
    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(device)
    example = torch.randn(
        1,
        3,
        imgsz,
        imgsz,
        device=device,
        requires_grad=True,
    )
    dg = build_dependency_graph(model, example, selected_mode)

    modules = dict(model.named_modules())
    names_by_id = {id(module): name for name, module in model.named_modules()}

    final_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    details_dir = domain_dir / "selected_group_details"
    details_dir.mkdir(exist_ok=True)

    for item in generic_rows:
        group_id = item["group_id"]
        root_path = item["representative_root"]
        module = modules[root_path]
        expected = validated_by_root.get(root_path, {}).get(
            "gen_depgraph_operations"
            if domain == "GEN"
            else "snow_depgraph_operations",
            "",
        )

        try:
            group = dg.get_pruning_group(
                module,
                tp.prune_conv_out_channels,
                idxs=[0],
            )
            valid = bool(dg.check_pruning_group(group))
            actual = len(group)
            status = "OK" if valid else "INVALID_GROUP"
            error = ""
            details = group.details()

            dependency_names: list[str] = []
            for dep, _ in group:
                try:
                    dependency_names.append(
                        names_by_id.get(id(dep.target.module), "<unresolved>")
                    )
                except Exception:
                    dependency_names.append("<unparsed>")
        except Exception as exc:
            group = None
            valid = False
            actual = ""
            status = "DEPGRAPH_ERROR"
            error = repr(exc)
            details = repr(exc)
            dependency_names = []

        (details_dir / f"{group_id}.txt").write_text(
            details,
            encoding="utf-8",
        )

        weights = module.weight.detach().float()
        l1_scores = weights.abs().sum(
            dim=tuple(range(1, weights.ndim))
        ).cpu()

        final_rows.append(
            {
                "group_id": group_id,
                "root_module": root_path,
                "status": status,
                "error": error,
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "expected_operations": expected,
                "actual_operations": actual,
                "operation_count_matches": (
                    str(expected) == str(actual)
                    if expected != "" and actual != ""
                    else ""
                ),
                "probe_group_valid": valid,
                "dependency_module_count": len(set(dependency_names)),
                "l1_channel_count": len(l1_scores),
                "l1_min": float(l1_scores.min()),
                "l1_mean": float(l1_scores.mean()),
                "l1_max": float(l1_scores.max()),
                "details_file": f"selected_group_details/{group_id}.txt",
            }
        )

        for channel_index, value in enumerate(l1_scores.tolist()):
            score_rows.append(
                {
                    "domain": domain,
                    "group_id": group_id,
                    "root_module": root_path,
                    "root_channel_index": channel_index,
                    "l1_score": float(value),
                }
            )

    score_rows.sort(
        key=lambda row: (
            row["l1_score"],
            row["group_id"],
            row["root_channel_index"],
        )
    )
    for rank, row in enumerate(score_rows, start=1):
        row["global_l1_rank"] = rank

    write_csv(domain_dir / "selected_42_root_dependency_audit.csv", final_rows)
    write_csv(domain_dir / "global_l1_ranking_42_generic_roots.csv", score_rows)

    selected_ok = sum(row["status"] == "OK" for row in final_rows)
    summary = {
        "domain": domain,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "parameters": parameters,
        "expected_parameters": EXPECTED_PARAMS[domain],
        "parameter_match": parameters == EXPECTED_PARAMS[domain],
        "forward_pass_ok": True,
        "selected_trace_mode": selected_mode,
        "selected_generic_roots_ok": selected_ok,
        "generic_roots_requested": len(generic_rows),
        "global_l1_channel_scores": len(score_rows),
        "trace_mode_results": mode_summaries,
        "audit_only": True,
    }
    (domain_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-model", type=Path, required=True)
    parser.add_argument("--snow-model", type=Path, required=True)
    parser.add_argument("--t4", type=Path, required=True)
    parser.add_argument("--validated-manifest", type=Path, required=True)
    parser.add_argument("--protected-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    for label, path in (
        ("GEN model", args.gen_model),
        ("SNOW model", args.snow_model),
        ("T4 table", args.t4),
        ("validated manifest", args.validated_manifest),
        ("protected manifest", args.protected_manifest),
    ):
        if not path.is_file():
            raise SystemExit(f"{label} not found: {path}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    t4_rows = read_csv(args.t4)
    validated_rows = read_csv(args.validated_manifest)
    protected_rows = read_csv(args.protected_manifest)

    generic_rows = [
        row for row in t4_rows if row["group_kind"] == "GENERIC"
    ]
    custom_rows = [
        row for row in t4_rows if row["group_kind"] == "CUSTOM"
    ]
    validated_by_root = {
        row["root_module_path"]: row for row in validated_rows
    }

    write_csv(output_dir / "t4_generic_42_scope.csv", generic_rows)
    write_csv(output_dir / "t4_custom_9_scope_pending_code.csv", custom_rows)
    write_csv(output_dir / "protected_54_roots.csv", protected_rows)

    environment = {
        "generated_unix": time.time(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": getattr(ultralytics, "__version__", "unknown"),
        "torch_pruning": getattr(tp, "__version__", "unknown"),
        "build_dependency_signature": str(
            inspect.signature(tp.DependencyGraph.build_dependency)
        ),
        "device": args.device,
        "audit_only": True,
    }
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2),
        encoding="utf-8",
    )

    scope_checks = {
        "t4_rows_51": len(t4_rows) == EXPECTED_TOTAL,
        "generic_rows_42": len(generic_rows) == EXPECTED_GENERIC,
        "custom_rows_9": len(custom_rows) == EXPECTED_CUSTOM,
        "protected_rows_54": len(protected_rows) == EXPECTED_PROTECTED,
    }

    device = torch.device(args.device)
    summaries = [
        audit_domain(
            "GEN",
            args.gen_model.resolve(),
            generic_rows,
            validated_by_root,
            output_dir,
            args.imgsz,
            device,
        ),
        audit_domain(
            "SNOW",
            args.snow_model.resolve(),
            generic_rows,
            validated_by_root,
            output_dir,
            args.imgsz,
            device,
        ),
    ]

    model_checks = {
        "both_parameter_counts_match": all(
            summary["parameter_match"] for summary in summaries
        ),
        "both_forward_passes_ok": all(
            summary["forward_pass_ok"] for summary in summaries
        ),
        "both_have_42_valid_generic_roots": all(
            summary["selected_generic_roots_ok"] == EXPECTED_GENERIC
            for summary in summaries
        ),
        "both_have_3528_l1_channel_scores": all(
            summary["global_l1_channel_scores"] == 3528
            for summary in summaries
        ),
        "no_pruning_performed": True,
    }

    passed = all(scope_checks.values()) and all(model_checks.values())

    final = {
        "status": (
            "PASSED_GENERIC_42_TRACE_AUDIT"
            if passed
            else "FAILED_TRACE_AUDIT"
        ),
        "scope_checks": scope_checks,
        "model_checks": model_checks,
        "models": summaries,
        "cause_of_v2_failure": (
            "V2 built DepGraph from the default YOLO26 evaluation output. "
            "The selected end-to-end inference output did not retain the "
            "42 T4 backbone/neck roots in the traced graph."
        ),
        "custom_scope_limitation": (
            "The 9 custom C3k2/C2PSA implementations are still absent. "
            "The current reproducible comparator is limited to the 42 "
            "validated generic roots."
        ),
        "safe_next_step": (
            "If status passes, build the temporary-copy parameter-target "
            "search over the 42 generic roots using the selected trace mode."
        ),
    }
    (output_dir / "stage5a_v3_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )

    print("\n===== STAGE 5A V3 COMPLETE =====")
    print(json.dumps(final, indent=2))
    print(f"\nOutput: {output_dir}")
    print("No pruning or training was performed.")


if __name__ == "__main__":
    main()
