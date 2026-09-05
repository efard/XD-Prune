#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer

from layer_replacement_adapter import (
    DynamicNoParamLayerAdapter,
    reduced_fraction,
)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in value:
            out.extend(flatten_tensors(item))
        return out
    if isinstance(value, dict):
        out: list[torch.Tensor] = []
        for item in value.values():
            out.extend(flatten_tensors(item))
        return out
    return []


def metric_value(metrics: Any, path: str) -> float | None:
    value = metrics
    for component in path.split("."):
        value = getattr(value, component, None)
        if value is None:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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


def architecture_rows(model: nn.Module) -> list[dict[str, Any]]:
    rows = []
    for name, module in model.named_modules():
        rows.append(
            {
                "name": name or "<root>",
                "type": module.__class__.__name__,
                "direct_params": sum(
                    p.numel()
                    for p in module.parameters(recurse=False)
                ),
            }
        )
    return rows


def architecture_sha256(model: nn.Module) -> str:
    payload = json.dumps(
        architecture_rows(model),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def count_bn(model: nn.Module) -> int:
    return sum(
        1
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
    )


def head_training_capability(model: nn.Module) -> dict[str, Any]:
    head = model.model[-1]
    cv2 = getattr(head, "cv2", None)
    cv3 = getattr(head, "cv3", None)
    return {
        "head_type": head.__class__.__name__,
        "end2end": bool(getattr(head, "end2end", False)),
        "cv2_present": cv2 is not None,
        "cv3_present": cv3 is not None,
        "one2many_training_head_present": (
            cv2 is not None and cv3 is not None
        ),
    }


def assert_trainable_unfused(model: nn.Module, label: str) -> None:
    head = head_training_capability(model)
    bn_count = count_bn(model)

    print(
        f"{label}: params={count_params(model):,}, "
        f"BatchNorm2d={bn_count}, "
        f"head_cv2={head['cv2_present']}, "
        f"head_cv3={head['cv3_present']}"
    )

    if bn_count == 0:
        raise RuntimeError(
            f"{label} appears fused: no BatchNorm2d layers remain."
        )

    if not head["one2many_training_head_present"]:
        raise RuntimeError(
            f"{label} is inference-fused: YOLO26 one-to-many "
            "training head is missing."
        )


def check_forward(
    model: nn.Module,
    imgsz: int,
    device: torch.device,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    with torch.no_grad():
        output = model(
            torch.zeros(1, 3, imgsz, imgsz, device=device)
        )
    tensors = flatten_tensors(output)
    if was_training:
        model.train()
    return {
        "imgsz": imgsz,
        "forward_ok": bool(tensors),
        "all_finite": bool(tensors)
        and all(
            bool(torch.isfinite(t).all().item())
            for t in tensors
        ),
        "output_tensor_count": len(tensors),
    }


def discover_single_layer_csv(project_root: Path) -> Path:
    candidates = []
    for base in [
        project_root / "5_reproduction/stage2_results",
        project_root / "4_swprunability/results",
    ]:
        if base.is_dir():
            candidates.extend(base.rglob("*.csv"))

    candidates = sorted(
        {p.resolve() for p in candidates if p.is_file()},
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    required = {"layer_id", "status", "ad_gen_map50_95"}

    for path in candidates:
        try:
            with path.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                fields = set(reader.fieldnames or [])
        except Exception:
            continue
        if required.issubset(fields):
            return path

    raise FileNotFoundError(
        "Could not locate previous single-layer replacement CSV."
    )


def load_candidates(
    csv_path: Path,
    model: nn.Module,
    baseline_params: int,
) -> list[dict[str, Any]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    candidates = []

    for row in rows:
        if str(row.get("status", "")).lower() != "ok":
            continue

        try:
            layer_id = int(row["layer_id"])
        except Exception:
            continue

        if not 0 <= layer_id < len(model.model):
            continue

        layer = model.model[layer_id]

        if layer.__class__.__name__ == "Detect":
            continue

        layer_params = count_params(layer)
        if layer_params <= 0:
            continue

        ad = parse_float(row.get("ad_gen_map50_95"))
        if ad is None:
            continue

        candidates.append(
            {
                "layer_id": layer_id,
                "layer_type": layer.__class__.__name__,
                "layer_params": layer_params,
                "reduction_percent_of_unfused_baseline": (
                    100.0 * layer_params / baseline_params
                ),
                "single_layer_gen_ad": ad,
            }
        )

    candidates.sort(key=lambda row: row["layer_id"])

    if not candidates:
        raise RuntimeError("No usable layer candidates found.")

    if len(candidates) > 22:
        raise RuntimeError(
            f"Too many candidates for subset search: {len(candidates)}"
        )

    return candidates


def rank_layers_by_efficiency(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank individual layers by:
        parameter_reduction_percent / previous_GEN_accuracy_drop

    Higher is better.
    Stage 2 GEN has no negative AD values.
    """
    ranked = []

    for row in candidates:
        ad = float(row["single_layer_gen_ad"])
        reduction = float(
            row["reduction_percent_of_unfused_baseline"]
        )

        if ad <= 0:
            efficiency = float("inf")
        else:
            efficiency = reduction / ad

        ranked.append(
            {
                **row,
                "efficiency_param_percent_per_ad": efficiency,
            }
        )

    ranked.sort(
        key=lambda row: (
            -row["efficiency_param_percent_per_ad"],
            -row["reduction_percent_of_unfused_baseline"],
            row["single_layer_gen_ad"],
            row["layer_id"],
        )
    )

    cumulative = 0.0
    for rank, row in enumerate(ranked, start=1):
        cumulative += row[
            "reduction_percent_of_unfused_baseline"
        ]
        row["efficiency_rank"] = rank
        row["cumulative_reduction_percent"] = cumulative

    return ranked


def greedy_select_until_target(
    ranked_layers: list[dict[str, Any]],
    target_percent: float,
) -> list[dict[str, Any]]:
    """
    Add ranked layers one-by-one until cumulative parameter reduction
    reaches or exceeds target_percent.
    """
    selected = []
    cumulative = 0.0

    for row in ranked_layers:
        selected.append(row)
        cumulative += row[
            "reduction_percent_of_unfused_baseline"
        ]

        if cumulative >= target_percent:
            break

    if cumulative < target_percent:
        raise RuntimeError(
            f"All eligible layers together reach only "
            f"{cumulative:.4f}% < target {target_percent:.4f}%."
        )

    return selected


def capture_interfaces(
    model: nn.Module,
    selected_ids: set[int],
    imgsz: int,
    device: torch.device,
) -> dict[int, dict[str, Any]]:
    interfaces: dict[int, dict[str, Any]] = {}
    handles = []

    def make_hook(layer_id: int):
        def hook(
            module: nn.Module,
            inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            sources = flatten_tensors(inputs)
            outputs = flatten_tensors(output)

            if not sources or len(outputs) != 1:
                raise RuntimeError(
                    f"Layer {layer_id} interface capture failed."
                )

            reference = sources[0]
            out = outputs[0]

            if reference.ndim != 4 or out.ndim != 4:
                raise RuntimeError(
                    f"Layer {layer_id} is not NCHW-to-NCHW."
                )

            h_num, h_den = reduced_fraction(
                int(out.shape[-2]),
                int(reference.shape[-2]),
            )
            w_num, w_den = reduced_fraction(
                int(out.shape[-1]),
                int(reference.shape[-1]),
            )

            interfaces[layer_id] = {
                "target_channels": int(out.shape[1]),
                "spatial_h_num": h_num,
                "spatial_h_den": h_den,
                "spatial_w_num": w_num,
                "spatial_w_den": w_den,
            }

        return hook

    for layer_id in sorted(selected_ids):
        handles.append(
            model.model[layer_id].register_forward_hook(
                make_hook(layer_id)
            )
        )

    try:
        model.eval()
        with torch.no_grad():
            model(
                torch.zeros(
                    1,
                    3,
                    imgsz,
                    imgsz,
                    device=device,
                )
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = selected_ids - set(interfaces)
    if missing:
        raise RuntimeError(
            f"Missing captured interfaces for {sorted(missing)}"
        )

    return interfaces


def attach_metadata(
    adapter: nn.Module,
    old_layer: nn.Module,
    layer_id: int,
) -> None:
    adapter.f = getattr(old_layer, "f", -1)
    adapter.i = getattr(old_layer, "i", layer_id)
    adapter.type = (
        "DynamicNoParamLayerAdapter_for_"
        + old_layer.__class__.__name__
    )
    adapter.np = 0


def build_raw_model(
    baseline_path: Path,
    selected_ids: list[int],
    imgsz: int,
    device: torch.device,
    out_path: Path,
) -> dict[str, Any]:
    yolo = YOLO(str(baseline_path))
    yolo.model = yolo.model.to(device).float()

    assert_trainable_unfused(
        yolo.model,
        "Fresh baseline builder",
    )

    interfaces = capture_interfaces(
        yolo.model,
        set(selected_ids),
        imgsz,
        device,
    )

    replacement_rows = []

    for layer_id in sorted(selected_ids):
        old_layer = yolo.model.model[layer_id]
        old_params = count_params(old_layer)
        interface = interfaces[layer_id]

        adapter = DynamicNoParamLayerAdapter(
            target_channels=interface["target_channels"],
            spatial_h_num=interface["spatial_h_num"],
            spatial_h_den=interface["spatial_h_den"],
            spatial_w_num=interface["spatial_w_num"],
            spatial_w_den=interface["spatial_w_den"],
            multi_input_policy="concat",
        ).to(device)

        attach_metadata(adapter, old_layer, layer_id)
        yolo.model.model[layer_id] = adapter

        replacement_rows.append(
            {
                "layer_id": layer_id,
                "old_type": old_layer.__class__.__name__,
                "removed_parameters": old_params,
            }
        )

    assert_trainable_unfused(
        yolo.model,
        "Raw replacement model before save",
    )

    forward_640 = check_forward(
        yolo.model,
        imgsz,
        device,
    )
    forward_320 = check_forward(
        yolo.model,
        320,
        device,
    )

    if not (
        forward_640["forward_ok"]
        and forward_640["all_finite"]
        and forward_320["forward_ok"]
        and forward_320["all_finite"]
    ):
        raise RuntimeError("Replacement model forward failed.")

    params = count_params(yolo.model)
    arch_sha = architecture_sha256(yolo.model)
    bn_count = count_bn(yolo.model)
    head = head_training_capability(yolo.model)

    yolo.save(str(out_path))

    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Reload WITHOUT validation to prove disk checkpoint is still unfused.
    reloaded = YOLO(str(out_path))
    reloaded.model = reloaded.model.to(device).float()
    assert_trainable_unfused(
        reloaded.model,
        "Reloaded raw checkpoint",
    )

    if count_params(reloaded.model) != params:
        raise RuntimeError("Raw checkpoint parameter count changed on reload.")

    if architecture_sha256(reloaded.model) != arch_sha:
        raise RuntimeError("Raw checkpoint architecture changed on reload.")

    reload_forward = check_forward(
        reloaded.model,
        imgsz,
        device,
    )

    del reloaded
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "parameters": params,
        "architecture_sha256": arch_sha,
        "batchnorm_layers": bn_count,
        "head": head,
        "forward_640": forward_640,
        "forward_320": forward_320,
        "reload_forward_640": reload_forward,
        "replacement_rows": replacement_rows,
    }


def validate_checkpoint(
    path: Path,
    data: Path,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    seed: int,
) -> dict[str, float | None]:
    # validation may fuse THIS IN-MEMORY INSTANCE.
    # It is intentionally loaded only for evaluation and discarded.
    yolo = YOLO(str(path))
    metrics = yolo.val(
        data=str(data.resolve()),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        augment=False,
        plots=False,
        save_json=False,
        verbose=True,
        seed=seed,
    )
    result = {
        "map50_95": metric_value(metrics, "box.map"),
        "map50": metric_value(metrics, "box.map50"),
        "map75": metric_value(metrics, "box.map75"),
        "precision": metric_value(metrics, "box.mp"),
        "recall": metric_value(metrics, "box.mr"),
    }
    del yolo
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_results_csv(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise RuntimeError("Training results.csv is empty.")

    nonfinite = []
    parsed_rows = []

    for row_index, row in enumerate(rows, start=1):
        parsed = {}
        for key, value in row.items():
            if value in (None, ""):
                continue
            try:
                number = float(value)
            except ValueError:
                continue
            parsed[key] = number
            if not math.isfinite(number):
                nonfinite.append(
                    {
                        "row": row_index,
                        "column": key,
                        "value": value,
                    }
                )
        parsed_rows.append(parsed)

    return {
        "rows": len(rows),
        "has_expected_rows": len(rows) == 20,
        "all_numeric_values_finite": not nonfinite,
        "nonfinite_values": nonfinite,
        "best_training_map50_95": max(
            (
                row["metrics/mAP50-95(B)"]
                for row in parsed_rows
                if "metrics/mAP50-95(B)" in row
            ),
            default=None,
        ),
        "last_numeric_values": parsed_rows[-1],
    }


class ExactStructureTrainer(DetectionTrainer):
    external_model: nn.Module | None = None

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose=True,
    ) -> nn.Module:
        model = type(self).external_model

        if model is None:
            raise RuntimeError(
                "ExactStructureTrainer did not receive "
                "the raw unfused model object."
            )

        type(self).external_model = None

        assert_trainable_unfused(
            model,
            "Trainer-injected raw model",
        )

        model = model.float()

        for parameter in model.parameters():
            parameter.requires_grad_(True)

        return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--single-layer-csv", type=Path)
    parser.add_argument("--target-percent", type=float, default=50.0)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for path in [args.baseline_model, args.data]:
        if not path.is_file():
            raise SystemExit(f"Required file missing: {path}")

    set_deterministic(args.seed)

    device = torch.device(
        "cuda:0"
        if args.device != "cpu"
        and torch.cuda.is_available()
        else "cpu"
    )

    # ---------------------------------------------------------------
    # STEP 1: plan and build from a FRESH, NEVER-VALIDATED checkpoint.
    # ---------------------------------------------------------------
    print("\n===== STEP 1: BUILD UNFUSED ~50% RAW MODEL =====")

    planner = YOLO(str(args.baseline_model.resolve()))
    planner.model = planner.model.to(device).float()

    baseline_params = count_params(planner.model)
    baseline_bn = count_bn(planner.model)
    baseline_head = head_training_capability(planner.model)

    assert_trainable_unfused(
        planner.model,
        "Unfused baseline planner",
    )

    single_layer_csv = (
        args.single_layer_csv.resolve()
        if args.single_layer_csv is not None
        else discover_single_layer_csv(
            args.project_root.resolve()
        )
    )

    candidates = load_candidates(
        single_layer_csv,
        planner.model,
        baseline_params,
    )

    ranked_layers = rank_layers_by_efficiency(
        candidates
    )

    with (
        args.out_dir
        / "layer_efficiency_ranking.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "efficiency_rank",
                "layer_id",
                "layer_type",
                "layer_params",
                "reduction_percent_of_unfused_baseline",
                "single_layer_gen_ad",
                "efficiency_param_percent_per_ad",
                "cumulative_reduction_percent",
            ],
        )
        writer.writeheader()
        for row in ranked_layers:
            writer.writerow(
                {
                    key: row[key]
                    for key in writer.fieldnames
                }
            )

    selected_rows = greedy_select_until_target(
        ranked_layers,
        args.target_percent,
    )

    selected_ids = [
        int(row["layer_id"])
        for row in selected_rows
    ]

    expected_reduction = sum(
        row["reduction_percent_of_unfused_baseline"]
        for row in selected_rows
    )

    print("\n===== EFFICIENCY RANKING / GREEDY SELECTION =====")
    for row in ranked_layers:
        marker = "*" if row["layer_id"] in selected_ids else " "
        efficiency = row["efficiency_param_percent_per_ad"]
        efficiency_text = (
            "inf"
            if math.isinf(efficiency)
            else f"{efficiency:.6f}"
        )
        print(
            f"{marker} rank={row['efficiency_rank']:2d} "
            f"L{row['layer_id']:02d} "
            f"{row['layer_type']:<6s} "
            f"reduce={row['reduction_percent_of_unfused_baseline']:.4f}% "
            f"AD={row['single_layer_gen_ad']:.6f} "
            f"eff={efficiency_text} "
            f"cum={row['cumulative_reduction_percent']:.4f}%"
        )

    print(
        "\nSelected in ranking order:",
        [row["layer_id"] for row in selected_rows],
    )
    print(
        "Selected layer IDs sorted:",
        sorted(selected_ids),
    )
    print(
        f"Expected cumulative reduction: "
        f"{expected_reduction:.4f}%"
    )

    del planner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    raw_path = (
        args.out_dir
        / "GEN_layer_replacement_efficiency_greedy_raw_unfused.pt"
    )

    print(
        "\nBuilding exact greedy-selected raw model with layers:",
        sorted(selected_ids),
    )

    build_info = build_raw_model(
        args.baseline_model.resolve(),
        selected_ids,
        args.imgsz,
        device,
        raw_path,
    )

    chosen = {
        "layer_ids": selected_ids,
        "expected_reduction_percent": expected_reduction,
        "selection_order": [
            row["layer_id"]
            for row in selected_rows
        ],
        "efficiency_values": {
            str(row["layer_id"]): (
                None
                if math.isinf(
                    row[
                        "efficiency_param_percent_per_ad"
                    ]
                )
                else row[
                    "efficiency_param_percent_per_ad"
                ]
            )
            for row in selected_rows
        },
    }

    raw_params = build_info["parameters"]
    actual_reduction = (
        100.0
        * (baseline_params - raw_params)
        / baseline_params
    )

    print("\n===== CHOSEN GREEDY PLAN =====")
    print(
        "Selection order:",
        chosen["selection_order"],
    )
    print(
        "Layers sorted:",
        sorted(chosen["layer_ids"]),
    )
    print(f"Baseline params: {baseline_params:,}")
    print(f"Raw params:      {raw_params:,}")
    print(f"Actual reduction: {actual_reduction:.4f}%")
    print(
        f"Raw BatchNorm2d layers: "
        f"{build_info['batchnorm_layers']}"
    )
    print(
        "Training head present:",
        build_info["head"][
            "one2many_training_head_present"
        ],
    )

    with (
        args.out_dir / "selected_layers.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "selection_order",
                "efficiency_rank",
                "layer_id",
                "old_type",
                "removed_parameters",
                "single_layer_gen_ad",
                "individual_reduction_percent",
                "efficiency_param_percent_per_ad",
                "cumulative_reduction_percent",
            ],
        )
        writer.writeheader()

        replacement_by_id = {
            int(row["layer_id"]): row
            for row in build_info["replacement_rows"]
        }

        cumulative = 0.0
        for order, row in enumerate(
            selected_rows,
            start=1,
        ):
            cumulative += row[
                "reduction_percent_of_unfused_baseline"
            ]
            replacement = replacement_by_id[
                int(row["layer_id"])
            ]
            writer.writerow(
                {
                    "selection_order": order,
                    "efficiency_rank": row[
                        "efficiency_rank"
                    ],
                    "layer_id": row["layer_id"],
                    "old_type": replacement[
                        "old_type"
                    ],
                    "removed_parameters": replacement[
                        "removed_parameters"
                    ],
                    "single_layer_gen_ad": row[
                        "single_layer_gen_ad"
                    ],
                    "individual_reduction_percent": row[
                        "reduction_percent_of_unfused_baseline"
                    ],
                    "efficiency_param_percent_per_ad": row[
                        "efficiency_param_percent_per_ad"
                    ],
                    "cumulative_reduction_percent": cumulative,
                }
            )

    adapter_copy = (
        args.out_dir
        / "layer_replacement_adapter.py"
    )
    shutil.copy2(
        Path(__file__).resolve().parent
        / "layer_replacement_adapter.py",
        adapter_copy,
    )

    # ---------------------------------------------------------------
    # STEP 2: validation on throwaway model instances only.
    # ---------------------------------------------------------------
    print("\n===== STEP 2: BASELINE GEN VALIDATION =====")
    baseline_metrics = validate_checkpoint(
        args.baseline_model,
        args.data,
        args.imgsz,
        args.batch,
        args.workers,
        args.device,
        args.seed,
    )

    print("\n===== STEP 3: RAW GEN VALIDATION =====")
    raw_metrics = validate_checkpoint(
        raw_path,
        args.data,
        args.imgsz,
        args.batch,
        args.workers,
        args.device,
        args.seed,
    )

    # Prove validation did NOT mutate the saved disk checkpoint.
    raw_after_validation = YOLO(str(raw_path))
    raw_after_validation.model = (
        raw_after_validation.model.to(device).float()
    )
    assert_trainable_unfused(
        raw_after_validation.model,
        "Disk raw checkpoint after validation",
    )

    if count_params(raw_after_validation.model) != raw_params:
        raise RuntimeError(
            "Saved raw checkpoint was unexpectedly changed."
        )

    del raw_after_validation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # STEP 4: training from ANOTHER fresh load of the unfused raw .pt.
    # ---------------------------------------------------------------
    print(
        "\n===== STEP 4: 20-EPOCH EXACT-STRUCTURE RECOVERY ====="
    )

    training_source = YOLO(str(raw_path))
    training_source.model = (
        training_source.model.to(device).float()
    )

    assert_trainable_unfused(
        training_source.model,
        "Fresh raw training source",
    )

    if count_params(training_source.model) != raw_params:
        raise RuntimeError(
            "Training source parameter mismatch."
        )

    if (
        architecture_sha256(training_source.model)
        != build_info["architecture_sha256"]
    ):
        raise RuntimeError(
            "Training source architecture mismatch."
        )

    ExactStructureTrainer.external_model = (
        training_source.model
    )

    train_overrides = {
        "model": str(raw_path),
        "data": str(args.data.resolve()),
        "task": "detect",
        "mode": "train",
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device": args.device,
        "workers": args.workers,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "pretrained": False,
        "amp": True,
        "deterministic": True,
        "seed": args.seed,
        "rect": False,
        "mosaic": 1.0,
        "close_mosaic": 0,
        "val": True,
        "save": True,
        "save_period": -1,
        "plots": False,
        "verbose": True,
        "cache": False,
        "resume": False,
        "project": str(args.out_dir),
        "name": "train_run",
        "exist_ok": True,
        "patience": 100,
    }

    trainer = ExactStructureTrainer(
        overrides=train_overrides
    )

    started = time.time()
    trainer.train()
    training_elapsed = time.time() - started

    best_path = Path(trainer.best)
    last_path = Path(trainer.last)
    results_csv = Path(trainer.csv)

    training_results = parse_results_csv(
        results_csv
    )

    recovered_best = (
        args.out_dir
        / "GEN_layer_replacement_efficiency_greedy_recovered_best.pt"
    )
    recovered_last = (
        args.out_dir
        / "GEN_layer_replacement_efficiency_greedy_recovered_last.pt"
    )

    shutil.copy2(best_path, recovered_best)
    shutil.copy2(last_path, recovered_last)

    # Load before final validation to verify training architecture.
    best_verify = YOLO(str(recovered_best))
    best_verify.model = best_verify.model.to(device).float()

    best_params = count_params(best_verify.model)
    best_arch = architecture_sha256(best_verify.model)
    best_bn = count_bn(best_verify.model)
    best_head = head_training_capability(best_verify.model)

    if best_params != raw_params:
        raise RuntimeError(
            f"Recovered best params changed: "
            f"{best_params} != {raw_params}"
        )

    if best_arch != build_info["architecture_sha256"]:
        raise RuntimeError(
            "Recovered best architecture changed."
        )

    del best_verify
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---------------------------------------------------------------
    # STEP 5: final explicit evaluation on a throwaway load.
    # ---------------------------------------------------------------
    print("\n===== STEP 5: FINAL RECOVERED GEN VALIDATION =====")

    recovered_metrics = validate_checkpoint(
        recovered_best,
        args.data,
        args.imgsz,
        args.batch,
        args.workers,
        args.device,
        args.seed,
    )

    baseline_map = baseline_metrics["map50_95"]
    raw_map = raw_metrics["map50_95"]
    recovered_map = recovered_metrics["map50_95"]

    if (
        baseline_map is None
        or raw_map is None
        or recovered_map is None
    ):
        raise RuntimeError(
            "Could not extract complete mAP50-95 metrics."
        )

    checks = {
        "unfused_baseline_used_for_planning": (
            baseline_bn > 0
            and baseline_head[
                "one2many_training_head_present"
            ]
        ),
        "raw_checkpoint_unfused": (
            build_info["batchnorm_layers"] > 0
            and build_info["head"][
                "one2many_training_head_present"
            ]
        ),
        "raw_parameter_target_met": (
            actual_reduction >= args.target_percent
        ),
        "training_completed": True,
        "training_has_20_rows": (
            training_results["has_expected_rows"]
        ),
        "training_values_finite": (
            training_results[
                "all_numeric_values_finite"
            ]
        ),
        "recovered_params_preserved": (
            best_params == raw_params
        ),
        "recovered_architecture_preserved": (
            best_arch
            == build_info["architecture_sha256"]
        ),
        "final_validation_completed": True,
    }

    summary = {
        "status": (
            "PASSED_GEN50_EFFICIENCY_GREEDY_FULL"
            if all(checks.values())
            else "FAILED_GEN50_LAYER_REPLACEMENT_FULL_V2"
        ),
        "checks": checks,
        "method_note": (
            "All planning/building uses the unfused checkpoint. "
            "Validation is performed only on throwaway model "
            "instances because YOLO26 validation may fuse the "
            "in-memory model and remove its one-to-many training head."
        ),
        "selection_note": (
            "Individual layers are ranked by parameter-reduction "
            "percentage divided by their previously measured Stage 2 "
            "GEN mAP50-95 accuracy drop. Layers are then added greedily "
            "from highest efficiency to lowest until cumulative "
            "parameter reduction reaches or exceeds the target."
        ),
        "selected_layers": sorted(chosen["layer_ids"]),
        "selection_order": chosen["selection_order"],
        "expected_reduction_percent_from_stage2_ranking": (
            chosen["expected_reduction_percent"]
        ),
        "baseline_parameters": baseline_params,
        "raw_parameters": raw_params,
        "parameters_removed": (
            baseline_params - raw_params
        ),
        "parameter_reduction_percent": (
            actual_reduction
        ),
        "baseline_batchnorm_layers": baseline_bn,
        "raw_batchnorm_layers": (
            build_info["batchnorm_layers"]
        ),
        "baseline_metrics": baseline_metrics,
        "raw_metrics": {
            **raw_metrics,
            "signed_ad_map50_95": (
                baseline_map - raw_map
            ),
            "accuracy_retention_percent": (
                100.0 * raw_map / baseline_map
            ),
        },
        "recovered_metrics": {
            **recovered_metrics,
            "signed_ad_map50_95": (
                baseline_map - recovered_map
            ),
            "accuracy_retention_percent": (
                100.0
                * recovered_map
                / baseline_map
            ),
        },
        "raw_model": str(raw_path.resolve()),
        "raw_model_sha256": sha256_file(raw_path),
        "recovered_best_model": str(
            recovered_best.resolve()
        ),
        "recovered_best_sha256": sha256_file(
            recovered_best
        ),
        "recovered_last_model": str(
            recovered_last.resolve()
        ),
        "training_results_csv": str(
            results_csv.resolve()
        ),
        "training_results": training_results,
        "training_elapsed_seconds": (
            training_elapsed
        ),
        "build_failures_before_selected_plan": (
            build_failures
        ),
        "recovered_best_batchnorm_layers": best_bn,
        "recovered_best_head": best_head,
    }

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    report_table = args.out_dir / "report_table.csv"
    with report_table.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "parameters",
                "parameter_reduction_percent",
                "map50_95",
                "map50",
                "signed_accuracy_drop_map50_95",
                "accuracy_retention_percent",
            ],
        )
        writer.writeheader()

        writer.writerow(
            {
                "stage": "baseline",
                "parameters": baseline_params,
                "parameter_reduction_percent": 0.0,
                "map50_95": baseline_map,
                "map50": baseline_metrics["map50"],
                "signed_accuracy_drop_map50_95": 0.0,
                "accuracy_retention_percent": 100.0,
            }
        )

        writer.writerow(
            {
                "stage": "raw_around50",
                "parameters": raw_params,
                "parameter_reduction_percent": actual_reduction,
                "map50_95": raw_map,
                "map50": raw_metrics["map50"],
                "signed_accuracy_drop_map50_95": (
                    baseline_map - raw_map
                ),
                "accuracy_retention_percent": (
                    100.0 * raw_map / baseline_map
                ),
            }
        )

        writer.writerow(
            {
                "stage": "recovered_around50",
                "parameters": raw_params,
                "parameter_reduction_percent": actual_reduction,
                "map50_95": recovered_map,
                "map50": recovered_metrics["map50"],
                "signed_accuracy_drop_map50_95": (
                    baseline_map - recovered_map
                ),
                "accuracy_retention_percent": (
                    100.0
                    * recovered_map
                    / baseline_map
                ),
            }
        )

    readme = (
        "GEN >=50% WHOLE-LAYER REPLACEMENT — EFFICIENCY GREEDY\n"
        "================================================\n\n"
        f"Selected layers: {','.join(str(x) for x in chosen['layer_ids'])}\n"
        f"Baseline parameters: {baseline_params}\n"
        f"Raw parameters: {raw_params}\n"
        f"Reduction: {actual_reduction:.6f}%\n"
        f"Baseline mAP50-95: {baseline_map:.9f}\n"
        f"Raw mAP50-95: {raw_map:.9f}\n"
        f"Recovered mAP50-95: {recovered_map:.9f}\n\n"
        "This version deliberately keeps the raw training checkpoint "
        "unfused. Evaluation uses separate throwaway model instances.\n"
    )

    readme_path = args.out_dir / "README.txt"
    readme_path.write_text(
        readme,
        encoding="utf-8",
    )

    bundle = (
        args.out_dir
        / "GEN_layer_replacement_efficiency_greedy_full_bundle.zip"
    )

    with zipfile.ZipFile(
        bundle,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        for file in [
            raw_path,
            recovered_best,
            recovered_last,
            adapter_copy,
            summary_path,
            report_table,
            args.out_dir / "selected_layers.csv",
            args.out_dir
            / "ranked_around50_candidate_plans.csv",
            results_csv,
            readme_path,
        ]:
            if file.is_file():
                archive.write(
                    file,
                    arcname=file.name,
                )

    print("\n===== COMPLETE =====")
    print(json.dumps(summary, indent=2))
    print("\nBundle:", bundle)


if __name__ == "__main__":
    main()
