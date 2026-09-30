#!/usr/bin/env python3
"""
PURE GLOBAL-L1 SELECTION RULE
- L1 score = sum(abs(weights)) for each output channel of each supported group representative root.
- Select the globally smallest current L1 channel.
- Prune its complete Torch-Pruning dependency group.
- Stop at the step closest to the requested whole-model parameter reduction target (default 56.81%).

The script:
1. Global-L1 search from the trained ACDC baseline.
2. Exact replay on a fresh baseline to create the raw model.
3. Raw validation.
4. 20-epoch exact-structure recovery.
5. Best-model validation.
6. CSV/JSON evidence and final PT files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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
        raise RuntimeError("Model output did not contain tensors for DepGraph trace.")
    return tuple(tensors)


def build_dependency_graph(
    model: nn.Module,
    imgsz: int,
    device: torch.device,
) -> tp.DependencyGraph:
    # This exact trace style is inherited from the previously working Stage-5
    # Global-L1 implementation, but no Stage-5 pruning-scope restrictions are
    # used by this pure algorithm.
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    model.zero_grad(set_to_none=True)

    example = torch.zeros(
        1,
        3,
        imgsz,
        imgsz,
        device=device,
        requires_grad=True,
    )

    kwargs: dict[str, Any] = {"example_inputs": example}
    signature = inspect.signature(tp.DependencyGraph.build_dependency)
    if "output_transform" not in signature.parameters:
        raise RuntimeError(
            "Installed Torch-Pruning does not expose output_transform; "
            "this YOLO26 trace requires it."
        )
    kwargs["output_transform"] = all_tensor_output_transform

    with torch.enable_grad():
        return tp.DependencyGraph().build_dependency(model, **kwargs)


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


def final_detect_output_conv_names(model: nn.Module) -> set[str]:
    """Find fixed final prediction Conv2d layers in the Detect head.

    Internal Detect-head convolutions remain eligible for Global-L1. Only the
    final class/regression projection convs are protected because changing
    their output width would alter the object-detection task definition.
    """
    head = model.model[-1]
    names_by_id = {id(module): name for name, module in model.named_modules()}
    protected: set[str] = set()

    for attribute in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branches = getattr(head, attribute, None)
        if branches is None:
            continue
        try:
            iterable = list(branches)
        except TypeError:
            continue

        for branch in iterable:
            convs = [module for module in branch.modules() if isinstance(module, nn.Conv2d)]
            if not convs:
                continue
            name = names_by_id.get(id(convs[-1]))
            if name:
                protected.add(name)

    if protected:
        return protected

    # Defensive architecture-preservation fallback: if this YOLO26 Detect
    # implementation uses different branch attribute names, protect the head's
    # Conv2d outputs rather than risk changing nc/regression dimensions.
    head_ids = {
        id(module)
        for module in head.modules()
        if isinstance(module, nn.Conv2d)
    }
    return {
        name
        for name, module in model.named_modules()
        if id(module) in head_ids
    }


def is_depthwise_conv2d(module: nn.Module) -> bool:
    """Return True for YOLO26's channel-wise DWConv Conv2d."""
    return (
        isinstance(module, nn.Conv2d)
        and int(module.groups) == int(module.in_channels)
        and int(module.in_channels) == int(module.out_channels)
        and int(module.groups) > 1
    )


def root_pruning_handler(module: nn.Conv2d):
    """Use the structurally correct Torch-Pruning output-channel handler."""
    if is_depthwise_conv2d(module):
        return tp.prune_depthwise_conv_out_channels
    if int(module.groups) == 1:
        return tp.prune_conv_out_channels
    raise RuntimeError(
        "Non-depthwise grouped Conv2d cannot be pruned one output channel "
        f"at a time safely: in={module.in_channels}, out={module.out_channels}, "
        f"groups={module.groups}"
    )


def detached_one2one_depthwise_root_names(model: nn.Module) -> set[str]:
    """Find detached one2one first-DWConv modules that are dependency-only.

    YOLO26 computes the one2one head from x.detach(). That severs the graph
    from the first one2one classification DWConv back to the shared neck
    feature. Because a depthwise output prune is also an input prune, such a
    module is not a valid independent pruning root. It remains prunable when
    an upstream shared feature channel is selected and mirrored into the
    duplicated Detect branches.
    """
    head = model.model[-1]
    branches = getattr(head, "one2one_cv3", None)
    if branches is None:
        return set()

    names_by_id = {id(module): name for name, module in model.named_modules()}
    excluded: set[str] = set()
    try:
        iterable = list(branches)
    except TypeError:
        return excluded

    for branch in iterable:
        conv = first_conv2d(branch)
        if conv is not None and is_depthwise_conv2d(conv):
            name = names_by_id.get(id(conv))
            if name:
                excluded.add(name)
    return excluded


def discover_global_l1_roots(
    model: nn.Module,
    protected_output_convs: set[str],
) -> tuple[list[str], list[dict[str, str]]]:
    """Discover roots using structural validity only, not pruning heuristics."""
    detached_dw = detached_one2one_depthwise_root_names(model)
    roots: list[str] = []
    excluded: list[dict[str, str]] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue
        if name in protected_output_convs:
            excluded.append({"module": name, "reason": "fixed Detect prediction output"})
            continue
        if module.out_channels <= 1:
            excluded.append({"module": name, "reason": "would reach zero output channels"})
            continue
        if name in detached_dw:
            excluded.append({
                "module": name,
                "reason": "one2one x.detach() makes first DWConv dependency-only, not an independent root",
            })
            continue
        if int(module.groups) > 1 and not is_depthwise_conv2d(module):
            excluded.append({
                "module": name,
                "reason": "non-depthwise grouped Conv2d is not safely one-channel-prunable",
            })
            continue
        roots.append(name)

    if not roots:
        raise RuntimeError("No structurally eligible Conv2d roots were discovered.")
    return roots, excluded


def pure_l1_candidates(model: nn.Module, roots: list[str]) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    candidates: list[dict[str, Any]] = []

    for root_name in roots:
        module = modules.get(root_name)
        if not isinstance(module, nn.Conv2d):
            continue
        if module.out_channels <= 1:
            continue

        weight = module.weight.detach().float()
        scores = weight.abs().sum(
            dim=tuple(range(1, weight.ndim))
        ).cpu()

        for channel_index, score in enumerate(scores.tolist()):
            candidates.append(
                {
                    "root_module": root_name,
                    "current_channel_index": int(channel_index),
                    "l1_score": float(score),
                    "current_root_out_channels": int(module.out_channels),
                }
            )

    candidates.sort(
        key=lambda row: (
            row["l1_score"],
            row["root_module"],
            row["current_channel_index"],
        )
    )
    return candidates


def first_conv2d(module: nn.Module) -> nn.Conv2d | None:
    """Return the first Conv2d executed inside a Detect branch."""
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            return child
    return None


def detect_shared_input_consumers(
    model: nn.Module,
) -> dict[int, list[tuple[str, nn.Conv2d]]]:
    """Map each Detect feature scale to all first Conv2d consumers.

    YOLO26 has one-to-many and one-to-one Detect branches that consume the same
    backbone/neck feature tensors. Torch-Pruning can miss one side of this
    duplicated consumer relationship. The mapping is used only to mirror an
    already-selected dependency input-channel deletion across every consumer of
    the same feature tensor; it does not change Global-L1 ranking or eligibility.
    """
    head = model.model[-1]
    groups: dict[int, list[tuple[str, nn.Conv2d]]] = {}

    for attribute in ("cv2", "cv3", "one2one_cv2", "one2one_cv3"):
        branches = getattr(head, attribute, None)
        if branches is None:
            continue
        try:
            branch_list = list(branches)
        except TypeError:
            continue

        for scale_index, branch in enumerate(branch_list):
            conv = first_conv2d(branch)
            if conv is None:
                continue
            groups.setdefault(scale_index, []).append((attribute, conv))

    return groups


def get_detect_branch_module(
    model: nn.Module,
    branch_name: str,
    scale_index: int,
) -> nn.Module:
    head = model.model[-1]
    branches = getattr(head, branch_name, None)
    if branches is None:
        raise RuntimeError(f"Missing Detect branch collection: {branch_name}")
    try:
        return list(branches)[scale_index]
    except (TypeError, IndexError) as exc:
        raise RuntimeError(
            f"Missing Detect branch: {branch_name}[{scale_index}]"
        ) from exc


def prune_detect_branch_input_channels(
    model: nn.Module,
    branch_name: str,
    scale_index: int,
    indices: list[int],
) -> dict[str, Any]:
    """Mirror one shared input-channel deletion into a Detect branch safely.

    Ordinary Conv2d consumers only need input slicing. YOLO26's class branch
    begins with DWConv, where input and output channels plus groups are coupled.
    Generic tp.prune_conv_in_channels() is invalid for that case and can create
    a [C, 0, k, k] weight. A branch-local DependencyGraph is used instead so
    the DWConv, BN and following pointwise Conv are updated together.
    """
    branch = get_detect_branch_module(model, branch_name, scale_index)
    first = first_conv2d(branch)
    if first is None:
        raise RuntimeError(f"No first Conv2d in {branch_name}[{scale_index}]")

    idxs = sorted(set(int(i) for i in indices))
    before_in = int(first.in_channels)
    bad = [i for i in idxs if i < 0 or i >= before_in]
    if bad:
        raise RuntimeError(
            f"Detect sync index out of range for {branch_name}[{scale_index}]: "
            f"in={before_in}, bad={bad}"
        )
    expected = before_in - len(idxs)
    if expected < 1:
        raise RuntimeError(
            f"Detect sync would remove all inputs from {branch_name}[{scale_index}]"
        )

    if is_depthwise_conv2d(first):
        was_training = branch.training
        branch.eval()
        example = torch.zeros(
            1, before_in, 8, 8,
            device=first.weight.device,
            dtype=first.weight.dtype,
            requires_grad=True,
        )
        with torch.enable_grad():
            graph = tp.DependencyGraph().build_dependency(
                branch,
                example_inputs=example,
            )
            group = graph.get_pruning_group(
                first,
                tp.prune_depthwise_conv_out_channels,
                idxs=idxs,
            )
            if not graph.check_pruning_group(group):
                raise RuntimeError(
                    f"Branch-local DepGraph rejected depthwise sync for "
                    f"{branch_name}[{scale_index}], idxs={idxs}"
                )
            group.prune()

        first_after = first_conv2d(branch)
        if first_after is None:
            raise RuntimeError("Depthwise sync removed first Conv2d unexpectedly.")
        if not is_depthwise_conv2d(first_after):
            raise RuntimeError(
                f"Depthwise structure broken in {branch_name}[{scale_index}]: "
                f"in={first_after.in_channels}, out={first_after.out_channels}, "
                f"groups={first_after.groups}"
            )
        if int(first_after.in_channels) != expected:
            raise RuntimeError(
                f"Depthwise sync width mismatch in {branch_name}[{scale_index}]: "
                f"{first_after.in_channels} != {expected}"
            )

        branch.eval()
        with torch.no_grad():
            branch(torch.zeros(
                1, expected, 8, 8,
                device=first_after.weight.device,
                dtype=first_after.weight.dtype,
            ))
        branch.train(was_training)
        return {
            "mode": "branch_local_depthwise_dependency_group",
            "before_in_channels": before_in,
            "after_in_channels": int(first_after.in_channels),
        }

    if int(first.groups) != 1:
        raise RuntimeError(
            f"Unsupported grouped first Detect consumer {branch_name}[{scale_index}]: "
            f"in={first.in_channels}, out={first.out_channels}, groups={first.groups}"
        )

    tp.prune_conv_in_channels(first, idxs=idxs)
    if int(first.in_channels) != expected:
        raise RuntimeError(
            f"Regular Detect sync width mismatch in {branch_name}[{scale_index}]: "
            f"{first.in_channels} != {expected}"
        )
    return {
        "mode": "ordinary_conv_input_slice",
        "before_in_channels": before_in,
        "after_in_channels": int(first.in_channels),
    }


def is_conv_input_pruning_handler(handler: Any) -> bool:
    """Recognize ordinary and depthwise Conv input pruning handlers."""
    if handler in (tp.prune_conv_in_channels, tp.prune_depthwise_conv_in_channels):
        return True
    name = getattr(handler, "__name__", str(handler))
    return "prune_in_channels" in name


def plan_detect_shared_input_sync(
    model: nn.Module,
    group: Any,
    names_by_id: dict[int, str],
) -> list[dict[str, Any]]:
    """Plan structural synchronization for duplicated YOLO26 Detect inputs.

    This is not a pruning heuristic. It mirrors the exact same dependency index
    into all first Detect-head consumers of one shared feature map when the
    DependencyGraph records that deletion for only a subset of the consumers.
    """
    consumers = detect_shared_input_consumers(model)
    module_to_scale: dict[int, int] = {}
    for scale_index, items in consumers.items():
        for _, module in items:
            module_to_scale[id(module)] = scale_index

    indices_by_scale: dict[int, set[int]] = {}
    for dependency, indices in group:
        module = dependency.target.module
        scale_index = module_to_scale.get(id(module))
        if scale_index is None:
            continue
        if not is_conv_input_pruning_handler(dependency.handler):
            continue
        indices_by_scale.setdefault(scale_index, set()).update(
            flatten_indices(indices)
        )

    plan: list[dict[str, Any]] = []
    for scale_index, raw_indices in sorted(indices_by_scale.items()):
        unique_indices = sorted(set(int(i) for i in raw_indices))
        if not unique_indices:
            continue

        before_consumers: list[dict[str, Any]] = []
        for branch_name, module in consumers.get(scale_index, []):
            module_name = names_by_id.get(id(module), "<unresolved>")
            before_in = int(module.in_channels)
            bad = [i for i in unique_indices if i < 0 or i >= before_in]
            if bad:
                raise RuntimeError(
                    f"Detect shared-input sync index out of range for "
                    f"{module_name}: in_channels={before_in}, bad={bad}"
                )
            before_consumers.append(
                {
                    "branch": branch_name,
                    "module_name": module_name,
                    "module_id": id(module),
                    "before_in_channels": before_in,
                }
            )

        plan.append(
            {
                "scale_index": scale_index,
                "indices": unique_indices,
                "consumers": before_consumers,
            }
        )

    return plan


def apply_detect_shared_input_sync(
    model: nn.Module,
    sync_plan: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Apply only missing shared-Detect dependency operations safely."""
    actions: list[dict[str, Any]] = []

    for item in sync_plan:
        scale_index = int(item["scale_index"])
        indices = [int(i) for i in item["indices"]]
        before_by_branch = {entry["branch"]: entry for entry in item["consumers"]}
        current_by_branch = dict(
            detect_shared_input_consumers(model).get(scale_index, [])
        )

        for branch, before in before_by_branch.items():
            module = current_by_branch.get(branch)
            if module is None:
                raise RuntimeError(
                    f"Detect branch disappeared: {branch}[{scale_index}]"
                )

            before_in = int(before["before_in_channels"])
            expected = before_in - len(indices)
            current_in = int(module.in_channels)
            sync_mode = "dependency_graph"

            if current_in == expected:
                status = "already_pruned_by_dependency_graph"
            elif current_in == before_in:
                details = prune_detect_branch_input_channels(
                    model, branch, scale_index, indices
                )
                refreshed = dict(
                    detect_shared_input_consumers(model).get(scale_index, [])
                )[branch]
                current_in = int(refreshed.in_channels)
                sync_mode = str(details["mode"])
                if current_in != expected:
                    raise RuntimeError(
                        f"Detect sync failed for {branch}[{scale_index}]: "
                        f"{current_in} != {expected}"
                    )
                status = "mirrored_missing_dependency"
            else:
                raise RuntimeError(
                    f"Unexpected Detect width for {branch}[{scale_index}]: "
                    f"before={before_in}, current={current_in}, expected={expected}"
                )

            actions.append(
                {
                    "module_name": before["module_name"],
                    "module_type": "Conv2d",
                    "handler": "YOLO26_shared_detect_input_sync",
                    "is_out_channel_prune": False,
                    "indices": ";".join(map(str, indices)),
                    "index_count": len(indices),
                    "detect_scale_index": scale_index,
                    "detect_branch": branch,
                    "sync_status": status,
                    "sync_mode": sync_mode,
                    "before_in_channels": before_in,
                    "after_in_channels": current_in,
                }
            )

    return actions


def verify_detect_shared_input_widths(model: nn.Module) -> None:
    """Verify all duplicated Detect branches accept the same feature width."""
    consumers = detect_shared_input_consumers(model)
    for scale_index, items in consumers.items():
        widths = {branch: int(module.in_channels) for branch, module in items}
        if len(set(widths.values())) > 1:
            raise RuntimeError(
                f"Detect shared-input width mismatch at scale {scale_index}: {widths}"
            )


def snapshot_detect_shared_inputs(model: nn.Module) -> dict[int, dict[str, dict[str, Any]]]:
    """Snapshot Detect consumer input widths and weights before one prune step.

    Torch-Pruning may apply an input-channel deletion to only some consumers of
    a shared YOLO26 feature tensor.  The weights are kept only for this one
    pruning step so the exact deleted input index can be inferred afterwards if
    DependencyGraph did not expose that consumer operation in advance.
    """
    snapshot: dict[int, dict[str, dict[str, Any]]] = {}
    for scale_index, items in detect_shared_input_consumers(model).items():
        scale_snapshot: dict[str, dict[str, Any]] = {}
        for branch, module in items:
            scale_snapshot[branch] = {
                "before_in_channels": int(module.in_channels),
                "before_weight": module.weight.detach().clone(),
            }
        snapshot[scale_index] = scale_snapshot
    return snapshot


def infer_removed_input_indices(
    before_weight: torch.Tensor,
    after_weight: torch.Tensor,
) -> list[int] | None:
    """Infer exact Conv2d input channels removed by structural slicing.

    Torch-Pruning deletes channels by index while preserving the order and
    values of surviving weights.  Therefore the post-prune input-channel axis
    must be an exact subsequence of the pre-prune axis.  This function does not
    score or choose a channel; it only recovers the dependency index that was
    already structurally deleted from a sibling Detect consumer.
    """
    if before_weight.ndim != 4 or after_weight.ndim != 4:
        return None
    if before_weight.shape[0] != after_weight.shape[0]:
        return None
    if before_weight.shape[2:] != after_weight.shape[2:]:
        return None

    before_in = int(before_weight.shape[1])
    after_in = int(after_weight.shape[1])
    removed_count = before_in - after_in
    if removed_count <= 0:
        return [] if removed_count == 0 else None

    removed: list[int] = []
    after_index = 0
    for before_index in range(before_in):
        if after_index < after_in and torch.equal(
            before_weight[:, before_index],
            after_weight[:, after_index],
        ):
            after_index += 1
        else:
            removed.append(before_index)

    if after_index != after_in or len(removed) != removed_count:
        return None
    return removed


def repair_detect_shared_inputs_from_snapshot(
    model: nn.Module,
    snapshot: dict[int, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Repair duplicated YOLO26 Detect consumers after one pruning step.

    The exact shared channel already removed from an ordinary sibling is
    inferred from its before/after weights. That same dependency is then
    mirrored to wider siblings. Depthwise first consumers are updated with a
    branch-local DependencyGraph, not generic Conv2d input slicing.
    """
    actions: list[dict[str, Any]] = []
    current = detect_shared_input_consumers(model)

    for scale_index, items in current.items():
        current_by_branch = dict(items)
        widths = {branch: int(module.in_channels) for branch, module in items}
        if len(set(widths.values())) <= 1:
            continue

        before_scale = snapshot.get(scale_index)
        if not before_scale:
            raise RuntimeError(
                f"No Detect snapshot for mismatched scale {scale_index}: {widths}"
            )

        before_widths = {
            branch: int(entry["before_in_channels"])
            for branch, entry in before_scale.items()
        }
        if len(set(before_widths.values())) != 1:
            raise RuntimeError(
                f"Detect inputs inconsistent before prune at scale {scale_index}: "
                f"{before_widths}"
            )

        target = min(widths.values())
        inferred_sets: list[tuple[int, ...]] = []

        for branch, module in current_by_branch.items():
            before = before_scale.get(branch)
            if before is None:
                continue
            before_in = int(before["before_in_channels"])
            if int(module.in_channels) >= before_in:
                continue
            if int(module.in_channels) != target:
                continue

            inferred = infer_removed_input_indices(
                before["before_weight"],
                module.weight.detach(),
            )
            if inferred:
                inferred_sets.append(tuple(int(i) for i in inferred))

        if not inferred_sets:
            raise RuntimeError(
                f"Could not infer shared Detect input index at scale {scale_index}: "
                f"before={before_widths}, after={widths}"
            )

        unique_sets = set(inferred_sets)
        if len(unique_sets) != 1:
            raise RuntimeError(
                f"Detect siblings imply different removed indices at scale "
                f"{scale_index}: {sorted(unique_sets)}"
            )
        indices = list(next(iter(unique_sets)))

        for branch, module in list(current_by_branch.items()):
            if int(module.in_channels) == target:
                continue

            before = before_scale.get(branch)
            if before is None:
                raise RuntimeError(
                    f"Missing pre-prune snapshot for {branch}[{scale_index}]"
                )
            before_in = int(before["before_in_channels"])
            if int(module.in_channels) != before_in:
                raise RuntimeError(
                    f"Unexpected partially-pruned {branch}[{scale_index}]: "
                    f"before={before_in}, current={module.in_channels}, target={target}"
                )

            details = prune_detect_branch_input_channels(
                model, branch, scale_index, indices
            )
            refreshed = dict(
                detect_shared_input_consumers(model).get(scale_index, [])
            )[branch]
            after_in = int(refreshed.in_channels)
            if after_in != target:
                raise RuntimeError(
                    f"Detect repair failed for {branch}[{scale_index}]: "
                    f"{after_in} != {target}"
                )

            actions.append(
                {
                    "module_name": f"Detect.{branch}[{scale_index}].first_conv",
                    "module_type": "Conv2d",
                    "handler": "YOLO26_post_prune_dependency_repair",
                    "is_out_channel_prune": False,
                    "indices": ";".join(map(str, indices)),
                    "index_count": len(indices),
                    "detect_scale_index": scale_index,
                    "detect_branch": branch,
                    "sync_status": "mirrored_from_auto_pruned_sibling",
                    "sync_mode": details["mode"],
                    "before_in_channels": before_in,
                    "after_in_channels": after_in,
                }
            )

    verify_detect_shared_input_widths(model)
    return actions


def group_is_structurally_valid(
    graph: tp.DependencyGraph,
    group: Any,
    names_by_id: dict[int, str],
    protected_output_convs: set[str],
) -> tuple[bool, str, list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []

    for dependency, indices in group:
        module = dependency.target.module
        module_name = names_by_id.get(id(module), "<unresolved>")
        unique_indices = sorted(set(flatten_indices(indices)))
        is_out_prune = graph.is_out_channel_pruning_fn(dependency.handler)

        operation = {
            "module_name": module_name,
            "module_type": module.__class__.__name__,
            "handler": getattr(dependency.handler, "__name__", str(dependency.handler)),
            "is_out_channel_prune": bool(is_out_prune),
            "indices": ";".join(map(str, unique_indices)),
            "index_count": len(unique_indices),
        }

        if is_out_prune and isinstance(module, nn.Conv2d):
            projected = int(module.out_channels) - len(unique_indices)
            operation["current_out_channels"] = int(module.out_channels)
            operation["projected_out_channels"] = projected

            if module_name in protected_output_convs:
                return (
                    False,
                    f"would change fixed Detect output channels: {module_name}",
                    operations + [operation],
                )

            # This is not a pruning-ratio/cap heuristic. Zero-channel modules
            # are not a valid neural-network architecture.
            if projected < 1:
                return (
                    False,
                    f"would reduce Conv2d {module_name} to zero channels",
                    operations + [operation],
                )

        operations.append(operation)

    if not graph.check_pruning_group(group):
        return False, "DependencyGraph rejected pruning group", operations

    return True, "", operations


def verify_conv_structures(model: nn.Module) -> None:
    """Verify Conv2d metadata matches its weight tensor, including groups."""
    for name, module in model.named_modules():
        if not isinstance(module, nn.Conv2d):
            continue

        in_channels = int(module.in_channels)
        out_channels = int(module.out_channels)
        groups = int(module.groups)

        if min(in_channels, out_channels, groups) < 1:
            raise RuntimeError(
                f"Invalid Conv2d dimensions at {name}: "
                f"in={in_channels}, out={out_channels}, groups={groups}"
            )
        if in_channels % groups != 0 or out_channels % groups != 0:
            raise RuntimeError(
                f"Grouped Conv2d divisibility failure at {name}: "
                f"in={in_channels}, out={out_channels}, groups={groups}"
            )

        expected = (out_channels, in_channels // groups)
        actual = tuple(int(x) for x in module.weight.shape[:2])
        if actual != expected:
            raise RuntimeError(
                f"Conv2d weight metadata mismatch at {name}: "
                f"weight[:2]={actual}, expected={expected}, "
                f"in={in_channels}, out={out_channels}, groups={groups}"
            )


def forward_check(model: nn.Module, imgsz: int, device: torch.device) -> None:
    verify_detect_shared_input_widths(model)
    verify_conv_structures(model)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    tensors = flatten_tensors(output)
    if not tensors:
        raise RuntimeError("Forward verification produced no tensors.")
    if not all(bool(torch.isfinite(tensor).all().item()) for tensor in tensors):
        raise RuntimeError("Forward verification produced non-finite tensors.")


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

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)



T4_GENERIC_SCOPE_DEFAULT = Path(
    "/home/afm176/yolo_project/5_reproduction/reproducing_files_for_hoyin/"
    "tables/T4_25pct_group_ranking.csv"
)
PROTECTED_MANIFEST_DEFAULT = Path(
    "/home/afm176/yolo_project/5_reproduction/reproducing_files_for_hoyin/"
    "group_definitions/protected_root_manifest.csv"
)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dependency_safe_scope(
    model: nn.Module,
    t4_path: Path,
    protected_manifest_path: Path,
) -> tuple[set[str], dict[str, str], set[str], list[dict[str, str]]]:
    if not t4_path.is_file():
        raise RuntimeError(f"T4 dependency-group table not found: {t4_path}")
    if not protected_manifest_path.is_file():
        raise RuntimeError(
            f"Protected-root manifest not found: {protected_manifest_path}"
        )

    t4_rows = read_csv_rows(t4_path)
    generic_rows = [row for row in t4_rows if row.get("group_kind") == "GENERIC"]
    custom_rows = [row for row in t4_rows if row.get("group_kind") == "CUSTOM"]
    if len(generic_rows) != 42 or len(custom_rows) != 9:
        raise RuntimeError(
            "Unexpected dependency-group scope: "
            f"GENERIC={len(generic_rows)}, CUSTOM={len(custom_rows)}; expected 42/9."
        )

    protected_rows = read_csv_rows(protected_manifest_path)
    allowed_roots = {row["representative_root"] for row in generic_rows}
    group_id_by_root = {
        row["representative_root"]: row["group_id"] for row in generic_rows
    }
    protected_roots = {row["module_path"] for row in protected_rows}

    modules = dict(model.named_modules())
    missing = sorted(allowed_roots - set(modules))
    if missing:
        raise RuntimeError(
            "The ACDC YOLO26n architecture does not match the validated "
            f"dependency-group scope. Missing roots: {missing}"
        )
    wrong_type = sorted(
        name for name in allowed_roots if not isinstance(modules[name], nn.Conv2d)
    )
    if wrong_type:
        raise RuntimeError(f"Validated roots are not Conv2d: {wrong_type}")

    scope_rows = []
    for row in generic_rows:
        name = row["representative_root"]
        module = modules[name]
        scope_rows.append({
            "group_id": row["group_id"],
            "group_kind": row["group_kind"],
            "representative_root": name,
            "initial_out_channels": int(module.out_channels),
            "selection_rule": "raw output-filter L1, globally ranked",
            "per_root_pruning_cap": "NONE",
            "minimum_channels_rule": "only structural >=1 output channel",
        })
    return allowed_roots, group_id_by_root, protected_roots, scope_rows


def safe_scope_candidates(
    model: nn.Module,
    allowed_roots: set[str],
    group_id_by_root: dict[str, str],
) -> list[dict[str, Any]]:
    modules = dict(model.named_modules())
    candidates: list[dict[str, Any]] = []
    for root_name in sorted(allowed_roots):
        module = modules.get(root_name)
        if not isinstance(module, nn.Conv2d) or int(module.out_channels) <= 1:
            continue
        weight = module.weight.detach().float()
        scores = weight.abs().sum(dim=tuple(range(1, weight.ndim))).cpu()
        for channel_index, score in enumerate(scores.tolist()):
            candidates.append({
                "group_id": group_id_by_root[root_name],
                "root_module": root_name,
                "current_channel_index": int(channel_index),
                "l1_score": float(score),
                "current_root_out_channels": int(module.out_channels),
            })
    candidates.sort(
        key=lambda row: (
            row["l1_score"],
            row["group_id"],
            row["current_channel_index"],
        )
    )
    return candidates


def safe_scope_group_check(
    graph: tp.DependencyGraph,
    group: Any,
    names_by_id: dict[int, str],
    protected_roots: set[str],
) -> tuple[bool, str, list[dict[str, Any]]]:
    operations: list[dict[str, Any]] = []
    for dependency, indices in group:
        module = dependency.target.module
        module_name = names_by_id.get(id(module), "<unresolved>")
        unique_indices = sorted(set(flatten_indices(indices)))
        is_out_prune = graph.is_out_channel_pruning_fn(dependency.handler)
        operation = {
            "module_name": module_name,
            "module_type": module.__class__.__name__,
            "handler": getattr(dependency.handler, "__name__", str(dependency.handler)),
            "is_out_channel_prune": bool(is_out_prune),
            "indices": ";".join(map(str, unique_indices)),
            "index_count": len(unique_indices),
        }
        if is_out_prune and isinstance(module, nn.Conv2d):
            projected = int(module.out_channels) - len(unique_indices)
            operation["current_out_channels"] = int(module.out_channels)
            operation["projected_out_channels"] = projected
            if module_name in protected_roots:
                return False, f"would prune protected output root {module_name}", operations + [operation]
            if projected < 1:
                return False, f"would reduce Conv2d {module_name} to zero channels", operations + [operation]
        operations.append(operation)
    if not graph.check_pruning_group(group):
        return False, "DependencyGraph rejected pruning group", operations
    return True, "", operations

def search_plan(
    checkpoint: Path,
    target_percent: float,
    imgsz: int,
    device: torch.device,
    output_dir: Path,
    max_steps: int,
    t4_path: Path,
    protected_manifest_path: Path,
) -> dict[str, Any]:
    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(device).float().eval()
    baseline_params = count_parameters(model)
    target_params = int(round(baseline_params * (1.0 - target_percent / 100.0)))

    allowed_roots, group_id_by_root, protected_roots, scope_rows = (
        load_dependency_safe_scope(model, t4_path, protected_manifest_path)
    )
    write_csv(output_dir / "dependency_safe_generic_scope.csv", scope_rows)
    (output_dir / "eligible_roots.txt").write_text(
        "\n".join(sorted(allowed_roots)) + "\n", encoding="utf-8"
    )
    (output_dir / "protected_roots.txt").write_text(
        "\n".join(sorted(protected_roots)) + "\n", encoding="utf-8"
    )

    # Baseline forward before any pruning.
    model.eval()
    with torch.no_grad():
        if not flatten_tensors(model(torch.zeros(1, 3, imgsz, imgsz, device=device))):
            raise RuntimeError("Baseline forward produced no tensors.")

    step_rows: list[dict[str, Any]] = []
    operation_rows: list[dict[str, Any]] = []
    rejection_rows: list[dict[str, Any]] = []
    chosen_step_count: int | None = None

    for step in range(1, max_steps + 1):
        step_started = time.time()
        graph = build_dependency_graph(model, imgsz, device)
        modules = dict(model.named_modules())
        names_by_id = {id(module): name for name, module in model.named_modules()}
        candidates = safe_scope_candidates(model, allowed_roots, group_id_by_root)
        if not candidates:
            break

        selected = None
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
                valid, reason, operations = safe_scope_group_check(
                    graph, group, names_by_id, protected_roots
                )
            except Exception as exc:
                valid = False
                reason = repr(exc)
                operations = []
                group = None

            if valid:
                selected = dict(candidate)
                selected["global_candidate_rank_this_step"] = global_rank
                selected_group = group
                selected_operations = operations
                break

            rejection_rows.append({
                "step": step,
                "global_candidate_rank_this_step": global_rank,
                **candidate,
                "rejection_reason": reason,
            })

        if selected is None or selected_group is None:
            break

        before_params = count_parameters(model)
        before_root_channels = int(modules[selected["root_module"]].out_channels)
        selected_group.prune()
        after_params = count_parameters(model)
        modules_after = dict(model.named_modules())
        after_root_channels = int(modules_after[selected["root_module"]].out_channels)

        if after_params >= before_params:
            raise RuntimeError(f"Step {step}: pruning did not reduce parameter count.")

        # The previously validated generic groups are expected to preserve a full
        # YOLO26 forward. Check every step because there is no per-root safety cap.
        model.eval()
        with torch.no_grad():
            output = model(torch.zeros(1, 3, imgsz, imgsz, device=device))
        if not flatten_tensors(output):
            raise RuntimeError(f"Step {step}: forward produced no tensors.")

        reduction = 100.0 * (baseline_params - after_params) / baseline_params
        row = {
            "step": step,
            "group_id": selected["group_id"],
            "root_module": selected["root_module"],
            "current_channel_index_pruned": selected["current_channel_index"],
            "l1_score_at_selection": selected["l1_score"],
            "global_candidate_rank_this_step": selected["global_candidate_rank_this_step"],
            "root_out_channels_before": before_root_channels,
            "root_out_channels_after": after_root_channels,
            "parameters_before": before_params,
            "parameters_after": after_params,
            "parameters_removed_this_step": before_params - after_params,
            "cumulative_parameter_reduction_percent": reduction,
            "target_parameters": target_params,
            "target_error_after": after_params - target_params,
            "absolute_target_error_after": abs(after_params - target_params),
            "forward_pass_ok": True,
            "step_elapsed_seconds": time.time() - step_started,
        }
        step_rows.append(row)

        for operation_index, operation in enumerate(selected_operations, start=1):
            operation_rows.append({
                "step": step,
                "operation_index": operation_index,
                "selected_group_id": selected["group_id"],
                "selected_root_module": selected["root_module"],
                **operation,
            })

        write_csv(output_dir / "global_l1_selection_log.csv", step_rows)
        write_csv(output_dir / "dependency_operations.csv", operation_rows)
        write_csv(output_dir / "rejected_candidates.csv", rejection_rows)

        print(
            f"step={step:04d} group={selected['group_id']} "
            f"root={selected['root_module']} idx={selected['current_channel_index']} "
            f"L1={selected['l1_score']:.8g} params={after_params:,} "
            f"reduction={reduction:.4f}% elapsed={row['step_elapsed_seconds']:.2f}s",
            flush=True,
        )

        if after_params <= target_params:
            previous_error = abs(before_params - target_params)
            current_error = abs(after_params - target_params)
            chosen_step_count = step if current_error <= previous_error else step - 1
            break

    if not step_rows:
        raise RuntimeError("Global-L1 search could not prune any valid generic dependency group.")

    if chosen_step_count is None:
        closest = min(step_rows, key=lambda item: item["absolute_target_error_after"])
        chosen_step_count = int(closest["step"])

    chosen_plan = [row for row in step_rows if int(row["step"]) <= chosen_step_count]
    write_csv(output_dir / "chosen_global_l1_plan.csv", chosen_plan)
    chosen_params = baseline_params if chosen_step_count == 0 else int(chosen_plan[-1]["parameters_after"])

    summary = {
        "baseline_parameters": baseline_params,
        "target_reduction_percent": target_percent,
        "target_parameters": target_params,
        "search_steps_executed": len(step_rows),
        "chosen_steps": chosen_step_count,
        "chosen_parameters": chosen_params,
        "chosen_parameter_reduction_percent": 100.0 * (baseline_params - chosen_params) / baseline_params,
        "generic_dependency_groups_included": len(allowed_roots),
        "custom_dependency_groups_not_supported": 9,
        "per_root_pruning_cap": None,
        "minimum_four_channels_rule": False,
        "selection_uses_accuracy_or_sensitivity": False,
        "scope_table": str(t4_path.resolve()),
        "protected_manifest": str(protected_manifest_path.resolve()),
        "algorithm": (
            "raw current output-filter L1 ranked globally across all 42 supported "
            "generic dependency-group representative roots; one dependency group "
            "pruned per step; no per-root channel cap and no minimum-4 rule"
        ),
    }
    (output_dir / "search_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def replay_plan(
    checkpoint: Path,
    plan_path: Path,
    expected_parameters: int,
    imgsz: int,
    device: torch.device,
    output_path: Path,
    t4_path: Path,
    protected_manifest_path: Path,
) -> dict[str, Any]:
    with plan_path.open(newline="", encoding="utf-8") as handle:
        plan = list(csv.DictReader(handle))
    if not plan:
        raise RuntimeError("Chosen Global-L1 replay plan is empty.")

    yolo = YOLO(str(checkpoint))
    model = yolo.model.to(device).float().eval()
    allowed_roots, _, protected_roots, _ = load_dependency_safe_scope(
        model, t4_path, protected_manifest_path
    )

    for row_number, row in enumerate(plan, start=1):
        graph = build_dependency_graph(model, imgsz, device)
        modules = dict(model.named_modules())
        names_by_id = {id(module): name for name, module in model.named_modules()}
        root_name = row["root_module"]
        channel_index = int(row["current_channel_index_pruned"])
        if root_name not in allowed_roots:
            raise RuntimeError(f"Replay selected root outside validated generic scope: {root_name}")
        module = modules.get(root_name)
        if not isinstance(module, nn.Conv2d):
            raise RuntimeError(f"Replay root missing/wrong type: {root_name}")

        scores = module.weight.detach().float().abs().sum(
            dim=tuple(range(1, module.weight.ndim))
        ).cpu()
        current_score = float(scores[channel_index].item())
        recorded_score = float(row["l1_score_at_selection"])
        if not math.isclose(current_score, recorded_score, rel_tol=1e-5, abs_tol=1e-6):
            raise RuntimeError(
                f"Replay L1 mismatch at step {row_number}, root={root_name}, "
                f"idx={channel_index}: current={current_score}, recorded={recorded_score}"
            )

        group = graph.get_pruning_group(
            module, tp.prune_conv_out_channels, idxs=[channel_index]
        )
        valid, reason, _ = safe_scope_group_check(
            graph, group, names_by_id, protected_roots
        )
        if not valid:
            raise RuntimeError(f"Replay group invalid at step {row_number}: {reason}")
        group.prune()

        expected_after = int(row["parameters_after"])
        actual_after = count_parameters(model)
        if actual_after != expected_after:
            raise RuntimeError(
                f"Replay parameter mismatch at step {row_number}: "
                f"{actual_after} != {expected_after}"
            )

    final_params = count_parameters(model)
    if final_params != expected_parameters:
        raise RuntimeError(
            f"Replay final parameter mismatch: {final_params} != {expected_parameters}"
        )

    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, imgsz, imgsz, device=device))
    if not flatten_tensors(output):
        raise RuntimeError("Final replay forward produced no tensors.")

    yolo.save(str(output_path))
    reloaded = YOLO(str(output_path))
    reload_params = count_parameters(reloaded.model)
    if reload_params != final_params:
        raise RuntimeError("Saved raw checkpoint parameter count changed on reload.")

    return {
        "raw_model": str(output_path.resolve()),
        "raw_parameters": final_params,
        "sha256": sha256_file(output_path),
    }


def metric_value(metrics: Any, path: str) -> float | None:
    value = metrics
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    try:
        return float(value)
    except Exception:
        return None


def validate_model(
    model_path: Path,
    data_yaml: Path,
    baseline_params: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
) -> dict[str, Any]:
    model = YOLO(str(model_path))
    params = count_parameters(model.model)

    metrics = model.val(
        data=str(data_yaml.resolve()),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        half=False,
        rect=True,
        conf=0.001,
        iou=0.70,
        max_det=300,
        augment=False,
        plots=False,
        save_json=False,
        verbose=True,
        seed=42,
    )

    speed = getattr(metrics, "speed", {}) or {}
    preprocess = float(speed.get("preprocess", 0.0) or 0.0)
    inference = float(speed.get("inference", 0.0) or 0.0)
    postprocess = float(speed.get("postprocess", 0.0) or 0.0)
    avg_latency = preprocess + inference + postprocess

    return {
        "model_path": str(model_path.resolve()),
        "parameters": params,
        "param_reduction_percent": (
            100.0 * (baseline_params - params) / baseline_params
        ),
        "map50_95": metric_value(metrics, "box.map"),
        "map50": metric_value(metrics, "box.map50"),
        "map75": metric_value(metrics, "box.map75"),
        "precision": metric_value(metrics, "box.mp"),
        "recall": metric_value(metrics, "box.mr"),
        "preprocess_ms_per_image": preprocess,
        "inference_ms_per_image": inference,
        "postprocess_ms_per_image": postprocess,
        "avg_latency_ms": avg_latency,
        "fps": (1000.0 / avg_latency) if avg_latency > 0 else None,
    }


def write_single_row_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


class ExactStructureTrainer(DetectionTrainer):
    external_model: nn.Module | None = None

    def get_model(self, cfg=None, weights=None, verbose=True) -> nn.Module:
        model = type(self).external_model
        if model is None:
            raise RuntimeError("Exact pruned model object was not supplied.")
        type(self).external_model = None
        model = model.float()
        for parameter in model.parameters():
            parameter.requires_grad_(True)
        return model


def recovery_train(
    raw_model: Path,
    data_yaml: Path,
    output_dir: Path,
    expected_params: int,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    seed: int,
    device: str,
) -> tuple[Path, Path]:
    source = YOLO(str(raw_model))
    if count_parameters(source.model) != expected_params:
        raise RuntimeError("Raw model parameter count changed before recovery.")

    ExactStructureTrainer.external_model = source.model

    overrides = {
        "model": str(raw_model.resolve()),
        "data": str(data_yaml.resolve()),
        "task": "detect",
        "mode": "train",
        "epochs": epochs,
        "batch": batch,
        "imgsz": imgsz,
        "device": device,
        "workers": workers,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_epochs": 1.0,
        "pretrained": False,
        "amp": True,
        "deterministic": True,
        "seed": seed,
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
        "project": str(output_dir),
        "name": "recovery_train",
        "exist_ok": True,
        "patience": 100,
    }

    trainer = ExactStructureTrainer(overrides=overrides)
    trainer.train()

    best = Path(trainer.best)
    last = Path(trainer.last)
    if not best.is_file() or not last.is_file():
        raise RuntimeError("Recovery training did not produce best.pt/last.pt.")

    best_copy = output_dir / "ACDC_pure_GlobalL1_56p81_recovered_best.pt"
    last_copy = output_dir / "ACDC_pure_GlobalL1_56p81_recovered_last.pt"
    shutil.copy2(best, best_copy)
    shutil.copy2(last, last_copy)

    args_path = output_dir / "recovery_train/args.yaml"
    results_path = output_dir / "recovery_train/results.csv"
    if args_path.is_file():
        shutil.copy2(args_path, output_dir / "recovery_args.yaml")
    if results_path.is_file():
        shutil.copy2(results_path, output_dir / "recovery_training_results.csv")

    return best_copy, last_copy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-percent", type=float, default=56.81)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-search-steps", type=int, default=5000)
    parser.add_argument("--t4", type=Path, default=T4_GENERIC_SCOPE_DEFAULT)
    parser.add_argument(
        "--protected-manifest",
        type=Path,
        default=PROTECTED_MANIFEST_DEFAULT,
    )
    args = parser.parse_args()

    if not args.baseline_model.is_file():
        raise SystemExit(f"Baseline model not found: {args.baseline_model}")
    if not args.data.is_file():
        raise SystemExit(f"Dataset YAML not found: {args.data}")
    if not args.t4.is_file():
        raise SystemExit(f"Dependency-group table not found: {args.t4}")
    if not args.protected_manifest.is_file():
        raise SystemExit(f"Protected-root manifest not found: {args.protected_manifest}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)

    device = torch.device(
        "cuda:0" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )

    baseline_probe = YOLO(str(args.baseline_model))
    baseline_params = count_parameters(baseline_probe.model)
    del baseline_probe

    print("===== PURE GLOBAL-L1 SEARCH =====")
    search_dir = args.output_dir / "search"
    search_dir.mkdir(parents=True, exist_ok=True)

    search_summary = search_plan(
        args.baseline_model.resolve(),
        args.target_percent,
        args.imgsz,
        device,
        search_dir,
        args.max_search_steps,
        args.t4.resolve(),
        args.protected_manifest.resolve(),
    )

    chosen_plan = search_dir / "chosen_global_l1_plan.csv"
    raw_model = args.output_dir / "ACDC_pure_GlobalL1_56p81_raw.pt"

    print("===== REPLAY + RAW MODEL =====")
    replay_summary = replay_plan(
        args.baseline_model.resolve(),
        chosen_plan,
        int(search_summary["chosen_parameters"]),
        args.imgsz,
        device,
        raw_model,
        args.t4.resolve(),
        args.protected_manifest.resolve(),
    )

    raw_row = validate_model(
        raw_model,
        args.data.resolve(),
        baseline_params,
        args.imgsz,
        args.batch,
        args.workers,
        args.device,
    )
    raw_row = {"stage": "raw", **raw_row}
    write_single_row_csv(args.output_dir / "raw_validation.csv", raw_row)

    print("===== 20-EPOCH RECOVERY =====")
    best_model, last_model = recovery_train(
        raw_model,
        args.data.resolve(),
        args.output_dir,
        int(search_summary["chosen_parameters"]),
        args.epochs,
        args.imgsz,
        args.batch,
        args.workers,
        args.seed,
        args.device,
    )

    best_row = validate_model(
        best_model,
        args.data.resolve(),
        baseline_params,
        args.imgsz,
        args.batch,
        args.workers,
        args.device,
    )
    best_row = {"stage": "recovered_best", **best_row}
    write_single_row_csv(args.output_dir / "best_validation.csv", best_row)

    with (args.output_dir / "summary_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(raw_row)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(raw_row)
        writer.writerow(best_row)

    final_summary = {
        "status": "PASSED_ACDC_PURE_GLOBAL_L1_56P81",
        "baseline_model": str(args.baseline_model.resolve()),
        "baseline_parameters": baseline_params,
        "target_parameter_reduction_percent": args.target_percent,
        "actual_parameter_reduction_percent": search_summary[
            "chosen_parameter_reduction_percent"
        ],
        "chosen_parameters": search_summary["chosen_parameters"],
        "raw_model": str(raw_model.resolve()),
        "recovered_best_model": str(best_model.resolve()),
        "recovered_last_model": str(last_model.resolve()),
        "search_summary": search_summary,
        "replay_summary": replay_summary,
        "raw_validation": raw_row,
        "best_validation": best_row,
        "algorithm_constraints": {
            "per_root_channel_cap": None,
            "minimum_four_channels_rule": False,
            "dependency_group_scope": "42 validated GENERIC representative roots",
            "custom_groups_excluded": 9,
            "t4_42_root_scope": True,
            "scope_reason": "structural dependency-group representation, not a pruning quota",
            "sensitivity_weighting": False,
            "accuracy_used_for_channel_selection": False,
            "yolo26_detect_dependency_sync_changes_ranking": False,
            "only_structural_constraints": [
                "do not prune a module to zero output channels",
                "do not change fixed final Detect output dimensions",
                "Torch-Pruning dependency group must be valid",
                "use only prevalidated generic dependency-group representative roots",
                "exclude unsupported custom C3k2/C2PSA groups from the generic pruning core",
            ],
        },
    }
    (args.output_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
