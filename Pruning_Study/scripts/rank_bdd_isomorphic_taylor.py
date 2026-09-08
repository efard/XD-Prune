"""Create a controlled Isomorphic-Taylor ranking for BDD YOLO26n pruning.

This is an adaptation of Fang et al.'s *Isomorphic Pruning for Vision Models*
(ECCV 2024) to the project's already validated YOLO26n structural groups.
It uses the paper's first-order Taylor importance and the official
Torch-Pruning isomorphic topology signature.  The original implementation
targets ImageNet CNNs/transformers; YOLO26n's custom C3k2/C2PSA units are
therefore represented by their validated coupled logical units rather than
silently applying an unsupported generic Conv2d operation.

The script performs ranking only.  It uses each domain's frozen training split
for loss gradients, never validation/test data, and makes no optimizer or
BatchNorm-statistics update.  A deterministic isomorphic round-robin adapter
forms one shared GEN2/NGN2 queue so the downstream 56% structural and T7
protocol remains matched to the completed L1/FPGM comparison.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback
import types
from typing import Any


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT.parent
RGP_HELPER_PATH = STUDY_ROOT / "scripts" / "rank_bdd_rgp_competitor.py"
COMPETITOR_HELPER_PATH = STUDY_ROOT / "scripts" / "rank_bdd_l1_fpgm_competitors.py"
# v1 used only two gradient batches as an integration smoke test.  The
# immutable v2 experiment follows the official implementation's default of
# 50 Taylor gradient batches and is the only run intended for comparison.
OUTPUT_ROOT = STUDY_ROOT / "results" / "pruning" / "bdd_isomorphic_taylor_rankings_v2" / "isomorphic_taylor_37_5pct_taylor50b"
DOMAINS = ("GEN2", "NGN2")
METHOD = "isomorphic_taylor"
LOCAL_PERCENT = 37.5
SEED = 42
PAPER_TAYLOR_BATCHES = 50
PROTECTED_GROUPS = {
    "DG001": "stem remains protected",
    "DG020": "excluded by the completed cumulative-collapse audit",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


rgp = load_module("_bdd_isomorphic_taylor_rgp_helpers", RGP_HELPER_PATH)
competitor = load_module("_bdd_isomorphic_taylor_competitor_helpers", COMPETITOR_HELPER_PATH)
bdd = competitor.bdd
base = competitor.base
engine = competitor.engine


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def relative(path: Path) -> str:
    return base.relative(path)


def all_group_ids() -> list[str]:
    groups = engine.all_group_ids()
    if len(groups) != 51 or len(set(groups)) != 51:
        raise RuntimeError("Expected the frozen catalogue of 51 legal YOLO26n groups")
    return groups


def topology_signature(graph: Any, group: Any) -> str:
    """Match Torch-Pruning's ``isomorphic=True`` group-tag construction."""
    parts: list[str] = []
    for dependency, _ in group:
        source = f"{type(dependency.source.module)}_{'out' if graph.is_out_channel_pruning_fn(dependency.handler) else 'in'}"
        target = f"{type(dependency.target.module)}_{'out' if graph.is_out_channel_pruning_fn(dependency.handler) else 'in'}"
        parts.append(f"{source}_{target}")
    if not parts:
        raise RuntimeError("An isomorphic group has no dependency signature")
    return "Isomorphic_" + "".join(parts)


def custom_signature(model: Any, group_id: str, selection_unit: str) -> str:
    row = engine.custom_rows()[group_id]
    block = model.model[int(row["block_index"])]
    return f"CUSTOM_{type(block)}_{row['rule_family']}_{selection_unit}"


def taylor_gradient_accumulator(model: Any, checkpoint: Path, dataset_yaml: Path, max_batches: int) -> dict[str, Any]:
    """Accumulate training-loss gradients without updating model parameters."""
    import torch

    if max_batches != PAPER_TAYLOR_BATCHES:
        raise ValueError(
            f"This comparison requires exactly {PAPER_TAYLOR_BATCHES} Taylor "
            "gradient batches, matching the official implementation default."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Isomorphic-Taylor loss-gradient saliency")
    rgp.configure_reproducible_gradients()
    device = torch.device("cuda:0")
    loss_configuration = rgp.configure_detection_loss(model, checkpoint)
    model.float().to(device).requires_grad_(True).train()
    batchnorm_layers = rgp.freeze_batchnorm_statistics(model)
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not convolutions:
        raise RuntimeError("No Conv2d modules were found")
    accumulated = {id(module): torch.zeros_like(module.weight, dtype=torch.float64, device="cpu") for module in convolutions}
    loader, dataset_images = rgp.build_calibration_loader(dataset_yaml)
    batches_seen = 0
    images_seen = 0
    loss_sum = 0.0
    try:
        for batch in loader:
            if max_batches and batches_seen >= max_batches:
                break
            prepared = rgp.prepare_loss_batch(batch, device)
            model.zero_grad(set_to_none=True)
            result = model.loss(prepared)
            loss_components = result[0] if isinstance(result, tuple) else result
            if not isinstance(loss_components, torch.Tensor) or not bool(torch.isfinite(loss_components).all()):
                raise RuntimeError("Taylor calibration produced non-finite loss components")
            loss = loss_components.sum()
            if loss.numel() != 1 or not bool(torch.isfinite(loss)):
                raise RuntimeError("Taylor calibration produced a non-finite scalar loss")
            loss.backward()
            for module in convolutions:
                gradient = module.weight.grad
                if gradient is None or not bool(torch.isfinite(gradient).all()):
                    raise RuntimeError("A convolution weight did not receive a finite loss gradient")
                accumulated[id(module)].add_(gradient.detach().to(device="cpu", dtype=torch.float64))
            batches_seen += 1
            images_seen += int(prepared["img"].shape[0])
            loss_sum += float(loss.detach().cpu())
        if batches_seen == 0:
            raise RuntimeError("Taylor calibration did not receive a training batch")
    finally:
        del loader

    model.cpu().eval()
    for module in convolutions:
        module.weight.grad = (accumulated[id(module)] / batches_seen).to(dtype=module.weight.dtype)
    torch.cuda.empty_cache()
    return {
        "calibration_dataset_images": dataset_images,
        "calibration_batches_seen": batches_seen,
        "calibration_images_seen": images_seen,
        "mean_loss_across_batches": loss_sum / batches_seen,
        "batchnorm_layers_forced_eval": batchnorm_layers,
        "conv_layers_scored": len(convolutions),
        "gradient_accumulation": "mean loss gradient per Conv2d weight over deterministic training batches",
        "loss_configuration": loss_configuration,
    }


def generic_taylor_scores_and_signature(model: Any, group_id: str) -> tuple[Any, str]:
    """Use official GroupTaylorImportance for a live DepGraph group."""
    import torch
    import torch_pruning as tp

    row = engine.generic_rows()[group_id]
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: representative root is not Conv2d")
    width = int(root.out_channels)
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    head = model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("Expected the frozen YOLO26 end-to-end Detect head")
    head.forward = types.MethodType(base.trace_detect_forward, head)

    class TraceWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, images: Any) -> tuple[Any, ...]:
            outputs = tuple(base.flatten_tensors(self.inner(images)))
            if not outputs:
                raise RuntimeError("Trace wrapper produced no tensor outputs")
            return outputs

    try:
        graph = tp.DependencyGraph().build_dependency(
            TraceWrapper(model),
            example_inputs=torch.zeros(1, 3, engine.TRACE_SIZE, engine.TRACE_SIZE),
        )
        group = graph.get_pruning_group(root, tp.prune_conv_out_channels, idxs=list(range(width)))
        importance = tp.importance.GroupTaylorImportance(
            group_reduction="mean", normalizer="mean", bias=False
        )(group)
        if importance is None or importance.numel() != width or not bool(torch.isfinite(importance).all()):
            raise RuntimeError(f"{group_id}: invalid GroupTaylorImportance vector")
        return importance.detach().float().cpu(), topology_signature(graph, group)
    finally:
        if "forward" in head.__dict__:
            delattr(head, "forward")
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def custom_taylor_scores_and_signature(model: Any, group_id: str) -> tuple[Any, str, Any | None, str]:
    """Taylor counterpart for the project's validated coupled custom units."""
    import torch

    row = engine.custom_rows()[group_id]
    block_index = int(row["block_index"])
    block = model.model[block_index]
    gradient = block.cv1.conv.weight.grad
    if gradient is None:
        raise RuntimeError(f"{group_id}: custom cv1 has no accumulated gradient")
    values = (block.cv1.conv.weight.detach().float() * gradient.detach().float()).abs().flatten(start_dim=1).sum(dim=1).cpu()
    c = int(block.c)
    if values.numel() != 2 * c:
        raise RuntimeError(f"{group_id}: custom cv1 width changed unexpectedly")
    layout = None
    if block_index not in (10, 22):
        scores = (values[:c] + values[c : 2 * c]) / 2.0
        unit = "hidden_channel"
    else:
        if block_index == 10:
            _, layout = engine.c2psa_head_aware_importance(block)
        else:
            _, layout = engine.attention_c3k2_head_aware_importance(block)
        paired: list[Any] = []
        for head_index in range(layout.num_heads):
            offset = head_index * layout.head_dim
            for position in range(layout.key_dim):
                paired.append((values[offset + position] + values[offset + layout.key_dim + position]) / 2.0)
        scores = torch.stack(paired)
        unit = "paired_attention_unit"
    mean = float(scores.mean())
    if not mean > 0.0 or not bool(torch.isfinite(scores).all()):
        raise RuntimeError(f"{group_id}: invalid custom Taylor scores")
    return scores / mean, custom_signature(model, group_id, unit), layout, unit


def select_units(scores: Any, group_id: str, attention_layout: Any | None, unit: str) -> tuple[list[int], list[int]]:
    import torch

    values = scores.detach().float().cpu().flatten()
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"{group_id}: non-finite Taylor score")
    if attention_layout is not None:
        selected = engine.balanced_attention_indices(values, attention_layout, competitor.LOCAL_FRACTION)
        ordered = sorted(selected, key=lambda index: (float(values[index]), index))
        return sorted(ordered), ordered
    remove_count = engine.exact_count(int(values.numel()), competitor.LOCAL_FRACTION, f"{group_id} selectable units")
    ordered = [int(index) for index in torch.argsort(values, stable=True)[:remove_count].tolist()]
    return sorted(ordered), ordered


def score_domain(model: Any, domain: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    masks: dict[str, Any] = {}
    for group_id in all_group_ids():
        if group_id.startswith("CDG"):
            scores, signature, layout, unit = custom_taylor_scores_and_signature(model, group_id)
            root_width = int(model.model[int(engine.custom_rows()[group_id]["block_index"])].c)
        else:
            scores, signature = generic_taylor_scores_and_signature(model, group_id)
            layout = None
            unit = "output_channel"
            root_width = int(scores.numel())
        selected, rank_order = select_units(scores, group_id, layout, unit)
        values = scores.detach().float().cpu().flatten()
        result = {
            "group_kind": "CUSTOM" if group_id.startswith("CDG") else "GENERIC",
            "selection_unit": unit,
            "score_basis": "GroupTaylorImportance_mean_mean" if not group_id.startswith("CDG") else "coupled_cv1_abs_weight_times_mean_gradient",
            "isomorphic_scope": signature,
            "root_channels_before": root_width,
            "selectable_units_before": int(values.numel()),
            "selected_indices": selected,
            "selection_rank_order": rank_order,
            "selected_scores_rank_order": [float(values[index]) for index in rank_order],
            "prune_first_mean_score": float(values[selected].mean()),
            "prune_first_max_score": float(values[selected].max()),
            "all_unit_score_mean": float(values.mean()),
            "all_unit_score_std": float(values.std(unbiased=False)),
        }
        masks[group_id] = result
        rows.append({
            "domain": domain,
            "group_id": group_id,
            "group_kind": result["group_kind"],
            "eligible_for_cumulative_raw": group_id not in PROTECTED_GROUPS,
            "protected_reason": PROTECTED_GROUPS.get(group_id, ""),
            "selection_unit": unit,
            "score_basis": result["score_basis"],
            "isomorphic_scope": signature,
            "root_channels_before": root_width,
            "selectable_units_before": int(values.numel()),
            "selected_unit_count": len(selected),
            "prune_first_mean_score": result["prune_first_mean_score"],
            "prune_first_max_score": result["prune_first_max_score"],
            "all_unit_score_mean": result["all_unit_score_mean"],
            "all_unit_score_std": result["all_unit_score_std"],
        })
    return rows, masks


def assign_isomorphic_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank only within topology-matched scopes, then interleave those scopes."""
    eligible = [row for row in rows if row["eligible_for_cumulative_raw"]]
    scopes: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        scopes.setdefault(str(row["isomorphic_scope"]), []).append(row)
    for scope_rows in scopes.values():
        scope_rows.sort(key=lambda row: (float(row["prune_first_mean_score"]), str(row["group_id"])))
        for rank, row in enumerate(scope_rows, start=1):
            row["isomorphic_scope_rank"] = rank
            row["isomorphic_scope_size"] = len(scope_rows)
    queue: list[dict[str, Any]] = []
    depth = 0
    while True:
        appended = False
        for scope in sorted(scopes):
            scope_rows = scopes[scope]
            if depth < len(scope_rows):
                queue.append(scope_rows[depth])
                appended = True
        if not appended:
            break
        depth += 1
    for index, row in enumerate(queue, start=1):
        row["isomorphic_queue_index"] = index
    for row in rows:
        row.setdefault("isomorphic_scope_rank", "")
        row.setdefault("isomorphic_scope_size", "")
        row.setdefault("isomorphic_queue_index", "")
    return queue


def preflight() -> dict[str, Any]:
    import torch
    import ultralytics
    from importlib.metadata import version
    from ultralytics import YOLO

    freeze = bdd.read_freeze()
    bdd.verify_frozen_inputs(freeze)
    generic, custom = engine.generic_rows(), engine.custom_rows()
    if len(generic) != 42 or len(custom) != 9:
        raise RuntimeError("Expected 42 generic plus 9 custom frozen groups")
    for domain in DOMAINS:
        config = bdd.domain_config(domain)
        checkpoint, dataset_yaml = PROJECT_ROOT / config["path"], PROJECT_ROOT / config["dataset_yaml"]
        if not checkpoint.is_file() or not dataset_yaml.is_file():
            raise FileNotFoundError(f"Missing frozen {domain} checkpoint or dataset")
        if base.sha256(checkpoint) != config["sha256"] or base.sha256(dataset_yaml) != config["dataset_yaml_sha256"]:
            raise RuntimeError(f"Frozen {domain} input hash changed")
    probe = YOLO(str(PROJECT_ROOT / bdd.domain_config("GEN2")["path"]), task="detect").model.float().cpu().eval()
    _, c2_layout = engine.c2psa_head_aware_importance(probe.model[10])
    _, c3_layout = engine.attention_c3k2_head_aware_importance(probe.model[22])
    engine.exact_count(c2_layout.key_dim, competitor.LOCAL_FRACTION, "C2PSA units per head")
    engine.exact_count(c3_layout.key_dim, competitor.LOCAL_FRACTION, "attention C3k2 units per head")
    del probe
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Isomorphic-Taylor ranking")
    return {
        "schema": "bdd_isomorphic_taylor_ranking_preflight_v2",
        "method": METHOD,
        "domains": list(DOMAINS),
        "groups": 51,
        "generic_groups": 42,
        "custom_groups": 9,
        "local_pruning_percent": LOCAL_PERCENT,
        "calibration_split": "train",
        "calibration_order": "dataset order, shuffle=False, augmentation disabled",
        "calibration_batch_size": rgp.CALIBRATION_BATCH_SIZE,
        "calibration_batches": PAPER_TAYLOR_BATCHES,
        "calibration_batches_policy": "Exactly 50 training batches, matching the official Isomorphic-Pruning implementation default.",
        "seed": SEED,
        "protected_groups": PROTECTED_GROUPS,
        "runtime_versions": {"torch": torch.__version__, "torch_pruning": version("torch-pruning"), "ultralytics": ultralytics.__version__},
        "cuda_device": torch.cuda.get_device_name(0),
        "paper_basis": "Fang et al., Isomorphic Pruning for Vision Models, ECCV 2024",
        "adaptation": {
            "importance": "first-order Taylor abs(weight * mean training-loss gradient), using Torch-Pruning GroupTaylorImportance for generic DepGraph groups",
            "isomorphic_scope": "official Torch-Pruning isomorphic topology signature for generic groups; validated custom-rule signature for C3k2/C2PSA units",
            "shared_queue": "equal-weight mean of domain-specific isomorphic round-robin queue indices; required only to keep the BDD 56% structural protocol common across GEN2 and NGN2",
            "one_shot_masks": "channel/unit masks are frozen from the baseline scoring pass; raw replay never uses validation/test data or re-ranks",
        },
    }


def run(calibration_batches: int) -> int:
    from ultralytics import YOLO

    manifest_path = OUTPUT_ROOT / "ranking_manifest.json"
    if manifest_path.is_file():
        raise RuntimeError(f"Output already exists: {OUTPUT_ROOT}. Create a new version instead of overwriting evidence.")
    evidence = preflight()
    manifest = {**evidence, "schema": "bdd_isomorphic_taylor_ranking_manifest_v2", "status": "RUNNING", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "script": relative(SELF), "script_sha256": base.sha256(SELF), "calibration_batches_requested": calibration_batches}
    atomic_json(manifest_path, manifest)
    try:
        per_domain: dict[str, list[dict[str, Any]]] = {}
        masks: dict[str, dict[str, Any]] = {}
        calibration: dict[str, Any] = {}
        for domain in DOMAINS:
            config = bdd.domain_config(domain)
            checkpoint = PROJECT_ROOT / config["path"]
            dataset_yaml = PROJECT_ROOT / config["dataset_yaml"]
            model = YOLO(str(checkpoint), task="detect").model.float().eval()
            calibration[domain] = taylor_gradient_accumulator(model, checkpoint, dataset_yaml, calibration_batches)
            rows, domain_masks = score_domain(model, domain)
            queue = assign_isomorphic_queue(rows)
            per_domain[domain], masks[domain] = rows, domain_masks
            atomic_csv(OUTPUT_ROOT / "tables" / f"{domain}_ISOMORPHIC_TAYLOR_37_5PCT.csv", list(rows[0]), rows)
            atomic_json(OUTPUT_ROOT / "masks" / f"{domain}_isomorphic_taylor_37_5pct_masks.json", {"schema": "bdd_isomorphic_taylor_masks_v2", "domain": domain, "calibration": calibration[domain], "masks": domain_masks})
            atomic_json(OUTPUT_ROOT / "queues" / f"{domain}_isomorphic_queue.json", {"domain": domain, "policy": "rank within topology-matched scope then deterministic round-robin interleave", "entries": queue})
            del model

        by_domain = {(domain, row["group_id"]): row for domain, rows in per_domain.items() for row in rows}
        shared: list[dict[str, Any]] = []
        for group_id in all_group_ids():
            gen, ngn = by_domain[("GEN2", group_id)], by_domain[("NGN2", group_id)]
            if not bool(gen["eligible_for_cumulative_raw"]):
                continue
            shared.append({
                "method": METHOD,
                "group_id": group_id,
                "group_kind": gen["group_kind"],
                "GEN2_isomorphic_queue_index": gen["isomorphic_queue_index"],
                "NGN2_isomorphic_queue_index": ngn["isomorphic_queue_index"],
                "mean_domain_isomorphic_queue_index": (int(gen["isomorphic_queue_index"]) + int(ngn["isomorphic_queue_index"])) / 2.0,
                "GEN2_scope": gen["isomorphic_scope"],
                "NGN2_scope": ngn["isomorphic_scope"],
            })
        shared.sort(key=lambda row: (float(row["mean_domain_isomorphic_queue_index"]), str(row["group_id"])))
        for index, row in enumerate(shared, start=1):
            row["cumulative_queue_index"] = index
        atomic_csv(OUTPUT_ROOT / "tables" / "ISOMORPHIC_TAYLOR_SHARED_CUMULATIVE_QUEUE.csv", list(shared[0]), shared)
        atomic_json(OUTPUT_ROOT / "shared_cumulative_queue.json", {"schema": "bdd_isomorphic_taylor_shared_queue_v2", "method": METHOD, "policy": "equal-weight mean of GEN2 and NGN2 topology-scoped queue positions; ties by group_id", "protected_groups": PROTECTED_GROUPS, "entries": shared})
        (OUTPUT_ROOT / "README.md").write_text(
            "# BDD Isomorphic-Taylor competitor\n\n"
            "Controlled adaptation of Isomorphic Pruning (Fang et al., ECCV 2024) using first-order Taylor importance. "
            "The original paper/source targets ImageNet CNNs and transformers, so YOLO26n custom C3k2/C2PSA units use this project's validated coupled structural rules. "
            "This is a one-shot ranking/mask stage; no validation/test data, pruning, or recovery occurs here.\n",
            encoding="utf-8",
        )
        manifest.update({"status": "PASS", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "calibration": calibration, "candidate_count": len(shared), "shared_queue": relative(OUTPUT_ROOT / "shared_cumulative_queue.json")})
        atomic_json(manifest_path, manifest)
        print(json.dumps({"status": "PASS", "output": relative(OUTPUT_ROOT), "candidate_count": len(shared)}, sort_keys=True))
        return 0
    except Exception as error:
        manifest.update({"status": "FAIL", "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}})
        atomic_json(manifest_path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-batches", type=int, default=PAPER_TAYLOR_BATCHES)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.calibration_batches != PAPER_TAYLOR_BATCHES:
        parser.error(f"--calibration-batches must be {PAPER_TAYLOR_BATCHES} for this controlled comparison")
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    return run(args.calibration_batches)


if __name__ == "__main__":
    raise SystemExit(main())
