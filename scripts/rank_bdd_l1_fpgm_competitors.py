"""Create ranking-only BDD competitors for Global-L1 and FPGM.

This is intentionally a *ranking* stage.  It does not structurally prune a
model, evaluate validation data, update BatchNorm statistics, fine-tune, or
write a T5/T7 artifact.  It freezes the channel/unit masks and one shared
GEN2/NGN2 candidate order which a later structural runner must consume.

Both competitors use the same 51 physically validated legal groups as the
BDD proposed method.  Per-domain masks are selected from that domain's frozen
baseline.  A single BDD order is then formed by the equal-weight mean of the
two per-domain ranks; no AD/NAD, domain sensitivity, or proposed-formula term
is used in either competitor.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
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
PROJECT_ROOT = STUDY_ROOT
BDD_RUNNER_PATH = STUDY_ROOT / "scripts" / "run_bdd_t1_t2_sweep.py"
# V1 is retained as preliminary evidence.  V2 is the first publication-ready
# comparator ranking: generic L1 is computed from the live DepGraph group,
# rather than a representative root alone.
OUTPUT_BASE = STUDY_ROOT / "results" / "pruning" / "bdd_gen2_ngn2_competitor_rankings_v2"
LOCAL_FRACTION = Fraction(3, 8)
LOCAL_PERCENT = 37.5
DOMAINS = ("GEN2", "NGN2")
METHODS = ("global_l1", "fpgm")
PROTECTED_GROUPS = {
    "DG001": "stem remains protected",
    "DG020": "excluded by the completed cumulative-collapse audit",
}


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Existing historical scripts use local sibling imports.
for import_root in (
    STUDY_ROOT / "scripts",
    STUDY_ROOT / "results" / "pruning" / "prune_25",
    STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


bdd = load_module("_bdd_competitor_bdd_runner", BDD_RUNNER_PATH)
base = bdd.base
engine = bdd.engine


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


def output_root(method: str) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    return OUTPUT_BASE / f"{method}_{str(LOCAL_PERCENT).replace('.', '_')}pct"


def all_group_ids() -> list[str]:
    groups = engine.all_group_ids()
    if len(groups) != 51 or len(set(groups)) != 51:
        raise RuntimeError("Expected exactly 51 unique validated groups")
    return groups


def exact_count(width: int, label: str) -> int:
    return engine.exact_count(width, LOCAL_FRACTION, label)


def rank_rows(rows: list[dict[str, Any]], key: str, rank_key: str) -> None:
    rows.sort(key=lambda row: (float(row[key]), str(row["group_id"])))
    for rank, row in enumerate(rows, start=1):
        row[rank_key] = rank


def fpgm_distances(vectors: Any) -> Any:
    """Distance of each filter/unit vector from its geometric median.

    Weiszfeld iterations are deterministic on CPU and make no CUDA allocation.
    The returned lower score means a more redundant FPGM candidate.
    """

    import torch

    points = vectors.detach().float().cpu()
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError(f"FPGM requires an N x D matrix with N >= 2, got {tuple(points.shape)}")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("FPGM received non-finite filter values")
    median = points.mean(dim=0)
    for _ in range(256):
        distance = torch.linalg.vector_norm(points - median, dim=1)
        zero = distance <= 1e-12
        if bool(zero.any()):
            median = points[zero].mean(dim=0)
            break
        weights = distance.reciprocal()
        next_median = (points * weights[:, None]).sum(dim=0) / weights.sum()
        if float(torch.linalg.vector_norm(next_median - median)) <= 1e-7:
            median = next_median
            break
        median = next_median
    result = torch.linalg.vector_norm(points - median, dim=1)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("FPGM produced a non-finite distance")
    return result


def generic_root_vectors(model: Any, group_id: str) -> Any:
    import torch

    row = engine.generic_rows()[group_id]
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: representative root is not Conv2d")
    if int(root.out_channels) != int(row["root_out_channels"]):
        raise RuntimeError(f"{group_id}: root width differs from the frozen catalogue")
    return root.weight.detach().float().cpu().flatten(start_dim=1)


def generic_depgraph_l1_scores(model: Any, group_id: str) -> Any:
    """Return the established DepGraph group-aware L1 vector for one root.

    This mirrors the generic score construction used by the frozen T1/T2
    engine, without applying a pruning operation.  It makes the L1 competitor
    genuinely ``DepGraph + GroupMagnitude(L1)`` rather than representative
    filter L1 alone.
    """

    import torch
    import torch_pruning as tp

    row = engine.generic_rows()[group_id]
    root = base.find_module(model, row["representative_root"])
    if not isinstance(root, torch.nn.Conv2d):
        raise TypeError(f"{group_id}: representative root is not Conv2d")
    channels = int(root.out_channels)
    if channels != int(row["root_out_channels"]):
        raise RuntimeError(f"{group_id}: root width differs from the frozen catalogue")
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    head = model.model[-1]
    if not getattr(head, "end2end", False):
        raise RuntimeError("Expected a YOLO26 end-to-end Detect head")
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
        graph = tp.DependencyGraph().build_dependency(
            TraceWrapper(model),
            example_inputs=torch.zeros(1, 3, engine.TRACE_SIZE, engine.TRACE_SIZE),
        )
        full_group = graph.get_pruning_group(
            root, tp.prune_conv_out_channels, idxs=list(range(channels))
        )
        importance = tp.importance.GroupMagnitudeImportance(
            p=1, group_reduction="mean", normalizer="mean", bias=False
        )(full_group)
        if importance is None or importance.numel() != channels or not bool(torch.isfinite(importance).all()):
            raise RuntimeError(f"{group_id}: invalid DepGraph group-aware L1 vector")
        return importance.detach().float().cpu()
    finally:
        if "forward" in head.__dict__:
            delattr(head, "forward")
        model.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.requires_grad_(False)


def custom_root_vectors(model: Any, group_id: str) -> tuple[Any, str]:
    """One coupled representative vector per custom logical pruning unit.

    A custom block cannot be pruned as one unconstrained Conv2d filter.  The
    canonical ``cv1`` output is therefore used as its representative root: the
    two coupled split branches are concatenated for non-attention C3k2, while
    the two embedding channels belonging to each attention unit are
    concatenated for attention-bearing blocks.
    """

    import torch

    row = engine.custom_rows()[group_id]
    block_index = int(row["block_index"])
    block = model.model[block_index]
    c = int(block.c)
    weight = block.cv1.conv.weight.detach().float().cpu().flatten(start_dim=1)
    if int(block.cv1.conv.out_channels) != 2 * c:
        raise RuntimeError(f"{group_id}: custom cv1 no longer has two hidden branches")
    if block_index not in (10, 22):
        return torch.cat((weight[:c], weight[c : 2 * c]), dim=1), "paired_cv1_split_filters"

    if block_index == 10:
        _, layout = engine.c2psa_head_aware_importance(block)
    else:
        _, layout = engine.attention_c3k2_head_aware_importance(block)
    vectors: list[Any] = []
    for head in range(layout.num_heads):
        base_index = head * layout.head_dim
        for position in range(layout.key_dim):
            vectors.append(torch.cat((weight[base_index + position], weight[base_index + layout.key_dim + position])))
    result = torch.stack(vectors, dim=0)
    if result.shape[0] != layout.total_units:
        raise RuntimeError(f"{group_id}: custom attention unit count changed")
    return result, "paired_cv1_attention_embedding_filters"


def select_attention(scores: Any, layout: Any) -> tuple[list[int], list[int]]:
    selected = engine.balanced_attention_indices(scores, layout, LOCAL_FRACTION)
    rank_order = sorted(selected, key=lambda index: (float(scores[index]), index))
    return sorted(selected), rank_order


def l1_custom_scores(model: Any, group_id: str) -> tuple[Any, str, Any | None]:
    row = engine.custom_rows()[group_id]
    block_index = int(row["block_index"])
    block = model.model[block_index]
    if block_index in engine.custom.NONATTENTION_BLOCKS:
        return engine.nonattention_c3k2_logical_importance(block), "coupled_logical_l1", None
    if block_index == 10:
        scores, layout = engine.c2psa_head_aware_importance(block)
        return scores, "coupled_head_aware_l1", layout
    if block_index == 22:
        scores, layout = engine.attention_c3k2_head_aware_importance(block)
        return scores, "coupled_head_aware_l1", layout
    raise ValueError(f"Unsupported custom block index: {block_index}")


def score_group(model: Any, group_id: str, method: str) -> dict[str, Any]:
    import torch

    is_custom = group_id.startswith("CDG")
    if method == "global_l1":
        if is_custom:
            scores, score_basis, attention_layout = l1_custom_scores(model, group_id)
        else:
            scores = generic_depgraph_l1_scores(model, group_id)
            score_basis, attention_layout = "DepGraph_GroupMagnitudeImportance_p1_mean_mean", None
    elif method == "fpgm":
        vectors, score_basis = (
            custom_root_vectors(model, group_id) if is_custom else (generic_root_vectors(model, group_id), "canonical_root_filters")
        )
        scores = fpgm_distances(vectors)
        attention_layout = None
        if is_custom:
            block_index = int(engine.custom_rows()[group_id]["block_index"])
            if block_index == 10:
                _, attention_layout = engine.c2psa_head_aware_importance(model.model[block_index])
            elif block_index == 22:
                _, attention_layout = engine.attention_c3k2_head_aware_importance(model.model[block_index])
    else:
        raise ValueError(f"Unknown method: {method}")

    scores = scores.detach().float().cpu().flatten()
    if not bool(torch.isfinite(scores).all()):
        raise RuntimeError(f"{group_id}: non-finite {method} scores")
    if attention_layout is not None:
        selected, rank_order = select_attention(scores, attention_layout)
        selection_unit = "paired_attention_unit"
        root_width = int(model.model[int(engine.custom_rows()[group_id]["block_index"])].c)
    else:
        width = int(scores.numel())
        remove_count = exact_count(width, f"{group_id} selectable units")
        rank_order = [int(value) for value in torch.argsort(scores, stable=True)[:remove_count].tolist()]
        selected = sorted(rank_order)
        selection_unit = "hidden_channel" if is_custom else "output_channel"
        root_width = int(engine.custom_rows()[group_id]["hidden_channels_before"]) if is_custom else width
    if not selected:
        raise RuntimeError(f"{group_id}: selection is empty")
    return {
        "group_kind": "CUSTOM" if is_custom else "GENERIC",
        "selection_unit": selection_unit,
        "score_basis": score_basis,
        "root_channels_before": root_width,
        "selectable_units_before": int(scores.numel()),
        "selected_indices": selected,
        "selection_rank_order": rank_order,
        "selected_scores_rank_order": [float(scores[index]) for index in rank_order],
        "prune_first_mean_score": float(scores[selected].mean()),
        "prune_first_max_score": float(scores[selected].max()),
        "all_unit_score_mean": float(scores.mean()),
        "all_unit_score_std": float(scores.std(unbiased=False)),
    }


def preflight() -> dict[str, Any]:
    import torch
    import ultralytics
    from importlib.metadata import version
    from ultralytics import YOLO

    freeze = bdd.read_freeze()
    bdd.verify_frozen_inputs(freeze)
    generic, custom, groups = engine.generic_rows(), engine.custom_rows(), all_group_ids()
    if len(generic) != 42 or len(custom) != 9:
        raise RuntimeError("The frozen catalogue no longer contains 42 generic + 9 custom groups")
    for group_id, row in generic.items():
        exact_count(int(row["root_out_channels"]), f"{group_id} root")
    for group_id, row in custom.items():
        exact_count(int(row["hidden_channels_before"]), f"{group_id} hidden width")
    for domain in DOMAINS:
        config = bdd.domain_config(domain)
        checkpoint = PROJECT_ROOT / config["path"]
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if base.sha256(checkpoint) != config["sha256"]:
            raise RuntimeError(f"{domain} baseline checkpoint hash changed")
    probe = YOLO(str(PROJECT_ROOT / bdd.domain_config("GEN2")["path"]), task="detect").model.float().cpu().eval()
    _, c2_layout = engine.c2psa_head_aware_importance(probe.model[10])
    _, c3_layout = engine.attention_c3k2_head_aware_importance(probe.model[22])
    exact_count(c2_layout.key_dim, "C2PSA units per head")
    exact_count(c3_layout.key_dim, "attention C3k2 units per head")
    del probe
    return {
        "schema": "bdd_l1_fpgm_competitor_ranking_preflight_v1",
        "domains": list(DOMAINS),
        "groups": len(groups),
        "generic_groups": len(generic),
        "custom_groups": len(custom),
        "local_pruning_fraction": str(LOCAL_FRACTION),
        "local_pruning_percent": LOCAL_PERCENT,
        "protected_groups": PROTECTED_GROUPS,
        "runtime_versions": {
            "torch": torch.__version__,
            "torch_pruning": version("torch-pruning"),
            "ultralytics": ultralytics.__version__,
        },
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "input_freeze": relative(bdd.FREEZE_PATH),
        "input_freeze_sha256": base.sha256(bdd.FREEZE_PATH),
    }


def run(method: str) -> int:
    import torch
    from ultralytics import YOLO

    output = output_root(method)
    manifest_path = output / "ranking_manifest.json"
    if manifest_path.is_file():
        raise RuntimeError(f"Ranking output already exists: {output}. Create a new versioned output instead of overwriting it.")
    evidence = preflight()
    manifest = {
        **evidence,
        "schema": "bdd_l1_fpgm_competitor_ranking_manifest_v1",
        "status": "RUNNING",
        "method": method,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": relative(SELF),
        "script_sha256": base.sha256(SELF),
        "policy": {
            "ranking_only": True,
            "validation_data_used": False,
            "fine_tuning": False,
            "batchnorm_update": False,
            "structural_pruning_performed": False,
            "same_51_legal_groups_as_proposed_method": True,
            "shared_order": "equal-weight mean of independent GEN2 and NGN2 per-domain ranks; ties by group_id",
            "global_l1": "lower DepGraph group-aware L1 (generic) or validated coupled-unit L1 (custom) is pruned earlier",
            "fpgm": "lower distance to a canonical-root/coupled-unit geometric median is pruned earlier; DepGraph/custom rules enforce structure later",
        },
    }
    atomic_json(manifest_path, manifest)
    try:
        per_domain: dict[str, list[dict[str, Any]]] = {}
        masks: dict[str, dict[str, Any]] = {}
        for domain in DOMAINS:
            config = bdd.domain_config(domain)
            checkpoint = PROJECT_ROOT / config["path"]
            model = YOLO(str(checkpoint), task="detect").model.float().cpu().eval()
            rows: list[dict[str, Any]] = []
            domain_masks: dict[str, Any] = {}
            for group_id in all_group_ids():
                result = score_group(model, group_id, method)
                row = {
                    "domain": domain,
                    "group_id": group_id,
                    "group_kind": result["group_kind"],
                    "eligible_for_cumulative_raw": group_id not in PROTECTED_GROUPS,
                    "selection_unit": result["selection_unit"],
                    "score_basis": result["score_basis"],
                    "root_channels_before": result["root_channels_before"],
                    "selectable_units_before": result["selectable_units_before"],
                    "selected_unit_count": len(result["selected_indices"]),
                    "prune_first_mean_score": result["prune_first_mean_score"],
                    "prune_first_max_score": result["prune_first_max_score"],
                    "all_unit_score_mean": result["all_unit_score_mean"],
                    "all_unit_score_std": result["all_unit_score_std"],
                }
                rows.append(row)
                domain_masks[group_id] = result
            rank_rows(rows, "prune_first_mean_score", "domain_rank_prune_first_ascending")
            per_domain[domain] = rows
            masks[domain] = domain_masks
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        by_domain_group = {
            (domain, row["group_id"]): row
            for domain, rows in per_domain.items()
            for row in rows
        }
        shared: list[dict[str, Any]] = []
        for group_id in all_group_ids():
            gen = by_domain_group[("GEN2", group_id)]
            ngn = by_domain_group[("NGN2", group_id)]
            shared.append({
                "method": method,
                "group_id": group_id,
                "group_kind": gen["group_kind"],
                "eligible_for_cumulative_raw": group_id not in PROTECTED_GROUPS,
                "protected_reason": PROTECTED_GROUPS.get(group_id, ""),
                "GEN2_rank": gen["domain_rank_prune_first_ascending"],
                "NGN2_rank": ngn["domain_rank_prune_first_ascending"],
                "mean_domain_rank": (gen["domain_rank_prune_first_ascending"] + ngn["domain_rank_prune_first_ascending"]) / 2.0,
                "GEN2_prune_first_mean_score": gen["prune_first_mean_score"],
                "NGN2_prune_first_mean_score": ngn["prune_first_mean_score"],
                "GEN2_selected_unit_count": gen["selected_unit_count"],
                "NGN2_selected_unit_count": ngn["selected_unit_count"],
            })
        shared.sort(key=lambda row: (float(row["mean_domain_rank"]), str(row["group_id"])))
        for rank, row in enumerate(shared, start=1):
            row["shared_rank_prune_first_ascending"] = rank
        queue = [row for row in shared if row["eligible_for_cumulative_raw"]]
        for queue_index, row in enumerate(queue, start=1):
            row["cumulative_queue_index"] = queue_index
        for row in shared:
            row.setdefault("cumulative_queue_index", "")

        domain_fields = list(per_domain["GEN2"][0])
        shared_fields = [
            "method", "shared_rank_prune_first_ascending", "cumulative_queue_index", "group_id", "group_kind",
            "eligible_for_cumulative_raw", "protected_reason", "GEN2_rank", "NGN2_rank", "mean_domain_rank",
            "GEN2_prune_first_mean_score", "NGN2_prune_first_mean_score", "GEN2_selected_unit_count", "NGN2_selected_unit_count",
        ]
        for domain in DOMAINS:
            atomic_csv(output / "tables" / f"{domain}_{method.upper()}_UNIT_SALIENCY_37_5PCT.csv", domain_fields, per_domain[domain])
            atomic_json(output / "masks" / f"{domain}_{method}_37_5pct_masks.json", {
                "schema": "bdd_competitor_domain_masks_v1",
                "method": method,
                "domain": domain,
                "local_pruning_percent": LOCAL_PERCENT,
                "baseline_checkpoint": bdd.domain_config(domain)["path"],
                "baseline_checkpoint_sha256": bdd.domain_config(domain)["sha256"],
                "masks": masks[domain],
            })
        atomic_csv(output / "tables" / f"SHARED_{method.upper()}_RANKING_37_5PCT.csv", shared_fields, shared)
        atomic_json(output / "shared_cumulative_queue.json", {
            "schema": "bdd_competitor_shared_queue_v1",
            "method": method,
            "local_pruning_percent": LOCAL_PERCENT,
            "aggregation": "equal-weight mean of independent GEN2 and NGN2 ranks; ties by group_id",
            "protected_groups": PROTECTED_GROUPS,
            "entries": queue,
        })
        readme = (
            f"# BDD {method.upper()} ranking at 37.5% local pruning\n\n"
            "This folder is ranking-only. It contains no cumulative model, validation metric, fine-tuning, or T7 result. "
            "The channel/unit masks are domain-specific, but the candidate order is the equal-weight mean of GEN2 and NGN2 ranks. "
            "The 49-entry queue excludes DG001 and DG020 solely because those protections also apply to the proposed-method raw56 workflow.\n"
        )
        (output / "README.md").write_text(readme, encoding="utf-8")
        manifest.update({
            "status": "PASS",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "ranking_table": relative(output / "tables" / f"SHARED_{method.upper()}_RANKING_37_5PCT.csv"),
            "shared_queue": relative(output / "shared_cumulative_queue.json"),
            "candidate_count": len(queue),
        })
        atomic_json(manifest_path, manifest)
        print(json.dumps({"status": "PASS", "method": method, "output": relative(output), "candidate_count": len(queue)}, sort_keys=True))
        return 0
    except Exception as error:
        manifest.update({
            "status": "FAIL",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "error": {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()},
        })
        atomic_json(manifest_path, manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        print(json.dumps(preflight(), indent=2, sort_keys=True))
        return 0
    return run(args.method)


if __name__ == "__main__":
    raise SystemExit(main())
