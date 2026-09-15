"""Create a reproducible BDD RGP-style filter-saliency ranking for YOLO26n.

This is an equation-based *adaptation* of Rigorous Gradation Pruning (RGP)
for the existing BDD GEN2/NGN2 YOLO26n experiment.  The authors' advertised
repository was unavailable when this implementation was prepared, so this is
not represented as an execution of their code.

For one frozen domain baseline, the script uses only that domain's training
split (never validation/test data) to accumulate first-order loss gradients.
For every Conv2d weight element it applies the final displayed RGP Equation 4,
``exp(weight * dL/dweight)``.  It aggregates element scores into output-filter
scores, mean-normalizes scores within the same convolution, and then maps these
scores to the project's validated DepGraph/custom logical units.

The result is deliberately domain-specific: RGP is a single-dataset method.
It does not mix GEN2 and NGN2 gradients, AD/NAD, alpha weighting, or the
proposed-method score.  It performs no structural pruning, validation,
BatchNorm update, recovery, or checkpoint write.
"""

from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Iterable


SELF = Path(__file__).resolve()
STUDY_ROOT = SELF.parents[1]
PROJECT_ROOT = STUDY_ROOT
BDD_RUNNER_PATH = STUDY_ROOT / "scripts" / "run_bdd_t1_t2_sweep.py"
OUTPUT_BASE = STUDY_ROOT / "results" / "pruning" / "bdd_rgp_taylor_gradation_rankings_v1"

DOMAINS = ("GEN2", "NGN2")
LOCAL_FRACTION = Fraction(3, 8)
LOCAL_PERCENT = 37.5
CALIBRATION_BATCH_SIZE = 16
CALIBRATION_WORKERS = 2
SEED = 42
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


# The validated BDD runner imports historical sibling modules dynamically.
for import_root in (
    STUDY_ROOT / "scripts",
    STUDY_ROOT / "results" / "pruning" / "prune_25",
    STUDY_ROOT / "results" / "pruning" / "prune_12_5" / "scripts",
):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


bdd = load_module("_bdd_rgp_runner", BDD_RUNNER_PATH)
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


def output_root(domain: str, calibration_batches: int) -> Path:
    if domain not in DOMAINS:
        raise ValueError(f"Unknown BDD domain: {domain}")
    if calibration_batches < 0:
        raise ValueError("calibration_batches cannot be negative")
    suffix = "full_train" if calibration_batches == 0 else f"first_{calibration_batches:04d}_batches"
    return OUTPUT_BASE / f"{domain}_rgp_taylor_gradation_{suffix}"


def all_group_ids() -> list[str]:
    groups = engine.all_group_ids()
    if len(groups) != 51 or len(set(groups)) != 51:
        raise RuntimeError("Expected exactly 51 unique validated BDD groups")
    return groups


def exact_count(width: int, label: str) -> int:
    return engine.exact_count(width, LOCAL_FRACTION, label)


def rank_rows(rows: list[dict[str, Any]], score_key: str) -> None:
    rows.sort(key=lambda row: (float(row[score_key]), str(row["group_id"])))
    for rank, row in enumerate(rows, start=1):
        row["domain_rank_prune_first_ascending"] = rank


def preflight(domain: str, calibration_batches: int) -> dict[str, Any]:
    import torch
    import ultralytics
    from importlib.metadata import version
    from ultralytics import YOLO

    if calibration_batches < 0:
        raise ValueError("calibration_batches cannot be negative")
    freeze = bdd.read_freeze()
    bdd.verify_frozen_inputs(freeze)
    generic, custom, groups = engine.generic_rows(), engine.custom_rows(), all_group_ids()
    if len(generic) != 42 or len(custom) != 9:
        raise RuntimeError("Expected 42 generic plus 9 custom validated groups")
    for group_id, row in generic.items():
        exact_count(int(row["root_out_channels"]), f"{group_id} root")
    for group_id, row in custom.items():
        exact_count(int(row["hidden_channels_before"]), f"{group_id} hidden width")
    config = bdd.domain_config(domain)
    checkpoint = PROJECT_ROOT / config["path"]
    dataset_yaml = PROJECT_ROOT / config["dataset_yaml"]
    if not checkpoint.is_file() or not dataset_yaml.is_file():
        raise FileNotFoundError(f"Missing frozen input for {domain}")
    if base.sha256(checkpoint) != config["sha256"]:
        raise RuntimeError(f"{domain} baseline checkpoint hash changed")
    if base.sha256(dataset_yaml) != config["dataset_yaml_sha256"]:
        raise RuntimeError(f"{domain} dataset YAML hash changed")
    probe = YOLO(str(checkpoint), task="detect").model.float().cpu().eval()
    _, c2_layout = engine.c2psa_head_aware_importance(probe.model[10])
    _, c3_layout = engine.attention_c3k2_head_aware_importance(probe.model[22])
    exact_count(c2_layout.key_dim, "C2PSA units per head")
    exact_count(c3_layout.key_dim, "attention C3k2 units per head")
    del probe
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to calculate RGP loss-gradient saliency")
    return {
        "schema": "bdd_rgp_taylor_gradation_ranking_preflight_v1",
        "domain": domain,
        "groups": len(groups),
        "generic_groups": len(generic),
        "custom_groups": len(custom),
        "local_pruning_fraction": str(LOCAL_FRACTION),
        "local_pruning_percent": LOCAL_PERCENT,
        "protected_groups": PROTECTED_GROUPS,
        "calibration_split": "train",
        "calibration_order": "dataset order, shuffle=False, augmentation disabled",
        "calibration_batch_size": CALIBRATION_BATCH_SIZE,
        "seed": SEED,
        "cudnn_deterministic": True,
        "calibration_batches_requested": calibration_batches,
        "calibration_batches_policy": "0 means every training batch exactly once",
        "baseline_checkpoint": relative(checkpoint),
        "baseline_checkpoint_sha256": base.sha256(checkpoint),
        "dataset_yaml": relative(dataset_yaml),
        "dataset_yaml_sha256": base.sha256(dataset_yaml),
        "runtime_versions": {
            "torch": torch.__version__,
            "torch_pruning": version("torch-pruning"),
            "ultralytics": ultralytics.__version__,
        },
        "cuda_device": torch.cuda.get_device_name(0),
        "equation_interpretation": {
            "paper": "RGP Equation 4 final displayed form",
            "element_score": "exp(weight * dL_dweight)",
            "absolute_value_inside_exponent": False,
            "reason": "The published derivation discusses absolute values, but its final displayed Equation 4 uses exp(weight * dL/dweight); the advertised source repository was unavailable. This adaptation follows that final displayed expression exactly.",
            "filter_aggregation": "mean of element scores in each output filter",
            "layer_normalization": "filter score divided by the mean filter score of the same Conv2d layer",
        },
    }


def build_calibration_loader(dataset_yaml: Path) -> tuple[Any, int]:
    from ultralytics.cfg import get_cfg
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset

    data = check_det_dataset(str(dataset_yaml), autodownload=False)
    cfg = get_cfg(
        overrides={
            "task": "detect",
            "imgsz": 640,
            "rect": False,
            "cache": False,
            "single_cls": False,
            "classes": None,
            "fraction": 1.0,
        }
    )
    dataset = build_yolo_dataset(
        cfg,
        img_path=data["train"],
        batch=CALIBRATION_BATCH_SIZE,
        data=data,
        mode="val",  # Uses the training paths but disables train augmentation.
        rect=False,
        stride=32,
    )
    loader = build_dataloader(
        dataset,
        batch=CALIBRATION_BATCH_SIZE,
        workers=CALIBRATION_WORKERS,
        shuffle=False,
        rank=-1,
        pin_memory=True,
    )
    return loader, len(dataset)


def prepare_loss_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    prepared: dict[str, Any] = {}
    for key, value in batch.items():
        prepared[key] = value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
    image = prepared.get("img")
    if not isinstance(image, torch.Tensor):
        raise RuntimeError("RGP calibration batch has no image tensor")
    prepared["img"] = image.float().div_(255.0)
    return prepared


def freeze_batchnorm_statistics(model: Any) -> int:
    import torch

    count = 0
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def configure_reproducible_gradients() -> None:
    """Set the scoring-only random state before constructing the loader."""

    import torch

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    # The loader is ordered and uses no augmentation. These settings also keep
    # cuDNN algorithm choice from perturbing otherwise identical Taylor scores.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def configure_detection_loss(model: Any, checkpoint: Path) -> dict[str, Any]:
    """Restore the frozen baseline's training-loss hyperparameters.

    A serialized Ultralytics model keeps ``args`` as a plain dictionary.  Its
    loss implementation requires attribute access (``hyp.box``, ``hyp.cls``,
    and ``hyp.dfl``), so scoring must restore the exact baseline `args.yaml`
    configuration before the first call to ``model.loss``.
    """

    import yaml
    from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg

    args_path = checkpoint.parents[1] / "args.yaml"
    if not args_path.is_file():
        raise FileNotFoundError(f"Frozen baseline training arguments are missing: {args_path}")
    overrides = yaml.safe_load(args_path.read_text(encoding="utf-8"))
    if not isinstance(overrides, dict):
        raise RuntimeError(f"Expected a YAML mapping in {args_path}")
    model.args = get_cfg(DEFAULT_CFG_DICT, overrides=overrides)
    # Ensure an earlier caller cannot leave a criterion built against stale
    # serialized arguments.  The next model.loss call rebuilds it correctly.
    if hasattr(model, "criterion"):
        delattr(model, "criterion")
    return {
        "training_args": relative(args_path),
        "training_args_sha256": base.sha256(args_path),
        "box_gain": float(model.args.box),
        "cls_gain": float(model.args.cls),
        "dfl_gain": float(model.args.dfl),
    }


def normalize_rgp_log_element_sums(
    log_element_sums: dict[int, Any],
    batches_seen: int,
) -> tuple[dict[int, Any], list[float]]:
    """Normalize exact Equation-4 scores without materializing large exponentials.

    Each input tensor stores ``log(sum_b exp(weight * gradient_b))`` for every
    convolution weight element.  The original RGP aggregation first averages
    those scores over calibration batches and filter elements, then divides
    every filter by its layer mean.  Performing those reductions with
    log-sum-exp is mathematically equivalent but remains finite when a valid
    ``weight * gradient`` value is too large for a direct exponential.
    """

    import torch

    if batches_seen < 1:
        raise ValueError("batches_seen must be positive")
    normalized: dict[int, Any] = {}
    log_layer_means: list[float] = []
    for module_id, log_sums in log_element_sums.items():
        values = log_sums.detach().to(device="cpu", dtype=torch.float64)
        if values.ndim < 2 or values.shape[0] < 1:
            raise RuntimeError("RGP log accumulator has an invalid convolution shape")
        if not bool(torch.isfinite(values).all()):
            raise RuntimeError("RGP log accumulator contains a non-finite value")
        flattened = values.flatten(start_dim=1)
        elements_per_filter = int(flattened.shape[1])
        log_filters = torch.logsumexp(flattened, dim=1) - math.log(
            batches_seen * elements_per_filter
        )
        log_layer_mean = torch.logsumexp(log_filters, dim=0) - math.log(
            int(log_filters.numel())
        )
        scores = torch.exp(log_filters - log_layer_mean)
        if not bool(torch.isfinite(scores).all()) or not bool((scores > 0).all()):
            raise RuntimeError("RGP log-space filter normalization produced invalid scores")
        normalized[module_id] = scores.float().cpu()
        log_layer_means.append(float(log_layer_mean))
    if not normalized:
        raise RuntimeError("RGP log-space normalization received no convolution scores")
    return normalized, log_layer_means


def rgp_filter_scores(
    model: Any,
    checkpoint: Path,
    dataset_yaml: Path,
    max_batches: int,
) -> tuple[dict[int, Any], dict[str, Any]]:
    """Return normalized RGP scores keyed by the identity of each Conv2d module."""

    import torch

    if max_batches < 0:
        raise ValueError("max_batches cannot be negative")
    configure_reproducible_gradients()
    device = torch.device("cuda:0")
    loss_configuration = configure_detection_loss(model, checkpoint)
    model.float().to(device)
    # Ultralytics marks serialized inference checkpoints as frozen.  RGP needs
    # gradients with respect to the original weights, but does not perform an
    # optimizer update or mutate those weights.
    model.requires_grad_(True)
    model.train()
    batchnorm_layers = freeze_batchnorm_statistics(model)
    convolutions = [module for module in model.modules() if isinstance(module, torch.nn.Conv2d)]
    if not convolutions:
        raise RuntimeError("No Conv2d modules were found")
    loader, dataset_images = build_calibration_loader(dataset_yaml)
    log_sums = {
        id(module): torch.full_like(
            module.weight,
            -torch.inf,
            dtype=torch.float64,
            device="cpu",
        )
        for module in convolutions
    }
    batches_seen = 0
    images_seen = 0
    loss_sum = 0.0
    try:
        for batch in loader:
            if max_batches and batches_seen >= max_batches:
                break
            prepared = prepare_loss_batch(batch, device)
            model.zero_grad(set_to_none=True)
            result = model.loss(prepared)
            loss_components = result[0] if isinstance(result, tuple) else result
            if not isinstance(loss_components, torch.Tensor) or not bool(torch.isfinite(loss_components).all()):
                raise RuntimeError("RGP calibration produced non-finite loss components")

            # Ultralytics returns one loss value per component. Its own trainer
            # sums that vector before backpropagating, so use the same objective
            # for the Taylor-gradient calculation.
            loss = loss_components.sum()
            if loss.numel() != 1 or not bool(torch.isfinite(loss)):
                raise RuntimeError("RGP calibration produced a non-finite scalar loss")
            loss.backward()
            for module in convolutions:
                gradient = module.weight.grad
                if gradient is None:
                    raise RuntimeError("A convolution weight did not receive a loss gradient")
                product = module.weight.detach().float() * gradient.detach().float()
                if not bool(torch.isfinite(product).all()):
                    raise RuntimeError("RGP Equation 4 received a non-finite weight-gradient product")
                product_cpu = product.detach().to(device="cpu", dtype=torch.float64)
                log_sums[id(module)] = torch.logaddexp(log_sums[id(module)], product_cpu)
            batches_seen += 1
            images_seen += int(prepared["img"].shape[0])
            loss_sum += float(loss.detach().cpu())
        if batches_seen == 0:
            raise RuntimeError("RGP calibration did not receive any training batches")
    finally:
        model.zero_grad(set_to_none=True)
        del loader

    normalized, log_layer_means = normalize_rgp_log_element_sums(log_sums, batches_seen)
    model.eval().cpu()
    torch.cuda.empty_cache()
    return normalized, {
        "calibration_dataset_images": dataset_images,
        "calibration_batches_seen": batches_seen,
        "calibration_images_seen": images_seen,
        "mean_loss_across_batches": loss_sum / batches_seen,
        "batchnorm_layers_forced_eval": batchnorm_layers,
        "conv_layers_scored": len(convolutions),
        "raw_layer_filter_score_log_mean_min": min(log_layer_means),
        "raw_layer_filter_score_log_mean_max": max(log_layer_means),
        "raw_layer_filter_score_log_base": "e",
        "numerical_accumulation": "exact per-element batch log-sum-exp; filter and layer means normalized in log space",
        "loss_configuration": loss_configuration,
    }


def group_scores(model: Any, group_id: str, normalized_filters: dict[int, Any]) -> tuple[Any, str, Any | None]:
    """Map per-filter RGP scores to validated generic/custom pruning units."""

    import torch

    if group_id.startswith("CDG"):
        row = engine.custom_rows()[group_id]
        block_index = int(row["block_index"])
        block = model.model[block_index]
        values = normalized_filters.get(id(block.cv1.conv))
        if values is None:
            raise RuntimeError(f"{group_id}: missing cv1 RGP filter scores")
        c = int(block.c)
        if values.numel() != 2 * c:
            raise RuntimeError(f"{group_id}: custom cv1 width changed unexpectedly")
        if block_index not in (10, 22):
            return (values[:c] + values[c : 2 * c]) / 2.0, "paired_cv1_filter_score_mean", None
        if block_index == 10:
            _, layout = engine.c2psa_head_aware_importance(block)
        else:
            _, layout = engine.attention_c3k2_head_aware_importance(block)
        units: list[Any] = []
        for head in range(layout.num_heads):
            offset = head * layout.head_dim
            for position in range(layout.key_dim):
                units.append((values[offset + position] + values[offset + layout.key_dim + position]) / 2.0)
        score = torch.stack(units)
        if score.numel() != layout.total_units:
            raise RuntimeError(f"{group_id}: attention RGP unit count changed unexpectedly")
        return score, "paired_attention_cv1_filter_score_mean", layout

    row = engine.generic_rows()[group_id]
    root = base.find_module(model, row["representative_root"])
    values = normalized_filters.get(id(root))
    if values is None:
        raise RuntimeError(f"{group_id}: missing root RGP filter scores")
    # Cumulative structural replay can change a root's live width.  The
    # catalogue width is the baseline reference, not a replay-time invariant.
    if values.numel() != int(root.out_channels):
        raise RuntimeError(f"{group_id}: RGP root score width does not match the live root")
    return values, "root_filter_score", None


def select_units(scores: Any, group_id: str, attention_layout: Any | None) -> tuple[list[int], list[int], str, int]:
    import torch

    values = scores.detach().float().cpu().flatten()
    if not bool(torch.isfinite(values).all()):
        raise RuntimeError(f"{group_id}: non-finite RGP unit score")
    if attention_layout is not None:
        selected = engine.balanced_attention_indices(values, attention_layout, LOCAL_FRACTION)
        rank_order = sorted(selected, key=lambda index: (float(values[index]), index))
        return sorted(selected), rank_order, "paired_attention_unit", int(attention_layout.total_units)
    remove_count = exact_count(int(values.numel()), f"{group_id} selectable units")
    rank_order = [int(value) for value in torch.argsort(values, stable=True)[:remove_count].tolist()]
    selection_unit = "hidden_channel" if group_id.startswith("CDG") else "output_channel"
    return sorted(rank_order), rank_order, selection_unit, int(values.numel())


def score_all_groups(
    model: Any,
    normalized_filters: dict[int, Any],
    domain: str,
    group_ids: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    known_groups = all_group_ids()
    requested_groups = known_groups if group_ids is None else list(group_ids)
    if len(requested_groups) != len(set(requested_groups)):
        raise ValueError("RGP scoring group_ids must be unique")
    unknown_groups = sorted(set(requested_groups) - set(known_groups))
    if unknown_groups:
        raise ValueError(f"Unknown RGP scoring groups: {unknown_groups}")

    rows: list[dict[str, Any]] = []
    masks: dict[str, Any] = {}
    for group_id in requested_groups:
        scores, basis, attention_layout = group_scores(model, group_id, normalized_filters)
        selected, rank_order, unit, width = select_units(scores, group_id, attention_layout)
        values = scores.detach().float().cpu().flatten()
        root_width = (
            int(model.model[int(engine.custom_rows()[group_id]["block_index"])].c)
            if group_id.startswith("CDG")
            else width
        )
        result = {
            "group_kind": "CUSTOM" if group_id.startswith("CDG") else "GENERIC",
            "selection_unit": unit,
            "score_basis": basis,
            "root_channels_before": root_width,
            "selectable_units_before": width,
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
            "selection_unit": result["selection_unit"],
            "score_basis": result["score_basis"],
            "root_channels_before": result["root_channels_before"],
            "selectable_units_before": result["selectable_units_before"],
            "selected_unit_count": len(selected),
            "prune_first_mean_score": result["prune_first_mean_score"],
            "prune_first_max_score": result["prune_first_max_score"],
            "all_unit_score_mean": result["all_unit_score_mean"],
            "all_unit_score_std": result["all_unit_score_std"],
        })
    rank_rows(rows, "prune_first_mean_score")
    return rows, masks


def run(domain: str, calibration_batches: int) -> int:
    from ultralytics import YOLO

    root = output_root(domain, calibration_batches)
    manifest_path = root / "ranking_manifest.json"
    if manifest_path.is_file():
        raise RuntimeError(f"RGP ranking already exists: {root}. Create a new versioned output rather than overwriting it.")
    evidence = preflight(domain, calibration_batches)
    manifest = {
        **evidence,
        "schema": "bdd_rgp_taylor_gradation_ranking_manifest_v1",
        "status": "RUNNING",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "script": relative(SELF),
        "script_sha256": base.sha256(SELF),
        "policy": {
            "ranking_only": True,
            "domain_specific": True,
            "training_split_used": True,
            "validation_data_used": False,
            "test_data_used": False,
            "fine_tuning": False,
            "optimizer_updates": False,
            "batchnorm_running_stats_updated": False,
            "structural_pruning_performed": False,
            "same_51_legal_groups_as_proposed_method": True,
            "local_rgp_score": "per-filter Equation-4 score, mean-normalized within each Conv2d layer",
            "group_priority": "ascending mean score of the locally selected 37.5% validated structural units; ties by group_id",
        },
    }
    atomic_json(manifest_path, manifest)
    try:
        config = bdd.domain_config(domain)
        checkpoint = PROJECT_ROOT / config["path"]
        dataset_yaml = PROJECT_ROOT / config["dataset_yaml"]
        model = YOLO(str(checkpoint), task="detect").model.float().eval()
        normalized, calibration = rgp_filter_scores(model, checkpoint, dataset_yaml, calibration_batches)
        rows, masks = score_all_groups(model, normalized, domain)
        queue = [row for row in rows if row["eligible_for_cumulative_raw"]]
        queue.sort(key=lambda row: int(row["domain_rank_prune_first_ascending"]))
        for index, row in enumerate(queue, start=1):
            row["gradation_queue_index"] = index
        for row in rows:
            row.setdefault("gradation_queue_index", "")
        fields = list(rows[0])
        atomic_csv(root / "tables" / f"{domain}_RGP_TAYLOR_LOCAL_FILTER_SALIENCY_37_5PCT.csv", fields, rows)
        atomic_json(root / "masks" / f"{domain}_rgp_taylor_37_5pct_masks.json", {
            "schema": "bdd_rgp_taylor_domain_masks_v1",
            "domain": domain,
            "local_pruning_percent": LOCAL_PERCENT,
            "baseline_checkpoint": config["path"],
            "baseline_checkpoint_sha256": config["sha256"],
            "masks": masks,
        })
        atomic_json(root / "gradation_queue.json", {
            "schema": "bdd_rgp_taylor_gradation_queue_v1",
            "domain": domain,
            "ranking_policy": "ascending mean score of the locally selected 37.5% RGP-normalized structural units; ties by group_id",
            "protected_groups": PROTECTED_GROUPS,
            "entries": queue,
        })
        readme = (
            f"# BDD {domain} RGP Taylor/gradation ranking\n\n"
            "This is a domain-specific, ranking-only, equation-based RGP adaptation. It uses only the "
            "frozen training split to calculate loss gradients; it does not evaluate validation/test data, prune, "
            "update BatchNorm statistics, or train. The advertised RGP source repository was unavailable when this "
            "artifact was generated. The implementation follows the final displayed Equation 4: exp(weight * dL/dweight).\n"
        )
        (root / "README.md").write_text(readme, encoding="utf-8")
        manifest.update({
            "status": "PASS",
            "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "calibration": calibration,
            "ranking_table": relative(root / "tables" / f"{domain}_RGP_TAYLOR_LOCAL_FILTER_SALIENCY_37_5PCT.csv"),
            "mask_file": relative(root / "masks" / f"{domain}_rgp_taylor_37_5pct_masks.json"),
            "gradation_queue": relative(root / "gradation_queue.json"),
            "candidate_count": len(queue),
        })
        atomic_json(manifest_path, manifest)
        print(json.dumps({"status": "PASS", "domain": domain, "output": relative(root), "candidate_count": len(queue)}, sort_keys=True))
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
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument(
        "--calibration-batches",
        type=int,
        default=0,
        help="Use this many deterministic training batches; default 0 consumes every training batch once.",
    )
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.calibration_batches < 0:
        parser.error("--calibration-batches cannot be negative")
    if args.preflight:
        print(json.dumps(preflight(args.domain, args.calibration_batches), indent=2, sort_keys=True))
        return 0
    return run(args.domain, args.calibration_batches)


if __name__ == "__main__":
    raise SystemExit(main())
