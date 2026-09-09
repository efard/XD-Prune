"""Reusable logical-channel pruning rules for stock YOLO26 blocks.

This module extends the generic DepGraph evidence rather than replacing it.
The first implemented family is non-attention C3k2.  One logical hidden
channel is mapped to every physical tensor position that must be removed from
the split, residual and concatenation paths while preserving the block's
external input and output widths.

C2PSA and attention-bearing C3k2 are deliberately rejected until their packed
Q/K/V and static attention constraints have their own validated rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch_pruning as tp


@dataclass(frozen=True)
class PrimitiveOperation:
    """One physically applied low-level pruning operation."""

    module_path: str
    operation: str
    indices: tuple[int, ...]
    channels_before: int
    channels_after: int


@dataclass(frozen=True)
class C3k2PruningResult:
    """Auditable result of one logical-width intervention."""

    family: str
    hidden_channels_before: int
    hidden_channels_removed: int
    hidden_channels_after: int
    logical_indices: tuple[int, ...]
    operations: tuple[PrimitiveOperation, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["operations"] = [asdict(operation) for operation in self.operations]
        return value


@dataclass(frozen=True)
class HeadAwarePruningResult:
    """Auditable result of a C2PSA or attention-C3k2 intervention."""

    family: str
    hidden_channels_before: int
    hidden_channels_removed: int
    hidden_channels_after: int
    attention_units_before: int
    attention_units_removed: int
    selected_attention_units: tuple[int, ...]
    embedding_indices: tuple[int, ...]
    qkv_output_indices: tuple[int, ...]
    operations: tuple[PrimitiveOperation, ...]

    def to_dict(self) -> dict:
        value = asdict(self)
        value["operations"] = [asdict(operation) for operation in self.operations]
        return value


@dataclass(frozen=True)
class AttentionUnitLayout:
    """Stock YOLO attention layout used for balanced logical-width pruning."""

    channels: int
    num_heads: int
    head_dim: int
    key_dim: int

    @property
    def units_per_head(self) -> int:
        return self.key_dim

    @property
    def total_units(self) -> int:
        return self.num_heads * self.key_dim

    @property
    def embedding_channels_per_unit(self) -> int:
        return self.head_dim // self.key_dim


def _unique_indices(indices: Iterable[int], width: int, label: str) -> list[int]:
    selected = sorted({int(index) for index in indices})
    if not selected:
        raise ValueError(f"{label}: at least one channel must be selected")
    invalid = [index for index in selected if not 0 <= index < width]
    if invalid:
        raise IndexError(f"{label}: indices outside [0, {width}): {invalid}")
    if len(selected) >= width:
        raise ValueError(f"{label}: pruning would remove every channel")
    return selected


def _normalized(vector: torch.Tensor, label: str) -> torch.Tensor:
    vector = vector.detach().float().cpu().flatten()
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label}: non-finite importance contribution")
    mean = vector.mean()
    if float(mean) <= 1e-12:
        return vector
    return vector / mean


def _conv_out_score(conv: nn.Conv2d, indices: Sequence[int] | None = None) -> torch.Tensor:
    values = conv.weight.detach().abs().mean(dim=(1, 2, 3))
    return values if indices is None else values[list(indices)]


def _conv_in_score(conv: nn.Conv2d, indices: Sequence[int] | None = None) -> torch.Tensor:
    if conv.groups != 1:
        raise ValueError("Logical C3k2 rule currently supports only groups=1 input pruning")
    values = conv.weight.detach().abs().mean(dim=(0, 2, 3))
    return values if indices is None else values[list(indices)]


def _bn_score(batchnorm: nn.BatchNorm2d, indices: Sequence[int] | None = None) -> torch.Tensor:
    values = batchnorm.weight.detach().abs()
    return values if indices is None else values[list(indices)]


def _module_kind(module: nn.Module) -> str:
    return module.__class__.__name__


def _validate_supported_nonattention_c3k2(block: nn.Module) -> None:
    if _module_kind(block) != "C3k2":
        raise TypeError(f"Expected C3k2, received {_module_kind(block)}")
    if not hasattr(block, "c") or not hasattr(block, "cv1") or not hasattr(block, "cv2"):
        raise TypeError("C3k2 block is missing c/cv1/cv2 attributes")
    if not hasattr(block, "m") or len(block.m) < 1:
        raise TypeError("C3k2 block has no internal modules")
    unsupported = [
        module.__class__.__name__
        for module in block.modules()
        if module.__class__.__name__ in {"PSABlock", "Attention"}
    ]
    if unsupported:
        raise TypeError("Attention-bearing C3k2 requires the separate head-aware rule")
    for child in block.m:
        if _module_kind(child) not in {"Bottleneck", "C3k"}:
            raise TypeError(f"Unsupported C3k2 child type: {_module_kind(child)}")


def nonattention_c3k2_logical_importance(block: nn.Module) -> torch.Tensor:
    """Return one normalized group-aware L1 score per logical hidden channel.

    Every coupled physical slice contributes one vector.  Each contribution is
    mean-normalized before group averaging, matching the scale-balancing intent
    of ``GroupMagnitudeImportance(... normalizer='mean', group_reduction='mean')``.
    """

    _validate_supported_nonattention_c3k2(block)
    c = int(block.c)
    n = len(block.m)
    if block.cv1.conv.out_channels != 2 * c:
        raise ValueError("C3k2 cv1 output is not exactly two logical-width branches")
    if block.cv2.conv.in_channels != (2 + n) * c:
        raise ValueError("C3k2 cv2 input does not match the concatenated logical-width branches")

    contributions: list[tuple[str, torch.Tensor]] = []

    def add(label: str, vector: torch.Tensor) -> None:
        if vector.numel() != c:
            raise ValueError(f"{label}: expected {c} scores, received {vector.numel()}")
        contributions.append((label, _normalized(vector, label)))

    first = list(range(c))
    second = list(range(c, 2 * c))
    add("cv1.conv.out.first", _conv_out_score(block.cv1.conv, first))
    add("cv1.conv.out.second", _conv_out_score(block.cv1.conv, second))
    add("cv1.bn.first", _bn_score(block.cv1.bn, first))
    add("cv1.bn.second", _bn_score(block.cv1.bn, second))

    for segment in range(2 + n):
        positions = list(range(segment * c, (segment + 1) * c))
        add(f"cv2.conv.in.segment{segment}", _conv_in_score(block.cv2.conv, positions))

    for child_index, child in enumerate(block.m):
        kind = _module_kind(child)
        if kind == "Bottleneck":
            add(f"m.{child_index}.cv1.conv.in", _conv_in_score(child.cv1.conv))
            add(f"m.{child_index}.cv2.conv.out", _conv_out_score(child.cv2.conv))
            add(f"m.{child_index}.cv2.bn", _bn_score(child.cv2.bn))
        elif kind == "C3k":
            add(f"m.{child_index}.cv1.conv.in", _conv_in_score(child.cv1.conv))
            add(f"m.{child_index}.cv2.conv.in", _conv_in_score(child.cv2.conv))
            add(f"m.{child_index}.cv3.conv.out", _conv_out_score(child.cv3.conv))
            add(f"m.{child_index}.cv3.bn", _bn_score(child.cv3.bn))
        else:  # guarded above; retained as a defensive assertion
            raise AssertionError(f"Unexpected C3k2 child: {kind}")

    stacked = torch.stack([vector for _, vector in contributions], dim=0)
    score = stacked.mean(dim=0)
    if score.numel() != c or not bool(torch.isfinite(score).all()):
        raise RuntimeError("Invalid logical C3k2 importance vector")
    return score


def _prune_conv_out(
    wrapper: nn.Module,
    indices: Sequence[int],
    path: str,
    operations: list[PrimitiveOperation],
) -> None:
    before = int(wrapper.conv.out_channels)
    selected = _unique_indices(indices, before, f"{path}.conv output")
    tp.prune_conv_out_channels(wrapper.conv, selected)
    if not hasattr(wrapper, "bn") or not isinstance(wrapper.bn, nn.BatchNorm2d):
        raise TypeError(f"{path}: expected an attached BatchNorm2d")
    tp.prune_batchnorm_out_channels(wrapper.bn, selected)
    after = int(wrapper.conv.out_channels)
    operations.append(PrimitiveOperation(f"{path}.conv", "prune_out_channels", tuple(selected), before, after))
    operations.append(PrimitiveOperation(f"{path}.bn", "prune_out_channels", tuple(selected), before, after))


def _prune_conv_in(
    wrapper: nn.Module,
    indices: Sequence[int],
    path: str,
    operations: list[PrimitiveOperation],
) -> None:
    before = int(wrapper.conv.in_channels)
    selected = _unique_indices(indices, before, f"{path}.conv input")
    tp.prune_conv_in_channels(wrapper.conv, selected)
    after = int(wrapper.conv.in_channels)
    operations.append(PrimitiveOperation(f"{path}.conv", "prune_in_channels", tuple(selected), before, after))


def validate_nonattention_c3k2_invariants(block: nn.Module) -> dict[str, int | str]:
    """Raise on any violated structural invariant and return a compact summary."""

    _validate_supported_nonattention_c3k2(block)
    c = int(block.c)
    n = len(block.m)
    if c <= 0:
        raise ValueError("C3k2 hidden width must remain positive")
    if block.cv1.conv.out_channels != 2 * c or block.cv1.bn.num_features != 2 * c:
        raise ValueError("C3k2 paired split width is inconsistent")
    if block.cv2.conv.in_channels != (2 + n) * c:
        raise ValueError("C3k2 concatenation width is inconsistent")

    for child_index, child in enumerate(block.m):
        kind = _module_kind(child)
        if kind == "Bottleneck":
            if child.cv1.conv.in_channels != c:
                raise ValueError(f"m.{child_index} Bottleneck input width is inconsistent")
            if child.cv2.conv.out_channels != c or child.cv2.bn.num_features != c:
                raise ValueError(f"m.{child_index} Bottleneck residual output width is inconsistent")
            if bool(child.add) and child.cv1.conv.in_channels != child.cv2.conv.out_channels:
                raise ValueError(f"m.{child_index} Bottleneck residual operands differ")
        elif kind == "C3k":
            if child.cv1.conv.in_channels != c or child.cv2.conv.in_channels != c:
                raise ValueError(f"m.{child_index} C3k branch input widths are inconsistent")
            if child.cv3.conv.out_channels != c or child.cv3.bn.num_features != c:
                raise ValueError(f"m.{child_index} C3k output width is inconsistent")
        else:
            raise AssertionError(f"Unexpected child type: {kind}")

    return {
        "family": "C3k2_NONATTENTION_LOGICAL_WIDTH_V1",
        "hidden_channels": c,
        "split_output_channels": int(block.cv1.conv.out_channels),
        "concat_input_channels": int(block.cv2.conv.in_channels),
        "internal_blocks": n,
    }


def prune_nonattention_c3k2_logical_channels(
    block: nn.Module,
    logical_indices: Sequence[int],
    module_path: str,
) -> C3k2PruningResult:
    """Physically remove selected logical hidden channels from one C3k2 block."""

    _validate_supported_nonattention_c3k2(block)
    c = int(block.c)
    n = len(block.m)
    selected = _unique_indices(logical_indices, c, f"{module_path} logical width")
    operations: list[PrimitiveOperation] = []

    paired_split_indices = selected + [c + index for index in selected]
    concat_indices = [segment * c + index for segment in range(2 + n) for index in selected]

    _prune_conv_out(block.cv1, paired_split_indices, f"{module_path}.cv1", operations)
    _prune_conv_in(block.cv2, concat_indices, f"{module_path}.cv2", operations)

    for child_index, child in enumerate(block.m):
        child_path = f"{module_path}.m.{child_index}"
        kind = _module_kind(child)
        if kind == "Bottleneck":
            _prune_conv_in(child.cv1, selected, f"{child_path}.cv1", operations)
            _prune_conv_out(child.cv2, selected, f"{child_path}.cv2", operations)
        elif kind == "C3k":
            _prune_conv_in(child.cv1, selected, f"{child_path}.cv1", operations)
            _prune_conv_in(child.cv2, selected, f"{child_path}.cv2", operations)
            _prune_conv_out(child.cv3, selected, f"{child_path}.cv3", operations)
        else:
            raise AssertionError(f"Unexpected child type: {kind}")

    block.c = c - len(selected)
    validate_nonattention_c3k2_invariants(block)
    return C3k2PruningResult(
        family="C3k2_NONATTENTION_LOGICAL_WIDTH_V1",
        hidden_channels_before=c,
        hidden_channels_removed=len(selected),
        hidden_channels_after=int(block.c),
        logical_indices=tuple(selected),
        operations=tuple(operations),
    )


def _attention_layout(attention: nn.Module) -> AttentionUnitLayout:
    required = ("num_heads", "head_dim", "key_dim", "qkv", "proj", "pe")
    if _module_kind(attention) != "Attention" or not all(hasattr(attention, name) for name in required):
        raise TypeError("Expected a stock Ultralytics Attention module")
    layout = AttentionUnitLayout(
        channels=int(attention.num_heads * attention.head_dim),
        num_heads=int(attention.num_heads),
        head_dim=int(attention.head_dim),
        key_dim=int(attention.key_dim),
    )
    if layout.num_heads < 1 or layout.key_dim < 1:
        raise ValueError("Attention head dimensions must be positive")
    if layout.head_dim != 2 * layout.key_dim:
        raise ValueError("V1 head-aware rule requires the stock 0.5 attention ratio")
    if attention.qkv.conv.in_channels != layout.channels:
        raise ValueError("Attention qkv input width differs from the stored head layout")
    expected_qkv = layout.num_heads * (2 * layout.key_dim + layout.head_dim)
    if attention.qkv.conv.out_channels != expected_qkv:
        raise ValueError("Attention qkv output width differs from the packed head layout")
    return layout


def _unit_embedding_indices(layout: AttentionUnitLayout, units: Sequence[int]) -> list[int]:
    selected = _unique_indices(units, layout.total_units, "attention logical units")
    indices: list[int] = []
    for unit in selected:
        head, position = divmod(unit, layout.key_dim)
        base = head * layout.head_dim
        indices.extend((base + position, base + layout.key_dim + position))
    return sorted(indices)


def _unit_qkv_output_indices(layout: AttentionUnitLayout, units: Sequence[int]) -> list[int]:
    selected = _unique_indices(units, layout.total_units, "attention logical units")
    per_head_packed = 2 * layout.key_dim + layout.head_dim
    indices: list[int] = []
    for unit in selected:
        head, position = divmod(unit, layout.key_dim)
        base = head * per_head_packed
        indices.extend(
            (
                base + position,
                base + layout.key_dim + position,
                base + 2 * layout.key_dim + position,
                base + 3 * layout.key_dim + position,
            )
        )
    return sorted(indices)


def _embedding_vector_to_units(vector: torch.Tensor, layout: AttentionUnitLayout, label: str) -> torch.Tensor:
    vector = vector.detach().float().cpu().flatten()
    if vector.numel() != layout.channels:
        raise ValueError(f"{label}: expected {layout.channels} embedding scores, received {vector.numel()}")
    values: list[torch.Tensor] = []
    for head in range(layout.num_heads):
        base = head * layout.head_dim
        for position in range(layout.key_dim):
            values.append(torch.stack((vector[base + position], vector[base + layout.key_dim + position])).mean())
    return torch.stack(values)


def _qkv_vector_to_units(vector: torch.Tensor, layout: AttentionUnitLayout, label: str) -> torch.Tensor:
    vector = vector.detach().float().cpu().flatten()
    expected = layout.num_heads * (2 * layout.key_dim + layout.head_dim)
    if vector.numel() != expected:
        raise ValueError(f"{label}: expected {expected} packed qkv scores, received {vector.numel()}")
    per_head_packed = 2 * layout.key_dim + layout.head_dim
    values: list[torch.Tensor] = []
    for head in range(layout.num_heads):
        base = head * per_head_packed
        for position in range(layout.key_dim):
            values.append(
                torch.stack(
                    (
                        vector[base + position],
                        vector[base + layout.key_dim + position],
                        vector[base + 2 * layout.key_dim + position],
                        vector[base + 3 * layout.key_dim + position],
                    )
                ).mean()
            )
    return torch.stack(values)


def _psablock_unit_contributions(psa: nn.Module, prefix: str) -> tuple[AttentionUnitLayout, list[tuple[str, torch.Tensor]]]:
    if _module_kind(psa) != "PSABlock" or not hasattr(psa, "attn") or not hasattr(psa, "ffn"):
        raise TypeError(f"{prefix}: expected a stock PSABlock")
    layout = _attention_layout(psa.attn)
    attention = psa.attn
    if len(psa.ffn) != 2:
        raise ValueError(f"{prefix}: expected a two-layer feed-forward network")
    contributions: list[tuple[str, torch.Tensor]] = []

    def embedding(label: str, vector: torch.Tensor) -> None:
        contributions.append((label, _embedding_vector_to_units(vector, layout, label)))

    def packed(label: str, vector: torch.Tensor) -> None:
        contributions.append((label, _qkv_vector_to_units(vector, layout, label)))

    embedding(f"{prefix}.attn.qkv.conv.in", _conv_in_score(attention.qkv.conv))
    packed(f"{prefix}.attn.qkv.conv.out", _conv_out_score(attention.qkv.conv))
    packed(f"{prefix}.attn.qkv.bn", _bn_score(attention.qkv.bn))
    embedding(f"{prefix}.attn.proj.conv.in", _conv_in_score(attention.proj.conv))
    embedding(f"{prefix}.attn.proj.conv.out", _conv_out_score(attention.proj.conv))
    embedding(f"{prefix}.attn.proj.bn", _bn_score(attention.proj.bn))
    embedding(f"{prefix}.attn.pe.conv", _conv_out_score(attention.pe.conv))
    embedding(f"{prefix}.attn.pe.bn", _bn_score(attention.pe.bn))
    embedding(f"{prefix}.ffn.0.conv.in", _conv_in_score(psa.ffn[0].conv))
    embedding(f"{prefix}.ffn.1.conv.out", _conv_out_score(psa.ffn[1].conv))
    embedding(f"{prefix}.ffn.1.bn", _bn_score(psa.ffn[1].bn))
    return layout, contributions


def _mean_normalized_contributions(
    contributions: Sequence[tuple[str, torch.Tensor]], expected: int
) -> torch.Tensor:
    if not contributions:
        raise ValueError("No importance contributions were provided")
    normalized: list[torch.Tensor] = []
    for label, vector in contributions:
        if vector.numel() != expected:
            raise ValueError(f"{label}: expected {expected} scores, received {vector.numel()}")
        normalized.append(_normalized(vector, label))
    score = torch.stack(normalized, dim=0).mean(dim=0)
    if score.numel() != expected or not bool(torch.isfinite(score).all()):
        raise RuntimeError("Invalid head-aware logical importance vector")
    return score


def select_balanced_attention_units(
    scores: torch.Tensor, layout: AttentionUnitLayout, fraction: float = 0.125
) -> list[int]:
    """Select equal numbers of logical units from every attention head."""

    if abs(fraction - 0.125) > 1e-12:
        raise ValueError("V1 validation freezes the attention-unit fraction at one eighth")
    if scores.numel() != layout.total_units:
        raise ValueError("Attention score count differs from the head layout")
    remove_per_head = layout.key_dim // 8
    if remove_per_head < 1 or layout.key_dim % 8:
        raise ValueError("One-eighth pruning is not integral within each attention head")
    selected: list[int] = []
    for head in range(layout.num_heads):
        offset = head * layout.key_dim
        local = scores[offset : offset + layout.key_dim]
        local_rank = torch.argsort(local, stable=True)[:remove_per_head].tolist()
        selected.extend(offset + int(position) for position in local_rank)
    return sorted(selected)


def c2psa_head_aware_importance(block: nn.Module) -> tuple[torch.Tensor, AttentionUnitLayout]:
    """Return group-aware scores for stock C2PSA paired head units."""

    if _module_kind(block) != "C2PSA" or not hasattr(block, "c") or not hasattr(block, "m"):
        raise TypeError("Expected a stock C2PSA block")
    c = int(block.c)
    if block.cv1.conv.out_channels != 2 * c or block.cv2.conv.in_channels != 2 * c:
        raise ValueError("C2PSA paired split/concatenation widths are inconsistent")
    layouts: list[AttentionUnitLayout] = []
    contributions: list[tuple[str, torch.Tensor]] = []
    for index, psa in enumerate(block.m):
        layout, child_contributions = _psablock_unit_contributions(psa, f"m.{index}")
        layouts.append(layout)
        contributions.extend(child_contributions)
    if not layouts or any(layout != layouts[0] for layout in layouts):
        raise ValueError("C2PSA attention blocks do not share one head layout")
    layout = layouts[0]
    if layout.channels != c:
        raise ValueError("C2PSA hidden width differs from its attention width")

    for label, vector in (
        ("cv1.conv.out.first", _conv_out_score(block.cv1.conv, range(c))),
        ("cv1.conv.out.second", _conv_out_score(block.cv1.conv, range(c, 2 * c))),
        ("cv1.bn.first", _bn_score(block.cv1.bn, range(c))),
        ("cv1.bn.second", _bn_score(block.cv1.bn, range(c, 2 * c))),
        ("cv2.conv.in.first", _conv_in_score(block.cv2.conv, range(c))),
        ("cv2.conv.in.second", _conv_in_score(block.cv2.conv, range(c, 2 * c))),
    ):
        contributions.append((label, _embedding_vector_to_units(vector, layout, label)))
    return _mean_normalized_contributions(contributions, layout.total_units), layout


def attention_c3k2_head_aware_importance(block: nn.Module) -> tuple[torch.Tensor, AttentionUnitLayout]:
    """Return group-aware scores for stock attention-bearing C3k2 paired units."""

    if _module_kind(block) != "C3k2" or not hasattr(block, "c") or len(block.m) < 1:
        raise TypeError("Expected an attention-bearing stock C3k2 block")
    c = int(block.c)
    n = len(block.m)
    if block.cv1.conv.out_channels != 2 * c or block.cv2.conv.in_channels != (2 + n) * c:
        raise ValueError("Attention C3k2 split/concatenation widths are inconsistent")

    contributions: list[tuple[str, torch.Tensor]] = []
    layouts: list[AttentionUnitLayout] = []
    for child_index, child in enumerate(block.m):
        if not isinstance(child, nn.Sequential) or len(child) != 2:
            raise TypeError("Attention C3k2 V1 expects Bottleneck + PSABlock sequences")
        bottleneck, psa = child[0], child[1]
        if _module_kind(bottleneck) != "Bottleneck":
            raise TypeError("Attention C3k2 sequence does not begin with Bottleneck")
        layout, psa_contributions = _psablock_unit_contributions(psa, f"m.{child_index}.1")
        layouts.append(layout)
        contributions.extend(psa_contributions)
        for label, vector in (
            (f"m.{child_index}.0.cv1.conv.in", _conv_in_score(bottleneck.cv1.conv)),
            (f"m.{child_index}.0.cv2.conv.out", _conv_out_score(bottleneck.cv2.conv)),
            (f"m.{child_index}.0.cv2.bn", _bn_score(bottleneck.cv2.bn)),
        ):
            contributions.append((label, _embedding_vector_to_units(vector, layout, label)))
    if not layouts or any(layout != layouts[0] for layout in layouts):
        raise ValueError("Attention C3k2 sequences do not share one head layout")
    layout = layouts[0]
    if layout.channels != c:
        raise ValueError("Attention C3k2 hidden width differs from its attention width")

    outer: list[tuple[str, torch.Tensor]] = [
        ("cv1.conv.out.first", _conv_out_score(block.cv1.conv, range(c))),
        ("cv1.conv.out.second", _conv_out_score(block.cv1.conv, range(c, 2 * c))),
        ("cv1.bn.first", _bn_score(block.cv1.bn, range(c))),
        ("cv1.bn.second", _bn_score(block.cv1.bn, range(c, 2 * c))),
    ]
    for segment in range(2 + n):
        outer.append(
            (
                f"cv2.conv.in.segment{segment}",
                _conv_in_score(block.cv2.conv, range(segment * c, (segment + 1) * c)),
            )
        )
    for label, vector in outer:
        contributions.append((label, _embedding_vector_to_units(vector, layout, label)))
    return _mean_normalized_contributions(contributions, layout.total_units), layout


def _prune_depthwise_out(
    wrapper: nn.Module,
    indices: Sequence[int],
    path: str,
    operations: list[PrimitiveOperation],
) -> None:
    conv = wrapper.conv
    before = int(conv.out_channels)
    if conv.in_channels != before or conv.groups != before:
        raise TypeError(f"{path}: expected a depthwise convolution")
    selected = _unique_indices(indices, before, f"{path}.conv depthwise channels")
    tp.prune_depthwise_conv_out_channels(conv, selected)
    tp.prune_batchnorm_out_channels(wrapper.bn, selected)
    after = int(conv.out_channels)
    operations.append(PrimitiveOperation(f"{path}.conv", "prune_depthwise_channels", tuple(selected), before, after))
    operations.append(PrimitiveOperation(f"{path}.bn", "prune_out_channels", tuple(selected), before, after))


def _prune_psablock_width(
    psa: nn.Module,
    layout: AttentionUnitLayout,
    selected_units: Sequence[int],
    path: str,
    operations: list[PrimitiveOperation],
) -> tuple[list[int], list[int]]:
    live_layout = _attention_layout(psa.attn)
    if live_layout != layout:
        raise ValueError(f"{path}: attention layout changed before pruning")
    embedding_indices = _unit_embedding_indices(layout, selected_units)
    qkv_indices = _unit_qkv_output_indices(layout, selected_units)
    attention = psa.attn

    _prune_conv_in(attention.qkv, embedding_indices, f"{path}.attn.qkv", operations)
    _prune_conv_out(attention.qkv, qkv_indices, f"{path}.attn.qkv", operations)
    _prune_conv_in(attention.proj, embedding_indices, f"{path}.attn.proj", operations)
    _prune_conv_out(attention.proj, embedding_indices, f"{path}.attn.proj", operations)
    _prune_depthwise_out(attention.pe, embedding_indices, f"{path}.attn.pe", operations)
    _prune_conv_in(psa.ffn[0], embedding_indices, f"{path}.ffn.0", operations)
    _prune_conv_out(psa.ffn[1], embedding_indices, f"{path}.ffn.1", operations)

    channels_after = layout.channels - len(embedding_indices)
    if channels_after % layout.num_heads:
        raise ValueError(f"{path}: remaining width is not divisible by the head count")
    attention.head_dim = channels_after // layout.num_heads
    attention.key_dim = attention.head_dim // 2
    attention.scale = attention.key_dim**-0.5
    return embedding_indices, qkv_indices


def _validate_psablock_width(psa: nn.Module, channels: int, path: str) -> dict[str, int]:
    layout = _attention_layout(psa.attn)
    attention = psa.attn
    if layout.channels != channels:
        raise ValueError(f"{path}: stored attention layout differs from the residual width")
    expected_qkv = layout.num_heads * (2 * layout.key_dim + layout.head_dim)
    checks = {
        "qkv_in": attention.qkv.conv.in_channels,
        "qkv_out": attention.qkv.conv.out_channels,
        "proj_in": attention.proj.conv.in_channels,
        "proj_out": attention.proj.conv.out_channels,
        "pe_in": attention.pe.conv.in_channels,
        "pe_out": attention.pe.conv.out_channels,
        "pe_groups": attention.pe.conv.groups,
        "ffn_in": psa.ffn[0].conv.in_channels,
        "ffn_out": psa.ffn[1].conv.out_channels,
    }
    expected = {
        "qkv_in": channels,
        "qkv_out": expected_qkv,
        "proj_in": channels,
        "proj_out": channels,
        "pe_in": channels,
        "pe_out": channels,
        "pe_groups": channels,
        "ffn_in": channels,
        "ffn_out": channels,
    }
    if checks != expected:
        raise ValueError(f"{path}: attention/FFN invariant mismatch: {checks} != {expected}")
    return {
        "channels": channels,
        "num_heads": layout.num_heads,
        "head_dim": layout.head_dim,
        "key_dim": layout.key_dim,
        "qkv_channels": expected_qkv,
    }


def validate_c2psa_invariants(block: nn.Module) -> dict:
    if _module_kind(block) != "C2PSA":
        raise TypeError("Expected C2PSA")
    c = int(block.c)
    if block.cv1.conv.out_channels != 2 * c or block.cv1.bn.num_features != 2 * c:
        raise ValueError("C2PSA split width is inconsistent")
    if block.cv2.conv.in_channels != 2 * c:
        raise ValueError("C2PSA concatenation width is inconsistent")
    children = [_validate_psablock_width(psa, c, f"m.{index}") for index, psa in enumerate(block.m)]
    return {"family": "C2PSA_HEAD_AWARE_LOGICAL_WIDTH_V1", "hidden_channels": c, "children": children}


def validate_attention_c3k2_invariants(block: nn.Module) -> dict:
    if _module_kind(block) != "C3k2":
        raise TypeError("Expected attention-bearing C3k2")
    c = int(block.c)
    n = len(block.m)
    if block.cv1.conv.out_channels != 2 * c or block.cv1.bn.num_features != 2 * c:
        raise ValueError("Attention C3k2 split width is inconsistent")
    if block.cv2.conv.in_channels != (2 + n) * c:
        raise ValueError("Attention C3k2 concatenation width is inconsistent")
    children: list[dict] = []
    for index, child in enumerate(block.m):
        if not isinstance(child, nn.Sequential) or len(child) != 2:
            raise TypeError("Attention C3k2 V1 expects Bottleneck + PSABlock sequences")
        bottleneck, psa = child[0], child[1]
        if bottleneck.cv1.conv.in_channels != c or bottleneck.cv2.conv.out_channels != c:
            raise ValueError(f"m.{index}.0 Bottleneck residual width is inconsistent")
        children.append(_validate_psablock_width(psa, c, f"m.{index}.1"))
    return {"family": "C3K2_ATTENTION_HEAD_AWARE_LOGICAL_WIDTH_V1", "hidden_channels": c, "children": children}


def prune_c2psa_head_aware_units(
    block: nn.Module, selected_units: Sequence[int], module_path: str
) -> HeadAwarePruningResult:
    scores, layout = c2psa_head_aware_importance(block)
    del scores
    selected = _unique_indices(selected_units, layout.total_units, f"{module_path} attention units")
    embedding_indices = _unit_embedding_indices(layout, selected)
    qkv_indices = _unit_qkv_output_indices(layout, selected)
    operations: list[PrimitiveOperation] = []
    c = int(block.c)
    _prune_conv_out(block.cv1, embedding_indices + [c + index for index in embedding_indices], f"{module_path}.cv1", operations)
    _prune_conv_in(block.cv2, embedding_indices + [c + index for index in embedding_indices], f"{module_path}.cv2", operations)
    for index, psa in enumerate(block.m):
        _prune_psablock_width(psa, layout, selected, f"{module_path}.m.{index}", operations)
    block.c = c - len(embedding_indices)
    validate_c2psa_invariants(block)
    return HeadAwarePruningResult(
        family="C2PSA_HEAD_AWARE_LOGICAL_WIDTH_V1",
        hidden_channels_before=c,
        hidden_channels_removed=len(embedding_indices),
        hidden_channels_after=int(block.c),
        attention_units_before=layout.total_units,
        attention_units_removed=len(selected),
        selected_attention_units=tuple(selected),
        embedding_indices=tuple(embedding_indices),
        qkv_output_indices=tuple(qkv_indices),
        operations=tuple(operations),
    )


def prune_attention_c3k2_head_aware_units(
    block: nn.Module, selected_units: Sequence[int], module_path: str
) -> HeadAwarePruningResult:
    scores, layout = attention_c3k2_head_aware_importance(block)
    del scores
    selected = _unique_indices(selected_units, layout.total_units, f"{module_path} attention units")
    embedding_indices = _unit_embedding_indices(layout, selected)
    qkv_indices = _unit_qkv_output_indices(layout, selected)
    operations: list[PrimitiveOperation] = []
    c = int(block.c)
    n = len(block.m)
    paired_split = embedding_indices + [c + index for index in embedding_indices]
    concat_indices = [segment * c + index for segment in range(2 + n) for index in embedding_indices]
    _prune_conv_out(block.cv1, paired_split, f"{module_path}.cv1", operations)
    _prune_conv_in(block.cv2, concat_indices, f"{module_path}.cv2", operations)
    for index, child in enumerate(block.m):
        bottleneck, psa = child[0], child[1]
        _prune_conv_in(bottleneck.cv1, embedding_indices, f"{module_path}.m.{index}.0.cv1", operations)
        _prune_conv_out(bottleneck.cv2, embedding_indices, f"{module_path}.m.{index}.0.cv2", operations)
        _prune_psablock_width(psa, layout, selected, f"{module_path}.m.{index}.1", operations)
    block.c = c - len(embedding_indices)
    validate_attention_c3k2_invariants(block)
    return HeadAwarePruningResult(
        family="C3K2_ATTENTION_HEAD_AWARE_LOGICAL_WIDTH_V1",
        hidden_channels_before=c,
        hidden_channels_removed=len(embedding_indices),
        hidden_channels_after=int(block.c),
        attention_units_before=layout.total_units,
        attention_units_removed=len(selected),
        selected_attention_units=tuple(selected),
        embedding_indices=tuple(embedding_indices),
        qkv_output_indices=tuple(qkv_indices),
        operations=tuple(operations),
    )
