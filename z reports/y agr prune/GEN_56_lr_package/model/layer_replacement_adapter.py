from __future__ import annotations

from fractions import Fraction
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNoParamLayerAdapter(nn.Module):
    """Parameter-free whole-layer replacement with dynamic output geometry."""

    def __init__(
        self,
        target_channels: int,
        spatial_h_num: int,
        spatial_h_den: int,
        spatial_w_num: int,
        spatial_w_den: int,
        multi_input_policy: str = "concat",
    ) -> None:
        super().__init__()
        if target_channels < 1:
            raise ValueError("target_channels must be positive")
        self.target_channels = int(target_channels)
        self.spatial_h_num = int(spatial_h_num)
        self.spatial_h_den = int(spatial_h_den)
        self.spatial_w_num = int(spatial_w_num)
        self.spatial_w_den = int(spatial_w_den)
        self.multi_input_policy = str(multi_input_policy)

    @staticmethod
    def _collect_tensors(value: Any) -> list[torch.Tensor]:
        if torch.is_tensor(value):
            return [value]
        if isinstance(value, (tuple, list)):
            out: list[torch.Tensor] = []
            for item in value:
                out.extend(DynamicNoParamLayerAdapter._collect_tensors(item))
            return out
        return []

    def _target_hw(self, x: torch.Tensor) -> tuple[int, int]:
        h = max(
            1,
            round(
                x.shape[-2]
                * self.spatial_h_num
                / self.spatial_h_den
            ),
        )
        w = max(
            1,
            round(
                x.shape[-1]
                * self.spatial_w_num
                / self.spatial_w_den
            ),
        )
        return int(h), int(w)

    @staticmethod
    def _match_spatial(
        x: torch.Tensor,
        target_hw: tuple[int, int],
    ) -> torch.Tensor:
        if tuple(x.shape[-2:]) == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="nearest")

    def _match_channels(self, x: torch.Tensor) -> torch.Tensor:
        c = int(x.shape[1])
        if c == self.target_channels:
            return x
        if c > self.target_channels:
            return x[:, : self.target_channels, ...]
        pad = x.new_zeros(
            (x.shape[0], self.target_channels - c, *x.shape[2:])
        )
        return torch.cat((x, pad), dim=1)

    def forward(self, x: Any) -> torch.Tensor:
        sources = self._collect_tensors(x)
        if not sources:
            raise RuntimeError("Adapter received no tensor input")
        if any(t.ndim != 4 for t in sources):
            raise RuntimeError("Adapter supports NCHW tensors only")

        target_hw = self._target_hw(sources[0])

        if self.multi_input_policy == "first" or len(sources) == 1:
            out = self._match_spatial(sources[0], target_hw)
        else:
            out = torch.cat(
                [self._match_spatial(t, target_hw) for t in sources],
                dim=1,
            )

        return self._match_channels(out)


def reduced_fraction(
    output_size: int,
    input_size: int,
) -> tuple[int, int]:
    fraction = Fraction(
        int(output_size),
        int(input_size),
    ).limit_denominator(64)
    return int(fraction.numerator), int(fraction.denominator)
