from __future__ import annotations

from fractions import Fraction
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicNoParamLayerAdapter(nn.Module):
    """Parameter-free replacement that preserves a layer's dynamic output geometry.

    The adapter carries no trainable tensors. It forwards incoming feature information,
    resizes spatial dimensions with nearest-neighbour interpolation, and crops or zero-pads
    channels to match the original layer's output interface.
    """

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
        for name, value in {
            "spatial_h_num": spatial_h_num,
            "spatial_h_den": spatial_h_den,
            "spatial_w_num": spatial_w_num,
            "spatial_w_den": spatial_w_den,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if multi_input_policy not in {"concat", "first"}:
            raise ValueError(f"Unsupported multi_input_policy: {multi_input_policy}")

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
            tensors: list[torch.Tensor] = []
            for item in value:
                tensors.extend(DynamicNoParamLayerAdapter._collect_tensors(item))
            return tensors
        return []

    def _dynamic_target_hw(self, reference: torch.Tensor) -> tuple[int, int]:
        height = max(1, round(reference.shape[-2] * self.spatial_h_num / self.spatial_h_den))
        width = max(1, round(reference.shape[-1] * self.spatial_w_num / self.spatial_w_den))
        return int(height), int(width)

    @staticmethod
    def _match_spatial(x: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
        if tuple(x.shape[-2:]) == target_hw:
            return x
        return F.interpolate(x, size=target_hw, mode="nearest")

    def _match_channels(self, x: torch.Tensor) -> torch.Tensor:
        current_channels = int(x.shape[1])
        if current_channels == self.target_channels:
            return x
        if current_channels > self.target_channels:
            return x[:, : self.target_channels, ...]

        pad_channels = self.target_channels - current_channels
        padding = x.new_zeros((x.shape[0], pad_channels, *x.shape[2:]))
        return torch.cat((x, padding), dim=1)

    def forward(self, x: Any) -> torch.Tensor:
        sources = self._collect_tensors(x)
        if not sources:
            raise RuntimeError("DynamicNoParamLayerAdapter received no tensor input")
        if any(source.ndim != 4 for source in sources):
            shapes = [tuple(source.shape) for source in sources]
            raise RuntimeError(f"Adapter supports 4D NCHW tensors only; got {shapes}")

        target_hw = self._dynamic_target_hw(sources[0])
        if self.multi_input_policy == "first" or len(sources) == 1:
            output = self._match_spatial(sources[0], target_hw)
        else:
            resized = [self._match_spatial(source, target_hw) for source in sources]
            output = torch.cat(resized, dim=1)

        return self._match_channels(output)

    def extra_repr(self) -> str:
        return (
            f"target_channels={self.target_channels}, "
            f"h_scale={self.spatial_h_num}/{self.spatial_h_den}, "
            f"w_scale={self.spatial_w_num}/{self.spatial_w_den}, "
            f"multi_input_policy={self.multi_input_policy}"
        )


def reduced_fraction(output_size: int, input_size: int) -> tuple[int, int]:
    if output_size < 1 or input_size < 1:
        raise ValueError(f"Invalid spatial sizes: output={output_size}, input={input_size}")
    fraction = Fraction(int(output_size), int(input_size)).limit_denominator(64)
    return int(fraction.numerator), int(fraction.denominator)
