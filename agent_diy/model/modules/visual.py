"""Visual (spatial) encoder based on ConvNeXt.

Adapted from reference_ppo.
Ref: https://arxiv.org/abs/2201.03545
"""

import math
import torch
from torch import nn
import torch.nn.functional as F


def _trunc_normal(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal initialization (a=-2std, b=+2std)."""
    with torch.no_grad():
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    return tensor


class LayerNorm2D(nn.Module):
    """LayerNorm that supports both channels-last and channels-first layouts."""

    def __init__(self, dim: int, eps: float = 1e-6, channels_last: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.channels_last = channels_last
        self.normalized_shape = (dim,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class ConvNeXtBlock(nn.Module):
    """Single ConvNeXt block: depthwise conv 7x7 + inverted bottleneck."""

    def __init__(self, dim: int, layer_scale_init: float = 1e-6):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.block = nn.Sequential(
            LayerNorm2D(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
            LayerScale(dim, layer_scale_init),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)      # NCHW -> NHWC
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)      # NHWC -> NCHW
        return identity + x


class ConvNeXtEncoder(nn.Sequential):
    """Multi-stage ConvNeXt encoder ending with AdaptiveAvgPool -> LayerNorm.

    Args:
        in_channels:      input channels (e.g. 8 for local map, 4 for global)
        dims:             output channels per stage, e.g. [32, 64, 128]
        depths:           number of ConvNeXtBlocks per stage
        downsample_sizes: stride of each stage's stem conv
    """

    def __init__(
        self,
        in_channels: int,
        dims: list[int],
        depths: list[int],
        downsample_sizes: list[int],
        layer_scale_init: float = 1e-6,
    ):
        blocks: list[nn.Module] = []
        ch_in = in_channels
        for i, ch_out in enumerate(dims):
            if i == 0:
                stem = [
                    nn.Conv2d(ch_in, ch_out, kernel_size=downsample_sizes[i], stride=downsample_sizes[i]),
                    LayerNorm2D(ch_out, eps=1e-6, channels_last=False),
                ]
            else:
                stem = [
                    LayerNorm2D(ch_in, eps=1e-6, channels_last=False),
                    nn.Conv2d(ch_in, ch_out, kernel_size=downsample_sizes[i], stride=downsample_sizes[i]),
                ]
            stage = [ConvNeXtBlock(ch_out, layer_scale_init) for _ in range(depths[i])]
            blocks.extend(stem)
            blocks.extend(stage)
            ch_in = ch_out

        blocks.extend([
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LayerNorm(ch_in, eps=1e-6),
        ])
        super().__init__(*blocks)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            _trunc_normal(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
