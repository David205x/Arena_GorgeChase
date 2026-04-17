"""Vector (scalar) encoders: get_fc_layer, ResidualBlock, SimbaEncoder.

Adapted from reference_ppo (Simba architecture).
"""

import math
import torch
from torch import nn


def get_fc_layer(
    in_dim: int,
    out_dim: int,
    orthogonal_init: bool = True,
    gain: float = 1.0,
) -> nn.Linear:
    fc = nn.Linear(in_dim, out_dim)
    if orthogonal_init:
        nn.init.orthogonal_(fc.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(fc.weight)
    nn.init.zeros_(fc.bias)
    return fc


class ResidualBlock(nn.Module):
    """Pre-norm residual block with inverted bottleneck (4x expansion)."""

    def __init__(self, hidden_dim: int, inv_bottle_size: int = 4):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            get_fc_layer(hidden_dim, inv_bottle_size * hidden_dim, orthogonal_init=True),
            nn.SiLU(),
            get_fc_layer(inv_bottle_size * hidden_dim, hidden_dim, orthogonal_init=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class SimbaEncoder(nn.Sequential):
    """Linear projection -> N ResidualBlocks -> LayerNorm."""

    def __init__(self, input_dim: int, hidden_dim: int, block_num: int = 2):
        blocks: list[nn.Module] = [get_fc_layer(input_dim, hidden_dim)]
        for _ in range(block_num):
            blocks.append(ResidualBlock(hidden_dim))
        blocks.append(nn.LayerNorm(hidden_dim))
        super().__init__(*blocks)
