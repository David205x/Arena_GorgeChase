# 针对向量输入的编码器, 主要借鉴了Simba
# https://openreview.net/forum?id=jXLiDKsuDo
# https://github.com/SonyResearch/simba
import math
import torch
from torch import nn


def get_fc_layer(
    in_dim: int,
    out_dim: int,
    orthogonal_init: bool = True,
    gain: float = 1,
):
    fc_layer = nn.Linear(in_dim, out_dim)
    if orthogonal_init:
        nn.init.orthogonal_(fc_layer.weight, gain=gain)
    else:
        nn.init.kaiming_normal_(fc_layer.weight)
    nn.init.zeros_(fc_layer.bias)
    return fc_layer


class MLPBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        use_silu: bool = False,
    ):
        super().__init__(
            get_fc_layer(input_dim, hidden_dim, gain=math.sqrt(2)),
            nn.SiLU() if use_silu else nn.ReLU(),
            get_fc_layer(hidden_dim, hidden_dim, gain=math.sqrt(2)),
            nn.SiLU() if use_silu else nn.ReLU(),
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        inv_bottle_size: int = 4,
    ):
        super().__init__()
        # ffn
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            get_fc_layer(
                hidden_dim, inv_bottle_size * hidden_dim, orthogonal_init=False
            ),
            nn.SiLU(),
            get_fc_layer(
                inv_bottle_size * hidden_dim, hidden_dim, orthogonal_init=False
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(x)


class SimbaEncoder(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        block_num: int,
    ):
        blocks = []
        fc_layer = get_fc_layer(input_dim, hidden_dim)
        blocks.append(fc_layer)
        for _ in range(block_num):
            blocks.append(ResidualBlock(hidden_dim))
        blocks.append(nn.LayerNorm(hidden_dim))
        super().__init__(*blocks)


class SimbaEncoderHeadless(nn.Sequential):
    def __init__(
        self,
        hidden_dim: int,
        block_num: int,
    ):
        blocks = []
        for _ in range(block_num):
            blocks.append(ResidualBlock(hidden_dim))
        blocks.append(nn.LayerNorm(hidden_dim))
        super().__init__(*blocks)
