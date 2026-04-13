# 针对视觉输入的编码器, 主要借鉴了ConvNeXt
# https://github.com/facebookresearch/ConvNeXt
# https://arxiv.org/abs/2201.03545
import math
import torch
from torch import nn
import torch.nn.functional as F


def trunc_normal_(
    tensor: torch.Tensor,
    mean: float = 0.0,
    std: float = 1.0,
    a: float = -2.0,
    b: float = 2.0,
) -> torch.Tensor:
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
    return tensor


class SquashDims(nn.Module):
    def __init__(self, ndims_in: int = 3):
        super().__init__()
        self.ndims_in = ndims_in

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value.flatten(-self.ndims_in, -1)
        return value


class LayerScale(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class LayerNorm2D(nn.Module):
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        channels_last: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.channels_last = channels_last
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.channels_last:
            return F.layer_norm(
                x, self.normalized_shape, self.weight, self.bias, self.eps
            )
        else:
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.block = nn.Sequential(
            LayerNorm2D(dim, eps=1e-6),
            nn.Linear(dim, 4 * dim),
            nn.SiLU(),
            nn.Linear(4 * dim, dim),
            LayerScale(dim, layer_scale_init_value),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.block(x)
        x = x.permute(0, 3, 1, 2)
        return identity + x


class ConvNeXtEncoder(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        dims: list[int],
        depths: list[int],
        downsample_sizes: list[int],
        layer_scale_init_value: float = 1e-6,
    ):
        blocks = []
        _in = in_channels
        for i, _out in enumerate(dims):
            if i == 0:
                downsample_layer = [
                    nn.Conv2d(
                        _in,
                        _out,
                        kernel_size=downsample_sizes[i],
                        stride=downsample_sizes[i],
                    ),
                    LayerNorm2D(_out, eps=1e-6, channels_last=False),
                ]
            else:
                downsample_layer = [
                    LayerNorm2D(_in, eps=1e-6, channels_last=False),
                    nn.Conv2d(
                        _in,
                        _out,
                        kernel_size=downsample_sizes[i],
                        stride=downsample_sizes[i],
                    ),
                ]
            stage = [
                ConvNeXtBlock(dim=_out, layer_scale_init_value=layer_scale_init_value)
                for _ in range(depths[i])
            ]
            _in = _out
            blocks.extend(downsample_layer)
            blocks.extend(stage)
        blocks.extend(
            [
                nn.AdaptiveAvgPool2d((1, 1)),
                SquashDims(),
                nn.LayerNorm(_in, eps=1e-6),
            ]
        )
        super().__init__(*blocks)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=0.02)
            nn.init.constant_(m.bias, 0)
