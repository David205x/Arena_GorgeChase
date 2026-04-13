import torch
from torch import nn
from torch.nn import functional as F
from agent_diy.model.modules import get_fc_layer


# 暂未完成
class VectorMerge(nn.Module):
    def __init__(
        self,
        input_sizes: list[int],
        output_size: int,
        pointwise_gate: bool = True,
        use_layer_norm: bool = True,
    ):
        """Initializes VectorMerge module.

        Args:
            input_sizes: A dictionary mapping input names to their size (a single
                integer for 1d inputs, or None for 0d inputs).
                If an input size is None, we assume it's ().
            output_size: The size of the output vector.
            pointwise_gate: The type of gating mechanism to use.
            use_layer_norm: Whether to use layer normalization.
        """
        super().__init__()
        self._input_sizes = input_sizes
        self._output_size = output_size
        self._pointwise_gate = pointwise_gate
        self._use_layer_norm = use_layer_norm
        self._gate_size = self._output_size if self._pointwise_gate else 1

        self.gates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(input_size),
                    nn.SiLU(),
                    get_fc_layer(input_size, output_size, orthogonal_init=False),
                )
                for input_size in input_sizes
            ]
        )
