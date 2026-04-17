"""
Three-branch ActorCritic network for Gorge Chase.

Architecture:
    scalar (134,)    → SimbaEncoder  → 256
    local  (8,21,21) → ConvNeXtEncoder → 128
    global (4,64,64) → ConvNeXtEncoder → 128
                                          ───
    Concat(256+128+128=512) → SimbaEncoder → 512
        ├─ Policy head  : ResBlock → LN → Linear(→16)   + action mask
        └─ Value head   : ResBlock → LN → Linear(→n_bins) distributional
"""

import torch
from torch import nn
import torch.nn.functional as F

from agent_diy.model.modules import (
    get_fc_layer,
    ResidualBlock,
    SimbaEncoder,
    ConvNeXtEncoder,
)

# ======================== default hyper-params ========================
SCALAR_DIM = 134
LOCAL_CH = 8
GLOBAL_CH = 4
ACTION_NUM = 16         # 8 move + 8 flash

VF_N_BINS = 51
VF_MIN = -30.0
VF_MAX = 30.0

VECTOR_EMBED = 256
VISION_EMBED = 128
TORSO_DIM = VECTOR_EMBED + VISION_EMBED * 2   # 512


class Model(nn.Module):
    """Shared-backbone ActorCritic with distributional value head."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_diy"
        self.device = device

        # ==================== Encoders ====================
        self.scalar_encoder = SimbaEncoder(
            input_dim=SCALAR_DIM,
            hidden_dim=VECTOR_EMBED,
            block_num=2,
        )
        self.local_encoder = ConvNeXtEncoder(
            in_channels=LOCAL_CH,
            dims=[32, 64, VISION_EMBED],
            depths=[1, 1, 1],
            downsample_sizes=[2, 2, 2],
        )
        self.global_encoder = ConvNeXtEncoder(
            in_channels=GLOBAL_CH,
            dims=[32, 64, VISION_EMBED],
            depths=[1, 1, 1],
            downsample_sizes=[2, 2, 2],
        )

        # ==================== Fusion (Torso) ====================
        self.fusion = SimbaEncoder(
            input_dim=TORSO_DIM,
            hidden_dim=TORSO_DIM,
            block_num=2,
        )

        # ==================== Policy Head ====================
        self.policy_head = nn.Sequential(
            ResidualBlock(TORSO_DIM),
            nn.LayerNorm(TORSO_DIM),
            get_fc_layer(TORSO_DIM, ACTION_NUM, gain=0.01),
        )

        # ==================== Value Head (distributional HL-Gauss) ====================
        self.vf_n_bins = VF_N_BINS
        support = torch.linspace(VF_MIN, VF_MAX, VF_N_BINS + 1)
        self.register_buffer("vf_support", support)
        self.register_buffer("vf_centers", (support[:-1] + support[1:]) / 2)
        self.value_head = nn.Sequential(
            ResidualBlock(TORSO_DIM),
            nn.LayerNorm(TORSO_DIM),
            get_fc_layer(TORSO_DIM, VF_N_BINS, gain=0.01),
        )

    # ------------------------------------------------------------------ forward
    def forward(self, scalar, local_map, global_map, legal_action):
        """
        Args:
            scalar:       (B, SCALAR_DIM)
            local_map:    (B, LOCAL_CH, 21, 21)
            global_map:   (B, GLOBAL_CH, 64, 64)
            legal_action: (B, ACTION_NUM)  float mask, 1=legal 0=illegal

        Returns:
            policy_probs:  (B, ACTION_NUM)  probability over actions
            value_logits:  (B, VF_N_BINS)   raw logits for distributional value
        """
        scalar_embed = self.scalar_encoder(scalar)
        local_embed = self.local_encoder(local_map)
        global_embed = self.global_encoder(global_map)

        torso_in = torch.cat([scalar_embed, local_embed, global_embed], dim=-1)
        torso_out = self.fusion(torso_in)

        # policy
        policy_logits = self.policy_head(torso_out)
        policy_logits = self._mask_illegal(policy_logits, legal_action)
        policy_probs = F.softmax(policy_logits, dim=-1)

        # value (distributional)
        value_logits = self.value_head(torso_out)

        return policy_probs, value_logits

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _mask_illegal(logits: torch.Tensor, legal_action: torch.Tensor) -> torch.Tensor:
        """Numerically stable illegal-action masking (reference_ppo style)."""
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = logits - label_max
        logits = logits * legal_action
        logits = logits + 1e5 * (legal_action - 1.0)
        return logits

    def value_expected(self, value_logits: torch.Tensor) -> torch.Tensor:
        """Convert distributional logits → scalar expected value.

        Args:
            value_logits: (B, VF_N_BINS)
        Returns:
            (B, 1)
        """
        probs = F.softmax(value_logits, dim=-1)
        return (probs * self.vf_centers).sum(dim=-1, keepdim=True)

    # ------------------------------------------------------------------ mode
    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
