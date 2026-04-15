"""
PPO algorithm for agent_diy (multi-input model + distributional value).

Enhancements over baseline agent_ppo:
  - HL-Gauss distributional value loss (cross-entropy on soft categorical)
  - Log-space importance ratio for numerical stability
  - Ratio hard-clamp [0, 3] to prevent gradient explosion
  - Dual-Clip PPO for negative-advantage protection
  - Per-batch advantage normalisation
  - channels_last memory format for CNN acceleration
"""

import os
import time

import torch
import torch.nn.functional as F

from agent_diy.model.model import Model, VF_N_BINS, VF_MIN, VF_MAX, ACTION_NUM
from agent_diy.algorithm.objectives import HLGaussLoss

# ======================== hyper-parameters ========================
CLIP_PARAM = 0.2
DUAL_CLIP = 3.0            # 0 to disable
VF_COEF = 0.5
ENTROPY_COEF = 0.01
VF_SIGMA = 0.75
GRAD_CLIP_NORM = 0.5
LR = 3e-4
ADV_NORM = True
LOG_INTERVAL = 60          # seconds between monitor reports


class Algorithm:
    def __init__(self, device, logger=None, monitor=None):
        self.device = device

        # model
        self.model = Model(device=device).to(device)
        self.model = self.model.to(memory_format=torch.channels_last)

        # optimiser
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8,
        )
        self.parameters = [
            p for pg in self.optimizer.param_groups for p in pg["params"]
        ]

        # HL-Gauss value objective
        self.hl_gauss = HLGaussLoss(
            min_value=VF_MIN,
            max_value=VF_MAX,
            num_bins=VF_N_BINS,
            sigma=VF_SIGMA,
            device=device,
        )

        # PPO clip bounds
        self.clip_param = CLIP_PARAM
        self.clip_high = 1.0 + CLIP_PARAM
        self.clip_low = 1.0 / self.clip_high
        self.dual_clip = DUAL_CLIP
        self.vf_coef = VF_COEF
        self.entropy_coef = ENTROPY_COEF

        # monitoring
        self.logger = logger
        self.monitor = monitor
        self.last_report_time = 0
        self.train_step = 0

    # ------------------------------------------------------------------ learn
    def learn(self, batch: dict[str, torch.Tensor]):
        """Single PPO update on a collated batch.

        Expected keys
        -------------
        scalar       (B, 134)
        local_map    (B, 8, 21, 21)
        global_map   (B, 4, 64, 64)
        legal_action (B, 16)       float mask 1/0
        old_action   (B, 1)        int64 action index
        old_prob     (B, 1)        float  probability of old_action under old policy
        reward       (B,)
        advantage    (B,)
        td_return    (B,)          discounted return (value target)
        """
        scalar = batch["scalar"].to(self.device)
        local_map = batch["local_map"].to(self.device)
        global_map = batch["global_map"].to(self.device)
        legal_action = batch["legal_action"].to(self.device)
        old_action = batch["old_action"].to(self.device).long()
        old_prob = batch["old_prob"].to(self.device)
        reward = batch["reward"].to(self.device)
        adv = batch["advantage"].to(self.device)
        td_return = batch["td_return"].to(self.device)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        new_probs, value_logits = self.model(scalar, local_map, global_map, legal_action)

        total_loss, info = self._compute_loss(
            new_probs, value_logits, old_action, old_prob, adv, td_return,
        )
        total_loss.backward()

        if GRAD_CLIP_NORM > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters, GRAD_CLIP_NORM)
        self.optimizer.step()
        self.train_step += 1

        self._maybe_report(total_loss, info, reward)

    # ----------------------------------------------------------- loss
    def _compute_loss(self, new_probs, value_logits, old_action, old_prob, adv, td_return):
        # ---- value loss (HL-Gauss cross-entropy) ----
        value_loss = self.hl_gauss(value_logits, td_return).mean()

        # ---- entropy ----
        entropy_loss = -(new_probs * torch.log(new_probs.clamp(1e-9, 1.0))).sum(-1).mean()

        # ---- importance ratio (log-space) ----
        new_prob = torch.gather(new_probs, dim=-1, index=old_action.view(-1, 1))
        ratio = torch.exp(
            torch.log(new_prob.clamp(1e-9, 1.0)) - torch.log(old_prob.clamp(1e-9, 1.0))
        )
        ratio = ratio.clamp(0.0, 3.0)

        # ---- advantage normalisation ----
        adv_std, adv_mean = torch.std_mean(adv)
        if ADV_NORM:
            adv = (adv - adv_mean) / adv_std.clamp_min(1e-7)
        adv = adv.view(-1, 1)

        # ---- PPO clipped objective ----
        surr1 = ratio * adv
        surr2 = ratio.clamp(self.clip_low, self.clip_high) * adv
        if self.dual_clip > 0:
            clip1 = torch.minimum(surr1, surr2)
            clip2 = torch.maximum(clip1, self.dual_clip * adv)
            clipped_obj = -torch.where(adv < 0, clip2, clip1)
        else:
            clipped_obj = -torch.minimum(surr1, surr2)
        policy_loss = clipped_obj.mean()

        with torch.no_grad():
            clipfrac = (ratio.gt(self.clip_high) | ratio.lt(self.clip_low)).float().mean().item()

        # ---- total ----
        total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy_loss

        info = {
            "value_loss": value_loss.detach(),
            "policy_loss": policy_loss.detach(),
            "entropy": entropy_loss.detach(),
            "clipfrac": clipfrac,
            "adv_mean": adv_mean.item(),
            "adv_std": adv_std.item(),
            "td_return_mean": td_return.mean().item(),
        }
        return total_loss, info

    # ----------------------------------------------------------- monitor
    def _maybe_report(self, total_loss, info, reward):
        now = time.time()
        if now - self.last_report_time < LOG_INTERVAL:
            return
        self.last_report_time = now

        results = {
            "total_loss": round(total_loss.item(), 4),
            "value_loss": round(info["value_loss"].item(), 4),
            "policy_loss": round(info["policy_loss"].item(), 4),
            "entropy": round(info["entropy"].item(), 4),
            "clipfrac": round(info["clipfrac"], 4),
            "adv_mean": round(info["adv_mean"], 4),
            "reward": round(reward.mean().item(), 4),
        }

        if self.logger:
            self.logger.info(
                f"[train step={self.train_step}] "
                f"loss={results['total_loss']} "
                f"pi={results['policy_loss']} "
                f"vf={results['value_loss']} "
                f"ent={results['entropy']} "
                f"clip={results['clipfrac']}"
            )
        if self.monitor:
            self.monitor.put_data({os.getpid(): results})

    # ----------------------------------------------------------- mode helpers
    def set_train_mode(self):
        self.model.set_train_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()
