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

import math
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

MONITOR_BATCH_KEYS = (
    "reward",
    "adv_mean",
    "adv_std",
    "adv_min",
    "adv_max",
    "td_return_mean",
    "td_return_min",
    "td_return_max",
    "old_prob_min",
    "old_prob_max",
    "new_prob_min",
    "new_prob_max",
    "explained_var",
    "grad_norm",
    "grad_norm_post_clip",
    "grad_clip_ratio",
    "param_update_rms",
    "batch_non_finite_count",
    "loss_non_finite",
    "grad_non_finite",
    "param_non_finite",
)


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
        self.last_monitor_results = {key: 0.0 for key in MONITOR_BATCH_KEYS}

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

        batch_stats = self._collect_batch_stats(reward, adv, td_return, old_prob)

        params_before_step = [param.detach().clone() for param in self.parameters]
        new_probs, value_logits = self.model(scalar, local_map, global_map, legal_action)
        value_pred = self.model.value_expected(value_logits).squeeze(-1)

        total_loss, info = self._compute_loss(
            new_probs, value_logits, old_action, old_prob, adv, td_return,
        )
        batch_stats["new_prob_min"] = float(new_probs.min().item())
        batch_stats["new_prob_max"] = float(new_probs.max().item())
        batch_stats["explained_var"] = self._explained_variance(value_pred.detach(), td_return)
        batch_stats["loss_non_finite"] = float(
            not all(
                torch.isfinite(x).all().item()
                for x in (total_loss, info["value_loss"], info["policy_loss"], info["entropy"])
            )
        )
        total_loss.backward()

        grad_norm = 0.0
        if GRAD_CLIP_NORM > 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(self.parameters, GRAD_CLIP_NORM).item())
        else:
            grad_norm = self._grad_norm()
        batch_stats["grad_norm"] = grad_norm
        post_clip_grad_norm = self._grad_norm()
        batch_stats["grad_norm_post_clip"] = post_clip_grad_norm
        batch_stats["grad_clip_ratio"] = (
            post_clip_grad_norm / grad_norm if math.isfinite(grad_norm) and grad_norm > 1e-12 else 1.0
        )
        batch_stats["grad_non_finite"] = float(self._has_non_finite_grad())
        self.optimizer.step()
        batch_stats["param_update_rms"] = self._param_update_rms(params_before_step)
        batch_stats["param_non_finite"] = float(self._has_non_finite_param())
        self.train_step += 1

        return self._maybe_report(total_loss, info, batch_stats)

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

    def _collect_batch_stats(self, reward, adv, td_return, old_prob):
        tensors = {
            "reward": reward,
            "adv": adv,
            "td_return": td_return,
            "old_prob": old_prob,
        }
        non_finite_count = 0
        for tensor in tensors.values():
            non_finite_count += int((~torch.isfinite(tensor)).sum().item())

        return {
            "reward": float(reward.mean().item()),
            "adv_mean": float(adv.mean().item()),
            "adv_std": float(adv.std(unbiased=False).item()),
            "adv_min": float(adv.min().item()),
            "adv_max": float(adv.max().item()),
            "td_return_mean": float(td_return.mean().item()),
            "td_return_min": float(td_return.min().item()),
            "td_return_max": float(td_return.max().item()),
            "old_prob_min": float(old_prob.min().item()),
            "old_prob_max": float(old_prob.max().item()),
            "new_prob_min": 0.0,
            "new_prob_max": 0.0,
            "grad_norm": 0.0,
            "batch_non_finite_count": float(non_finite_count),
            "loss_non_finite": 0.0,
            "grad_non_finite": 0.0,
            "param_non_finite": 0.0,
        }

    def _has_non_finite_grad(self) -> bool:
        for param in self.parameters:
            if param.grad is not None and not torch.isfinite(param.grad).all().item():
                return True
        return False

    def _grad_norm(self) -> float:
        total_sq = 0.0
        for param in self.parameters:
            if param.grad is None:
                continue
            grad_sq = float(param.grad.detach().pow(2).sum().item())
            total_sq += grad_sq
        return math.sqrt(total_sq)

    def _param_update_rms(self, params_before_step: list[torch.Tensor]) -> float:
        update_sq = 0.0
        param_count = 0
        for before, after in zip(params_before_step, self.parameters):
            delta = after.detach() - before
            update_sq += float(delta.pow(2).sum().item())
            param_count += int(delta.numel())
        if param_count == 0:
            return 0.0
        return math.sqrt(update_sq / param_count)

    @staticmethod
    def _explained_variance(value_pred: torch.Tensor, td_return: torch.Tensor) -> float:
        target_var = torch.var(td_return, unbiased=False)
        if not torch.isfinite(target_var).all().item() or float(target_var.item()) <= 1e-12:
            return 0.0
        residual_var = torch.var(td_return - value_pred, unbiased=False)
        if not torch.isfinite(residual_var).all().item():
            return 0.0
        ev = 1.0 - float(residual_var.item() / target_var.item())
        return max(-1.0, min(1.0, ev))

    def _has_non_finite_param(self) -> bool:
        for param in self.parameters:
            if not torch.isfinite(param).all().item():
                return True
        return False

    def _warn_if_unstable(self, total_loss, info, stats):
        if not self.logger:
            return
        unstable = (
            stats["batch_non_finite_count"] > 0
            or stats["loss_non_finite"] > 0
            or stats["grad_non_finite"] > 0
            or stats["param_non_finite"] > 0
            or not math.isfinite(stats["grad_norm"])
        )
        if not unstable:
            return
        self.logger.error(
            "[train unstable] "
            f"step={self.train_step} "
            f"loss={float(total_loss.detach().item()) if torch.isfinite(total_loss).all().item() else 'nan'} "
            f"pi={float(info['policy_loss'].item()) if torch.isfinite(info['policy_loss']).all().item() else 'nan'} "
            f"vf={float(info['value_loss'].item()) if torch.isfinite(info['value_loss']).all().item() else 'nan'} "
            f"ent={float(info['entropy'].item()) if torch.isfinite(info['entropy']).all().item() else 'nan'} "
            f"reward={stats['reward']:.4f} adv=[{stats['adv_min']:.4f},{stats['adv_max']:.4f}] "
            f"td=[{stats['td_return_min']:.4f},{stats['td_return_max']:.4f}] "
            f"old_prob=[{stats['old_prob_min']:.6f},{stats['old_prob_max']:.6f}] "
            f"new_prob=[{stats['new_prob_min']:.6f},{stats['new_prob_max']:.6f}] "
            f"grad_norm={stats['grad_norm']} batch_bad={int(stats['batch_non_finite_count'])} "
            f"loss_bad={int(stats['loss_non_finite'])} grad_bad={int(stats['grad_non_finite'])} param_bad={int(stats['param_non_finite'])}"
        )

    # ----------------------------------------------------------- monitor
    def _maybe_report(self, total_loss, info, stats):
        self._warn_if_unstable(total_loss, info, stats)

        results = {
            "total_loss": round(total_loss.item(), 4),
            "value_loss": round(info["value_loss"].item(), 4),
            "policy_loss": round(info["policy_loss"].item(), 4),
            "entropy": round(info["entropy"].item(), 4),
            "clipfrac": round(info["clipfrac"], 4),
            "reward": round(stats["reward"], 4),
            "adv_mean": round(stats["adv_mean"], 4),
            "adv_std": round(stats["adv_std"], 4),
            "adv_min": round(stats["adv_min"], 4),
            "adv_max": round(stats["adv_max"], 4),
            "td_return_mean": round(stats["td_return_mean"], 4),
            "td_return_min": round(stats["td_return_min"], 4),
            "td_return_max": round(stats["td_return_max"], 4),
            "old_prob_min": round(stats["old_prob_min"], 6),
            "old_prob_max": round(stats["old_prob_max"], 6),
            "new_prob_min": round(stats["new_prob_min"], 6),
            "new_prob_max": round(stats["new_prob_max"], 6),
            "explained_var": round(stats["explained_var"], 6),
            "grad_norm": round(stats["grad_norm"], 6) if math.isfinite(stats["grad_norm"]) else stats["grad_norm"],
            "grad_norm_post_clip": round(stats["grad_norm_post_clip"], 6) if math.isfinite(stats["grad_norm_post_clip"]) else stats["grad_norm_post_clip"],
            "grad_clip_ratio": round(stats["grad_clip_ratio"], 6) if math.isfinite(stats["grad_clip_ratio"]) else stats["grad_clip_ratio"],
            "param_update_rms": round(stats["param_update_rms"], 8) if math.isfinite(stats["param_update_rms"]) else stats["param_update_rms"],
            "batch_non_finite_count": int(stats["batch_non_finite_count"]),
            "loss_non_finite": int(stats["loss_non_finite"]),
            "grad_non_finite": int(stats["grad_non_finite"]),
            "param_non_finite": int(stats["param_non_finite"]),
        }
        self.last_monitor_results = results

        now = time.time()
        if now - self.last_report_time < LOG_INTERVAL:
            return results
        self.last_report_time = now

        if self.logger:
            self.logger.info(
                f"[train step={self.train_step}] "
                f"loss={results['total_loss']} "
                f"pi={results['policy_loss']} "
                f"vf={results['value_loss']} "
                f"ent={results['entropy']} "
                f"clip={results['clipfrac']} "
                f"ev={results['explained_var']} "
                f"adv=[{results['adv_min']},{results['adv_max']}] "
                f"td=[{results['td_return_min']},{results['td_return_max']}] "
                f"grad_pre={results['grad_norm']} "
                f"grad_post={results['grad_norm_post_clip']} "
                f"upd_rms={results['param_update_rms']}"
            )
        if self.monitor:
            self.monitor.put_data({os.getpid(): results})
        return results

    # ----------------------------------------------------------- mode helpers
    def set_train_mode(self):
        self.model.set_train_mode()

    def set_eval_mode(self):
        self.model.set_eval_mode()
