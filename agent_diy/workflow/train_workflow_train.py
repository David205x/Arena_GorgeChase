#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Gorge Chase agent_diy.
峡谷追猎 agent_diy 训练工作流。

Design highlights
-----------------
- Train / val map split  (train=2, 4-10, val=1, 3)
- Curriculum learning    (4 stages driven by train episode count)
- Per-episode monitor reporting:
    · episode-level stats    from extractor.build_monitor_metrics()
    · reward sub-items       accumulated step-by-step from reward_info
    · algorithm metrics      handled asynchronously inside Algorithm._maybe_report()
- Periodic model save    (every Config.SAVE_INTERVAL seconds)
- Best model save        (whenever val total_score improves)
- Model reload           (every episode, non-blocking try/except)
"""

import copy
import os
import random
import time
from pathlib import Path

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


# ── train / val map split ─────────────────────────────────────────────────────
TRAIN_MAPS = [2, 4, 5, 6, 7, 8, 9, 10]   # maps 1-8
VAL_MAPS   = [1, 3]             # maps 9-10

# Run 1 validation episode after every VAL_EVERY training episodes.
VAL_EVERY = 20

# ── curriculum learning ───────────────────────────────────────────────────────
# Matches the P佬 curriculum table (monster_interval / monster_speedup use
# the midpoint of the documented ranges).
# Format: (up_to_train_episode, env_conf_override)
_CURRICULUM: list[tuple[int | None, dict]] = [
    (150,  {"treasure_count": 9, "buff_count": 2, "monster_interval": 260, "monster_speedup": 410}),
    (500,  {"treasure_count": 8, "buff_count": 1, "monster_interval": 220, "monster_speedup": 330}),
    (900,  {"treasure_count": 7, "buff_count": 1, "monster_interval": 170, "monster_speedup": 250}),
    (None, {"treasure_count": 6, "buff_count": 1, "monster_interval": 220, "monster_speedup": 280}),
]

# ── reward_info keys that map to monitor Group-2 (need "reward_" prefix) ─────
_REWARD_OVERVIEW_KEYS = frozenset({
    "total", "alpha",
    "survival", "survival_weighted",
    "explore", "explore_weighted",
    "terminal",
})

# reward_info alpha is a coefficient (mean across steps); all others are sums.
_REWARD_MEAN_KEYS = frozenset({"alpha"})

# ── timing constants ──────────────────────────────────────────────────────────
_MONITOR_INTERVAL = 60   # seconds: min gap between monitor uploads (train mode)
_METRICS_INTERVAL = 60   # seconds: polling cadence for get_training_metrics()

# ── eval snapshot export ──────────────────────────────────────────────────────
_EVAL_SNAPSHOT_DIR = Path('	/workspace/log/trajectory').resolve()


# ═════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═════════════════════════════════════════════════════════════════════════════

def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    """Framework entry point called by the training harness."""
    last_save_time = time.time()
    env   = envs[0]
    agent = agents[0]

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    runner = EpisodeRunner(
        env=env,
        agent=agent,
        base_usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in runner.run_episode():
            agent.send_sample_data(g_data)
            g_data.clear()

            if time.time() - last_save_time >= Config.SAVE_INTERVAL:
                agent.save_model()
                last_save_time = time.time()


# ═════════════════════════════════════════════════════════════════════════════
# EpisodeRunner
# ═════════════════════════════════════════════════════════════════════════════

class EpisodeRunner:
    """Manages episode-level execution: env interaction, sample collection,
    curriculum scheduling, monitoring and model persistence."""

    def __init__(self, env, agent, base_usr_conf, logger, monitor):
        self.env           = env
        self.agent         = agent
        self.base_usr_conf = base_usr_conf
        self.logger        = logger
        self.monitor       = monitor

        # episode counters
        self.train_ep_cnt = 0
        self.val_ep_cnt   = 0

        # best val score tracker for best-model checkpointing
        self.best_val_score = float("-inf")

        # track which train_ep_cnt last triggered a val run (prevents val loop)
        self._last_val_at_train_ep = 0

        # rate-limiting timestamps
        self._last_monitor_time  = 0.0
        self._last_metrics_time  = 0.0
        self._last_log_time      = 0.0

    # ── public ────────────────────────────────────────────────────────────────

    def run_episode(self):
        """Execute one episode (train or val) and yield list[SampleData].

        For val episodes nothing is yielded (we only collect metrics).
        """
        is_val   = self._should_run_val()
        usr_conf = self._build_usr_conf(is_val)
        tag      = "VAL" if is_val else "TRAIN"

        # poll learner-side training metrics periodically
        self._maybe_poll_training_metrics(tag)

        # reset env
        env_obs = self.env.reset(usr_conf)
        if handle_disaster_recovery(env_obs, self.logger):
            return

        # reset agent + try to load latest weights
        self.agent.reset(env_obs)
        try:
            self.agent.load_model(id="latest")
        except Exception as exc:
            self.logger.info(f"[{tag}] load_model skipped: {exc}")

        # process initial observation (step-0, no action taken yet)
        obs_data, remain_info = self.agent.observation_process(env_obs)

        # bump episode counter after env reset succeeds
        if is_val:
            self.val_ep_cnt += 1
            ep_id = self.val_ep_cnt
        else:
            self.train_ep_cnt += 1
            ep_id = self.train_ep_cnt

        self.logger.info(f"[{tag} ep={ep_id}] start | "
                         f"train_ep={self.train_ep_cnt} val_ep={self.val_ep_cnt}")

        # ── episode loop ──────────────────────────────────────────────────────
        collector:           list[SampleData]  = []
        reward_info_accum:   dict[str, float]  = {}
        reward_info_steps:   int               = 0
        ep_reward:           float             = 0.0
        step:                int               = 0
        done:                bool              = False

        while not done:
            # predict
            act_data = self.agent.predict(list_obs_data=[obs_data])[0]
            act      = self.agent.action_process(act_data)

            # step
            env_reward, env_obs = self.env.step(act)
            if handle_disaster_recovery(env_obs, self.logger):
                break

            terminated = bool(env_obs.get("terminated", False))
            truncated  = bool(env_obs.get("truncated",  False))
            step      += 1
            done       = terminated or truncated

            # next observation & shaped reward (includes terminal component)
            _obs_data, _remain_info = self.agent.observation_process(env_obs)

            reward_arr = np.array(
                _remain_info.get("reward", [0.0]), dtype=np.float32
            )
            ep_reward += float(reward_arr[0])

            # accumulate reward sub-items for episode-level monitoring
            r_info: dict = _remain_info.get("reward_info", {})
            _accumulate_reward_info(reward_info_accum, r_info)
            reward_info_steps += 1

            # build SampleData (train episodes only)
            if not is_val:
                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array([act_data.action[0]], dtype=np.float32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    reward=reward_arr,
                    value=np.array(act_data.value, dtype=np.float32).flatten()[:1],
                    next_value=np.zeros(1, dtype=np.float32),
                    td_return=np.zeros(1, dtype=np.float32),
                    advantage=np.zeros(1, dtype=np.float32),
                    done=np.array([float(done)], dtype=np.float32),
                )
                collector.append(frame)

            if done:
                if is_val:
                    self._save_eval_extractor_snapshot(step)
                self._on_episode_end(
                    tag=tag,
                    ep_id=ep_id,
                    step=step,
                    ep_reward=ep_reward,
                    reward_info_accum=reward_info_accum,
                    reward_info_steps=reward_info_steps,
                    is_val=is_val,
                    env_obs=env_obs,
                )
                if not is_val and collector:
                    yield sample_process(collector)
                break

            # advance state
            obs_data    = _obs_data
            remain_info = _remain_info

    # ── curriculum ────────────────────────────────────────────────────────────

    def _should_run_val(self) -> bool:
        """Return True when a validation episode should be run next."""
        if (
            self.train_ep_cnt > 0
            and self.train_ep_cnt % VAL_EVERY == 0
            and self.train_ep_cnt != self._last_val_at_train_ep
        ):
            self._last_val_at_train_ep = self.train_ep_cnt
            return True
        return False

    def _get_curriculum_conf(self) -> dict:
        """Return the env_conf override for the current curriculum stage."""
        for up_to, override in _CURRICULUM:
            if up_to is None or self.train_ep_cnt < up_to:
                return override
        return _CURRICULUM[-1][1]

    def _build_usr_conf(self, is_val: bool) -> dict:
        """Build a deep-copied usr_conf with correct map list and curriculum."""
        conf    = copy.deepcopy(self.base_usr_conf)
        ec: dict = conf["env_conf"]

        # map selection
        ec["map"]        = list(VAL_MAPS if is_val else TRAIN_MAPS)
        ec["map_random"] = True

        # curriculum overrides (train only)
        if not is_val:
            for k, v in self._get_curriculum_conf().items():
                ec[k] = v

        return conf

    # ── episode end callbacks ─────────────────────────────────────────────────

    def _on_episode_end(
        self,
        tag:               str,
        ep_id:             int,
        step:              int,
        ep_reward:         float,
        reward_info_accum: dict,
        reward_info_steps: int,
        is_val:            bool,
        env_obs:           dict,
    ) -> None:
        now        = time.time()
        terminated = bool(env_obs.get("terminated", False))
        truncated  = bool(env_obs.get("truncated",  False))
        env_info   = env_obs.get("observation", {}).get("env_info", {})
        total_score: float = float(env_info.get("total_score", 0.0))

        result = "FAIL" if terminated else ("WIN" if truncated else "TRUNC")

        # ── logger ────────────────────────────────────────────────────────────
        if now - self._last_log_time >= Config.LOG_INTERVAL or is_val:
            stage_label = self._curriculum_stage_label()
            self.logger.info(
                f"[{tag} ep={ep_id}] steps={step} result={result} "
                f"score={total_score:.1f} ep_reward={ep_reward:.4f} "
                f"curriculum={stage_label}"
            )
            self._last_log_time = now

        # ── best model (val only) ─────────────────────────────────────────────
        if is_val and total_score > self.best_val_score:
            self.best_val_score = total_score
            try:
                self.agent.save_model(id="best")
                self.logger.info(
                    f"[VAL] new best score={total_score:.1f} → model saved (id=best)"
                )
            except Exception as exc:
                self.logger.warning(f"[VAL] best model save failed: {exc}")

        # ── monitor ───────────────────────────────────────────────────────────
        if self.monitor is None:
            return
        # val episodes always report; train episodes respect the interval
        if not is_val and now - self._last_monitor_time < _MONITOR_INTERVAL:
            return

        monitor_data = self._build_monitor_data(
            ep_reward=ep_reward,
            reward_info_accum=reward_info_accum,
            reward_info_steps=reward_info_steps,
        )
        self.monitor.put_data({os.getpid(): monitor_data})
        self._last_monitor_time = now

    # ── monitor data assembly ─────────────────────────────────────────────────

    def _build_monitor_data(
        self,
        ep_reward:         float,
        reward_info_accum: dict,
        reward_info_steps: int,
    ) -> dict[str, float]:
        """Assemble the full monitor dict for one episode.

        Key mapping
        -----------
        Group 6-9 (episode stats):   from extractor.build_monitor_metrics()
        Group 2   (reward overview):  reward_info keys → "reward_{key}" (sum or mean)
        Group 3   (survival sub):     "s_*" keys → same key (episode sum)
        Group 4   (explore sub):      "e_*" keys → same key (episode sum)
        Group 5   (terminal sub):     "t_*" keys → same key (episode sum)
        """
        data: dict[str, float] = {}
        n = max(reward_info_steps, 1)

        # episode stats (Groups 6-9)
        try:
            data.update(self.agent.extractor.build_monitor_metrics())
        except Exception:
            pass

        # reward sub-items (Groups 2-5)
        for k, v_sum in reward_info_accum.items():
            if k == "t_type":        # string field, skip
                continue
            if k in _REWARD_OVERVIEW_KEYS:
                # Group 2: add "reward_" prefix
                monitor_key = f"reward_{k}"
                # alpha is a coefficient → report mean; others are reward sums
                data[monitor_key] = round(v_sum / n if k in _REWARD_MEAN_KEYS else v_sum, 6)
            else:
                # Groups 3-5: key already has s_ / e_ / t_ prefix
                data[k] = round(v_sum, 6)

        # episode total reward (convenience)
        data["ep_reward"] = round(ep_reward, 4)

        return data

    # ── helpers ───────────────────────────────────────────────────────────────

    def _maybe_poll_training_metrics(self, tag: str) -> None:
        now = time.time()
        if now - self._last_metrics_time < _METRICS_INTERVAL:
            return
        self._last_metrics_time = now
        try:
            metrics = get_training_metrics()
            if metrics is not None:
                self.logger.info(f"[{tag}] training_metrics={metrics}")
        except Exception:
            pass

    def _curriculum_stage_label(self) -> str:
        ep = self.train_ep_cnt
        if ep < 150:
            return "warmup_stable"
        if ep < 500:
            return "mid_pressure"
        if ep < 900:
            return "late_speedup_survival"
        return "hard_generalization"

    def _save_eval_extractor_snapshot(self, step: int) -> None: 
        rand = random.random()

        extractor = getattr(self.agent, "extractor", None)
        if extractor is None or rand > 0.2:
            return

        pid = os.getpid()
        _EVAL_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
        np.save(_EVAL_SNAPSHOT_DIR / f"{step:07d}_{pid}_map_full.npy", extractor.map_full)
        np.save(_EVAL_SNAPSHOT_DIR / f"{step:07d}_{pid}_visit_coverage.npy", extractor.visit_coverage)


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _accumulate_reward_info(accum: dict, r_info: dict) -> None:
    """Add numeric entries of r_info into accum (in-place)."""
    for k, v in r_info.items():
        if isinstance(v, (int, float)):
            accum[k] = accum.get(k, 0.0) + float(v)
