#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs

import numpy as np

from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

from agent_diy.feature.extractor import Extractor
from agent_diy.feature.reward import compute_reward
from agent_diy.monitor.web_control_server import WebControlServer


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_get_training_metrics_time = 0
        self.web_server = WebControlServer(logger=logger)
        self.extractor = Extractor()

    def _split_env_obs(self, env_obs):
        observation = env_obs.get("observation")
        extra_info = env_obs.get("extra_info")
        terminated = bool(env_obs.get("terminated", False))
        truncated = bool(env_obs.get("truncated", False))
        return observation, extra_info, terminated, truncated

    def _build_extractor_view(self, env_obs):
        observation, extra_info, terminated, truncated = self._split_env_obs(env_obs)
        if not isinstance(observation, dict):
            return {
                "available": False,
                "error": {
                    "message": "observation 不可用，无法更新 extractor",
                    "extra_info": extra_info or {},
                },
            }

        self.extractor.update(
            env_obs=observation,
            extra_info=extra_info,
            terminated=terminated,
            truncated=truncated,
        )
        debug_state = self.extractor.build_debug_state()
        # `current/previous` 与 obs/reward_state 内容高度重叠，网页中保留展开后的详细状态即可。
        debug_state.pop("current", None)
        debug_state.pop("previous", None)
        debug_state.pop("obs_state", None)
        debug_state.pop("reward_state", None)
        debug_state.pop("monitor_metrics", None)
        reward_state = self.extractor.build_reward_state()
        _, reward_info = compute_reward(reward_state)
        return {
            "available": True,
            "obs_state": self.extractor.build_obs_state(),
            "reward_state": reward_state,
            "reward_info": reward_info,
            "monitor_metrics": self.extractor.build_monitor_metrics(),
            "debug_state": debug_state,
        }

    def run(self):
        self.web_server.start()
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if metrics is not None:
                    self.logger.info(f"training_metrics is {metrics}")

            env_obs = self.env.reset(self.usr_conf)
            self.web_server.reset_episode_history()
            if handle_disaster_recovery(env_obs, self.logger):
                continue
            self.extractor.reset()
            extractor_view = self._build_extractor_view(env_obs)

            if hasattr(self.agent, "reset"):
                self.agent.reset(env_obs)
            try:
                self.agent.load_model(id="latest")
            except Exception as exc:
                self.logger.info(f"load_model skipped: {exc}")

            self.episode_cnt += 1
            step = 0
            done = False
            self.web_server.publish_obs(
                env_obs,
                self.episode_cnt,
                step,
                "waiting for web action",
                extractor_view=extractor_view,
            )

            while not done:
                action = self.web_server.wait_for_action()
                env_reward, env_obs = self.env.step(action)
                with open('/data/projects/gorge_chase/agent_diy/sample_obs.json', 'w') as f:
                    json.dump(env_obs, f, indent=2)

                if handle_disaster_recovery(env_obs, self.logger):
                    extractor_view = self._build_extractor_view(env_obs)
                    self.web_server.publish_obs(
                        env_obs,
                        self.episode_cnt,
                        step,
                        "disaster recovery triggered",
                        env_reward,
                        True,
                        extractor_view=extractor_view,
                    )
                    break

                step += 1
                terminated = bool(env_obs.get("terminated", False))
                truncated = bool(env_obs.get("truncated", False))
                done = terminated or truncated
                status = "episode finished" if done else "waiting for web action"
                extractor_view = self._build_extractor_view(env_obs)
                self.web_server.publish_obs(
                    env_obs,
                    self.episode_cnt,
                    step,
                    status,
                    env_reward,
                    done,
                    extractor_view=extractor_view,
                )


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env, agent = envs[0], agents[0]
    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return
    EpisodeRunner(env=env, agent=agent, usr_conf=usr_conf, logger=logger, monitor=monitor).run()

