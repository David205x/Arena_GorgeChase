import torch
from torch.nn import functional as F
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
    BaseAgent,
)

from collections import deque
import random
import numpy as np
from kaiwu_agent.utils.common_func import attached
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.conf.conf import Config
from agent_ppo.model.model import NetworkModelActor
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.definition import SampleData, ObsData, ActData, SampleManager


def random_choice(p):
    r = random.random() * sum(p)
    s = 0
    for i in range(len(p)):
        if r > s and r <= s + p[i]:
            return i, p[i]
        s += p[i]
    return len(p) - 1, p[len(p) - 1]


_support = torch.linspace(
    Config.VF_MIN,
    Config.VF_MAX,
    Config.VF_BINS + 1,
    dtype=torch.float32,
)
_centers = (_support[:-1] + _support[1:]) / 2


def logits_to_value(logits: torch.Tensor) -> torch.Tensor:
    value_probs = F.softmax(logits, dim=-1)
    return torch.sum(value_probs * _centers, dim=-1)


@attached
class Agent(BaseAgent):
    def __init__(self, agent_type='player', device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)

        self.model = NetworkModelActor()
        self.model = self.model.to(memory_format=torch.channels_last)
        self.algorithm = Algorithm(device=device, logger=logger, monitor=monitor)
        self.preprocessor = Preprocessor()
        self.sample_manager = SampleManager()
        self.win_history = deque(maxlen=100)
        self.logger = logger
        self.reset()

    def update_win_rate(self, is_win):
        self.win_history.append(is_win)
        return (
            sum(self.win_history) / len(self.win_history)
            if len(self.win_history) > 10
            else 0
        )

    def _predict(self, obs, legal_action):
        with torch.no_grad():
            inputs = self.model.format_data(obs, legal_action)
            prob, value_logits = self.model(*inputs)
            value = logits_to_value(value_logits)

        np_output_list = [
            prob.numpy().flatten(),
            value.numpy().flatten(),
        ]

        return np_output_list

    def predict_process(self, obs, legal_action):
        obs = np.array([obs])
        legal_action = np.array([legal_action])
        probs, value = self._predict(obs, legal_action)
        return probs, value

    def observation_process(self, obs):
        feature, legal_action = self.preprocessor.obs_process(obs)
        return ObsData(
            feature=feature,
            legal_action=legal_action,
        )

    def reward_process(self, extra_info, _extra_info):
        reward = self.preprocessor.reward_process(extra_info, _extra_info)
        return reward

    @predict_wrapper
    def predict(self, list_obs_data):
        feature = list_obs_data[0].feature
        legal_action = list_obs_data[0].legal_action
        probs, value = self.predict_process(feature, legal_action)
        action, prob = random_choice(probs)
        return [ActData(probs=probs, value=value, action=action, prob=prob)]

    def action_process(self, act_data):
        return act_data.action

    @exploit_wrapper
    def exploit(self, observation):
        obs_data = self.observation_process(observation['obs'])
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        probs, value = self.predict_process(feature, legal_action)
        action, prob = random_choice(probs)
        act = self.action_process(
            ActData(probs=probs, value=value, action=action, prob=prob)
        )
        return act

    def reset(self):
        self.preprocessor.reset()
        self.last_prob = 0
        self.last_action = -1

    @learn_wrapper
    def learn(self, list_sample_data):
        self.algorithm.learn(list_sample_data)

    @save_model_wrapper
    def save_model(self, path=None, id='1'):
        """保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段"""
        model_file_path = f'{path}/model.ckpt-{str(id)}.pkl'
        torch.save(self.algorithm.model.state_dict(), model_file_path)
        self.logger.info(f'save model {model_file_path} successfully')

    @load_model_wrapper
    def load_model(self, path=None, id='1'):
        """加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致"""
        model_file_path = f'{path}/model.ckpt-{str(id)}.pkl'
        self.model.load_state_dict(torch.load(model_file_path, map_location='cpu'))
        self.logger.info(f'load model {model_file_path} successfully')
