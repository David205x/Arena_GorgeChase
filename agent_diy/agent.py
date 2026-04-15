"""
Agent for agent_diy: multi-input ConvNeXt + Simba PPO.

Data flow:
    env_obs → Extractor.update() → build_obs_state()
            → obs.construct_obs_scaler()   → (134,)
            → obs.construct_obs_matrix()    → local(8,21,21) + global(4,64,64)
            → flatten & pack into ObsData.feature → (20046,)
    predict:
            → unpack feature → model(scalar, local, global, legal) → probs, value
            → sample action → ActData
    learn:
            → collate list[SampleData] → batch dict → Algorithm.learn()
"""

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np
from kaiwudrl.interface.agent import BaseAgent

from agent_diy.conf.conf import Config
from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.feature.definition import ObsData, ActData
from agent_diy.feature.extractor import Extractor
from agent_diy.feature.obs import construct_obs_scaler, construct_obs_matrix
from agent_diy.feature.reward import compute_reward

# flat obs layout: [scalar | local_flat | global_flat]
_S = Config.SCALAR_DIM                # 134
_L = Config.LOCAL_FLAT                # 3528
_G = Config.GLOBAL_FLAT               # 16384
_LC = Config.LOCAL_CH                  # 8
_LH = Config.LOCAL_H                  # 21
_LW = Config.LOCAL_W                  # 21
_GC = Config.GLOBAL_CH                # 4
_GD = Config.GLOBAL_DS                # 64


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        self.device = device or torch.device("cpu")
        self.logger = logger
        self.monitor = monitor

        self.algorithm = Algorithm(device=self.device, logger=logger, monitor=monitor)
        self.model = self.algorithm.model

        self.extractor = Extractor()
        self.last_action = -1

        super().__init__(agent_type, device, logger, monitor)

    # ------------------------------------------------------------------ reset
    def reset(self, env_obs=None):
        self.extractor.reset()
        self.last_action = -1

    # ------------------------------------------------------------------ observation
    def observation_process(self, env_obs):
        """Convert raw env_obs → ObsData + remain_info."""
        observation = env_obs["observation"]
        extra_info = env_obs.get("extra_info")
        terminated = bool(env_obs.get("terminated", False))
        truncated = bool(env_obs.get("truncated", False))

        self.extractor.update(
            env_obs=observation,
            extra_info=extra_info,
            terminated=terminated,
            truncated=truncated,
            last_action=self.last_action,
        )
        data = self.extractor.build_obs_state()

        scalar = construct_obs_scaler(data)       # (134,)
        matrices = construct_obs_matrix(data)      # local(8,21,21), global(4,64,64)

        flat_feature = np.concatenate([
            scalar,
            matrices["local"].reshape(-1),
            matrices["global"].reshape(-1),
        ]).astype(np.float32)

        legal_action = data.get("legal_action")
        if legal_action is None:
            legal_action = [1] * Config.ACTION_NUM
        else:
            legal_action = [int(v) for v in legal_action]

        obs_data = ObsData(
            feature=list(flat_feature),
            legal_action=legal_action,
        )

        reward_data = self.extractor.build_reward_state()
        reward_value, reward_info = compute_reward(reward_data)
        remain_info = {"reward": [reward_value], "reward_info": reward_info}
        return obs_data, remain_info

    # ------------------------------------------------------------------ predict
    def predict(self, list_obs_data):
        """Stochastic inference (training exploration)."""
        obs_data = list_obs_data[0]
        feature = np.array(obs_data.feature, dtype=np.float32)
        legal_action = np.array(obs_data.legal_action, dtype=np.float32)

        probs_np, value_scalar = self._run_model(feature, legal_action)

        action = self._sample(probs_np, use_max=False)
        d_action = self._sample(probs_np, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(probs_np),
                value=[value_scalar],
            )
        ]

    # ------------------------------------------------------------------ exploit
    def exploit(self, env_obs):
        """Greedy inference (evaluation)."""
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])
        return self.action_process(act_data[0], is_stochastic=False)

    # ------------------------------------------------------------------ action
    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return int(action[0])

    # ------------------------------------------------------------------ learn
    def learn(self, list_sample_data):
        batch = self._collate(list_sample_data)
        self.algorithm.learn(batch)

    # ------------------------------------------------------------------ save / load
    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{str(id)}.pkl"
        self.model.load_state_dict(
            torch.load(model_file_path, map_location=self.device)
        )
        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    # ================================================================== private
    def _run_model(self, flat_feature, legal_action_np):
        """Run model forward on a single observation.

        Returns:
            probs_np:     (ACTION_NUM,) probability distribution
            value_scalar: float expected value
        """
        self.model.set_eval_mode()

        scalar_t = torch.from_numpy(flat_feature[:_S]).unsqueeze(0).to(self.device)
        local_t = torch.from_numpy(
            flat_feature[_S:_S + _L].reshape(_LC, _LH, _LW)
        ).unsqueeze(0).to(self.device)
        global_t = torch.from_numpy(
            flat_feature[_S + _L:].reshape(_GC, _GD, _GD)
        ).unsqueeze(0).to(self.device)
        legal_t = torch.from_numpy(legal_action_np).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, value_logits = self.model(scalar_t, local_t, global_t, legal_t)
            value = self.model.value_expected(value_logits)

        probs_np = probs.cpu().numpy()[0]
        value_scalar = float(value.cpu().numpy()[0, 0])
        return probs_np, value_scalar

    def _collate(self, list_sample_data):
        """Convert list[SampleData] → batch dict for Algorithm.learn()."""
        obs_flat = torch.stack([
            torch.as_tensor(s.obs, dtype=torch.float32) for s in list_sample_data
        ])
        batch = {
            "scalar":       obs_flat[:, :_S],
            "local_map":    obs_flat[:, _S:_S + _L].reshape(-1, _LC, _LH, _LW),
            "global_map":   obs_flat[:, _S + _L:].reshape(-1, _GC, _GD, _GD),
            "legal_action": torch.stack([torch.as_tensor(s.legal_action, dtype=torch.float32) for s in list_sample_data]),
            "old_action":   torch.stack([torch.as_tensor(s.act, dtype=torch.long) for s in list_sample_data]).view(-1, 1),
            "old_prob":     torch.stack([
                torch.as_tensor(
                    [s.prob[int(s.act[0])] if hasattr(s.act, '__getitem__') else s.prob[int(s.act)]],
                    dtype=torch.float32,
                ) for s in list_sample_data
            ]),
            "reward":       torch.stack([torch.as_tensor(s.reward, dtype=torch.float32) for s in list_sample_data]).squeeze(-1),
            "advantage":    torch.stack([torch.as_tensor(s.advantage, dtype=torch.float32) for s in list_sample_data]).squeeze(-1),
            "td_return":    torch.stack([torch.as_tensor(s.td_return, dtype=torch.float32) for s in list_sample_data]).squeeze(-1),
        }
        return batch

    @staticmethod
    def _sample(probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))
