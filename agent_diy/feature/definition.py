"""
Data definitions & GAE computation for agent_diy.

ObsData  — inference path, carries structured multi-modal features
ActData  — action output from model
SampleData — flat serialisable sample for aisrv ↔ learner transport
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_diy.conf.conf import Config

# ======================== ObsData ========================
ObsData = create_cls(
    "ObsData",
    feature=None,         # flat np.float32 (OBS_FLAT_DIM,) for framework serialisation
    legal_action=None,    # list[int] length=16
)

# ======================== ActData ========================
ActData = create_cls(
    "ActData",
    action=None,          # [int]     sampled action
    d_action=None,        # [int]     greedy (deterministic) action
    prob=None,            # list      full prob distribution (16,)
    value=None,           # list/ndarray scalar expected value
)

# ======================== SampleData ========================
SampleData = create_cls(
    "SampleData",
    obs=Config.OBS_FLAT_DIM,           # 20046
    legal_action=Config.ACTION_NUM,    # 16
    act=Config.ACTION_LEN,             # 1
    prob=Config.ACTION_NUM,            # 16 full probability distribution
    reward=Config.VALUE_NUM,           # 1
    value=Config.VALUE_NUM,            # 1  scalar expected value
    next_value=Config.VALUE_NUM,       # 1  (filled by sample_process)
    td_return=Config.VALUE_NUM,        # 1  (filled by sample_process)
    advantage=Config.VALUE_NUM,        # 1  (filled by sample_process)
    done=1,                            # 1
)


# ======================== reward_shaping ========================
def reward_shaping(frame_no, score, terminated, truncated, remain_info, _remain_info, obs, _obs):
    """Framework hook — extracts reward computed in observation_process.

    TODO: implement full reward shaping with extractor's extra_info data.
    """
    reward = remain_info.get("reward", [0.0]) if remain_info else [0.0]
    if terminated:
        pass
    if truncated:
        pass
    return reward


# ======================== sample_process ========================
def sample_process(list_sample_data):
    """Fill next_value, compute GAE advantage and td_return.

    Called by the framework after an episode ends.
    """
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gamma = Config.GAMMA
    lamda = Config.TDLAMBDA
    gae = 0.0
    for sample in reversed(list_sample_data):
        done_mask = 1.0 - float(sample.done[0]) if hasattr(sample.done, '__getitem__') else 1.0 - float(sample.done)
        nv = float(sample.next_value[0]) if hasattr(sample.next_value, '__getitem__') else float(sample.next_value)
        v = float(sample.value[0]) if hasattr(sample.value, '__getitem__') else float(sample.value)
        r = float(sample.reward[0]) if hasattr(sample.reward, '__getitem__') else float(sample.reward)

        delta = r + gamma * nv * done_mask - v
        gae = delta + gamma * lamda * done_mask * gae
        sample.advantage = [gae]
        sample.td_return = [gae + v]
