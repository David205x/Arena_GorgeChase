from agent_ppo.conf.conf import Config
from kaiwu_agent.utils.common_func import create_cls, attached
import numpy as np

# create_cls函数用于动态创建一个类，函数第一个参数为类型名称，剩余参数为类的属性，属性默认值应设为None
ObsData = create_cls(
    'ObsData',
    feature=None,
    legal_action=None,
    reward=None,
)

ActData = create_cls(
    'ActData',
    probs=None,
    value=None,
    target=None,
    predict=None,
    action=None,
    prob=None,
)

SampleData = create_cls('SampleData', npdata=None)

RelativeDistance = {
    'RELATIVE_DISTANCE_NONE': 0,
    'VerySmall': 1,
    'Small': 2,
    'Medium': 3,
    'Large': 4,
    'VeryLarge': 5,
}

RelativeDirection = {
    'East': 1,
    'NorthEast': 2,
    'North': 3,
    'NorthWest': 4,
    'West': 5,
    'SouthWest': 6,
    'South': 7,
    'SouthEast': 8,
}

DirectionAngles = {
    1: 0,
    2: 45,
    3: 90,
    4: 135,
    5: 180,
    6: 225,
    7: 270,
    8: 315,
}


class SampleManager:
    """
    SampleManager在环境交互过程中顺序存储各决策步对应的数据, 形成一个轨迹(Trajectory)
    在对局完成后, 它将该轨迹逆序遍历以计算GAE和TD(λ)的价值估计
    """

    def __init__(
        self,
        gamma=0.99,
        tdlambda=0.95,
    ):
        self.gamma = Config.GAMMA
        self.tdlambda = Config.TDLAMBDA

        self.feature = []
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def compute_gae(self):
        last_gae = 0
        next_val = 0
        for i in reversed(range(self.count)):
            reward = self.reward[i]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.tdlambda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val
            next_val = val

    def finalize_trajectory(self):
        self.compute_gae()
        traj = self._convert2samples()
        self.samples = []
        return traj

    def _convert2samples(self):
        feature = np.array(self.feature).transpose()
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward).transpose()
        value = np.array(self.value).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()

        data = np.concatenate(
            [feature, reward, value, tdlamret, adv, actions, probs, legal_action]
        ).transpose()

        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))

        return samples


@attached
def SampleData2NumpyData(g_data):
    return g_data.npdata


@attached
def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)
