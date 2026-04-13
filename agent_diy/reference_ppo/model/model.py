import torch
from torch import nn
from torch.nn import functional as F
from agent_ppo.conf.conf import Config
from agent_ppo.model.modules import (
    get_fc_layer,
    ResidualBlock,
    SimbaEncoder,
    ConvNeXtEncoder,
    UnitEncoder,
)

import sys
import os

if os.path.basename(sys.argv[0]) == 'learner.py':
    import torch

    torch.set_num_interop_threads(2)
    torch.set_num_threads(2)
else:
    import torch

    torch.set_num_interop_threads(4)
    torch.set_num_threads(4)


class NetworkModelBase(nn.Module):
    def __init__(self):
        super().__init__()
        # 特征配置参数
        self.data_split_shape = Config.DATA_SPLIT_SHAPE
        self.feature_split_shape = Config.FEATURE_SPLIT_SHAPE
        self.feature_len = Config.FEATURE_LEN
        action_num = Config.ACTION_NUM
        value_num = Config.VF_BINS

        self.entropy_weight = Config.ENTROPY_COEF
        self.value_weight = Config.VALUE_COEF
        self.clip_param = Config.CLIP_PARAM
        self.data_len = Config.data_len

        max_organ_len = Config.NUM_ORGAN_MAX

        self.vector_embed_dim = 256
        self.vision_embed_dim = 256
        self.unit_token_dim = 64
        self.torso_out_dim = 512

        # ****************************** Encoders 编码器 ******************************
        # 视觉编码器为ConvNeXt, 取其输出特征图的channels作为视觉编码
        self.vision_encoder = ConvNeXtEncoder(
            in_channels=5,
            dims=[64, 128, self.vision_embed_dim],
            depths=[1, 2, 1],
            downsample_sizes=[2, 2, 2],
        )
        # 向量观测编码器为Simba Network, 可以看作是MLP Pro Max版
        self.vector_encoder = SimbaEncoder(
            input_dim=Config.DIM_FEATURE_VECTOR,
            hidden_dim=self.vector_embed_dim,
            block_num=2,
        )
        # Organ编码器, 使用一系列MLP对输入的Organ序列进行编码, 其中不可用的Organ编码会被填充全0
        self.organ_encoder = UnitEncoder(
            Config.DIM_FEATURE_ORGAN,
            self.unit_token_dim,
        )
        # ****************************** Torso 躯干 ******************************
        # 将编码器产出的embedding拼接在一起, 然后使用SimbaEncoder产生一个全局信息编码
        embed_dim = (
            self.vector_embed_dim
            + self.vision_embed_dim
            + self.unit_token_dim * max_organ_len
        )
        self.main_encoder = SimbaEncoder(
            input_dim=embed_dim,
            hidden_dim=self.torso_out_dim,
            block_num=2,
        )
        # ****************************** Heads 策略/价值头 ******************************
        self.policy_head = nn.Sequential(
            ResidualBlock(self.torso_out_dim),
            nn.LayerNorm(self.torso_out_dim),
            get_fc_layer(self.torso_out_dim, action_num),
        )
        self.value_head = nn.Sequential(
            ResidualBlock(self.torso_out_dim),
            nn.LayerNorm(self.torso_out_dim),
            get_fc_layer(self.torso_out_dim, value_num),
        )

    def forward(self, feature, legal_action):
        # Unpack Inputs
        vector_input, vision_input, organ_inputs, organ_mask = self.split_data(feature)
        # Encoders
        vector_embed = self.vector_encoder(vector_input)
        vision_embed = self.vision_encoder(vision_input)
        organ_tokens = self.organ_encoder(organ_inputs, organ_mask)
        organ_tokens_flatten = organ_tokens.flatten(-2)
        # Torso
        torso_input = torch.concat(
            [vector_embed, vision_embed, organ_tokens_flatten], dim=-1
        )
        torso_embed = self.main_encoder(torso_input)
        # Policy Output
        policy_logits = self.policy_head(torso_embed)
        policy_logits = self.process_legal_action(policy_logits, legal_action)
        policy_probs = F.softmax(policy_logits, dim=-1)
        # Value Output
        # 注意这里因为我用了值分布预测, 策略网络的输出是一个概率分布
        # 参考: https://arxiv.org/abs/2403.03950
        value_logits = self.value_head(torso_embed)

        return policy_probs, value_logits

    def process_legal_action(self, label, legal_action):
        """
        在策略网络输出的logits当中'不合法动作'的位置减去一个巨大的数字
        这样子确保后续使用Softmax将logits转换为概率值的时候, 非法动作的概率一定是0
        """
        label_max, _ = torch.max(label * legal_action, 1, True)
        label = label - label_max
        label = label * legal_action
        label = label + 1e5 * (legal_action - 1)
        return label

    def split_data(self, feature):
        vector_input, organ_inputs, organ_mask, vision_input = torch.split(
            feature,
            Config.FEATURES,
            dim=-1,
        )
        organ_inputs = organ_inputs.view(
            -1,
            Config.NUM_ORGAN_MAX,
            Config.DIM_FEATURE_ORGAN,
        )
        organ_mask = organ_mask.view(
            -1,
            Config.NUM_ORGAN_MAX,
        )
        vision_input = vision_input.view(
            -1,
            Config.DIM_VISION_CHANNELS,
            Config.VISION_SIZE,
            Config.VISION_SIZE,
        )
        return vector_input, vision_input, organ_inputs, organ_mask


class NetworkModelActor(NetworkModelBase):
    def format_data(self, obs, legal_action):
        return (
            torch.tensor(obs).to(torch.float32),
            torch.tensor(legal_action).to(torch.float32),
        )


class NetworkModelLearner(NetworkModelBase):
    def format_data(self, datas):
        return datas.view(-1, self.data_len).float().split(self.data_split_shape, dim=1)

    def forward(self, data_list, inference=False):
        feature = data_list[0]
        legal_action = data_list[-1]
        return super().forward(feature, legal_action)
