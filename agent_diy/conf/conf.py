#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


import math


# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:
    """****************************** 奖励权重 ******************************"""
    REWARD_WEIGHTS = {
        # 每步给予的负常数惩罚, 防止智能体赖着不去终点
        'reward_step': -0.002,
        # 闪现给予惩罚, 防止智能体乱交闪
        'reward_flash': -0.1,
        # 获取Buff给予微弱奖励
        'reward_buff': 0.01,
        # 获取宝箱给予大额奖励
        'reward_treasure': 1.0,
        # 鼓励智能体靠近L1距离最近的宝箱
        'reward_treasure_dist': 0.02,
        # 鼓励智能体靠近终点
        'reward_end_dist': 0.02,
        # 若走在已经过的位置, 给予惩罚, 防止智能体赖在一个地方不动
        'reward_memory': -0.2,
        # 如果行动前后没有发生位置改变(撞墙), 给予惩罚
        'reward_stiff': -0.1,
        # 到达终点的奖励
        'reward_win': 0.5,
        # 超时惩罚, 建议不给
        'reward_lose': -0.0,
    }
    """****************************** 事件间隔 ******************************"""
    # Logger报告间隔 (秒)
    LOG_INTERVAL = 15
    # 产出模型间隔 (秒)
    SAVE_INTERVAL = 900
    # 读取最新模型间隔 (step)
    LOAD_MODEL_INTERVAL = 256
    """****************************** 算法超参 ******************************"""
    # RL中的回报折扣GAMMA
    GAMMA = 0.997
    # GAE Lambda, 建议0.95
    TDLAMBDA = 0.95
    # PPO clip系数, 越低则越不接受差异大的数据, 建议0.1 ~ 0.2
    CLIP_PARAM = 0.2
    # Dual-clip系数, 建议3.0
    DUAL_CLIP = 3.0
    # 价值函数损失权重
    VALUE_COEF = 0.5
    # 熵正则化权重 (鼓励策略维持高熵, 保证探索能力), 建议5e-4 (别改太大)
    ENTROPY_COEF = 5e-4
    # 优势归一化
    # 建议打开优势归一化: 否则如果优势估计的绝对值太大, 其引导的一次"剧烈"的策略梯度更新可能会导致策略分布坍塌
    WITH_ADV_NORM = True
    """此处我使用HL-Gauss Loss来优化价值网络: https://arxiv.org/abs/2403.03950"""
    # 价值分布的支撑数量, 建议51或101
    VF_BINS = 101
    # 价值下界 (这里很重要, 如果实际的学习目标越界了是学习不了的)
    VF_MIN = -50.0
    # 价值上界 (这里很重要, 如果实际的学习目标越界了是学习不了的)
    VF_MAX = 50.0
    # 价值目标的"坍塌"系数, 建议0.5 ~ 0.75
    VF_SIGMA = 0.75
    """****************************** 优化系数 ******************************"""
    # 初始的学习率
    START_LR = 3e-4
    # 梯度裁剪范数, 设置为0则不裁剪, 建议0.5 ~ 5.0
    CLIP_GRAD_NORM = 3.0
    """****************************** 环境参数(勿修改) ******************************"""
    # 智能体视野半径, 高级赛道为5
    VISION_R = 5
    # 智能体完整视野大小: 1格自身位置 + 2 × 视野半径
    VISION_SIZE = 1 + 2 * VISION_R
    # 地图尺寸
    MAP_LEN = 128
    # Buff持续时长
    BUFF_SUSTAIN_TIME = 50
    # Buff冷却时长
    BUFF_COOLDOWN_TIME = 100
    # 闪现冷却时长
    SKILL_COOLDOWN_TIME = 100
    # 动作维度
    ACTION_LEN = 1
    # 动作总数
    ACTION_NUM = 16
    # 奖励维度
    VALUE_NUM = 1

    # 固定终点位置 (暂时不用, 现在的配置是随机终点位置)
    END_POS = 80, 114
    # 固定Buff位置 (暂时不用, 现在的配置是随机Buff位置)
    BUFF_POS = 49, 28
    """****************************** 特征维度 ******************************"""
    # 标量特征维度 (obs.construct_obs_scaler 输出)
    SCALAR_DIM = 134
    # 局部地图: 8通道 × 21 × 21
    LOCAL_CH = 8
    LOCAL_H = VISION_SIZE        # 21
    LOCAL_W = VISION_SIZE        # 21
    LOCAL_FLAT = LOCAL_CH * LOCAL_H * LOCAL_W   # 3528
    # 全局地图: 4通道 × 64 × 64
    GLOBAL_CH = 4
    GLOBAL_DS = 64
    GLOBAL_FLAT = GLOBAL_CH * GLOBAL_DS * GLOBAL_DS   # 16384
    # 观测打平后总维度 (scalar + local_flat + global_flat)
    OBS_FLAT_DIM = SCALAR_DIM + LOCAL_FLAT + GLOBAL_FLAT   # 20046
    """****************************** 数据流相关(勿修改) ******************************"""
    DATA_SPLIT_SHAPE = [
        OBS_FLAT_DIM,      # obs (scalar + local_map + global_map flattened)
        VALUE_NUM,         # reward
        VALUE_NUM,         # value  (scalar expected value)
        VALUE_NUM,         # td_return
        VALUE_NUM,         # advantage
        ACTION_LEN,        # action
        ACTION_LEN,        # prob   (taken action probability)
        ACTION_NUM,        # legal_action
    ]
    data_len = sum(DATA_SPLIT_SHAPE)

    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = data_len

