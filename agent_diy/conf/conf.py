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
    # 最大的宝箱数量 (如果宝箱数量超过它, 则不会被纳入特征)
    NUM_TREASURE_MAX = 9
    # 最大的Organ数量: 终点 + Buff + 宝箱
    NUM_ORGAN_MAX = 1 + 1 + NUM_TREASURE_MAX
    # Buff剩余生效时间的尺度编码大小
    DIM_BUFF_SUSTAIN_ENCODE = math.floor(math.sqrt(BUFF_SUSTAIN_TIME))
    # Buff冷却时间的尺度编码大小
    DIM_BUFF_COOLDOWN_ENCODE = math.floor(math.sqrt(BUFF_COOLDOWN_TIME - 1))
    # 闪现冷却时间的尺度编码大小
    DIM_SKILL_COOLDOWN_ENCODE = math.floor(math.sqrt(SKILL_COOLDOWN_TIME - 1))
    # Organ与智能体之间L2距离的尺度编码大小
    DIM_RELATIVE_DIST_ENCODE = math.floor(math.sqrt(1.41 * MAP_LEN))
    # 坐标二进制编码的长度
    DIM_COORD_BINARY = (MAP_LEN - 1).bit_length()
    """********** Organ特征 **********"""
    DIM_FEATURE_ORGAN = (
        # Organ种类的onehot编码 (宝箱/Buff/终点)
        3
        # Organ状态 (绝对坐标是否已知, 是否可拾取(这条是专门针对Buff的))
        + 2
        # 相对距离 (归一化的距离, 归一化的相对x/z坐标, 归一化的绝对x/z坐标)
        + 5
        # L2距离的尺度编码
        + (DIM_RELATIVE_DIST_ENCODE + 1)
        # Organ绝对x/z坐标的二进制编码
        + 2 * (DIM_COORD_BINARY)
        # 相对方向的OneHot编码
        + 8
        # 相对距离的OneHot编码
        + 5
    )
    """********** 向量特征 **********"""
    DIM_FEATURE_VECTOR = (
        # 智能体绝对x/z坐标的二进制编码
        2 * DIM_COORD_BINARY
        # 智能体自身状态 (闪现状态, Buff状态, 闪现冷却, Buff冷却)
        + 4
        # Buff剩余生效时间的尺度编码
        + (DIM_BUFF_SUSTAIN_ENCODE + 1)
        # Buff冷却时间的尺度编码
        + (DIM_BUFF_COOLDOWN_ENCODE + 1)
        # 闪现冷却时间的尺度编码
        + (DIM_SKILL_COOLDOWN_ENCODE + 1)
        # 剩余宝箱数量的OneHot编码
        + (NUM_TREASURE_MAX + 1)
    )
    """********** 视觉特征 **********"""
    # 视觉特征通道数 (障碍, 记忆, 宝箱, Buff, 终点)
    DIM_VISION_CHANNELS = 5
    DIM_FEATURE_VISION = VISION_SIZE * VISION_SIZE * DIM_VISION_CHANNELS
    """********** 完整特征 **********"""
    FEATURES = [
        # 向量特征
        DIM_FEATURE_VECTOR,
        # N个Organ特征
        NUM_ORGAN_MAX * DIM_FEATURE_ORGAN,
        # N个Organ特征掩码, 指示了当前位置的Organ是否可用
        NUM_ORGAN_MAX,
        # 视觉特征
        DIM_FEATURE_VISION,
    ]
    """****************************** 数据流相关(勿修改) ******************************"""
    # 开悟框架会先将特征flatten以便存储, 而优化/推理时则需要先将展平的特征恢复到原始形状
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    # PPO专用的样本布局, 构造过程可见definition.py
    DATA_SPLIT_SHAPE = [
        # 观测
        FEATURE_LEN,
        # 奖励
        VALUE_NUM,
        # 价值网络输出
        VALUE_NUM,
        # TD(λ)价值估计
        VALUE_NUM,
        # GAE优势估计
        VALUE_NUM,
        # 执行的动作
        ACTION_LEN,
        # 该动作对应的动作概率
        ACTION_LEN,
        # 合法动作列表
        ACTION_NUM,
    ]
    # 一个样本的大小
    data_len = sum(DATA_SPLIT_SHAPE)

    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = data_len

