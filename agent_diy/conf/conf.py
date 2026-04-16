#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""


# 配置，包含维度设置，算法参数设置，文件的最后一些配置是开悟平台使用不要改动
class Config:
    """****************************** 事件间隔 ******************************"""
    # Logger报告间隔 (秒)
    LOG_INTERVAL = 30
    # 产出模型间隔 (秒)
    SAVE_INTERVAL = 1200
    # 读取最新模型间隔 (step)
    LOAD_MODEL_INTERVAL = 256
    """****************************** 算法超参 ******************************"""
    # RL中的回报折扣GAMMA (definition.py 中 GAE 计算使用)
    GAMMA = 0.997
    # GAE Lambda (definition.py 中 GAE 计算使用)
    TDLAMBDA = 0.95
    """****************************** 环境参数(勿修改) ******************************"""
    # 智能体视野半径
    VISION_R = 10
    # 智能体完整视野大小: 1格自身位置 + 2 × 视野半径
    VISION_SIZE = 1 + 2 * VISION_R   # 21
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
        ACTION_NUM,        # prob   (full probability distribution)
        ACTION_NUM,        # legal_action
    ]
    data_len = sum(DATA_SPLIT_SHAPE)

    # learner上reverb样本的输入维度
    # **注意**，此项必须正确配置，应该与definition.py中的NumpyData2SampleData函数数据对齐，否则可能报样本维度错误
    SAMPLE_DIM = data_len

