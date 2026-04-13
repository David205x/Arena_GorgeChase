#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors
"""

import numpy as np
import torch
from torch.nn import functional as F
import os
import time
from agent_diy.model.model import NetworkModelLearner
from agent_diy.conf.conf import Config
from agent_diy.algorithm.objectives import HLGaussLoss


class Algorithm:
    def __init__(self, model, optimizer, scheduler, device=None, logger=None, monitor=None):
        # Hyperparams
        self.label_size = Config.ACTION_NUM
        self.var_beta = Config.ENTROPY_COEF
        self.vf_coef = Config.VALUE_COEF
        self.clip_param = Config.CLIP_PARAM
        self.dual_clip = Config.DUAL_CLIP
        self.clip_high = 1.0 + self.clip_param
        self.clip_low = 1.0 / self.clip_high
        self.grad_clip = Config.CLIP_GRAD_NORM
        # Model
        self.device = device
        self.model = NetworkModelLearner().to(self.device)
        self.model = self.model.to(memory_format=torch.channels_last)
        self.lr = Config.START_LR
        self.optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8
        )
        self.parameters = [
            p
            for param_group in self.optimizer.param_groups
            for p in param_group['params']
        ]
        self.hl_gauss_loss = HLGaussLoss(
            min_value=Config.VF_MIN,
            max_value=Config.VF_MAX,
            num_bins=Config.VF_BINS,
            sigma=Config.VF_SIGMA,
            device=self.device,
        )
        # Monitor
        self.logger = logger
        self.monitor = monitor
        self.last_report_monitor_time = 0

    def learn(self, list_sample_data):
        results = {}
        self.model.train()
        self.optimizer.zero_grad()

        list_npdata = [
            torch.as_tensor(sample_data.npdata, device=self.device)
            for sample_data in list_sample_data
        ]
        _input_datas = torch.stack(list_npdata, dim=0)
        data_list = self.model.format_data(_input_datas)
        rst_list = self.model(data_list)
        total_loss, info_list = self.compute_loss(data_list, rst_list)
        results['total_loss'] = total_loss.item()
        total_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.parameters, self.grad_clip)
        self.optimizer.step()

        _info_list = []
        for info in info_list:
            if isinstance(info, list):
                _info = [
                    i.detach().cpu().item() if torch.is_tensor(i) else i for i in info
                ]
            else:
                _info = (
                    info.detach().mean().cpu().item() if torch.is_tensor(info) else info
                )
            _info_list.append(_info)

        if self.monitor:
            now = time.time()
            if now - self.last_report_monitor_time >= Config.LOG_INTERVAL:
                results['value_loss'] = round(_info_list[1], 2)
                results['policy_loss'] = round(_info_list[2], 2)
                results['entropy_loss'] = round(_info_list[3], 2)

                results['reward'] = _info_list[-1]

                results['diy_3'] = round(_info_list[4], 2)
                results['diy_4'] = round(_info_list[5], 2)
                results['diy_5'] = round(_info_list[0], 2)
                self.monitor.put_data({os.getpid(): results})
                self.last_report_monitor_time = now

    def compute_loss(self, data_list, rst_list):
        (
            feature,
            reward,
            old_value,
            tdret,
            adv,
            old_action,
            old_prob,
            legal_action,
        ) = data_list
        new_probs, value_logits = rst_list
        # -------------------- Value loss --------------------
        # (基于交叉熵的)价值损失
        # 参考: https://arxiv.org/abs/2403.03950
        value_entropy = self.hl_gauss_loss(value_logits, tdret)
        value_loss = value_entropy.mean()
        # -------------------- Entropy loss --------------------
        # 熵损失
        entropy = -new_probs * torch.log(new_probs.clamp(min=1e-9, max=1))
        entropy_loss = entropy.mean()
        # -------------------- Policy ratio --------------------
        # 新旧策略的动作概率比值
        # 此处将除法操作转换为"log之差的exp", 以减小精度误差
        new_prob = torch.gather(new_probs, dim=-1, index=old_action.long())
        new_log_prob = torch.log(new_prob.clamp(min=1e-9, max=1))
        old_log_prob = torch.log(old_prob.clamp(min=1e-9, max=1))
        ratio = torch.exp(new_log_prob - old_log_prob)
        # 计算之前提前裁剪掉过大的数据, 以免发生数值溢出
        ratio = torch.clamp(ratio, 0.0, 3.0)
        # 优势归一化, 避免过大的策略梯度估计摧毁策略网络
        adv_std, adv_mean = torch.std_mean(adv)
        if Config.WITH_ADV_NORM:
            adv = (adv - adv_mean) / torch.clamp_min(
                adv_std, 1e-7
            )  # normalize advantage
        # PPO损失
        surr1 = ratio * adv
        surr2 = ratio.clamp(self.clip_low, self.clip_high) * adv
        if self.dual_clip > 0:
            # 使用Dual-Clip
            clip1 = torch.minimum(surr1, surr2)
            clip2 = torch.maximum(clip1, self.dual_clip * adv)
            clipped_objective = -torch.where(adv < 0, clip2, clip1)
        else:
            clipped_objective = -torch.minimum(surr1, surr2)
        policy_loss = clipped_objective.mean()
        with torch.no_grad():
            # 被Clipped掉的数据比率
            clipped = ratio.gt(self.clip_high) | ratio.lt(self.clip_low)
            clipfrac = clipped.float().mean().item()
        # -------------------- Total loss --------------------
        total_loss = (
            policy_loss + self.vf_coef * value_loss - self.var_beta * entropy_loss
        )
        info_list = [
            tdret.mean(),
            value_loss,
            policy_loss,
            entropy_loss,
            clipfrac,
            adv_mean,
            adv_std,
            reward.mean(),
        ]
        return total_loss, info_list

