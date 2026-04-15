#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Monitor panel configuration builder for Gorge Chase (agent_diy).
峡谷追猎监控面板配置构建器（agent_diy 版）。

Groups
------
1. 算法指标      — PPO 训练过程核心指标
2. 奖励总览      — 奖励加权合并结果与 alpha 系数
3. 生存奖励子项   — survival bucket 各分项
4. 探索奖励子项   — explore bucket 各分项
5. 终局奖励子项   — terminal reward 各分项
6. 对局全局状态   — 得分、步数、资源等全局统计
7. 对局阶段状态   — 加速前后 / 各 stage 步数分布
8. 终态危险分析   — 终局时刻的危险度、空间、闪现状态
9. 全局平均状态   — 全局路径信号均值
"""

from kaiwudrl.common.monitor.monitor_config_builder import MonitorConfigBuilder


def build_monitor():
    """
    # This function is used to create monitoring panel configurations for custom indicators.
    # 该函数用于创建自定义指标的监控面板配置。
    """
    m = MonitorConfigBuilder()
    m.title("峡谷追猎")

    # ================================================================
    # Group 1: 算法指标
    # ================================================================
    m.add_group(group_name="算法指标", group_name_en="algorithm")
    m.add_panel(name="累积回报", name_en="reward", type="line")
    m.add_metric(metrics_name="reward", expr="avg(reward{})")
    m.end_panel()
    m.add_panel(name="总损失", name_en="total_loss", type="line")
    m.add_metric(metrics_name="total_loss", expr="avg(total_loss{})")
    m.end_panel()
    m.add_panel(name="策略损失", name_en="policy_loss", type="line")
    m.add_metric(metrics_name="policy_loss", expr="avg(policy_loss{})")
    m.end_panel()
    m.add_panel(name="价值损失", name_en="value_loss", type="line")
    m.add_metric(metrics_name="value_loss", expr="avg(value_loss{})")
    m.end_panel()
    m.add_panel(name="策略熵", name_en="entropy", type="line")
    m.add_metric(metrics_name="entropy", expr="avg(entropy{})")
    m.end_panel()
    m.add_panel(name="clip比例", name_en="clipfrac", type="line")
    m.add_metric(metrics_name="clipfrac", expr="avg(clipfrac{})")
    m.end_panel()
    m.add_panel(name="advantage均值", name_en="adv_mean", type="line")
    m.add_metric(metrics_name="adv_mean", expr="avg(adv_mean{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 2: 奖励总览
    # ================================================================
    m.add_group(group_name="奖励总览", group_name_en="reward_overview")
    m.add_panel(name="total", name_en="reward_total", type="line")
    m.add_metric(metrics_name="reward_total", expr="avg(reward_total{})")
    m.end_panel()
    m.add_panel(name="alpha", name_en="reward_alpha", type="line")
    m.add_metric(metrics_name="reward_alpha", expr="avg(reward_alpha{})")
    m.end_panel()
    m.add_panel(name="survival", name_en="reward_survival", type="line")
    m.add_metric(metrics_name="reward_survival", expr="avg(reward_survival{})")
    m.end_panel()
    m.add_panel(name="survival_weighted", name_en="reward_survival_weighted", type="line")
    m.add_metric(metrics_name="reward_survival_weighted", expr="avg(reward_survival_weighted{})")
    m.end_panel()
    m.add_panel(name="explore", name_en="reward_explore", type="line")
    m.add_metric(metrics_name="reward_explore", expr="avg(reward_explore{})")
    m.end_panel()
    m.add_panel(name="explore_weighted", name_en="reward_explore_weighted", type="line")
    m.add_metric(metrics_name="reward_explore_weighted", expr="avg(reward_explore_weighted{})")
    m.end_panel()
    m.add_panel(name="terminal", name_en="reward_terminal", type="line")
    m.add_metric(metrics_name="reward_terminal", expr="avg(reward_terminal{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 3: 生存奖励子项
    # ================================================================
    m.add_group(group_name="生存奖励子项", group_name_en="survival_components")
    m.add_panel(name="s_step_score", name_en="s_step_score", type="line")
    m.add_metric(metrics_name="s_step_score", expr="avg(s_step_score{})")
    m.end_panel()
    m.add_panel(name="s_monster_dist", name_en="s_monster_dist", type="line")
    m.add_metric(metrics_name="s_monster_dist", expr="avg(s_monster_dist{})")
    m.end_panel()
    m.add_panel(name="s_encircle", name_en="s_encircle", type="line")
    m.add_metric(metrics_name="s_encircle", expr="avg(s_encircle{})")
    m.end_panel()
    m.add_panel(name="s_space", name_en="s_space", type="line")
    m.add_metric(metrics_name="s_space", expr="avg(s_space{})")
    m.end_panel()
    m.add_panel(name="s_topology", name_en="s_topology", type="line")
    m.add_metric(metrics_name="s_topology", expr="avg(s_topology{})")
    m.end_panel()
    m.add_panel(name="s_no_move", name_en="s_no_move", type="line")
    m.add_metric(metrics_name="s_no_move", expr="avg(s_no_move{})")
    m.end_panel()
    m.add_panel(name="s_flash_escape", name_en="s_flash_escape", type="line")
    m.add_metric(metrics_name="s_flash_escape", expr="avg(s_flash_escape{})")
    m.end_panel()
    m.add_panel(name="s_flash_low", name_en="s_flash_low", type="line")
    m.add_metric(metrics_name="s_flash_low", expr="avg(s_flash_low{})")
    m.end_panel()
    m.add_panel(name="s_revisit", name_en="s_revisit", type="line")
    m.add_metric(metrics_name="s_revisit", expr="avg(s_revisit{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 4: 探索奖励子项
    # ================================================================
    m.add_group(group_name="探索奖励子项", group_name_en="explore_components")
    m.add_panel(name="e_treasure_score", name_en="e_treasure_score", type="line")
    m.add_metric(metrics_name="e_treasure_score", expr="avg(e_treasure_score{})")
    m.end_panel()
    m.add_panel(name="e_treasure_approach", name_en="e_treasure_approach", type="line")
    m.add_metric(metrics_name="e_treasure_approach", expr="avg(e_treasure_approach{})")
    m.end_panel()
    m.add_panel(name="e_map_explore", name_en="e_map_explore", type="line")
    m.add_metric(metrics_name="e_map_explore", expr="avg(e_map_explore{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 5: 终局奖励子项
    # ================================================================
    m.add_group(group_name="终局奖励子项", group_name_en="terminal_components")
    m.add_panel(name="t_complete_bonus", name_en="t_complete_bonus", type="line")
    m.add_metric(metrics_name="t_complete_bonus", expr="avg(t_complete_bonus{})")
    m.end_panel()
    m.add_panel(name="t_death_stage_pen", name_en="t_death_stage_pen", type="line")
    m.add_metric(metrics_name="t_death_stage_pen", expr="avg(t_death_stage_pen{})")
    m.end_panel()
    m.add_panel(name="t_encircle_pen", name_en="t_encircle_pen", type="line")
    m.add_metric(metrics_name="t_encircle_pen", expr="avg(t_encircle_pen{})")
    m.end_panel()
    m.add_panel(name="t_dead_end_pen", name_en="t_dead_end_pen", type="line")
    m.add_metric(metrics_name="t_dead_end_pen", expr="avg(t_dead_end_pen{})")
    m.end_panel()
    m.add_panel(name="t_same_side_reduce", name_en="t_same_side_reduce", type="line")
    m.add_metric(metrics_name="t_same_side_reduce", expr="avg(t_same_side_reduce{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 6: 对局全局状态
    # ================================================================
    m.add_group(group_name="对局全局状态", group_name_en="episode_global")
    m.add_panel(name="episode_steps", name_en="episode_steps", type="line")
    m.add_metric(metrics_name="episode_steps", expr="avg(episode_steps{})")
    m.end_panel()
    m.add_panel(name="episode_total_score", name_en="episode_total_score", type="line")
    m.add_metric(metrics_name="episode_total_score", expr="avg(episode_total_score{})")
    m.end_panel()
    m.add_panel(name="episode_step_score", name_en="episode_step_score", type="line")
    m.add_metric(metrics_name="episode_step_score", expr="avg(episode_step_score{})")
    m.end_panel()
    m.add_panel(name="episode_treasure_score", name_en="episode_treasure_score", type="line")
    m.add_metric(metrics_name="episode_treasure_score", expr="avg(episode_treasure_score{})")
    m.end_panel()
    m.add_panel(name="episode_treasures", name_en="episode_treasures", type="line")
    m.add_metric(metrics_name="episode_treasures", expr="avg(episode_treasures{})")
    m.end_panel()
    m.add_panel(name="episode_buffs", name_en="episode_buffs", type="line")
    m.add_metric(metrics_name="episode_buffs", expr="avg(episode_buffs{})")
    m.end_panel()
    m.add_panel(name="episode_flash_count", name_en="episode_flash_count", type="line")
    m.add_metric(metrics_name="episode_flash_count", expr="avg(episode_flash_count{})")
    m.end_panel()
    m.add_panel(name="speedup_reached", name_en="speedup_reached", type="line")
    m.add_metric(metrics_name="speedup_reached", expr="avg(speedup_reached{})")
    m.end_panel()
    m.add_panel(name="terminated", name_en="terminated", type="line")
    m.add_metric(metrics_name="terminated", expr="avg(terminated{})")
    m.end_panel()
    m.add_panel(name="completed", name_en="completed", type="line")
    m.add_metric(metrics_name="completed", expr="avg(completed{})")
    m.end_panel()
    m.add_panel(name="abnormal_truncated", name_en="abnormal_truncated", type="line")
    m.add_metric(metrics_name="abnormal_truncated", expr="avg(abnormal_truncated{})")
    m.end_panel()
    m.add_panel(name="post_terminated", name_en="post_terminated", type="line")
    m.add_metric(metrics_name="post_terminated", expr="avg(post_terminated{})")
    m.end_panel()
    m.add_panel(name="final_stage", name_en="final_stage", type="line")
    m.add_metric(metrics_name="final_stage", expr="avg(final_stage{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 7: 对局阶段状态
    # ================================================================
    m.add_group(group_name="对局阶段状态", group_name_en="episode_phase")
    m.add_panel(name="pre_steps", name_en="pre_steps", type="line")
    m.add_metric(metrics_name="pre_steps", expr="avg(pre_steps{})")
    m.end_panel()
    m.add_panel(name="post_steps", name_en="post_steps", type="line")
    m.add_metric(metrics_name="post_steps", expr="avg(post_steps{})")
    m.end_panel()
    m.add_panel(name="stage1_steps", name_en="stage1_steps", type="line")
    m.add_metric(metrics_name="stage1_steps", expr="avg(stage1_steps{})")
    m.end_panel()
    m.add_panel(name="stage2_steps", name_en="stage2_steps", type="line")
    m.add_metric(metrics_name="stage2_steps", expr="avg(stage2_steps{})")
    m.end_panel()
    m.add_panel(name="stage3_steps", name_en="stage3_steps", type="line")
    m.add_metric(metrics_name="stage3_steps", expr="avg(stage3_steps{})")
    m.end_panel()
    m.add_panel(name="flash_escape_success_count", name_en="flash_escape_success_count", type="line")
    m.add_metric(metrics_name="flash_escape_success_count", expr="avg(flash_escape_success_count{})")
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 8: 终态危险分析
    # ================================================================
    m.add_group(group_name="终态危险分析", group_name_en="episode_terminal_state")
    m.add_panel(name="final_nearest_monster_path_dist", name_en="final_nearest_monster_path_dist", type="line")
    m.add_metric(
        metrics_name="final_nearest_monster_dist_est",
        expr="avg(final_nearest_monster_dist_est{})",
    )
    m.end_panel()
    m.add_panel(name="final_capture_margin_path", name_en="final_capture_margin_path", type="line")
    m.add_metric(
        metrics_name="final_capture_margin_path_estimate",
        expr="avg(final_capture_margin_path_estimate{})",
    )
    m.end_panel()
    m.add_panel(name="final_encirclement_cosine", name_en="final_encirclement_cosine", type="line")
    m.add_metric(
        metrics_name="final_encirclement_path_cosine_estimate",
        expr="avg(final_encirclement_path_cosine_estimate{})",
    )
    m.end_panel()
    m.add_panel(name="final_safe_direction_count", name_en="final_safe_direction_count", type="line")
    m.add_metric(
        metrics_name="final_safe_direction_path_count_estimate",
        expr="avg(final_safe_direction_path_count_estimate{})",
    )
    m.end_panel()
    m.add_panel(name="final_visible_treasure_ratio", name_en="final_visible_treasure_ratio", type="line")
    m.add_metric(metrics_name="final_visible_treasure_ratio", expr="avg(final_visible_treasure_ratio{})")
    m.end_panel()
    m.add_panel(name="last_flash_used", name_en="last_flash_used", type="line")
    m.add_metric(metrics_name="last_flash_used", expr="avg(last_flash_used{})")
    m.end_panel()
    m.add_panel(name="last_flash_ready", name_en="last_flash_ready", type="line")
    m.add_metric(metrics_name="last_flash_ready", expr="avg(last_flash_ready{})")
    m.end_panel()
    m.add_panel(name="last_flash_legal_ratio", name_en="last_flash_legal_ratio", type="line")
    m.add_metric(metrics_name="last_flash_legal_ratio", expr="avg(last_flash_legal_ratio{})")
    m.end_panel()
    m.add_panel(name="last_flash_escape_improved", name_en="last_flash_escape_improved", type="line")
    m.add_metric(
        metrics_name="last_flash_escape_improved_estimate",
        expr="avg(last_flash_escape_improved_estimate{})",
    )
    m.end_panel()
    m.end_group()

    # ================================================================
    # Group 9: 全局平均状态
    # ================================================================
    m.add_group(group_name="全局平均状态", group_name_en="episode_mean_state")
    m.add_panel(name="mean_nearest_monster_path_dist", name_en="mean_nearest_monster_path_dist", type="line")
    m.add_metric(
        metrics_name="mean_nearest_monster_dist_est",
        expr="avg(mean_nearest_monster_dist_est{})",
    )
    m.end_panel()
    m.add_panel(name="mean_capture_margin_path", name_en="mean_capture_margin_path", type="line")
    m.add_metric(
        metrics_name="mean_capture_margin_path_estimate",
        expr="avg(mean_capture_margin_path_estimate{})",
    )
    m.end_panel()
    m.add_panel(name="mean_encirclement_cosine", name_en="mean_encirclement_cosine", type="line")
    m.add_metric(
        metrics_name="mean_encirclement_path_cosine_estimate",
        expr="avg(mean_encirclement_path_cosine_estimate{})",
    )
    m.end_panel()
    m.add_panel(name="mean_safe_direction_count", name_en="mean_safe_direction_count", type="line")
    m.add_metric(
        metrics_name="mean_safe_direction_path_count_estimate",
        expr="avg(mean_safe_direction_path_count_estimate{})",
    )
    m.end_panel()
    m.end_group()

    return m.build()
