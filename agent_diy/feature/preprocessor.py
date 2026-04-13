#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计。
"""

import numpy as np

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128.0
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5.0
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50.0


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def predict_flash_pos(map_view: np.ndarray, x: int, z: int) -> list[tuple[int, int]]:
    """
    预测8方向的闪现落点
    :param map_view: 21*21的矩阵, 其中1为可行动, 0为不可行动
    :param x
    :param z
    :return 按[右、右上、上、...] 返回8方向的落点坐标
    """
    pass


def is_pos_neighbor(x1: int, z1: int, x2: int, z2: int) -> bool:
    dx = abs(x1 - x2)
    dz = abs(z1 - z2)
    if dx + dz <= 2:
        return True
    return False


class Character:
    def __init__(self, obs: dict):
        self.id: int = obs.get('hero_id', 0) or obs.get('monster_id', 0) or obs.get('config_id', 0)
        self.x: int = obs['pos']['x']
        """[0, 127]"""
        self.z: int = obs['pos']['z']
        """[0, 127]"""
        self.hero_l2_distance: int = obs.get('hero_l2_distance', 0)
        """
        与英雄的欧氏距离桶编号 (0-5), 均匀划分.
        128×128 地图均匀划分：
        0=[0,30), 1=[30,60), 
        2=[60,90), 3=[90,120), 
        4=[120,150), 5=[150,180]
        """
        self.hero_relative_direction: int = obs.get('hero_relative_direction', 0)
        """
        相对于英雄的方位 (0-8)
        0=重叠/无效，
        1=东，2=东北，
        3=北，4=西北，
        5=西，6=西南，
        7=南，8=东南
        """


class Hero(Character):
    def __init__(self, obs: dict):
        super(Hero, self).__init__(obs)
        self.buff_remaining_time: int = obs['buff_remaining_time']
        self.flash_cooldown: int = obs['flash_cooldown']

    @property
    def can_flash(self) -> bool:
        return self.flash_cooldown == 0


class Monster(Character):
    def __init__(self, obs: dict):
        super(Monster, self).__init__(obs)
        self.monster_interval: int = obs['monster_interval']
        self.speed: int = obs['speed']
        self.is_in_view: bool = bool(obs['is_in_view'])


class Organ(Character):
    def __init__(self, obs: dict):
        super(Organ, self).__init__(obs)
        self.status: bool = bool(obs['status'])
        """1=可获取, 0=不可获取, 实际上被收集后organ会直接消失, 该属性意味不明"""
        self.sub_type: int = bool(obs['status'])
        """1=宝箱, 2=加速 buff"""
        # ========== custom
        self.cooldown: int = 0


class RawObs:
    def __init__(self, env_obs: dict):
        # ==========
        frame_state: dict = env_obs['frame_state']
        env_info: dict = env_obs['env_info']
        # ========== current env
        self.step: int = env_obs['step_no']
        self.legal_action: list[bool] = env_obs['legal_action']
        self.map_view: np.ndarray = np.array(env_obs['map_info'])
        self.hero: Hero = Hero(frame_state['heroes'])
        self.monsters: list[Monster] = [Monster(d) for d in frame_state['monsters']]
        self.treasures: list[Organ] = [Organ(d) for d in frame_state['organs'] if d['sub_type'] == 1]
        self.buffs: list[Organ] = [Organ(d) for d in frame_state['organs'] if d['sub_type'] == 2]
        self.treasure_id: list[int] = env_info['treasure_id']
        """该变量记录的是本局内**没有被收集**的宝箱序号, 并非所有"""
        # ========== statistic
        self.collected_buff: int = env_info['collected_buff']
        self.flash_count: int = env_info['flash_count']
        self.step_score: float = env_info['step_score']
        self.total_score: float = env_info['total_score']
        self.treasure_score: int = env_info['treasure_score']
        self.treasures_collected: int = env_info['treasures_collected']
        # ========== env setting
        self.buff_refresh_time: int = env_info['buff_refresh_time']
        self.flash_cooldown_max: int = env_info['flash_cooldown_max']
        self.max_step: int = env_info['max_step']
        self.monster_init_speed: int = env_info['monster_init_speed']
        self.monster_interval: int = env_info['monster_interval']
        self.monster_speed_boost_step: int = env_info['monster_speed_boost_step']
        self.total_buff: int = env_info['total_buff']
        self.total_treasure: int = env_info['total_treasure']
        self.total_buff: int = env_info['total_buff']


class FullObs(RawObs):
    def __init__(self, env_obs: dict):
        super().__init__(env_obs)
        # ========== hero
        self.hero_last: Hero | None = None
        self.action_preferred = [1] * len(self.legal_action)
        """过滤撞墙和不可用动作"""
        # ========== map
        self.map_full: np.ndarray = np.full((int(MAP_SIZE), int(MAP_SIZE)), -1)
        self.map_explore_rate: float = 0.
        self.map_new_discover: int = 0
        # ========== organ
        self.treasure_full: list[Organ | None] = [None] * self.total_treasure
        self.buff_full: dict[Organ | None] = {}
        for i in range(self.total_treasure, self.total_treasure + self.total_buff):
            self.buff_full[i] = None

        self.update(self)

    def update(self, obs: RawObs):
        # !!! self.hero is old before update !!!
        self.update_info(obs)
        # attributes in self is new now

        # update preferred action
        direction_index = [5, 2, 1, 0, 3, 6, 7, 8]
        map_slice = self.map_view[10:13, 10:13].reshape(-1)
        self.action_preferred = self.legal_action.copy()
        for i, idx in enumerate(direction_index):
            self.action_preferred[i] = map_slice[idx]

        # update full map
        unknown_count_old = np.sum(self.map_full == -1)
        self.update_map(self.hero.x, self.hero.z, obs.map_view)
        unknown_count = np.sum(self.map_full == -1)
        self.map_explore_rate = 1 - (unknown_count / MAP_SIZE ** 2)
        self.map_new_discover = unknown_count_old - unknown_count

        # update organ
        self.update_organ(obs)




    def update_info(self, obs: RawObs):
        # TODO 部分增量没有记录
        self.step = obs.step
        self.legal_action = obs.legal_action
        self.map_view = obs.map_view
        self.hero_last = self.hero
        self.hero = obs.hero
        self.monsters = obs.monsters
        self.treasures = obs.treasures
        self.buffs = obs.buffs
        # ========== statistic
        self.collected_buff = obs.collected_buff
        self.flash_count = obs.flash_count
        self.step_score = obs.step_score
        self.total_score = obs.step_score
        self.treasure_score = obs.treasure_score
        self.treasures_collected = obs.treasures_collected
        # ========== don't need to update env setting

    def update_map(self, x:int, z:int, map_view: np.ndarray):
        view_size = 21
        half_size = view_size // 2

        x_min = x - half_size
        x_max = x + half_size + 1
        z_min = z - half_size
        z_max = z + half_size + 1

        global_x_start = max(0, x_min)
        global_x_end = min(128, x_max)
        global_z_start = max(0, z_min)
        global_z_end = min(128, z_max)

        view_x_start = max(0, -x_min)
        view_x_end = view_size - max(0, x_max - 128)
        view_z_start = max(0, -z_min)
        view_z_end = view_size - max(0, z_max - 128)

        self.map_full[global_x_start:global_x_end, global_z_start:global_z_end] = \
            map_view[view_x_start:view_x_end, view_z_start:view_z_end]

    def update_organ(self, obs: RawObs):
        # add new seen treasures
        for o in obs.treasures:
            if self.treasure_full[o.id] is None:
                self.treasure_full[o.id] = o
        # mark collected treasures
        for o in self.treasure_full:
            if o and o.id not in obs.treasure_id:
                o.status = 0
        # add new seen buffs
        for o in obs.buffs:
            if self.buff_full[o.id] is None:
                self.buff_full[o.id] = o
        # refresh buffs cooldown
        for o in self.buff_full:
            condition = [
                o,
                is_pos_neighbor(self.hero.x, self.hero.z, o.x, o.z),
                self.hero.buff_remaining_time == 49
            ]
            if all(condition):
                o.cooldown = self.buff_refresh_time
            # count down
            o.cooldown = max(o.cooldown - 1, 0)




class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 200
        self.last_min_monster_dist_norm = 0.5

    def feature_process(self, env_obs, last_action):
        """Process env_obs into feature vector, legal_action mask, and reward.

        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        map_info = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no = observation["step_no"]
        self.max_step = env_info.get("max_step", 200)

        # Hero self features (4D) / 英雄自身特征
        hero = frame_state["heroes"]
        hero_pos = hero["pos"]
        hero_x_norm = _norm(hero_pos["x"], MAP_SIZE)
        hero_z_norm = _norm(hero_pos["z"], MAP_SIZE)
        flash_cd_norm = _norm(hero["flash_cooldown"], MAX_FLASH_CD)
        buff_remain_norm = _norm(hero["buff_remaining_time"], MAX_BUFF_DURATION)

        hero_feat = np.array([hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm], dtype=np.float32)

        # Monster features (5D x 2) / 怪物特征
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m["pos"]
                if is_in_view:
                    m_x_norm = _norm(m_pos["x"], MAP_SIZE)
                    m_z_norm = _norm(m_pos["z"], MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1), MAX_MONSTER_SPEED)

                    # Euclidean distance / 欧式距离
                    raw_dist = np.sqrt((hero_pos["x"] - m_pos["x"]) ** 2 + (hero_pos["z"] - m_pos["z"]) ** 2)
                    dist_norm = _norm(raw_dist, MAP_SIZE * 1.41)
                else:
                    m_x_norm = 0.0
                    m_z_norm = 0.0
                    m_speed_norm = 0.0
                    dist_norm = 1.0
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm], dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # Local map features (16D) / 局部地图特征
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center = len(map_info) // 2
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < len(map_info[0]):
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # Legal action mask (8D) / 合法动作掩码
        legal_action = [1] * 8
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 8}
                legal_action = [1 if j in valid_set else 0 for j in range(8)]

        if sum(legal_action) == 0:
            legal_action = [1] * 8

        # Progress features (2D) / 进度特征
        step_norm = _norm(self.step_no, self.max_step)
        survival_ratio = step_norm
        progress_feat = np.array([step_norm, survival_ratio], dtype=np.float32)

        # Concatenate features / 拼接特征
        feature = np.concatenate(
            [
                hero_feat,
                monster_feats[0],
                monster_feats[1],
                map_feat,
                np.array(legal_action, dtype=np.float32),
                progress_feat,
            ]
        )

        # Step reward / 即时奖励
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:
                cur_min_dist_norm = min(cur_min_dist_norm, m_feat[4])

        survive_reward = 0.01
        dist_shaping = 0.1 * (cur_min_dist_norm - self.last_min_monster_dist_norm)

        self.last_min_monster_dist_norm = cur_min_dist_norm

        reward = [survive_reward + dist_shaping]

        return feature, legal_action, reward
