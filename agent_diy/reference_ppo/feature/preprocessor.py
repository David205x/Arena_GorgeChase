import os
import json
import numpy as np
import math
from agent_ppo.feature.definition import (
    RelativeDistance,
    RelativeDirection,
)
from agent_ppo.conf.conf import Config

# 8个移动方向的坐标偏移量, 用于构造Legal Action
_biases = [
    # 0: x++, z
    [(1, 0)],
    # 1: x++, z++
    [(1, 1)],
    # 2: x, z++
    [(0, 1)],
    # 3: x--, z++
    [(-1, 1)],
    # 4: x--, z
    [(-1, 0)],
    # 5: x--, z--
    [(-1, -1)],
    # 6: x, z--
    [(0, -1)],
    # 7: x++, z--
    [(1, -1)],
]
_biases = np.array(_biases, dtype=np.int8)
_biases_x = _biases[:, :, 0]
_biases_z = _biases[:, :, 1]
_vision_r = Config.VISION_R
# 坐标的二进制数量, 用于构造坐标的二进制编码
_num_bits = Config.DIM_COORD_BINARY
_bit_mask = 1 << np.arange(_num_bits)


def norm(v, max_v, min_v=0):
    """给定数值和期望的上下界, 将其投射至[0, 1]范围内"""
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v)


def check_coord(x: int, z: int):
    """检查坐标是否在合法范围内"""
    return 0 <= x < 128 and 0 <= z < 128


def _binary_scale_embedding(to_encode: int) -> np.ndarray:
    """将数值二进制编码(这里只用于编码智能体/物件的绝对位置)"""
    pos = to_encode * np.ones(_num_bits, dtype=np.uint8)
    result = np.not_equal(np.bitwise_and(pos, _bit_mask), 0)
    return result.astype(np.float32)


def _encode_sqrt_one_hot(x: int | float, num: int):
    """
    一种用于指示数值'尺度'的编码方式
    将数值开方, 然后向下取整, 用onehot来编码所得到的数字
    """
    y = math.floor(math.sqrt(x))
    y = min(y, num)
    onehot = np.zeros(num + 1)
    onehot[y] = 1
    return onehot


class Preprocessor:
    def __init__(self):
        # 读取(全局)障碍地图, 方便判断闪现是否可行
        dir_name = os.path.join(os.path.dirname(__file__), 'fish.json')
        self.global_map = (
            np.array(json.load(open(dir_name)), dtype=np.bool_)
            .reshape(128, 128)
            .transpose()
        )
        # 记忆地图, 每踩过一个坐标就给对应位置加0.2, 最大加到1.0
        self.memory_map = np.zeros(
            [128 + 2 * _vision_r, 128 + 2 * _vision_r], dtype=np.float32
        )
        # 高级赛道当中只有视野内有Organ时才会给出绝对位置信息, 这里我们尝试记录各个Organ的状态
        self.memory_organ_status = {}
        self.memory_organ_pos = {}
        self.reset()

    def reset(self):
        """将各种记忆信息初始化"""
        self.memory_map = np.pad(
            np.zeros([128, 128], dtype=np.float32),
            ((_vision_r, _vision_r), (_vision_r, _vision_r)),
            'constant',
            constant_values=1.0,
        )
        # -1已拾取 0未发现 1待拾取
        self.memory_organ_status = {}
        self.memory_organ_pos = {}

    def _update_memory(self, pb_state):
        """更新记忆地图, 每踩过一个坐标就给对应位置加0.2, 最大加到1.0"""
        hero = pb_state['frame_state']['heroes'][0]
        grid_pos_x, grid_pos_z = (
            _vision_r + hero['pos']['x'],
            _vision_r + hero['pos']['z'],
        )
        self.memory_map[grid_pos_x, grid_pos_z] = min(
            1, 0.2 + self.memory_map[grid_pos_x, grid_pos_z]
        )

    def obs_process(self, pb_state):
        """智能体决策时调用的观测特征编码流程"""
        # 特征
        feature = self.observation_process(pb_state)
        # 合法动作
        legal_action = self.get_legal_action(pb_state)
        self._update_memory(pb_state)
        return feature, legal_action

    def organ_feature(
        self,
        is_collectable: bool | int,
        hero_pos: tuple[int, int],
        organ: dict,
    ):
        """构造单个Organ的特征"""
        organ_id = organ['config_id']
        vector_feature = np.zeros(7)
        subtype_encode = np.zeros(3)
        dist_encode = np.zeros(Config.DIM_RELATIVE_DIST_ENCODE + 1)
        x_encode = np.zeros(_num_bits)
        z_encode = np.zeros(_num_bits)
        dir_encode = np.zeros(8, dtype=np.float32)
        dis_encode = np.zeros(5, dtype=np.float32)
        # Organ特征1: 类别 (宝箱/Buff/终点)
        # 这里我们不使用起点, 它没什么意义
        # 1: treasure, 2: buff, 3: begin, 4: end
        match organ['sub_type']:
            case 1:
                subtype_encode[0] = 1.0
            case 2:
                subtype_encode[1] = 1.0
            case 4:
                subtype_encode[2] = 1.0
        # Organ特征2.1: 绝对位置是否已知
        vector_feature[0] = self.memory_organ_status[organ_id] == 1
        # Organ特征2.2: 是否可以拾取 (针对Buff, 它在刷新之前是捡不起来的)
        vector_feature[1] = is_collectable
        if self.memory_organ_status[organ_id] == 1:
            # 如果绝对位置可知, 则进一步构造绝对位置特征
            target_pos = self.memory_organ_pos[organ_id]
            relative_pos = tuple(y - x for x, y in zip(hero_pos, target_pos))
            dist = np.linalg.norm(relative_pos)

            # Organ特征2.3: 归一化的L2距离
            vector_feature[2] = dist / (1.41 * 128)
            # Organ特征2.4: 归一化的相对坐标(x)
            vector_feature[3] = relative_pos[0] / 128.0
            # Organ特征2.5: 归一化的相对坐标(z)
            vector_feature[4] = relative_pos[1] / 128.0
            # Organ特征2.6: 归一化的绝对坐标(x)
            vector_feature[5] = target_pos[0] / max(dist, 1e-4)
            # Organ特征2.7: 归一化的绝对坐标(z)
            vector_feature[6] = target_pos[1] / max(dist, 1e-4)

            # Organ特征3: L2距离的尺度编码
            dist_encode[:] = _encode_sqrt_one_hot(dist, Config.DIM_RELATIVE_DIST_ENCODE)
            # Organ特征4: 绝对坐标的二进制编码(x)
            x_encode[:] = _binary_scale_embedding(target_pos[0] + 1)
            # Organ特征5: 绝对坐标的二进制编码(z)
            z_encode[:] = _binary_scale_embedding(target_pos[1] + 1)

        # 下面这两个特征是Organ信息中自带的, 即使其绝对位置未知也可以进行模糊估计
        # Organ特征6: 相对方向的OneHot编码
        organ_dir = RelativeDirection[organ['relative_pos']['direction']]
        # Organ特征7: 相对距离的OneHot编码
        organ_dis = RelativeDistance[organ['relative_pos']['l2_distance']]
        if organ_dir > 0:
            organ_dir = organ_dir - 1  # 舍掉第0个
            dir_encode[organ_dir] = 1.0
        if organ_dis > 0:
            organ_dis = organ_dis - 1  # 舍掉第0个
            dis_encode[organ_dis] = 1.0
        # 拼接Organ特征
        feature = np.concatenate(
            [
                subtype_encode,
                vector_feature,
                dist_encode,
                x_encode,
                z_encode,
                dis_encode,
                dir_encode,
            ],
            dtype=np.float32,
        )
        return feature

    def observation_process(self, pb_state: dict):
        """构造完整的Observation特征"""
        # 数据解包
        frame_state = pb_state['frame_state']
        score_info = pb_state['score_info']
        map_info = pb_state['map_info']
        step_no = frame_state['step_no']
        hero = frame_state['heroes'][0]
        hero_pos = hero['pos']['x'], hero['pos']['z']
        organs = frame_state['organs']
        # 扫描Organ
        for organ in organs:
            organ_id = organ['config_id']
            # 若未见过Organ的id, 在字典当中将其初始化为"未观测到"状态
            if organ_id not in self.memory_organ_status:
                self.memory_organ_status[organ_id] = 0
            # 若此前Organ状态为"未观测到", 尝试判断该时刻是否有观测/拾取到它
            if self.memory_organ_status[organ_id] == 0:
                if organ['status'] == 1:
                    self.memory_organ_status[organ_id] = 1
                    self.memory_organ_pos[organ_id] = (
                        organ['pos']['x'],
                        organ['pos']['z'],
                    )
                elif organ['status'] == 0:
                    self.memory_organ_status[organ_id] = 1
                    self.memory_organ_pos[organ_id] = (
                        organ['pos']['x'],
                        organ['pos']['z'],
                    )
                    if organ['sub_type'] == 1:
                        # 对于已拾取的宝箱, 特别将其记录为"已拾取", 此后不再参与特征构造
                        self.memory_organ_status[organ_id] = -1
            # 对于已拾取的宝箱, 特别将其记录为"已拾取", 此后不再参与特征构造
            elif self.memory_organ_status[organ_id] == 1:
                if organ['sub_type'] == 1 and organ['status'] == 0:
                    self.memory_organ_status[organ_id] = -1
        # 筛选出当前时刻的所有(未拾取过)的宝箱
        treasures = [
            organ
            for organ in organs
            if organ['sub_type'] == 1
            and self.memory_organ_status[organ['config_id']] != -1
        ]
        end_organ = organs[-1]
        buff_organ = organs[-3]
        # Buff的冷却时间是全局可见的, 不需要额外记录
        buff_cooldown = buff_organ['cooldown']
        # 特征1：Buff/闪现存续
        buff_status = hero['speed_up'] == 1
        buff_sustain = hero['buff_remain_time'] / Config.BUFF_SUSTAIN_TIME
        talent_status = hero['talent']['status'] == 1
        talent_remain = hero['talent']['cooldown'] / Config.SKILL_COOLDOWN_TIME
        self_related = np.array(
            [buff_status, buff_sustain, talent_status, talent_remain],
            dtype=np.float32,
        )
        # 特征2：Buff/闪现存续时间的尺度编码
        buff_sustain_onehot = _encode_sqrt_one_hot(
            hero['buff_remain_time'], Config.DIM_BUFF_SUSTAIN_ENCODE
        )
        buff_cooldown_onehot = _encode_sqrt_one_hot(
            buff_cooldown, Config.DIM_BUFF_COOLDOWN_ENCODE
        )
        talent_cooldown_onehot = _encode_sqrt_one_hot(
            hero['talent']['cooldown'], Config.DIM_SKILL_COOLDOWN_ENCODE
        )
        # 特征3: 智能体坐标的二进制编码
        pos_encode_row = _binary_scale_embedding(hero_pos[0] + 1)
        pos_encode_col = _binary_scale_embedding(hero_pos[1] + 1)

        # 特征4: 各个Organ的特征
        organ_features = np.zeros(
            [Config.NUM_TREASURE_MAX + 2, Config.DIM_FEATURE_ORGAN], dtype=np.float32
        )
        organ_mask = np.zeros(2 + Config.NUM_TREASURE_MAX)

        def get_treasure_dist(x):
            match self.memory_organ_status[x['config_id']]:
                case 0:
                    return (
                        dist_label := RelativeDistance[x['relative_pos']['l2_distance']]
                    ) * 20 + (80 if dist_label == 5 else 0)
                case 1:
                    return np.linalg.norm(
                        (
                            self.memory_organ_pos[x['config_id']][0] - hero_pos[0],
                            self.memory_organ_pos[x['config_id']][1] - hero_pos[1],
                        )
                    )
                case -1:
                    return 999.0

        # 首先将宝箱进行从近到远的排序
        treasures = sorted(treasures, key=get_treasure_dist)
        # 前2位固定为终点和Buff
        organ_features[0] = self.organ_feature(1, hero_pos, end_organ)
        organ_features[1] = self.organ_feature(buff_cooldown == 0, hero_pos, buff_organ)
        organ_mask[0] = 1.0
        organ_mask[1] = 1.0
        # 按顺序遍历宝箱, 逐个加入特征
        for idx, treasure in enumerate(treasures):
            if idx >= Config.NUM_TREASURE_MAX:
                break
            organ_features[2 + idx] = self.organ_feature(1, hero_pos, treasure)
            organ_mask[2 + idx] = 1.0
        # 特征5: 剩余宝箱数量的OneHot编码
        num_treasure_left = min(Config.NUM_TREASURE_MAX, len(treasures))
        num_treasure_encode = np.zeros(Config.NUM_TREASURE_MAX + 1)
        num_treasure_encode[num_treasure_left] = 1.0

        # 特征6: 视觉特征
        np_map = np.zeros([5, 1 + 2 * _vision_r, 1 + 2 * _vision_r])
        # 特征6.通道1 - 障碍
        np_map[0] = (
            np.logical_not(
                np.array([col['values'] for col in map_info], dtype=np.bool_)
            )
        ).astype(np.float32)
        # 特征6.通道2 - 记忆
        # 记忆地图
        padded_x, padded_z = hero_pos[0] + _vision_r, hero_pos[1] + _vision_r
        left = padded_x - _vision_r
        right = padded_x + _vision_r + 1  # 取值时左闭右开, 所以+1
        top = padded_z - _vision_r
        bot = padded_z + _vision_r + 1
        np_map[1] = self.memory_map[left:right, top:bot]
        for organ in organs:
            # 对于可以看见的Organ, 绘制在地图上
            if organ['status'] == 1:
                organ_pos = organ['pos']['x'], organ['pos']['z']
                relative_pos = tuple(y - x for x, y in zip(hero_pos, organ_pos))
                map_pos = tuple(x + _vision_r for x in relative_pos)
                if (
                    0 <= map_pos[0] <= 2 * _vision_r
                    and 0 <= map_pos[1] <= 2 * _vision_r
                ):
                    # 如果在局部观测范围内，进行绘制
                    match organ['sub_type']:
                        case 1:
                            # 特征6.通道3 - 宝箱
                            np_map[2, map_pos[0], map_pos[1]] = 1.0
                        case 2:
                            # 特征6.通道4 - Buff
                            np_map[3, map_pos[0], map_pos[1]] = 1.0
                        case 4:
                            # 特征6.通道5 - 终点
                            np_map[4, map_pos[0], map_pos[1]] = 1.0
        # 打包全部特征
        feature = np.concatenate(
            [
                pos_encode_row,
                pos_encode_col,
                buff_sustain_onehot,
                buff_cooldown_onehot,
                talent_cooldown_onehot,
                self_related,
                num_treasure_encode,
                organ_features.flatten(),
                organ_mask,
                np_map.flatten(),
            ],
            dtype=np.float32,
        )
        return feature

    def reward_process(self, extra_info, _extra_info):
        """
        通过全局信息构造奖励
        extra_info为当前时刻的全局信息
        _extra_info为下一时刻的全局信息
        为方便阅读, 这里统一将属于"下一时刻"的变量名前面加上下划线
        奖励的权重可以在conf/conf.py里面改
        """
        # 当前时刻的基础信息
        game_info = extra_info['game_info']
        frame_state = extra_info['frame_state']
        hero = frame_state['heroes'][0]
        hero_pos = hero['pos']['x'], hero['pos']['z']
        organs = frame_state['organs']
        end_organ = organs[-1]
        treasures = [organ for organ in organs if organ['sub_type'] == 1]
        # 下一时刻的基础信息
        _game_info = _extra_info['game_info']
        _frame_state = _extra_info['frame_state']
        _hero = _frame_state['heroes'][0]
        _hero_pos = _hero['pos']['x'], _hero['pos']['z']
        _organs = _frame_state['organs']
        _end_organ = _organs[-1]
        _treasures = [_organ for _organ in _organs if _organ['sub_type'] == 1]
        # 奖励构造
        # 1. 步数奖励
        reward_step = 1.0
        # 2. 闪现奖励
        reward_flash = 0.0
        if hero['talent']['status'] == 1 and _hero['talent']['status'] == 0:
            reward_flash = 1.0
        # 3. Buff奖励
        reward_buff = 0.0
        if hero['speed_up'] == 0 and _hero['speed_up'] == 1:
            reward_buff = 1.0
        # 4. 宝箱奖励
        reward_treasure = 0.0
        # 5. 宝箱L1距离奖励
        reward_treasure_dist = 0.0
        # 6. 终点L1距离奖励
        reward_end_dist = 0.0
        if (
            _game_info['treasure_collected_count']
            > game_info['treasure_collected_count']
        ):
            reward_treasure = 1.0
        elif game_info['treasure_collected_count'] < game_info['treasure_count'] - 2:
            treasure_remain = (
                game_info['treasure_count'] - 2 - game_info['treasure_collected_count']
            )
            treasure_dists = [
                (
                    (
                        abs(treasure['pos']['x'] - hero_pos[0])
                        + abs(treasure['pos']['z'] - hero_pos[1])
                    )
                    if treasure['status'] == 1
                    else 999.0
                )
                for treasure in treasures
            ]
            _treasure_dists = [
                (
                    (
                        abs(treasure['pos']['x'] - _hero_pos[0])
                        + abs(treasure['pos']['z'] - _hero_pos[1])
                    )
                    if treasure['status'] == 1
                    else 999.0
                )
                for treasure in treasures
            ]
            nearest_idx = np.argmin(treasure_dists)
            dist = treasure_dists[nearest_idx]
            _dist = _treasure_dists[nearest_idx]
            reward_treasure_dist = dist - _dist
        else:
            # 终点奖励
            end_dist = abs(end_organ['pos']['x'] - hero_pos[0]) + abs(
                end_organ['pos']['z'] - hero_pos[1]
            )
            _end_dist = abs(end_organ['pos']['x'] - _hero_pos[0]) + abs(
                end_organ['pos']['z'] - _hero_pos[1]
            )
            reward_end_dist = end_dist - _end_dist
        # 7. 撞墙惩罚
        reward_stiff = 0.0
        if hero_pos[0] == _hero_pos[0] and hero_pos[1] == _hero_pos[1]:
            reward_stiff = 1.0
        # 8. 重复探索惩罚
        reward_memory = self.memory_map[
            _hero_pos[0] + _vision_r, _hero_pos[1] + _vision_r
        ]
        reward = (
            reward_step * Config.REWARD_WEIGHTS['reward_step']
            + reward_flash * Config.REWARD_WEIGHTS['reward_flash']
            + reward_buff * Config.REWARD_WEIGHTS['reward_buff']
            + reward_treasure * Config.REWARD_WEIGHTS['reward_treasure']
            + reward_treasure_dist * Config.REWARD_WEIGHTS['reward_treasure_dist']
            + reward_end_dist * Config.REWARD_WEIGHTS['reward_end_dist']
            + reward_memory * Config.REWARD_WEIGHTS['reward_memory']
            + reward_stiff * Config.REWARD_WEIGHTS['reward_stiff']
        )
        return reward

    def get_legal_action(self, pb_state):
        """通过障碍地图判断移动/闪现是否可用"""
        hero = pb_state['frame_state']['heroes'][0]
        pos_x, pos_z = hero['pos']['x'], hero['pos']['z']
        legal_action = np.zeros([16], dtype=np.bool_)

        map_info = pb_state['map_info']
        map_self = (
            np.array([col['values'] for col in map_info], dtype=np.bool_)
        ).astype(np.float32)
        biased_x = _biases_x + Config.VISION_R
        biased_z = _biases_z + Config.VISION_R
        values = map_self[biased_x, biased_z]
        legal_action[:8] = np.all(values, axis=-1)
        if hero['talent']['status'] == 1:
            target_x, target_z = pos_x + 16, pos_z
            legal_action[8] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x + 11, pos_z + 11
            legal_action[9] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x, pos_z + 16
            legal_action[10] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x - 11, pos_z + 11
            legal_action[11] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x - 16, pos_z
            legal_action[12] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x - 11, pos_z - 11
            legal_action[13] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x, pos_z - 16
            legal_action[14] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
            target_x, target_z = pos_x + 11, pos_z - 11
            legal_action[15] = (
                check_coord(target_x, target_z) and self.global_map[target_x, target_z]
            )
        return legal_action
