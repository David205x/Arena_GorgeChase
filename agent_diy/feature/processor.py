import numpy as np
from .dataclass import *

# Map size / 地图尺寸（128×128）
MAP_SIZE = 128
# Max monster speed / 最大怪物速度
MAX_MONSTER_SPEED = 5
# Max distance bucket / 距离桶最大值
MAX_DIST_BUCKET = 5
# Max flash cooldown / 最大闪现冷却步数
MAX_FLASH_CD = 2000
# Max buff duration / buff最大持续时间
MAX_BUFF_DURATION = 50
# Vision Radius
VISION_RADIUS = 10


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1].

    将值归一化到 [0, 1]。
    """
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def construct_obs(data: dict) -> np.ndarray:
    # ============================== hero
    raw: RawObs = data['raw']
    hero: Hero = data['hero']
    hero_x = _norm(hero.x, 0, MAP_SIZE)
    hero_z = _norm(hero.z, 0, MAP_SIZE)
    hero_pos = np.array([hero_x, hero_z])
    # | 英雄当前速度 | `current.hero_speed` |
    hero_speed = _norm(data['hero_speed'], 0, 2)
    hero_buff_remaining = _norm(hero.buff_remaining_time, 0, MAX_BUFF_DURATION)
    hero_can_flash = int(hero.can_flash)
    hero_flash_cooldown = _norm(hero.flash_cooldown, 0, raw.flash_cooldown_max)

    # ============================== last action
    action_last: ActionLast = data['action_last']
    x_delta, z_delta = action_last.moved_delta
    x_delta = _norm(x_delta, 0, VISION_RADIUS)
    z_delta = _norm(z_delta, 0, VISION_RADIUS)
    moved_delta = np.array([x_delta, z_delta])
    # | 上次动作后最近怪物距离是否增加 | `action_last.nearest_monster_distance_increased` |
    picked_buff = int(action_last.picked_buff)
    picked_treasure = int(action_last.picked_treasure)
    map_exploered_

    # ============================== action

    # | 上次动作带来的地图探索率增量 | 当前实现未直接提供探索率增量；已提供
    # `map_new_discover` / `action_last.explored_new_area` |
    # | 上次动作带来的环境收益 | `action_last.reward_delta` |
    # | | |
    # | 普通移动
    # 8
    # 个方向可行性mask | `action_predict.move_valid_mask` |
    # | 闪现动作8个方向的可行性mask | `action_predict.flash_valid_mask` |
    # | 闪现带来的距离 | `action_predict.flash_distance` |
    # | 闪现是否穿墙 | `action_predict.flash_across_wall` |