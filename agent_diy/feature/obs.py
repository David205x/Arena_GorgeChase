from .dataclass import *
from .utils import distance_l2

# ======================== processor-local constants ========================
MAX_MONSTER_SPEED = 5       # 怪物速度上限
MAX_DIST_BUCKET = 5         # 最大距离桶取值
MAX_BUFF_DURATION = 50      # 最大BUFF持续时间
MAX_FLASH_DIST = 10         # 最大闪现距离
MAX_L2_DIST = 182.0         # 最大L2距离
TOTAL_TREASURE_SLOTS = 10   # 宝箱槽位上限
TOTAL_BUFF_SLOTS = 3        # BUFF槽位上限


def _norm(v, v_max, v_min=0.0):
    """Normalize value to [0, 1]."""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


def _encode_monster_dist(is_in_view: bool, chebyshev_dist, bucket) -> float:
    """怪物距离分段编码。

    视野内:  chebyshev / 30          → [0, 1/3]
    视野外:  1/3 + log₂(b+2)/3 × 2/3 → (1/3, 1)   对低桶更敏感
    不存在:  1.0
    """
    if chebyshev_dist is None and bucket is None:
        return 1.0
    if is_in_view and chebyshev_dist is not None:
        return min(float(chebyshev_dist) / 30.0, 1.0 / 3.0)
    if bucket is not None:
        return 1.0 / 3.0 + float(np.log2(bucket + 2)) / 3.0 * 2.0 / 3.0
    return 1.0


def _safe_l2_dist(l2_d) -> float:
    """仅使用 L2 距离；缺失时返回 1.0。"""
    if l2_d is None:
        return 1.0
    return _norm(l2_d, MAX_L2_DIST)


# ======================== scalar obs dimension breakdown ========================
# hero:       6
# last:       7
# action:    32  (8+8+8+8)
# monster:   16  (7+7+1+1)
# resource:  62  (10 + 10*4 + 3*4)
# space:      8
# stage:      3
# -------------------
# TOTAL:    134
SCALAR_DIM = 134


def construct_obs_scaler(data: dict) -> np.ndarray:
    """将 extractor.build_obs_state() 的结构化状态编码为定长 scalar 向量。

    返回 shape=(SCALAR_DIM,) 的 float32 一维数组。
    局部 / 全局地图矩阵不在此函数处理，由调用方单独获取。
    """
    raw: RawObs = data["raw"]
    hero: Hero = data["hero"]

    # ============================== 1. hero state  (6)
    hero_vec = [
        _norm(hero.x, MAP_SIZE),
        _norm(hero.z, MAP_SIZE),
        _norm(data["hero_speed"], 2),
        _norm(hero.buff_remaining_time, MAX_BUFF_DURATION),
        float(hero.can_flash),
        _norm(hero.flash_cooldown, raw.flash_cooldown_max) if raw.flash_cooldown_max > 0 else 0.0,
    ]

    # ============================== 2. last action feedback  (7)
    al: ActionLast = data["action_last"]
    dx_last, dz_last = al.moved_delta
    last_vec = [
        float(np.clip(dx_last / 10.0, -1.0, 1.0)),
        float(np.clip(dz_last / 10.0, -1.0, 1.0)),
        float(al.nearest_monster_distance_increased),
        float(al.picked_treasure),
        float(al.picked_buff),
        float(np.clip(al.map_explore_rate_delta * 100.0, 0.0, 1.0)),
        float(np.clip(al.reward_delta.step_score_delta, -5.0, 5.0)) / 5.0,
    ]

    # ============================== 3. action predict  (32)
    ap: ActionPredict = data["action_predict"]
    action_vec = (
        [float(v) for v in ap.move_valid_mask]                      # 8
        + [float(v) for v in ap.flash_valid_mask]                    # 8
        + [_norm(d, MAX_FLASH_DIST) for d in ap.flash_distance]     # 8
        + [float(v) for v in ap.flash_across_wall]                   # 8
    )

    # ============================== 4. monster pressure  (16)
    ms: MonsterSummary = data["monster_summary"]

    def _monster_vec(idx: int) -> list[float]:
        exists = getattr(ms, f"monster{idx}_exists")
        if not exists:
            return [0.0] * 7
        steps = getattr(ms, f"monster{idx}_steps_to_appear")
        rel_pos = getattr(ms, f"monster{idx}_relative_position")
        cheb = getattr(ms, f"monster{idx}_distance_chebyshev")
        bucket = getattr(ms, f"monster{idx}_distance_bucket")
        speed = getattr(ms, f"monster{idx}_speed")
        is_nearest = getattr(ms, f"monster{idx}_is_nearest")
        raw_m = raw.monsters[idx - 1] if len(raw.monsters) >= idx else None
        in_view = raw_m.is_in_view if raw_m is not None else False
        return [
            1.0,
            _norm(steps, raw.max_step),
            float(np.clip(rel_pos[0] / MAP_SIZE, -1.0, 1.0)),
            float(np.clip(rel_pos[1] / MAP_SIZE, -1.0, 1.0)),
            _encode_monster_dist(in_view, cheb, bucket),
            _norm(speed, MAX_MONSTER_SPEED),
            float(is_nearest),
        ]

    monster_vec = (
        _monster_vec(1)                                                                      # 7
        + _monster_vec(2)                                                                    # 7
        + [float(ms.relative_direction_cosine) if ms.relative_direction_cosine is not None else 0.0]  # 1
        + [_norm(ms.average_monster_distance or 0, MAP_SIZE)]                                # 1
    )

    # ============================== 5. resource  (62)
    rs: ResourceSummary = data["resource_summary"]

    resource_summary_vec = [
        _norm(rs.treasure_discovered_count, raw.total_treasure) if raw.total_treasure > 0 else 0.0,
        rs.treasure_progress,
        float(rs.nearest_known_treasure_direction[0]),
        float(rs.nearest_known_treasure_direction[1]),
        _safe_l2_dist(rs.nearest_known_treasure_distance_l2),
        _norm(rs.buff_discovered_count, raw.total_buff) if raw.total_buff > 0 else 0.0,
        rs.buff_progress,
        float(rs.nearest_known_buff_direction[0]),
        float(rs.nearest_known_buff_direction[1]),
        _safe_l2_dist(rs.nearest_known_buff_distance_l2),
    ]  # 10

    treasure_full: dict[int, Organ | None] = data.get("treasure_full", {})
    treasure_slot_vec: list[float] = []
    for slot_id in range(1, TOTAL_TREASURE_SLOTS + 1):
        t = treasure_full.get(slot_id)
        if t is None:
            treasure_slot_vec += [0.0, 0.0, 0.0, 0.0]
        else:
            status_val = 1.0 if t.status == 1 else 0.5
            d = distance_l2(hero.x, hero.z, t.x, t.z)
            treasure_slot_vec += [
                status_val,
                float(np.clip((t.x - hero.x) / MAP_SIZE, -1.0, 1.0)),
                float(np.clip((t.z - hero.z) / MAP_SIZE, -1.0, 1.0)),
                _norm(d, MAX_L2_DIST),
            ]
    # 40

    buff_full: dict[int, Organ | None] = data.get("buff_full", {})
    sorted_buff_keys = sorted(buff_full.keys())
    cooldown_max = max(int(raw.buff_refresh_time), 1)
    buff_slot_vec: list[float] = []
    for i in range(TOTAL_BUFF_SLOTS):
        b = buff_full.get(sorted_buff_keys[i]) if i < len(sorted_buff_keys) else None
        if b is None:
            buff_slot_vec += [-1.0, 0.0, 0.0, 0.0]
        else:
            if b.status == 1 and b.cooldown == 0:
                status_val = 1.0
            elif b.cooldown > 0:
                status_val = float(np.clip(1.0 - (b.cooldown / cooldown_max), 0.0, 1.0))
            else:
                status_val = 0.0
            d = distance_l2(hero.x, hero.z, b.x, b.z)
            buff_slot_vec += [
                status_val,
                float(np.clip((b.x - hero.x) / MAP_SIZE, -1.0, 1.0)),
                float(np.clip((b.z - hero.z) / MAP_SIZE, -1.0, 1.0)),
                _norm(d, MAX_L2_DIST),
            ]
    # 12

    resource_vec = resource_summary_vec + treasure_slot_vec + buff_slot_vec  # 62

    # ============================== 6. space  (8)
    ss: SpaceSummary = data["space_summary"]
    space_vec = [_norm(c, VIEW_SIZE) for c in ss.corridor_lengths]  # 8

    # ============================== 7. stage & tempo  (3)
    step_progress = _norm(raw.step, raw.max_step)
    stage2_cd = _norm(max(0, raw.monster_interval - raw.step), raw.max_step) if raw.monster_interval >= 0 else 0.0
    stage3_cd = _norm(max(0, raw.monster_speed_boost_step - raw.step), raw.max_step) if raw.monster_speed_boost_step >= 0 else 0.0
    stage_vec = [step_progress, stage2_cd, stage3_cd]  # 3

    # ============================== concat
    scalar = np.array(
        hero_vec + last_vec + action_vec + monster_vec
        + resource_vec + space_vec + stage_vec,
        dtype=np.float32,
    )
    assert scalar.shape == (SCALAR_DIM,), f"obs dim mismatch: expected {SCALAR_DIM}, got {scalar.shape[0]}"
    return scalar


GLOBAL_DS_SIZE = 64
VISIT_LOG_CAP = float(np.log1p(100.0))

# ======================== matrix obs dimension breakdown ========================
# local:   8 channels × 21 × 21
#   0 obstacle   1 hero   2 monster   3 treasure   4 buff
#   5 visit(log-norm)   6 flash_landing   7 visit_coverage(log-norm)
# global:  4 channels × 64 × 64  (128×128 downsample 2×)
#   0 walkability(-1/0/1 → 0/0.25/1)   1 visit_coverage(log-norm)
#   2 treasure_available(max-pool)      3 buff_known(max-pool)
LOCAL_CH = 8
GLOBAL_CH = 4


def _downsample_mean(arr: np.ndarray, target: int) -> np.ndarray:
    factor = arr.shape[0] // target
    return arr.reshape(target, factor, target, factor).mean(axis=(1, 3))


def _downsample_max(arr: np.ndarray, target: int) -> np.ndarray:
    factor = arr.shape[0] // target
    return arr.reshape(target, factor, target, factor).max(axis=(1, 3))


def _log_norm_map(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.log1p(arr.astype(np.float32)) / VISIT_LOG_CAP, 0.0, 1.0)


def construct_obs_matrix(data: dict) -> dict[str, np.ndarray]:
    """构造局部和全局地图的多通道矩阵表示。

    Returns:
        dict with:
          'local':  shape (LOCAL_CH,  21, 21)  float32
          'global': shape (GLOBAL_CH, 64, 64)  float32
    """
    # ============================== local map  (8ch × 21 × 21)
    layers: LocalMapLayers = data["local_map_layers"]

    local_map = np.stack([
        layers.obstacle.astype(np.float32),
        layers.hero.astype(np.float32),
        layers.monster.astype(np.float32),
        layers.treasure.astype(np.float32),
        layers.buff.astype(np.float32),
        _log_norm_map(layers.visit),
        layers.flash_landing.astype(np.float32),
        _log_norm_map(layers.visit_coverage),
    ], axis=0)

    # ============================== global map  (4ch × 128×128 → 32×32)
    map_full: np.ndarray = data["global_map_full"]
    visit_cov: np.ndarray = data["global_visit_coverage"]
    treasure_map: np.ndarray = data["global_treasure_available_map"]
    buff_map: np.ndarray = data["global_buff_known_map"]

    walkable = np.where(
        map_full == 1, 1.0,
        np.where(map_full == 0, 0.25, 0.0),
    ).astype(np.float32)

    global_map = np.stack([
        _downsample_mean(walkable, GLOBAL_DS_SIZE),
        _downsample_mean(_log_norm_map(visit_cov), GLOBAL_DS_SIZE),
        _downsample_max(treasure_map.astype(np.float32), GLOBAL_DS_SIZE),
        _downsample_max(buff_map.astype(np.float32), GLOBAL_DS_SIZE),
    ], axis=0)

    return {"local": local_map, "global": global_map}
