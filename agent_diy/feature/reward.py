"""Reward shaping for agent_diy.

Implements the staged reward formula from 设计清单/2_reward.md:

    total = α · survival + (1−α) · explore + terminal

where α = stage_info.alpha increases with danger (0.25 → 0.60 → 0.90).

Survival bucket  (×α)  — step score, monster distance, encirclement,
                         space, topology, action quality
Explore  bucket  (×1−α) — treasure score, treasure approach, map explore
Terminal         (once) — completion bonus / death penalties
"""

from __future__ import annotations

from agent_diy.feature.dataclass import (
    RawObs,
    ActionLast,
    MonsterSummary,
    ResourceSummary,
    SpaceSummary,
    StageInfo,
)

# ======================== helpers ========================

# --- 4.5  explore: treasure / buff approach ---
W_TREASURE_APPROACH      = 0.010
W_BUFF_APPROACH          = 0.008
TREASURE_APPROACH_CLIP   = 5.0
BUFF_APPROACH_CLIP       = 5.0
RESOURCE_NEAR_BUCKET_MAX = 1
RESOURCE_MID_BUCKET_MAX  = 3
RESOURCE_FAR_BUCKET_MAX  = 6
RESOURCE_NEAR_GAIN       = 1.00
RESOURCE_MID_GAIN        = 0.45
RESOURCE_FAR_GAIN        = 0.18
RESOURCE_DIST_GAIN       = 0.08


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _resource_distance_gain(distance: int | float | None) -> float:
    if distance is None:
        return 0.0
    d = float(distance)
    if d <= RESOURCE_NEAR_BUCKET_MAX:
        return RESOURCE_NEAR_GAIN
    if d <= RESOURCE_MID_BUCKET_MAX:
        return RESOURCE_MID_GAIN
    if d <= RESOURCE_FAR_BUCKET_MAX:
        return RESOURCE_FAR_GAIN
    return RESOURCE_DIST_GAIN


# ======================== tunable weights ========================

# --- 4.3  env raw score ---
W_STEP_SCORE             = 1.0 / 50.0
W_TREASURE_SCORE         = 1.0 / 50.0

# --- 4.4  survival: monster distance ---
SAFE_DIST                = 20.0
W_DIST_DELTA             = 0.015
DIST_DELTA_CLIP          = 5.0

# --- 4.4  survival: encirclement ---
ENCIRCLE_AVG_SAFE        = 50.0
W_ENCIRCLE               = 0.020

# --- 4.4  survival: traversable space ---
W_SPACE_DELTA            = 0.001
SPACE_DELTA_CLIP         = 10.0

# --- 4.4  survival: dangerous topology ---
DEAD_END_PEN             = -0.04
CORRIDOR_PEN             = -0.02
LOW_OPEN_PEN             = -0.01

# --- 4.5  explore: map exploration ---
W_EXPLORE                = 50.0

# --- 4.6  action quality (survival-aligned, ×α) ---
NO_MOVE_PEN              = -0.2
LOW_FLASH_RATIO          = 0.4
LOW_FLASH_PEN            = -0.2
FLASH_ESCAPE_BONUS       = 0.06
FLASH_ACROSS_WALL_BONUS  = 0.1
VISIT_THRESH             = 5
W_REVISIT                = 0.1
REVISIT_CAP              = 0.1

# --- 4.7  terminal ---
COMPLETE_BONUS           = 1.0
DEATH_STAGE_PEN          = {1: -0.6, 2: -0.4, 3: -0.2}
DEATH_ENCIRCLE_PEN       = -0.25
DEATH_DEAD_END_PEN       = -0.20
DEATH_SAME_SIDE_REDUCE   = 0.10


# ======================== main entry ========================

def compute_reward(data: dict) -> tuple[float, dict]:
    """Compute shaped reward from ``extractor.build_reward_state()``.

    Returns:
        (total_reward, reward_info) where reward_info contains all sub-components.
    """
    raw: RawObs | None = data["raw"]
    if raw is None:
        return 0.0, {}

    terminated: bool = data["terminated"]
    truncated: bool = data["truncated"]
    abnormal_truncated: bool = data["abnormal_truncated"]

    al: ActionLast = data["action_last"]
    ms: MonsterSummary = data["monster_summary"]
    rs: ResourceSummary = data["resource_summary"]
    ss: SpaceSummary = data["space_summary"]
    si: StageInfo = data["stage_info"]
    rd = al.reward_delta

    alpha = si.alpha

    survival, survival_info = _survival(rd, ms, ss, al, data)
    explore, explore_info = _explore(rd, rs, al)
    terminal_value, terminal_info = _terminal(terminated, truncated, abnormal_truncated, si, ms, ss)

    total = alpha * survival + (1.0 - alpha) * explore + terminal_value

    reward_info = {
        "total": round(total, 6),
        "alpha": round(alpha, 4),
        "survival": round(survival, 6),
        "survival_weighted": round(alpha * survival, 6),
        "explore": round(explore, 6),
        "explore_weighted": round((1.0 - alpha) * explore, 6),
        "terminal": round(terminal_value, 6),
        **{f"s_{k}": v for k, v in survival_info.items()},
        **{f"e_{k}": v for k, v in explore_info.items()},
        **terminal_info,
    }

    return total, reward_info


# ======================== component functions ========================

def _survival(rd, ms: MonsterSummary, ss: SpaceSummary, al: ActionLast, data: dict) -> tuple[float, dict]:
    """Survival-aligned shaping (§4.3-step, §4.4, §4.6)."""
    info: dict[str, float] = {}

    step_score = rd.step_score_delta * W_STEP_SCORE
    info["step_score"] = round(step_score, 6)

    monster_dist = 0.0
    if ms.nearest_monster_distance is not None and ms.nearest_monster_distance_delta is not None:
        danger = max(0.0, 1.0 - ms.nearest_monster_distance / SAFE_DIST)
        delta = _clip(float(ms.nearest_monster_distance_delta),
                      -DIST_DELTA_CLIP, DIST_DELTA_CLIP)
        monster_dist = delta * W_DIST_DELTA * (1.0 + danger)
    info["monster_dist"] = round(monster_dist, 6)

    encircle = 0.0
    if ms.monster_count >= 2 and ms.relative_direction_cosine is not None:
        avg_d = ms.average_monster_distance or 999.0
        avg_mod = max(0.0, 1.0 - avg_d / ENCIRCLE_AVG_SAFE)
        threat = max(0.0, -ms.relative_direction_cosine)
        encircle = -(threat * avg_mod * W_ENCIRCLE)
    info["encircle"] = round(encircle, 6)

    sd = _clip(float(ss.traversable_space_delta), -SPACE_DELTA_CLIP, SPACE_DELTA_CLIP)
    space = sd * W_SPACE_DELTA
    info["space"] = round(space, 6)

    topology = 0.0
    # if ss.is_dead_end:
    #     topology += DEAD_END_PEN
    # if ss.is_corridor:
    #     topology += CORRIDOR_PEN
    # if ss.is_low_openness:
    #     topology += LOW_OPEN_PEN
    # info["topology"] = round(topology, 6)

    no_move = 0.0
    action_id = al.last_action_id
    if action_id >= 0 and not al.moved:
        no_move = NO_MOVE_PEN
    info["no_move"] = round(no_move, 6)

    flash_low = 0.0
    flash_escape = 0.0
    flash_across_wall = 0.0
    if rd.flash_count_delta > 0:
        if action_id >= 8:
            prev_distances: list[float] = data["prev_flash_distance"]
            max_d = max(prev_distances) if prev_distances else 0.0
            flash_dir = action_id - 8
            used_d = prev_distances[flash_dir] if flash_dir < len(prev_distances) else 0.0
            prev_across_wall: list[bool] = data["prev_flash_across_wall"]
            used_across_wall = prev_across_wall[flash_dir] if flash_dir < len(prev_across_wall) else False
            if max_d > 0 and used_d < max_d * LOW_FLASH_RATIO:
                flash_low = LOW_FLASH_PEN
            if used_across_wall:
                flash_across_wall = FLASH_ACROSS_WALL_BONUS
        if data["flash_escape_improved_estimate"]:
            flash_escape = FLASH_ESCAPE_BONUS
    info["flash_low"] = round(flash_low, 6)
    info["flash_escape"] = round(flash_escape, 6)
    info["flash_across_wall"] = round(flash_across_wall, 6)

    revisit = 0.0
    vc: int = data["hero_visit_count"]
    if vc > VISIT_THRESH:
        revisit = -min((vc - VISIT_THRESH) * W_REVISIT, REVISIT_CAP)
    info["revisit"] = round(revisit, 6)

    r = step_score + monster_dist + encircle + space + topology + no_move + flash_low + flash_escape + flash_across_wall + revisit
    return r, info


def _explore(rd, rs: ResourceSummary, al: ActionLast) -> tuple[float, dict]:
    """Exploration / resource-aligned shaping (§4.3-treasure, §4.5)."""
    info: dict[str, float] = {}

    treasure_score = rd.treasure_score_delta * W_TREASURE_SCORE
    info["treasure_score"] = round(treasure_score, 6)

    treasure_approach = 0.0
    td = rs.nearest_known_treasure_distance_path_delta
    if td is not None:
        approach = -_clip(float(td), -TREASURE_APPROACH_CLIP, TREASURE_APPROACH_CLIP)
        treasure_approach = approach * W_TREASURE_APPROACH * _resource_distance_gain(rs.nearest_known_treasure_distance_path)
    info["treasure_approach"] = round(treasure_approach, 6)

    buff_approach = 0.0
    bd = rs.nearest_known_buff_distance_path_delta
    if bd is not None:
        approach = -_clip(float(bd), -BUFF_APPROACH_CLIP, BUFF_APPROACH_CLIP)
        buff_approach = approach * W_BUFF_APPROACH * _resource_distance_gain(rs.nearest_known_buff_distance_path)
    info["buff_approach"] = round(buff_approach, 6)

    map_explore = al.map_explore_rate_delta * W_EXPLORE
    info["map_explore"] = round(map_explore, 6)

    r = treasure_score + treasure_approach + buff_approach + map_explore
    return r, info


def _terminal(terminated: bool, truncated: bool, abnormal_truncated: bool,
              si: StageInfo, ms: MonsterSummary, ss: SpaceSummary) -> tuple[float, dict]:
    """One-shot terminal reward (§4.7)."""
    info: dict = {}

    if abnormal_truncated:
        info["t_type"] = "abnormal"
        return 0.0, info

    if truncated and not terminated:
        info["t_type"] = "completed"
        info["t_complete_bonus"] = COMPLETE_BONUS
        return COMPLETE_BONUS, info

    if not terminated:
        info["t_type"] = "ongoing"
        return 0.0, info

    # ---- death analysis ----
    info["t_type"] = "death"
    stage = si.stage
    r = DEATH_STAGE_PEN.get(stage, -0.4)
    info["t_death_stage_pen"] = round(r, 6)

    is_encircled = (
        ms.monster_count >= 2
        and ms.relative_direction_cosine is not None
        and ms.relative_direction_cosine < -0.3
    )
    t_encircle = 0.0
    if is_encircled:
        t_encircle = DEATH_ENCIRCLE_PEN
    r += t_encircle
    info["t_encircle_pen"] = round(t_encircle, 6)

    t_dead_end = 0.0
    if ss.is_dead_end:
        t_dead_end = DEATH_DEAD_END_PEN
    r += t_dead_end
    info["t_dead_end_pen"] = round(t_dead_end, 6)

    t_same_side = 0.0
    if stage >= 2 and not is_encircled and not ss.is_dead_end:
        t_same_side = DEATH_SAME_SIDE_REDUCE
    r += t_same_side
    info["t_same_side_reduce"] = round(t_same_side, 6)

    return r, info
