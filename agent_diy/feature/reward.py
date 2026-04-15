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

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ======================== tunable weights ========================

# --- 4.3  env raw score ---
W_STEP_SCORE             = 1.0 / 100.0
W_TREASURE_SCORE         = 1.0 / 100.0

# --- 4.4  survival: monster distance ---
SAFE_DIST                = 40.0
W_DIST_DELTA             = 0.015
DIST_DELTA_CLIP          = 5.0

# --- 4.4  survival: encirclement ---
ENCIRCLE_AVG_SAFE        = 50.0
W_ENCIRCLE               = 0.020

# --- 4.4  survival: traversable space ---
W_SPACE_DELTA            = 0.003
SPACE_DELTA_CLIP         = 10.0

# --- 4.4  survival: dangerous topology ---
DEAD_END_PEN             = -0.04
CORRIDOR_PEN             = -0.02
LOW_OPEN_PEN             = -0.01

# --- 4.5  explore: treasure approach ---
W_TREASURE_APPROACH      = 0.010
TREASURE_APPROACH_CLIP   = 5.0

# --- 4.5  explore: map exploration ---
W_EXPLORE                = 3.0

# --- 4.6  action quality (survival-aligned, ×α) ---
NO_MOVE_PEN              = -0.06
LOW_FLASH_RATIO          = 0.5
LOW_FLASH_PEN            = -0.04
FLASH_ESCAPE_BONUS       = 0.06
VISIT_THRESH             = 5
W_REVISIT                = 0.004
REVISIT_CAP              = 0.04

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

    survival = _survival(rd, ms, ss, al, data)
    explore = _explore(rd, rs, al)
    terminal_value, terminal_info = _terminal(terminated, truncated, abnormal_truncated, si, ms, ss)

    total = alpha * survival + (1.0 - alpha) * explore + terminal_value

    reward_info = {
        "alpha": round(alpha, 4),
        "survival": round(survival, 6),
        "explore": round(explore, 6),
        "terminal": round(terminal_value, 6),
        "total": round(total, 6),
        **terminal_info,
    }

    return total, reward_info


# ======================== component functions ========================

def _survival(rd, ms: MonsterSummary, ss: SpaceSummary, al: ActionLast, data: dict) -> float:
    """Survival-aligned shaping (§4.3-step, §4.4, §4.6)."""
    r = 0.0

    # ---- §4.3  env step score  ↑🔊 ----
    r += rd.step_score_delta * W_STEP_SCORE

    # ---- §4.4  nearest monster distance change  ±↑🔊 ----
    if ms.nearest_monster_distance is not None and ms.nearest_monster_distance_delta is not None:
        danger = max(0.0, 1.0 - ms.nearest_monster_distance / SAFE_DIST)
        delta = _clip(float(ms.nearest_monster_distance_delta),
                      -DIST_DELTA_CLIP, DIST_DELTA_CLIP)
        r += delta * W_DIST_DELTA * (1.0 + danger)

    # ---- §4.4  encirclement penalty  -↑🔊 (2+ monsters) ----
    if ms.monster_count >= 2 and ms.relative_direction_cosine is not None:
        avg_d = ms.average_monster_distance or 999.0
        avg_mod = max(0.0, 1.0 - avg_d / ENCIRCLE_AVG_SAFE)
        # cosine < 0 → monsters on opposite sides → hero sandwiched
        threat = max(0.0, -ms.relative_direction_cosine)
        r -= threat * avg_mod * W_ENCIRCLE

    # ---- §4.4  traversable space change  ±↑🔊 ----
    sd = _clip(float(ss.traversable_space_delta), -SPACE_DELTA_CLIP, SPACE_DELTA_CLIP)
    r += sd * W_SPACE_DELTA

    # ---- §4.4  dangerous topology  -↑🔉 ----
    if ss.is_dead_end:
        r += DEAD_END_PEN
    if ss.is_corridor:
        r += CORRIDOR_PEN
    if ss.is_low_openness:
        r += LOW_OPEN_PEN

    # ---- §4.6  no-move penalty  -↑🔉 ----
    action_id = al.last_action_id
    if action_id >= 0 and not al.moved:
        r += NO_MOVE_PEN

    # ---- §4.6  flash quality  ----
    if rd.flash_count_delta > 0:
        # low-value flash penalty: 闪现位移 < 0.5 × 最长可达位移  -↑🔉
        if action_id >= 8:
            prev_distances: list[float] = data["prev_flash_distance"]
            max_d = max(prev_distances) if prev_distances else 0.0
            flash_dir = action_id - 8
            used_d = prev_distances[flash_dir] if flash_dir < len(prev_distances) else 0.0
            if max_d > 0 and used_d < max_d * LOW_FLASH_RATIO:
                r += LOW_FLASH_PEN

        # flash escape bonus: 脱险或进入更优位置  +↑🔉
        if data["flash_escape_improved_estimate"]:
            r += FLASH_ESCAPE_BONUS

    # ---- §4.6  revisit penalty (visit count)  -↑🔉 ----
    vc: int = data["hero_visit_count"]
    if vc > VISIT_THRESH:
        r -= min((vc - VISIT_THRESH) * W_REVISIT, REVISIT_CAP)

    return r


def _explore(rd, rs: ResourceSummary, al: ActionLast) -> float:
    """Exploration / resource-aligned shaping (§4.3-treasure, §4.5)."""
    r = 0.0

    # ---- §4.3  env treasure score  +↓🔉 ----
    r += rd.treasure_score_delta * W_TREASURE_SCORE

    # ---- §4.5  treasure approach  ±↓🔊 ----
    td = rs.nearest_known_treasure_distance_path_delta
    if td is not None:
        # delta < 0 → getting closer → positive reward
        approach = -_clip(float(td), -TREASURE_APPROACH_CLIP, TREASURE_APPROACH_CLIP)
        r += approach * W_TREASURE_APPROACH

    # ---- §4.5  exploration increment  +↓🔊 ----
    r += al.map_explore_rate_delta * W_EXPLORE

    return r


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
