from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Self

from .constant import *


def direction_to_vector(direction: int) -> tuple[int, int]:
    direction_map = {
        0: (0, 0),
        1: (1, 0),
        2: (1, -1),
        3: (0, -1),
        4: (-1, -1),
        5: (-1, 0),
        6: (-1, 1),
        7: (0, 1),
        8: (1, 1),
    }
    return direction_map.get(int(direction), (0, 0))


@dataclass(slots=True)
class Character:
    id: int
    x: int
    """[0, 127]"""
    z: int
    """[0, 127]"""
    hero_l2_distance: int = 0
    """
    与英雄的欧氏距离桶编号 (0-5), 来自环境原始字段。
    """
    hero_relative_direction: tuple[int, int] = (0, 0)
    """
    相对于英雄的方位方向向量, 由环境原始方向编号 (0-8) 转换而来。
    """

    @classmethod
    def from_env(cls, obs: dict[str, Any]) -> Self:
        pos = obs["pos"]
        entity_id = int(obs.get("hero_id", 0) or obs.get("monster_id", 0) or obs.get("config_id", 0))
        return cls(
            id=entity_id,
            x=int(pos["x"]),
            z=int(pos["z"]),
            hero_l2_distance=int(obs.get("hero_l2_distance", 0)),
            hero_relative_direction=direction_to_vector(int(obs.get("hero_relative_direction", 0))),
        )


@dataclass(slots=True)
class Hero(Character):
    buff_remaining_time: int = 0
    flash_cooldown: int = 0

    @property
    def can_flash(self) -> bool:
        return self.flash_cooldown == 0

    @classmethod
    def from_env(cls, obs: dict[str, Any]) -> Hero:
        base = Character.from_env(obs)
        return cls(
            id=base.id,
            x=base.x,
            z=base.z,
            hero_l2_distance=base.hero_l2_distance,
            hero_relative_direction=base.hero_relative_direction,
            buff_remaining_time=int(obs["buff_remaining_time"]),
            flash_cooldown=int(obs["flash_cooldown"]),
        )


@dataclass(slots=True)
class Monster(Character):
    monster_interval: int = 0
    speed: int = 0
    is_in_view: bool = False

    @classmethod
    def from_env(cls, obs: dict[str, Any]) -> Monster:
        base = Character.from_env(obs)
        return cls(
            id=base.id,
            x=base.x,
            z=base.z,
            hero_l2_distance=base.hero_l2_distance,
            hero_relative_direction=base.hero_relative_direction,
            monster_interval=int(obs.get("monster_interval", 0)),
            speed=int(obs.get("speed", 0)),
            is_in_view=bool(obs.get("is_in_view", 1)),
        )


@dataclass(slots=True)
class Organ(Character):
    status: int = 0
    """
    1=可获取, 0=不可获取。
    """
    sub_type: int = 0
    """
    1=宝箱, 2=buff。
    """
    cooldown: int = 0
    """仅对 buff 侧维护自定义 cooldown。"""

    @property
    def is_treasure(self) -> bool:
        return self.sub_type == 1

    @property
    def is_buff(self) -> bool:
        return self.sub_type == 2

    @classmethod
    def from_env(cls, obs: dict[str, Any]) -> Organ:
        base = Character.from_env(obs)
        return cls(
            id=base.id,
            x=base.x,
            z=base.z,
            hero_l2_distance=base.hero_l2_distance,
            hero_relative_direction=base.hero_relative_direction,
            status=int(obs["status"]),
            sub_type=int(obs["sub_type"]),
            cooldown=0,
        )


@dataclass(slots=True)
class RawObs:
    # ========== current env
    step: int
    """env.reset() 后的首帧 step=0"""
    legal_action: list[bool]
    map_view: np.ndarray
    hero: Hero
    monsters: list[Monster]
    treasures: list[Organ]
    buffs: list[Organ]
    treasure_id: list[int]
    """该变量记录的是本局内尚未被收集的宝箱序号。"""
    # ========== statistic
    collected_buff: int
    flash_count: int
    step_score: float
    total_score: float
    treasure_score: float
    treasures_collected: int
    # ========== env setting
    buff_refresh_time: int
    flash_cooldown_max: int
    max_step: int
    monster_init_speed: int
    monster_interval: int
    monster_speed_boost_step: int
    total_buff: int
    total_treasure: int

    @classmethod
    def from_env(cls, env_obs: dict[str, Any]) -> RawObs:
        frame_state = env_obs["frame_state"]
        env_info = env_obs["env_info"]
        organs = frame_state["organs"]

        treasures = [Organ.from_env(d) for d in organs if int(d["sub_type"]) == 1]
        buffs = [Organ.from_env(d) for d in organs if int(d["sub_type"]) == 2]

        return cls(
            step=int(env_obs["step_no"]),
            legal_action=[bool(v) for v in env_obs["legal_action"]],
            map_view=np.asarray(env_obs["map_info"], dtype=np.int8),
            hero=Hero.from_env(frame_state["heroes"]),
            monsters=[Monster.from_env(d) for d in frame_state["monsters"]],
            treasures=treasures,
            buffs=buffs,
            treasure_id=[int(i) for i in env_info["treasure_id"]],
            collected_buff=int(env_info["collected_buff"]),
            flash_count=int(env_info["flash_count"]),
            step_score=float(env_info["step_score"]),
            total_score=float(env_info["total_score"]),
            treasure_score=float(env_info["treasure_score"]),
            treasures_collected=int(env_info["treasures_collected"]),
            buff_refresh_time=int(env_info["buff_refresh_time"]),
            flash_cooldown_max=int(env_info["flash_cooldown_max"]),
            max_step=int(env_info["max_step"]),
            monster_init_speed=int(env_info["monster_init_speed"]),
            monster_interval=int(env_info["monster_interval"]),
            monster_speed_boost_step=int(env_info["monster_speed_boost_step"]),
            total_buff=int(env_info["total_buff"]),
            total_treasure=int(env_info["total_treasure"]),
        )


@dataclass(slots=True)
class ExtraInfo:
    map_id: int = -1
    result_code: int = 0
    result_message: str = ""
    hero: Hero | None = None
    monsters: list[Monster] = field(default_factory=list)
    treasures: list[Organ] = field(default_factory=list)
    buffs: list[Organ] = field(default_factory=list)

    @classmethod
    def from_env(cls, extra_info: dict[str, Any] | None) -> ExtraInfo | None:
        if not extra_info:
            return None

        payload = extra_info.get("extra_info", extra_info)
        frame_state = payload.get("frame_state")
        if not isinstance(frame_state, dict):
            return None

        heroes_raw = frame_state.get("heroes")
        if isinstance(heroes_raw, list):
            hero_raw = heroes_raw[0] if heroes_raw else None
        else:
            hero_raw = heroes_raw

        hero = Hero.from_env(hero_raw) if isinstance(hero_raw, dict) else None

        monsters_raw = frame_state.get("monsters", [])
        organs_raw = frame_state.get("organs", [])

        monsters = [Monster.from_env(d) for d in monsters_raw if isinstance(d, dict)]
        treasures = [Organ.from_env(d) for d in organs_raw if isinstance(d, dict) and int(d.get("sub_type", 0)) == 1]
        buffs = [Organ.from_env(d) for d in organs_raw if isinstance(d, dict) and int(d.get("sub_type", 0)) == 2]

        return cls(
            map_id=int(payload.get("map_id", -1)),
            result_code=int(payload.get("result_code", 0)),
            result_message=str(payload.get("result_message", "")),
            hero=hero,
            monsters=monsters,
            treasures=treasures,
            buffs=buffs,
        )


@dataclass(slots=True)
class RewardDelta:
    total_score_delta: float = 0.0
    step_score_delta: float = 0.0
    treasure_score_delta: float = 0.0
    treasures_collected_delta: int = 0
    collected_buff_delta: int = 0
    flash_count_delta: int = 0


@dataclass(slots=True)
class ActionPredict:
    move_valid_mask: list[bool] = field(default_factory=lambda: [False] * 8)
    flash_pos: list[tuple[int, int]] = field(default_factory=list)
    flash_pos_relative: list[tuple[int, int]] = field(default_factory=list)
    flash_valid_mask: list[bool] = field(default_factory=lambda: [False] * 8)
    flash_distance: list[float] = field(default_factory=lambda: [0.0] * 8)
    flash_across_wall: list[bool] = field(default_factory=lambda: [False] * 8)
    action_preferred: list[bool] = field(default_factory=list)


@dataclass(slots=True)
class ActionLast:
    last_action_id: int = -1
    """上一步执行的动作编号 (0-15), -1 表示首帧无动作"""
    moved: bool = False
    moved_delta: tuple[int, int] = (0, 0)
    nearest_monster_distance_increased: bool = False
    picked_treasure: bool = False
    picked_buff: bool = False
    explored_new_area: bool = False
    map_explore_rate_delta: float = 0.0
    reward_delta: RewardDelta = field(default_factory=RewardDelta)


@dataclass(slots=True)
class MonsterSummary:
    monster_count: int = 0
    nearest_monster: Monster | None = None
    second_monster: Monster | None = None
    nearest_monster_distance: int | None = None
    second_monster_distance: int | None = None
    nearest_monster_distance_last: int | None = None
    nearest_monster_distance_delta: int | None = None
    average_monster_distance: float | None = None
    monster1_exists: bool = False
    monster2_exists: bool = False
    monster1_steps_to_appear: int = 0
    monster2_steps_to_appear: int = 0
    monster1_relative_position: tuple[int, int] = (0, 0)
    monster2_relative_position: tuple[int, int] = (0, 0)
    monster1_relative_direction: tuple[int, int] = (0, 0)
    monster2_relative_direction: tuple[int, int] = (0, 0)
    monster1_distance_chebyshev: int | None = None
    monster2_distance_chebyshev: int | None = None
    monster1_distance_l2: float | None = None
    monster2_distance_l2: float | None = None
    monster1_distance_bucket: int | None = None
    monster2_distance_bucket: int | None = None
    monster1_speed: int = 0
    monster2_speed: int = 0
    monster1_is_nearest: bool = False
    monster2_is_nearest: bool = False
    relative_direction_cosine: float | None = None


@dataclass(slots=True)
class ResourceSummary:
    nearest_known_treasure: Organ | None = None
    nearest_known_treasure_distance_l2: float | None = None
    nearest_known_treasure_distance_path: int | None = None
    nearest_known_treasure_direction: tuple[int, int] = (0, 0)
    nearest_known_buff: Organ | None = None
    nearest_known_buff_distance_l2: float | None = None
    nearest_known_buff_distance_path: int | None = None
    nearest_known_buff_direction: tuple[int, int] = (0, 0)
    nearest_known_treasure_distance_path_last: int | None = None
    nearest_known_treasure_distance_path_delta: int | None = None
    treasure_discovered_count: int = 0
    buff_discovered_count: int = 0
    treasure_progress: float = 0.0
    buff_progress: float = 0.0


@dataclass(slots=True)
class SpaceSummary:
    corridor_lengths: list[int] = field(default_factory=lambda: [0] * 8)
    traversable_space: int = 0
    openness: int = 0
    safe_direction_count: int = 0
    traversable_space_delta: int = 0
    is_dead_end: bool = False
    is_corridor: bool = False
    is_low_openness: bool = False


@dataclass(slots=True)
class GlobalSummary:
    nearest_monster: Monster | None = None
    second_monster: Monster | None = None
    nearest_monster_distance: int | None = None
    second_monster_distance: int | None = None
    nearest_monster_distance_last: int | None = None
    nearest_monster_distance_delta: int | None = None
    average_monster_distance: float | None = None
    safe_direction_count: int = 0
    safe_direction_count_last: int = 0
    safe_direction_count_delta: int = 0
    nearest_monster_path_distance_estimate: int | None = None
    second_monster_path_distance_estimate: int | None = None
    nearest_monster_path_distance_last_estimate: int | None = None
    nearest_monster_path_distance_delta_estimate: int | None = None
    average_monster_path_distance_estimate: float | None = None
    capture_margin_path_estimate: int | None = None
    capture_margin_path_last_estimate: int | None = None
    capture_margin_path_delta_estimate: int | None = None
    nearest_monster_approach_direction_estimate: tuple[int, int] = (0, 0)
    second_monster_approach_direction_estimate: tuple[int, int] = (0, 0)
    encirclement_path_cosine_estimate: float | None = None
    encirclement_path_cosine_last_estimate: float | None = None
    encirclement_path_cosine_delta_estimate: float | None = None
    safe_direction_path_count_estimate: int = 0
    safe_direction_path_count_last_estimate: int = 0
    safe_direction_path_count_delta_estimate: int = 0
    dead_end_under_pressure_estimate: bool = False


@dataclass(slots=True)
class EpisodeStats:
    map_id: int = -1
    result_code: int = 0
    episode_steps: int = 0
    stage1_steps: int = 0
    stage2_steps: int = 0
    stage3_steps: int = 0
    pre_steps: int = 0
    post_steps: int = 0
    speedup_reached: bool = False
    terminated: bool = False
    truncated: bool = False
    completed: bool = False
    abnormal_truncated: bool = False
    post_terminated: bool = False
    final_stage: int = 0
    final_total_score: float = 0.0
    final_step_score: float = 0.0
    final_treasure_score: float = 0.0
    final_treasures: int = 0
    final_buffs: int = 0
    final_flash_count: int = 0
    final_nearest_monster_dist_est: int = 0
    final_capture_margin_path_estimate: int = 0
    final_encirclement_path_cosine_estimate: float = 0.0
    final_safe_direction_path_count_estimate: int = 0
    final_visible_treasure_ratio: float = 0.0
    last_flash_used: bool = False
    last_flash_ready: bool = False
    last_flash_legal_ratio: float = 0.0
    last_flash_escape_improved_estimate: bool = False
    path_signal_steps: int = 0
    nearest_monster_path_distance_estimate_sum: float = 0.0
    capture_margin_path_estimate_sum: float = 0.0
    encirclement_path_cosine_estimate_sum: float = 0.0
    safe_direction_path_count_estimate_sum: float = 0.0
    flash_escape_success_count: int = 0

    def as_dict(self) -> dict[str, float]:
        signal_steps = max(self.path_signal_steps, 1)
        return {
            "map_id": float(self.map_id),
            "result_code": float(self.result_code),
            "episode_steps": float(self.episode_steps),
            "stage1_steps": float(self.stage1_steps),
            "stage2_steps": float(self.stage2_steps),
            "stage3_steps": float(self.stage3_steps),
            "pre_steps": float(self.pre_steps),
            "post_steps": float(self.post_steps),
            "speedup_reached": float(self.speedup_reached),
            "terminated": float(self.terminated),
            "truncated": float(self.truncated),
            "completed": float(self.completed),
            "abnormal_truncated": float(self.abnormal_truncated),
            "post_terminated": float(self.post_terminated),
            "final_stage": float(self.final_stage),
            "episode_total_score": float(self.final_total_score),
            "episode_step_score": float(self.final_step_score),
            "episode_treasure_score": float(self.final_treasure_score),
            "episode_treasures": float(self.final_treasures),
            "episode_buffs": float(self.final_buffs),
            "episode_flash_count": float(self.final_flash_count),
            "final_nearest_monster_dist_est": float(self.final_nearest_monster_dist_est),
            "final_capture_margin_path_estimate": float(self.final_capture_margin_path_estimate),
            "final_encirclement_path_cosine_estimate": float(self.final_encirclement_path_cosine_estimate),
            "final_safe_direction_path_count_estimate": float(self.final_safe_direction_path_count_estimate),
            "final_visible_treasure_ratio": float(self.final_visible_treasure_ratio),
            "last_flash_used": float(self.last_flash_used),
            "last_flash_ready": float(self.last_flash_ready),
            "last_flash_legal_ratio": float(self.last_flash_legal_ratio),
            "last_flash_escape_improved_estimate": float(self.last_flash_escape_improved_estimate),
            "mean_nearest_monster_dist_est": self.nearest_monster_path_distance_estimate_sum / signal_steps,
            "mean_capture_margin_path_estimate": self.capture_margin_path_estimate_sum / signal_steps,
            "mean_encirclement_path_cosine_estimate": self.encirclement_path_cosine_estimate_sum / signal_steps,
            "mean_safe_direction_path_count_estimate": self.safe_direction_path_count_estimate_sum / signal_steps,
            "flash_escape_success_count": float(self.flash_escape_success_count),
        }


@dataclass(slots=True)
class StageInfo:
    stage: int = 0
    """1=单怪阶段, 2=双怪阶段, 3=怪物加速阶段（暂定）"""
    has_second_monster: bool = False
    is_speed_boost_stage: bool = False
    steps_to_next_stage: int = 0
    alpha: float = 0.0


@dataclass(slots=True)
class LocalMapLayers:
    obstacle: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8))
    hero: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8))
    monster: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8))
    treasure: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8))
    buff: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8))
    visit: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.float32))
    visit_coverage: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.float32))
    flash_landing: np.ndarray = field(default_factory=lambda: np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8))

    def as_stack(self) -> np.ndarray:
        return np.stack([
            self.obstacle,
            self.hero,
            self.monster,
            self.treasure,
            self.buff,
            self.visit,
            self.flash_landing,
        ], axis=0,)


@dataclass(slots=True)
class ExtractorSnapshot:
    raw: RawObs | None = None
    extra: ExtraInfo | None = None
    hero_speed: int = 1
    map_explore_rate: float = 0.0
    map_new_discover: int = 0
    action_predict: ActionPredict = field(default_factory=ActionPredict)
    action_last: ActionLast = field(default_factory=ActionLast)
    monster_summary: MonsterSummary = field(default_factory=MonsterSummary)
    resource_summary: ResourceSummary = field(default_factory=ResourceSummary)
    space_summary: SpaceSummary = field(default_factory=SpaceSummary)
    global_summary: GlobalSummary = field(default_factory=GlobalSummary)
    stage_info: StageInfo = field(default_factory=StageInfo)
    local_map_layers: LocalMapLayers = field(default_factory=LocalMapLayers)
