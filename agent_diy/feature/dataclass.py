from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Any

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
    def from_env(cls, obs: dict[str, Any]) -> Character:
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
            monster_interval=int(obs["monster_interval"]),
            speed=int(obs["speed"]),
            is_in_view=bool(obs["is_in_view"]),
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
class RewardDelta:
    total_score_delta: float = 0.0
    step_score_delta: float = 0.0
    treasure_score_delta: float = 0.0
    treasures_collected_delta: int = 0
    collected_buff_delta: int = 0
    flash_count_delta: int = 0


@dataclass(slots=True)
class ActionResult:
    move_valid_mask: list[bool] = field(default_factory=lambda: [False] * 8)
    flash_pos: list[tuple[int, int]] = field(default_factory=list)
    flash_pos_relative: list[tuple[int, int]] = field(default_factory=list)
    flash_valid_mask: list[bool] = field(default_factory=lambda: [False] * 8)
    flash_distance: list[float] = field(default_factory=lambda: [0.0] * 8)
    action_preferred: list[bool] = field(default_factory=list)


@dataclass(slots=True)
class ActionFeedback:
    moved: bool = False
    moved_effectively: bool = False
    nearest_monster_distance_increased: bool = False
    picked_treasure: bool = False
    picked_buff: bool = False
    gained_resource: bool = False
    explored_new_area: bool = False
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


@dataclass(slots=True)
class ResourceSummary:
    nearest_known_treasure: Organ | None = None
    nearest_known_treasure_distance: float | None = None
    nearest_known_buff: Organ | None = None
    nearest_known_buff_distance: float | None = None
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

    def as_stack(self) -> np.ndarray:
        return np.stack([
            self.obstacle,
            self.hero,
            self.monster,
            self.treasure,
            self.buff,
            self.visit,
        ], axis=0,)


@dataclass(slots=True)
class ExtractorSnapshot:
    raw: RawObs | None = None
    hero_speed: int = 1
    map_explore_rate: float = 0.0
    map_new_discover: int = 0
    action_result: ActionResult = field(default_factory=ActionResult)
    action_feedback: ActionFeedback = field(default_factory=ActionFeedback)
    monster_summary: MonsterSummary = field(default_factory=MonsterSummary)
    resource_summary: ResourceSummary = field(default_factory=ResourceSummary)
    space_summary: SpaceSummary = field(default_factory=SpaceSummary)
    stage_info: StageInfo = field(default_factory=StageInfo)
    local_map_layers: LocalMapLayers = field(default_factory=LocalMapLayers)
