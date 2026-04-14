from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

from .constant import *


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
    hero_relative_direction: int = 0
    """
    相对于英雄的方位 (0-8), 来自环境原始字段。
    """

    @classmethod
    def from_env(cls, obs: dict) -> Character:
        raise NotImplementedError


@dataclass(slots=True)
class Hero(Character):
    buff_remaining_time: int = 0
    flash_cooldown: int = 0

    @property
    def can_flash(self) -> bool:
        return self.flash_cooldown == 0

    @classmethod
    def from_env(cls, obs: dict) -> Hero:
        raise NotImplementedError


@dataclass(slots=True)
class Monster(Character):
    monster_interval: int = 0
    speed: int = 0
    is_in_view: bool = False

    @classmethod
    def from_env(cls, obs: dict) -> Monster:
        raise NotImplementedError


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
    def from_env(cls, obs: dict) -> Organ:
        raise NotImplementedError


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
    def from_env(cls, env_obs: dict) -> RawObs:
        raise NotImplementedError


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
        raise NotImplementedError


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
