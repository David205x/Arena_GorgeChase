from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque

from .dataclass import *
from .constant import *
from .utils import build_local_window, is_pos_neighbor


class Extractor:
    def __init__(self):
        # ========== current / previous
        self.current: ExtractorSnapshot = ExtractorSnapshot()
        self.current_extra: ExtractorSnapshot = ExtractorSnapshot()
        self.previous: ExtractorSnapshot = ExtractorSnapshot()
        # ========== map cache
        self.map_id: int
        self.map_full: np.ndarray = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int8)
        """-1=未探索, 0=已探索不可通行, 1=已探索可通行"""
        self.visit_count: np.ndarray = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
        # ========== resource cache
        self.treasure_full: list[Organ | None] = []
        """索引与 treasure config_id 对齐。"""
        self.buff_full: dict[int, Organ | None] = {}
        # ========== trajectory cache
        self.pos_history: Deque[tuple[int, int]] = deque(maxlen=POS_HISTORY_LEN)
        # ========== lifecycle
        self.initialized: bool = False
        self.terminated: bool = False
        self.truncated: bool = False

    def reset(self) -> None:
        """
        重置 extractor 的全部跨步状态与缓存。
        - 在每局开始前调用，与环境 `reset()` 保持同步。
        功能：
        - 清空 `current` / `previous` 快照。
        - 清空全局地图缓存、访问计数、资源缓存、历史轨迹。
        - 恢复 `initialized`、`terminated`、`truncated` 等生命周期状态。
        """
        self.current = ExtractorSnapshot()
        self.current_extra = ExtractorSnapshot()
        self.previous = ExtractorSnapshot()
        self.map_id = -1
        self.map_full = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int8)
        self.visit_count = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
        self.treasure_full = []
        self.buff_full = {}
        self.pos_history = deque(maxlen=POS_HISTORY_LEN)
        self.initialized = False
        self.terminated = False
        self.truncated = False

    def update(
        self,
        env_obs: dict,
        extra_info: dict | None = None,
        terminated: bool = False,
        truncated: bool = False
    ) -> ExtractorSnapshot:
        """
        单步更新 extractor 主状态。

        调用顺序建议：
        1. 构造 RawObs
        2. 滚动 previous/current
        3. 更新地图缓存、访问缓存、资源缓存
        4. 更新阶段信息、对象摘要、空间统计、动作反馈
        5. 返回 current snapshot
        """
        raw = self.build_raw_obs(env_obs)
        self.terminated = terminated
        self.truncated = truncated
        self.roll_snapshots(raw)
        self.init_resource_cache(raw)
        self.update_map_full(raw)
        self.update_visit_count(raw)
        self.update_pos_history()
        self.update_treasure_cache(raw)
        self.update_buff_cache(raw)
        self.initialized = True
        return self.current

    # ==================== raw / lifecycle ====================
    def build_raw_obs(self, env_obs: dict) -> RawObs:
        """
        将单帧环境观测解包为 `RawObs`。

        使用场景：
        - `update()` 的最开始阶段。
        - 需要把原始环境字典统一转换成 extractor 内部标准结构时。

        功能：
        - 读取 `step`、`legal_action`、`map_view`、英雄、怪物、资源与统计字段。
        - 仅做“忠实映射”的原始封装，不做跨步比较，不更新缓存。

        输入：
        - `env_obs`: 单帧 observation 字典，结构应与环境返回的 `observation` 一致。

        输出：
        - `RawObs`: 当前帧的原始结构化观测对象。
        """
        return RawObs.from_env(env_obs)

    def roll_snapshots(self, raw: RawObs) -> None:
        """
        将 extractor 的当前快照滚动为上一帧，并为当前帧准备新的快照容器。

        使用场景：
        - `build_raw_obs()` 之后、各类缓存更新和派生计算之前。
        - 所有依赖“当前帧 vs 上一帧”的逻辑前置步骤。

        功能：
        - 将旧的 `current` 迁移到 `previous`。
        - 基于本帧 `raw` 初始化新的 `current` 基础状态。
        - 为奖励增量、动作反馈、地图增量等跨步逻辑提供统一时间线。

        输入：
        - `raw`: 当前帧的 `RawObs`。

        输出：
        - 无返回值，直接更新 `self.current` 与 `self.previous`。
        """
        self.previous = self.current
        self.current = ExtractorSnapshot(raw=raw)

    # ==================== cache update ====================
    def init_resource_cache(self, raw: RawObs) -> None:
        """
        按当前环境配置初始化资源缓存容器。

        使用场景：
        - 通常在首帧更新时调用。
        - 当 `treasure_full` / `buff_full` 还未根据总资源数量建立结构时调用。

        功能：
        - 根据 `raw.total_treasure` 和 `raw.total_buff` 创建资源缓存。
        - 为后续“发现 / 未发现 / 已收集 / 冷却中”的状态维护提供存储空间。

        输入：
        - `raw`: 当前帧原始观测，主要提供资源总量配置。

        输出：
        - 无返回值，直接更新 extractor 内部资源缓存结构。
        """
        if not self.treasure_full:
            self.treasure_full = [None] * raw.total_treasure
        if not self.buff_full:
            self.buff_full = {i: None for i in range(raw.total_treasure, raw.total_treasure + raw.total_buff)}

    def update_map_full(self, raw: RawObs) -> None:
        """
        将当前 21×21 局部视野写回全局地图缓存 `map_full`。

        使用场景：
        - 每帧更新时调用。
        - 当 `obs` / `reward` 需要使用历史探索信息、全局已知地图时调用前置。

        功能：
        - 根据英雄当前位置，将 `raw.map_view` 对齐写入全局坐标系。
        - 维护“未探索 / 已探索可通行 / 已探索不可通行”的地图状态。

        输入：
        - `raw`: 当前帧原始观测，至少需要英雄坐标与局部地图。

        输出：
        - 无返回值，直接更新 `self.map_full`。
        """
        self.map_full = build_local_window(self.map_full, raw.hero.x, raw.hero.z)
        x_min = raw.hero.x - VIEW_CENTER
        x_max = raw.hero.x + VIEW_CENTER + 1
        z_min = raw.hero.z - VIEW_CENTER
        z_max = raw.hero.z + VIEW_CENTER + 1

        global_x_start = max(0, x_min)
        global_x_end = min(MAP_SIZE, x_max)
        global_z_start = max(0, z_min)
        global_z_end = min(MAP_SIZE, z_max)

        view_x_start = max(0, -x_min)
        view_x_end = VIEW_SIZE - max(0, x_max - MAP_SIZE)
        view_z_start = max(0, -z_min)
        view_z_end = VIEW_SIZE - max(0, z_max - MAP_SIZE)

        self.map_full[global_z_start:global_z_end, global_x_start:global_x_end] = raw.map_view[
            view_z_start:view_z_end,
            view_x_start:view_x_end,
        ]

    def update_visit_count(self, raw: RawObs) -> None:
        """
        更新英雄位置访问计数。

        使用场景：
        - 每帧更新时调用。
        - 当 reward 需要评估低效移动、重复绕路，或 obs 需要构造 visit 通道时使用。

        功能：
        - 对当前英雄所在格子的访问次数加一。
        - 为局部访问热度、历史访问分布等派生信息提供底层数据。

        输入：
        - `raw`: 当前帧原始观测，主要使用英雄绝对坐标。

        输出：
        - 无返回值，直接更新 `self.visit_count`。
        """
        self.visit_count[raw.hero.z, raw.hero.x] += 1

    def update_pos_history(self) -> None:
        """
        维护英雄历史位置轨迹缓存。

        使用场景：
        - 在当前帧与上一帧快照均已就绪后调用。
        - 当需要支持怪物追踪分析、路径重复检测、低效移动判定时使用。

        功能：
        - 将上一帧英雄位置写入 `pos_history`。
        - 保持固定长度的时间窗口轨迹，供后续空间与行为分析复用。

        输入：
        - 无显式参数，内部依赖 `self.previous` / `self.current`。

        输出：
        - 无返回值，直接更新 `self.pos_history`。
        """
        if self.previous.raw is None:
            return
        hero = self.previous.raw.hero
        self.pos_history.append((hero.x, hero.z))

    def update_treasure_cache(self, raw: RawObs) -> None:
        """
        更新已知宝箱的全局缓存状态。

        使用场景：
        - 每帧更新资源信息时调用。
        - 当 obs / reward 需要基于“历史已知宝箱”而非“当前可见宝箱”做判断时使用。

        功能：
        - 将当前视野内宝箱写入 `treasure_full`。
        - 结合 `raw.treasure_id` 维护宝箱的发现、未收集、已收集状态。
        - 为最近已知宝箱、宝箱发现进度、宝箱收集进度提供底层缓存。

        输入：
        - `raw`: 当前帧原始观测，包含当前可见宝箱与剩余宝箱 id 列表。

        输出：
        - 无返回值，直接更新 `self.treasure_full`。
        """
        for treasure in raw.treasures:
            self.treasure_full[treasure.id] = treasure

        for treasure in self.treasure_full:
            if treasure is None:
                continue
            if treasure.id not in raw.treasure_id:
                treasure.status = 0

    def update_buff_cache(self, raw: RawObs) -> None:
        """
        更新已知 buff 点的全局缓存状态。

        使用场景：
        - 每帧更新资源信息时调用。
        - 当 reward 需要识别 buff 获取事件，或 obs 需要查询最近已知 buff 时使用。

        功能：
        - 将当前视野内 buff 写入 `buff_full`。
        - 基于历史记录与环境规则维护 buff 是否可获取、是否处于 cooldown。
        - 为最近已知 buff、buff 发现进度、buff 刷新状态提供统一缓存来源。

        输入：
        - `raw`: 当前帧原始观测，包含当前可见 buff 与 buff 相关统计字段。

        输出：
        - 无返回值，直接更新 `self.buff_full`。
        """
        for buff in raw.buffs:
            self.buff_full[buff.id] = buff

        for buff in self.buff_full.values():
            if buff is None:
                continue
            if is_pos_neighbor(raw.hero.x, raw.hero.z, buff.x, buff.z) and raw.hero.buff_remaining_time == 49:
                buff.cooldown = raw.buff_refresh_time
            buff.cooldown = max(buff.cooldown - 1, 0)

    # ==================== derived summaries ====================
    def compute_hero_speed(self) -> int:
        """
        计算英雄当前帧的等效速度。

        使用场景：
        - 更新当前帧快照的基础派生信息时调用。
        - 当 obs / reward 需要区分普通状态与 buff 加速状态时使用。

        功能：
        - 根据英雄 buff 持续时间与环境规则，推导当前每步移动能力。
        - 为动作质量分析、怪物压力评估与空间估计提供基础量。

        输入：
        - 无显式参数，内部通常读取 `self.current.raw.hero`。

        输出：
        - `int`: 当前英雄速度。
        """
        raise NotImplementedError

    def compute_map_statistics(self) -> tuple[float, int]:
        """
        统计当前全局地图缓存的探索进度与本步新增探索量。

        使用场景：
        - 每帧地图缓存更新后调用。
        - 当 obs 需要地图探索程度、reward 需要探索增量 shaping 时使用。

        功能：
        - 计算当前已探索格子占全图的比例。
        - 对比上一帧统计量，得到本步新探索格数。

        输入：
        - 无显式参数，内部依赖 `self.map_full` 以及前一帧快照统计值。

        输出：
        - `tuple[float, int]`: `(map_explore_rate, map_new_discover)`。
        """
        raise NotImplementedError

    def compute_reward_delta(self) -> RewardDelta:
        """
        汇总当前帧相对于上一帧的环境得分增量。

        使用场景：
        - reward 相关状态更新时调用。
        - 当需要统一访问总分、步数分、宝箱分、资源获取次数等跨步增量时使用。

        功能：
        - 比较当前帧与上一帧的分数和统计字段。
        - 生成 `RewardDelta`，供动作反馈和 reward 导出复用。

        输入：
        - 无显式参数，内部依赖 `self.current.raw` 与 `self.previous.raw`。

        输出：
        - `RewardDelta`: 当前帧相对上一帧的环境增量摘要。
        """
        raise NotImplementedError

    def compute_action_feedback(self) -> ActionFeedback:
        """
        计算上一动作在当前帧体现出的结果反馈。

        使用场景：
        - 每帧派生动作后果时调用。
        - 当 obs 需要历史行为反馈、reward 需要动作质量 shaping 时使用。

        功能：
        - 判断是否发生有效移动。
        - 判断最近怪距离是否增加、是否拾取资源、是否带来新探索。
        - 汇总与动作结果强相关的环境反馈信号。

        输入：
        - 无显式参数，内部依赖当前帧与上一帧快照及 `RewardDelta`。

        输出：
        - `ActionFeedback`: 当前帧对上一动作的结构化反馈结果。
        """
        raise NotImplementedError

    def compute_action_result(self) -> ActionResult:
        """
        计算当前帧面向下一步决策的动作结果摘要。

        使用场景：
        - obs 构造前调用。
        - 当策略或规则逻辑需要查询普通移动合法性、闪现落点与动作先验时使用。

        功能：
        - 生成普通移动 8 方向可行性。
        - 生成闪现 8 方向落点、相对位移、有效性与位移长度。
        - 维护当前局面下的基础动作偏好信息。

        输入：
        - 无显式参数，内部依赖当前帧局部地图、英雄位置与动作规则。

        输出：
        - `ActionResult`: 面向下一步动作选择的结构化结果。
        """
        raise NotImplementedError

    def compute_monster_summary(self) -> MonsterSummary:
        """
        汇总当前帧怪物压力相关的基础摘要。

        使用场景：
        - 每帧派生怪物信息时调用。
        - 当 obs / reward 需要最近怪物、第二怪物、距离变化等统一入口时使用。

        功能：
        - 统计怪物数量。
        - 识别最近怪物与第二怪物，并计算对应距离。
        - 生成与上一帧比较所需的基础压力量。

        输入：
        - 无显式参数，内部依赖当前帧和上一帧怪物状态。

        输出：
        - `MonsterSummary`: 当前帧怪物压力摘要。
        """
        raise NotImplementedError

    def compute_resource_summary(self) -> ResourceSummary:
        """
        汇总当前帧资源机会相关的基础摘要。

        使用场景：
        - 每帧派生资源信息时调用。
        - 当 obs / reward 需要最近已知宝箱、最近已知 buff、发现进度等统一入口时使用。

        功能：
        - 查找最近已知宝箱与最近已知 buff。
        - 统计宝箱 / buff 的发现进度与收集进度。
        - 为资源推进类观测和 shaping 提供标准化输入。

        输入：
        - 无显式参数，内部依赖资源全局缓存与当前英雄位置。

        输出：
        - `ResourceSummary`: 当前帧资源机会摘要。
        """
        raise NotImplementedError

    def compute_space_summary(self) -> SpaceSummary:
        """
        汇总当前帧局部地形与活动空间的基础统计。

        使用场景：
        - 每帧更新空间特征时调用。
        - 当 obs / reward 需要通路长度、开阔度、死角风险等信息时使用。

        功能：
        - 计算八方向通路长度、可活动空间、开阔度、安全方向数量。
        - 生成死角、走廊、低开阔等局部结构判定。
        - 为后续空间变化惩罚和生存分析提供基础量。

        输入：
        - 无显式参数，内部依赖当前局部地图与必要的历史统计。

        输出：
        - `SpaceSummary`: 当前帧空间结构摘要。
        """
        raise NotImplementedError

    def compute_stage_info(self) -> StageInfo:
        """
        计算当前帧所属阶段及阶段权重相关信息。

        使用场景：
        - 每帧更新节奏信息时调用。
        - 当 obs 需要阶段观测、reward 需要阶段权重 `alpha` 时使用。

        功能：
        - 结合当前步数、怪物数量、怪物加速节点判断当前阶段。
        - 计算距离下一个关键阶段的剩余步数。
        - 生成 reward 混合权重所需的阶段参数。

        输入：
        - 无显式参数，内部依赖当前原始观测与环境配置字段。

        输出：
        - `StageInfo`: 当前帧阶段与节奏摘要。
        """
        raise NotImplementedError

    def compute_local_map_layers(self) -> LocalMapLayers:
        """
        构造当前帧局部地图多通道表达。

        使用场景：
        - obs 需要直接消费 `21×21×n` 局部地图输入时调用。
        - 调试局部空间表达是否完整、各通道是否对齐时使用。

        功能：
        - 生成障碍、英雄、怪物、宝箱、buff、访问次数等局部地图通道。
        - 统一局部图的坐标对齐方式，作为视觉/空间输入的底层表示。

        输入：
        - 无显式参数，内部依赖当前视野、实体位置和历史访问缓存。

        输出：
        - `LocalMapLayers`: 当前帧局部地图多通道结构。
        """
        raise NotImplementedError

    # ==================== shared query interface ====================
    def get_nearest_monsters(self) -> tuple[Monster | None, Monster | None]:
        """
        查询当前帧距离英雄最近的两只怪物。

        使用场景：
        - `compute_monster_summary()` 的底层查询接口。
        - reward / debug 需要直接访问最近怪与第二怪时使用。

        功能：
        - 对当前怪物按有效距离排序。
        - 返回最近怪物和第二怪物，不存在时返回 `None`。

        输入：
        - 无显式参数，内部依赖当前帧怪物状态。

        输出：
        - `tuple[Monster | None, Monster | None]`: `(nearest_monster, second_monster)`。
        """
        raise NotImplementedError

    def get_nearest_known_treasure(self) -> Organ | None:
        """
        查询当前已知宝箱中距离英雄最近的一个。

        使用场景：
        - `compute_resource_summary()` 内部调用。
        - reward 需要比较是否更接近最近宝箱时使用。

        功能：
        - 基于历史已知且满足可用条件的宝箱缓存进行距离筛选。
        - 返回最近目标，不存在时返回 `None`。

        输入：
        - 无显式参数，内部依赖 `treasure_full` 与当前英雄位置。

        输出：
        - `Organ | None`: 最近已知宝箱。
        """
        raise NotImplementedError

    def get_nearest_known_buff(self) -> Organ | None:
        """
        查询当前已知 buff 点中距离英雄最近的一个。

        使用场景：
        - `compute_resource_summary()` 内部调用。
        - reward / obs 需要评估最近 buff 机会时使用。

        功能：
        - 基于历史已知 buff 缓存筛选最近目标。
        - 可结合是否可获取、是否处于 cooldown 等条件过滤。

        输入：
        - 无显式参数，内部依赖 `buff_full` 与当前英雄位置。

        输出：
        - `Organ | None`: 最近已知 buff；不存在时返回 `None`。
        """
        raise NotImplementedError

    def get_known_treasures(self, only_available: bool = True) -> list[Organ]:
        """
        返回当前缓存中满足条件的已知宝箱列表。

        使用场景：
        - 资源摘要、目标选择、调试输出时使用。
        - 当需要统一遍历“历史已知宝箱”而非仅当前视野宝箱时使用。

        功能：
        - 从 `treasure_full` 中筛选有效条目。
        - 根据 `only_available` 控制是否仅返回当前仍可收集的宝箱。

        输入：
        - `only_available`: 是否只返回仍可获取的宝箱。

        输出：
        - `list[Organ]`: 满足条件的宝箱列表。
        """
        raise NotImplementedError

    def get_known_buffs(self, only_available: bool = False) -> list[Organ]:
        """
        返回当前缓存中满足条件的已知 buff 列表。

        使用场景：
        - 资源摘要、reward 辅助判断、调试输出时使用。
        - 当需要统一遍历“历史已知 buff 点”时使用。

        功能：
        - 从 `buff_full` 中筛选有效条目。
        - 根据 `only_available` 控制是否只返回当前可获取的 buff。

        输入：
        - `only_available`: 是否仅返回当前可用的 buff。

        输出：
        - `list[Organ]`: 满足条件的 buff 列表。
        """
        raise NotImplementedError

    def get_move_valid_mask(self) -> list[bool]:
        """
        返回普通移动 8 方向的有效性掩码。

        使用场景：
        - `compute_action_result()` 内部使用。
        - 策略前处理或规则过滤需要快速获取移动可行方向时使用。

        功能：
        - 根据当前局部地图和移动规则判断八方向移动是否可执行。
        - 输出与动作定义顺序一致的布尔掩码。

        输入：
        - 无显式参数，内部依赖当前局部地图与英雄所在位置。

        输出：
        - `list[bool]`: 八方向普通移动有效性列表。
        """
        raise NotImplementedError

    def get_flash_result(self) -> ActionResult:
        """
        返回当前帧与闪现动作相关的结果集合。

        使用场景：
        - obs 需要直接使用闪现落点、位移长度、有效性时使用。
        - 规则或 reward 需要分析闪现质量时使用。

        功能：
        - 统一暴露闪现 8 方向落点、相对位移、有效性和位移长度。
        - 避免上层重复调用底层闪现推导逻辑。

        输入：
        - 无显式参数，内部通常复用 `compute_action_result()` 的结果。

        输出：
        - `ActionResult`: 包含闪现相关字段的动作结果结构。
        """
        raise NotImplementedError

    def get_stage_alpha(self) -> float:
        """
        返回当前阶段对应的 reward 混合权重 `alpha`。

        使用场景：
        - reward 计算中需要按阶段平衡“生存项”和“资源推进项”时使用。
        - 调试阶段切换是否合理时也可直接查询。

        功能：
        - 从当前阶段信息中提取统一的阶段权重。
        - 避免 reward 模块自行重复推导阶段系数。

        输入：
        - 无显式参数，内部依赖 `self.current.stage_info`。

        输出：
        - `float`: 当前阶段的混合权重。
        """
        raise NotImplementedError

    # ==================== obs / reward export ====================
    def build_obs_state(self) -> dict:
        """
        构造供 obs 模块消费的统一结构化状态。

        使用场景：
        - obs 构建流程的直接输入接口。
        - 当上层不希望直接访问 extractor 内部细粒度字段时使用。

        功能：
        - 将英雄、怪物、资源、空间、阶段、局部地图、动作结果等信息整理为稳定字典。
        - 屏蔽底层缓存细节，向 obs 暴露语义稳定的数据视图。

        输入：
        - 无显式参数，内部依赖当前快照与各类缓存。

        输出：
        - `dict`: 面向 obs 模块的结构化状态字典。
        """
        raise NotImplementedError

    def build_reward_state(self) -> dict:
        """
        构造供 reward 模块消费的统一结构化状态。

        使用场景：
        - reward 计算流程入口。
        - 当 reward 需要统一访问变化型字段、阶段信息、空间风险、资源推进信息时使用。

        功能：
        - 汇总 reward 所需的环境增量、动作反馈、怪物压力、资源机会、空间结构与阶段权重。
        - 将 reward 逻辑依赖的数据集中暴露，避免 reward 直接回到底层缓存取数。

        输入：
        - 无显式参数，内部依赖当前/上一帧快照与各类缓存。

        输出：
        - `dict`: 面向 reward 模块的结构化状态字典。
        """
        raise NotImplementedError

    def build_debug_state(self) -> dict:
        """
        构造供日志、调试、分析使用的辅助状态。

        使用场景：
        - 调试 extractor 内部状态是否正确。
        - 分析地图缓存、资源缓存、阶段切换和动作反馈是否符合预期。

        功能：
        - 暴露当前快照、上一帧快照、地图缓存、资源缓存等调试价值高的数据。
        - 为实验排查与可视化分析提供统一的辅助出口。

        输入：
        - 无显式参数，内部依赖 extractor 当前内部完整状态。

        输出：
        - `dict`: 面向日志与调试分析的辅助状态字典。
        """
        raise NotImplementedError
