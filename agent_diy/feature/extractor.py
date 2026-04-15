from __future__ import annotations
import json
import time
from collections import deque
from pathlib import Path
from typing import Deque

from .dataclass import *
from .constant import *
from .utils import *


class _StepProfiler:
    """轻量级单步计时器，用于诊断 extractor 各阶段开销。"""

    __slots__ = ("_enabled", "_marks", "_last", "_step_count",
                 "_accum", "_report_interval", "_last_report")

    def __init__(self, enabled: bool = False, report_interval: int = 200):
        self._enabled = enabled
        self._marks: list[tuple[str, float]] = []
        self._last: float = 0.0
        self._step_count: int = 0
        self._accum: dict[str, float] = {}
        self._report_interval = report_interval
        self._last_report: int = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, v: bool) -> None:
        self._enabled = v

    def begin(self) -> None:
        if not self._enabled:
            return
        self._marks.clear()
        self._last = time.perf_counter()

    def mark(self, label: str) -> None:
        if not self._enabled:
            return
        now = time.perf_counter()
        self._marks.append((label, now - self._last))
        self._last = now

    def finish(self) -> dict[str, float] | None:
        """返回本步各阶段耗时 (ms)；累计到内部统计。"""
        if not self._enabled:
            return None
        result: dict[str, float] = {}
        total = 0.0
        for label, dt in self._marks:
            ms = dt * 1000.0
            result[label] = ms
            total += ms
            self._accum[label] = self._accum.get(label, 0.0) + ms
        result["_total"] = total
        self._accum["_total"] = self._accum.get("_total", 0.0) + total
        self._step_count += 1
        return result

    def should_report(self) -> bool:
        return self._enabled and self._step_count > 0 and (self._step_count - self._last_report) >= self._report_interval

    def report(self, logger=None) -> str:
        """返回并重置累计统计的格式化字符串。"""
        n = max(self._step_count - self._last_report, 1)
        lines = [f"[Extractor Profiler] avg over {n} steps (ms):"]
        for label in list(self._accum.keys()):
            avg = self._accum[label] / n
            lines.append(f"  {label:30s} {avg:8.3f}")
        report_str = "\n".join(lines)
        if logger:
            logger.info(report_str)
        self._accum.clear()
        self._last_report = self._step_count
        return report_str


class Extractor:
    def __init__(self, profile: bool = False):
        # ========== current / previous
        self.current: ExtractorSnapshot = ExtractorSnapshot()
        self.previous: ExtractorSnapshot = ExtractorSnapshot()
        self.current_extra: ExtraInfo | None = None
        self.previous_extra: ExtraInfo | None = None
        # ========== map cache
        self.map_id: int = -1
        self.map_full: np.ndarray = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int8)
        """-1=未探索, 0=已探索不可通行, 1=已探索可通行"""
        self.map_static: np.ndarray | None = None
        self.map_static_id: int = -1
        self.visit_count: np.ndarray = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
        self.visit_coverage: np.ndarray = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        # ========== resource cache
        self.treasure_full: dict[int, Organ | None] = {}
        """key 为从 1 开始的连续 treasure config_id。"""
        self.buff_full: dict[int, Organ | None] = {}
        # ========== trajectory cache
        self.pos_history: Deque[tuple[int, int]] = deque(maxlen=POS_HISTORY_LEN)
        # ========== lifecycle
        self.initialized: bool = False
        self.terminated: bool = False
        self.truncated: bool = False
        self._last_action: int = -1
        self.episode_stats: EpisodeStats = EpisodeStats()
        # ========== profiling
        self.profiler = _StepProfiler(enabled=profile)

    def reset(self) -> None:
        """在每局开始前调用"""
        self.current = ExtractorSnapshot()
        self.previous = ExtractorSnapshot()
        self.current_extra = None
        self.previous_extra = None
        self.map_id = -1
        self.map_full = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int8)
        self.map_static = None
        self.map_static_id = -1
        self.visit_count = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int32)
        self.visit_coverage = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        self.treasure_full = {}
        self.buff_full = {}
        self.pos_history = deque(maxlen=POS_HISTORY_LEN)
        self.initialized = False
        self.terminated = False
        self.truncated = False
        self._last_action = -1
        self.episode_stats = EpisodeStats()

    def update(
        self,
        env_obs: dict,
        extra_info: dict | None = None,
        terminated: bool = False,
        truncated: bool = False,
        last_action: int = -1,
    ) -> ExtractorSnapshot:
        """单步更新 extractor 主状态"""
        p = self.profiler
        p.begin()

        self.terminated = terminated
        self.truncated = truncated
        self._last_action = last_action

        raw = RawObs.from_env(env_obs)
        self.previous = self.current
        self.previous_extra = self.current_extra
        self.current_extra = ExtraInfo.from_env(extra_info)
        self.current = ExtractorSnapshot(raw=raw, extra=self.current_extra)
        if self.current_extra is not None and self.current_extra.map_id >= 0:
            self.map_id = self.current_extra.map_id
            self.ensure_static_map_loaded(self.map_id)
        p.mark("parse_obs")

        self.init_resource_cache(raw)
        self.update_map_full(raw)
        self.update_visit_count(raw)
        self.update_visit_coverage(raw)
        self.update_pos_history()
        self.update_treasure_cache(raw)
        self.update_buff_cache(raw)
        p.mark("cache_update")

        self.current.hero_speed = self.compute_hero_speed()
        self.current.map_explore_rate, self.current.map_new_discover = self.compute_map_statistics()
        p.mark("map_statistics")

        self.current.monster_summary = self.compute_monster_summary()
        p.mark("monster_summary")

        self.current.resource_summary = self.compute_resource_summary()
        p.mark("resource_summary")

        self.current.stage_info = self.compute_stage_info()
        p.mark("stage_info")

        self.current.action_predict = self.compute_action_predict()
        p.mark("action_predict")

        self.current.local_map_layers = self.compute_local_map_layers()
        p.mark("local_map_layers")

        self.current.action_last = self.compute_action_last()
        p.mark("action_last")

        self.current.space_summary = self.compute_space_summary()
        p.mark("space_summary")

        self.current.global_summary = self.compute_global_summary()
        p.mark("global_summary")

        self.update_episode_stats()
        p.mark("episode_stats")

        p.finish()
        self.initialized = True
        return self.current

    # ======================================== cache update
    def init_resource_cache(self, raw: RawObs) -> None:
        """按当前环境配置初始化资源缓存容器"""
        if not self.treasure_full:
            self.treasure_full = {i: None for i in range(1, raw.total_treasure + 1)}
        if not self.buff_full:
            self.buff_full = {i: None for i in range(raw.total_treasure, raw.total_treasure + raw.total_buff)}

    def update_map_full(self, raw: RawObs) -> None:
        """
        将当前 21×21 局部视野写回全局地图缓存 `map_full`
        """
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
        """更新英雄位置访问计数"""
        self.visit_count[raw.hero.z, raw.hero.x] += 1

    def update_visit_coverage(self, raw: RawObs) -> None:
        """更新带邻域覆盖的访问热度"""
        hero_x, hero_z = raw.hero.x, raw.hero.z
        self.visit_coverage[hero_z, hero_x] += 1.0

        for dz in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dz == 0:
                    continue

                global_x = hero_x + dx
                global_z = hero_z + dz
                if global_x < 0 or global_x >= MAP_SIZE or global_z < 0 or global_z >= MAP_SIZE:
                    continue

                local_x = VIEW_CENTER + dx
                local_z = VIEW_CENTER + dz
                if raw.map_view[local_z, local_x] == 1:
                    self.visit_coverage[global_z, global_x] += 0.5

    def update_pos_history(self) -> None:
        if self.previous.raw is None:
            return
        hero = self.previous.raw.hero
        self.pos_history.append((hero.x, hero.z))

    def update_treasure_cache(self, raw: RawObs) -> None:
        """更新已知宝箱的全局缓存状态"""
        for treasure in raw.treasures:
            self.treasure_full[treasure.id] = treasure

        for treasure in self.treasure_full.values():
            if treasure is None:
                continue
            if treasure.id not in raw.treasure_id:
                treasure.status = 0

    def update_buff_cache(self, raw: RawObs) -> None:
        """更新已知 buff 点的全局缓存状态"""
        for buff in raw.buffs:
            self.buff_full[buff.id] = buff

        for buff in self.buff_full.values():
            if buff is None:
                continue
            if is_pos_neighbor(raw.hero.x, raw.hero.z, buff.x, buff.z) and raw.hero.buff_remaining_time == 49:
                buff.cooldown = raw.buff_refresh_time
            buff.cooldown = max(buff.cooldown - 1, 0)

    def ensure_static_map_loaded(self, map_id: int) -> None:
        """
        加载地图真值通行图，仅供 reward / monitor 侧做路径估计。
        """
        if self.map_static is not None and self.map_static_id == map_id:
            return

        map_path = Path(__file__).resolve().parents[1] / "ref" / "map" / f"gorge_chase_map_{map_id}.json"
        if not map_path.exists():
            self.map_static = None
            self.map_static_id = -1
            return

        with map_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        matrix = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        for cell in data.get("cells", []):
            x = int(cell.get("x", -1))
            z = int(cell.get("z", -1))
            if 0 <= x < MAP_SIZE and 0 <= z < MAP_SIZE and int(cell.get("type_id", 0)) == 1:
                matrix[z, x] = 1
        self.map_static = matrix
        self.map_static_id = map_id

    # ======================================== derived summaries
    def compute_hero_speed(self) -> int:
        """
        计算英雄当前帧的等效速度。
        """
        raw = self.current.raw
        if raw is None:
            return 1
        return 2 if raw.hero.buff_remaining_time > 0 else 1

    def compute_map_statistics(self) -> tuple[float, int]:
        """
        统计当前全局地图缓存的探索进度与本步新增探索量。
        """
        explored_count = int(np.count_nonzero(self.map_full != -1))
        map_explore_rate = explored_count / float(MAP_SIZE * MAP_SIZE)
        previous_rate = self.previous.map_explore_rate if self.previous is not None else 0.0
        previous_explored_count = int(round(previous_rate * MAP_SIZE * MAP_SIZE))
        map_new_discover = max(0, explored_count - previous_explored_count)
        return map_explore_rate, map_new_discover

    def compute_reward_delta(self) -> RewardDelta:
        current = self.current.raw
        previous = self.previous.raw
        if current is None:
            return RewardDelta()
        if previous is None:
            return RewardDelta(
                total_score_delta=current.total_score,
                step_score_delta=current.step_score,
                treasure_score_delta=current.treasure_score,
                treasures_collected_delta=current.treasures_collected,
                collected_buff_delta=current.collected_buff,
                flash_count_delta=current.flash_count,
            )
        return RewardDelta(
            total_score_delta=current.total_score - previous.total_score,
            step_score_delta=current.step_score - previous.step_score,
            treasure_score_delta=current.treasure_score - previous.treasure_score,
            treasures_collected_delta=current.treasures_collected - previous.treasures_collected,
            collected_buff_delta=current.collected_buff - previous.collected_buff,
            flash_count_delta=current.flash_count - previous.flash_count,
        )

    def compute_action_last(self) -> ActionLast:
        """
        计算上一动作在当前帧体现出的结果反馈。
        """
        current = self.current.raw
        previous = self.previous.raw
        reward_delta = self.compute_reward_delta()

        if current is None or previous is None:
            return ActionLast(reward_delta=reward_delta)

        moved = (current.hero.x != previous.hero.x) or (current.hero.z != previous.hero.z)
        moved_delta = (current.hero.x - previous.hero.x, current.hero.z - previous.hero.z)

        nearest_current = self.current.monster_summary.nearest_monster_distance
        nearest_last = self.previous.monster_summary.nearest_monster_distance
        nearest_monster_distance_increased = False
        if nearest_current is not None and nearest_last is not None:
            nearest_monster_distance_increased = nearest_current > nearest_last

        picked_treasure = reward_delta.treasures_collected_delta > 0
        picked_buff = reward_delta.collected_buff_delta > 0
        map_explore_rate_delta = self.current.map_explore_rate - self.previous.map_explore_rate

        return ActionLast(
            last_action_id=self._last_action,
            moved=moved,
            moved_delta=moved_delta,
            nearest_monster_distance_increased=nearest_monster_distance_increased,
            picked_treasure=picked_treasure,
            picked_buff=picked_buff,
            explored_new_area=self.current.map_new_discover > 0,
            map_explore_rate_delta=map_explore_rate_delta,
            reward_delta=reward_delta,
        )

    def compute_action_predict(self) -> ActionPredict:
        """计算当前帧面向下一步决策的动作预测摘要。"""
        move_valid_mask = self.get_move_valid_mask()
        flash_result = self.get_flash_result()

        return ActionPredict(
            move_valid_mask=move_valid_mask,
            flash_pos=flash_result.flash_pos,
            flash_pos_relative=flash_result.flash_pos_relative,
            flash_valid_mask=flash_result.flash_valid_mask,
            flash_distance=flash_result.flash_distance,
            flash_across_wall=flash_result.flash_across_wall,
            action_preferred=move_valid_mask + flash_result.flash_valid_mask, #TODO
        )

    def compute_monster_summary(self) -> MonsterSummary:
        raw = self.current.raw
        if raw is None:
            return MonsterSummary()

        hero = raw.hero
        visible_monsters = sorted(
            raw.monsters,
            key=lambda m: chebyshev_distance(hero.x, hero.z, m.x, m.z),
        )
        nearest_monster = visible_monsters[0] if len(visible_monsters) >= 1 else None
        second_monster = visible_monsters[1] if len(visible_monsters) >= 2 else None

        nearest_distance = None
        second_distance = None
        average_distance = None
        if nearest_monster is not None:
            nearest_distance = chebyshev_distance(hero.x, hero.z, nearest_monster.x, nearest_monster.z)
        if second_monster is not None:
            second_distance = chebyshev_distance(hero.x, hero.z, second_monster.x, second_monster.z)

        distances = [d for d in [nearest_distance, second_distance] if d is not None]
        if distances:
            average_distance = float(sum(distances)) / len(distances)

        last_summary = self.previous.monster_summary
        nearest_distance_last = last_summary.nearest_monster_distance
        nearest_distance_delta = None
        if nearest_distance is not None and nearest_distance_last is not None:
            nearest_distance_delta = nearest_distance - nearest_distance_last

        monster_count = len(raw.monsters)
        monster1 = raw.monsters[0] if len(raw.monsters) >= 1 else None
        monster2 = raw.monsters[1] if len(raw.monsters) >= 2 else None

        def relative_position(monster: Monster | None) -> tuple[int, int]:
            if monster is None:
                return (0, 0)
            return (monster.x - hero.x, monster.z - hero.z)

        def l2_distance(monster: Monster | None) -> float | None:
            if monster is None:
                return None
            return distance_l2(hero.x, hero.z, monster.x, monster.z)

        def bucket_distance(monster: Monster | None) -> int | None:
            if monster is None:
                return None
            return int(monster.hero_l2_distance)

        def cosine_from_vectors(v1: tuple[int, int], v2: tuple[int, int]) -> float | None:
            a = np.asarray(v1, dtype=np.float32)
            b = np.asarray(v2, dtype=np.float32)
            norm = float(np.linalg.norm(a) * np.linalg.norm(b))
            if norm <= 1e-6:
                return None
            return float(np.clip(np.dot(a, b) / norm, -1.0, 1.0))

        monster1_exists = monster1 is not None
        monster2_exists = monster2 is not None
        monster1_steps_to_appear = 0
        monster2_steps_to_appear = 0 if monster2_exists else max(0, raw.monster_interval - raw.step) if raw.monster_interval >= 0 else 0
        monster1_relative_position = relative_position(monster1)
        monster2_relative_position = relative_position(monster2)
        monster1_relative_direction = monster1.hero_relative_direction if monster1 is not None else (0, 0)
        monster2_relative_direction = monster2.hero_relative_direction if monster2 is not None else (0, 0)
        monster1_distance_chebyshev = chebyshev_distance(hero.x, hero.z, monster1.x, monster1.z) if monster1 is not None else None
        monster2_distance_chebyshev = chebyshev_distance(hero.x, hero.z, monster2.x, monster2.z) if monster2 is not None else None
        monster1_distance_l2 = l2_distance(monster1)
        monster2_distance_l2 = l2_distance(monster2)
        monster1_distance_bucket = bucket_distance(monster1)
        monster2_distance_bucket = bucket_distance(monster2)
        monster1_speed = monster1.speed if monster1 is not None else 0
        monster2_speed = monster2.speed if monster2 is not None else 0
        monster1_is_nearest = nearest_monster is not None and monster1 is not None and nearest_monster.id == monster1.id
        monster2_is_nearest = nearest_monster is not None and monster2 is not None and nearest_monster.id == monster2.id
        relative_direction_cosine = cosine_from_vectors(monster1_relative_direction, monster2_relative_direction)

        return MonsterSummary(
            monster_count=monster_count,
            nearest_monster=nearest_monster,
            second_monster=second_monster,
            nearest_monster_distance=nearest_distance,
            second_monster_distance=second_distance,
            nearest_monster_distance_last=nearest_distance_last,
            nearest_monster_distance_delta=nearest_distance_delta,
            average_monster_distance=average_distance,
            monster1_exists=monster1_exists,
            monster2_exists=monster2_exists,
            monster1_steps_to_appear=monster1_steps_to_appear,
            monster2_steps_to_appear=monster2_steps_to_appear,
            monster1_relative_position=monster1_relative_position,
            monster2_relative_position=monster2_relative_position,
            monster1_relative_direction=monster1_relative_direction,
            monster2_relative_direction=monster2_relative_direction,
            monster1_distance_chebyshev=monster1_distance_chebyshev,
            monster2_distance_chebyshev=monster2_distance_chebyshev,
            monster1_distance_l2=monster1_distance_l2,
            monster2_distance_l2=monster2_distance_l2,
            monster1_distance_bucket=monster1_distance_bucket,
            monster2_distance_bucket=monster2_distance_bucket,
            monster1_speed=monster1_speed,
            monster2_speed=monster2_speed,
            monster1_is_nearest=monster1_is_nearest,
            monster2_is_nearest=monster2_is_nearest,
            relative_direction_cosine=relative_direction_cosine,
        )

    def compute_resource_summary(self) -> ResourceSummary:
        raw = self.current.raw
        if raw is None:
            return ResourceSummary()

        hero = raw.hero
        treasures = self.get_known_treasures(only_available=True)
        buffs = self.get_known_buffs(only_available=True)

        # L2 pre-sort for early-termination efficiency
        treasures.sort(key=lambda t: (t.x - hero.x) ** 2 + (t.z - hero.z) ** 2)
        buffs.sort(key=lambda b: (b.x - hero.x) ** 2 + (b.z - hero.z) ** 2)

        # single BFS from hero on known map, early-terminate when all targets found
        targets: set[tuple[int, int]] = set()
        for t in treasures:
            targets.add((t.x, t.z))
        for b in buffs:
            targets.add((b.x, b.z))
        hero_dist = self._bfs_from_hero_known(targets)

        # find nearest treasure (path-first, L2 tiebreaker)
        nearest_treasure: Organ | None = None
        nearest_treasure_distance_path: int | None = None
        nearest_treasure_distance_l2: float | None = None
        nearest_treasure_direction: tuple[int, int] = (0, 0)
        for t in treasures:
            d = int(hero_dist[t.z, t.x])
            pd = d if d >= 0 else None
            l2d = distance_l2(hero.x, hero.z, t.x, t.z)
            if pd is not None:
                if nearest_treasure_distance_path is None or pd < nearest_treasure_distance_path or (
                    pd == nearest_treasure_distance_path and l2d < (nearest_treasure_distance_l2 if nearest_treasure_distance_l2 is not None else float("inf"))
                ):
                    nearest_treasure = t
                    nearest_treasure_distance_path = pd
                    nearest_treasure_distance_l2 = l2d
            elif nearest_treasure_distance_path is None and (nearest_treasure_distance_l2 is None or l2d < nearest_treasure_distance_l2):
                nearest_treasure = t
                nearest_treasure_distance_l2 = l2d
        if nearest_treasure is not None:
            nearest_treasure_direction = (
                int(np.sign(nearest_treasure.x - hero.x)),
                int(np.sign(nearest_treasure.z - hero.z)),
            )

        # find nearest buff (same logic)
        nearest_buff: Organ | None = None
        nearest_buff_distance_path: int | None = None
        nearest_buff_distance_l2: float | None = None
        nearest_buff_direction: tuple[int, int] = (0, 0)
        for b in buffs:
            d = int(hero_dist[b.z, b.x])
            pd = d if d >= 0 else None
            l2d = distance_l2(hero.x, hero.z, b.x, b.z)
            if pd is not None:
                if nearest_buff_distance_path is None or pd < nearest_buff_distance_path or (
                    pd == nearest_buff_distance_path and l2d < (nearest_buff_distance_l2 if nearest_buff_distance_l2 is not None else float("inf"))
                ):
                    nearest_buff = b
                    nearest_buff_distance_path = pd
                    nearest_buff_distance_l2 = l2d
            elif nearest_buff_distance_path is None and (nearest_buff_distance_l2 is None or l2d < nearest_buff_distance_l2):
                nearest_buff = b
                nearest_buff_distance_l2 = l2d
        if nearest_buff is not None:
            nearest_buff_direction = (
                int(np.sign(nearest_buff.x - hero.x)),
                int(np.sign(nearest_buff.z - hero.z)),
            )

        treasure_discovered_count = len(self.get_known_treasures(only_available=False))
        buff_discovered_count = len(self.get_known_buffs(only_available=False))

        treasure_progress = 0.0
        if raw.total_treasure > 0:
            treasure_progress = raw.treasures_collected / raw.total_treasure

        buff_progress = 0.0
        if raw.total_buff > 0:
            buff_progress = raw.collected_buff / raw.total_buff

        prev_rs = self.previous.resource_summary
        treasure_path_last = prev_rs.nearest_known_treasure_distance_path
        treasure_path_delta = None
        if nearest_treasure_distance_path is not None and treasure_path_last is not None:
            treasure_path_delta = nearest_treasure_distance_path - treasure_path_last

        return ResourceSummary(
            nearest_known_treasure=nearest_treasure,
            nearest_known_treasure_distance_l2=nearest_treasure_distance_l2,
            nearest_known_treasure_distance_path=nearest_treasure_distance_path,
            nearest_known_treasure_direction=nearest_treasure_direction,
            nearest_known_buff=nearest_buff,
            nearest_known_buff_distance_l2=nearest_buff_distance_l2,
            nearest_known_buff_distance_path=nearest_buff_distance_path,
            nearest_known_buff_direction=nearest_buff_direction,
            nearest_known_treasure_distance_path_last=treasure_path_last,
            nearest_known_treasure_distance_path_delta=treasure_path_delta,
            treasure_discovered_count=treasure_discovered_count,
            buff_discovered_count=buff_discovered_count,
            treasure_progress=treasure_progress,
            buff_progress=buff_progress,
        )

    def compute_space_summary(self) -> SpaceSummary:
        """
        汇总当前帧局部地形与活动空间的基础统计。
        """
        raw = self.current.raw
        if raw is None:
            return SpaceSummary()

        map_view = raw.map_view
        cx, cz = VIEW_CENTER, VIEW_CENTER

        def is_walkable(nx: int, nz: int) -> bool:
            if nx < 0 or nx >= VIEW_SIZE or nz < 0 or nz >= VIEW_SIZE:
                return False
            return bool(map_view[nz, nx] == 1)

        # 1) 八方向通路长度
        corridor_lengths: list[int] = []
        for dx, dz in MOVE_DIR_VEC:
            length = 0
            nx, nz = cx + dx, cz + dz
            while 0 <= nx < VIEW_SIZE and 0 <= nz < VIEW_SIZE and is_walkable(nx, nz):
                length += 1
                nx += dx
                nz += dz
            corridor_lengths.append(length)

        # 2) 局部可活动空间
        traversable_space = int(np.count_nonzero(map_view == 1))

        # 3) 英雄周围开阔度（8 邻域可走数）
        openness = 0
        for dx, dz in MOVE_DIR_VEC:
            if is_walkable(cx + dx, cz + dz):
                openness += 1

        # 4) 安全方向数：下一步可走，且目标格不与怪物相邻
        monster_positions = {(m.x, m.z) for m in raw.monsters}
        safe_direction_count = 0
        for dx, dz in MOVE_DIR_VEC:
            nx_local = cx + dx
            nz_local = cz + dz
            if not is_walkable(nx_local, nz_local):
                continue

            nx_global = raw.hero.x + dx
            nz_global = raw.hero.z + dz

            is_safe = True
            for mx, mz in monster_positions:
                if chebyshev_distance(nx_global, nz_global, mx, mz) <= 1:
                    is_safe = False
                    break

            if is_safe:
                safe_direction_count += 1

        # 5) 与上一帧比较
        traversable_space_last = self.previous.space_summary.traversable_space
        traversable_space_delta = traversable_space - traversable_space_last

        # 6) 局部结构判定
        is_dead_end = safe_direction_count <= 1
        is_low_openness = openness <= 2
        is_corridor = is_low_openness and max(corridor_lengths, default=0) >= 2

        return SpaceSummary(
            corridor_lengths=corridor_lengths,
            traversable_space=traversable_space,
            openness=openness,
            safe_direction_count=safe_direction_count,
            traversable_space_delta=traversable_space_delta,
            is_dead_end=is_dead_end,
            is_corridor=is_corridor,
            is_low_openness=is_low_openness,
        )

    def compute_global_summary(self) -> GlobalSummary:
        """
        使用 extra_info 与静态全图做更准确的 reward-side 局势估计。

        优化：从英雄做 1 次 BFS（而非从每个怪物各做一次全图 BFS），
        早期终止 + 纯 ndarray（无 parents dict），开销从 O(W²×M) 降为 O(W²)。
        """
        raw = self.current.raw
        extra = self.current_extra
        hero = extra.hero if extra is not None and extra.hero is not None else (raw.hero if raw is not None else None)
        monsters = extra.monsters if extra is not None and extra.monsters else (raw.monsters if raw is not None else [])

        if hero is None or not monsters:
            return GlobalSummary()

        monster_targets = {(m.x, m.z) for m in monsters}
        hero_dist, direction_from = self._bfs_from_hero_static(monster_targets)

        monster_entries: list[dict] = []
        for monster in monsters:
            pd = self._lookup_static_dist(hero_dist, monster.x, monster.z)
            approach = self._trace_approach_direction(
                hero_dist, direction_from, monster.x, monster.z, hero.x, hero.z,
            )
            monster_entries.append({
                "monster": monster,
                "path_distance_estimate": pd,
                "approach_direction_estimate": approach,
            })

        monster_entries.sort(
            key=lambda item: (
                item["path_distance_estimate"] is None,
                item["path_distance_estimate"] if item["path_distance_estimate"] is not None else chebyshev_distance(hero.x, hero.z, item["monster"].x, item["monster"].z),
            )
        )

        nearest_entry = monster_entries[0] if len(monster_entries) >= 1 else None
        second_entry = monster_entries[1] if len(monster_entries) >= 2 else None
        nearest = nearest_entry["monster"] if nearest_entry is not None else None
        second = second_entry["monster"] if second_entry is not None else None

        nearest_distance = chebyshev_distance(hero.x, hero.z, nearest.x, nearest.z) if nearest is not None else None
        second_distance = chebyshev_distance(hero.x, hero.z, second.x, second.z) if second is not None else None
        distances = [d for d in [nearest_distance, second_distance] if d is not None]
        average_distance = float(sum(distances)) / len(distances) if distances else None

        nearest_path_distance_estimate = nearest_entry["path_distance_estimate"] if nearest_entry is not None else None
        second_path_distance_estimate = second_entry["path_distance_estimate"] if second_entry is not None else None
        path_distances = [d for d in [nearest_path_distance_estimate, second_path_distance_estimate] if d is not None]
        average_path_distance_estimate = float(sum(path_distances)) / len(path_distances) if path_distances else None

        capture_margin_path_estimate = max(0, nearest_path_distance_estimate - 1) if nearest_path_distance_estimate is not None else None
        encirclement_path_cosine_estimate = self.compute_approach_cosine_estimate(
            nearest_entry["approach_direction_estimate"] if nearest_entry is not None else (0, 0),
            second_entry["approach_direction_estimate"] if second_entry is not None else (0, 0),
        )
        safe_direction_count = self.count_safe_directions(monsters)
        safe_direction_path_count_estimate = self._count_safe_dirs_adjacent(hero, monsters)

        last_summary = self.previous.global_summary
        nearest_distance_last = last_summary.nearest_monster_distance
        nearest_distance_delta = None
        if nearest_distance is not None and nearest_distance_last is not None:
            nearest_distance_delta = nearest_distance - nearest_distance_last

        nearest_path_distance_last_estimate = last_summary.nearest_monster_path_distance_estimate
        nearest_path_distance_delta_estimate = None
        if nearest_path_distance_estimate is not None and nearest_path_distance_last_estimate is not None:
            nearest_path_distance_delta_estimate = nearest_path_distance_estimate - nearest_path_distance_last_estimate

        capture_margin_path_last_estimate = last_summary.capture_margin_path_estimate
        capture_margin_path_delta_estimate = None
        if capture_margin_path_estimate is not None and capture_margin_path_last_estimate is not None:
            capture_margin_path_delta_estimate = capture_margin_path_estimate - capture_margin_path_last_estimate

        encirclement_path_cosine_last_estimate = last_summary.encirclement_path_cosine_estimate
        encirclement_path_cosine_delta_estimate = None
        if encirclement_path_cosine_estimate is not None and encirclement_path_cosine_last_estimate is not None:
            encirclement_path_cosine_delta_estimate = encirclement_path_cosine_estimate - encirclement_path_cosine_last_estimate

        return GlobalSummary(
            nearest_monster=nearest,
            second_monster=second,
            nearest_monster_distance=nearest_distance,
            second_monster_distance=second_distance,
            nearest_monster_distance_last=nearest_distance_last,
            nearest_monster_distance_delta=nearest_distance_delta,
            average_monster_distance=average_distance,
            safe_direction_count=safe_direction_count,
            safe_direction_count_last=last_summary.safe_direction_count,
            safe_direction_count_delta=safe_direction_count - last_summary.safe_direction_count,
            nearest_monster_path_distance_estimate=nearest_path_distance_estimate,
            second_monster_path_distance_estimate=second_path_distance_estimate,
            nearest_monster_path_distance_last_estimate=nearest_path_distance_last_estimate,
            nearest_monster_path_distance_delta_estimate=nearest_path_distance_delta_estimate,
            average_monster_path_distance_estimate=average_path_distance_estimate,
            capture_margin_path_estimate=capture_margin_path_estimate,
            capture_margin_path_last_estimate=capture_margin_path_last_estimate,
            capture_margin_path_delta_estimate=capture_margin_path_delta_estimate,
            nearest_monster_approach_direction_estimate=nearest_entry["approach_direction_estimate"] if nearest_entry is not None else (0, 0),
            second_monster_approach_direction_estimate=second_entry["approach_direction_estimate"] if second_entry is not None else (0, 0),
            encirclement_path_cosine_estimate=encirclement_path_cosine_estimate,
            encirclement_path_cosine_last_estimate=encirclement_path_cosine_last_estimate,
            encirclement_path_cosine_delta_estimate=encirclement_path_cosine_delta_estimate,
            safe_direction_path_count_estimate=safe_direction_path_count_estimate,
            safe_direction_path_count_last_estimate=last_summary.safe_direction_path_count_estimate,
            safe_direction_path_count_delta_estimate=safe_direction_path_count_estimate - last_summary.safe_direction_path_count_estimate,
            dead_end_under_pressure_estimate=(
                safe_direction_path_count_estimate <= 1
                and capture_margin_path_estimate is not None
                and capture_margin_path_estimate <= 2
            ),
        )

    def compute_stage_info(self) -> StageInfo:
        """
        计算当前帧所属阶段及阶段权重相关信息。
        """
        raw = self.current.raw
        if raw is None:
            return StageInfo()

        has_second_monster = len(raw.monsters) >= 2
        is_speed_boost_stage = raw.step >= raw.monster_speed_boost_step

        if is_speed_boost_stage:
            stage = 3
        elif has_second_monster:
            stage = 2
        else:
            stage = 1

        if stage == 1:
            if raw.monster_interval >= 0:
                steps_to_next_stage = max(0, raw.monster_interval - raw.step)
            else:
                steps_to_next_stage = 0
        elif stage == 2:
            if raw.monster_speed_boost_step >= 0:
                steps_to_next_stage = max(0, raw.monster_speed_boost_step - raw.step)
            else:
                steps_to_next_stage = 0
        else:
            steps_to_next_stage = 0

        return StageInfo(
            stage=stage,
            has_second_monster=has_second_monster,
            is_speed_boost_stage=is_speed_boost_stage,
            steps_to_next_stage=steps_to_next_stage,
            alpha=ALPHA_MAP[stage],
        )

    def compute_local_map_layers(self) -> LocalMapLayers:
        """
        构造当前帧局部地图多通道表达。
        """
        raw = self.current.raw
        if raw is None:
            return LocalMapLayers()

        obstacle = (raw.map_view == 0).astype(np.int8)

        hero = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8)
        hero[VIEW_CENTER, VIEW_CENTER] = 1

        monster = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8)
        for m in raw.monsters:
            dx = m.x - raw.hero.x
            dz = m.z - raw.hero.z
            lx = VIEW_CENTER + dx
            lz = VIEW_CENTER + dz
            if 0 <= lx < VIEW_SIZE and 0 <= lz < VIEW_SIZE:
                monster[lz, lx] = 1

        treasure = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8)
        for t in raw.treasures:
            dx = t.x - raw.hero.x
            dz = t.z - raw.hero.z
            lx = VIEW_CENTER + dx
            lz = VIEW_CENTER + dz
            if 0 <= lx < VIEW_SIZE and 0 <= lz < VIEW_SIZE:
                treasure[lz, lx] = 1

        buff = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8)
        for b in raw.buffs:
            dx = b.x - raw.hero.x
            dz = b.z - raw.hero.z
            lx = VIEW_CENTER + dx
            lz = VIEW_CENTER + dz
            if 0 <= lx < VIEW_SIZE and 0 <= lz < VIEW_SIZE:
                buff[lz, lx] = 1

        visit = build_local_window(self.visit_count, raw.hero.x, raw.hero.z, pad_value=0).astype(np.float32)
        visit_coverage = build_local_window(self.visit_coverage, raw.hero.x, raw.hero.z, pad_value=0).astype(np.float32)
        flash_landing = np.zeros((VIEW_SIZE, VIEW_SIZE), dtype=np.int8)
        if raw.hero.can_flash:
            ap = self.current.action_predict
            for (dx, dz), is_valid in zip(ap.flash_pos_relative, ap.flash_valid_mask):
                if not is_valid:
                    continue
                lx = VIEW_CENTER + dx
                lz = VIEW_CENTER + dz
                if 0 <= lx < VIEW_SIZE and 0 <= lz < VIEW_SIZE:
                    flash_landing[lz, lx] = 1

        return LocalMapLayers(
            obstacle=obstacle,
            hero=hero,
            monster=monster,
            treasure=treasure,
            buff=buff,
            visit=visit,
            visit_coverage=visit_coverage,
            flash_landing=flash_landing,
        )

    def estimate_path_distance_on_known_map(self, start_x: int, start_z: int, target_x: int, target_z: int) -> int | None:
        if not (0 <= start_x < MAP_SIZE and 0 <= start_z < MAP_SIZE and 0 <= target_x < MAP_SIZE and 0 <= target_z < MAP_SIZE):
            return None
        if self.map_full[start_z, start_x] != 1 or self.map_full[target_z, target_x] != 1:
            return None

        distances = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int16)
        queue = deque([(start_x, start_z)])
        distances[start_z, start_x] = 0

        while queue:
            x, z = queue.popleft()
            if x == target_x and z == target_z:
                return int(distances[z, x])
            for dx, dz in MOVE_DIR_VEC:
                nx, nz = x + dx, z + dz
                if not self.can_step_known_map(x, z, nx, nz):
                    continue
                if distances[nz, nx] != -1:
                    continue
                distances[nz, nx] = distances[z, x] + 1
                queue.append((nx, nz))
        return None

    def can_step_known_map(self, x: int, z: int, nx: int, nz: int) -> bool:
        if nx < 0 or nx >= MAP_SIZE or nz < 0 or nz >= MAP_SIZE:
            return False
        if self.map_full[nz, nx] != 1:
            return False
        dx = nx - x
        dz = nz - z
        if abs(dx) > 1 or abs(dz) > 1:
            return False
        if dx != 0 and dz != 0:
            side1_ok = 0 <= x + dx < MAP_SIZE and self.map_full[z, x + dx] == 1
            side2_ok = 0 <= z + dz < MAP_SIZE and self.map_full[z + dz, x] == 1
            return bool(side1_ok or side2_ok)
        return True

    def is_walkable_static(self, x: int, z: int) -> bool:
        if x < 0 or x >= MAP_SIZE or z < 0 or z >= MAP_SIZE:
            return False
        if self.map_static is None:
            return False
        return bool(self.map_static[z, x] == 1)

    def can_step_static(self, x: int, z: int, nx: int, nz: int) -> bool:
        if not self.is_walkable_static(nx, nz):
            return False
        dx = nx - x
        dz = nz - z
        if abs(dx) > 1 or abs(dz) > 1:
            return False
        if dx != 0 and dz != 0:
            return self.is_walkable_static(x + dx, z) or self.is_walkable_static(x, z + dz)
        return True

    # ==================== BFS helpers (optimized) ====================

    def _bfs_from_hero_static(
        self, targets: set[tuple[int, int]]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        从英雄位置在静态地图上做一次 BFS。
        返回 (distances, direction_from)，均为 ndarray。
        direction_from[z, x] = 到达该格时使用的 MOVE_DIR_VEC 索引 (0-7), -1=未访问。
        当 targets 全部找到时早期终止。
        """
        raw = self.current.raw
        extra = self.current_extra
        hero = extra.hero if extra is not None and extra.hero is not None else (raw.hero if raw is not None else None)

        distances = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int16)
        direction_from = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int8)

        if hero is None or not self.is_walkable_static(hero.x, hero.z):
            return distances, direction_from

        hx, hz = hero.x, hero.z
        distances[hz, hx] = 0
        queue = deque([(hx, hz)])
        remaining = set(targets)
        remaining.discard((hx, hz))

        while queue:
            x, z = queue.popleft()
            if not remaining:
                break
            for di, (dx, dz) in enumerate(MOVE_DIR_VEC):
                nx, nz = x + dx, z + dz
                if not self.can_step_static(x, z, nx, nz):
                    continue
                if distances[nz, nx] != -1:
                    continue
                distances[nz, nx] = distances[z, x] + 1
                direction_from[nz, nx] = di
                queue.append((nx, nz))
                remaining.discard((nx, nz))

        return distances, direction_from

    @staticmethod
    def _lookup_static_dist(distances: np.ndarray, x: int, z: int) -> int | None:
        if x < 0 or x >= MAP_SIZE or z < 0 or z >= MAP_SIZE:
            return None
        d = int(distances[z, x])
        return None if d < 0 else d

    def _trace_approach_direction(
        self,
        hero_distances: np.ndarray,
        direction_from: np.ndarray,
        target_x: int,
        target_z: int,
        hero_x: int,
        hero_z: int,
        tail_steps: int = 3,
    ) -> tuple[int, int]:
        """
        从 target 沿前驱链回溯到 hero，取路径末尾 tail_steps 计算接近方向。
        语义与原始 compute_path_tail_direction_estimate 一致。
        """
        d = int(hero_distances[target_z, target_x]) if (
            0 <= target_x < MAP_SIZE and 0 <= target_z < MAP_SIZE
        ) else -1
        if d <= 0:
            return (0, 0)

        path: list[tuple[int, int]] = [(target_x, target_z)]
        x, z = target_x, target_z
        while int(hero_distances[z, x]) > 0:
            di = int(direction_from[z, x])
            if di < 0:
                break
            dx, dz = MOVE_DIR_VEC[di]
            x, z = x - dx, z - dz
            path.append((x, z))

        if len(path) < 2:
            return (0, 0)
        anchor_idx = max(0, len(path) - 1 - tail_steps)
        return (
            int(np.sign(path[-1][0] - path[anchor_idx][0])),
            int(np.sign(path[-1][1] - path[anchor_idx][1])),
        )

    def _bfs_from_hero_known(self, targets: set[tuple[int, int]]) -> np.ndarray:
        """
        从英雄位置在已探索地图 (map_full) 上做 BFS。
        当 targets 全部找到时早期终止。
        """
        raw = self.current.raw
        hero = raw.hero if raw is not None else None
        distances = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int16)

        if hero is None or self.map_full[hero.z, hero.x] != 1:
            return distances

        hx, hz = hero.x, hero.z
        distances[hz, hx] = 0
        queue = deque([(hx, hz)])
        remaining = set(targets)
        remaining.discard((hx, hz))

        while queue:
            x, z = queue.popleft()
            if not remaining:
                break
            for dx, dz in MOVE_DIR_VEC:
                nx, nz = x + dx, z + dz
                if not self.can_step_known_map(x, z, nx, nz):
                    continue
                if distances[nz, nx] != -1:
                    continue
                distances[nz, nx] = distances[z, x] + 1
                queue.append((nx, nz))
                remaining.discard((nx, nz))

        return distances

    def _count_safe_dirs_adjacent(self, hero, monsters) -> int:
        """
        等价于原 count_safe_directions_path_estimate 的 distance<=1 判断，
        但无需怪物 BFS 距离场。
        判断英雄邻居格子是否在任一怪物的一步可达范围内。
        """
        safe_count = 0
        for dx, dz in MOVE_DIR_VEC:
            nx, nz = hero.x + dx, hero.z + dz
            if not self.can_step_static(hero.x, hero.z, nx, nz):
                continue
            is_safe = True
            for m in monsters:
                if (nx == m.x and nz == m.z) or self.can_step_static(m.x, m.z, nx, nz):
                    is_safe = False
                    break
            if is_safe:
                safe_count += 1
        return safe_count

    def build_distance_field_estimate(
        self, start_x: int, start_z: int
    ) -> tuple[np.ndarray | None, dict[tuple[int, int], tuple[int, int] | None]]:
        if not self.is_walkable_static(start_x, start_z):
            return None, {}

        distances = np.full((MAP_SIZE, MAP_SIZE), -1, dtype=np.int16)
        parents: dict[tuple[int, int], tuple[int, int] | None] = {(start_x, start_z): None}
        queue = deque([(start_x, start_z)])
        distances[start_z, start_x] = 0

        while queue:
            x, z = queue.popleft()
            for dx, dz in MOVE_DIR_VEC:
                nx, nz = x + dx, z + dz
                if not self.can_step_static(x, z, nx, nz):
                    continue
                if distances[nz, nx] != -1:
                    continue
                distances[nz, nx] = distances[z, x] + 1
                parents[(nx, nz)] = (x, z)
                queue.append((nx, nz))
        return distances, parents

    def lookup_distance_estimate(self, distance_field: np.ndarray | None, x: int, z: int) -> int | None:
        if distance_field is None or x < 0 or x >= MAP_SIZE or z < 0 or z >= MAP_SIZE:
            return None
        dist = int(distance_field[z, x])
        return None if dist < 0 else dist

    def reconstruct_path_estimate(
        self,
        parents: dict[tuple[int, int], tuple[int, int] | None],
        target: tuple[int, int],
    ) -> list[tuple[int, int]]:
        if target not in parents:
            return []
        path: list[tuple[int, int]] = []
        current: tuple[int, int] | None = target
        while current is not None:
            path.append(current)
            current = parents.get(current)
        path.reverse()
        return path

    def compute_path_tail_direction_estimate(
        self,
        path_estimate: list[tuple[int, int]],
        tail_steps: int = 3,
    ) -> tuple[int, int]:
        if len(path_estimate) < 2:
            return (0, 0)
        anchor_idx = max(0, len(path_estimate) - 1 - tail_steps)
        anchor_x, anchor_z = path_estimate[anchor_idx]
        target_x, target_z = path_estimate[-1]
        dx = int(np.sign(target_x - anchor_x))
        dz = int(np.sign(target_z - anchor_z))
        return (dx, dz)

    def compute_approach_cosine_estimate(
        self,
        direction1: tuple[int, int],
        direction2: tuple[int, int],
    ) -> float | None:
        v1 = np.asarray(direction1, dtype=np.float32)
        v2 = np.asarray(direction2, dtype=np.float32)
        norm = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if norm <= 1e-6:
            return None
        return float(np.clip(np.dot(v1, v2) / norm, -1.0, 1.0))

    def count_safe_directions_path_estimate(self, hero: Hero, monster_entries: list[dict]) -> int:
        safe_count = 0
        for dx, dz in MOVE_DIR_VEC:
            nx, nz = hero.x + dx, hero.z + dz
            if not self.can_step_static(hero.x, hero.z, nx, nz):
                continue

            is_safe = True
            for entry in monster_entries:
                distance_estimate = self.lookup_distance_estimate(entry["distance_field"], nx, nz)
                if distance_estimate is not None and distance_estimate <= 1:
                    is_safe = False
                    break
            if is_safe:
                safe_count += 1
        return safe_count

    def count_safe_directions(self, monsters: list[Monster]) -> int:
        raw = self.current.raw
        if raw is None:
            return 0

        map_view = raw.map_view
        cx, cz = VIEW_CENTER, VIEW_CENTER
        monster_positions = {(m.x, m.z) for m in monsters}

        def is_walkable(nx: int, nz: int) -> bool:
            if nx < 0 or nx >= VIEW_SIZE or nz < 0 or nz >= VIEW_SIZE:
                return False
            return bool(map_view[nz, nx] == 1)

        safe_count = 0
        for dx, dz in MOVE_DIR_VEC:
            nx_local = cx + dx
            nz_local = cz + dz
            if not is_walkable(nx_local, nz_local):
                continue

            nx_global = raw.hero.x + dx
            nz_global = raw.hero.z + dz
            if all(chebyshev_distance(nx_global, nz_global, mx, mz) > 1 for mx, mz in monster_positions):
                safe_count += 1
        return safe_count


    def is_abnormal_truncated(self) -> bool:
        if not self.truncated:
            return False
        if self.current_extra is not None and self.current_extra.result_code != 0:
            return True
        raw = self.current.raw
        if raw is None:
            return False
        return raw.step < raw.max_step

    def update_episode_stats(self) -> None:
        raw = self.current.raw
        if raw is None:
            return

        stats = self.episode_stats
        global_summary = self.current.global_summary
        reward_delta = self.current.action_last.reward_delta
        flash_escape_improved_estimate = (
            reward_delta.flash_count_delta > 0
            and (
                (global_summary.nearest_monster_path_distance_delta_estimate or 0) > 0
                or (global_summary.capture_margin_path_delta_estimate or 0) > 0
                or global_summary.safe_direction_path_count_delta_estimate > 0
            )
        )
        step_delta = raw.step - (self.previous.raw.step if self.previous.raw is not None else 0)
        step_delta = max(step_delta, 0)

        stats.map_id = self.map_id
        stats.result_code = self.current_extra.result_code if self.current_extra is not None else 0
        stats.episode_steps = raw.step
        stats.final_stage = self.current.stage_info.stage
        stats.final_total_score = raw.total_score
        stats.final_step_score = raw.step_score
        stats.final_treasure_score = raw.treasure_score
        stats.final_treasures = raw.treasures_collected
        stats.final_buffs = raw.collected_buff
        stats.final_flash_count = raw.flash_count
        stats.final_nearest_monster_dist_est  = int(global_summary.nearest_monster_path_distance_estimate or 0)
        stats.final_capture_margin_path_estimate = int(global_summary.capture_margin_path_estimate or 0)
        stats.final_encirclement_path_cosine_estimate = float(global_summary.encirclement_path_cosine_estimate or 0.0)
        stats.final_safe_direction_path_count_estimate = int(global_summary.safe_direction_path_count_estimate)
        stats.final_visible_treasure_ratio = len(raw.treasures) / max(len(raw.treasure_id), 1)
        stats.speedup_reached = stats.speedup_reached or self.current.stage_info.is_speed_boost_stage

        if step_delta > 0:
            if self.current.stage_info.stage == 1:
                stats.stage1_steps += step_delta
            elif self.current.stage_info.stage == 2:
                stats.stage2_steps += step_delta
            else:
                stats.stage3_steps += step_delta

            if self.current.stage_info.is_speed_boost_stage:
                stats.post_steps += step_delta
            else:
                stats.pre_steps += step_delta

            if global_summary.nearest_monster_path_distance_estimate is not None:
                stats.nearest_monster_path_distance_estimate_sum += global_summary.nearest_monster_path_distance_estimate
                stats.path_signal_steps += 1
            if global_summary.capture_margin_path_estimate is not None:
                stats.capture_margin_path_estimate_sum += global_summary.capture_margin_path_estimate
            if global_summary.encirclement_path_cosine_estimate is not None:
                stats.encirclement_path_cosine_estimate_sum += global_summary.encirclement_path_cosine_estimate
            stats.safe_direction_path_count_estimate_sum += global_summary.safe_direction_path_count_estimate
            if flash_escape_improved_estimate:
                stats.flash_escape_success_count += 1

        if self.terminated or self.truncated:
            stats.terminated = self.terminated
            stats.truncated = self.truncated
            stats.abnormal_truncated = self.is_abnormal_truncated()
            stats.completed = self.truncated and not stats.abnormal_truncated
            stats.post_terminated = self.terminated and self.current.stage_info.is_speed_boost_stage
            stats.last_flash_used = reward_delta.flash_count_delta > 0
            stats.last_flash_escape_improved_estimate = flash_escape_improved_estimate
            if self.previous.raw is not None:
                stats.last_flash_ready = self.previous.raw.hero.can_flash
                stats.last_flash_legal_ratio = (
                    sum(self.previous.action_predict.flash_valid_mask) / 8.0
                    if self.previous.action_predict.flash_valid_mask
                    else 0.0
                )

    def get_nearest_monsters(self) -> tuple[Monster | None, Monster | None]:
        raw = self.current.raw
        if raw is None:
            return None, None

        hero = raw.hero
        monsters = sorted(
            raw.monsters,
            key=lambda m: chebyshev_distance(hero.x, hero.z, m.x, m.z),
        )
        nearest = monsters[0] if len(monsters) >= 1 else None
        second = monsters[1] if len(monsters) >= 2 else None
        return nearest, second

    def get_nearest_known_treasure(self) -> Organ | None:
        raw = self.current.raw
        if raw is None:
            return None
        treasures = self.get_known_treasures(only_available=True)
        if not treasures:
            return None
        hero = raw.hero

        best_treasure: Organ | None = None
        best_path_distance: int | None = None
        best_l2_distance: float | None = None
        for treasure in treasures:
            path_distance = self.estimate_path_distance_on_known_map(hero.x, hero.z, treasure.x, treasure.z)
            l2_distance = distance_l2(hero.x, hero.z, treasure.x, treasure.z)
            if path_distance is not None:
                if best_path_distance is None or path_distance < best_path_distance or (
                    path_distance == best_path_distance and (best_l2_distance is None or l2_distance < best_l2_distance)
                ):
                    best_treasure = treasure
                    best_path_distance = path_distance
                    best_l2_distance = l2_distance
            elif best_path_distance is None and (best_l2_distance is None or l2_distance < best_l2_distance):
                best_treasure = treasure
                best_l2_distance = l2_distance
        return best_treasure

    def get_nearest_known_buff(self) -> Organ | None:
        raw = self.current.raw
        if raw is None:
            return None
        buffs = self.get_known_buffs(only_available=True)
        if not buffs:
            return None
        hero = raw.hero

        best_buff: Organ | None = None
        best_path_distance: int | None = None
        best_l2_distance: float | None = None
        for buff in buffs:
            path_distance = self.estimate_path_distance_on_known_map(hero.x, hero.z, buff.x, buff.z)
            l2_distance = distance_l2(hero.x, hero.z, buff.x, buff.z)
            if path_distance is not None:
                if best_path_distance is None or path_distance < best_path_distance or (
                    path_distance == best_path_distance and (best_l2_distance is None or l2_distance < best_l2_distance)
                ):
                    best_buff = buff
                    best_path_distance = path_distance
                    best_l2_distance = l2_distance
            elif best_path_distance is None and (best_l2_distance is None or l2_distance < best_l2_distance):
                best_buff = buff
                best_l2_distance = l2_distance
        return best_buff

    def get_known_treasures(self, only_available: bool = True) -> list[Organ]:
        treasures = [t for t in self.treasure_full.values() if t is not None]
        if only_available:
            treasures = [t for t in treasures if t.status == 1]
        return treasures

    def get_known_buffs(self, only_available: bool = False) -> list[Organ]:
        buffs = [b for b in self.buff_full.values() if b is not None]
        if only_available:
            buffs = [b for b in buffs if b.status == 1 and b.cooldown == 0]
        return buffs

    def build_global_treasure_available_map(self) -> np.ndarray:
        layer = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.int8)
        for treasure in self.get_known_treasures(only_available=True):
            if 0 <= treasure.x < MAP_SIZE and 0 <= treasure.z < MAP_SIZE:
                layer[treasure.z, treasure.x] = 1
        return layer

    def build_global_buff_known_map(self) -> np.ndarray:
        layer = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
        raw = self.current.raw
        cooldown_max = max(int(raw.buff_refresh_time), 1) if raw is not None else 1
        for buff in self.get_known_buffs(only_available=False):
            if 0 <= buff.x < MAP_SIZE and 0 <= buff.z < MAP_SIZE:
                cooldown_ratio = float(np.clip(buff.cooldown / cooldown_max, 0.0, 1.0))
                layer[buff.z, buff.x] = max(0.1, cooldown_ratio)
        return layer

    def get_move_valid_mask(self) -> list[bool]:
        """返回普通移动 8 方向的有效性掩码"""
        raw = self.current.raw
        if raw is None:
            return [False] * 8

        map_view = raw.map_view
        cx, cz = VIEW_CENTER, VIEW_CENTER

        def is_walkable(nx: int, nz: int) -> bool:
            if nx < 0 or nx >= VIEW_SIZE or nz < 0 or nz >= VIEW_SIZE:
                return False
            return bool(map_view[nz, nx] == 1)

        mask: list[bool] = []
        for dx, dz in MOVE_DIR_VEC:
            nx, nz = cx + dx, cz + dz
            if dx != 0 and dz != 0:
                ok = is_walkable(nx, nz) and (
                    is_walkable(cx + dx, cz) or is_walkable(cx, cz + dz)
                )
                mask.append(ok)
            else:
                mask.append(is_walkable(nx, nz))
        return mask

    def get_flash_result(self) -> ActionPredict:
        """
        返回当前帧与闪现动作相关的结果集合。
        """
        raw = self.current.raw
        if raw is None:
            return ActionPredict(
                flash_pos=[(0, 0)] * 8,
                flash_pos_relative=[(0, 0)] * 8,
                flash_valid_mask=[False] * 8,
                flash_distance=[0.0] * 8,
                flash_across_wall=[False] * 8,
            )

        local_flash_pos = predict_flash_pos(raw.map_view, VIEW_CENTER, VIEW_CENTER)
        relative = flash_pos_relative(local_flash_pos, VIEW_CENTER, VIEW_CENTER)
        valid = flash_validation(relative)
        global_flash_pos = [(raw.hero.x + dx, raw.hero.z + dz) for dx, dz in relative]
        flash_distance = [float(max(abs(dx), abs(dz))) for dx, dz in relative]
        flash_across_wall: list[bool] = []

        for (dx, dz), distance in zip(relative, flash_distance):
            if distance <= 1.0 or (dx == 0 and dz == 0):
                flash_across_wall.append(False)
                continue

            step_dx = int(np.sign(dx))
            step_dz = int(np.sign(dz))
            crossed_wall = False
            for step in range(1, int(distance) + 1):
                nx = VIEW_CENTER + step_dx * step
                nz = VIEW_CENTER + step_dz * step
                if raw.map_view[nz, nx] == 0:
                    crossed_wall = True
                    break
            flash_across_wall.append(crossed_wall)

        return ActionPredict(
            flash_pos=global_flash_pos,
            flash_pos_relative=relative,
            flash_valid_mask=valid,
            flash_distance=flash_distance,
            flash_across_wall=flash_across_wall,
        )

    def get_stage_alpha(self) -> float:
        """
        返回当前阶段对应的 reward 混合权重 `alpha`。
        """
        return self.current.stage_info.alpha

    # ======================================== building

    def build_obs_state(self) -> dict:
        """构造供 obs 模块消费的统一结构化状态。

        消费方通过 summary 对象访问属性，不再展开到顶层。
        """
        raw = self.current.raw
        return {
            "raw": raw,
            'raw_previous': self.previous.raw,
            "hero": raw.hero if raw is not None else None,
            "hero_speed": self.current.hero_speed,
            "legal_action": raw.legal_action if raw is not None else None,
            # summaries (消费方通过 .attr 访问)
            "monster_summary": self.current.monster_summary,
            "resource_summary": self.current.resource_summary,
            "space_summary": self.current.space_summary,
            "stage_info": self.current.stage_info,
            "action_predict": self.current.action_predict,
            "action_last": self.current.action_last,
            # maps
            "local_map_layers": self.current.local_map_layers,
            "global_map_full": self.map_full,
            "global_visit_coverage": self.visit_coverage,
            "global_treasure_available_map": self.build_global_treasure_available_map(),
            "global_buff_known_map": self.build_global_buff_known_map(),
            # per-slot resource caches
            "treasure_full": self.treasure_full,
            "buff_full": self.buff_full,
        }

    def build_reward_state(self) -> dict:
        """构造供 reward 模块消费的统一结构化状态。

        消费方通过 summary 对象访问属性，不再展开到顶层。
        """
        raw = self.current.raw
        reward_delta = self.current.action_last.reward_delta
        global_summary = self.current.global_summary
        flash_escape_improved_estimate = (
            reward_delta.flash_count_delta > 0
            and (
                (global_summary.nearest_monster_path_distance_delta_estimate or 0) > 0
                or (global_summary.capture_margin_path_delta_estimate or 0) > 0
                or global_summary.safe_direction_path_count_delta_estimate > 0
            )
        )

        return {
            # ===== lifecycle
            "raw": raw,
            "extra": self.current_extra,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "abnormal_truncated": self.is_abnormal_truncated(),
            # summaries (消费方通过 .attr 访问)
            "action_last": self.current.action_last,
            "monster_summary": self.current.monster_summary,
            "resource_summary": self.current.resource_summary,
            "space_summary": self.current.space_summary,
            "stage_info": self.current.stage_info,
            "global_summary": global_summary,
            # derived scalars that don't live on any summary
            "flash_escape_improved_estimate": flash_escape_improved_estimate,
            # previous action_predict (上一步的闪现预测，而非本帧)
            "prev_flash_across_wall": self.previous.action_predict.flash_across_wall,
            "prev_flash_distance": self.previous.action_predict.flash_distance,
            # local visit pressure
            "hero_visit_count": int(self.visit_count[raw.hero.z, raw.hero.x]) if raw is not None else 0,
        }

    def build_monitor_metrics(self) -> dict[str, float]:
        """
        返回用于监控面板投送的 episode/step 指标。
        """
        return self.episode_stats.as_dict()

    def build_debug_state(self) -> dict:
        """
        构造供日志、调试、分析使用的辅助状态。
        """
        return {
            # ===== lifecycle
            "initialized": self.initialized,
            "terminated": self.terminated,
            "truncated": self.truncated,

            # ===== snapshots
            "current": self.current,
            "previous": self.previous,
            "current_extra": self.current_extra,
            "previous_extra": self.previous_extra,
            "current_raw": self.current.raw,
            "previous_raw": self.previous.raw,

            # ===== cache
            "map_id": self.map_id,
            "map_full": self.map_full,
            "visit_count": self.visit_count,
            "visit_coverage": self.visit_coverage,
            "treasure_available_map": self.build_global_treasure_available_map(),
            "buff_known_map": self.build_global_buff_known_map(),
            "treasure_full": self.treasure_full,
            "buff_full": self.buff_full,
            "pos_history": list(self.pos_history),

            # ===== current derived
            "hero_speed": self.current.hero_speed,
            "map_explore_rate": self.current.map_explore_rate,
            "map_new_discover": self.current.map_new_discover,
            "monster_summary": self.current.monster_summary,
            "resource_summary": self.current.resource_summary,
            "space_summary": self.current.space_summary,
            "stage_info": self.current.stage_info,
            "action_predict": self.current.action_predict,
            "action_last": self.current.action_last,
            "global_summary": self.current.global_summary,
            "local_map_layers": self.current.local_map_layers,
            "episode_stats": self.episode_stats,
            "monitor_metrics": self.build_monitor_metrics(),

            # ===== helper exports
            "obs_state": self.build_obs_state(),
            "reward_state": self.build_reward_state(),
        }