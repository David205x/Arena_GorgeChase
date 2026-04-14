from __future__ import annotations
import numpy as np
from collections import deque
from typing import Deque

from .dataclass import *
from .constant import *
from .utils import *


class Extractor:
    def __init__(self):
        # ========== current / previous
        self.current: ExtractorSnapshot = ExtractorSnapshot()
        self.current_extra: ExtractorSnapshot = ExtractorSnapshot()
        self.previous: ExtractorSnapshot = ExtractorSnapshot()
        # ========== map cache
        self.map_id: int = -1
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
        """在每局开始前调用"""
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
        """单步更新 extractor 主状态"""
        
        self.terminated = terminated
        self.truncated = truncated
        
        raw = RawObs.from_env(env_obs)
        # ===== update step
        self.previous = self.current
        self.current = ExtractorSnapshot(raw=raw)

        # 2) 更新各类缓存
        self.init_resource_cache(raw)
        self.update_map_full(raw)
        self.update_visit_count(raw)
        self.update_pos_history()
        self.update_treasure_cache(raw)
        self.update_buff_cache(raw)
        # 3) 计算基础派生信息
        self.current.hero_speed = self.compute_hero_speed()
        self.current.map_explore_rate, self.current.map_new_discover = self.compute_map_statistics()
        # 4) 先计算不依赖 action_feedback 的摘要
        self.current.monster_summary = self.compute_monster_summary()
        self.current.resource_summary = self.compute_resource_summary()
        self.current.stage_info = self.compute_stage_info()
        self.current.local_map_layers = self.compute_local_map_layers()
        # 5) 动作结果 / 动作反馈
        self.current.action_result = self.compute_action_result()
        self.current.action_feedback = self.compute_action_feedback()
        # 6) 空间信息（若当前实现已完成）
        self.current.space_summary = self.compute_space_summary()
        self.initialized = True
        return self.current

    # ======================================== cache update
    def init_resource_cache(self, raw: RawObs) -> None:
        """按当前环境配置初始化资源缓存容器"""
        if not self.treasure_full:
            self.treasure_full = [None] * raw.total_treasure
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

    def update_pos_history(self) -> None:
        if self.previous.raw is None:
            return
        hero = self.previous.raw.hero
        self.pos_history.append((hero.x, hero.z))

    def update_treasure_cache(self, raw: RawObs) -> None:
        """更新已知宝箱的全局缓存状态"""
        for treasure in raw.treasures:
            self.treasure_full[treasure.id] = treasure

        for treasure in self.treasure_full:
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

    def compute_action_feedback(self) -> ActionFeedback:
        """
        计算上一动作在当前帧体现出的结果反馈。
        """
        current = self.current.raw
        previous = self.previous.raw
        reward_delta = self.compute_reward_delta()

        if current is None or previous is None:
            return ActionFeedback(reward_delta=reward_delta)

        moved = (current.hero.x != previous.hero.x) or (current.hero.z != previous.hero.z)

        nearest_current = self.current.monster_summary.nearest_monster_distance
        nearest_last = self.previous.monster_summary.nearest_monster_distance
        nearest_monster_distance_increased = False
        if nearest_current is not None and nearest_last is not None:
            nearest_monster_distance_increased = nearest_current > nearest_last

        picked_treasure = reward_delta.treasures_collected_delta > 0
        picked_buff = reward_delta.collected_buff_delta > 0

        return ActionFeedback(
            moved=moved,
            moved_effectively=moved,    # TODO
            nearest_monster_distance_increased=nearest_monster_distance_increased,
            picked_treasure=picked_treasure,
            picked_buff=picked_buff,
            gained_resource=(picked_treasure or picked_buff),
            explored_new_area=self.current.map_new_discover > 0,
            reward_delta=reward_delta,
        )

    def compute_action_result(self) -> ActionResult:
        """计算当前帧面向下一步决策的动作结果摘要。"""
        move_valid_mask = self.get_move_valid_mask()
        flash_result = self.get_flash_result()

        return ActionResult(
            move_valid_mask=move_valid_mask,
            flash_pos=flash_result.flash_pos,
            flash_pos_relative=flash_result.flash_pos_relative,
            flash_valid_mask=flash_result.flash_valid_mask,
            flash_distance=flash_result.flash_distance,
            action_preferred=move_valid_mask + flash_result.flash_valid_mask, #TODO
        )

    def compute_monster_summary(self) -> MonsterSummary:
        nearest_monster, second_monster = self.get_nearest_monsters()
        nearest_distance = None
        second_distance = None
        average_distance = None
        hero = self.current.raw.hero if self.current.raw is not None else None

        if hero is not None and nearest_monster is not None:
            nearest_distance = chebyshev_distance(hero.x, hero.z, nearest_monster.x, nearest_monster.z)
        if hero is not None and second_monster is not None:
            second_distance = chebyshev_distance(hero.x, hero.z, second_monster.x, second_monster.z)

        distances = [d for d in [nearest_distance, second_distance] if d is not None]
        if distances:
            average_distance = float(sum(distances)) / len(distances)

        last_summary = self.previous.monster_summary
        nearest_distance_last = last_summary.nearest_monster_distance
        nearest_distance_delta = None
        if nearest_distance is not None and nearest_distance_last is not None:
            nearest_distance_delta = nearest_distance - nearest_distance_last

        monster_count = len(self.current.raw.monsters) if self.current.raw is not None else 0
        return MonsterSummary(
            monster_count=monster_count,
            nearest_monster=nearest_monster,
            second_monster=second_monster,
            nearest_monster_distance=nearest_distance,
            second_monster_distance=second_distance,
            nearest_monster_distance_last=nearest_distance_last,
            nearest_monster_distance_delta=nearest_distance_delta,
            average_monster_distance=average_distance,
        )

    def compute_resource_summary(self) -> ResourceSummary:
        raw = self.current.raw
        if raw is None:
            return ResourceSummary()

        nearest_treasure = self.get_nearest_known_treasure()
        nearest_buff = self.get_nearest_known_buff()

        nearest_treasure_distance = None
        if nearest_treasure is not None:
            nearest_treasure_distance = distance_l2(raw.hero.x, raw.hero.z, nearest_treasure.x, nearest_treasure.z)

        nearest_buff_distance = None
        if nearest_buff is not None:
            nearest_buff_distance = distance_l2(raw.hero.x, raw.hero.z, nearest_buff.x, nearest_buff.z)

        treasure_discovered_count = len(self.get_known_treasures(only_available=False))
        buff_discovered_count = len(self.get_known_buffs(only_available=False))

        treasure_progress = 0.0
        if raw.total_treasure > 0:
            treasure_progress = raw.treasures_collected / raw.total_treasure

        buff_progress = 0.0
        if raw.total_buff > 0:
            buff_progress = raw.collected_buff / raw.total_buff

        return ResourceSummary(
            nearest_known_treasure=nearest_treasure,
            nearest_known_treasure_distance=nearest_treasure_distance,
            nearest_known_buff=nearest_buff,
            nearest_known_buff_distance=nearest_buff_distance,
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

        return LocalMapLayers(
            obstacle=obstacle,
            hero=hero,
            monster=monster,
            treasure=treasure,
            buff=buff,
            visit=visit,
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
        return min(treasures, key=lambda o: distance_l2(hero.x, hero.z, o.x, o.z))

    def get_nearest_known_buff(self) -> Organ | None:
        raw = self.current.raw
        if raw is None:
            return None
        buffs = self.get_known_buffs(only_available=True)
        if not buffs:
            return None
        hero = raw.hero
        return min(buffs, key=lambda o: distance_l2(hero.x, hero.z, o.x, o.z))

    def get_known_treasures(self, only_available: bool = True) -> list[Organ]:
        treasures = [t for t in self.treasure_full if t is not None]
        if only_available:
            treasures = [t for t in treasures if t.status == 1]
        return treasures

    def get_known_buffs(self, only_available: bool = False) -> list[Organ]:
        buffs = [b for b in self.buff_full.values() if b is not None]
        if only_available:
            buffs = [b for b in buffs if b.status == 1 and b.cooldown == 0]
        return buffs

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

    def get_flash_result(self) -> ActionResult:
        """
        返回当前帧与闪现动作相关的结果集合。
        """
        raw = self.current.raw
        if raw is None:
            return ActionResult(
                flash_pos=[(0, 0)] * 8,
                flash_pos_relative=[(0, 0)] * 8,
                flash_valid_mask=[False] * 8,
                flash_distance=[0.0] * 8,
            )

        local_flash_pos = predict_flash_pos(raw.map_view, VIEW_CENTER, VIEW_CENTER)
        relative = flash_pos_relative(local_flash_pos, VIEW_CENTER, VIEW_CENTER)
        valid = flash_validation(relative)
        global_flash_pos = [(raw.hero.x + dx, raw.hero.z + dz) for dx, dz in relative]
        flash_distance = [float(max(abs(dx), abs(dz))) for dx, dz in relative]

        return ActionResult(
            flash_pos=global_flash_pos,
            flash_pos_relative=relative,
            flash_valid_mask=valid,
            flash_distance=flash_distance,
        )

    def get_stage_alpha(self) -> float:
        """
        返回当前阶段对应的 reward 混合权重 `alpha`。
        """
        return self.current.stage_info.alpha

    # ======================================== building

    def build_obs_state(self) -> dict:
        """
        构造供 obs 模块消费的统一结构化状态。
        """
        raw = self.current.raw
        monster_summary = self.current.monster_summary
        resource_summary = self.current.resource_summary
        space_summary = self.current.space_summary
        stage_info = self.current.stage_info
        action_result = self.current.action_result
        action_feedback = self.current.action_feedback
        local_map_layers = self.current.local_map_layers

        return {
            # ===== raw / hero
            "raw": raw,
            "hero": raw.hero if raw is not None else None,
            "hero_speed": self.current.hero_speed,
            "legal_action": raw.legal_action if raw is not None else None,
            "step": raw.step if raw is not None else 0,

            # ===== map / explore
            "map_explore_rate": self.current.map_explore_rate,
            "map_new_discover": self.current.map_new_discover,
            "local_map_layers": local_map_layers,
            "local_map_stack": local_map_layers.as_stack(),

            # ===== monster
            "monster_summary": monster_summary,
            "nearest_monster": monster_summary.nearest_monster,
            "second_monster": monster_summary.second_monster,
            "nearest_monster_distance": monster_summary.nearest_monster_distance,
            "second_monster_distance": monster_summary.second_monster_distance,
            "nearest_monster_distance_last": monster_summary.nearest_monster_distance_last,
            "nearest_monster_distance_delta": monster_summary.nearest_monster_distance_delta,
            "average_monster_distance": monster_summary.average_monster_distance,
            "monster_count": monster_summary.monster_count,

            # ===== resource
            "resource_summary": resource_summary,
            "nearest_known_treasure": resource_summary.nearest_known_treasure,
            "nearest_known_treasure_distance": resource_summary.nearest_known_treasure_distance,
            "nearest_known_buff": resource_summary.nearest_known_buff,
            "nearest_known_buff_distance": resource_summary.nearest_known_buff_distance,
            "treasure_discovered_count": resource_summary.treasure_discovered_count,
            "buff_discovered_count": resource_summary.buff_discovered_count,
            "treasure_progress": resource_summary.treasure_progress,
            "buff_progress": resource_summary.buff_progress,

            # ===== space
            "space_summary": space_summary,
            "corridor_lengths": space_summary.corridor_lengths,
            "traversable_space": space_summary.traversable_space,
            "openness": space_summary.openness,
            "safe_direction_count": space_summary.safe_direction_count,
            "traversable_space_delta": space_summary.traversable_space_delta,
            "is_dead_end": space_summary.is_dead_end,
            "is_corridor": space_summary.is_corridor,
            "is_low_openness": space_summary.is_low_openness,

            # ===== stage
            "stage_info": stage_info,
            "stage": stage_info.stage,
            "has_second_monster": stage_info.has_second_monster,
            "is_speed_boost_stage": stage_info.is_speed_boost_stage,
            "steps_to_next_stage": stage_info.steps_to_next_stage,
            "alpha": stage_info.alpha,

            # ===== action
            "action_result": action_result,
            "move_valid_mask": action_result.move_valid_mask,
            "flash_pos": action_result.flash_pos,
            "flash_pos_relative": action_result.flash_pos_relative,
            "flash_valid_mask": action_result.flash_valid_mask,
            "flash_distance": action_result.flash_distance,
            "action_preferred": action_result.action_preferred,

            # ===== last action feedback
            "action_feedback": action_feedback,
            "moved": action_feedback.moved,
            "moved_effectively": action_feedback.moved_effectively,
            "nearest_monster_distance_increased": action_feedback.nearest_monster_distance_increased,
            "picked_treasure": action_feedback.picked_treasure,
            "picked_buff": action_feedback.picked_buff,
            "gained_resource": action_feedback.gained_resource,
            "explored_new_area": action_feedback.explored_new_area,
        }

    def build_reward_state(self) -> dict:
        """
        构造供 reward 模块消费的统一结构化状态。
        """
        raw = self.current.raw
        action_feedback = self.current.action_feedback
        reward_delta = action_feedback.reward_delta
        monster_summary = self.current.monster_summary
        resource_summary = self.current.resource_summary
        space_summary = self.current.space_summary
        stage_info = self.current.stage_info

        return {
            # ===== raw
            "raw": raw,
            "terminated": self.terminated,
            "truncated": self.truncated,

            # ===== reward delta
            "reward_delta": reward_delta,
            "total_score_delta": reward_delta.total_score_delta,
            "step_score_delta": reward_delta.step_score_delta,
            "treasure_score_delta": reward_delta.treasure_score_delta,
            "treasures_collected_delta": reward_delta.treasures_collected_delta,
            "collected_buff_delta": reward_delta.collected_buff_delta,
            "flash_count_delta": reward_delta.flash_count_delta,

            # ===== action feedback
            "action_feedback": action_feedback,
            "moved": action_feedback.moved,
            "moved_effectively": action_feedback.moved_effectively,
            "nearest_monster_distance_increased": action_feedback.nearest_monster_distance_increased,
            "picked_treasure": action_feedback.picked_treasure,
            "picked_buff": action_feedback.picked_buff,
            "gained_resource": action_feedback.gained_resource,
            "explored_new_area": action_feedback.explored_new_area,

            # ===== monster pressure
            "monster_summary": monster_summary,
            "monster_count": monster_summary.monster_count,
            "nearest_monster": monster_summary.nearest_monster,
            "second_monster": monster_summary.second_monster,
            "nearest_monster_distance": monster_summary.nearest_monster_distance,
            "second_monster_distance": monster_summary.second_monster_distance,
            "nearest_monster_distance_last": monster_summary.nearest_monster_distance_last,
            "nearest_monster_distance_delta": monster_summary.nearest_monster_distance_delta,
            "average_monster_distance": monster_summary.average_monster_distance,

            # ===== resource opportunity
            "resource_summary": resource_summary,
            "nearest_known_treasure": resource_summary.nearest_known_treasure,
            "nearest_known_treasure_distance": resource_summary.nearest_known_treasure_distance,
            "nearest_known_buff": resource_summary.nearest_known_buff,
            "nearest_known_buff_distance": resource_summary.nearest_known_buff_distance,
            "treasure_discovered_count": resource_summary.treasure_discovered_count,
            "buff_discovered_count": resource_summary.buff_discovered_count,
            "treasure_progress": resource_summary.treasure_progress,
            "buff_progress": resource_summary.buff_progress,

            # ===== space risk
            "space_summary": space_summary,
            "corridor_lengths": space_summary.corridor_lengths,
            "traversable_space": space_summary.traversable_space,
            "openness": space_summary.openness,
            "safe_direction_count": space_summary.safe_direction_count,
            "traversable_space_delta": space_summary.traversable_space_delta,
            "is_dead_end": space_summary.is_dead_end,
            "is_corridor": space_summary.is_corridor,
            "is_low_openness": space_summary.is_low_openness,

            # ===== stage
            "stage_info": stage_info,
            "stage": stage_info.stage,
            "has_second_monster": stage_info.has_second_monster,
            "is_speed_boost_stage": stage_info.is_speed_boost_stage,
            "steps_to_next_stage": stage_info.steps_to_next_stage,
            "alpha": stage_info.alpha,
        }

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
            "current_raw": self.current.raw,
            "previous_raw": self.previous.raw,

            # ===== cache
            "map_id": self.map_id,
            "map_full": self.map_full,
            "visit_count": self.visit_count,
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
            "action_result": self.current.action_result,
            "action_feedback": self.current.action_feedback,
            "local_map_layers": self.current.local_map_layers,

            # ===== helper exports
            "obs_state": self.build_obs_state(),
            "reward_state": self.build_reward_state(),
        }