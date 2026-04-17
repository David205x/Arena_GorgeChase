# 0. extractor 当前实现

## 一、总体分析

### 1. 定位
当前 `extractor` 已不是简单的原始观测解包器，而是一个面向 `obs`、`reward`、monitor、debug 的统一状态中台：

- 下接环境原始 `obs`
- 中间维护跨步状态、地图缓存、资源缓存、轨迹缓存
- 上接结构化摘要与导出接口

它大体形成了三层结构：

1. 原始封装层：`RawObs` / `ExtraInfo` / `Hero` / `Monster` / `Organ`
2. 派生汇总层：各种 `Summary`、`ActionLast`、`ActionPredict`、`LocalMapLayers`
3. 消费接口层：`build_obs_state()` / `build_reward_state()` / `build_monitor_metrics()` / `build_debug_state()`

### 2. 实现意图
我对当前实现意图的理解是：

> 用一套统一、跨步维护的结构，把环境中与决策相关的“当前状态 + 历史状态 + 近似推断状态”整理出来，避免 `obs` 和 `reward` 重复实现底层逻辑。

它的设计重点不是“尽量少做事”，而是“把真正会重复用到的状态尽量在 extractor 里收敛掉”。

### 3. 与设计清单的对应
和 `ref/设计清单/0_extractor.md` 对照后，当前实现已经较完整覆盖：

- 原始观测封装层
- 跨步完整状态层
- 地图探索缓存
- 资源全局缓存
- 英雄动作反馈缓存
- 闪现落点 / 动作有效性预测
- 怪物压力基础摘要
- 阶段信息派生
- 对 `obs` / `reward` 的统一接口

比设计清单更进一步的部分主要是：

- `global_summary`：基于静态图和完整 `extra_info` 的路径级危险估计
- `episode_stats`：整局监控与终局分析指标
- profiling 支持

---

## 二、代码结构

## 1. `constant.py`
定义了 extractor 的基础规则参数：

- 地图与视野：`MAP_SIZE=128`、`VIEW_SIZE=21`、`VIEW_CENTER=10`
- 闪现：`FLASH_DISTANCE=10`、`FLASH_DISTANCE_DIAGONAL=8`
- 历史轨迹：`POS_HISTORY_LEN=10`
- 方向：`FLASH_DIR_VEC`、`MOVE_DIR_VEC`
- 阶段权重：`ALPHA_MAP={1:0.5, 2:0.6, 3:0.90}`

### 印象
- 常量层很克制
- 方向顺序统一，便于动作掩码、闪现预测、空间统计复用
- `ALPHA_MAP` 说明 extractor 已直接服务 reward 阶段权重

## 2. `utils.py`
主要提供三类基础函数：

### 距离 / 邻接
- `chebyshev_distance()`
- `distance_l2()`
- `is_pos_neighbor()`

### 坐标 / 窗口
- `clamp_map_coord()`
- `build_local_window()`：从 `128×128` 全局图裁出以英雄为中心的 `21×21` 窗口

### 闪现规则复刻
- `predict_flash_pos()`
- `flash_pos_relative()`
- `flash_validation()`

其中 `predict_flash_pos()` 基本复刻了环境真实闪现逻辑：直线 10、斜向 8，从最远点往回找可通行点，否则原地。

### 印象
`utils.py` 内容不多，但都很关键，尤其是闪现预测，是当前 extractor 动作侧建模的关键底层支撑。

## 3. `dataclass.py`
这层是当前实现最成熟、最清晰的部分。

### 3.1 基础实体类
- `Character`：通用实体基类，封装 `id/x/z/hero_l2_distance/hero_relative_direction`
- `Hero`：增加 `buff_remaining_time`、`flash_cooldown`、`can_flash`
- `Monster`：增加 `monster_interval`、`speed`、`is_in_view`
- `Organ`：增加 `status`、`sub_type`、`cooldown`，并区分 `is_treasure` / `is_buff`

### 3.2 原始观测层
- `RawObs`：封装当前步原始观测主数据
- `ExtraInfo`：封装全局完整帧、`map_id`、`result_code` 等额外信息

`RawObs` 包含：
- 当前步、`legal_action`、`map_view`
- 英雄、怪物、宝箱、buff
- `treasure_id`
- 各类 score / count
- 环境配置字段：`buff_refresh_time`、`monster_interval`、`monster_speed_boost_step` 等

### 3.3 跨步和摘要层
- `RewardDelta`：当前与上一帧的分数 / 资源 / flash 增量
- `ActionPredict`：动作预测结果
- `ActionLast`：上一动作在当前帧体现出的结果反馈
- `MonsterSummary`：怪物距离、方向、速度、相对关系摘要
- `ResourceSummary`：最近已知宝箱 / buff 摘要
- `SpaceSummary`：局部地形与活动空间摘要
- `GlobalSummary`：基于静态图和 extra 的全局危险估计
- `StageInfo`：阶段与阶段权重
- `LocalMapLayers`：局部多通道地图
- `EpisodeStats`：整局监控统计
- `ExtractorSnapshot`：某一帧的完整 extractor 快照

### 印象
- `Snapshot + Summary` 组织方式非常适合上层消费
- dataclass 数量较多，但并没有失控，反而有效收拢了复杂度

---

## 三、Extractor 主流程

## 1. 生命周期
### `__init__`
初始化：
- `current` / `previous`
- `current_extra` / `previous_extra`
- 地图缓存：`map_full`、`map_static`、`visit_count`、`visit_coverage`
- 资源缓存：`treasure_full`、`buff_full`
- 轨迹缓存：`pos_history`
- 生命周期标记：`initialized`、`terminated`、`truncated`、`_last_action`
- `episode_stats`
- profiler

### `reset()`
每局开始前彻底清空所有缓存和统计。

### 印象
- 生命周期管理完整
- reset 非常彻底
- profiler 说明实现者明确关心 extractor 性能

## 2. `update()` 单步主流程
`update()` 是 extractor 核心入口，数据流非常明确：

### 2.1 解析当前输入
1. 保存 `terminated` / `truncated` / `last_action`
2. `raw = RawObs.from_env(env_obs)`
3. `previous <- current`
4. `current_extra = ExtraInfo.from_env(extra_info)`
5. `current = ExtractorSnapshot(raw=raw, extra=current_extra)`
6. 若有 `map_id`，加载静态地图

### 2.2 更新缓存
1. `init_resource_cache(raw)`
2. `update_map_full(raw)`
3. `update_visit_count(raw)`
4. `update_visit_coverage(raw)`
5. `update_pos_history()`
6. `update_treasure_cache(raw)`
7. `update_buff_cache(raw)`

### 2.3 计算派生信息
1. `compute_hero_speed()`
2. `compute_map_statistics()`
3. `compute_monster_summary()`
4. `compute_resource_summary()`
5. `compute_stage_info()`
6. `compute_action_predict()`
7. `compute_local_map_layers()`
8. `compute_action_last()`
9. `compute_space_summary()`
10. `compute_global_summary()`
11. `update_episode_stats()`

最后返回 `self.current`。

### 数据流印象
可以概括为：

**原始 obs → 缓存更新 → 派生摘要 → 统一导出**

而且顺序有明确依赖意识，例如：
- 先算 `action_predict`，因为 `local_map_layers.flash_landing` 依赖它
- 先有局部摘要，再做更重的 `global_summary`

---

## 四、缓存系统

## 1. 地图缓存
### `map_full`
`update_map_full()` 把当前 `21×21` 视野写回全局 `128×128` 地图：
- `-1=未探索`
- `0=已知不可通行`
- `1=已知可通行`

用途：
- 地图探索率统计
- 已知图路径估计
- 资源距离估计

### `map_static`
通过 `_load_all_static_maps()` 与 `map_id` 加载真值通行图，仅供 reward / monitor 估计使用。

### `visit_count`
只对英雄当前位置格子加一。

### `visit_coverage`
当前位置 `+1.0`，其周围可通行 8 邻格 `+0.5`。

### 印象
- `visit_coverage` 比单纯 visit count 更适合表示“区域重复逗留”
- `map_static` 让 extractor 不再只是“局部观测整理器”，而兼具分析支撑角色

## 2. 资源缓存
### `treasure_full`
`dict[int, Organ | None]`，记录已发现宝箱；若某已知宝箱不再出现在 `raw.treasure_id` 中，则记为 `status=0`。

它维护了三种隐含状态：
- 未发现
- 已发现未收集
- 已发现已收集

### `buff_full`
记录已知 buff 位置及 extractor 自己推断的 `cooldown`。

推断逻辑：
- 当前帧可见 buff 直接写入缓存
- 若英雄与 buff 邻接，且 `buff_remaining_time == 49`，则认为刚吃到 buff，设 `cooldown = raw.buff_refresh_time`
- 此后每步 `cooldown -= 1`

### 印象记录
1. 宝箱缓存思路非常清楚，主要服务“最近已知宝箱”和资源进度统计。
2. buff cooldown 是 extractor 自己补出来的状态，因为环境原始观测不直接给全局刷新倒计时。
3. `buff_remaining_time == 49` 带有对 buff 持续时长默认值的隐含依赖。

### 编号约定印象
当前实现初始化缓存时带有明显假设：

- `treasure_full = {1..total_treasure}`
- `buff_full = {total_treasure .. total_treasure + total_buff - 1}`

这意味着当前代码更像默认“宝箱编号从 1 开始”。而你的备注里提到 treasure `config_id` 是从 `0` 开始连续编号，所以这里我记录为：

> 当前实现在资源编号约定上带有“treasure 从 1 开始”的实现假设，需要和真实环境继续核对。

---

## 五、派生信息

## 1. 英雄与探索
### `compute_hero_speed()`
- `buff_remaining_time > 0` → `2`
- 否则 `1`

### `compute_map_statistics()`
计算：
- `map_explore_rate`
- `map_new_discover`

### 印象
这部分是比赛默认规则驱动的工程实现，而不是做成完全配置化。

## 2. 变化量与上一动作反馈
### `compute_reward_delta()`
计算当前帧相对上一帧的 score / count 增量。

### `compute_action_last()`
总结上一动作结果：
- 是否发生有效移动
- 位移量
- 最近怪距离是否增加
- 是否捡到宝箱 / buff
- 是否探索到新区
- 地图探索率变化
- reward delta

### 印象
`action_last` 是非常重要的桥梁层，它把 reward / monitor 常用的“上一动作后果”显式化了。

## 3. 动作预测
### `get_move_valid_mask()`
实现普通移动 8 方向有效性判断：
- 直走：目标格可通行
- 斜走：目标格可通行，且横 / 竖侧边至少一边可通行

### `get_flash_result()`
计算：
- `flash_pos`
- `flash_pos_relative`
- `flash_valid_mask`
- `flash_distance`
- `flash_across_wall`

### `compute_action_predict()`
把移动掩码与闪现结果汇总成 `ActionPredict`。

其中 `action_preferred = move_valid_mask + flash_valid_mask`，当前仍是占位式定义，注释里也标了 `#TODO`。

### 印象
- 这是当前 extractor 很强的一块
- 它解决了环境 `legal_action` 对移动 / 闪现质量信息不足的问题
- `flash_across_wall` 是很有用的额外信号

## 4. 怪物摘要 `compute_monster_summary()`
主要输出：
- 最近怪 / 第二怪
- 各类距离（切比雪夫、L2、bucket）
- 最近怪距离变化
- 第一怪 / 第二怪相对位置与方向
- 速度
- 哪只是最近怪
- 双怪方向余弦
- 第二怪出现倒计时

### 印象记录
当前实现直接对 `raw.monsters` 排序，没有主动按 `is_in_view` 做保护，说明它更信任输入数据本身，而不是在 extractor 内二次防御。

## 5. 资源摘要 `compute_resource_summary()`
逻辑比单纯最近欧氏距离更进一步：

### 候选筛选
- 从缓存中取所有已知可用宝箱 / buff
- 按平方欧氏距离排序
- 各取前 3 个

### 距离估计
- 距离较近（`chebyshev <= 30`）时，优先在 `map_full` 上做 BFS
- 否则退化为：切比雪夫距离 + `_line_obstruction_penalty()`

### 输出
- 最近已知宝箱 / buff
- path 距离 / L2 距离 / 方向
- 距离变化
- 已发现数量
- treasure / buff progress

### 印象
这是“已知世界可达性近似”的实现，而不是只看几何距离，工程上很实用。

## 6. 空间摘要 `compute_space_summary()`
当前空间摘要属于基础版：
- 八方向通路长度
- 局部可通行格数
- 开阔度
- 安全方向数
- 可活动空间变化
- `is_dead_end`
- `is_corridor`
- `is_low_openness`

### 印象
能支撑基础生存 shaping，但还没有特别深入地表达“未来逃逸空间”。

## 7. 阶段摘要 `compute_stage_info()`
阶段定义：
- `stage=1`：单怪阶段
- `stage=2`：双怪阶段
- `stage=3`：怪物加速阶段

同时输出：
- `has_second_monster`
- `is_speed_boost_stage`
- `steps_to_next_stage`
- `alpha`

### 印象
这是设计清单里“阶段与节奏信息派生”的直接落地，语义很统一。

## 8. 局部地图通道 `compute_local_map_layers()`
维护 8 个通道：
- `obstacle`
- `hero`
- `monster`
- `treasure`
- `buff`
- `visit`
- `visit_coverage`
- `flash_landing`

但 `LocalMapLayers.as_stack()` 当前只 stack 了 7 个通道，没有把 `visit_coverage` 放进去。

### 印象记录
> 当前代码里实际维护了 8 个局部图层，但默认堆叠导出只使用了 7 个，`visit_coverage` 暂未进入默认 stack。

## 9. 全局危险摘要 `compute_global_summary()`
这是当前 extractor 最偏 reward / monitor 的高阶部分。

### 数据来源
优先使用：
- `current_extra.hero`
- `current_extra.monsters`

### 核心思路
在静态真值地图上：
- 从英雄做一次 BFS
- 查各怪物到英雄的路径距离估计
- 回溯怪物接近方向
- 估计最近怪、第二怪、包夹余弦、抓捕裕度、安全方向数

### 代表性输出
- `nearest_monster_path_distance_estimate`
- `capture_margin_path_estimate`
- `encirclement_path_cosine_estimate`
- `safe_direction_path_count_estimate`
- `dead_end_under_pressure_estimate`

### 印象
这是当前 extractor 最有价值、也最“侵略性”的部分：已经不只是整理信息，而是在做 reward-side 的局势估计中台。

---

## 六、路径与辅助方法
当前 extractor 还维护了不少已知图 / 静态图辅助方法：

### 已知图
- `estimate_path_distance_on_known_map()`
- `can_step_known_map()`
- `_bfs_from_hero_known()`

### 静态图
- `is_walkable_static()`
- `can_step_static()`
- `_bfs_from_hero_static()`
- `_lookup_static_dist()`
- `_trace_approach_direction()`
- `_count_safe_dirs_adjacent()`

### 更通用的旧式路径工具
- `build_distance_field_estimate()`
- `lookup_distance_estimate()`
- `reconstruct_path_estimate()`
- `compute_path_tail_direction_estimate()`
- `compute_approach_cosine_estimate()`
- `count_safe_directions_path_estimate()`

### 印象
可以看出实现有明显演化痕迹：
- 先有通用版 distance field / parents 方法
- 后来又加入从英雄单次 BFS 的优化实现

这说明 extractor 是边实验边收敛出来的，而不是一次性设计成型的。

---

## 七、整局统计 `update_episode_stats()`
这一部分把 step 级状态累计为 episode 级监控指标：

- 地图 / 结果码
- 总步数
- 各阶段步数
- `pre_steps` / `post_steps`
- 是否进入怪物加速阶段
- 最终总分 / 宝箱 / buff / flash 次数
- 终局危险指标估计
- 最终可见宝箱比例
- 最后一步闪现相关信息
- 路径危险信号均值累计
- 闪现脱险成功累计

### 印象
这部分很明显是把“训练监控要看什么”直接落到了 extractor 中，和设计清单相比更偏实验分析中台。

---

## 八、对外接口

## 1. `build_obs_state()`
返回给 obs 模块的统一结构，包括：
- `raw` / `raw_previous`
- `hero` / `hero_speed`
- `legal_action`
- `monster_summary`
- `resource_summary`
- `space_summary`
- `stage_info`
- `action_predict`
- `action_last`
- `local_map_layers`
- 全局缓存图
- `treasure_full` / `buff_full`

## 2. `build_reward_state()`
返回给 reward 模块的统一结构，包括：
- 生命周期信息
- `action_last`
- `monster_summary`
- `resource_summary`
- `space_summary`
- `stage_info`
- `global_summary`
- `flash_escape_improved_estimate`
- `prev_flash_across_wall`
- `prev_flash_distance`
- `hero_visit_count`

## 3. `build_monitor_metrics()`
导出 `episode_stats.as_dict()`。

## 4. `build_debug_state()`
导出几乎所有内部状态，供日志与分析使用。

### 印象
对外接口分层非常清楚，符合“统一语义源”的设计目标。

---

## 九、数据流总结
当前 extractor 的数据流可以概括为：

1. 输入层：`env_obs` / `extra_info` / `terminated` / `truncated` / `last_action`
2. 原始封装层：`RawObs` / `ExtraInfo`
3. 状态滚动层：`previous <- current`，`current <- snapshot`
4. 缓存更新层：地图、访问、轨迹、资源缓存
5. 派生汇总层：动作、怪物、资源、空间、阶段、局部图、全局危险
6. 监控累计层：`episode_stats`
7. 导出层：`obs_state` / `reward_state` / `monitor_metrics` / `debug_state`

### 总体印象
整体数据流清晰，已经形成了一个相当稳定的中台式结构。

---

## 十、总体印象记录

### 1. 明显面向当前赛题
这不是一个完全泛化的 extractor，而是围绕 Gorge Chase 当前任务强定制的实现：
- 闪现规则直接写死
- 阶段定义直接围绕单怪 / 双怪 / 加速
- 速度按当前比赛默认值处理

### 2. reward-side 支持很强
尤其是：
- `global_summary`
- `episode_stats`
- `flash_escape_improved_estimate`

都说明 extractor 已不只是 obs 的底层支撑，而是 reward / monitor 的核心数据源。

### 3. 有“先实用，再优化”的工程痕迹
能看到 heuristic 与 BFS 优化并存，说明它是在实验推动下逐步长出来的。

### 4. 复杂度不低，但还没失控
复杂性大多被 dataclass 和 helper 函数收住了，没有完全塌成混乱大函数。

---

## 十一、一句话总结
当前 `extractor` 已实现为一个 **跨步状态管理 + 特征中间层 + reward 分析支撑 + monitor 指标生成** 的综合模块。

它真正的核心价值不在“解包 obs”，而在：

> 把环境里零散、跨步、局部可见、需要近似推断的信息，统一整理成稳定、可消费、可分析的结构化状态表达。
