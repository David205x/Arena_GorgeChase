# obs_extractor 预选派生/缓存字段清单

本文档只整理**候选项**，用于后续讨论整体任务思路时做取舍。

目标不是一次性全部实现，而是先把可能有价值的字段分层列出，后续根据：
- 训练效果
- 代码复杂度
- 信息增益
- 是否容易引入噪声

再决定保留哪些。

---

## 一、强相关且优先级较高的候选字段

这些字段和当前比赛任务“躲怪 + 吃宝箱 + 合理使用闪现”直接相关，通常优先考虑。

### 1. 闪现相关

#### `flash_pos`
- 含义：8 个方向的闪现落点局部坐标 `(x, z)`
- 来源：`predict_flash_pos(map_view, x, z)`
- 价值：可直接反映每个方向闪现后的着陆位置

#### `flash_pos_relative`
- 含义：8 个方向的相对位移 `(dz, dx)`
- 来源：由 `flash_pos` 与当前中心位置做差得到
- 价值：比绝对落点更适合特征工程和规则判断

#### `flash_valid_mask`
- 含义：8 个方向是否产生非零位移
- 来源：`flash_validation(flash_pos_relative)`
- 价值：可过滤“原地闪现但仍消耗冷却”的无效闪现方向

#### `flash_distance`
- 含义：各方向实际闪现位移长度
- 来源：由 `flash_pos_relative` 派生
- 价值：可区分“短闪”“远闪”，有助于策略判断

#### `flash_escape_score`
- 含义：各方向闪现后相对怪物的安全性评分
- 来源：闪现落点 + 怪物位置
- 价值：适合做启发式策略或奖励塑形
- 备注：实现复杂度略高，可后置

---

### 2. 怪物相关

#### `monsters_in_view`
- 含义：当前视野内怪物列表
- 来源：`monsters` 过滤 `is_in_view == True`
- 价值：便于统一后续处理逻辑

#### `nearest_monster`
- 含义：最近怪物对象/索引
- 来源：从视野内怪物中选取
- 价值：躲避策略中的核心对象

#### `nearest_monster_dist`
- 含义：最近怪物距离
- 来源：英雄与最近怪物坐标差
- 价值：高价值危险度特征

#### `monster_count_current`
- 含义：当前已出现怪物数量
- 来源：`len(monsters)`
- 价值：判断是否进入双怪阶段

#### `monster_phase`
- 含义：当前阶段，如“单怪 / 双怪 / 怪物已加速”
- 来源：`monster_count_current`、`step`、`monster_speed_boost_step`
- 价值：帮助策略切换

---

### 3. 宝箱/buff 相关

#### `visible_treasures`
- 含义：当前视野内宝箱列表
- 来源：`treasures`
- 价值：便于统一做目标选择

#### `nearest_treasure`
- 含义：最近宝箱对象/索引
- 来源：当前视野宝箱
- 价值：最直接的导航目标

#### `nearest_treasure_dist`
- 含义：最近宝箱距离
- 来源：英雄与最近宝箱坐标差
- 价值：对“是否顺路吃箱子”判断很有帮助

#### `remaining_treasure_count`
- 含义：当前剩余宝箱数
- 来源：`len(treasure_id)`
- 价值：反映任务进度

#### `treasure_progress`
- 含义：宝箱收集进度
- 来源：`treasures_collected / total_treasure`
- 价值：训练中可作为进度特征

#### `visible_buffs`
- 含义：当前视野内 buff 列表
- 来源：`buffs`
- 价值：和宝箱类似，便于统一处理

#### `nearest_buff`
- 含义：最近 buff 对象/索引
- 来源：当前视野 buff
- 价值：适合做机动性资源决策

#### `nearest_buff_dist`
- 含义：最近 buff 距离
- 来源：英雄与最近 buff 坐标差
- 价值：适用于“是否绕路吃 buff”判断

---

### 4. 英雄状态相关

#### `hero_has_buff`
- 含义：英雄当前是否拥有加速 buff
- 来源：`hero.buff_remaining_time > 0`
- 价值：比直接使用剩余时间更适合布尔化特征

#### `hero_speed`
- 含义：英雄本步移动能力
- 来源：是否有 buff，必要时结合配置
- 价值：影响逃跑/追宝箱策略

#### `hero_can_flash`
- 含义：当前是否可闪现
- 来源：`hero.flash_cooldown == 0`
- 价值：策略分支的重要条件

---

## 二、中等优先级候选字段

这些字段可能有帮助，但不是最先必须实现的。

### 1. 地图/探索相关

#### `frontier_cells`
- 含义：已探索区域与未探索区域交界处
- 来源：`map_full`
- 价值：若策略包含主动搜图，会很有用

#### `unexplored_ratio`
- 含义：未探索区域占比
- 来源：`map_explore_rate`
- 价值：与搜图策略相关

#### `local_obstacle_density`
- 含义：局部区域障碍密度
- 来源：`map_view`
- 价值：判断当前是否在狭窄地形/开阔地形

#### `corridor_flag`
- 含义：当前位置是否在通道中
- 来源：局部地图结构
- 价值：对躲怪和闪现很敏感

---

### 2. 动作可行性相关

#### `move_valid_mask_local`
- 含义：8 个基础移动方向中，哪些方向局部不撞墙
- 来源：`map_view` 中心 3×3 + 斜向防穿角规则
- 价值：比原始 `legal_action[:8]` 更接近真实移动效果

#### `flash_landing_openess`
- 含义：闪现落点周围开阔程度
- 来源：落点周围局部地图
- 价值：可辅助判断落点安全性

#### `preferred_action_mask`
- 含义：对原始动作空间的启发式偏好过滤
- 来源：局部图、闪现落点等
- 价值：适合 rule-based 辅助策略

---

### 3. 风险评估相关

#### `danger_score_local`
- 含义：当前局部危险度评分
- 来源：怪物距离、数量、地形、是否有闪现
- 价值：适合奖励塑形或规则切换

#### `safe_direction_mask`
- 含义：哪些方向会让最近怪物距离变大
- 来源：动作后位置与怪物位置关系
- 价值：可用于动作筛选

---

## 三、跨步缓存候选字段

这些字段的价值主要体现在：
- 环境本身不直接提供
- 但跨步比较后会得到强信息

### 1. 已有缓存（建议保留）

#### `hero_last`
- 用途：比较前后帧英雄状态

#### `pos_history`
- 用途：和第二只怪物生成规则直接相关

#### `map_full`
- 用途：维护全图探索信息

#### `treasure_full`
- 用途：跨步追踪宝箱是否已被收集

#### `buff_full`
- 用途：跨步追踪 buff 是否进入刷新/重新出现

---

### 2. 值得考虑新增的缓存

#### `flash_pos_last`
- 含义：上一帧 8 个方向闪现落点
- 价值：比较“因为位置变化，哪些闪现方向刚变可用/刚失效”

#### `flash_valid_mask_last`
- 含义：上一帧闪现有效掩码
- 价值：做变化检测

#### `nearest_monster_dist_last`
- 含义：上一帧最近怪物距离
- 价值：构造距离变化特征

#### `nearest_treasure_dist_last`
- 含义：上一帧最近宝箱距离
- 价值：衡量是否接近目标

#### `total_score_last`
- 含义：上一帧总分
- 价值：奖励塑形/debug 方便

#### `treasure_score_last`
- 含义：上一帧宝箱分
- 价值：判断本步是否刚吃到宝箱

#### `flash_count_last`
- 含义：上一帧闪现次数
- 价值：判断本步是否刚使用闪现

#### `collected_buff_last`
- 含义：上一帧 buff 获取次数
- 价值：判断本步是否刚吃到 buff

#### `monster_count_last`
- 含义：上一帧怪物数量
- 价值：检测第二只怪物刚生成的时刻

---

## 四、按“讨论优先级”建议的分组

如果后续要讨论“哪些该上、哪些先不做”，建议按下面三组来评估。

### A 组：最值得先讨论
这些通常信息增益高、实现成本低：
- `flash_pos`
- `flash_pos_relative`
- `flash_valid_mask`
- `nearest_monster_dist`
- `nearest_treasure_dist`
- `remaining_treasure_count`
- `hero_has_buff`
- `hero_can_flash`
- `monster_count_current`

### B 组：任务思路明确后再决定
这些有潜力，但取决于你最终采用的训练/规则思路：
- `flash_escape_score`
- `move_valid_mask_local`
- `danger_score_local`
- `safe_direction_mask`
- `frontier_cells`
- `local_obstacle_density`

### C 组：偏缓存/分析用途
这些更适合做调试、reward shaping 或复杂策略：
- `flash_pos_last`
- `flash_valid_mask_last`
- `nearest_monster_dist_last`
- `nearest_treasure_dist_last`
- `total_score_last`
- `flash_count_last`
- `collected_buff_last`
- `monster_count_last`

---

## 五、当前建议

现阶段不建议一次性把所有候选字段都塞进 `obs_extractor.py`。

更合理的思路是：
1. 先确定整体任务思路
2. 明确你更偏：
   - 纯 PPO 学习
   - PPO + 规则辅助
   - 更强的 reward shaping
3. 再从上面的 A/B/C 组里做筛选

如果只是为了先推进一版可训练方案，通常先从 A 组挑即可。
