# 1. obs 当前实现

## 一、总体分析

### 1. 定位
当前 `obs.py` 的职责很明确：

- 输入：`extractor.build_obs_state()` 产出的结构化状态
- 输出：两类模型输入
  - `scalar`：定长一维向量
  - `matrix`：局部图 / 全局图多通道矩阵

也就是说，当前 `obs` **不直接解析环境原始字典**，而是完全建立在 extractor 已整理好的标准语义之上。这和设计清单里的模块边界是一致的：

- `extractor` 负责整理状态
- `obs` 负责选择和编码状态

### 2. 当前实现意图
当前 obs 的核心意图不是“把所有字段尽量塞满”，而是采用清晰的双路输入结构：

- `scalar` 路：承载高层语义、阶段、动作、资源槽位等结构化信息
- `matrix` 路：承载局部 / 全局空间结构

所以当前 obs 的思路可以概括为：

> 用 `scalar` 表达决策摘要，用 `matrix` 表达空间结构。

### 3. 冲突时按代码为准的结论
和 `ref/设计清单/1_obs.md` 对照后，需要明确以下几点以代码为准：

1. **怪物距离编码已经定案**，不是设计稿里的口头描述，而是 `_encode_monster_dist()`：
   - 视野内：`chebyshev / 30`，并截到 `1/3`
   - 视野外：`1/3 + log2(bucket+2)/3 * 2/3`
   - 不存在：`1.0`

2. **空间 scalar 当前只用了 `corridor_lengths` 8 维**，没有把 `openness`、`safe_direction_count`、`is_dead_end` 等直接放进 obs。

3. **阶段信息当前只编码了连续进度量**：
   - 当前步数进度
   - 第二阶段倒计时
   - 第三阶段倒计时
   没有直接编码 `stage one-hot` 或 `alpha`。

4. **资源是固定槽位展开的**：
   - 宝箱槽位上限 `10`
   - buff 槽位上限 `3`

5. **全局矩阵已经定案为 `64×64`**，即 `128 -> 64` 的 2 倍下采样。

---

## 二、文件结构

`obs.py` 可以分成四块：

1. 常量与归一化辅助函数
2. `construct_obs_scaler()`：构造 `134` 维标量观测
3. 矩阵辅助函数：下采样与对数归一化
4. `construct_obs_matrix()`：构造局部 / 全局图观测

---

## 三、常量与辅助函数

## 1. 常量
当前 obs 自己定义了一组编码时使用的上界：

- `MAX_MONSTER_SPEED = 5`
- `MAX_DIST_BUCKET = 5`
- `MAX_BUFF_DURATION = 50`
- `MAX_FLASH_DIST = 10`
- `MAX_PATH_DIST = 200`
- `MAX_L2_DIST = 182.0`
- `TOTAL_TREASURE_SLOTS = 10`
- `TOTAL_BUFF_SLOTS = 3`
- `GLOBAL_DS_SIZE = 64`
- `VISIT_LOG_CAP = log1p(100.0)`

### 印象
这些常量说明 obs 不是纯粹搬运 extractor 数据，而是主动承担了“编码尺度设计”的职责。

## 2. 归一化与编码辅助

### `_norm(v, v_max, v_min=0.0)`
将数值裁剪到指定范围后映射到 `[0,1]`。

### `_safe_dist(path_d, l2_d)`
优先取路径距离 `path_d`，没有则退回 `l2_d`，都没有则返回 `1.0`。

### `_encode_monster_dist(is_in_view, chebyshev_dist, bucket)`
这是当前 obs 怪物距离编码的关键函数：

- 不存在：`1.0`
- 视野内：`min(cheb / 30, 1/3)`
- 视野外：`1/3 + log2(bucket+2)/3 * 2/3`

它的目的很明确：
- 近距离高分辨率
- 远距离压缩表达
- 保持“越危险越小”的单调关系

### 印象
怪物距离编码明显不是“简单归一化”，而是在有意识地区分近距压力和远距粗略感知。

---

## 四、标量观测 `construct_obs_scaler()`

## 1. 总体结构
当前标量维度固定为：

- `SCALAR_DIM = 134`

组成如下：

- hero：6
- last：7
- action：32
- monster：16
- resource：62
- space：8
- stage：3

说明当前 obs 已经非常明确地采用了“按语义分块、最后拼接”的结构。

---

## 2. hero 状态块（6维）
来源：
- `data["hero"]`
- `data["hero_speed"]`
- `raw.flash_cooldown_max`

编码内容：

1. `hero.x / MAP_SIZE`
2. `hero.z / MAP_SIZE`
3. `hero_speed / 2`
4. `buff_remaining_time / MAX_BUFF_DURATION`
5. `can_flash`
6. `flash_cooldown / raw.flash_cooldown_max`

### 实现含义
这一块覆盖了当前实现中最基础的英雄自身状态：
- 位置
- 速度
- buff 状态
- 闪现可用性
- 闪现剩余冷却

### 印象
- 绝对坐标直接进入 obs，说明当前设计不回避全局位置语义
- 英雄速度取的是 extractor 派生值，而不是自行判断

---

## 3. 上一步动作反馈块（7维）
来源：`ActionLast`

编码内容：

1. 上一步位移 `dx / 10`，裁到 `[-1,1]`
2. 上一步位移 `dz / 10`，裁到 `[-1,1]`
3. 最近怪距离是否增加
4. 是否捡到宝箱
5. 是否捡到 buff
6. 地图探索率增量 ×100，再裁到 `[0,1]`
7. `step_score_delta / 5`

### 代码层面实际选择
虽然 `ActionLast.reward_delta` 里有多种 delta，但当前 obs 实际只把 `step_score_delta` 放进来了，没有把 treasure / buff / total score 一起编码。

### 印象
这一块是“短期行为反馈”的精简版实现，既给模型一点历史感，又没有把历史状态堆得太重。

---

## 4. 动作预测块（32维）
来源：`ActionPredict`

编码内容按 8 个方向展开：

1. `move_valid_mask`：8维
2. `flash_valid_mask`：8维
3. `flash_distance / MAX_FLASH_DIST`：8维
4. `flash_across_wall`：8维

### 当前没有放进来的信息
虽然 extractor 里还有：
- `flash_pos`
- `flash_pos_relative`
- `action_preferred`

但当前 obs 并没有直接编码这些字段。

### 印象
当前 obs 对动作块的取舍很实用：
- 是否能走
- 闪现是否有效
- 闪现能走多远
- 闪现是否跨墙

这几项已经足够支撑大量局面决策。

---

## 5. 怪物压力块（16维）
来源：`MonsterSummary`

### 5.1 每只怪物 7 维
通过内部 `_monster_vec(idx)` 编码，若怪不存在则整组全 0；若存在则包含：

1. 是否存在
2. 出现倒计时归一化
3. 相对位置 `dx / MAP_SIZE`
4. 相对位置 `dz / MAP_SIZE`
5. 距离编码 `_encode_monster_dist(...)`
6. 速度归一化
7. 是否为最近怪

### 5.2 双怪组合额外 2 维
1. `relative_direction_cosine`
2. `average_monster_distance / MAP_SIZE`

### 重要实现细节
怪物是否在视野内不是从 `MonsterSummary` 直接拿，而是从 `raw.monsters[idx-1].is_in_view` 读取，然后决定如何编码距离。

### 印象
这块比较完整地落了设计稿里“单怪压力 + 双怪组合关系”的核心内容，但保持在低维摘要层面，没有把更复杂的空间压缩趋势直接塞进 monster block。

---

## 6. 资源块（62维）
这是当前 obs 里信息量最大的一块。

## 6.1 最近资源摘要（10维）
来源：`ResourceSummary`

### 宝箱 5 维
1. 宝箱发现进度
2. 宝箱收集进度
3. 最近宝箱方向 x
4. 最近宝箱方向 z
5. 最近宝箱距离（纯 L2 距离归一化）

### buff 5 维
6. buff 发现进度
7. buff 收集进度
8. 最近 buff 方向 x
9. 最近 buff 方向 z
10. 最近 buff 距离（纯 L2 距离归一化）

## 6.2 宝箱固定槽位（40维）
按 `slot_id = 1..10` 遍历 `treasure_full`，每个槽位 4 维：

1. 状态值
   - 未发现：`0.0`
   - `status == 1`：`1.0`
   - 其他已知状态：`0.5`
2. 相对 x
3. 相对 z
4. L2 距离归一化

## 6.3 buff 固定槽位（12维）
按 `sorted(buff_full.keys())` 取前 `TOTAL_BUFF_SLOTS=3` 个，每个槽位 4 维：

1. 状态值
   - 未发现：`-1.0`
   - 可立即获取：`1.0`
   - 冷却中：`1 - cooldown / buff_refresh_time`，即 `0~1` 表示冷却进度
   - 其他已知但当前不可立即获取：`0.0`
2. 相对 x
3. 相对 z
4. L2 距离归一化

### 重要实现记录
obs 这里的资源槽位编码严格继承 extractor 的缓存 key 约定，因此：

> 如果 extractor 那边的 treasure / buff `config_id` 假设与真实环境有偏差，obs 的固定槽位语义也会一起偏。

### 印象
资源块明显是当前 obs 最重的一块，说明当前实现非常重视：
- 最近目标
- 全局已知资源表
- 固定槽位的长期记忆语义

### 当前实现更新（2025-08）
- 最近 treasure / buff 摘要距离已不再依赖已知图 BFS 或 path 估计，统一改为 **纯 L2 距离归一化**。
- buff 槽位状态值已改为更强语义编码：
  - `-1` 表示未发现
  - `0~1` 表示冷却进度
  - `1` 表示当前可领取
- extractor 的 `update()` 主链已停用 obs / reward 侧 BFS 计算，不再在每步更新中执行相关路径搜索。
- 同时，静态地图在 `_load_all_static_maps()` 后会额外预计算并常驻保存“合法点-合法点最短距离缓存”，供后续 reward / analysis 查询复用。

---

## 7. 空间块（8维）
来源：`SpaceSummary`

当前只编码：
- `corridor_lengths` 的 8 个方向长度，并用 `VIEW_SIZE` 归一化

### 设计稿与实现的差异
虽然设计稿提到了：
- 可活动空间
- 开阔度
- 风险空间统计

但当前代码并没有把这些直接塞进 scalar。

### 印象
空间 scalar 现在走的是“先上最稳的一层”，只保留八方向通路长度。

---

## 8. 阶段与节奏块（3维）
编码内容：

1. `step_progress = raw.step / raw.max_step`
2. `stage2_cd = max(monster_interval - step, 0) / raw.max_step`
3. `stage3_cd = max(monster_speed_boost_step - step, 0) / raw.max_step`

### 代码优先的理解
当前 obs 不直接编码：
- 当前 stage 编号
- `stage one-hot`
- `alpha`

而是用两个关键节点的倒计时来间接表达节奏。

### 印象
这是一个很明显的实现风格：

> 当前 obs 更偏连续进度量，而不是离散阶段标签。

---

## 五、矩阵观测 `construct_obs_matrix()`

## 1. 总体结构
当前返回两个矩阵分支：

- `local`: `(8, 21, 21)`
- `global`: `(4, 64, 64)`

返回结构：

```python
{
    "local": local_map,
    "global": global_map,
}
```

---

## 2. 局部图 `local`（8通道）
来源：`LocalMapLayers`

通道顺序：

1. `obstacle`
2. `hero`
3. `monster`
4. `treasure`
5. `buff`
6. `visit`（`log1p` 归一化）
7. `flash_landing`
8. `visit_coverage`（`log1p` 归一化）

### 重要实现点
虽然 `LocalMapLayers.as_stack()` 默认没有包含 `visit_coverage`，但 `construct_obs_matrix()` 没有直接用它，而是手动 stack，所以 `visit_coverage` 在 obs 的局部图里**确实被用上了**。

### 印象
obs 层对最终图通道拥有明确控制权，而不是被 dataclass 默认方法绑定。

---

## 3. 全局图 `global`（4通道，64×64）
来源：
- `global_map_full`
- `global_visit_coverage`
- `global_treasure_available_map`
- `global_buff_known_map`

### 通道 1：walkability
`map_full` 中：
- `1 -> 1.0`
- `0 -> 0.25`
- `-1 -> 0.0`

然后做平均下采样到 `64×64`。

### 通道 2：visit coverage
先做 `log1p` 归一化，再平均下采样。

### 通道 3：treasure available
对宝箱图做 max-pool 下采样。

### 通道 4：buff known
对 buff 图做 max-pool 下采样。

### 注释与代码不一致的地方
函数中有一条注释写了 `128×128 → 32×32`，但这是旧注释；真实实现由 `GLOBAL_DS_SIZE = 64` 决定，所以实际输出是：

- `128×128 → 64×64`

### 印象
全局图采用的是比较克制的 4 通道方案，聚焦：
- 可通行性
- 访问热度
- 宝箱分布
- buff 分布

而不是一开始就做“大而全”的多通道全局图。

---

## 六、obs 的整体数据流

当前 obs 的数据流可以概括为：

1. `Extractor.build_obs_state()` 提供统一结构化状态
2. `construct_obs_scaler(data)` 取摘要字段，编码成 `134` 维标量向量
3. `construct_obs_matrix(data)` 取局部 / 全局缓存图，编码成两个矩阵分支
4. 最终模型输入由：
   - scalar branch
   - local matrix branch
   - global matrix branch
   组成

### 总体印象
这是一个非常明确的多分支输入设计，不是单一路径。

---

## 七、我对当前 obs 实现的印象记录

### 1. 风格非常克制
设计清单里有很多候选方向，但当前实现真正落地时比较节制，只保留了最有把握的一批信息。

### 2. 边界很健康
obs 几乎不重复做 extractor 已经做过的业务推理，而是消费 extractor 的标准 summary。

### 3. scalar + matrix 分工清晰
- scalar 放高层语义
- matrix 放空间结构

这是当前 obs 最稳定、最合理的地方。

### 4. 资源表达是当前 obs 最重的部分
62 维资源块说明当前实现非常重视“资源记忆”和“最近资源机会”，这和任务本身是对齐的。

### 5. 空间 scalar 仍偏保守
更复杂的空间语义主要还是留在矩阵分支和 reward 侧，scalar 里只放了八方向通路长度。

---

## 八、一句话总结
当前 `obs.py` 已实现为一个 **基于 extractor 的双路观测编码器**：

- 用 `134` 维 scalar 表达高层状态、动作、怪物、资源、阶段信息
- 用 `8×21×21` 的局部图和 `4×64×64` 的全局图表达空间结构

它的核心特点是：

> 不直接重复环境原始字段，而是围绕“危险、机会、动作空间、阶段压力”做选择性编码，并把高层语义与空间结构分路输入模型。
