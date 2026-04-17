# 2. reward 当前实现

## 一、总体分析

### 1. 定位
当前 `reward.py` 的职责非常明确：

- 输入：`extractor.build_reward_state()` 产出的结构化奖励状态
- 输出：
  - 一个标量总奖励 `total`
  - 一个详细的 `reward_info` 字典，用于调试与监控

它完全建立在 extractor 已经整理好的状态语义之上，不直接回头解析原始环境观测。

### 2. 当前实现意图
当前 reward 的实现思路和设计清单高度一致，核心公式已经落地为：

\[
\text{total} = \alpha \cdot \text{survival} + (1-\alpha) \cdot \text{explore} + \text{terminal}
\]

其中：
- `alpha = stage_info.alpha`
- `survival`：生存与动作质量相关 shaping
- `explore`：资源获取与探索相关 shaping
- `terminal`：终局一次性奖励 / 惩罚

这说明 reward 当前的核心目标很明确：

> 随着阶段推进，逐步提高“保命与局势质量”的权重，降低“探索与资源推进”的相对权重。

### 3. 冲突时按代码为准的结论
和 `ref/设计清单/2_reward.md` 对照后，当前实现需要按代码确认以下几点：

1. **环境原始奖励没有直接取环境 `reward.reward`**，而是取 extractor 里的分数增量：
   - `step_score_delta`
   - `treasure_score_delta`

2. **survival bucket 当前实际包含**：
   - 步数分增量
   - 最近怪距离变化
   - 夹击惩罚
   - 可活动空间变化
   - 无效移动惩罚
   - 低价值闪现惩罚
   - 闪现脱险奖励
   - 穿墙闪现奖励
   - 重复访问惩罚

3. **设计稿里的 topology 惩罚目前代码已注释掉**：
   - `is_dead_end`
   - `is_corridor`
   - `is_low_openness`
   虽然相关权重常量还在，但当前 reward 计算并未启用这部分。

4. **explore bucket 当前除了宝箱推进，还实际包含了 buff 接近奖励**，这一点设计稿里是弱提及，代码已明确实现。

5. **终局奖励目前比较克制**：
   - 完成：`+1.0`
   - 异常截断：`0.0`
   - 阵亡：按阶段基础惩罚，再叠加夹击 / 死角 / 同侧追赶修正

6. **当前 reward_info 设计得很细**，每个子项都会以 `s_` / `e_` / `t_` 前缀展开，方便调试。

---

## 二、文件结构

`reward.py` 可以分成四块：

1. 小工具函数
2. 超参数 / 权重常量
3. `compute_reward()` 主入口
4. 三个子桶函数：`_survival()`、`_explore()`、`_terminal()`

---

## 三、辅助函数与权重常量

## 1. 辅助函数

### `_clip(x, lo, hi)`
简单裁剪函数。

### `_resource_distance_gain(distance)`
根据资源距离返回一个分段权重：

- `<= 1`：`1.00`
- `<= 3`：`0.45`
- `<= 6`：`0.18`
- 更远：`0.08`
- `None`：`0.0`

### 印象
这个函数说明当前 reward 对“接近资源”的 shaping 不是一刀切，而是：

> 越靠近资源，距离改善的奖励越值钱。

---

## 2. 权重常量
当前 reward 常量按几个大类组织。

### 2.1 环境原始分数项
- `W_STEP_SCORE = 1/50`
- `W_TREASURE_SCORE = 1/50`

### 2.2 生存相关
- `SAFE_DIST = 20`
- `W_DIST_DELTA = 0.015`
- `DIST_DELTA_CLIP = 5`
- `ENCIRCLE_AVG_SAFE = 50`
- `W_ENCIRCLE = 0.020`
- `W_SPACE_DELTA = 0.001`
- `SPACE_DELTA_CLIP = 10`

### 2.3 地形惩罚（当前未启用）
- `DEAD_END_PEN = -0.04`
- `CORRIDOR_PEN = -0.02`
- `LOW_OPEN_PEN = -0.01`

### 2.4 探索 / 资源推进
- `W_EXPLORE = 50.0`
- `W_TREASURE_APPROACH = 0.010`
- `W_BUFF_APPROACH = 0.008`
- `TREASURE_APPROACH_CLIP = 5.0`
- `BUFF_APPROACH_CLIP = 5.0`

### 2.5 动作质量
- `NO_MOVE_PEN = -0.2`
- `LOW_FLASH_RATIO = 0.4`
- `LOW_FLASH_PEN = -0.2`
- `FLASH_ESCAPE_BONUS = 0.06`
- `FLASH_ACROSS_WALL_BONUS = 0.1`
- `VISIT_THRESH = 5`
- `W_REVISIT = 0.1`
- `REVISIT_CAP = 0.1`

### 2.6 终局项
- `COMPLETE_BONUS = 1.0`
- `DEATH_STAGE_PEN = {1:-0.6, 2:-0.4, 3:-0.2}`
- `DEATH_ENCIRCLE_PEN = -0.25`
- `DEATH_DEAD_END_PEN = -0.20`
- `DEATH_SAME_SIDE_REDUCE = 0.10`

### 印象
权重设计整体上比较保守，没有把 shaping 权重拉得特别大，仍保留“原始任务目标为主，shaping 为辅”的风格。

---

## 四、主入口 `compute_reward()`

## 1. 输入
输入来自 `extractor.build_reward_state()`，当前主入口会读取：

- `raw`
- `terminated`
- `truncated`
- `abnormal_truncated`
- `action_last`
- `monster_summary`
- `resource_summary`
- `space_summary`
- `stage_info`

以及 `action_last.reward_delta` 中的各类增量。

## 2. 空输入处理
若 `raw is None`：
- 直接返回 `(0.0, {})`

## 3. 核心流程
1. 取 `alpha = stage_info.alpha`
2. 调 `_survival(...)`
3. 调 `_explore(...)`
4. 调 `_terminal(...)`
5. 汇总： $$\text{total} = \alpha \cdot survival + (1-\alpha) \cdot explore + terminal$$

## 4. reward_info 输出
最终会输出一个扁平字典，包括：

- `total`
- `alpha`
- `survival`
- `survival_weighted`
- `explore`
- `explore_weighted`
- `terminal`
- 所有 survival 子项，前缀 `s_`
- 所有 explore 子项，前缀 `e_`
- 所有 terminal 子项，前缀保留 `t_`

### 印象
当前 reward 不是只返回一个标量，而是有明确的“可解释性接口”，这对后续调权重和诊断很有帮助。

---

## 五、生存项 `_survival()`

这一桶对应设计清单里的：
- 原始步数分
- 生存相关 shaping
- 动作质量 shaping

返回：
- 生存总奖励 `r`
- 每个子项组成的 `info`

---

## 1. 步数分 `step_score`
计算：

```python
step_score = rd.step_score_delta * W_STEP_SCORE
```

即将环境步数分增量按 `1/50` 缩放后作为 survival 的一部分。

### 含义
这说明当前 reward 明确把“多活一步”归到 survival 桶，而不是 explore 桶。

---

## 2. 最近怪距离变化 `monster_dist`
触发条件：
- `nearest_monster_distance` 不为空
- `nearest_monster_distance_delta` 不为空

计算过程：

1. 危险修饰项：

```python
danger = max(0.0, 1.0 - nearest_distance / SAFE_DIST)
```

2. 距离变化裁剪：

```python
delta = clip(nearest_monster_distance_delta, -DIST_DELTA_CLIP, DIST_DELTA_CLIP)
```

3. 最终项：

```python
monster_dist = delta * W_DIST_DELTA * (1.0 + danger)
```

### 含义
- 若最近怪距离拉大，则正奖励
- 若最近怪更近，则负奖励
- 而且越危险时，这项权重越大

### 印象
这是当前 reward 里最核心的 survival dense shaping 之一。

---

## 3. 夹击惩罚 `encircle`
触发条件：
- 怪物数 `>= 2`
- `relative_direction_cosine` 可用

计算过程：

1. 平均距离修饰：

```python
avg_mod = max(0.0, 1.0 - avg_distance / ENCIRCLE_AVG_SAFE)
```

2. 夹击强度：

```python
threat = max(0.0, -relative_direction_cosine)
```

3. 最终惩罚：

```python
encircle = -(threat * avg_mod * W_ENCIRCLE)
```

### 代码语义
- 只有当两个怪的相对方向余弦为负，即更像两侧 / 对向包夹时才会产生惩罚
- 而且怪越近，惩罚越明显

### 印象
这是对设计稿里“夹击趋势”比较干净的一版实现。

---

## 4. 活动空间变化 `space`
计算：

```python
sd = clip(ss.traversable_space_delta, -SPACE_DELTA_CLIP, SPACE_DELTA_CLIP)
space = sd * W_SPACE_DELTA
```

### 含义
- 可活动空间增加：正奖励
- 可活动空间减少：负奖励

### 印象
这一项很轻，但它把空间质量变化作为一个稳定稠密信号接进来了。

---

## 5. 地形惩罚 `topology`（当前未启用）
代码里相关逻辑存在，但被整体注释掉：

- `ss.is_dead_end`
- `ss.is_corridor`
- `ss.is_low_openness`

虽然常量还在，但当前 `topology = 0.0`，不参与总奖励。

### 代码优先结论
> 当前 reward 实际上并没有启用“进入死角 / 走廊 / 低开阔区域”的逐步惩罚。

### 印象
这看起来像是实现过一版，但后来为了稳定性暂时关掉了。

---

## 6. 无效移动惩罚 `no_move`
条件：
- `last_action_id >= 0`
- `not al.moved`

则：

```python
no_move = NO_MOVE_PEN
```

### 含义
任何执行了动作但没位移的情况都会受罚，包括：
- 撞墙
- 原地无效闪现
- 其他没产生位移的动作

### 印象
这是一个非常直接、解释性很强的动作质量惩罚。

---

## 7. 闪现相关 shaping
这部分只有在 `rd.flash_count_delta > 0` 时触发，说明当前步确实交了闪现。

### 7.1 低价值闪现 `flash_low`
若本次动作是闪现（`action_id >= 8`）：

- 读取上一步预测好的 `prev_flash_distance`
- 找到该方向的实际使用距离 `used_d`
- 若 `used_d < max_d * LOW_FLASH_RATIO`
  - 记 `flash_low = LOW_FLASH_PEN`

即：如果这次闪现距离明显低于当前可达到的最佳距离，会被认为是低价值闪现。

### 7.2 穿墙闪现奖励 `flash_across_wall`
若该闪现方向在 `prev_flash_across_wall` 中被标记为跨墙，则：

```python
flash_across_wall = FLASH_ACROSS_WALL_BONUS
```

### 7.3 闪现脱险奖励 `flash_escape`
若 extractor 提供的 `flash_escape_improved_estimate` 为真，则：

```python
flash_escape = FLASH_ESCAPE_BONUS
```

其语义是：
- 交闪后最近怪路径距离改善
- 或抓捕裕度提升
- 或安全方向数增加

### 印象
当前闪现 shaping 已经不再是“用了就奖 / 用了就罚”，而是很明确地区分：
- 闪得短 → 罚
- 跨墙 → 奖
- 真脱险 → 奖

这和设计清单的目标是一致的。

---

## 8. 重复访问惩罚 `revisit`
读取：
- `hero_visit_count`

若当前位置访问次数超过阈值：

```python
revisit = -min((vc - VISIT_THRESH) * W_REVISIT, REVISIT_CAP)
```

### 含义
当前位置访问次数越高，惩罚越强，但有上限。

### 印象
这是个轻量版“绕圈 / 低效移动”惩罚，只使用当前位置访问计数，不引入复杂轨迹逻辑。

---

## 9. survival 总和
当前 survival 总和为：

```python
step_score + monster_dist + encircle + space + topology + no_move + flash_low + flash_escape + flash_across_wall + revisit
```

其中 `topology` 目前恒为 0。

### 总体印象
survival 桶当前已经覆盖了：
- 原始生存收益
- 怪物压力变化
- 包夹压力
- 空间变化
- 动作质量
- 闪现质量
- 反绕圈约束

是当前 reward 的主体。

---

## 六、探索项 `_explore()`

这一桶对应设计清单里的：
- 宝箱得分
- 接近资源
- 探索未知区域

返回：
- 探索总奖励 `r`
- 各子项 `info`

---

## 1. 宝箱分 `treasure_score`
计算：

```python
treasure_score = rd.treasure_score_delta * W_TREASURE_SCORE
```

即环境宝箱分增量缩放后直接进入 explore 桶。

---

## 2. 接近宝箱 `treasure_approach`
若 `nearest_known_treasure_distance_path_delta` 可用：

1. 先对距离变化取负：
   - 距离变小 → 正奖励
   - 距离变大 → 负奖励

2. 裁剪到 `[-TREASURE_APPROACH_CLIP, TREASURE_APPROACH_CLIP]`

3. 再乘：
   - `W_TREASURE_APPROACH`
   - `_resource_distance_gain(current_distance)`

即：

```python
treasure_approach = (-clip(td)) * W_TREASURE_APPROACH * distance_gain
```

### 含义
越靠近最近已知宝箱，并且当前离宝箱本来就越近，这项奖励越明显。

---

## 3. 接近 buff `buff_approach`
逻辑和宝箱完全平行：

```python
buff_approach = (-clip(bd)) * W_BUFF_APPROACH * distance_gain
```

### 代码优先结论
虽然设计稿里“buff 本身就会提高步数分增量，本版本不给”主要是指显式收集奖励，但当前代码**确实给了 buff 接近 shaping**。

### 印象
这说明当前实现把 buff 当成了值得靠近的战略资源，而不是完全忽略。

---

## 4. 探索奖励 `map_explore`
直接使用：

```python
map_explore = al.map_explore_rate_delta * W_EXPLORE
```

### 含义
地图探索率只要有正增量，就会得到奖励。

### 印象
这项权重相对很大，但因为 `map_explore_rate_delta` 本身通常很小，所以整体仍是可控的 dense shaping。

---

## 5. explore 总和
当前 explore 总和为：

```python
treasure_score + treasure_approach + buff_approach + map_explore
```

### 总体印象
explore 桶比较纯粹，主要围绕：
- 资源分
- 靠近资源
- 探索未知区

没有把太多生存语义混进来。

---

## 七、终局项 `_terminal()`

这一部分是一次性奖励 / 惩罚，只在终局起作用。

---

## 1. 异常截断
若 `abnormal_truncated`：

- `t_type = "abnormal"`
- 返回 `0.0`

### 含义
异常截断当前不额外奖惩，只做标记。

---

## 2. 正常完成
若：
- `truncated == True`
- `terminated == False`

则：
- `t_type = "completed"`
- 给 `COMPLETE_BONUS = 1.0`

### 含义
走满步数被视为优质结局，有固定终局奖励。

---

## 3. 非终局状态
若当前没终局：
- `t_type = "ongoing"`
- 返回 `0.0`

---

## 4. 阵亡终局
若 `terminated == True`，则进入死亡分析。

### 4.1 阶段基础惩罚
根据当前阶段给基础惩罚：

- stage 1: `-0.6`
- stage 2: `-0.4`
- stage 3: `-0.2`

即：越后期死亡，基础惩罚越轻。

### 4.2 夹击死亡惩罚 `t_encircle`
若：
- 至少两只怪
- `relative_direction_cosine < -0.3`

则追加：

```python
DEATH_ENCIRCLE_PEN = -0.25
```

### 4.3 死角死亡惩罚 `t_dead_end`
若：
- `ss.is_dead_end`

则追加：

```python
DEATH_DEAD_END_PEN = -0.20
```

### 4.4 同侧追赶修正 `t_same_side`
若：
- `stage >= 2`
- 且不是夹击死亡
- 且不是死角死亡

则：

```python
t_same_side = DEATH_SAME_SIDE_REDUCE = 0.10
```

注意这是**加分修正**，用来减轻某类“不是明显蠢死”的死亡惩罚。

### 印象
终局项整体比较克制，没有做很复杂的死法分类，而是用：
- 阶段
- 是否夹击
- 是否死角
- 是否更像同侧追上

这四种语义做轻量修正。

---

## 八、reward 当前实现的数据流

可以把当前 reward 数据流概括为：

1. extractor 提供统一奖励状态
2. `compute_reward()` 读取 `stage_info.alpha`
3. `_survival()` 计算生存与动作质量桶
4. `_explore()` 计算资源与探索桶
5. `_terminal()` 计算一次性终局项
6. 三部分按阶段权重汇总成总奖励
7. 同时输出细粒度 `reward_info`

### 总体印象
当前 reward 的结构非常清晰，和设计清单高度同构，属于典型的“先搭稳定骨架，再微调各项权重”的实现方式。

---

## 九、我对当前 reward 实现的印象记录

### 1. 主体结构已经稳定
分成 survival / explore / terminal 三桶，再由 `alpha` 混合，这个主结构已经很清楚了。

### 2. 当前实现比设计稿更克制
虽然设计稿里提了不少候选 shaping，但代码实际只启用了其中一部分，而且都偏易解释、易稳定的项。

### 3. survival 桶明显更成熟
- 最近怪距离变化
- 夹击惩罚
- 闪现质量
- 无效移动
- 反绕圈

这些已经形成比较完整的一组生存 shaping。

### 4. topology shaping 暂时被关掉
这很像实验中发现直接按死角 / 走廊逐步惩罚可能不够稳定，所以保留接口但先不启用。

### 5. reward 可解释性做得很好
`reward_info` 的展开方式很利于训练期看板和调参。

### 6. 整体仍然没有偏离原始任务分数
当前 reward 虽然加了不少 shaping，但原始步数分 / 宝箱分仍然处在体系核心位置，符合“不要偏离比赛目标”的原则。

---

## 十、一句话总结
当前 `reward.py` 已实现为一个 **按阶段混合 survival / explore / terminal 三桶的 shaping reward 系统**。

它的核心特点是：

> 随着局势进入高压阶段，逐步提高生存与动作质量的权重，同时保留资源推进与探索信号，并在终局用轻量死法分析做一次性修正。
