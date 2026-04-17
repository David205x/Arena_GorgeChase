# 3. algorithm_structure 当前实现

## 一、总体分析

### 1. 定位
当前 `algorithm_structure` 已经落成一套完整的 **PPO + 三分支 ActorCritic 网络 + 分布式 value 学习** 实现，主要由三部分组成：

- `agent_diy/algorithm/objectives.py`：价值目标函数
- `agent_diy/algorithm/algorithm.py`：PPO 训练主体
- `agent_diy/model/model.py`：模型结构

从职责边界看：

- `obs` 决定输入内容与编码格式
- `model` 决定如何编码和融合这些输入
- `algorithm` 决定如何基于 PPO 学习策略与价值

### 2. 当前实现意图
我对当前实现意图的理解是：

> 在尽量复用 PPO 稳定训练范式的前提下，给当前任务做一个“结构化向量 + 局部图 + 全局图”的多分支输入网络，并把 critic 做成分布式 value 预测，以提升训练稳定性与表达能力。

相比设计清单，当前实现已经不是“结构化向量 + 局部地图”双分支，而是明确采用了：

- `scalar` 分支
- `local_map` 分支
- `global_map` 分支

也就是说，**全局图分支已经进入第一版正式实现**。

### 3. 冲突时按代码为准的结论
和 `ref/设计清单/3_algorithm_structure.md` 对照后，需要按代码明确以下几点：

1. **算法主体就是 PPO**，并且是较完整的 PPO 变体：
   - advantage normalization
   - PPO clip objective
   - dual-clip
   - entropy regularization
   - gradient clipping
   - importance ratio 的 log-space 计算

2. **critic 不是标量回归，也不是普通 MSE**，而是：
   - 分布式 value 头
   - `HLGaussLoss` 交叉熵目标

3. **输入结构不是双分支，而是三分支**：
   - `scalar (134,)`
   - `local_map (8,21,21)`
   - `global_map (4,64,64)`

4. **actor / critic 共享主干，最后分头输出**，这一点与设计稿一致。

5. **legal action mask 直接接在 policy logits 上**，不是额外的动作约束模块。

6. **当前实现没有实体 token / attention / 时序模块**，仍然走的是简单稳定路线。

---

## 二、文件分工

## 1. `objectives.py`
只做一件事：实现 value 学习目标 `HLGaussLoss`。

## 2. `algorithm.py`
负责 PPO 训练逻辑：
- 创建模型与优化器
- 执行前向
- 计算 PPO 损失
- 反向传播与梯度裁剪
- 记录训练监控指标

## 3. `model.py`
负责神经网络结构：
- 三路输入编码
- 特征融合
- actor head
- critic head
- action mask
- value expectation 辅助函数

### 印象
当前算法结构分工非常清晰，没有把网络结构和训练逻辑混在一起。

---

## 三、价值目标 `objectives.py`

## 1. `HLGaussLoss`
当前价值学习目标不是普通标量回归，而是使用 `HLGaussLoss`。

类的核心参数：
- `min_value`
- `max_value`
- `num_bins`
- `sigma`
- `device`

内部会构造：
- `support = linspace(min_value, max_value, num_bins + 1)`

### 2. `forward()`
输入：
- `logits`
- `target`

做法：
- 先把标量 target 转成分布 `transform_to_probs(target)`
- 然后对 `logits` 做 `cross_entropy`

### 3. `transform_to_probs()`
把标量目标映射到 soft categorical 分布：
- 通过误差函数 `erf` 计算各 bin 的累计概率
- 再求相邻 bin 边界差值得到 bin probability
- 最后归一化

### 4. `transform_from_probs()`
把分布概率转回期望值：
- 用每个 bin center 的加权和

### 5. 当前实现含义
这意味着 critic 当前学习的是：

> “回报在 value support 上的分布”

而不是一个单点标量。

### 印象
这部分是当前算法结构里最“非 baseline”的增强项，也是最能体现实现意图的一块：
- 不是改 PPO 主框架
- 而是增强 critic 的目标表达

---

## 四、模型结构 `model.py`

## 1. 总体结构
模型是一个三分支共享主干的 ActorCritic：

- `scalar (134,) -> SimbaEncoder -> 256`
- `local_map (8,21,21) -> ConvNeXtEncoder -> 128`
- `global_map (4,64,64) -> ConvNeXtEncoder -> 128`
- 三者拼接成 `512`
- 再经过一个共享 `SimbaEncoder` 融合
- 最后分成 policy head 和 value head

### 常量
- `SCALAR_DIM = 134`
- `LOCAL_CH = 8`
- `GLOBAL_CH = 4`
- `ACTION_NUM = 16`
- `VF_N_BINS = 51`
- `VF_MIN = -10.0`
- `VF_MAX = 10.0`
- `VECTOR_EMBED = 256`
- `VISION_EMBED = 128`
- `TORSO_DIM = 512`

### 印象
模型结构和 obs 编码是严格对齐的，说明这一版实现的输入-网络耦合关系是清楚的。

---

## 2. 三路输入编码器

## 2.1 scalar 分支
使用：
- `SimbaEncoder(input_dim=134, hidden_dim=256, block_num=2)`

输出：
- `scalar_embed: 256`

### 含义
这个分支负责编码高层结构化语义：
- 英雄状态
- 动作预测
- 怪物摘要
- 资源摘要
- 空间摘要
- 阶段信息

## 2.2 local_map 分支
使用：
- `ConvNeXtEncoder(in_channels=8, dims=[32,64,128], depths=[1,1,1], downsample_sizes=[2,2,2])`

输出：
- `local_embed: 128`

### 含义
这个分支负责编码局部空间结构：
- 地形
- 局部怪物
- 局部资源
- 闪现落点
- 访问热度等

## 2.3 global_map 分支
使用：
- `ConvNeXtEncoder(in_channels=4, dims=[32,64,128], depths=[1,1,1], downsample_sizes=[2,2,2])`

输出：
- `global_embed: 128`

### 含义
这个分支负责编码全局探索与资源分布信息。

### 和设计稿的差异
设计稿里“全局图暂不作为第一版主输入”，但当前代码里它已经正式进入主结构。

### 印象
当前模型结构比设计稿更积极，已经把全局图纳入主输入，而不是只保留摘要统计。

---

## 3. 融合主干（Torso）
三路编码结果先拼接：

- `256 + 128 + 128 = 512`

然后进入：
- `SimbaEncoder(input_dim=512, hidden_dim=512, block_num=2)`

得到共享表示 `torso_out`。

### 当前实现特点
- 采用的是**简单拼接 + 共享编码器**
- 没有 attention
- 没有 gating
- 没有跨模态复杂交互模块

### 印象
这一点和设计稿“先上简单稳定的融合，再考虑更复杂融合”高度一致。

---

## 4. Policy Head
策略头结构：

- `ResidualBlock(512)`
- `LayerNorm(512)`
- `Linear(512 -> 16)`

输出：
- `policy_logits`

然后会调用 `_mask_illegal()` 对非法动作做 mask，最后：
- `softmax(policy_logits)` 得到 `policy_probs`

### `_mask_illegal()` 实现
逻辑是：
1. 先取合法动作上的最大 logit 做平移
2. 非法动作位置加上一个很大的负值 `1e5 * (legal_action - 1.0)`

因为：
- 合法动作为 `1`
- 非法动作为 `0`
- 所以非法位置会被减去 `1e5`

### 含义
这是一种 reference PPO 风格的、数值稳定的 illegal-action masking。

### 印象
动作掩码接法非常直接，没有引入额外约束模块，和设计稿一致。

---

## 5. Value Head
价值头结构：

- `ResidualBlock(512)`
- `LayerNorm(512)`
- `Linear(512 -> VF_N_BINS)`

输出：
- `value_logits: (B, 51)`

此外模型还注册了：
- `vf_support`
- `vf_centers`

并提供 `value_expected()`：
- 先 softmax 成概率
- 再与 `vf_centers` 加权求期望

### 含义
critic 的直接输出不是标量 value，而是 value distribution 的 logits。

### 印象
这与 `HLGaussLoss` 是一套成体系的设计，不是单点增强，而是完整的 distributional critic 路线。

---

## 6. forward 接口
模型 `forward()` 输入：
- `scalar`
- `local_map`
- `global_map`
- `legal_action`

输出：
- `policy_probs`
- `value_logits`

当前没有返回中间 embedding，也没有额外 auxiliary head。

### 印象
接口非常简洁，说明当前实现优先的是主训练链路稳定，而不是做过多实验性扩展。

---

## 五、PPO 训练主体 `algorithm.py`

## 1. 总体说明
`algorithm.py` 里实现的是一个增强版 PPO，注释已经明确写了主要增强点：

- HL-Gauss distributional value loss
- importance ratio 的 log-space 计算
- ratio hard clamp `[0,3]`
- Dual-Clip PPO
- batch 内 advantage normalization
- `channels_last` 内存格式加速 CNN

这说明当前算法主体的思路是：

> PPO 主体不做大改，但在数值稳定性和 critic 表达能力上做增强。

---

## 2. 超参数
当前主要超参数：

- `CLIP_PARAM = 0.2`
- `DUAL_CLIP = 3.0`
- `VF_COEF = 0.5`
- `ENTROPY_COEF = 0.01`
- `VF_SIGMA = 0.75`
- `GRAD_CLIP_NORM = 0.5`
- `LR = 3e-4`
- `ADV_NORM = True`
- `LOG_INTERVAL = 60`

### 印象
整体是比较标准、偏稳定训练取向的一组配置。

---

## 3. `Algorithm.__init__()`
初始化时完成：

### 3.1 模型
- `self.model = Model(device=device).to(device)`
- 再转成 `channels_last` 内存格式

### 3.2 优化器
- `AdamW`
- `lr=3e-4`
- `betas=(0.9, 0.999)`
- `eps=1e-8`

### 3.3 value objective
- 创建 `HLGaussLoss(min=VF_MIN, max=VF_MAX, num_bins=VF_N_BINS, sigma=VF_SIGMA)`

### 3.4 PPO 配置缓存
缓存：
- `clip_high = 1 + clip_param`
- `clip_low = 1 / clip_high`
- `dual_clip`
- `vf_coef`
- `entropy_coef`

### 3.5 监控相关
- `logger`
- `monitor`
- `last_report_time`
- `train_step`
- `last_monitor_results`

### 印象
初始化阶段把训练主体、优化器、value objective 和监控都组织好了，结构很规整。

---

## 4. `learn()` 单次 PPO 更新

### 4.1 输入 batch 格式
注释里已经明确要求：

- `scalar       (B, 134)`
- `local_map    (B, 8, 21, 21)`
- `global_map   (B, 4, 64, 64)`
- `legal_action (B, 16)`
- `old_action   (B, 1)`
- `old_prob     (B, 1)`
- `reward       (B,)`
- `advantage    (B,)`
- `td_return    (B,)`

### 4.2 主流程
1. 把 batch 张量搬到 device
2. `self.model.set_train_mode()`
3. `optimizer.zero_grad()`
4. 收集 batch 统计 `_collect_batch_stats(...)`
5. 前向：
   - `new_probs, value_logits = self.model(...)`
6. 计算损失：
   - `total_loss, info = self._compute_loss(...)`
7. 记录 `new_prob_min/max` 与非有限值状态
8. `total_loss.backward()`
9. 梯度裁剪
10. 更新参数 `optimizer.step()`
11. 记录参数 / 梯度稳定性
12. `train_step += 1`
13. `_maybe_report(...)`

### 印象
`learn()` 结构清楚，属于标准 PPO update 流程，但加了比较完整的稳定性监控。

---

## 5. `_compute_loss()`
这是 PPO 的核心。

## 5.1 value loss
```python
value_loss = self.hl_gauss(value_logits, td_return).mean()
```

也就是：
- critic 学习目标是 `td_return`
- 但通过 HL-Gauss 分布式交叉熵来拟合

## 5.2 entropy
```python
entropy_loss = -(new_probs * log(new_probs)).sum(-1).mean()
```

是标准策略熵。

## 5.3 importance ratio
首先取出当前策略对旧动作的概率：

```python
new_prob = gather(new_probs, old_action)
```

然后在 log 空间里算比值：

```python
ratio = exp(log(new_prob) - log(old_prob))
```

再做硬裁剪：

```python
ratio = ratio.clamp(0.0, 3.0)
```

### 含义
相比直接除法：
- log-space 更稳定
- ratio 上限 3.0 可抑制梯度爆炸

## 5.4 advantage normalization
若 `ADV_NORM=True`：

```python
adv = (adv - mean) / std.clamp_min(1e-7)
```

## 5.5 PPO clip objective
先算：
- `surr1 = ratio * adv`
- `surr2 = ratio.clamp(clip_low, clip_high) * adv`

若 `dual_clip > 0`：
- 对负 advantage 采用 dual-clip 修正

最后：
- `policy_loss = clipped_obj.mean()`

### 含义
这说明当前实现不是最基础的 PPO，而是带 dual-clip 的版本。

## 5.6 total loss
最终总损失：

\[
L = policy\_loss + vf\_coef \cdot value\_loss - entropy\_coef \cdot entropy
\]

## 5.7 额外统计
返回的 `info` 包括：
- `value_loss`
- `policy_loss`
- `entropy`
- `clipfrac`
- `adv_mean`
- `adv_std`
- `td_return_mean`

### 印象
这一块非常标准而完整，明显是在 baseline PPO 上做了稳健性增强，而不是换算法范式。

---

## 6. 监控与稳定性检查
当前算法主体很重视训练稳定性。

### 6.1 `_collect_batch_stats()`
统计：
- reward 均值
- advantage 均值 / 方差 / min / max
- td_return 均值 / min / max
- old_prob min / max
- batch 中非有限值数量

### 6.2 `_has_non_finite_grad()` / `_has_non_finite_param()`
分别检查梯度和参数是否出现非有限值。

### 6.3 `_warn_if_unstable()`
若发现：
- batch 非有限值
- loss 非有限
- grad 非有限
- param 非有限
- grad_norm 非有限

则通过 logger 打印训练不稳定警告。

### 6.4 `_maybe_report()`
每隔 `LOG_INTERVAL=60s`：
- 汇总当前 loss / clipfrac / adv / td_return / prob / grad 等指标
- 写日志
- 推送 monitor

### 印象
当前实现对稳定性问题的监控已经比较完善，这也是 PPO 工程实现里很重要的一部分。

---

## 六、当前实现的整体结构图（概念）

可以把当前算法结构理解成：

1. `obs.py` 输出三路输入：
   - `scalar`
   - `local_map`
   - `global_map`

2. `model.py` 三路分别编码：
   - `SimbaEncoder`
   - `ConvNeXtEncoder`
   - `ConvNeXtEncoder`

3. 融合后进入共享主干：
   - `SimbaEncoder`

4. 分成两个头：
   - policy head → logits → mask → probs
   - value head → distributional logits

5. `algorithm.py` 用 PPO 目标训练：
   - clipped policy objective
   - entropy regularization
   - HL-Gauss value loss

6. `objectives.py` 提供分布式 value 目标

---

## 七、与设计清单的对应情况

## 1. 已落实的部分
- PPO 主体
- 结构化向量编码模块
- 局部地图卷积编码模块
- 多路特征融合模块
- actor 输出头
- critic 输出头
- legal action mask
- actor / critic 共享主干
- 分布式 value 预测

## 2. 比设计稿更进一步的部分
- 已经把全局图分支纳入主输入
- 价值函数已采用 distributional value + HL-Gauss
- 训练稳定性增强做得较完整（dual-clip、ratio hard clamp、non-finite monitor）

## 3. 尚未进入当前实现的部分
- 实体 token 编码
- attention / gating 融合
- 时序模块
- 更复杂的对象关系建模

### 印象
整体上，当前实现属于：

> 主结构比设计稿更实、更完整，但仍然保持了“先稳定主干，再考虑高级增强”的风格。

---

## 八、我对当前实现的印象记录

### 1. 当前实现是“PPO 主体稳住，输入结构做升级”
没有去发明新算法，而是：
- 让输入分支更贴近任务结构
- 让 critic 表达更强
- 让训练数值更稳

### 2. 三分支输入是当前结构核心
当前网络真正的结构特征，不是 PPO，而是：
- scalar / local / global 三分支编码再融合

### 3. distributional critic 是最显著的增强点
`HLGaussLoss + value_bins` 让这套结构和简单 baseline 形成了明显区别。

### 4. 融合策略依然克制
虽然输入分支更多了，但融合方式仍然是最稳的拼接 + 共享编码器，没有过度复杂化。

### 5. 工程味很强
- `channels_last`
- non-finite 检查
- monitor 数据
- dual-clip 和 hard clamp

都说明当前实现不是“概念验证”，而是明显朝可训练、可监控、可迭代的工程版本在写。

---

## 九、一句话总结
当前 `algorithm_structure` 已实现为一套 **三分支输入的 PPO ActorCritic 框架**：

- `scalar + local_map + global_map` 三路编码
- 共享融合主干
- `policy` 与 `distributional value` 双头输出
- PPO + dual-clip + advantage normalization + entropy regularization
- critic 使用 `HLGaussLoss` 进行分布式价值学习

它的核心特点是：

> 在保持 PPO 主体稳定的前提下，把网络结构做成贴合当前任务空间结构的多分支输入模型，并用更稳健的 critic 与训练细节提升训练质量。
