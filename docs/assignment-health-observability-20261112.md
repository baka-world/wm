# 解决方案尝试：assignment-specific 特征与离线候选重排（本地，未提交）

日期：2026-11-12  
性质：在 PR #16 的 48 个冻结决策/174 个候选上做离线重排实验；不新增仿真、不训练大模型、不修改仓库。  
数据：本地 feature records（由 PR #16 raw + PR #17 fault 标签 + MovingAI 地图 BFS 生成；原始数据未提交）。

## 0.1 与当前 V4-H20 进展的关系

观测到仓库 #28–#37 的最新进展（2026-11-12）：

- #31 后训练 V4 的独立 Brier 0.0673→0.0493，但 H20 首步候选 regret 未变；
- #32/#34 独立在线后训练没有稳定调度增益；#34 发现大量评分打平（shift 原 V4 14/44、Scratch 19/44）、原 V4 改选 4 次、Scratch 0 次；
- #33/#37 发现并修复“最后候选预测后未检查 0.5s”预算缺陷；#37 在新 24 组场景中，原 V4-H20 medium 47.5 vs CP-SAT 50.0、heavy 69.833 vs 74.5，但 shift 无收益，后训练 Warmstart/Scratch 无稳定优势；
- 因此当前项目结论是“原 V4-H20 有局部高负载收益、后训练不转化、预算/评分打平是瓶颈”。

本文的 assignment/health 分析是正交补充：

- 评分打平/候选特征坍缩与 #34 的现象一致；
- station health/telemetry 是 #28–#37 尚未使用的另一条轴；
- 本文的 187.3 是 PR16 旧候选集 + 合成精确 health 的离线/离线重放上界，不能与 #37 的 V4-H20 结果直接比较，也不代表 V4 集成。

## 1. 目的

上一轮归因发现：PR #16 的并列候选 `components` 完全相同、assignment 不同；候选评分特征没有分配关系。  
本实验直接验证：加入 **per-robot assignment-specific 距离/站点特征**，能否打破并列、降低后悔。

## 2. 方法

- 每个候选计划展开为 3 个机器人槽位（R1/R2/R3），每槽位包含：
  - 是否分配
  - 机器人到 pickup 的 BFS 距离、pickup 到 dropoff 的 BFS 距离、总距离
  - pickup/dropoff 可用性、容量、waiting/approaching 数、资源占用
  - 任务 release/stage/服务先验
- 计划级：sum/max/load imbalance、目标重复、不可用目标数、机器人间距离。
- 对照特征集：
  - `score_only`：PR #16 的 3 个 Structured score 分量（应复现 82.333）。
  - `drel`：per-robot assigned/d_total + plan sum/max/load（9 维，不含模型 score）。
  - `dscore`：`drel` + 3 个 Structured score（12 维）。
  - `dplus`：额外 per-robot d_pickup/d_drop（18 维）。
- 学习器：pairwise logistic regression（L2=1.0，lr=0.05，2000 epochs），按决策内 candidate pair 训练；用 leave-one-(layout,seed)-out。
- 评估：held-out 决策选 `argmax w·x`；regret = 所选真实 flow − 该决策最小 flow。真实 flow 只用于训练/评估标签，不进入特征。

## 3. 主结果

| 方法/设置 | 总后悔 | 命中 | fault 后悔 | nonfault 后悔 |
|---|---:|---:|---:|---:|
| Structured-5（基线） | 82.333 | 22/48 | 62.333 | 20.000 |
| `drel` cluster CV | **53.333** | 25/48 | **30.000** | 23.333 |
| `dscore` cluster CV | **50.500** | **28/48** | **30.000** | 20.500 |
| `dplus` cluster CV | 56.500 | 23/48 | 29.167 | 27.333 |
| shuffle-label control (`dscore`) | 90.333 | 16/48 | 48.333 | 42.000 |

- `dscore` 把总后悔从 82.333 降到 **50.500**（−31.833，−38.7%）；命中从 22/48 提到 28/48。
- fault 后悔从 62.333 降到 **30.000**（−32.333，−51.9%）；nonfault 从 20.000 到 20.500（基本不变）。
- `drel` 不含模型 score 也有 53.333，说明**per-robot 分配特征**是主要驱动，不是模型 score。
- shuffle 控制 90.333，差于基线；说明真实特征-结果关系存在，不是随机拟合。
- `dplus` 更差，提示增加过多 per-robot 特征会过拟合。

## 4. 分地图行为（关键）

| 地图 | 方法 | 总后悔 | fault | nonfault | 命中 |
|---|---|---:|---:|---:|---:|
| empty8 | Structured | 40.667 | 32.333 | 8.333 | 13/24 |
| empty8 | `drel` | 14.833 | **0.000** | 14.833 | 14/24 |
| empty8 | `dscore` | 12.000 | **0.000** | 12.000 | 17/24 |
| random32 | Structured | 41.667 | 30.000 | 11.667 | 9/24 |
| random32 | `drel` | 38.500 | 30.000 | 8.500 | 11/24 |
| random32 | `dscore` | 38.500 | 30.000 | 8.500 | 11/24 |

- **empty8 的 4 个故障状态全部被解决**（fault 32.333→0），但 nonfault 从 8.333 升到 12.0/14.833；有代价。
- **random32 的 4 个故障状态没有解决**（fault 30.0→30.0）；nonfault 11.667→8.5 有改善。
- 这说明：assignment 特征能利用 empty8 的几何/分配规律；random32 的故障 oracle 需要牺牲距离/避开 B0，单靠距离分配特征看不到 outage。

## 5. 泛化/泄漏检查

### 5.1 `nonfault -> fault`

只用 40 个 nonfault 决策训练，测试 8 个 fault 决策：

- `drel`/`dscore`：fault 后悔 **30.000**，命中 **4/8**。
- 4 个 empty8 故障全部命中；4 个 random32 故障全部失败（chosen 0，oracle 2，regret 6.0–9.5）。

这非常重要：

- empty8 的故障选择可以**在没有故障标签的情况下**由 per-robot 距离/分配特征选出；不是纯记忆 fault 标签。
- random32 的故障需要 outage 信息；距离特征反而选择 plan 0（更短总距离），而 oracle plan 2 牺牲距离避开 B0。**这正是可观测性限制的体现。**

### 5.2 `fault -> nonfault`

只用 8 个 fault 决策训练，测试 40 个 nonfault：

- `drel` 44.833；`dscore` 43.833；远差于 Structured 20.0。  
- 说明不能只用 fault 数据训练；需要 all-regime + fault augmentation，避免过拟合。

### 5.3 shuffle control

`dscore` 的 cluster CV 在训练标签随机打乱后总后悔 90.333，差于基线；真实模型 50.500。  
=> 提升不是标签泄漏/评估偶然。

### 5.4 学到的权重（nonfault -> fault，`dscore`）

```text
R1_d_total -0.775, R2_d_total +0.504, R3_d_total -0.209,
sum_d_total -0.612, load_imbalance +1.768,
score0 -0.697, score1 -0.760, score2 -0.252
```

- 模型学到了“偏好 R2 承担更多距离 / 惩罚 R1 距离 / 偏好更大 load imbalance”的组合；这在 empty8 故障上恰好对应 oracle，但在 random32 上失败。
- `load_imbalance` 的正权重很可能是 empty8 分配模式的 spurious correlation；需要正则化/特征选择/cross-map 验证，不能直接部署。

## 6. 结论

**assignment-specific per-robot 特征是有效的**：

- 它打破了 PR #16 的特征坍缩；cluster CV 总后悔 82.333→50.500，fault 62.333→30.000。
- 在 empty8，仅用 nonfault 训练就能解决 4/8 故障状态，说明距离/分配关系确实携带可学习的决策信号。
- 但 random32 的 4 个故障状态仍为 30.0，说明剩余部分是**不可观测的 outage/queue 信息**，不是单纯特征问题；而且简单线性 ranker 会学到 spurious load-balance 权重，nonfault 也有代价。

因此下一步不是“继续加参数”，而是：

1. **数据/训练平衡**：加入更多、位置/时间变化的故障样本；all-regime 混合训练，避免 fault-only 过拟合；对 nonfault 代价做约束。
2. **稳健关系特征**：per-robot assignment distance、相对距离（相对该机器人最优候选）、站点队列/容量、路径替代方案；去掉/正则化 raw `load_imbalance` 这类易 spurious 的聚合特征。
3. **轻量 relational scorer**：保留 per-robot/per-task 身份，先线性/attention，不直接上大 GNN。
4. **在线验证**：在 PR #27 式 paired online protocol 上比较，而不是只看离线 candidate replay；同时报告 nonfault 和 fault 分组。
5. **可观测性**：random32 的剩余 30.0 需要设备健康/故障前信号；没有它，距离特征只能猜测。

**关键限制**：本实验的重排器使用 PR #16 对每个候选的 counterfactual 真实 flow 作为训练标签；生产环境只能观察被选候选的结果，不能同时观察所有候选。因此 50.5 是“离线候选重放/上界诊断”，不是可部署策略。要部署，需要训练一个从可观察 assignment 特征预测 flow/regret 的世界模型（可在模拟器中用 counterfactual 标签训练），再用 PR #27 式在线配对验证。另一个限制是 empty8 的 fault 增益可能利用了几何与故障的相关性；random32 的 30.0 残差和非故障代价说明简单线性 ranker 仍会过拟合/偏移（例如 `load_imbalance` 的 spurious 正权重）。


## 6.2 第二轮：robust assignment features + fault data

在 scale-0（不使用 B0 手工 outage 特征）下，使用 **13 维** 特征：

- per-robot: `assigned`, `d_total`
- plan: `max_d_total`, `n_unavailable_targets`, `n_target_duplicates`, `min_pairwise_robot_distance`
- 原有 3 个 Structured score

cluster CV 结果：

| 设置 | 总后悔 | 命中 | fault | nonfault |
|---|---:|---:|---:|---:|
| Structured 基线 | 82.333 | 22/48 | 62.333 | 20.000 |
| robust13 scale0 | **33.333** | **29/48** | **0.000** | 33.333 |
| robust13 shuffle | 90.833 | 17/48 | 62.167 | 28.667 |

- **fault 62.333→0**，total 82.333→33.333；但 **nonfault 20.0→33.333**，正常场景有明显代价。
- 手工 B0 wait 特征反而更差：scale=1.0 时 total 39.500、fault 7.333、non 32.167；B0 特征可以删除。
- **fault 数据是必要的**：`nonfault -> fault` 训练时 fault=62.167、命中 0/8；没有 fault 样本，assignment 特征本身不能解决 fault。
- **跨地图**：`empty8 -> random32` fault=0、non=19.333；`random32 -> empty8` fault=32.333、non=24.667；单向泛化，故障模式仍依赖具体地图/几何。
- **cost-sensitive**：normal weight 1/2/4/8 → total 33.333/54.167/85.500/89.833，fault 0/23.0/55.167/61.167，non 33.333/31.167/30.333/28.667。提高正常权重几乎不能把 nonfault 拉回 20，反而迅速丢失 fault 收益；说明单一 B0/tick10 故障模式被模型学成了“避免某个分配模式”，正常场景也被误伤。

第二轮结论：

- assignment-specific 特征确实能解决 fault 的**可学习部分**；fault 样本是前提，手工 B0 特征不是。
- 当前 fault 数据只有 8 个、同一位置/时间模式，模型过拟合到该模式，normal 代价高；需要**多样故障数据**（不同站点/时刻/时长/前兆）和 risk-aware 目标，而不是继续调权重。
- 离线 ranker 仍使用 counterfactual 标签；下一步需要 flow/regret 世界模型 + 在线 paired 验证。


## 6.3 多样故障数据与 station-conditioned 策略（最新一轮）

用 PR16 的 `BusinessSimulator` 重新生成多样故障数据：8 个 base case（empty8/random32 × 4 seeds）× 31 个 variant（5 个故障站点 × 3 个 onset × 2 个 duration，加 1 个 normal）× 4 个候选 plan = **992 条记录 / 248 个决策**；全部 completed、0 安全违规。

### 可达性能层级（fault 240 个 variant）

| 策略 | fault 总后悔 | fault 平均后悔 | 命中 | fault 平均 flow | normal 平均 flow |
|---|---:|---:|---:|---:|---:|
| Structured-5 | 376.333 | 1.568 | 71/240 | 44.358 | 41.667 |
| best fixed plan（每 base 一个，oracle） | 233.000 | 0.971 | 179/240 | 43.760 | 40.917 |
| station lookup（按站点选训练均值最优 plan） | 191.333 | 0.797 | 134/240 | 43.587 | 41.792 |
| **per-station ridge（assignment 特征 + 站点 gate）** | **187.333** | **0.781** | **140/240** | **43.570** | **41.562** |
| best station oracle（每 base/站点最优） | 112.833 | 0.470 | 193/240 | 43.260 | 40.917 |
| full oracle（每 variant 最优） | 0 | 0 | 240/240 | 42.79 | — |

- 全局 logistic ranker（risk-neutral/health 特征）只有 323.8/308.7 总后悔，**不如 per-station ridge 和 station lookup**；说明“按站点条件化”比把所有故障混进一个线性模型更有效。
- **per-station ridge**：每个站点单独训练一个 ridge flow 预测器（13 维 assignment 特征），用站点健康信号选模型；fault 376.3→**187.3**，normal 41.667→**41.562**，没有明显正常代价。
- **station lookup** 191.3 已经很强；ridge 只小幅超过；说明主要收益来自**站点级健康/条件化**，不是复杂模型。
- best station oracle 112.8 是“已知 base 的站点最优 plan”的上界；full oracle 0 是“已知每个 variant 精确 outage 时间”的上界。当前 187.3 与 112.8 的差距来自 base 间泛化误差；112.8→0 的差距来自 start/duration 等未观测的 timing 信息。

### 解释

- 站点级 health/risk gate 是正确方向：它让模型按站点选择不同 plan，同时 normal 模型保持正常 plan；这避免了上一轮“全局避开 B0 导致 normal 受损”的问题。
- 剩余提升需要：
  1. 更强的 per-station 模型/更多特征（相对距离、队列、容量、路径替代），缩小 187.3→112.8；
  2. 可观测的 timing/outage 信息（设备告警/维护计划/历史故障窗口），缩小 112.8→0；
  3. 在线 PR27 式验证；当前仍是离线 counterfactual 标签。
- health flag 在本轮是**合成的站点 outage 信号**（oracle）；真实部署需要设备健康/故障历史遥测。如果没有它，risk-neutral 的可达上界是 best fixed 233，而不是 187。


## 6.4 可观测 health 信号敏感性

per-station ridge 在 perfect station signal 下 fault mean 0.781（total 187.4）；把 signal 改为“检测率 detect / 误报率 fp”：

| detect | fp | fault 平均后悔 | fault 总后悔（240） | fault 命中 | normal 平均 flow | normal 总后悔（8） |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 0.00 | 0.781 | 187.4 | 140/240 | 0.646 | 5.17 |
| 0.75 | 0.00 | 0.983 | 235.9 | 123/240 | 0.646 | 5.17 |
| 0.50 | 0.00 | 1.180 | 283.2 | 106/240 | 0.646 | 5.17 |
| 0.25 | 0.00 | 1.372 | 329.3 | 89/240 | 0.646 | 5.17 |
| 0.00 | 0.00 | 1.568 | 376.3（structured fallback） | 71/240 | 0.646 | 5.17 |
| 1.00 | 0.10 | 0.781 | 187.4 | 140/240 | 0.666 | 5.33 |
| 1.00 | 0.25 | 0.781 | 187.4 | 140/240 | 0.722 | 5.78 |

- 站点 health 检测率是主要因素：perfect 187.4 → 75% 235.9 → 50% 283.2 → no signal 376.3。
- 误报对 normal 的影响较小：fp 10% 使 normal 平均 flow +0.02，25% +0.076；但 fault gain 不受影响（因为 fault 用真站点信号）。
- 结论：**要保留大部分收益，站点级 health/risk 检测率需要大约 ≥70–75%**；误报可以容忍，但仍需校准。

## 6.5 特征丰富度

per-station ridge 的 13 维特征（per-robot assigned/d_total + plan max/n_unavailable/n_duplicates/min_pairwise + 3 score）在 lam=1 时最好：fault total 187.3、normal 5.17。加入 per-robot d_pickup/d_drop 或 queue/capacity 特征后：

- kind2/lam1: fault 214.0、normal 5.5（变差）
- kind3/lam1: fault 214.0、normal 5.5（变差）
- kind3/lam10: fault 187.3、normal 7.0（fault 相同但 normal 变差）

说明当前 168 个 per-station 训练决策下，更多原始特征只会过拟合；下一步要靠更好的特征设计/更多数据，而不是堆特征。


## 6.6 historical prior alone 不够

用一个非均匀站点 prior（A0 0.25 / A1 0.15 / B0 0.40 / B1 0.15 / C 0.05）评估“没有 per-decision health、只有历史站点风险”的策略：

| 策略 | 加权 fault 平均后悔 | 说明 |
|---|---:|---|
| Structured-5 | 2.106 | 基线 |
| best fixed plan per base（prior-weighted oracle） | 1.200 | 用测试 base 的真实 flow 选一个固定 plan；不可部署 |
| **deployable prior lookup**（训练集 prior 均值选 plan） | **2.373** | 比 Structured 还差；全局 prior plan 不可泛化 |
| per-station ridge + exact health | **0.820** | 需要 per-station health 信号 |
| best station oracle | 0.470 | 每 base/站点最优；不可部署 |

- 只有历史站点频率、没有 per-decision/per-station health 时，最好的固定 plan（oracle）是 1.2，但可部署的 prior lookup 是 2.373，**比 Structured 2.106 还差**。
- 这说明：**历史先验不能替代站点级 health 信号**；需要 per-decision 的设备健康/故障历史/告警/队列状态，而不是一个全局站点频率。
- exact health 的 0.820 是当前可达到的方向；但 health flag 在本轮是合成的，真实部署需要遥测。


## 6.7 health 信号延迟（在线实验）

在 `BusinessSimulator` 中真实执行：tick < latency 时策略不派单，tick == latency 时使用 per-station ridge 选出的 plan，之后 nearest；health 为精确站点信号。

| latency (ticks) | fault 平均后悔 | fault 总后悔 | fault 命中 | fault 平均 flow | normal 平均 flow |
|---:|---:|---:|---:|---:|---:|
| 0 | **0.781** | **187.3** | **140/240** | **43.57** | **41.562** |
| 2 | 5.381 | 1291.5 | 1/240 | 48.171 | 45.917 |
| 5 | 9.251 | 2220.2 | 1/240 | 52.040 | 49.792 |
| 10 | 13.910 | 3338.3 | 0/240 | 56.699 | 54.479 |

- latency=0 与离线 187.3 完全一致，说明离线 plan flow 就是真实 simulator 执行结果。
- **只要延迟 2 tick，等待 health 的代价就吞掉了全部收益**：fault 后悔 0.781→5.381，normal 也 +5.0 flow；延迟 5/10 tick 更差，远不如 Structured。
- 结论：**health 信号必须在首次派单前/当 tick 到达**；迟到信号不能靠“等”来使用。调度器应先派单，不能为了 health 空等。
- 这也说明“可观测性”不仅是检测率，还包括**时序**；离线 187.3 是“立即且准确 health”的上界。


## 6.8 partial commitment / 迟到 health 的尝试

为了不空等 health，尝试 tick 0 先派一个 assignment，health 到达后再派剩余：

| 策略 | latency | fault 平均后悔 | fault 总后悔 | fault 命中 | normal 平均 flow |
|---|---:|---:|---:|---:|---:|
| full plan（immediate health） | 0 | **0.781** | **187.3** | **140/240** | **41.562** |
| structured-first partial | 2 | 1.886 | 452.7 | 40/240 | 42.042 |
| structured-first partial | 5 | 3.480 | 835.2 | 14/240 | 43.479 |
| robust-first partial | 2 | 3.831 | 919.3 | 22/240 | 43.562 |
| robust-first partial | 5 | 5.667 | 1360.0 | 7/240 | 46.000 |
| Structured | 0 | 1.568 | 376.3 | 71/240 | 41.667 |
| wait-all | 2 | 5.381 | 1291.5 | 1/240 | 45.917 |

- tick 0 先派一个 assignment + 迟到 health 的 partial commitment **仍然差于 Structured**：latency 2 最好也只有 452.7，latency 5 835.2；robust-first 更差。
- 迟到 health 无法弥补首个 assignment 的错配；等待更差。
- 只有 **immediate health + full plan**（187.3）明显优于 Structured；否则 Structured/CP-SAT 是更稳的默认策略。
- 结论：**health 必须在首次派单当 tick 到达**；如果不能在 dispatch 前获得，不应等待或部分承诺，直接用默认规则。

## 7. 仓库内证据与本地原始产物

仓库内：

```text
docs/assignment-health-observability-20261112.md
examples/evidence/assignment-health-20261112/
  README.md
  summary.json
  hierarchy.csv
  health-sensor.csv
  health-latency.csv
  partial-commitment.csv
  prior-policy.csv
  SHA256SUMS
```

本地未提交：feature records（86 维）、992 条 diverse records（约 2.5 MB）、分析脚本、原始 PR16/17 artifacts。没有检查点或大训练数据进入 Git。

本报告与精简证据随 PR 提交；未修改模型/训练/评估代码，未合并 main。
