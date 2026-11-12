# 三批独立严格预算验证：原 V4-H20 世界模型在合成多机器人调度中的条件决策价值

日期：2026-11-12。**本文件是先前两批48组只读归档的增量复核，旧结果不修改。**

## 一、真实已执行来源

在初次完成[PR36 + PR37 两源48组分析](h20-strict-budget-independent-cohort-replication.md)后，仓库另有 [PR #38](https://github.com/lieuyimo/wm/pull/38) 完成第三批独立新seed的严格预算在线仿真。直接读取其成功归档，不重抽seed、不改模型、不推理、不训练、不生成新轨迹。

| 独立来源 | 本源正式模型和门禁 | 原始成功运行与SHA检验文件 |
|---|---|---|
| [PR36](https://github.com/lieuyimo/wm/pull/36) | Rule/V4/Scratch progress H20 + 独立累计代价臂；仅取progress臂，候选后置预算检查 | [Actions #43147122344](https://github.com/lieuyimo/wm/actions/runs/43147122344)，artifact [13912451562](https://github.com/lieuyimo/wm/actions/runs/43147122344/artifacts/13912451562) |
| [PR37](https://github.com/lieuyimo/wm/pull/37) | 原V4/Warmstart/Scratch/Rule H20，候选后置及最终核验 + 人工时钟回归 | [Actions #43147634979](https://github.com/lieuyimo/wm/actions/runs/43147634979)，artifact [13911523921](https://github.com/lieuyimo/wm/actions/runs/43147634979/artifacts/13911523921) |
| [PR38](https://github.com/lieuyimo/wm/pull/38) | 原V4/Scratch/Rule H20，候选后置核验和CPU边界检查 | [Actions #43148139899](https://github.com/lieuyimo/wm/actions/runs/43148139899)，artifact [13912587157](https://github.com/lieuyimo/wm/actions/runs/43148139899/artifacts/13912587157) |

**本次源码证据**：[GitHub Actions #43149875212](https://github.com/lieuyimo/wm/actions/runs/43149875212) **SUCCESS**。从上述三个 artifact 恢复和校验所有原 `evidence_sha256.txt` / `SHA256SUMS`，6项内部Schema、独立性、违规决策测试通过；[72个场景的复核JSON及 SHA256 artifact #13913291389](https://github.com/lieuyimo/wm/actions/runs/43149875212/artifacts/13913291389)。**本轮 0 新仿真、0 新模型调用、0 训练、0 GPU**。

## 二、方法与可比性

三个来源各24组独立新seed，每种负载6组，合并为每负载18组，共72组。每组相同：
- 基础业务原始 BusinessSimulator 与 BusinessCPSAT，轮式机器人及样本运输/工作站容量，随机机制是原 keyed 服务 jitter；
- H20，3预测粒子，最多4组CP-SAT候选，**工序完成期望最大化**，平局选择CP-SAT第一提案；
- 原V4权重SHA256 `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c`；Scratch是PR31验证集预选checkpoint；
- 0.5 CPU process second CPU预算，候选预测完成后检查，超时回退CP-SAT第一方案。

**三个来源在决策门禁实现细节上有轻微差异，不冒称同一个版本/单次预注册的72场景随机试验。** 本文校验了三源的按tick记录：**实际CPU超预算却继续执行非首选模型方案 = 0**。预算回退并不能抢占一次不可中断的预测过程，因此不能称为硬实时wall-clock SLA。

场景层参数：
- light：2jobs/2robots/共享容量2，服务附加0，jitter1，deadline +60；
- medium：3jobs/2robots/共享容量2，服务附加1，jitter2，deadline +55；
- heavy：5jobs/2robots/共享容量1，服务附加1，jitter2，deadline +48；
- shift：5jobs/2robots/共享容量1，服务附加3，jitter4，deadline +48。
每组都是合成场景，**不是企业现场实验**，后验汇总只是跨独立复现的一种描述性证据。

## 三、72组真实在线结果：完成时间 tick，越小越好

| 业务条件 | CP-SAT | Rule-H20 | 原V4-H20 | PR31 Scratch-H20 |
|---|---:|---:|---:|---:|
| light（18组） | 19.889 | 19.889 | 19.889 | 19.889 |
| medium（18组） | 49.389 | 49.000 | **46.889** | 46.944 |
| heavy（18组） | 74.722 | 73.667 | **72.111** | 72.778 |
| shift（18组） | **109.667** | **109.667** | 109.778 | 109.889 |

### 真实H20原V4相对于CP-SAT

| 负载 | 完成时间平均差值 tick | 场景级胜/平/负 | 三来源分层bootstrap 95%**描述区间** |
|---|---:|---:|---:|
| light | 0.000 | 0/18/0 | [0, 0] |
| **medium** | **−2.500** | **13/3/2** | **[−3.722, −1.167]** |
| **heavy** | **−2.611** | **12/6/0** | **[−3.667, −1.500]** |
| shift | +0.111 | 2/14/2 | [−0.444, +0.667] |

三个来源在 medium 和 heavy 均有同方向均值：
- medium：PR36 **−1.0**、PR37 **−2.5**、PR38 **−4.0** tick；
- heavy：PR36 **−1.333**、PR37 **−4.667**、PR38 **−1.833** tick。

### 学习V4相对于**同样H20**的规则世界模型

| 负载 | 原V4 − Rule tick | 胜/平/负 | 三来源分层bootstrap 95%**描述区间** |
|---|---:|---:|---:|
| light | 0.000 | 0/18/0 | [0,0] |
| **medium** | **−2.111** | 11/4/3 | **[−3.500, −0.556]** |
| **heavy** | **−1.556** | 9/7/2 | **[−2.778, −0.333]** |
| shift | +0.111 | 2/14/2 | [−0.444, +0.667] |

因此，**同样 H20 预测跨度并不能完全解释 medium/heavy 的收益**，原冻结V4在部分工作负载下确实能提供比纯规则模型更有决策价值的概率动态信息。与此前只有单批/两批数据时相比，这是更可靠的重复方向证据；但仍是合成机制、仅18组每条件、同一架构/小场景，不得外推真实实验室生产。

### 后训练Scratch是否比原V4更好？

| 条件 | Scratch − 原V4 完成tick | 胜/平/负 |
|---|---:|---:|
| light | 0.000 | 0/18/0 |
| medium | +0.056 | 0/17/1 |
| heavy | +0.667 | 2/9/7 |
| shift | +0.111 | 1/15/2 |

与PR31独立事件 Brier改善形成对照：**事件概率更准确不等于直接改善调度动作。** 当前后训练没有稳定提升实际完成时间，不继续无证据追加epoch。

## 四、当前重点及下一项工程开发门槛

1. **世界模型第一优先级**。现有数据支持在中、高负载下有条件使用原V4-H20，保留原Rule-H20和CP-SAT作为明确基线。不要把“预测跨度长”或“CP-SAT本身更好”混入学习模型独立贡献。
2. **泛化边界**。服务时间和抖动参数发生变化时，世界模型优势不稳定。训练下一版本之前应调查如何观测 service duration/health 信号，并以原始可观察事实验证，禁止事后改变仿真机制制造数据优势。
3. **有限预算执行**。三批数据均仅证明超时后安全回退，未解决单个不可中断rollout可能造成的wall-clock响应超限。后续借鉴已公开 Event-Triggered MPC/自适应规划与计算缓存的来源实现，先论证与离散业务/CP-SAT实际相容，再设定新独立测试协议。
4. **原模型可恢复性保持**。所有再次训练必须先捕获commit、pip环境、数据sha及依赖、源代码和CPU/GPU，保存best/latest模型、Adam动量和Python/NumPy/Torch RNG，并实测断点恢复；沿用 [PR31可恢复归档](https://github.com/lieuyimo/wm/releases/tag/untagged-871b60f7bf013400cb45) 做参考。
5. **方法迁移、非造轮子**。长时粒子/CP-SAT调度已经遵循原RHWM及PETS-style MPC，若引入新策略必须说明论文/官方源码/许可证/具体可迁移模块，再实施独立测试。不要用已经看过的72组调阈值后又称为盲测。

**阶段结论：三批合成独立场景稳健支持“原V4-H20在medium/heavy具有额外条件决策信息”；不支持“任意服务变化条件下普遍优于Rule/CP-SAT”或“后训练一定提高业务绩效”。**
