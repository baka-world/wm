# PR #29：冻结 V4 的预测跨度 × 场景泛化与受限候选决策价值实验

日期：2026-11-12
状态：**代码、冻结预注册、CPU 真仿真、读档复核全部完成。阶段裁决为“条件正信号、泛化未确立”。**

## 1. 方法来源、研究范围和严格边界

**没有训练模型，没有更换网络、调参或修改场景结果后重新抽样。** 本 PR 从 Draft PR #28 的 HEAD `c266ee1bc4d9e40fd1d4dd0ffc3b8662882451d2` 继续。

- 动态业务场景定义借鉴 Reijnen et al. (2026), *Job shop scheduling benchmark: environments and instances for learning and non-learning methods*, DOI 10.1007/s10472-026-10007-3；[TU/e 开源基准 (MIT)](https://github.com/ai-for-decision-making-tue/Job_Shop_Scheduling_Benchmark_Environments_and_Instances)。本项目只借鉴工作量分层与标准化对照，并未复制其数据集或网络。
- 加工时长不确定性与服务期限借鉴 Flores-Gómez & Dauzère-Pérès (2026), *Makespan service level for the flexible job-shop scheduling problem under machine-related uncertainty*, DOI 10.1007/s10479-026-07197-8。服务过程仍保持仓库原本的 keyed duration-jitter，未增添新随机机制。
- 不确定性传播的有限粒子方法参照 Chua et al. (2018) PETS；这里只使用已存在的 RHWM 三粒子、合法业务状态转移，不复现 PETS 连续控制 CEM 等训练/规划模块。
- 下游受限候选集 regret 的评价思想参考 Decision-Focused Learning (Wilder et al., 2019, arXiv:1809.05504)；**没有复现其 decision-focused 训练损失**。

此阶段评估的是**冻结 V4 在新合成负载参数下的场景级预测与首步选案表现**，不是企业实际泛化，不是未知紧急到达，不是多时刻在线滚动调度，不允许宣称“新算法”。

## 2. 实验前冻结的协议

[\`pr29_protocol.json\`](../experiments/world_model_integration/horizon_generalization/pr29_protocol.json) 在任何实验运行前提交：**4 个场景层 × 每层 8 组全新独立 seed = 32 组**。每组真实运行原始 BusinessSimulator，tick0 第一或第二套**不同合法 CP-SAT 方案**作为唯一干预，tick1 起统一采用原版 BusinessCPSAT 第一方案，在每条事实分支自己的公开状态上重排。所有任务/阶段服务时长仍按同 seed/job/phase/station 的 keyed 随机数产生，保留原运行规则和安全检查，所有场景开始前固定参数：

| 场景层 | Jobs | Robots | 共享资源 A 容量 | 额外服务 ticks | Jitter | 截止间隔 | Release |
|---|---:|---:|---:|---:|---:|---:|---|
| light | 2 | 2 | 2 | 0 | 1 | +60 | 全部 t0 |
| medium | 3 | 2 | 2 | 1 | 2 | +55 | 全部 t0 |
| heavy_new_seed | 5 | 2 | 1 | 1 | 2 | +48 | 全部 t0 |
| heavy_service_shift | 5 | 2 | 1 | 3 | 4 | +48 | 全部 t0 |

这些数字只是合成实验设定，不是企业的实测业务量；`heavy_service_shift` 属于**受控参数变化测试**，不能从中推断真实生产分布外泛化。

固定预测：原版 Rule 与 **冻结 V4，检查点 SHA256 `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c`**。每种模型分别在 H5/H10/H20，3 粒子，候选首步动作一/二上预测。续调度仅根据各粒子**当前公开预测状态**调用同一个 BusinessCPSAT 规则，不使用事实未来。模型评分在所有层完全固定为**选期望完成样本工序数较多的方案；期望相同选择原 CP-SAT 第一方案**，没有回头调权重。

评估：事实终局 makespan、逾期、候选 regret，模型的预期完成进展方向和未来完整状态预测误差；按**场景 seed**聚合，粒子不是独立样本。对已经完成而不再有 h 时刻公开观察的场景，不编造终止状态来评估；明确报 coverage。CPU 预算：总进程 ≤480 秒，*仅预测调用*的单个候选对 ≤0.5 秒；所有新训练、GPU=0。

### 受限候选集后悔值（不是全局最优）

当两条完整事实轨迹均完成时：
```
R_model(h) = makespan(selected_by_model_at_h)
             - min(makespan(CP_SAT_proposal_0),makespan(CP_SAT_proposal_1))
```
在线选择过程不访问任一事实 makespan。两条事实结果仅由离线评估器用于报告后悔值。若存在删失，不将最大 tick 当成真实完成时间。与 PR27 的完整在线世界模型策略实验性质不同。

## 3. 已完成的事实实验

[CPU Actions #43138824417](https://github.com/lieuyimo/wm/actions/runs/43138824417) **SUCCESS**；[完整按场景结果、测试及 SHA256 artifact #13908108784](https://github.com/lieuyimo/wm/actions/runs/43138824417/artifacts/13908108784)。原始 32 组、64 条真正执行的业务轨迹；每组前两套 CP-SAT 候选均存在且可以执行，所有相关测试 5+3 项通过，主 CPU 计算约 **21.014 秒**，0 GPU，0 新训练。

下表为**每组等权，真实最终完成时间（ticks）的受限候选集平均 regret，越低越好**：

| 8 组独立场景 | CP-SAT 第一方案 | Rule-H20 | V4-H5 | V4-H10 | V4-H20 | 两方案实际 makespan 有差异 |
|---|---:|---:|---:|---:|---:|---:|
| light | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 8/8，原 CP-SAT 全部选得更好 |
| medium | 2.125 | 2.125 | 2.125 | 2.125 | **0.750** | 6/8 |
| heavy_new_seed | 1.500 | 1.500 | 1.500 | 1.500 | 1.500 | 8/8 |
| heavy_service_shift | 0.625 | 0.625 | 0.625 | 0.625 | 0.625 | 3/8 |

**最重要的真实结果：仅在 medium 层，V4-H20 将平均候选 regret 由 2.125 降至 0.750 ticks；其它层没有选案收益。** 这不能解释为“模型越长预测越准确”或“跨场景稳定泛化”。

### 原始结果的独立只读核对

[CPU Actions #43139228253](https://github.com/lieuyimo/wm/actions/runs/43139228253) **SUCCESS**。在计算前先从原始 artifact #13908108784 恢复 `generalization.json`，用同包 `evidence_sha256.txt` 校验，再只读计算选择正确性、配对改善、描述性场景级 bootstrap 区间与 horizon coverage；**0 新轨迹，0 新模型预测/训练**。[只读审计 artifact #13908867189](https://github.com/lieuyimo/wm/actions/runs/43139228253/artifacts/13908867189)。

- **medium H20**：8/8 场景实际选择了第二候选；对比原 CP-SAT 4 胜、2 平、2 负。相对原策略的完成时间差均值 **−1.375 tick**（相对改进），固定 seed 场景级 percentile bootstrap 95% 区间 **[−3.625, +0.875] ticks**，跨过 0。**结果只能支持“小样本条件正信号”，不支持稳定优越或显著泛化的结论**。
- **heavy_new_seed**：8/8 组两个方案最终 makespan 不同，但 H20 V4 **0/8 改变原 CP-SAT 第一方案**；它的阶段完成进展排序在这8组对真实 makespan 只 4/8 方向一致。这是和 PR28 的 H20 5/5 阶段进展、3/4 终局方向信号的明确差异。
- **heavy_service_shift**：3/8 组最终 makespan 有差异；V4 H20 **0/8 改变第一方案**，其中真实最优方向对了 1/3，不构成迁移提升。
- **light**：实际 8/8 两方案完成时间都不同，但 CP-SAT 第一方案在 8 组中都是最优，世界模型无可改进余地；不要写成“因无敏感性所以无收益”。
- 本轮所有层的候选对 H5/H10/H20 **预测调用 CPU 时间均未超过预设 0.5 秒**，但这是**不包含模型构建/检查点加载与完整业务策略其他成本**的调用时间；不得宣称已通过完整在线服务延迟预算。

### 完整状态预测指标：不能跨 horizon 直接比较成“越长越准”

冻结 V4 不同 horizon 的`sample_status_error`（场景均值，误差越低越好）：

| 条件 | H5 | H10 | H20 |
|---|---:|---:|---:|
| light | 0.15625 | 0.31771 | 0.04167 |
| medium | 0.16667 | 0.05556 | 0.13750 |
| heavy_new_seed | 0.08750 | 0.07500 | 0.00000 |
| heavy_service_shift | 0.06667 | 0.00000 | 0.17014 |

- **轻载 H20 只有 11/16 个候选分支在该时刻仍有可观察真实状态**；H5/H10 和其他场景多数为 16/16。轻载 H20 较低状态错误率不能直接被称作从 H10 改善。
- H5/H10/H20 评价的是**不同真实未来状态**；误差值不相同不能直接用于“预测跨度越长越准确”的结论。
- 同 horizon、同真值下比较 Rule 与 V4 可以衡量是否有已学习动力学预测优势；并不意味着这种优势能够改变有收益的调度动作。

## 4. 技术说明及当前漏洞防护

- 原 PR28 的 `run_factual` 已通过新增 `public_observations` 只读返回其本来记录的事实公开时序。未修改业务模拟器状态、未把真未来传给模型；仅在评估器中使用它与预测时刻的同 tick 状态比较。
- 未新增设备故障、AGV 私有随机异常或突然到达任务；没有重复使用 PR28、PR24～#27 的 seed 分组，也没有改 V4 checkpoint。
- 允许任一候选对提前完成后，状态误差的该候选对应 horizon **不入分母**；决策 regret 仍使用实际完成的完整轨迹。这造成各 horizon 有不同覆盖率，是解释 H20 状态误差时必须披露的限制。
- 当前选择规则是**工序进展最大化的文献启发工程实现**，不是原 PETS 或 Decision-Focused Learning 论文的完整复现，不得作为独创排序算法。
- 所有统计区间都只是 8 组小样本 descriptive interval，尤其 medium H20 的区间跨 0；原始事实只支持内部控制场景的初步适用范围研究。

## 5. 阶段结论与后续研究门槛

**代码、离线评价 GO；V4-H20 已实现跨情景泛化或在线稳定收益，NO-GO。**

后续重点不是继续增加时长、换评分刷本批场景：需要参考已有模型预测控制与动态调度工作，在**新的事先冻结的高负载/服务变化场景**上进行多时刻在线选案；必须与 CP-SAT 和 Rule 同成本约束比较，报告实际 late jobs、完成时间、资源等待及模型初始化/单次评分全成本，尤其检验当前高负载的无增益能否被明确解释。未经独立证据，不晋升 H20 世界模型为默认调度策略。

所需正式论文口径：`在部分中负载合成场景中，较长跨度的概率世界模型对候选方案具有初步区分能力，但在更高负载和服务参数变化场景下未形成稳定调度收益。`
