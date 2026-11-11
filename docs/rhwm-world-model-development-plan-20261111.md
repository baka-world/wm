# 本科毕业设计世界模型研发实施方案（RHWM，2026-11-11）

> 适用于《基于世界模型的样本转运调度系统设计与实现》
> 起点：Draft PR #18 `feature/world-model-decision-integration-20261110`，本计划不修改 PR #19，不合并历史分支。
> 本文是**实施计划**，不是已完成的模型实验或新训练结果。执行前先检查 PR #18 HEAD。

## 0. 基本决策与论文问题

**P0 世界模型优先**：设计和实现一个可复用的、动作条件化的未来业务状态预测模型 `P(S_{t+h} | S_t, A_t, π)`；其中 `S_t` 必须是当前真实可观测系统状态，`A_t` 是当前合法候选动作，`π` 是明确的未来续调度策略。输出完整的未来机器人、任务、工作站、共享资源状态及不确定性，而不是只有服务完成概率或动作价值分数。

**P1 是模型辅助派单**：CP-SAT 生成合法候选，世界模型预测其后果，统一评分挑选动作；PIBT/LaCAM3 等路径/冲突模块继续独立负责运动层，安全由规则保证。

**P2 是必要的系统演示**：只做支持 P0/P1 的运行面板、日志和可复现实验。不要将 UI、业务 CRUD、ROS/VLA/人形机器人的底层动作生成挤到 P0 前面。

本方案分两级模型，避免因一次复杂网络研发而耽误毕业答辩：

- **RHWM-v1：已有 V4 + 合法业务状态转移引擎**。完成可调用的完整状态预测和滚动分布接口。这是可交付的首个全系统混合世界模型，但需如实解释“神经学习服务概率、确定性模型推进其他业务状态”。
- **RHWM-v2：关系式多实体编码器 + 事件概率头 + 公共业务状态转移**。重点学习资源竞争与任务上下文对服务事件及等待风险的影响；不强迫神经网络学习容量、运动、工作流等完全已知的规则。旧 V4 是直接神经基线，Rule 是规则基线。

## 1. 现有基础及确切复用点

| 已存在 | 路径 | 本轮动作 |
|---|---|---|
| 富公开观测 | `lab_transport/state_v2.py`、`business_transport/observation.py` | 复用，版本化扩展，不破坏存量训练权重 |
| 当前可观测业务特征 | `business_transport/state_features.py` | 复用 known/value 掩码，添加带关系边的 graph encoder |
| V4 当前服务事件概率模型 | `business_transport/session_dynamics.py`、`experiments/world_model_integration/train_v4.py` | 冻结权重作为基线和首个完整状态预测服务头 |
| 工序、容量及状态跃迁 | `business_transport/dynamics.py`、`business_transport/workflow_transition.py` | 不重写合法性规则，只组合世界模型预测的随机事件 |
| 现有业务仿真和 CP-SAT | `business_transport/simulation.py`、`business_transport/dispatch.py` | 保持真实环境/调度器隔离，做模型推理与闭环对照 |
| 急诊及复检业务 | `experiments/world_model_integration/urgent_workflow_feasibility/core.py` | 测试场景之一，不把复检结果当成预测前已知 |
| 原来的预测稳定性证据 | `docs/pr18-final-mechanism-stop-and-engineering-handoff.md` | 已完成，禁止围绕旧5个案例再微调优化 |

目前 V4 是 56 维查询输入、Deep Sets 上下文、每个服务会话给出下步完成概率；并未单独学习完整系统转移。CPU 锁定依赖 `requirements-paper-reuse.lock` 为 Python3.10、Torch2.5.1+cpu、ortools9.14；原锁里没有 PyG。**不要在原生产依赖中直接升级 Torch**。

## 2. 世界模型状态、动作、输出契约（第1阶段必须提交）

### 2.1 状态与动作

观测 `S_t`：统一使用当前 `BusinessSampleV2 / BusinessStationV2 / RobotV2`（或兼容适配后的 `ObservationV2`），包含：
- 机器人：位置、任务归属、执行阶段、当前可观察服务进度；
- 样本：job/stage、状态、优先级、释放时间、截止时间和已公开后续工序；
- 工作站：服务中、设备公开可用、平均服务时长、服务进度代理；
- 共享资源池：容量、占用、可能等待的已公开请求；
- 图结构：机器人→执行样本、样本→取/放工作站、站点→资源池、机器人→目标站点、可观察的工序前后依赖、相邻通行关系；
- 候选动作 `A_t`：本次合法机器人/样本 assignment batch。将动作作为图里的实体关系和显式条件，而不是仅用于事后评分。

缺失信息必须有 mask。**禁止**输入模拟器的真实剩余时间、未来故障时刻、未来尚未释放的任务、未知复检结果。模型可以明确标为“不预测外生事件”，且以当前已知任务为主评测队列。

### 2.2 输出与约束

核心接口建议（名称可在实现前根据现有代码风格做微调）：

```python
class SystemWorldModel(Protocol):
    def predict_next(self, public_state, legal_assignments, *, sample_seed) -> ForecastStep: ...
    def rollout(self, public_state, first_action, *,
                continuation_policy, horizon_ticks, particles,
                sampling_seed) -> ForecastBatch: ...
```

`ForecastStep` 包含 `predicted_state: ObservationV2`、`service_events: [{query, p_finish}]`、`unknown_exogenous_mask`、`model_version`；`ForecastBatch` 除合法状态粒子外，返回按任务/资源的未来状态摘要（可比较候选的 deadline risk、remaining-time 及 station occupancy；统计项是从同一状态粒子计算，不能混成无定义的“价值网络”）。

纯规则确定转移拥有：路由推进、动作合法性、容量保护、服务后继工序释放、机器人运动/路径冲突。学习组件拥有：条件服务完成事件分布，以及可选的固定续策略下资源等待风险辅助预测。推理时抽样事件并经过合法转移函数生成整个 `S_{t+1}`。

**不允许**以违规预测状态直接交给生产调度，不允许把规则裁剪掉而让网络随意生成机器人位置或任务 ID。

### 2.3 多步语义

主评测 `h∈{1,5,10}`，20 tick 仅为工程观察。首步执行候选动作；未来 `t+1...` 的选择使用同一固定、可公开观测的 P1 continuation policy。开放环和闭环预测必须分开命名，不能拿真实未来计划喂给预测器。外生新增任务/故障不可预知，主要测试集用“当前已知任务的闭合队列”，次要测试明确报告外生事件覆盖限制。

## 3. 分阶段开发安排（总计约4周，预留1周缓冲）

### 阶段 A：2~3天，建立 RHWM-v1 与验收契约

新增/修改建议：
- `business_transport/world_model/contracts.py`：versioned ForecastStep/Batch；
- `business_transport/world_model/hybrid.py`：复用 `WorkflowDynamics` + `SessionRuleDynamics`（V4）和 Rule，提供标准 `predict_next` / `rollout`；
- `experiments/world_model_integration/system_world_model/tests/test_contracts.py`；
- `experiments/world_model_integration/system_world_model/protocol.json`。

交付：从一个观测与不同合法动作出发，产生各自完整的未来任务/机器人/站点状态粒子，并显示每个事件 `p_finish`。冻结 Rule 和 V4 权重；不用 GPU。

门禁：所有输入来源、输出字段和未来续策略可追溯；每步状态必须通过 `check_business_transition` 和 motion/capacity audits；空候选、终止状态、动态 workflow 扩展、服务暂停、共享容量至少有单测。

### 阶段 B：3~4天，真实观测下的全状态预测基线

使用旧仿真器导出的 `t,action,t+1` 公开观测进行 H1/H5/H10 预测；只读或重用已经定义好的场景，不在本阶段训练。记录：
- 服务事件 Brier/ECE；
- 每个任务的 stage/status accuracy、到站与工序完成事件 F1；
- 机器人任务/phase accuracy 与位置错误（运动由规则给出时单独统计）；
- 资源 occupied MAE、站点忙闲和等待人数 MAE；
- 所有粒子的合法转移率、rollout 每5步状态误差、p50/p95 推理时延。

对照 `Rule+WorkflowDynamics` 与 `V4+WorkflowDynamics`，保持相同公开输入和后续策略。成功标准：**两种模型都能可靠地输出完整合法预测状态及同一套指标；不要求 V4 在所有指标优于 Rule**。这样论文已有一个真实工作的世界模型系统内核。

### 阶段 C：5~7天，关系式世界模型 RHWM-v2（唯一一次架构升级）

增加：
- `business_transport/world_model/graph_encoder.py`：从公开状态和候选动作构造异质图（robot/sample/station/resource；公开关系边）；
- `business_transport/world_model/relational_event_model.py`：约64维隐层、2轮 message-passing，输出可观测服务会话的 `p_finish`；设置针对当前等待任务的“未来5步能否开始服务”的辅助概率头。这个辅助标签**只在固定的未来 P1 policy 下有效**；
- `business_transport/world_model/relational_dynamics.py`：概率事件与合法状态跃迁的组合层；
- `experiments/world_model_integration/system_world_model/{generate_data.py,train.py,evaluate.py}`。

优先在与当前 Torch2.5.1 隔离的开发依赖中试装 PyTorch Geometric 的 `HeteroData/HeteroConv`；正式引入前需 CPU 安装及 forward/backward smoke test。若安装/兼容困难，使用现有 PyTorch 的张量索引聚合实现限定两层的关系信息传递，不无限扩展新基础设施。**不要升级全仓 Torch/重构全部依赖。**

`loss = BCE(service_finish) + 0.25*BCE(wait_start_within_5)` 作为初始冻结目标；等待样本未观察到未来5步或外生事件导致标签歧义时只标为 censored/masked，不当成负例。冻结网络规模、损失比例、随机种子、分组/测试协议后，只允许一次预设训练；不通过观察测试集修改模型配置。

### 阶段 D：数据生成与训练（与 C 并行，约3~5天）

建议一次性冻结业务场景 group：
- TRAIN 48 个 group、VALIDATION 12、独立 IID TEST 12；
- OOD-SERVICE 8（服务时长跳出训练区间）、OOD-ROBOT 8（机器人数量/负载超训练范围，必须在场景结构支持下生成）；
- 每 group 两种事先指定的合法行为策略：nearest/现有 P1 urgent-first，总计 **176条合成轨迹**。每条策略轨迹使用真实模拟器执行、公开观测与真实下一状态标注，绝不把模型预测轨迹当事实。
- group 按 scenario seed + 结构标识预先拆分；同一个外生 scenario 的所有行为轨迹只能归属一个 split，验证及独立测试不共享轨迹实体。TEST/OOD 在冻结模型后只运行一次评测。
- 在模型编码前检查当前/未来标签时间戳、任务释放、未知复检与未知未来故障是否泄漏；保留每条记录的 hashes 和生成脚本。

资源硬上限：数据生成不超过60分钟CPU wall，模型训练首选CPU、总计不超过60分钟CPU wall；如 CPU 不够，必须先提交真实成本和理由，再决定是否使用单卡 V100 16GB，初始申请最多 **2 GPU卡小时**。上述都是**预算门槛，不是已测耗时**。

### 阶段 E：4~5天，冻结泛化测试与模型晋级

先锁定 `RHWM-v1`（现有 V4混合）与 `RHWM-v2` 两个模型，再在独立 IID、OOD-SERVICE、OOD-ROBOT 三类场景上分别报告 H1、H5、H10：
- H1：service Brier/ECE、状态变化准确性、状态约束；
- H5/H10：任务阶段、样本状态、站点占用、等待人数、任务交付时间/截止风险（若样本未交付，需明确 censored 处理）；
- 泛化：1~2 vs 3机器人、已见 vs 未见站点服务参数；样本按 group 汇总置信区间，**不要把每一步事件当成独立场景**；
- 性能：4候选×10 tick×8 particles 时模型部分 CPU p95 延迟目标 <500ms，整体含 CP-SAT 决策 p95 目标 <2s（目标而非已有保证）；安全资源/运动违规必须为0。

晋级条件（冻结前写入协议，不得看结果再改）：
1. 约束违规始终为0，模型可端到端运行；
2. IID service Brier 不比旧 V4 恶化超过绝对0.005；
3. 至少一个预定义的系统级指标（资源占用/等待误差或任务阶段事件）相对 V4 在独立测试有可观察改善，且另一个关键指标不出现预定严重回退；
4. CPU 模型时间预算达标，或有清晰可重现降级到 RHWM-v1 的回退。

如果 v2 不达标，**保留可用的 v1**，论文按“关系编码器未带来稳定泛化提升”如实展示。禁止新开无限制 V3/V4/V5 调权工程。这里的训练是**针对完整系统预测的新建模任务**，不是重新追逐旧 `906195` 个案。

### 阶段 F：4~5天，世界模型进入完整决策闭环

保持 CP-SAT 同一个候选列表，比较：
- P0：原始 P1/Rule 调度基线；
- P1：`RHWM-v1` 未来状态辅助调度；
- P2：`RHWM-v2` 未来状态辅助调度（仅达晋级门槛才上）。

固定同样总预测预算、H10、8 particles、候选上限4；使用相同公开状态和相同外生情景 seed 进行配对比较。评分事先规范为“紧急任务完成/逾期 → 常规平均流转 → 资源等待/拥堵”，并把词典序和业务最终指标的差异写清，不把一项改善偷换成另一项。记录 deadline on-time、迟到、完成任务数、mean restricted flow、站点/机器人利用率、时延、回退次数和约束违规。真实仿真器只作为评估真值，不参与任何候选预测。

**决策收益为系统级结果，不是训练主目标**。若独立场景没有稳定业务提升，就报告“完整世界模型及决策集成已实现、性能增益尚不成立”；不在 test 上反复调权重。

## 4. 最终必须交付的四个可以实际验收的产物

1. **世界模型**：版本化 `RHWM-v1` 与（通过门禁则）`RHWM-v2`，明确输入、输出、权重来源、known/unknown 与合法状态转移。
2. **可复现实验**：一键数据生成、训练、模型加载、H1/H5/H10 测试和业务 paired benchmark；保存 frozen config、sha256、CPU/GPU 时长与原始结果。
3. **预测与决策接口**：一次候选评分能追溯到“当前系统状态 → 未来状态粒子 → 任务/资源预测 → 最终选择”，提供 Rule 回退。
4. **论文/演示证据**：至少一个包含多任务、多个工作站和多机器人资源竞争的完整可视化场景；展示预测值与真实后续状态对比，诚实标注业务收益是否实证成立。

## 5. 工作中不能反复改动的边界

- 不能改写 PR #18 既有 V4 checkpoint/测试/历史归档或 PR #19；PR #18 仍 Draft。
- 不能把旧 5 个诊断状态用于 v2 超参选择、训练回归方向或新的“发现收益”故事。
- 先实现 **完整系统未来状态预测**，再决定是否要额外训练模型；不做与 P0 无关的大规模服务端/UI 工作。
- 仅小模型+CPU起步；无事前计算预算，不动 V100。
- 要与企业业务保持相似性，但没有真实企业日志就只能称为合成仿真验证，不得声称迈克生物现场可部署。

## 6. 相邻工作的归属与文献引用

- Zaheer et al., **Deep Sets**, NeurIPS 2017: https://proceedings.neurips.cc/paper_files/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html
- Sanchez-Gonzalez et al., **Graph Networks as Learnable Physics Engines for Inference and Control**, ICML 2018: https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html
- Sanchez-Gonzalez et al., **Learning to Simulate Complex Physics with Graph Networks**, ICML 2020: https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html
- Chua et al., **Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models (PETS)**, NeurIPS 2018: https://proceedings.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
- Zhou et al., **A multi-robot–multi-task scheduling system for autonomous chemistry laboratories**, Digital Discovery 2025: https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00313f
- PyG 官方异质图教程： https://pytorch-geometric.readthedocs.io/en/stable/tutorial/heterogeneous.html

**上述借鉴均不是“原创发明图网络/概率动态/MPC”的主张。论文的工程创新点是符合业务约束的可观测系统状态建模、可扩展关系式混合状态转移、有限预算下集成多机器人滚动调度和可复现系统性验证。**
