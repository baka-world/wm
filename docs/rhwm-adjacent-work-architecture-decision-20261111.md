# RHWM-v2 建模前相邻工作评估与架构取舍（2026-11-11）

> 状态：文献与现有代码对照后的架构决策草案。仅文献/只读代码审查；不代表已训练、已验证 RHWM-v2。继续保留 Draft PR #21，不动 PR #19 和主分支。

## 结论先行

**暂停“直接用异质图网络替换 V4 Deep Sets”的既定训练方案。** 在目前的事实生成机制中，服务时长由 `station.service_ticks + hash(seed, sample_id, phase, station_id) % (jitter+1)` 决定；`business_transport/observation.py::capacity_view` 的共享资源容量限制是明确的确定性规则。容量争用改变等待、调度与后续工作流，但**没有在模拟器的服务时长生成函数中引入排队长度、资源占用或机器人交互作为随机时长原因**。

因此，现有合成数据不足以证明“关系式图编码对服务完成概率有独立可学习的额外因果信息”。如果直接训练 GNN 服务头，可能只会将确定性规则和已有 V4 特征重新编码；即便在固定模拟器上取得数值提升，也不能据此声称识别了真实实验室的设备竞争机制。

世界模型目标仍是 `P(S_{t+h}|O_{<=t},A_t,π)`：**输出完整任务、机器人、工作站、共享资源状态分布**，并区分学习事件机制与规则合法转移。已实现 RHWM-v1 提供了这一整合接口；下一模型创新应以**真实可学的条件事件/时长信息**为首要目标，图只是可选编码器，而非成果本身。

## 一、直接可比邻近研究

| 研究 | 已有贡献与适用条件 | 对本项目的明确取舍 |
|---|---|---|
| *Modelling stochastic behaviour in simulation digital twins through neural nets* (2021) | ANN/混合密度网络预测离散事件仿真中依赖当前状态的事件结果分布 | 与 V4 “学习条件随机事件 + 确定性仿真器”最接近。不能把这种混合架构本身宣称为原创；优先借鉴**条件事件时长分布** |
| *Plug and Schedule* (2026) | 生产系统的异质图 State Modeler、事件驱动仿真和 PPO 调度 | 借鉴异质关系与**只在决策触发事件重算**，但其编码器主要产生调度状态嵌入，不能直接称为预测完整未来状态的世界模型 |
| *Dynamic heterogeneous graph combined with reinforcement learning for solving job shop scheduling problem* (2026) | 将机器作为节点、动作形成机器-工序边 | 借鉴机器人/任务/站点/资源动态边；不复制 RL 策略或把图表征当世界模型 |
| *Graph Networks as Learnable Physics Engines for Inference and Control* (2018) | 使用对象、边及消息传递学状态转移 | 图预测动力学的基础，不直接提供业务工作流离散状态或校准的事件时间分布 |
| *PETS* (NeurIPS 2018), mbrl-lib | 概率动力学与轨迹采样、集成不确定性估计 | 借鉴模型不确定性与滚动估计；当前事件逻辑是离散的，不强行迁移连续控制 ensemble 模型 |
| *A multi-robot–multi-task scheduling system for autonomous chemistry laboratories* (2025) | 约束规划+CBS，在 3 机器人、18 站点的实验室系统进行任务协作 | 保留 CP-SAT/MAPF 安全层，不能把多机器人任务并行、动态插单当原创算法 |
| *Graph neural networks for job shop scheduling problems: A survey* (2025) | 系统比较作业车间图表示及 GNN 调度方法 | 用于梳理任务-资源关系结构与已覆盖内容；绝大多数图模型为调度策略，并不是环境预测器 |

### 参考链接

- https://www.tandfonline.com/doi/full/10.1080/17477778.2021.1874844
- https://www.tandfonline.com/doi/full/10.1080/00207543.2026.2741380
- https://www.sciencedirect.com/science/article/pii/S0957417425038643
- https://proceedings.mlr.press/v80/sanchez-gonzalez18a.html
- https://proceedings.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html
- https://github.com/facebookresearch/mbrl-lib
- https://pubs.rsc.org/en/content/articlehtml/2025/dd/d4dd00313f
- https://github.com/pic-ai-robotic-chemistry/multi-bot-coordinator
- https://www.sciencedirect.com/science/article/pii/S0305054824003861

## 二、基于现有证据的三个候选方案

| 候选 | 新学习内容 | 与已知规则整合 | 优势 | 主要风险 | 现阶段裁决 |
|---|---|---|---|---|---|
| A：事件时间/剩余服务时长条件分布 | 在公开服务进度、站点信息及历史数据下预测剩余时长/完成危险率 | 事件抽样 → 现有 WorkflowDynamics/容量/运动/工序状态推进 | 可直接改进现有 V4 的主要随机事件头，CPU 成本低，适合事件驱动 | 现有模拟器服务时长的随机性过于简单；可能很快逼近可学习上限 | **首选建模候选，先做离线数据可识别性确认** |
| B：关系式图条件事件预测 | 任务/机器人/工作站/资源关系嵌入 → 事件概率分布 | 相同合法状态转移 | 可变机器人/样本数量、跨实体关联显式 | 当前随机服务时长并不依赖排队/共享资源负载；GNN 可能没有超越 V4 的真实新信号 | **条件 GO：先证明关系信息在给定 V4 观测后仍有增量预测价值** |
| C：完全可学习的多智能体状态生成网络 | 直接生成机器人/任务/资源未来状态 | 需复杂合法性投影与约束 | 研究空间大 | 数据/算力开销高，容易生成违法状态，重复学习已知确定性规则 | **NO-GO，本科周期不考虑** |

### 重要辨析：不是所有“系统级预测”都要求所有状态字段均由神经网络直接预测

混合世界模型可以通过学习 `P(e, Δt|O_t,A_t)` 与合法转移 `S_{t+Δt}=F_{legal}(S_t,A_t,e,Δt)` 组成完整未来状态分布。其中 `e` 是可发生的公开业务事件，`Δt` 是事件发生等待时间。能不能称为世界模型，取决于**动作条件化、未来状态分布、闭环更新和独立验证**，而非是否所有计算均由神经网络承担。

## 三、当前环境的信息瓶颈（基于源码，非推测）

- `lab_transport/simulation.py::_service_duration` 的时长生成不接收队列长度、资源占用、机器人数量或候选动作；只是当前站点服务基础时长 + 与 seed/sample/phase/station 有关的固定离散 jitter。
- `business_transport/observation.py::capacity_view` 通过当前公开状态和分配动作确定准入；共享容量会改变等待，但等候是**已知约束与未来服务事件组合后的结果**。
- 因此，以“5步内开始服务”为辅助头，若续策略已固定，它可能主要模仿调度及容量规则，不一定是新可学习的随机机制。应先比较基于已知规则的确定性等待推演与学习预测；不能把规则可直接计算的标签当神经网络额外信息。
- 现有 V4 前向概率只作用于合法已启动的服务会话；当前多步状态误差的改善并不直接证明需要图关系编码器。
- 不能为了使 GNN 优于 V4 而事后向仿真器引入人为的负载相关服务延迟，再当作真实实验室机制。可做**预注册的合成机制稳健性测试**，但需明确该机制来自假设，而非企业观测。

## 四、修订后的具体执行门槛

1. **先做“有无可学信息”的只读数据契约审查。** 写明想学的随机变量、当前可观测特征、事实标签的发生时点、删失处理、其与规则状态的条件独立关系。只读源码/现有数据，不重新审计 PR #18 旧五状态。
2. **选 A 作为低成本、已有文献支持的第一训练候选**：学习事件发生时间的离散分布（例如给定已经运行的时长、服务类型、站点配置和当前已知状态的剩余时间分布），保留 V4 原“下一步完成概率”作为直接基线。训练目标须与事实数据生成机制一致，并处理尚未完成的截尾服务时长。
3. **B 保留为可证伪的补充**：只有当企业已有可用的业务日志，或事前明确、带源头依据的多种合成事件机制，确实使共享资源/工序/机器人关系成为条件事件分布的有效信息时，再比较同一个事件头下 Deep Sets 与 relation-aware encoder。否则暂缓 GNN。
4. **事件驱动优化单独处理**：可以借鉴 DES 在服务完成、机器人抵达、设备停机、任务释放等事件附近触发重算，但必须验证“跳过空闲时钟”不会忽略其他合法状态变化。无需为了事件驱动而训练额外模型。
5. **独立测试保持严格**：用固定的 H1/H5/H10 整体状态误差、服务事件 Brier/校准、事件时间分布质量以及真实场景组的运行成本评价。已看过的 PR #21 六组不能再当盲测或用来选择下一模型；下阶段需新预先冻结的 group，按 group 汇总，不能以粒子/时刻充当独立样本。
6. **成果口径**：最多声称“基于既有条件事件建模/对象关系建模和约束状态推进，设计实现了面向样本转运的动作条件世界模型及集成验证”；不能把 hybrid DES、GNN、PETS 或 CP-SAT 算法单独称作原创。若创新性依赖实际提升，须等待独立场景证据。

## 五、推荐代码复用

- 保持：`business_transport/world_model/hybrid.py`、`business_transport/dynamics.py`、`workflow_transition.py`、`session_dynamics.py`、CP-SAT 和 PIBT/LaCAM3 等，避免大规模重构。
- 若研究事件时长：在当前服务事件模型旁新增 versioned `event_time_distribution`，而非覆盖 V4 检查点及历史数据；使用 PyTorch CPU 实现离散危险率/有限分布，训练协议冻结。
- 若确有关系增益：优先试 `PyTorch Geometric` 的 `HeteroData` 与 `HeteroConv`，仅作为事件预测编码器，先独立验证当前 Torch 版本兼容性，不升级全仓 PyTorch。
- 不引入 MAMBA/MARIE/Dreamer 的整套多智能体 RL 系统，因为现有上层调度仍由 CP-SAT 负责，模型任务是可观察任务/资源状态预测。

**决议：进入 RHWM-v2 前，先完成文献及可学习随机机制的识别，不立即训练 GNN。**
