# PR #18：预测驱动调度相邻工作深度复现审查与源码级迁移选型

**审计基线：** wm@d3932fa6b54c2e0a7caa5300c3f745657ad9d33b；PR #18 保持 Draft。外部官方代码 SCoBA.jl@e66633dcf044decfb63b2d3cb1b5b059f79cd6ea。  
**工作性质：** 文献、官方源码、内部源码及最小接口审计。0 GPU、0 训练、0 新 checkpoint、0 新仿真、0 独立测试复用。没有复现任何论文性能数值。  
**事实标记：** 【源码】代码直接支持；【论文】原文明确；【推导】为适配设计而推导，尚未运行；【待企业确认】不能视为企业已确认需求。

## 一、开头八个问题的直接回答

1. **最适合的预测驱动机制？** 条件优先选择 Fernández-Cuevas 等（2026）**预测式任务准入（Predictive Admission Control）与已承诺服务窗口保护**，而不是照搬机械臂系统。该论文直接解决“当前启动一个任务，是否会耽误未来不可错过的业务服务”。但本企业是否存在这种严格承诺窗口尚未确认，**最终门禁为 BLOCKED，条件优先不等于 GO**。
2. **预测怎样改变动作？** 2026 论文 Algorithm 1 将候选任务的未来占用时间与当前已承诺的后续服务窗口进行冲突检查，通过则 DISPATCH，否则 WAIT。SCoBA 在策略树成功/失败分支里使用时窗内成功概率，决定尝试/跳过与机器人匹配。Street 等通过未来任务出现的时间/地点分布计算等待点与竞拍增量成本，使机器人提前部署。这三项都实际改变动作，不是仅生成漂亮的未来轨迹。
3. **V4 能否提供相应预测？** **不能直接提供完整预测对象。** 冻结 V4 是合法观测下活动服务会话的下一步完成概率预测器，不是整条工作流在截止前完成的 CDF，不提供确定上界或未来设备健康、任务到达时空分布。可在适当条件下作为未来完成时长分布的局部 hazard，但不能直接替换 SCoBA 的 success_prob_fn 或 2026 论文硬时限所需的保守上界。
4. **现有 CP-SAT 与模拟器支持什么？** 支持给空闲机器人分配已释放的 waiting 任务、共享资源容量和多阶段约束。基础外部 DispatchPolicy **可以返回空分配**，仿真器会在下一 tick 再尝试；但正式 CP-SAT 候选并不包含一个主动的 WAIT 决策，优化目标优先最大化当前分配数。没有跨 tick 持久预约、无任务预定位或抢占。
5. **有规则/解析分布难以描述的真实条件信息吗？** 尚无证据。现有服务时长由公开 service_ticks 加上 SHA 键控、有限 jitter 生成，已知有效进度可产生解析条件 hazard。此前独立完成事件 Brier 为 Rule 0.06970、B2 经验条件概率 0.06369、V4 0.06482。V4 没有超越 B2 的信息证据，真实设备退化、批次负荷等信号尚未验证。
6. **如何分离新增动作与模型收益？** 使用同一场景、合法观察、候选、风险约束和 CPU 预算，先对比旧动作空间 Rule 与新增 WAIT/承诺规则 Rule，差异归属**动作扩展及新约束**；再把规则、解析 CDF、B2、冻结 V4 放在**同一个新动作空间**内公平对照，V4 相对强解析/B2 的额外收益才可能归于学习型预测。
7. **小改动/CPU 可行？** 静态约束合同、可观测性、模拟器空分配合法性和 CPU 级门禁可以低成本完成，不需要 GPU，也不必训练。若企业确认真实承诺窗口，后续才另行冻结新开发与独立实验，不动已查看的 16 组 tick0 Test 和 12 个动态开发状态。
8. **下一步复用哪篇、放弃哪些？** 条件优先移植**2026 生物医学论文 §2.7.2 的 Eq.(9)/(11) 冲突准入判据及 Algorithm 1 的 WAIT**，保留 CP-SAT、PIBT/LaCAM3；不移植单臂本体。SCoBA 的单机器人策略树仅作有真实严格时间窗口时的备选机制，不移植其冲突搜索；Street 需要全新到达预测与待命移动动作，不适合当前冻结 V4。若企业无真实承诺时限，本轮三个方向都 BLOCKED。

## 二、核查材料与源代码可用性

- **SCoBA，Choudhury et al.（RSS 2020）**：论文 [arXiv 全文 §III–V](https://arxiv.org/html/2005.13109v3)、[PDF](https://arxiv.org/pdf/2005.13109)；核实的 [官方 Julia 源码](https://github.com/sisl/SCoBA.jl/tree/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea)。已读取类型、策略树、冲突搜索、conveyor、routing 与 benchmark 入口。**源码级审查，不是运行复现**。
- **生物医学单臂调度，Fernández-Cuevas et al.（Robotics 2026，15(10):206）**：[全文 §2.3、§2.5–2.7、Algorithm 1、Propositions 1–2、§3–5](https://www.mdpi.com/2218-6581/15/10/206)，发表于 2026-10-20。论文 §2.1 提到 proprietary platform，Data Availability 表示实验数据须联系作者并经设备商允许。**未定位经核实的作者官方代码，不能宣称完成该实现的源码复现**。以下仅复核原文算法与数学表达。
- **Right Place, Right Time，Street et al.（JAIR 2024, 79:137–171）**：[论文 PDF §3–7](https://www.robots.ox.ac.uk/~nickh/papers/street24jair.pdf)，[Oxford 记录](https://ora.ox.ac.uk/objects/uuid:448f43b1-6837-4333-9bd4-532133a3d2b7)。论文已核对 CTMC、竞拍/IWP、性能基线、复杂度和局限。**未定位可靠的官方公开代码仓库**，不能声称执行过作者的 CTMC/IWP 程序。

### 二.1 SCoBA：数学目标、算法与官方源码逐项核对

**论文 §III Eq.(1–3)：** 机器人 n、任务 k 的时间窗 W(n,k)=[t_l,t_u]；任务时长的累积分布 τ(n,k,t)=P(n 在 t 内完成 k)。目标为最小化整个任务时域未完成任务的期望惩罚，默认每任务 J(k)=1。它优化的是**任务窗口内完成概率**，不是“一个服务会话下一步结束”的概率。

**§IV-A 单机器人策略树：** 按窗口开/闭事件展开 attempt 与 skip 决策和 success/failure 后果，用概率加权结果子树效用，再向根节点取最佳决策。该论文近似“尽早尝试并在窗口结束时观察后果”，不是通用的任意时刻服务结束模型。**§IV-B/C、Algorithm 1：** 高层 CBS 风格任务冲突树用排除任务的分支，重新求有冲突机器人策略树。§IV-C Proposition 1/2 的期望最优性与完备性依赖**没有新任务加入、低层完整时域、在窗口末确定完成**等条件，不能套用到本项目有限预算的动态资源调度。论文 §V 与 EDD、Hungarian、Q-Learning、MCTS 比较，模拟传送带和无人机业务；论文 CPU 设置是 6 核 3.7GHz、16GiB，不构成本项目算力要求。

**官方源码路径（均固定 SHA e66633d）：**

| 关键文件与源码行 | 核对的对象 | 具体含义及迁移边界 |
|---|---|---|
| [src/solver/types.jl#L3-L17](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/solver/types.jl#L3-L17) | MODE、InteractionEvent | START、FINISH、SUCCESS 是**事件时间戳**，不是成功概率。对象持有 agent_name、task_name、timestamps。 |
| [src/solver/types.jl#L22-L72](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/solver/types.jl#L22-L72) | DecisionNode、OutcomeNode、SearchTree、GenericAllocation | 记录分支概率与效用、窗口、分配和冲突阈值。 |
| [src/solver/scoba_tree_search.jl#L1-L62](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/solver/scoba_tree_search.jl#L1-L62) | insert_decision_node! | success_prob_fn(ref_time, interaction_event) 是返回单个 [0,1] 成功概率的回调；建立 p 与 1-p 的结果分支。 |
| [src/solver/scoba_tree_search.jl#L66-L279](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/solver/scoba_tree_search.jl#L66-L279) | generate_search_tree! | 输入 allocation、agent、任务集合、success_prob_fn、util_val_fn、downtime；计算树上**最大化成功收益**（与论文最小化失败罚分的固定统一罚分版本可对应）。不是简单乘概率后排序。 |
| [src/solver/scoba_tree_search.jl#L283-L315](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/solver/scoba_tree_search.jl#L283-L315) | get_next_attempt_idx | 可以跳过某任务，直到找到选择 attempt 的首节点；需要延后/不尝试的动作语义。 |
| [src/solver/scoba_conflict_resolution.jl#L18-L128](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/solver/scoba_conflict_resolution.jl#L18-L128) | coordinate_allocation! | 处理多个机器人争**同一任务**，重算低层策略树；超 conflict_threshold 时警告并返回当前分配。这不能替代当前多工作站 AddCumulative 的资源安全证明。 |
| [src/domains/conveyor_belt/conveyor_belt_scoba.jl#L1-L10](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/domains/conveyor_belt/conveyor_belt_scoba.jl#L1-L10) | cumul_grasp_prob | 传送带抓取成功率随可用步数的 CDF。 |
| [src/domains/routing/routing_scoba.jl#L7-L20](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/domains/routing/routing_scoba.jl#L7-L20) | delivery_success_prob | 使用 Epanechnikov 交付时间分布 CDF，作者代码自身标注 TODO，不能把分布选择当通用真理。 |
| [src/domains/routing/routing_scoba.jl#L25-L143](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/domains/routing/routing_scoba.jl#L25-L143) | scoba_routing! | 筛空闲 drone，包装成功概率、求每台策略树、调用 coordinate_allocation!，最后更新真实配送任务。 |
| [src/domains/routing/routing_simulator.jl#L241-L272](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/domains/routing/routing_simulator.jl#L241-L272) | update_time_windows! | 根据真实已出现包裹的窗口和预计返程构造 InteractionEvent。 |
| [src/domains/conveyor_belt/conveyor_belt_simulator.jl#L119-L155](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/src/domains/conveyor_belt/conveyor_belt_simulator.jl#L119-L155) | update_time_windows! | 从盒子运动轨迹构造开始/结束时间窗。 |
| [scripts/benchmark_routing.jl#L98-L196](https://github.com/sisl/SCoBA.jl/blob/e66633dcf044decfb63b2d3cb1b5b059f79cd6ea/scripts/benchmark_routing.jl#L98-L196) | 实际 benchmark | 入口可切换 edd、hungarian、scoba、MCTS；另有 benchmark_conveyor_belt.jl 和 scaling_*.jl。 |

**论文/源码要分开：** 论文用最小未完成罚分，Julia 树实际最大化成功效用；论文冲突定义包含时间窗口，源码高层主要检测同名 task 重复分配，而非时间资源占用矩阵。实践中低层任务数和冲突节点均设上限，可能返回非完整协调结果。因此不引入 SCoBA conflict tree 取代当前 CP-SAT。共享工作站容量、释放链和运动路径会产生跨机器人相关性，不能独立相乘单机器人成功概率。

### 二.2 生物医学单臂论文：精确的预测准入机制

来源：[§2.3、§2.5–2.7，Eq.(4–11)、Algorithm 1、Proposition 1/2](https://www.mdpi.com/2218-6581/15/10/206)。作者研究一台机械臂与 42 个加热反应室，将需独占机械臂的 blocking 操作与不需占用机械臂的孵育 non-blocking 阶段分开。

- **Eq.(5)**：孵育结束后 attending 操作必须在目标时长 D 与允许误差 Δ 的窗口内开始。**Eq.(7)/(8)**：活动孵育建立由 I（无需臂）与 B（占用臂）交替的未来 commitment chain，并可预测下次必须服务的预约窗口。
- **Eq.(9)/(10)**：要启动候选 c，预投影其未来 B 阶段，不得与任何已承诺活动孵育链的 B 阶段重叠。若只是一个不会释放孵育的普通 blocking 操作，**Eq.(11)** 要求该操作在最早承诺窗口开始前结束。
- **Algorithm 1**：从 READY 选出最高优先级 c；如果它就是到期 attending 或当前无活动承诺，派发；否则通过上述入场检查才派发，不通过返回 **WAIT**，待下一调度调用重试。**从预测到动作**的路径是时间区间可行性筛选，不是简单改一个软评分系数。
- **Proposition 1 的安全条件**：blocking 操作实际时间**不能超过**预测时间，并且到期 attending 的调度优先级高于所有普通任务。Proposition 2 的进展性也建立在相应有限任务/最终释放条件下。**随机均值时长、未校准 sigmoid、未知路上堵塞不满足“绝不低估”的条件**，本项目不能继承原保证。
- **实验 §3/讨论 §4**：开发平台 19 次实机实验、7 台设备，报告 4526/4528 次孵育符合 60 s 窗口；基于同工作负载的数字孪生仿真，混合协议相比批处理缩短 28%–66% makespan，并与 FIFO、due-date、最短任务、random 对照。论文明确该环境不是商用平台规格；这些数值**不是本仓库已验证性能**。论文 §2.7.3 复杂度主要是 O(|R|log|R|) 排序和随活动承诺数增长的检查，不能直接照搬为多机器人复杂度保证。

**迁移关键分歧：** 它以单臂“两个 B 区间不重叠”证明安全；本项目多个机器人与资源池容量 >1、运动路径冲突，需要用每个工作站资源池的容量-时间累计约束与独立运动安全验证；而非直接套 Eq.(10)。最重要的是：染色的孵育时限**不等于**本项目普通 deadline_tick。企业若没有“某阶段一旦启动就必须在未来时段完成下一阶段服务”的真实业务过程，不得人为引入并称为商业需求。**官方源码未核实**，本轮只能声称算法机制研究，不声称代码复现。

### 二.3 Right Place, Right Time：CTMC 到主动待命动作的完整链条

来源：[Street et al. JAIR PDF §3–7](https://www.robots.ox.ac.uk/~nickh/papers/street24jair.pdf)。

1. **§4.1 预测输入**：任务宣布/出现过程的 CTMC 当前可观测状态；吸收态代表未来宣布事件，带位置标签。Eq.(1) 用吸收概率形成未来任务地点的离散分布，首达时间形成出现时间分布。作者假设 CTMC 状态更新可以观察。**它不是服务结束预测**。
2. **§4.2/§5.1 合法主动动作**：任务宣布前，机器人被分配若干潜在任务地点 PTL，可朝 waiting point / intermediate waiting point（IWP）提前移动，并在那里等待；任务真正宣布后才去其实际地点执行。
3. **§5.1–5.3 调度**：Sequential Single-Item（SSI）竞拍，把 (PTL, task) 插入机器人未来计划，bid 是新增该对后的**期望总任务等待时长增量**。§5.3 计算 IWP 以减少到 PTL 的期望延迟；§5.4 提示期望成本计算是主要复杂度瓶颈，多 PTL 导致指数复杂度成分，因而需要限制候选。
4. **§6 公平基线**：非主动 SSI、只向最可能出现地点提前移动的 MLL、MLL+PTL、IWP、IWP+PTL。其结果依赖任务时空结构，完整 IWP+PTL **并非每种配置都最佳**。§7 明确最廉价插入与 greedy IWP 是次优启发式，且未完整考虑机器人物理交互。
5. **迁移边界**：当前未释放任务对在线 Observation 不可见，冻结 V4 不会预测到达地点/时间，空闲机器人默认向 home 行动而无通用无样本 StandbyMove。需要合法可观测的批次/上游工作站状态、新到达模型、新待命移动动作及堵塞/能耗限制。当前暂不适合；论文官方仓库**未核实**，不能说实现级移植已完成。

---

## 三、现有系统的真实动作能力与源码级调用链

以 [wm 固定基线 d3932fa6](https://github.com/lieuyimo/wm/tree/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b) 为准。**可在数据结构中表达**、**可由求解器生成**、**可由在线策略选中**、**可由事实仿真器执行**是四种不同等级，均不能冒充已获得企业业务授权。

| 机制 | 数据结构与求解 | 仿真执行路径 | 真实支持状态 |
|---|---|---|---|
| 空闲机器人接已释放任务 | Assignment(robot_id,sample_id)，CP-SAT 匹配布尔变量 | Simulator._dispatch_external 验证 idle×waiting，并绑定任务 | **全链路已支持** |
| 当前 tick 故意不派单（WAIT） | DispatchPolicy 可以返回空序列；BusinessCPSAT 在候选存在时优先最大化派单数，不将 WAIT 列为独立评分候选 | 空返回合法，当前 tick 无分配，下一 tick 再尝试 | **仿真执行原语支持，正式求解/价值排序不支持** |
| 延后已发布任务 | 未分配样本仍为 waiting；允许小于总数的匹配 | 未选任务等待后续时刻 | **被动延期支持；主动保留优先级/预约动机没有建模** |
| 预约未来工作站容量 | CP-SAT 内有临时的 OptionalInterval 和 AddCumulative | 无跨 tick 持久预约、无兑现预约/取消记录 | **不支持预约合同** |
| 为尚未发布任务预留机器人 | 未发布样本不在当前 Observation；机器人只有现行 task_id | 不存在预留未来任务的绑定语义 | **不支持** |
| 空载机器人主动预定位 | 地图与 home 可表示机器人当前位置 | 无样本时 _target 返回 home，无法下发任意待命点动作 | **不支持** |
| 调整已绑定任务后续顺序 | CP-SAT 对候选后续路线作规划假设 | 已绑定机器人不能重新参与当前任务匹配；完成阶段后才释放后继任务 | **不可重新排已绑定任务** |
| 服务抢占、取消与改派 | Assignment 没有抢占或取消类型 | 业务仿真器不允许服务中机器人改派 | **不支持** |
| 截止时间优先与强制时限 | deadline_tick 存在于 job/sample/Observation；BusinessCPSAT 未使用它建硬约束或 tardiness 目标 | 逾期计入 summary.late_count 等；旧基础 Simulator 的 rule/learned_rule 用 EDD，不等于正式业务 CP-SAT 路径 | **记录和基础策略部分支持，当前业务优化链路不支持** |
| 多机器人、阶段与工作站容量 | 每机器人/样本互斥，AddCumulative，配置工作流阶段路线 | 按资源池准入，服务完成才发布 successor stage，运动仍由原规划器保证 | **已支持，调度未来计划不是持久承诺** |

**关键源码与函数定位：**

- [business_transport/dispatch.py#L21-L155](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/dispatch.py#L21-L155)：BusinessCPSAT.proposals 从公开可达匹配生成候选，为活动会话创建粗略剩余占用区间，用 AddCumulative 保护资源，最小化 (horizon+1) × 未派数量 + makespan。该目标不是最迟交付或服务违约。只有无候选时返回空 fallback，**不是有选择的 WAIT**。
- [business_transport/policy.py#L47-L55](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/policy.py#L47-L55) 与 [#L87-L185](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/policy.py#L87-L185)：候选生成、相同预测预算和粒子种子、预测后一律 nearest_batch、按评分元组和原候选索引决策。没有显式等待候选或后续预约权。
- [business_transport/workflow_scoring.py#L29-L138](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/workflow_scoring.py#L29-L138)：主分是 unfinished_job_ticks+terminal_work，其后为 resource_contention_proxy、remaining_distance；**业务 deadline 没进入此评分**。
- [business_transport/scenario.py#L51-L111](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/scenario.py#L51-L111)：编译资源池、阶段与 job release/deadline；没有不可错过的阶段服务承诺，也没有经企业确认的迟到损失函数。
- [lab_transport/transition_schema.py#L56-L82](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/transition_schema.py#L56-L82) 和 [lab_transport/simulation.py#L230-L253](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/simulation.py#L230-L253)：外部策略可返回零个/多个合法 Assignment；只有 idle×waiting 不为空才调用，已绑定机器人与进行中的服务不是合法新匹配动作。
- [lab_transport/simulation.py#L449-L495](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/simulation.py#L449-L495) 与 [business_transport/simulation.py#L19-L54](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/simulation.py#L19-L54)：每 tick 更新事实观察、服务与运动；资源准入可能引发 resource_wait，stage_ready 为后继阶段合法发布点。
- [business_transport/observation.py#L48-L120](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/observation.py#L48-L120)、[lab_transport/state_v2.py#L80-L150](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/state_v2.py#L80-L150)：有效服务进度、暂停时间与历史服务统计依赖完整连续观测，不应从私有 simulator 剩余倒计时制造。
- [business_transport/workflow_transition.py#L99-L167](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/workflow_transition.py#L99-L167)：预测后的多阶段释放、资源容量和事实可观测字段的一致性检查。
- [lab_transport/simulation.py#L180-L215](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/simulation.py#L180-L215) 与 [#L497-L513](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/simulation.py#L497-L513)：旧内部规则策略包含 EDD 与最终 late_count 统计，不能误认为正式 CP-SAT 具有时间窗保证。
- [experiments/world_model_integration/dynamic_value_audit/audit.py#L114-L210](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/experiments/world_model_integration/dynamic_value_audit/audit.py#L114-L210) 和 [#L251-L345](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/experiments/world_model_integration/dynamic_value_audit/audit.py#L251-L345)：此前开发轨迹只证明部分非零时刻有合法区别动作，原 seed 反事实与 Rule/V4 预测差异已检查；新机制不得据此调参。
- [experiments/world_model_integration/value_learnability/run.py#L206-L308](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/experiments/world_model_integration/value_learnability/run.py#L206-L308)：OLS/Ridge 是探索性长期价值拟合，并未成为在线正式 ADP。

## 四、V4 信息能力及解析/经验概率的必要公平基线

### 四.1 当前冻结 V4 的预测对象

[session_events.py#L13-L46](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/session_events.py#L13-L46) 只对已开始、观测进度完整、设备与服务未暂停的有效会话查询；[session_dynamics.py#L50-L95](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/business_transport/session_dynamics.py#L50-L95) 用冻结神经网络计算“下一步完成”概率，抽样后交给合法 ProgressAwareRule 演算资源、机器人和后继工作流。这不是硬剩余时长上界、整个任务截止前成功率、设备健康预测或未来到达分布。[train_v4.py#L42-L59](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/experiments/world_model_integration/train_v4.py#L42-L59) 也声明冻结权重并未通过生产在线认证。

既有独立服务完成 Brier：Rule 0.06970、B2 条件经验概率 0.06369、V4 0.06482。V4 相比粗 Rule 有预测能力，但**不优于已检查的 B2 统计概率**；更不能宣称学习模型利用了传统方法无法获得的信息。

### 四.2 当前服务时长可以有合理解析分布

[lab_transport/simulation.py#L86-L93](https://github.com/lieuyimo/wm/blob/d3932fa6b54c2e0a7caa5300c3f745657ad9d33b/lab_transport/simulation.py#L86-L93) 的实际服务时长机制：

服务时长 D = station.service_ticks + keyed_hash(seed, sample_id, phase, station_id) mod (jitter_limit + 1)。

对未知 seed 而言，这是一个预设范围内的合成扰动，**不能把真实 seed 或私有 service_remaining 交给任何在线预测器**。但在公开服务先验与抖动配置可用的公平前提下，规则模型可以构造解析离散分布。已连续观测到活动服务有效执行 e tick 且尚未完成，其单步条件结束概率为：

\[
h(e,x) = P(D=e+1\mid D>e,x)
       = \frac{P(D=e+1\mid x)}{P(D>e\mid x)}.
\]

其中 x 是各比较方法共同合法可观察的站点公开先验、服务类型、进度与历史统计，分母需为正。设备暂停时不能把暂停墙钟当成有效服务进度。如果配置中的名义时长/抖动范围实际上**不会提供给生产调度器**，解析基线也不能偷偷获取它们；应对照同观测的经验 B2。

多个时间步的累计完成概率只有在每步 h_k 都是**沿前面未完成路径递推、基于更新后的合法条件状态**得到的 hazard 时，才可写为

\[
P(T\le H\mid s_t,a)=1-\prod_{k=0}^{H-1}(1-h_k).
\]

多阶段运输、排队、多机器人工作站竞争和设备故障有联合相关性，不能把孤立的 V4 sigmoid 输出简单相乘，冒充完整任务按时成功概率，也不能无条件给出硬安全上界。

### 四.3 预测信息真正能够改变调度的必要条件

固定相同的公开可观测状态 o、实际可执行动作集合 A(o)、相同业务成本 C、相同随机环境与调用预算。学习模型要值得单独引入，至少必须证明：

1. 存在关于未来有增量条件预测力、并且所有方法都有权读取的**合法现场观测**，而非只有 V4 拿到隐藏真值或从训练 seed 推断；
2. 在当前状态下至少两种**不同且安全**的动作有可操作的业务后果，不只是候选列表顺序不同；
3. 相关未来事件对候选间真实成本或违约风险有影响，不仅某模型的数值评分变化；
4. 预测的差异可改变候选的**相对价值或是否准入**；如果动作排序固定，改进下一步服务 Brier 也不能自动产生业务收益；
5. V4 相比同观测的**解析概率与 B2 经验条件概率**有可在独立组重现的**有效信息优势**，并能克服附加 CPU 与维护开销。

这是必要研究门禁，不是数学上的充分收益证明。现有结果对第 4、5 项均不支持。若正确解析分布使用同样信息达到相同或更好表现，应优先用解析方案。

### 四.4 数据真实性分级

| 字段/机制 | 当前依据 | 企业层面的真伪 |
|---|---|---|
| 站点名义时间、均匀有界 jitter、预先配置设备停机 | 合成仿真器机制 | **不是企业真实分布** |
| 合法当前服务进度、暂停、工位容量、阶段释放 | 当前观察与业务模拟器 | 模拟中可用；真实现场接口待确认 |
| 启动孵育后必须在后续窗口 attending | 2026 医学生物分析论文 | **不能推断迈克生物存在此流程** |
| 设备退化、温度、上游负荷与服务时长依赖 | 有待研究的候选业务信号 | **尚无真实证据** |
| 未来任务宣布的时间、起点概率 | Street 等模型假设 | **当前 V4 不具备，现场数据未知** |

## 五、三条迁移路径的最小接口与工程代价

### B1：时限任务概率决策，参考 SCoBA

**前提：** 已发布样本具有真实且有处罚/质量影响的完成时间窗，不同合法机器人匹配确实改变工作流整体按时完成概率。

- **数据结构：** 可选新增 TaskDueWindow(job_id, latest_completion, business_penalty)；ConditionalCompletionCDF(observation, legal_matching, horizon, calibration_scope)。
- **决策接口：** 读取现有 BusinessCPSAT.proposals 给出的合法候选，调用**相同信息权限**的解析时长分布、B2 与冻结 V4 局部模型（若能构造一致的完整 CDF），对比任务的期望逾期损失，不碰原 CP-SAT 的匹配和安全约束。
- **必要动作：** 至少已有合法匹配间可以区别选择；若策略要主动跳过、等待或者先执行关键任务，则需显式增加并验证 WAIT，不能把未选任务当已经具备预约语义。
- **目标：** 真实迟到成本、关键完成率、总流转时间，避免仅为了模型优势增加虚拟时限。
- **GPU/成本：** CPU 静态分布/小预检，先不训练。V4 冻结 CPU 推理只是可选。
- **主要风险：** 多机器人共享工作站造成任务成功概率强相关，SCoBA 单机器人树的假设失效；规则/解析 CDF 已经足够好；业务实际 deadline 不具有硬损失。

**不移植**其完整 Julia CBA/conflict tree 替换 CP-SAT，也不移植无人机 Epanechnikov 时长分布。

### B2：预测式准入与资源承诺保护，参考生物医学论文（条件第一）

**前提：** 企业确有“阶段启动后，下一步必须在已知最早/最迟窗口占用某资源，否则产生真实业务损失”的承诺，触发事件与资源可在线观察。

- **拟新增纯数据合同（不是现有实现）：**
  - CommitmentWindow(job_id, stage_id, resource_pool, earliest_start, latest_start, blocking_duration_bound, observed_source, criticality)；
  - DurationBound(resource, task_type, conservative_max_ticks, calibration_status)；
  - AdmissionCertificate(tick, candidate_matching, allowed, conflict_windows, violated_capacity, slack, predictor_kind)。
- **合法决策：** 当前候选 ASSIGN 或经业务允许的显式 WAIT；同一资源池未来区间必须满足 capacity，而不是照搬单臂的两个 B 区间零重叠。CP-SAT 负责当前可行匹配和能力上限；独立 AdmissionGuard.check() 使用公开观察、承诺表与合法时长边界给出可检验的准入证书。
- **风险约束：** 预测时间必须是可证明的保守上界才可沿用原论文 Proposition 1 的硬时限论证；有限分位数或 V4 sigmoid 至多支持**明确业务接受的概率风险约束**，不能写成“确保 100% 准时”。多机器人到站拥堵也需要计入保护边界。
- **最小变更位置（未来工程阶段，非本轮）：** 新的 commitment_contracts.py、admission_guard.py 和隔离的 predictive_admission/ 实验；只有企业和静态门禁通过，才考虑把 WAIT 作为 BusinessPolicy 候选，并添加公平的未来资源预约。BusinessSimulator 为事实权威，PIBT/LaCAM3 运动规划不重写。
- **公平基线：** 保守 Rule 上界、解析剩余时长分布、B2 经验条件概率、冻结 V4（只有它能合法提供该决策需要的条件风险时）。新机制不一定必须使用 V4。
- **GPU/成本：** CPU 即可做逻辑、约束与低预算预检，不需要新模型。
- **主要风险：** 样本转运或实验室流程**并不存在**孵育式未来 attending 承诺；新增 WAIT 可以改善业务，但其收益属于动作/时限建模，不能自动归功于学习型世界模型。

### B3：主动预定位，参考 Street 等 IWP

**前提：** 上游检验流程存在**当前可观察**的未来批次/地点线索；机器人可获许可空载移动到安全待命区。

- **数据合同：** ArrivalForecast(public_upstream_state, location_probabilities, arrival_time_distribution)；StandbyMove(robot_id, safe_waypoint)；SafeWaitingArea(waypoint, capacity, traffic_constraints)。
- **新增动作与模块：** 没有已发布样本时仍可设置安全移动目标，需要扩展机器人上层任务/目标状态和外部策略接口，以及到站等待与取消；路径仍交给已有规划器，必须检查交通堵塞、工作站通道占用、空驶里程、消耗与并发冲突。
- **需要预测：** 真实上游任务宣布的时空分布。先用相同事实历史构造简单频率/热点/CTMC 基线，再考虑学习模型。**冻结 V4 的服务完成模型不适配**，不能依靠它冒充到达模型。
- **业务指标：** 真正宣布任务到开始服务的等待时间、逾期、机器人多余里程和拥堵负面成本。
- **GPU/成本：** 传统到达频率和 CTMC 可 CPU，开发复杂度主要来自新动作与安全状态机，而非算力。
- **主要风险：** 没有可预报的任务到达，提前移动实际反而堵路；即使有益也要先把**动作扩展价值**与**预测信息价值**拆开。

## 六、统一比较与最终选型

| 维度 | SCoBA 时限概率分配 | 2026 预测式准入 | Street 主动预定位 |
|---|---|---|---|
| 业务真实性 | 需确认实际硬/软 deadline | **最接近已有实验室阶段流程，但未来承诺尚未确认** | 需确认可预报的上游批次/出现位置 |
| 文献成熟度 | RSS 2020，理论条件与两个领域模拟 | Robotics 2026，定理约束和实验室设备开发环境 | JAIR 2024，时空到达分布与成组模拟 |
| 可复用源码 | **有 Julia 官方源码**，完整冲突算法不建议替代 CP-SAT | **官方源码未核实**，仅复用清楚的准入判据与伪代码 | **官方源码未核实**，只参考 CTMC/IWP 原理 |
| 冻结 V4 契合度 | **局部 hazard，不等于 job CDF** | 可给活动会话 hazard，**无未来硬时限上界** | **不契合，缺任务到达模型** |
| 强公平基线 | 解析 CDF、经验 B2 | 保守规则与解析时长、经验 B2 | 频率热点、MLL、经验/CTMC |
| 当前动作能力 | 已有不同合法分配，缺完整主动 defer | **仿真可 WAIT，CP-SAT 未把它作为候选** | **无空载预定位动作** |
| 可关注业务后果 | 真实逾期率与最终任务完成 | 关键时窗违约、资源等待与总流转 | 宣布后的等待、里程、交通干扰 |
| 须补真实业务假设 | 有时间窗与迟到成本 | 有启动后不可错过的未来服务承诺 | 有可观测未来到达与待命地点 |
| 最小工程改动 | 中：时长 CDF 接口与期望损失评价 | **中：承诺合同+准入守卫+WAIT** | 高：新到达数据、待命移动状态机 |
| CPU/GPU | CPU，0 GPU | **CPU，0 GPU** | CPU，0 GPU（传统 CTMC 阶段） |
| 本科毕业设计适配 | 中，取决于 deadline 真实性 | **条件第一，最符合多阶段资源协同但需求需证实** | 中低，当前模型无法复用 |
| **本轮裁决** | **BLOCKED** | **BLOCKED，优先验证业务前置条件** | **BLOCKED** |

**没有任何方向达到无条件 GO。** 条件推荐 2026 的预测式任务准入，是因为它对业务时限、资源和当前动作之间具有最直接的因果连接，而不是因为已有 V4 必然能令它获益。若企业证明没有关键承诺时限，则应转向真实业务的具体调度痛点，不能在合成场景中制造窗口硬装企业需求。

## 七、企业待确认事项：用业务事实决定能否继续

1. 样本经过离心、上机、加试剂、保温等实际流程时，**是否存在 A 阶段启动后，必须在固定时窗由 B 工作站处理**的要求？违反后会损坏质量、违反流程规定，还是仅延迟产出？
2. **时钟起点是哪项可读事件？** 是仪器开始、样本从出口可取、试剂加入还是机器人接单？限制的是最早/最晚开始下一阶段，还是最晚交付完成？
3. **承诺资源是哪个？** 工作站端口、共享仪器、特定机器人还是通道？容量多少，能否更换不同机器人或工位？
4. **允许主动等候吗？** 当前有普通样本任务时可否暂不派车以保护即将到期的关键样本？最久可等多久，如何避免普通任务饥饿？
5. **不同业务任务的 deadline 与损失有什么区别？** 是否有真正不可违反的急诊样本优先级，还是只追求平均流转时间？
6. **哪个现在可观测的设备/样本历史变量真的影响未来服务时间？** 设备使用次数、健康、负载、温度、试管类型、实际工序，是否有可追踪脱敏历史记录用于识别条件相关性？
7. **上游是否能提前宣布将来在哪个站点产生样本任务？** 分析仪剩余处理时间、批次进度或订单队列是否向调度系统提供，精度如何？
8. **是否允许空载提前移动至待命点？** 待命点是否占用工作站端口或窄通道，移动是否有能耗与交通安全约束？

必须把回答原样形成可追溯业务约束；没有企业确认，不能宣称“迈克生物已要求预测式服务承诺/设备健康建模”。

## 八、下一阶段最小 CPU 实验协议（**本轮未执行**）

### 八.1 先冻结业务门禁

只有以下条件全部满足才允许提出下一轮代码实验：

- 企业确认一类**真实存在、能被当前合法观察识别、超窗具有业务后果**的未来服务承诺，至少提供触发事件、相关资源、窗口边界和任务处理规则；
- 允许安全的 WAIT 或至少存在两个可执行的不同匹配，且在同一观测下确有**不同后续业务价值**；
- 解析时间分布或可信 duration bound 可以对照，并且各预测器使用相同公开观测；不可从模拟器私有未来获取解析 Oracle；
- 资源池预约、工作流依赖与已有运动安全规则不被绕过；新 WAIT 具备避免死锁/饥饿的超时退出；
- 预注册目标、候选、时域、预测预算、随机种子、场景分组和停止条件，先封存后执行。

若真实业务没有硬窗口，**立即停止预测式准入路线**。若仅有软截止期，应另立软时限决策协议，不借 2026 原论文硬安全定理。

### 八.2 公平对照矩阵，分离新增动作与世界模型自身贡献

| 组 | 允许动作 | 预测来源 | 能归因的差异 |
|---|---|---|---|
| A0 | 原始 CP-SAT 候选 | Rule | 原有业务基准 |
| A1 | CP-SAT + 合法 WAIT/承诺检查 | **同一个 Rule/保守上界** | **新增动作与新业务约束的总体收益** |
| A2 | **与 A1 完全相同** | 公开条件解析时长分布 | 传统统计预测贡献 |
| A3 | **与 A1 完全相同** | 经验 B2 条件概率 | 经验统计预测贡献 |
| A4 | **与 A1 完全相同** | 冻结 V4（仅当能提供完整合法条件预测接口时） | V4 相对 A2/A3 的额外贡献 |

比较必须在相同任务、候选预算、后续策略、设备容量与安全检查、可观测字段及计算预算下完成。A1−A0 是**引入 WAIT/承诺约束的效应**，不应写成 V4 的收益；A4−A2/A3 在统一动作空间和相同观测时才可能反映学习预测的新增价值。

主指标应从**企业确认的关键阶段超窗率/违规数**选取（若无真实硬时限不得建立该指标）；副指标为全部样本完整流转时间、资源等待、到期完成率、移动里程、CPU 延迟，路径/容量/生命周期安全违规则停止实验。单次 seed 下的事实反事实要如实标注 single-realization；不能使用不同历史的 seed 混成同一状态的期望收益。

下轮开发场景和**真正新独立场景**应按原始场景分组、固定配对外生随机机制，避免已经看过的 16 个初始 Test 与 12 个动态开发状态再用于择优。首轮 CPU 开发机制检查可限制 30 分钟，无 GPU、无模型训练；独立评估须由新任务明确冻结后仅运行一次。本轮**没有**执行该协议、没有新实验结果。

### 八.3 最小改动清单（未来候选，当前未实现）

- 新增纯数据合同 **business_transport/commitment_contracts.py**：CommitmentWindow、DurationBound、AdmissionCertificate，并标注来自真实业务的触发点、是否硬上界、是否当前可观察；
- 新增只读**business_transport/admission_guard.py**：AdmissionGuard.check(observation, legal_matching, commitments, bound_provider) 不访问私有未来、不修改模拟器；返回准入理由/资源冲突证书；
- 新增隔离的 **experiments/world_model_integration/predictive_admission/**：仅 CPU、冻结协议与版本、可复现测试和详细运行记录，保留正式 CP-SAT/WorkflowPolicy/PIBT/LaCAM3 不动；
- 仅当前置业务与静态门禁通过后，才考虑扩展正式候选为明确 WAIT 与未来资源预约，同时保留 BusinessSimulator 作为事实、安全和工作流合法性权威；
- V4 不变且是**可选对照**，若无法给出所需的多步条件服务风险分布或通过独立校准，则不得接入安全准入判定。

## 九、最终 GO / BLOCKED / NO-GO 与本轮执行记录

**本轮最终裁决：三个迁移方向暂为 BLOCKED；2026 生物医学 Algorithm 1 的准入检查是业务确认优先级第一。** 原 V4 现有固定五步评分机制仍 NO-GO，本轮没有推翻；也没有证明新准入方式将使 V4 胜出。优先验证“存在真实未来承诺和可执行 WAIT、动作能影响业务价值”而不是继续盲目训练网络。

- **GO（仅授权下一轮开发预检）：** 企业有可追溯的未来关键服务时限、承诺资源可观察，两个合法动作可以改变真实违约风险，且解析/B2 可公平参照，安全与饥饿门禁可实现。
- **BLOCKED：** 任一业务条件尚未确认、关键观测/WAIT/预约无法表达、条件概率转工作流风险不成立或没有公平解析基线。
- **NO-GO：** 条件确认后仍不能产生差异合法动作，或者学习预测不优于解析/B2 并且没有决策增量，或者实现和安全代价过高。不能在原测试上反复改评分直到出现有利数字。

**执行事实：** 实际阅读 SCoBA 官方 Julia 的指定 solver、routing/conveyor domain 和 benchmark 调用链、上述论文算法章节与本仓库真实源码；没有运行 Julia 或作者程序，没有声称论文结果源码复现，没有训练新模型、运行新仿真或打开旧独立测试、没有修改 CP-SAT、仿真业务规则、运动规划器、V4 或 PR #19。本轮研究报告位于 docs/，以 [skip ci] 方式提交；已检查 .github/workflows/world-model-integration.yml 和 world-model-v3-training.yml 的文件触发路径，避免旧 CPU 训练再被不相干报告变更触发。最终是否有 CI 运行以 GitHub Actions 实际记录为准。
