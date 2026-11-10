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

**下半部分：本项目源码动作空间审计、V4 信息价值、三个最小迁移接口、统一选型、业务确认表和下一阶段冻结 CPU 实验设计，见本文后续章节。**
