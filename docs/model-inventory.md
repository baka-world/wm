# 世界模型来源、检查点与运行路径审计

> 审计基线：PR #17 HEAD ca49c84e26928d3f58dcad54ae9dde18f5fee0fa（2026-11-10）；仅对该提交及既有 PR / 归档进行只读核查。此文档是审计产物，不代表重新执行训练、加载权重或复现实验；未知项目明确保留为未知。PR #8、#9、#11、#15、#16、#17 均不得在本轮合并。

## 0. 三个直接结论

1. **最新完整在线业务闭环是 PR #15。** 其方法包含 nearest（无世界模型前向）、old_rule、workflow_rule、workflow_pets、workflow_structured。三种预测器分别为 ProgressAwareRule（无检查点）、PETSRuleDynamics（PR #11 归档 pets-service-cpu.pt）、CountingStructuredService（PR #9 归档 sequence.pt；只使用神经服务头，机器人运动由规则和 PIBT/LaCAM3 接管）。PR #16 不是新的完整在线学习策略训练，而是调用相同冻结模型对固定候选做离线短窗预测，并通过真实仿真反事实比较。
2. **CPU 足够处理当前的加载、已训练模型的小型前向、CP-SAT 枚举、PIBT/LaCAM3 规划与离线证据审计。** PR #15、#16 没有重训，PR #17 不导入模型也不进行前向。调整模型权重、采用新预测目标/特征结构、更新训练集后要求新的模型或重新校准时才涉及新训练；是否需要 GPU 应按数据规模和性能要求测量，不是这些轻量模型的必然条件。
3. **尚未证明学习世界模型整体优于 Rule。** PR #15 的工作流联合准确率和闭环均未建立学习模型优势；PR #16 的 48 个同状态决策中 PETS 与 Rule 选择相同 48/48，Structured 只改善一次候选选择；事后 Oracle 不是线上可用模型。

主要依据：[PR15 实验入口](../experiments/workflow_prediction/evaluate.py)、[PR15 原始汇总](../examples/evidence/workflow-prediction-20261110/confirmation/summary.json)、[PR16 入口](../experiments/oracle_value/evaluate.py)、[PR16 汇总](../docs/world-model-value-ceiling.md)、[PR17 只读脚本](../experiments/oracle_value/observable_audit.py)。

## 1. 边界与模型清单

| 名称 / 当前角色 | 实现路径 | 框架、参数量 | 训练数据来源 / 训练方式 | 检查点位置和 SHA-256 | 运行设备、入口、冻结状态 |
|---|---|---|---|---|---|
| Rule / PR15-16 主比较 | [ProgressAwareRule](../lab_transport/torch_structured_adapter.py)；[HybridDynamicsModel](../lab_transport/dynamics_model.py)；业务封装 [WorkflowDynamics](../business_transport/dynamics.py) | 项目自实现 Python 确定性规则；**0 个可训练神经参数** | 无离线神经训练；由当前可观察状态、地图、服务先验/已完成服务统计产生条件转移；已观察均值可在线更新，不等于梯度训练 | **无检查点；SHA-256 不适用** | CPU；[make_model](../experiments/business_constraints/run.py) 与 PR15/16；规则与算法参数冻结 |
| Structured / PR15-16 混合式服务头 | [StructuredNetwork / StructuredDynamics](../lab_transport/torch_structured_adapter.py)、[RuleMotionNeuralService](../lab_transport/service_dynamics.py)、[CountingStructuredService](../business_transport/models.py) | 自实现 PyTorch 实体池化双头 MLP；**4,642 个模型参数（含当前未使用的 motion head）**，参见 PR8 模型报告 | PR #9 **另外采集的合成事实轨迹**：80 轨迹、18,358 次转移；train 8,401、validation 4,052；5 epoch、CPU、种子 123；服务正类加权并加 3 步辅助损失，验证集校准阈值 0.60。监督标签来自真实执行下一观测；并非继续训练 A100 权重 | **[sequence.pt](../examples/evidence/service-dynamics-20261109/sequence.pt)**；SHA-256 **a3f5d5c3bae54d2889b8a92414c3763998a9a139006fc01b8df6be25b234a95e** | CPU，网络 eval、无梯度/优化器；[load_pr9_model](../experiments/service_value/counterfactual.py) → [make_model](../experiments/business_constraints/run.py) → PR15/16。仅 context + service head 用于业务前向，motion head 被规则替换 |
| PETS 风格服务分布模型 / PR15-16 | [PETSRuleDynamics / ServiceDistribution / build_network](../lab_transport/paper_reuse/pets_service.py) | 上游 **MBRL-Lib GaussianMLP** 概率集成，5 成员、2 隐藏层、宽 32、输入 8、输出 2（均值与 logvar）；按上游源码计算 **7,050 个可训练参数、7,052 个注册参数**（另 2 个冻结的 logvar 上下限） | PR #9 合成事实训练/验证轨迹派生的已完成服务时长 **352 train / 176 validation**；仅在服务已完成后取有效服务时长标签，服务暂停不计有效时长；log 时长 Gaussian NLL、35 epoch、CPU、种子 17；与 PR9 Structured **不同独立训练** | **[pets-service-cpu.pt](../examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt)**；SHA-256 **3748a5b96af507c1708355b6b33c25d1ecb56813a57b8469fc9144650bf35e38** | CPU，网络 eval、torch.inference_mode；[ServiceDistribution.load](../lab_transport/paper_reuse/pets_service.py) → [make_model](../experiments/business_constraints/run.py) → PR15/16；冻结 |
| PR #8 A100 Structured 历史基线 / **非当前加载** | 同 [StructuredNetwork](../lab_transport/torch_structured_adapter.py)，当时基于 main 0364a634 | PyTorch，**4,642 参数** | 独立合成数据，训练 11,848 次转移，随机初始化、10 epoch、A100 GPU，10/10 最优 | 历史文件名 **epoch-0010.pt**；原始完整路径 **未知**、**未入 Git**；报告中的 SHA-256 **3572c6c3697c4ca76944519ff0a612432bd65d922fd339dc0835036c6c75dfdf**，当前无法针对权重文件重新计算 | 历史训练 A100；PR15/16 **未加载此权重**；现存仅[冻结报告和精简证据](../examples/evidence/pr8-a100-baseline-20261109/README.md)，是否仍保存在某台本地实验机 **未知** |

参数量释义：Structured 的 4,642 是训练网络总参数数，不表示 PR15 使用了全部预测头；PETS 数量按本项目 [8→32→32→2 的 5 成员配置](../lab_transport/paper_reuse/pets_service.py) 及 [固定上游 GaussianMLP 层定义](https://github.com/facebookresearch/mbrl-lib/blob/3f93cccfc8d635f74e335a2f07aab6e9a48fc021/mbrl/models/gaussian_mlp.py) 静态推算：5×[(8×32+32)+(32×32+32)+(32×2+2)]=7,050，再加两个不可训练边界标量。未在本轮实例化模型核对张量。所有相对路径均以仓库根为基准；未知的具体 CPU 型号、PR8 原始权重绝对路径及独立前向毫秒数不填猜测值。

### 输入、输出、作用严格分层

- **Rule**：输入可观察机器人/样本/站点状态、当前候选分配、静态 Geometry、已观察有效服务进度/历史均值；输出下一步机器人/任务/服务关系的合法预测观察。运动与服务进度均有硬编码近似；不预测未知未来任务到达或突发故障。
- **Structured**：原网络输入 26 维机器人、样本、站点实体 token 的池化上下文、候选运动特征及服务特征，训练时输出运动候选分数和服务完成 logit。[编码与网络](../lab_transport/torch_structured_adapter.py)；**PR15/16 的业务变体仅使用服务 logit**：同一上下文送入 [ThresholdDynamics](../lab_transport/service_dynamics.py)，阈值为从 PR9 验证集冻结的 0.60；移动位置由 ProgressAwareRule 及下游 PIBT/LaCAM3 处理。网络不能直接预测整套工作流的后继产生。
- **PETS**：输入当前站点服务先验/观察历史计数和均值、服务阶段、站点可用性及队列/临近任务特征（7 维）+ 常数 0 的 MBRL-Lib 接口占位，共 8 维；输出 5 个成员对 **log 有效服务时长**的均值和方差，再以生存概率计算下一步完成概率，抽样服务完成；[PETSRuleDynamics](../lab_transport/paper_reuse/pets_service.py) 用规则拼接合法下一状态。它并非预训练的机器人视频世界模型，不学习机器人路径，也不是原 PETS 论文中的完整 MPC 控制器。
- **共同业务包装**：[CapacityDynamics/WorkflowDynamics](../business_transport/dynamics.py) 处理可见容量许可，[release_successors](../business_transport/workflow_transition.py) 用公开工作流模板在预测前一阶段完成后释放后继，严格校验动态 ID、容量与资源关系。该继承/释放逻辑是**项目确定性代码**，不是神经网络的输出头。
- **调度器不是世界模型**：[BusinessCPSAT](../business_transport/dispatch.py) 基于 OR-Tools CP-SAT 产生最多 4 个候选分配，建模阶段顺序、资源容量等；[BusinessPolicy](../business_transport/policy.py) 才借助模型展开候选、评分并选出动作。规则/神经预测器都不是 CP-SAT。
- **运动规划器不是世界模型**：[PIBTMotion](../business_transport/pibt_motion.py) 与 [LaCAM3Planner](../business_transport/lacam3_motion.py) / [RecoveryCoordinator](../business_transport/lacam3_coordinator.py) 负责联合通行、碰撞回避和卡死恢复；在预测时使用隔离的路径恢复快照，在真实执行时处理机器人动作。
- **真实仿真器不是世界模型**：[HybridBusinessSimulator](../business_transport/lacam3_integration.py) 继承 [CoordinatedBusinessSimulator](../business_transport/pibt_integration.py) / [BusinessSimulator](../business_transport/simulation.py) / [Simulator](../lab_transport/simulation.py)，负责实际离散状态、服务、故障、后继阶段和指标；模型预测的轨迹只用于候选评分，不替代事实推进。

## 2. 检查点溯源与归档状态

| PR / 实验 | 训练是否发生 | 产物与归档 | 与后续推理的关系 |
|---|---|---|---|
| **PR #8**（head 141500d1f69911ee2a7a9c04e9b932c6dcf971dc） | A100 从随机初始化训练 Structured 10 epoch；墙钟 60.41 s；PyTorch 2.5.1+cu124 | 只归档精简报告、指标与 SHA256SUMS；**epoch-0010.pt 未提交**；原始数据与完整日志未入库。详见 [PR8 来源说明](../docs/pr8-baseline-provenance.md) | **不是 PR #9/15/16 的加载权重**；原始权重现在是否在某本地机器上未知 |
| **PR #9**（head 96e5f4c1e1626171682549e796b774c14e91cad2） | 另外生成更有故障覆盖的合成训练数据，同初始化方案训练 original、weighted、sequence 三个独立变体；5 epoch，CPU | [original.pt](../examples/evidence/service-dynamics-20261109/original.pt) SHA256 **ae968d8a962188e782d151ad110b2e4dd2eb69393ebb9f1653b28c5d9149999d**；[weighted.pt](../examples/evidence/service-dynamics-20261109/weighted.pt) SHA256 **d1526f28399634a77ce559ecfad0e760099e380508c8cfe2ec9f99fb6106bfbe**；sequence.pt 如上；三者均归档，参见 [training.json](../examples/evidence/service-dynamics-20261109/training.json)、[SHA256SUMS](../examples/evidence/service-dynamics-20261109/SHA256SUMS) | PR15/16 只通过 [load_pr9_model()](../experiments/service_value/counterfactual.py) 加载 **sequence.pt**；同时核对文件 SHA-256、数据 manifest、源代码合同和阈值。sequence 指 3 步服务监督辅助目标，**不是原论文预训练权重，也不是 RNN 训练延续** |
| **PR #11**（head 0f7629b545de8abca0aad1ad2d36d9f066cbd006） | 用 PR9 事实训练/验证轨迹派生的 352/176 条**已完成服务**训练另一个 5 成员概率集成；35 epoch，CPU | [PETS 权重](../examples/evidence/paper-reuse-integration-20261109/pets-service-cpu.pt)、[training-metadata.json](../examples/evidence/paper-reuse-integration-20261109/training-metadata.json)、[SHA256SUMS](../examples/evidence/paper-reuse-integration-20261109/SHA256SUMS) 已归档；派生训练样本 [existing-service-labels.json.gz](../examples/evidence/paper-reuse-integration-20261109/existing-service-labels.json.gz) 亦归档 | PR15/16 [make_model('pets')](../experiments/business_constraints/run.py) 按固定路径创建 ServiceDistribution；不使用 PR8/9 的神经网络权重 |
| **PR #15 / PR #16** | **无训练**；只加载冻结产物、做 CPU 预测与真实仿真 | 结果与配置各自归档，未生成新的世界模型检查点 | PR15 是完整业务闭环，PR16 是固定候选反事实、推理、事后评分 |
| **PR #17**（head ca49c84e26928d3f58dcad54ae9dde18f5fee0fa） | **无训练、无模型前向、无仿真重跑** | 只读分析 PR16 固定归档；见 [CPU 工作流](../.github/workflows/observable-audit-cpu.yml) | 不是 PR15/16 的新模型版本 |

**校验范围说明**：PR9 加载器运行时自行校验 sequence.pt 的真实字节哈希；PETS 加载器校验模型元数据及结构，但 [ServiceDistribution.load](../lab_transport/paper_reuse/pets_service.py) 本身不单独执行归档 SHA256SUMS 的文件哈希校验。此处 PETS SHA 值来自仓库已提交的 SHA256SUMS，不可混称为加载时哈希断言。本轮未下载、执行或反序列化二进制权重，故不声称独立重新算过 SHA。PR8 所列哈希只是历史报告记录，不能证明当前可取得文件。

## 3. 已执行模型推理的代码级完整调用路径

### PR #15：最近一次完整业务调度闭环

1. [experiments/workflow_prediction/evaluate.py:478-489](../experiments/workflow_prediction/evaluate.py) 的 run() 在 CPU 单线程加载 PR9 sequence.pt（即使 nearest/Rule 并不调用其网络），遍历五种调度基线。配置见 [experiments/workflow_prediction/config.json](../experiments/workflow_prediction/config.json)：预测 horizon **5**、粒子 **3**、候选预算 **4**、CP-SAT 限时 **0.5 s**、运动 backend hybrid。
2. [policy_for / run_method](../experiments/workflow_prediction/evaluate.py) → [make_model](../experiments/business_constraints/run.py)：rule 构造 ProgressAwareRule；pets 每次构造 PETSRuleDynamics 并从冻结 pets-service-cpu.pt 加载 GaussianMLP；structured 构造 CountingStructuredService，使用已经加载的 PR9 网络与验证阈值。随后全部经 [RecoveryMovementModel](../business_transport/lacam3_integration.py) 包装运动恢复接口。与之对应的事实执行端构造 **HybridBusinessSimulator**。
3. 真实仿真器每个派工时刻调用 [BusinessPolicy.__call__](../business_transport/policy.py) → observer.observe() → candidates()；CP-SAT 方案由 [BusinessCPSAT.proposals()](../business_transport/dispatch.py) 构造，候选仅依赖当前可见任务、规则、地图和资源条件。
4. 对每个候选、每个粒子、每个模拟步，调用 [BusinessPolicy.evaluate()](../business_transport/policy.py) → [WorkflowDynamics.predict_next()](../business_transport/dynamics.py) → [CapacityDynamics.predict_next()](../business_transport/dynamics.py) → [RecoveryMovementModel.predict_next()](../business_transport/lacam3_integration.py) → Rule / PETS / Structured 服务预测。PETS 在需要完成判断时走 [ServiceDistribution.parameters()](../lab_transport/paper_reuse/pets_service.py) 的真正 GaussianMLP 前向；Structured 走 [StructuredDynamics.predict_next / ThresholdDynamics.service_advances](../lab_transport/service_dynamics.py) 的上下文与服务头前向。预测后由工作流转移释放合法后继，并做 [check_business_transition](../business_transport/workflow_transition.py) 校验。
5. 对候选的预测轨迹调用 [workflow_components / score_components](../business_transport/workflow_scoring.py)，比较「窗口内未完成业务任务步数 + 末端剩余距离与预计服务工作量」，再以资源竞争、剩余距离作字典序平局处理；[BusinessPolicy.__call__](../business_transport/policy.py) 选择最小分数候选，将其分配作为**实际派工动作**返还给真实仿真器。
6. [HybridBusinessSimulator](../business_transport/lacam3_integration.py) 使用真实离散仿真规则、当前外生故障与服务扰动和 PIBT/必要时 LaCAM3 实际推进；未来真实故障计划和隐藏服务倒计时**不传给预测器**。它的事实服务状态变化和任务完成才生成评估标签。[business_transport/simulation.py](../business_transport/simulation.py) 处理资源和阶段释放。

**特别说明**：PR15 的 [workflow_score](../business_transport/workflow_scoring.py) 按已知服务先验估计末端工作量，而不是逐候选调用 PETS 的 estimated_service_work()；PETS 的神经前向发生于服务完成预测及其递归状态变化。不能把 PR11 旧 RolloutDispatch 的「PETS 评分时前向」套用到 PR15 新的 workflow_score。

### PR #16：同一模型权重的固定候选离线反事实

[experiments/oracle_value/evaluate.py:313-342](../experiments/oracle_value/evaluate.py) 加载同一 sequence.pt，跑 nearest 事实轨迹；[evaluate_decisions](../experiments/oracle_value/evaluate.py) 只取合资格的事实观测状态，通过同一 BusinessCPSAT 取得固定候选；[replay_branch](../experiments/workflow_prediction/evaluate.py) **分别重新调用真实仿真器**，强制首个候选动作、之后统一 nearest_batch，收集完整事实结局；[future_metrics](../experiments/oracle_value/evaluate.py) 针对 Rule/PETS/Structured 调用同一 WorkflowPolicy.evaluate() 真正预测、评分与选动作，再独立做第一粒子的五步误差归因。Perfect-State-5/20 使用已真实重放的未来状态，只供**事后诊断**；Full-outcome Oracle 按完整反事实结果挑最优，也仅供事后定义上界。

PR16 共 **24 场景、8 个地图/种子簇、48 个决策点、174 个候选方案**。存在学习模型真实前向，但不更新参数，不产生可在线部署的 Oracle 模型。其汇总没有单列可直接引用的「GaussianMLP 单次纯前向毫秒」或全实验独立前向合计，标记为**未知**，不拿模型步数推算。

### PR #17：不发生模型前向

[experiments/oracle_value/observable_audit.py](../experiments/oracle_value/observable_audit.py) 只用 Python 标准库读取 PR16 confirmation / robustness 的固定 JSON/GZIP、识别真实 device_available 上升/下降沿，并汇总可观察状态与事后标签；[.github/workflows/observable-audit-cpu.yml](../.github/workflows/observable-audit-cpu.yml) 只运行该分析及测试，不导入 torch、模型、业务仿真器或执行候选。**PR17 本轮模型前向次数：0**。PR17 审计在真实故障前 8 个状态均未观察到可作为早期预警的服务历史；20 步事后 Oracle 优势不能称为线上可预测故障的收益，详见 [可观察性审计](../docs/observable-information-audit.md)。

## 4. CPU 时间、调用次数、闭环指标：三种互不替代的统计单位

### PR #11：CPU PETS 集成确认（独立旧实验口径）

来源 [paper_reuse_integration.md](../docs/paper_reuse_integration.md) / [confirmation-summary.json](../examples/evidence/paper-reuse-integration-20261109/confirmation-summary.json)。Python 3.10.22、PyTorch 2.5.1+cpu；16 个合成场景、96 个任务，全部完成。

| 策略 | 平均任务流转（仿真步） | 平均每次调度**CPU 进程时间** | 神经前向合计 |
|---|---:|---:|---:|
| nearest + Rule | 104.667 | 0.88 ms | 0 |
| CP-SAT + Rule | 101.823 | 9.63 ms | 0 |
| CP-SAT + PETS | 101.844 | 21.12 ms | **4,196** |

PR11 的 PETS 4,196 次包含 **620 次服务状态转移前向、3,576 次剩余服务评分前向**，平均每次决策 49.95 次；这是旧 RolloutDispatch 的调用分布，**不可迁移为 PR15 的统计**。仅有每决策 CPU 时间，**无纯神经网络单次前向延迟实测**。PETS 与 CP-SAT + Rule 平均流转差约 **+0.021 仿真步**，未证明性能优势。

### PR #15：完整业务工作流 CPU 闭环（当前最相关）

来源 [PR15 确认集 summary.json](../examples/evidence/workflow-prediction-20261110/confirmation/summary.json) 和 [报告](../docs/workflow-prediction-results.md)。本机 CPU，PyTorch 单线程；具体本机 CPU 型号 **未知**；8 场景、4 个布局/种子簇，每方案 48/48 业务任务完成。如下各时间均为 PR15 归档的**墙钟时间**：

| 策略 | 平均业务流转（仿真步） | 平均每次完整调度决策（ms） | predict_next 次数 / 平均每步整合预测（ms） | 实际神经头前向计数 |
|---|---:|---:|---:|---:|
| nearest | 44.854 | 0.17 | 0 / 不适用 | 0 |
| CP-SAT + Rule（旧状态递推/旧评分） | 43.125 | 52.70 | 2,520 / 1.162 | 0 |
| CP-SAT + Rule（完整工作流） | 42.979 | 67.71 | 2,505 / 1.347 | 0 |
| CP-SAT + PETS（完整工作流） | 43.542 | 76.81 | 2,535 / 1.554 | **2,038 次 GaussianMLP.forward** |
| CP-SAT + Structured（完整工作流） | 43.625 | 91.22 | 2,535 / 1.968 | **2,082 次服务头调用**，上下文前向另计 |

**predict_next 不是一次神经前向**：它包括构建可见状态、神经/规则预测、PIBT/LaCAM3 运动、容量与工作流转移。PETS 的 2,038 次是真正的 GaussianMLP 调用；Structured 的 2,082 次是有服务机会时的神经服务头调用，另有上下文编码，并非「整个 StructuredNetwork.forward 方法调用 2,082 次」。各法整合预测的累计墙钟：旧 Rule 2,928.16 ms；完整 Rule 3,373.60 ms；PETS 3,940.45 ms；Structured 4,989.27 ms，均来自同一 summary.json。

**完整调度决策**另包含 CP-SAT、对多个候选/粒子的多步 rollout、合同校验及评分，不能用一次 predict_next 代表决策延迟；**完整实验总耗时**还包括所有事实执行、反事实、文件处理和测试，不得与任何单步毫秒相混。

PR15 联合状态预测：一步 Rule **121/137**、PETS **116/137**、Structured **104/137**；三步 **72/103、49/103、34/103**；五步 **55/83、32/83、23/83**。以完整工作流 Rule 为参照，PETS / Structured 分别为 **1 场景较快、4 平、3 慢**；平均流转均更长。PR15 仅 4 个地图/种子簇，分层区间跨 0。不能从训练服务事件上局部改进推论总体调度收益。

### PR #16 / #17：离线反事实和只读分析

PR16 [confirmation/cpu.json](../examples/evidence/oracle-value-20261110/confirmation/cpu.json) 显示 **24 场景整批计算总墙钟 88.0192 s、总进程 CPU 时间 88.0011 s**；这不是模型单次推理，也不是一轮调度延迟。PR16 决策 regret（越低越好，按 8 簇等权）：Rule-5 **1.7639**、PETS-5 **1.7639**、Structured-5 **1.7153**、Perfect-State-5 **1.2257**、Perfect-State-20 **0.2812**、Full-outcome Oracle **0（定义）**。PETS 选择与 Rule **48/48** 相同；Structured **47/48** 相同。具体 PR16 的 PETS/Structured 总神经前向次数及纯前向耗时没有出现在所核查的汇总字段中，**未知**。PR17 不执行模型或仿真，额外前向 **0**；PR17 的数据处理耗时不属于推理成本。

### PR #8：禁止把 GPU 测试混成 CPU 指标

[PR8 GPU benchmark](../examples/evidence/pr8-a100-baseline-20261109/predict-bench.json) 中 learned 单步端到端预测平均 **0.9106 ms**，测量设备为 **cuda:0 A100**，不是 CPU、也不是纯网络 forward；训练墙钟 **60.41 s** 不是推理。其旧闭环 16 场景平均流转 nearest 75.864、规则 76.760、神经 77.344；此场景/数据与 PR15 不同，不能横向称为严格配对改善。

## 5. 论文、开源框架和自实现代码分别是什么

| 部件 | 实际可证明的来源 | 本项目自实现部分 | 是否复用论文预训练权重 |
|---|---|---|---|
| Structured | 通用 PyTorch 神经层；实体集合池化是通用架构形式，尚未核实其复现任何特定论文的公开模型代码 | [StructuredNetwork](../lab_transport/torch_structured_adapter.py)、token/动作特征、训练适配、服务阈值与规则运动混合 | **否**。PR8/PR9 权重由项目合成数据训练 |
| PETS 风格 | Chua 等（2018）PETS 概率动力学思想；实际调用 pinned [MBRL-Lib GaussianMLP](https://github.com/facebookresearch/mbrl-lib/blob/3f93cccfc8d635f74e335a2f07aab6e9a48fc021/mbrl/models/gaussian_mlp.py)、ModelTrainer、ReplayBuffer，源码 commit **3f93cccfc8d635f74e335a2f07aab6e9a48fc021**，MIT，[来源记录](../examples/evidence/paper-reuse-integration-20261109/upstream-provenance.json) | [service_dataset.py](../lab_transport/paper_reuse/service_dataset.py) 的七维服务协变量与事实提取；[pets_service.py](../lab_transport/paper_reuse/pets_service.py) 的对数时长标签、存活概率/条件完成、规则状态转移，以及业务/调度包装 | **否**。权重是 PR11 用项目事实合成轨迹训练，并非 MBRL-Lib 附送预训练模型 |
| CP-SAT | [Google OR-Tools](https://github.com/google/or-tools) 包 **9.14.6206**、示例 commit **100f66e6242ab8bf8d32feb8f3bf086db66ae2b5**，Apache-2.0 | [BusinessCPSAT](../business_transport/dispatch.py) 任务阶段、资源容量与候选接口 | 不适用，非世界模型 |
| PIBT / LaCAM3 | 分别复用项目固定的原生官方算法后端，许可/来源记录见 [PIBT 审计](../docs/pibt-motion-results.md)、[LaCAM3 审计](../docs/lacam3-recovery-results.md) | Python 桥接、冻结动作约束、预测分支缓存与恢复整合 | 不适用，非世界模型 |
| 真实仿真器、工作流与评分 | 项目现有模拟环境 | [BusinessSimulator](../business_transport/simulation.py)、[WorkflowDynamics](../business_transport/dynamics.py)、[workflow_scoring](../business_transport/workflow_scoring.py) | 不适用，真实执行不靠预训练权重 |

特别提醒：**“使用 PETS 论文思想和 MBRL-Lib 的 GaussianMLP”不等于“使用论文提供的 PETS 完整系统或预训练世界模型”。** Structured 也没有证据表明移植了某篇论文现成的预训练结构化模型。这两点在论文写作和工程交接中必须保持明确。

## 6. 冻结状态、已验证范围与当前未解决问题

- 所核查 PR #15、#16 推理入口既不调用 train() 也不调用 optimizer.step()；模型在评估/推理状态，训练与评估过程是不同实验。PR #17 工作流不导入模型/仿真器；本轮审计只读取 GitHub 源码、文本归档及元数据，**未训练、未使用 GPU、未改算法、未改历史结果**。
- checkpoint 字节实际可在仓库中定位：PR9 的 original/weighted/sequence 和 PR11 的 PETS；PR8 A100 仅有校验值与汇总，未找到二进制本体。**本地是否留存 PR8 检查点未知**。模型的进一步训练、新权重加载或 SHA 重校验须另立实验，不得由历史日志推断。
- PR17 [可观察性结果](../docs/observable-information-audit.md) 指出 8/48 个未来 20 步发生故障的决策点，起始时均无有效服务历史/活跃服务/已占用容量，故障在 tick 10 才发生；真实事后信息占据大部分 Perfect-State-20 优势。即使提高网络容量，也不能凭空预测没有观察先兆的外生故障。
- 要证明学习模型产生**可部署的增益**，需要在可用在线信号、固定候选和真实未来不可见的协议下，先建立优于规则的预测及决策增量证据；当前结果不支持在同一 5 步评分不变时直接盲目重训。
- 不详项：PR8 本地权重位置/现存与否；PR15 本机 CPU 型号；PR16 全部神经层级前向计数和纯前向延迟；PR15 纯网络前向延迟。所有项均不靠近似值或 PR11 不同协议的记录替代。

## 7. 可只读复核的入口

- 文件来源与 SHA 声明：[PR8 冻结记录](../docs/pr8-baseline-provenance.md)、[PR9 训练记录](../examples/evidence/service-dynamics-20261109/training.json)、[PR11 训练元数据](../examples/evidence/paper-reuse-integration-20261109/training-metadata.json)、[PR9 SHA256SUMS](../examples/evidence/service-dynamics-20261109/SHA256SUMS)、[PR11 SHA256SUMS](../examples/evidence/paper-reuse-integration-20261109/SHA256SUMS)。
- 运行路径与公平性：[PR15 代码](../experiments/workflow_prediction/evaluate.py)、[PR15 固定配置](../experiments/workflow_prediction/config.json)、[PR16 代码](../experiments/oracle_value/evaluate.py)、[PR16 固定配置](../experiments/oracle_value/config.json)、[PR17 只读分析脚本](../experiments/oracle_value/observable_audit.py)。
- 在**另行独立获授权**的具备完整依赖的 CPU 环境，可仅查看文本归档、运行 PR17 的标准库单元测试或对已取得文件做 sha256sum -c（校验不执行 torch.load 或模型前向）；不应在本轮运行 PR15/16 重放脚本，因为那属于新的仿真执行。

**最终审计判定**：世界模型的来源与权重身份已经能够按 Rule / PR9 sequence / PR11 PETS 三条互不混淆的线路追踪；PR8 A100 权重不在当前仓库；PR15/16 确实使用了固定学习型预测器进入候选评分，但没有证明调度优于 Rule；PR17 本身完全没有世界模型前向调用。
