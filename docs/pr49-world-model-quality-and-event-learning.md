# PR #49：充足预算下检验世界模型动态能力，而非优化 CPU 时延

## 研究路线修正

用户明确：最终世界模型不使用 CPU 作为推理运行平台，愿意继续投入计算预算；因此停止将 CPU 推理效率作为世界模型研究的优先目标。PR #48 的高负载在线退步包含 CPU 超时回退因素，不能直接据此判断学习型恢复动态没有决策价值。

PR49 的第一阶段采用**48个此前未见过的业务压力新种子**，原样复用 PR48 的三种环境机制（14任务拥堵、两处故障、分批任务发布），原样复用 PR30 的合法CP-SAT候选及H20/P3完成数评分、原V4权重和PR44站点条件恢复KM，唯一改变是对**两个世界模型同时给出宽裕30秒/决策的CPU-process异常保险限额**，避免模型有效性与CPU性能混淆。此时间限额**不是最终部署的时延目标**，CPU仅承担 GitHub Actions 可复现实验执行，最终模型部署面向 GPU。

协议预先冻结在 `experiments/world_model_integration/unconstrained_pr49/pr49_protocol.json`；每种压力场景16组、两个真实故障持续时间机制各8组，共48组新独立场景×3策略（原CP-SAT、原V4-H20、恢复V4-H20）=144次真实闭环业务仿真。种子未与PR44/45/46/47/48交叠。原业务模拟器和决策安全约束不变；未来故障结束时刻与未来任务发布时间**仅作为真实环境隐含变量**，不能交给模型。

本轮的核心比较是**当两个模型都能充分完成候选评分时**，恢复动态对不同类型业务压力是改善、无效还是变差。主指标为每场景实际 makespan、按场景等权的胜/平/负及各类分布。必须同时报告未完成、逾期、任何候选退回/预测错误及安全违规。如果任一模型仍发生候选评分回退，则明确标记这一场景并不据其宣称“完全评分”的模型质量差异，不会通过事后增预算或删样本包装成功。

## 模型现状与相邻方法迁移

截至PR48真实代码，`HybridSystemWorldModel` 是混合世界模型：V4仅学习设备服务完成概率，路径/资源容量/工作流状态遵守已有可检查的状态转移；PR45另行加入了已观测故障后的条件KM恢复事件。**还没有学习任务到达事件模型**。现有原 `check_business_transition` 禁止凭空生成新任务，以保证未来外生业务真值不会泄漏，这是必须保留的性质。

在控制动作的世界模型之外，异步外生事件可借鉴已有事件过程研究，而不是从零设计新网络。下一研发阶段的来源与迁移次序：

1. **基线：固定到达率/泊松事件过程**，仅从在线真正已出现的公开任务到达历史估计，到达未知则保持不确定；不能用预设未来任务表伪造预知。
2. **条件历史强度模型：Neural Hawkes / marked temporal point process**，用已观测的历史到达间隔、工序类型、资源拥堵等公开信息，预测下一个任务的到达时间与类别。迁移依据：Mei & Eisner (NeurIPS 2017) 的 Neural Hawkes 以及2025年事件过程综述。选型时明确是不是已有历史中真正存在可预测依赖，否则深度网络不会创造原本不可观察的规律。
3. **候选更完整的未来事件轨迹分布：EventFlow** 等条件事件序列分布建模，可用于多事件时间预测，但复杂度显著高于简易历史hazard，应先根据基线与验证数据必要性决定，不能只为追逐新模型而堆网络。
4. **条件恢复模型**保留PR45显式外生事件的合法转移合同，可将训练所得独立的未来任务到达事件与当前公开业务状态结合，在遵守工作流释放、资源容量、无未来信息泄漏情况下传播到H20完整未来状态。

相关已有工作：
- Mei & Eisner, [The Neural Hawkes Process](https://papers.nips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html)
- [Modeling events and interactions through temporal processes: A survey](https://doi.org/10.1016/j.neucom.2025.131191), *Neurocomputing*, 2025
- [EventFlow: Forecasting Temporal Point Processes with Flow Matching](https://pmc.ncbi.nlm.nih.gov/articles/PMC13580664/), 2026
- Chua et al., [PETS: Probabilistic Dynamics Model and Trajectory Sampling](https://arxiv.org/abs/1805.12114)
- Chen et al., [Specification-Driven Discrete-Event World Models via DEVS](https://arxiv.org/abs/2603.03784), 2026（公开状态契约/事件因果与可检查转移设计的相邻思路）

## 下一阶段训练纪律

下一阶段优先构造**训练/验证/独立测试三段场景**，仅从连续公开 `BusinessObserver` 状态提取任务到达事件、站点异常/恢复和服务事件时间序列。严格区分：

- 真实世界未来到达表、真实故障时刻表：只在事实环境生成；
- 当前及历史公开到达、站点、队列、机器人与资源状态：可作为模型输入；
- 模型预测的未来任务数/到达时间与类别：可作为不确定状态事件进入预测粒子；
- 尚未出现的未来具体任务ID、真实业务属性和时刻表：**不可直接预测为已知事实**，需要保守的匿名/概率业务需求表示及新的合法外生到达合同。

如离线离散事件模型的开发验证显示学习到比历史统计基线更好的可复现预测分布，再进行真正的神经模型训练与GPU端推理环境迁移。真正的训练与推理资源可加预算；必须保存完整随机种子、优化器、训练数据生成与代码环境、最佳/最新检查点。模型质量以 H5/H20 队列、任务完成、资源占用、服务/恢复概率校准及**新独立在线真实调度效果**评价，而不是CPU每次推理耗时。

## 运行入口

`python -m unittest experiments.world_model_integration.recovery_integration.test_pr45 experiments.world_model_integration.unconstrained_pr49.test_pr49 -v`

`python -m experiments.world_model_integration.unconstrained_pr49.pr49_online --checkpoint /path/to/sha-verified-v4-data.pt --pr44-result /path/to/sha-verified-pr44-train-only.json --regime congestion --out /tmp/pr49-congestion.json`

GitHub Actions `.github/workflows/pr49-high-compute-world-quality.yml` 并行执行三种环境，每个环境归档逐场景公开决策历史、模型、真实业务轨迹hash、依赖和原始结果；聚合节点仅在完整3×16新场景全部成功后分析。全程保持PR49 Draft、不给原业务生产流程打补丁。


## GPU 目标架构迁移审计（仅设计，尚未实际GPU执行）

源代码检查表明原 `train_v4.load_v4` 使用 `torch.load(..., map_location="cpu")` 并返回CPU网络；`business_transport/session_dynamics.py::SessionRuleDynamics.predict_next` 把 `encode_rows` 生成的CPU张量直接送入网络，再从torch输出 `.tolist()`。因此**原有V4不是GPU自动可用的部署路径**。不能将GitHub hosted CPU实验伪称为GPU推理实测。

后续GPU版本（新的研究性选择，不修改冻结对照模型）需要采用：
1. 保留原模型检查点的 SHA256 与权重结构，显式 `device=torch.device("cuda")`，CPU读取检查点后 `model.to(device).eval()`，输入tokens/masks/queries全部送往同一个GPU，并在需要Python事件抽样时从预测张量显式 `detach().cpu()`。
2. CPU与GPU在相同公开观测及合法动作、相同检查点上的**事件概率**数值等价，采用可解释的浮点容差；粒子样本由于临界概率差异不能擅自宣称逐轨迹完全相同。随机种子和独立粒子流需保存。
3. 原业务规则、容量、CP-SAT及合法状态转换继续作为可验证的结构化转移模块运行，学习型事件模型 GPU化；不要把迁移解释为在GPU上所有Python控制逻辑自动加速。
4. 在实际GPU执行环境绑定后，独立完成设备检测、显存与batch验证、CPU/GPU概率对齐、H20粒子回放、真实墙钟P50/P95和完整训练检查点恢复试验。**目前GitHub workflow和当前可访问的运行容器没有确认可用GPU，故此项仍为待执行工程检查而非已完成结果。**
5. 推理性能不作为下一轮世界模型有效性的选择标准；GPU运行时延仅用于系统部署验收，不能以CPU机型表现作为毕业设计的核心算法结论。

## 数据可识别性优先于盲目增加模型大小

PR48的 `staggered_arrivals` 合成机制按确定的发布tick 0/6/12/18生成任务，而原 V4/H20 不知道真实未来发布表。若在同一固定机制上训练事件到达网络，会很容易“背诵固定节拍”，并不证明对未知实验室任务到达的泛化。因此训练数据应当覆盖**多种合成到达机制**（定期、泊松、历史相关突发、负载相关），分离机制/seed/时间段的train、validation和完全独立test，先用可解释历史平均到达率与统计hazard验证信息增量，再使用Neural Hawkes等已有方法迁移。事件预测目标是给未来20tick可验证的概率分布，而不是把未来任务ID、真实发布tick、下游完成时刻伪造成公开输入。需要任务ID与业务属性的世界状态接入时，必须设计**外生到达第二状态转移合同**，完全独立于普通合法业务状态转移checker。


## 候选已有实现与模型迁移优先级（2026-11-12 进一步查证）

检索并核实了可直接复用的公开实现，以避免重新发明事件序列网络：

- **优先选型：[EasyTPP / EasyTemporalPointProcess](https://github.com/ant-research/EasyTemporalPointProcess)**，ICLR 2024公开开源框架，当前PyTorch版已包含 `NHP` (Neural Hawkes)、`RMTPP`、`IntensityFree`、`THP` 等已发表模型及标准数据预处理/训练/评估流程。特别适合对离散任务发布**时间+类型mark**建模。可以迁移其训练框架、输入事件序列契约及GPU模型，不从零设计神经架构。
- **对照原论文实现：[hongyuanmei/neurawkes](https://github.com/hongyuanmei/neurawkes)**，NIPS 2017 Neural Hawkes作者源码，作为方法核对；工程迁移优先采用EasyTPP的现代PyTorch接口而不是复制旧研究项目全部依赖。
- **长跨度事件时间研究备选：[GavinKerrigan/eventflow](https://github.com/GavinKerrigan/eventflow)**，AISTATS 2026 EventFlow的开源GPU训练/预测代码，直接模拟未来多个时间事件的联合分布；但原论文主要是**无mark事件时间**而我们需要未来任务类别/站点，作为二阶段长跨度对照而非第一实现。不能直接声称其原版能预测工作流标记。

优先做**公开观测历史→事件数据集→Poisson/历史hazard基线→EasyTPP NHP或IntensityFree→冻结独立测试**，训练、测试与模型选择必须在不同生成机制/场景种子上分离。没有看到真实前兆或历史事件结构时，继续追加GPU训练不会凭空创造信息。只有事件模型在独立H5/H20任务到达计数、发生时刻、mark及下游预测上改善时，才沿已验证的PR45显式外生事件合同接入完整世界模型进行闭环决策验证。
