# PR #51：Decision-Aware Relational World Model Learning 的反事实标签与训练目标门禁

## 研究定位

本科毕业设计《基于世界模型的样本转运调度系统设计与实现》及其方法研究定位为 **Decision-Aware Relational World Models for Dynamic Multi-Robot Scheduling**。

截至 PR #49，已有冻结 V4-H20 服务完成事件预测、已观测设备故障条件恢复的外生事件采样、严格业务资源/运动/工作流转移校验、CP-SAT 真实在线闭环候选排序与分负载下的实际收益。PR49 48组独立场景×3个完整在线臂在双方均完成评分的条件下：总体恢复版相对原V4平均−0.396tick，11胜32平5负；多设备故障−2.375tick，10胜6平0负；拥堵+0.563tick；分批到达+0.625tick。这些结果证明预测影响决策，却**未证明世界模型是按照决策质量训练的**。

本阶段明确把“面向决策的关系世界模型学习”定位为方法目标，而不是改名现有 V4、把 CP-SAT 训练为神经策略，或在原 V4后面加一个不反向影响预测动态的独立评分网络。

## 相邻方法和可直接迁移部分

- **Farahmand, Barreto & Nikovski, [Value-Aware Loss Function for Model-based Reinforcement Learning](https://proceedings.mlr.press/v54/farahmand17a.html), AISTATS 2017**：动态模型训练应考虑规划/价值损失，而非仅优化无差别的状态概率误差。
- **Voelcker et al., [VaGraM: Value Gradient Weighted Model-Based Reinforcement Learning](https://openreview.net/forum?id=4-D6CZkRXxI), ICLR 2022**；[作者开源实现](https://github.com/pairlab/vagram)：决策价值敏感度可引导动态模型拟合，尤其当状态部分维度无关时。这里**不把简单pairwise logistic冒充VaGraM原始实现**，只是借鉴“决策损失必须触及动态参数”的原则。
- **Sanchez-Gonzalez et al., [Learning to Simulate Complex Physics with Graph Networks](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html), ICML 2020**：可以迁移基于对象与关系的预测转移结构，而不是必须从零写新 GNN 论点。本项目实体包括机器人、可见工序、工作站及资源；边包括已分配、候选分配、工序到站点以及站点到资源。是否需要真正消息传递网络，应以和扁平编码、无决策损失的对照为准。
- **Voelcker et al., [Calibrated Value-Aware Model Learning with Probabilistic Environment Models](https://proceedings.mlr.press/v267/voelcker25a.html), ICML 2025**：朴素价值感知目标可能不校准。因此模型训练时必须同时监测服务/设备概率校准、H5/H20状态误差、候选排序错误和独立闭环收益；不能只凭联合训练目标下降宣称方法成功。

## 一、反事实监督真正需要的数据

存在两种截然不同的标签：

1. **预测标签**：系统在某实际执行动作下的下一步/多步公开状态、设备状态、资源占用和任务完成事件。
2. **决策标签**：在**完全相同**的公开状态、合法CP-SAT候选集合、此前实际选定动作序列以及同一次真实外生随机条件下，仅替换**当前第一个候选批次**，实际模拟完整后续任务的完成时间/逾期差异。

第二种标签不能直接从 PR49 中已执行的**一个**动作的结果推断其他未执行候选的真实效果，也不能将不同场景的完成时间直接拿来做 action ranking，因为负载、已知故障及后来出现的任务均不同。

此 PR 真实实现：`experiments/world_model_integration/decision_learning_pr51/pr51_labels.py`。

- 使用 **原 PR48真实BusinessSimulator**，原 BusinessObserver 的**连续可观测状态**，原 CP-SAT 合法提案（4候选、固定seed、原 deterministic/time限额）。
- 选择首次 tick≥7 且两个及以上CP-SAT不同合法提案的决策节点，不根据事后哪个动作有收益来选择节点。如果没有符合节点，记录 `labelable:false`，保留场景，不得事后筛除后声称代表整体。
- 将此节点前所有派单固定为第一个合法CP-SAT提案，节点处分别强制选第0、1、2、3个合法提案之一，之后**一律回到第一个合法CP-SAT提案**。
- 每一次分支都重新从真正的同一模拟器初始状态、同一个seed运行，业务服务时长由原模拟器**按场景seed/样本/站点稳定键控**，保证不同提案所见外生随机条件可比。分支前同一公开状态的 SHA256、CP-SAT候选池SHA256及此前动作序列均严格一致，否则程序报错而不是继续把不可比标签混入数据。
- 保存后续真实完整回合的 makespan、逾期、容量等待以及 anchor后H5/H20公开全系统状态摘要，均仅作**监督标签**；图网络输入只来自当时已公开的机器人、工序、工作站、资源、合法动作边，不读取真实未来工作任务发布时间或故障结束时刻。
- 原业务层容量、运动、工作流正确性检查每个反事实分支都必须通过。
- 预先冻结 **72个新种子、按场景类型独立划分**：拥堵、双设备故障、分批到达三类，每类训练12、验证4、测试8，共训练36、验证12、测试24；所有此前PR44–50种子不重叠。两种设备持续时间机制在每种划分内平衡。**72组是标签可行性试验，不是可支撑发表级泛化结论的最终训练样本量。**

该阶段不能替代真正训练。第一步先看：可标注节点比例、同状态候选的真实结果差异比例、每组成功结束/截尾比例，以及此问题有无真正可学习的决策信号。若大多数候选在真实业务目标上完全等价，不能通过随机把平局打破制造“决策标签”。

## 二、训练目标与模型结构

目标仍是一个**行动条件下预测未来系统状态**的世界模型：

```
(public robots/tasks/stations/resources graph, legal dispatch action edges)
     -> shared relation-aware transition dynamics
     -> predicted future business state distribution (H5/H20)
     -> differentiable predicted candidate consequence / value readout
```

可先比较结构化关系编码（复用GNS/PyG概念）与等参数量扁平编码，不在没有数据之前预设 GNN 最佳。

预测损失针对系统状态概率、服务事件、设备恢复、任务和资源关系定义；候选真实目标 `y_i` 为同样真实模拟器条件下单次动作干预的**完整业务完成时间**，越小越好。

本 PR 已实现一个**实验性pairwise排序替代损失**，非直接重现VaGraM论文：

```
L_dec = mean_{i<j, y_i != y_j, both_completed}
    softplus( - sign(y_i-y_j) * (c_hat_i-c_hat_j) )
L_total = L_pred + λ L_dec
```

- 候选完全平局**不随意指定谁赢**；未完成/被截尾候选**不得视作成功完成**，排名暂排除，原预测监督仍保留。
- 先用验证集选择有限个预先固定的λ值，正式测试集在模型与λ冻结之前完全不参与调参。
- **关键梯度门禁**：仅对 `L_dec` 反向传播，必须观察到共享**世界状态转移参数**的非零梯度。若更新只发生在末端ranker，便只训练了动作评分器，没有完成Decision-Aware World Model Learning。对应代码 `pr51_loss.py::gradient_path_present` 和 `test_pr51.py` 的正反例。
- **校准门禁**：比较 `L_pred` 与 `L_pred+λL_dec` 的 H5/H20多字段状态误差、服务/设备事件Brier、候选排序准确度、闭环完成时间与逾期，避免一个判分更好的模型牺牲世界状态物理/业务一致性。
- **核心消融**：a) 关系世界模型只用预测损失；b) 同网络加决策损失；c) 扁平编码加决策损失；d) 冻结V4另接ranker（**负对照**，不能当做决策世界模型学习）。
- 在线对照仍保留原CP-SAT生成合法候选与BusinessSimulator真实系统，保证改进来自预测模型内部而非调度候选产生器偷换。

## 三、阶段门禁与算力

本轮 GH Actions 的 CPU仅承担**真实模拟器反事实标签采样**，不使用GPU、没有神经训练，属于训练前必须具备的有效监督数据集。GitHub Actions已知没有可直接调用的GPU；最终目标GPU训练时要同时保存环境、代码、数据、optimizer状态、随机种子、best/latest checkpoint，确保恢复训练及概率校准可复现。

阶段依次为：
1. **PR51-A** 真实合法动作分支，得到可核验、可学习的同状态候选排序标签；各组三分层原始数据与 SHA256、锁文件归档。
2. **PR51-B / 后续PR** 把已验证的训练集扩展到足够规模，使用现有GNS/图神经实现的**共享关系动态网络**，先预训练预测任务，再做联合loss微调；保存梯度审计与checkpoint，GPU平台有实际资源时做完整GPU训练。
3. **独立验证**：预测-only vs 联合loss、关系 vs 扁平、冻结V4+ranker负对照，先看未见验证集，再锁权重和参数进入独立测试。未来需要跨地图、不同任务到达机制及业务负载，不能仅在同7×5地图说明广义迁移。
4. **真实闭环**：同GPU推理资源、同合法候选、同预测跨度、同外生机制、同安全fallback，记录makespan/逾期/真值决策后悔值。不允许通过只降低状态误差便宣布decision aware成功。

**任何一种结果都不自动保证方法有效。** 核心研究假设应表述为：*在维持多步状态预测可靠性及业务约束的条件下，把候选动作后果监督共同传入关系世界模型的状态转移参数，是否能够在独立复杂负载中减少选择后悔与实际在线任务完成时间？*

PR51保持Draft，不改既有正式V4/CP-SAT/PR19。PR50事件到达学习是**独立的外生动态数据研究支线**，不再被当作Decision-Aware Learning核心方法的替代。
