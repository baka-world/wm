# PR54：时间感知、动作条件化的关系世界状态转移学习

## 方法定位

项目`lieuyimo/wm`，论文方向 **Decision-Aware Relational World Models for Dynamic Multi-Robot Scheduling**。当前已完成PR50/PR51/PR52的H5/H20直接条件化逐实体预测、真实反事实决策监督和校准消融。PR52既有源代码的两层`RGCNConv` + 输出头（hidden48）并非完整的单步可重复状态转移：它从单个快照直接输出H5/H20的状态分布；对动作差异的预测经常非常接近，强制decision loss可降低后悔却伤害系统预测MSE。不能仅凭大网络更有表达力就断言应该扩容。

**本PR改进的是世界模型本身，不更改调度器的候选产生和评分算法。**

文献迁移依据：

- [Sanchez-Gonzalez et al., *Learning to Simulate Complex Physics with Graph Networks*, ICML 2020](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)：节点/关系上的可重复局部动态，消息传播深度及长时rollout误差管理；本项目实际使用已经验证的PyTorch Geometric RGCNConv，而非从零研发GNN。
- [Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models*, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)：PETS的不确定性、模型ensemble与粒子rollout思路仅在**单步可构造合法未来系统状态**且概率校准时才可迁移。PR54当前不是PETS复现。
- [Voelcker et al., *Value Gradient weighted Model-Based Reinforcement Learning*, ICLR 2022](https://github.com/pairlab/vagram)：预测目标/决策价值错配，且梯度加权需要有可信动态而不只是提高终局排序；当前阶段不叠加未经验证的排序损失。

## 现有实际实现与参数设计

核心路径：`experiments/world_model_integration/temporal_pr54/`。

```
真实 BusinessSimulator 的连续公开状态与合法已执行派单
  -> 前4个时刻（t-3..t）公开机器人/工序/工作站/资源状态
  -> PR51同源的关系图（任务取样/交付、站点资源、已分配/本次候选分配）
  -> 16维节点特征 + 历史可见mask
  -> 48维 GRU(4个公开时刻，每实体单独沿时间)
  -> 当前快照48维嵌入 + fusion
  -> 2层官方 PyG RGCNConv（6种业务关系+反向边，4 relation bases）
  -> t+1逐实体6字段合法状态概率 + 匿名新工序出现/完成计数
  -> 显式 node mask / task categorical simplex / new completed <= new visible
```

参数对照是**同一网络、同一参数量、同一训练目标与源数据**：
- `snapshot_only`：只保留t时刻公开状态，t-3..t-1历史被严格置零；
- `public_history_gru`：完整公开历史；若无历史数据或节点尚不存在，历史槽位使用0且可见mask为0。

这能判断历史可观测系统状态是否为预测新增信息，而非将更大参数量或换一个调度器混入对照。**t+1的公开状态仅作为真实标签，未知未来设备故障结束时刻、真实未来任务释放表不允许成为模型输入**。当前节点原始ID只用来在同一真实时序中对齐实体，模型仅见数字化节点字段和匿名关系ID。

业务观测历史来源为原`Simulator.run(on_transition=...)`及原`BusinessObserver`，真值业务合法性仍由原模拟器、容量审计、动作/路径/工作流审计保证。每次用的是**真实已执行CP-SAT当前动作**，不是根据真实未来选出来的反事实最优动作。尚未观察到的未来任务只能通过匿名事件概率而非提前创建task节点进入世界模型。

## 冻结CPU结构门禁与统计单位

原始协议`protocol.json`提供 **60个全新独立业务场景**，各压力类别20场景、TRAIN36/VALIDATION12/TEST12。每场景采样至多56条连续真实公开的`(t-3:t,action_t,state_t+1)`。共最多3360条事件对；但同场景56个步长高度相关，**推断统计单位始终是独立场景而不是3360个相互独立样本**。

原始真实数据：[GitHub Actions #43191970049](https://github.com/lieuyimo/wm/actions/runs/43191970049) **SUCCESS**：拥堵、多故障、分批到达各20场景/1120个真实一步样本，合计60/3360。原始归档包括全量公开图、先验历史观测、独立t+1标签、source_head、SHA256SUMS和pip freeze。

`pr54_train.py`：官方PyG真实RGCN + GRU，两个模型×种子1810/1811，每arm/seed以VALIDATION的场景宏平均实体MSE＋0.25设备Brier选择best epoch，保存best/last checkpoint、Adam、Torch RNG、环境。两个arm再根据VALIDATION两seed平均实体MSE/Brier选择一个，**写入selection.json后才读取物理独立TEST**。对每组源业务场景同等加权报告：

1. t+1逐实体masked MSE、站点设备可用Brier、新任务数MAE，记录负载分组；
2. 原公开状态**持续不变**的persistence基线，检验模型是否真正超过无学习的持久性预测；
3. history是否在同参数下超过snapshot，避免只论整体损失下降；
4. 每个模型实际参数量、真实checkpoint及源SHA；
5. 真实测试是否保持时间因果不泄漏与业务安全（预测本身未被用作真实控制指令）。

### 门禁解释

- 若history在新测试有可信改进且概率预测校准不劣化：进入大数据GPU容量消融与真正递归状态转移设计。
- 若history和snapshot都不超过persistence：优先检查状态目标不变比例、动态信息是否存在、未来外生变量可观测性与下一步标签平衡，禁止为追求分数无条件扩容。
- 若snapshot优于history：观察是否存在GRU训练不充分/长时状态干扰，但**不能据已公开TEST重新调window或参数**。必须通过新训练/验证/盲测再做一次确认。

## 下一阶段的GPU启动条件与任务交接

单步CPU验证不需要GPU。需要**多步状态递归、H20粒子rollout、多种训练seed、大于200独立场景与模型宽度96/128/层数3的控制变量消融**时，优先以GPU作为训练与未来最终部署目标。

已提交两个直接可用交接文件：

- [docs/gpu-handoff-pr54.md](gpu-handoff-pr54.md)：正式GPU任务清单、资源前提、数据恢复与SHA核验、GPU/CPU数值一致性、模型规模/历史窗口/关系传播消融、H5/H20递归rollout、任务与容量安全、best/latest/optimizer/RNG归档；启动时不覆盖本阶段。
- [docs/pr54-gpu-agent-prompt.md](pr54-gpu-agent-prompt.md)：可直接粘贴到有GPU环境的执行Agent的完整中文命令和验收Prompt。

**当前没有任何GPU执行承诺或GPU产出。正式GPU训练前需实际确认GPU可用，并在GPU分支冻结新大数据训练协议与独立测试集。**

## 当前结果与研究边界

当前PR54的训练质量结论依赖独立CPU模型工作流；尚不允许从本阶段的one-step测试推论“在线样本转运效率已提高”。最终研究目的仍是**高保真、合法、可递归、动作条件化的世界模型**优先，其次才是决策感知loss及真实在线CP-SAT候选重排收益。原V4、原CP-SAT正式系统和历史PR保持冻结，PR54 Draft不合并。

## 实际完成的模型训练与新TEST结果（2026-11-12）

[官方 PyG + BusinessSimulator 实际训练 Actions #43192730343](https://github.com/lieuyimo/wm/actions/runs/43192730343) **SUCCESS**，36独立train、12验证、12独立TEST，模型seed1810/1811，两个完全同参48维RGCN+GRU模型，16 epochs，预注册validation选中`public_history_gru`。训练4份best与4份last均保存、包括Adam状态、torch RNG、环境锁和源SHA；[原始权重与结果Artifact #13930509625](https://github.com/lieuyimo/wm/actions/runs/43192730343/artifacts/13930509625)。所有TEST均在VALIDATION选择文件写出后才读取。

| 模型 | 新12个独立TEST场景，2 seed宏平均逐实体masked MSE ↓ | 设备Brier ↓ | 来源 |
|---|---:|---:|---|
| 公开状态不变的persistence | 0.04099843 | 需要报告每个设备字段 | 真实当前→下一tick |
| 同参snapshot-only RGCN | 0.01213195 | **0.00933001** | 只看当前公开状态 |
| **同参history-GRU RGCN** | **0.01028620** | 0.00985017 | 历史t−3..t |

绝对MSE改善−0.00184575，约**相对降低15.2%**；两训练seed的状态MSE均明显低于snapshot。**但设备Brier反而增加+0.00052016（约5.6%相对劣化）**，不能宣称所有未来状态更可信。

[全部原始12场景配对只读统计 Actions #43193769866](https://github.com/lieuyimo/wm/actions/runs/43193769866) **SUCCESS**，[逐场景记录 Artifact #13932082660](https://github.com/lieuyimo/wm/actions/runs/43193769866/artifacts/13932082660)：0新增训练、0新事实轨迹，固定2模型seed在每个场景内部平均，再按拥堵/多故障/分批到达各4个场景分层bootstrap 5000次，12个真实独立场景才是统计单位。

- **实体状态**：12/12场景history更好；差值(history−snapshot) −0.00184575，95%场景Bootstrap条件区间 **[−0.00204430,−0.00165377]**。
- **站点可用性**：仅3/12场景history更好，9/12更差；Brier差值 +0.00052016，区间 **[+0.00015335,+0.00093816]**，明确存在校准退步。
- 这个区间仅条件于当前同地图合成三类负载与已固定的两个神经模型初始化；**不是训练seed变化和不同地图、不同厂商设备的泛化保证**。

### 当前阶段判断与GPU任务触发

**时序历史有价值：PASS。设备可用性概率校准：尚需改进。多步递归世界模型与真实在线调度收益：未验证。** 初步数据证明应当优先研究可观测历史而不是无限堆参数。进入下一阶段所需的是新独立数据、设备状态/历史down时长信息建模、可递归的t+1合法状态世界转移和真实GPU规模控制变量试验，**不是重调这12组公开TEST的学习率**。

下一GPU任务已经创建为 [GitHub Issue #54：完整GPU清单及资源验收](https://github.com/lieuyimo/wm/issues/54)。与源代码同步的[可直接下发执行Agent的Prompt](pr54-gpu-agent-prompt.md)、[GPU完整资源和阶段门禁](gpu-handoff-pr54.md) 均已提交仓库。**当前CPU动作与结果真实，尚无GPU训练或完整H20在线闭环结果。**
