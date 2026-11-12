# PR #51 / PR52：从聚合预测走向逐实体关系动态与决策感知学习

研究题目：**Decision-Aware Relational World Models for Dynamic Multi-Robot Scheduling**

## 研究起点与不重复的失败

[Draft PR #50](https://github.com/lieuyimo/wm/pull/50) 首次真实执行了动作条件化关系R-GCN的联合训练，获得72组同状态反事实场景、216个候选分支配对、65/72存在非平局真实业务后果；决策损失梯度确实进入共享R-GCN转移网络。不过，小样本的H5/H20聚合状态预测（十个数字）不足以稳定选择更好调度：prediction-only测试后悔2.75tick、joint(λ=.25) 2.208tick、验证选择的flat对照2.25tick，均差于原CP-SAT首候选1.708tick。并且原joint的预测状态MSE比prediction-only更差。

这些结果是本轮**完整保留**的失败对照，不允许重新使用此前24个test组优化λ或挑选模型。

## 可迁移的已发表方法

1. [**Learning to Simulate Complex Physics with Graph Networks**](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)：迁移typed object/relation + message passing动态表示的工程设计。机器人、公开工序、工作站和资源作为不同节点类型；当前任务-工作站、工作站-资源、候选机器人-任务分配作为图边。使用已存在的**PyTorch Geometric RGCNConv 2.6.1**，不是从零设计“新GNN”。
2. [**Value-aware Model Learning**](https://proceedings.mlr.press/v54/farahmand17a.html)：状态重建与决策效果的目标错配。
3. [**VaGraM (ICLR 2022) 原论文方法与作者开源代码**](https://github.com/pairlab/vagram)：价值梯度可表示不同预测变量影响决策的敏感程度。我们预注册的`rel_entity_gradweighted`是**严格标注的简化迁移**：由不含新训练参数的H20“预计完成工序数”读取，计算对当前已知工序完成概率和新增工序完成数的固定梯度，再对这些状态预测误差做加权；**不是对原版VaGraM value-function fitting或loss的准确复刻**。
4. [**Calibrated Value-Aware Model Learning (ICML 2025)**](https://proceedings.mlr.press/v267/voelcker25a.html)：不能只看候选排序准确度，必须报告多步世界状态预测、设备可用性概率Brier、真实候选后悔值，避免决策损失导致模型预测不校准。

## A. 新的真实反事实世界状态监督

预注册 `experiments/world_model_integration/entity_pr52/pr52_protocol.json`；**228个从未出现过的新独立场景**，三类压力机制：14任务拥堵、两个先后故障的工作站、0/6/12/18分批到达任务。每类 train48、validation12、test16，共**训练144、验证36、独立test48**，各分组包含一半原合成故障持续时间、一半更长的故障时间。

真实场景构建、故障发生与持续时间及未来发布仍由**原 BusinessSimulator**作为唯一事实权威。每个场景取tick≥7且≤70的首次至少两个合法CP-SAT候选决策，统一此前真实动作序列/当前公开状态/候选池，每个候选只干预**当前一步**，之后仍采用相同的原CP-SAT首提案。对每分支保存实际完整makespan、逾期，以及锚点后的H5/H20公开观察。

**输入图严格只包含决策当时可见的节点和关系**；节点的来源ID仅在仿真事实标签构造内部用于将当前实体与未来观察对齐，从未作为模型特征传入。新发布、原锚点时不存在的任务不被提前添加图节点；只对其H5/H20新增可见工序数量和已完成数量建立匿名全局事件目标。

**逐实体6字段监督**（分别在H5/H20）：

| 节点类型 | 可观测未来状态 | 目标 |
|---|---|---|
| 机器人 | 公共位置x/y、是否有任务、取样/交付阶段、服务中状态 | 未来执行进度与机器人资源占用 |
| 工序 | 等待、已分配、运输中、已完成、持有机器人、工序编号 | 分配改变如何影响具体样本工序 |
| 工作站 | 设备可用、服务准入可用、占用比、容量阻塞、等待/接近任务数 | 恢复故障如何传播到服务与队列 |
| 资源 | 占用/剩余容量、等待/接近任务数 | 多站共享资源下的未来约束 |
| 匿名未来新增工序 | H5/H20新增可见工序数/完成工序数（非图节点） | **仍未知的**任务外生事件及新工序发展 |

没有在未来观察中可确定追踪的旧节点采用显式`node_target_mask=0`，不会人为填“完成”或者“消失”。若缺H5/H20未来真实快照，整组标记截尾/排除预测监督，并在正式统计里报告，不因候选真实效果不佳而只删除负面案例。所有真实分支要经过原容量、运动与workflow合法性校验。

这比上一轮的十个全局数更加细致，但仍属于**从当前状态/单个候选动作预测H5/H20的节点条件未来状态**；尚非反复递归采样全部未来新节点的可部署随机世界模型，不得夸大成已完成完整world simulator。

## B. 真实训练、严格消融及检验指标

`pr52_train.py`复用先前PyG节点/typed edge编码，学习两层`RGCNConv`共享关系转移表征；node head只输出各节点未来状态概率，event head只输出未知新增工序未来数量概率。所有模型共同的**固定**动作评分仍是：
```
negative_expected_completed_H20 =
    -(sum_current_task_node_future_completed_probability
      + predicted_anonymous_new_stage_completions_H20)
```
两个未来组成都是受真实公共状态标签约束的世界状态输出，**不存在独立的trainable Q/ranking head**。

已冻结4组训练目标，使用完全相同的原始数据来源、node mask、候选集合、模型读出规则：

- `rel_entity_pred_only`：实体/新工序预测损失。
- `rel_entity_joint025`：在预测损失上加权0.25的**同公开状态真实后果排序**损失，验证该项梯度能进入共享RGCN。
- `rel_entity_gradweighted`：只对固定完成进度评分**对世界状态预测的梯度敏感维度**补充加权平方误差(0.25)，借鉴VaGraM动机，但不是官方算法。
- `flat_entity_joint025_control`：去除消息传递，仅使用相同节点输入与独立embedding/head，测试关系建模是否真的贡献增益。

每组随机初始化种子1810和1811，训练45轮，Adam学习率0.0015，hidden48。总计8个受控训练实例。验证指标按**场景种子**计算（不是同场景4候选当4个独立样本）：

- H5/H20逐实体有效标签masked MSE；设备可用Brier；全业务完成工序预测MAE；
- 非平局候选排序accuracy；在同公共状态真实动作分支上的regret vs 最佳合法候选；CP-SAT首提案regret；
- 真正训练共享关系动态的**decision-only/value-gradient-only非零梯度**；全部动效不能仅停留在读出层。

按**验证集的2训练种子平均真实候选后悔值，预测MSE次之**选择方法；所有训练/验证/测试**物理拆分成不同JSON文件**，选择文件写出后才打开48组独立test。不使用已公开PR50测试场景，也不因本轮test结果重选λ或结构。

保存8份模型权重、optimizer、PyTorch RNG、source commit、环境锁文件与真实原始结果。此轮GitHub现有CPU runner可承担可复现实验，但**不把CPU训练吞吐当模型研究指标，也不宣称有可访问GPU**；最终GPU部署另需checkpoint迁移、GPU概率对齐与真实显存/时延验收。

## C. 阶段门禁与完整闭环

此轮最终希望拿到两个独立结果：

1. **世界模型本身**：逐实体R-GCN在新独立场景中能更好预测任务、站点和资源的未来联合状态，并有明确的任务/故障关系价值；
2. **决策感知训练的额外贡献**：同结构`prediction+decision`或预指定价值敏感损失，在**不会显著损害状态预测校准**的情况下，比prediction-only降低同公共状态合法候选的真实regret。

如果这两个目标任一不能完成，应保留全部失败模型和结果，优先查模型表达能力及数据/目标错误，不能从测试集挑选胜者宣称成功。

**本阶段还不会将新状态预测网络接入正式真实闭环调度**。必须先通过独立同状态候选后果与预测校准门禁，然后作为后续新PR的工程适配加入原业务合法CP-SAT滚动规划、粒子外生事件转换和全在线完成时间比较。没有完整rollout之前，当前网络仍是action-conditioned multi-horizon learned dynamics原型。

## 代码与复现

- `experiments/world_model_integration/entity_pr52/pr52_protocol.json`：新228独立场景、4模型、2训练seed、全部先验验收。
- `pr52_labels.py` / `test_pr52_labels.py`：真实未来节点标签、公开输入对齐与新任务不泄露的合同测试。
- `pr52_train.py` / `test_pr52_train.py`：PyG关系世界模型训练、固定H20决策读出、价值敏感误差与真实反事实排序损失、原始模型梯度验证。
- `.github/workflows/pr52-entity-labels.yml`：3个独立CPU-hosted GitHub Actions，原始228场景数据与SHA256归档。
- 训练工作流以原始源Artifact哈希为输入，物理隔离test并存储optimizer/RNG后，在同一运行内选择和独立评测。

**所有 PR 保持 Draft，未合并历史PR/PR19/正式V4路径。**
