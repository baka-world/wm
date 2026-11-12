# PR #52: Action-Contrast and State-Consistent Relational World Model Learning

## 研究动机

延续此前 [Draft PR #51](https://github.com/lieuyimo/wm/pull/51) 真实逐实体世界模型学习成果。已在三类合成实验中获得228个独立真实场景、912次同一历史状态候选分支、8个真实PyG R-GCN训练检查点、H5/H20逐实体状态预测。此前模型在公开的新48组场景中，联合决策损失的2个训练seed平均后悔值从prediction-only 1.625降为1.240tick，但状态预测MSE 0.02571→0.02961，训练seed影响明显。

更严格的只读复现 [Actions #43180194193](https://github.com/lieuyimo/wm/actions/runs/43180194193)：原始八份冻结模型在同样48组的模型重放与训练报告完全一致。PR51模型中，仅预测组**38/48**场景的前两候选预测完成分数差小于1e-5，说明**当前动作变化未有效传播到可排序的未来状态**；PR51已公开test用来发现这个问题，因此不能继续拿它作为新方法的独立验证。

复现还发现：关系joint对prediction-only的后悔值差异−0.385tick，按真实场景分层bootstrap95%区间[−0.896,+0.094]跨零；value-gradient-inspired相对prediction-only −0.365tick，区间[−0.688,−0.094]。后者只是固定两训练seed与已公开场景上的线索，**不得事后改称原先验证选择的joint已变成价值梯度模型**。

## 相邻方法迁移边界

本次不增加一个可直接预测最终后悔的独立训练Q头，而是改良**动作条件下的世界状态动态**。

- [Learning to Simulate Complex Physics with Graph Networks (ICML 2020)](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html)：typed graph object interaction与动作条件化动态的设计经验。现有系统迁移为robot/task/station/resource节点及合法robot-task动作边，仍用官方PyTorch Geometric `RGCNConv`，不从零设计网络框架。
- [3D-OES: Viewpoint-Invariant Object-Factorized Environment Simulators (CoRL 2020)](https://proceedings.mlr.press/v155/tung21a.html)：对象分解和动作引起的变化建模，可作为世界状态逐实体预测的参考，**并非原样复现图像/3D体系**。
- [Calibrated Value-Aware Model Learning with Probabilistic Environment Models (ICML 2025)](https://proceedings.mlr.press/v267/voelcker25a.html)：未经校准的价值感知损失会偏离真实状态分布，故必须同时监测逐实体状态MSE和设备Brier。我们不自称实现其特定校准校正损失。
- [Boosting World Models Learning via Latent-Space Value Alignment (ICML 2026)](https://proceedings.mlr.press/v306/jiang26ai.html)：将重建可靠性和与价值有关的表征结合，说明只靠一般预测与只靠决策目标都不一定优。我们的损失不等价于该文具体的latent alignment方法。

**方法区别与创新检验目标**：同一可公开状态、同一合法CP-SAT候选池，在事实外生扰动相同条件下真实模拟一次动作差异，直接监督该差异对未来机器人/任务/站点/资源系统状态的作用；限制工序互斥状态和匿名新工序完成数的一致性。它是一项**提出并待验证的场景适配性反事实状态差监督**，不冒充已有论文精确公式。

## 真实业务数据：全部使用全新种子

`experiments/world_model_integration/action_pr53/pr53_protocol.json` 预注册三类源自原BusinessSimulator合成负载，拥堵/多设备故障/分批到达各76场景（train48、validation12、test16）。总**228个新独立场景**，train144、validation36、TEST48。全部种子为新的4300万命名空间，与PR50/PR51已公开数据完全不交叠。新真实事实标签采集直接复用先前审计过的`entity_pr52.pr52_labels.collect_case`，不修改原CP-SAT动作、状态可观测性、合法工作流、故障隐藏机制与运动约束。

原真实数据由BusinessSimulator计算、一次只变更一个当前合法候选动作，随后都按同一CP-SAT规则续执行。每个候选H5/H20真实逐实体状态与事件统计作为**预测标签**；完整makespan只作为训练决策目标，不作为模型输入。无未来任务表、真实设备故障结束时间及未来任务ID泄漏。测试选择只能在**新36验证集**完成；48个全新TEST物理分离并在选择文件保存后才读取。

## 关系世界模型训练架构

从上一轮真实完成训练的两层官方PyG R-GCN结构迁移，不加入独立候选评分参数。预测每个公开可见的机器人、任务工序、站点和资源H5/H20未来6字段，外加匿名新工序未来可见数和完成数。所有模型共同使用**原有固定读出**：选择H20预测完成工序数最大的合法CP-SAT候选，而不是训练一个新Q头。

新增两类系统世界预测约束：

1. **工序状态互斥**：每个已公开工序在同一未来时间点的waiting / assigned / in_transit / completed预测由softmax形成合法的概率单纯形（而不是四个可以同时为1的独立Bernoulli）。
2. **匿名未来工序一致性**：模型的未来新增完成工序数不能大于未来新增可见工序数。以`P(completed|visible) P(visible)`参数化，不把具体未来任务ID加到当前输入图。

新状态差预测损失，令`i=0`为同一状态的原CP-SAT首候选：

```
Δ_pred(i) = fθ(public_state, legal_action_i) - fθ(public_state, legal_action_0)
Δ_true(i) = next_state_truth_i - next_state_truth_0

L_delta = masked MSE(Δ_pred(i), Δ_true(i))
          + 0.3 × MSE(anonymous_future_event_deltas)
L_total = L_state + γ L_delta + λ L_dec + μ L_value_sensitivity
```

`L_delta` 使用的是H5/H20**真实公开未来状态差**而非总体episode收益标签。mask取两个候选实际可观测的相同旧节点交集；未知新任务没有单独节点目标。这个损失可直接反向更新共享R-GCN状态转移参数。已有`L_dec`延续真实同状态候选完整makespan的非平局pairwise损失；`L_value_sensitivity`是先前冻结世界状态完成数读出的简化价值梯度误差权重，不是VaGraM原式。

## 预注册5组方法和3训练seed

| Arm | 物理预测一致性 | γ 状态差 | λ 决策 | μ 价值梯度 |
|---|---|---:|---:|---:|
| `unconstrained_pred_control` | 否（历史原型） | 0 | 0 | 0 |
| `consistent_pred_only` | 是 | 0 | 0 | 0 |
| `consistent_action_delta` | 是 | 0.5 | 0 | 0 |
| `consistent_delta_joint` | 是 | 0.5 | 0.25 | 0 |
| `consistent_delta_value` | 是 | 0.5 | 0 | 0.25 |

每组用1810、1811、1812三个预先固定的初始化，共**15个真实世界模型训练实例**；45 epochs、hidden48、Adam 0.0015，固定同一个H20读出。仅依据验证集**三个训练seed的平均真实候选后悔值**选择模型，并加一道校准门槛：逐实体MSE及设备Brier均不得超出`consistent_pred_only`验证均值15%；同分选择状态MSE更低者。除非历史对照与新方法同样满足此门槛，否则不能以有害于状态预测校准的模型声称改进。

正式指标：逐实体masked MSE、工作站设备概率Brier、真实完成进度MAE、候选排序的非平局准确率、实际同状态真实候选后悔值、CP-SAT首提案后悔值、两候选分差小于1e-5/1e-3的场景数量。结果按场景等权，严格区分模型seed和真实业务场景；TEST绝不调损失权重。

本阶段依然是H5/H20的**直接条件化状态预测模型**，不是完整随机多步递归世界模型、也不是已验证的真正在线闭环部署。最终部署目标为GPU，当前GitHub CPU环境仅用于可复现实验和checkpoint归档，不把CPU时延作为研究选择目标。
