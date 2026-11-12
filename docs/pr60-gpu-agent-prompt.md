# 可直接交给 GPU Agent 的执行指令：PR60 → 可递归公开地图世界模型

你是一名熟悉 PyTorch Geometric、真实环境动作条件状态转移、概率预测校准、多机器人资源/路径/站点系统和可复现实验的研究工程师。**本任务必须真正使用用户授权的GPU执行；不得把CPU仿真或模拟指标伪报为GPU训练。**

## 0. 项目优先级与起点

仓库：<https://github.com/lieuyimo/wm>。在最新的`research/pr60-marginal-high-information-cpu-data-gpu-handoff-20261112`基础上核查最新HEAD，新建GPU实验分支和Draft PR，严禁合并历史PR、改原正式CP-SAT/V4或覆盖历史模型权重。

本科毕设：《基于世界模型的样本转运调度系统设计与实现》。上层多机器人样本运输调度，世界模型通过公开状态+合法当前动作学习任务、机器人、工作站、资源站点及可通行地图的未来状态转移；不训练底层VLA/人形运动控制。

**最高指令：世界模型状态转移学习是第一优先级。没有可靠的合法动态预测，不启动只针对在线分数的Q/ranker优化。不可为了支持预设结论而放宽门槛或在独立测试上调参。**

## 1. 开始前必须阅读与核验

1. `docs/pr60-gpu-research-handoff.md`，有完整原始数据Artifact、SOURCE-SHA、下载命令、真实测试集封存规则。
2. `experiments/world_model_integration/sealed_pr59/pr59_dual_model.py`，官方PyG的public map SAGEConv + GRU/RGCN world state transition，包含跨图batch独立ID空间。
3. `experiments/world_model_integration/sealed_pr59/pr59_gpu_train.py`，CUDA-only训练入口，产生best/last完整模型、Adam、CPU/CUDA RNG、validation-only选模和`selection.json`。
4. `experiments/world_model_integration/sealed_pr59/test_pr59.py`，CPU真实第一步公开业务状态、map batch索引与梯度合同。
5. `experiments/world_model_integration/sealed_pr59/protocol.json`、`experiments/world_model_integration/expand_pr60/protocol.json`，TRAIN地图、VALIDATION不同地图族、**BLIND TEST仅预注册未生成**，不得扩散私有未来信息。
6. PR59真实质量审计：[Actions #43235876892](https://github.com/lieuyimo/wm/actions/runs/43235876892)，通过`PASS_QUALITY_TO_HANDOFF_GRAPH_TRAINING`。
7. PR60真实数据质量审计：[Actions #43237354137](https://github.com/lieuyimo/wm/actions/runs/43237354137)，通过`PASS_60_TRAIN_TRUE_DYNAMICS_READY_FOR_GPU_AGENT`。数据真实来自BusinessSimulator，而不是文字计划。

核实数据：TRAIN60独立真实场景、9,046条完整相邻状态转移；VALIDATION12独立场景、1,517条；BLIND TEST12个仅有地图/场景配置的留出案例，**无测试标签**。新36训练有5个时间截尾，原24训练有3个截尾；截尾是合法但不完整，**不能把时限260 tick当真实完成时间**。新36里有36/36真实H20合法动作状态效应，36/36效果L1≥0.01；48故障/48恢复；相对旧24训练状态语义新颖度87.14%。

## 2. 资源与环境纪律

- 先确认真实`nvidia-smi`、设备型号、显存、CUDA运行时与驱动，再建立与当前GPU匹配的**CUDA版PyTorch**和兼容PyG环境，**不能直接沿用GitHub CPU专用torch构建**。
- 第一批申请或使用：**单张V100 16GB或以上，4 GPU卡时的阶段上限**。优先做两arm各2 epoch×1seed GPU smoke，测真实秒/epoch、峰值显存、map-encoder有效梯度和CPU/GPU预测数值差。
- 如实际显存/吞吐超过预算，停止并交出真实资源审计及下一批需求。不得直接上96/128维耗尽GPU。
- 本阶段所有源模拟数据已经在CPU完成。**不要重复采集或把模型训练进CPU数据质量工作流。**

## 3. 真实源恢复

必须严格执行`docs/pr60-gpu-research-handoff.md`中的`gh run download`与所有`sha256sum -c SHA256SUMS`，验明原PR59 24 TRAIN、12自然VAL，及PR60三组各12新增TRAIN与两个PASS质量档案。

在当前PR60源码上的命令（GPU smoke）：

```bash
python -m experiments.world_model_integration.sealed_pr59.pr59_gpu_train \
  --train /tmp/wm-gpu-source/pr59-train/data.json \
    /tmp/wm-gpu-source/pr60/pr60-true-new-training-43237354137-congestion/data.json \
    /tmp/wm-gpu-source/pr60/pr60-true-new-training-43237354137-multi_fault/data.json \
    /tmp/wm-gpu-source/pr60/pr60-true-new-training-43237354137-staggered_arrivals/data.json \
  --validation /tmp/wm-gpu-source/pr59-validation/data.json \
  --output /tmp/wm-gpu-artifacts/pr60-gpu-smoke-unique \
  --git-head "$(git rev-parse HEAD)" \
  --hidden 48 --epochs 2 --batch-size 16 --seeds 1810 \
  --arms map_off,map_on
```

`--output`必须不存在；不能覆盖前次检查点。先修复所有CUDA/PyG/批处理错误、NaN或模型梯度无法到达的情况，**重新跑真实烟测**；修复必须留审计commit与原因。

## 4. 正式第一批世界模型训练

固定模型架构：`PublicMapTemporalRelationalWorld`中**公开地图SAGEConv×2**、每实体公开t−3..t **GRU**、动作/资源站点**RGCN×2**，节点t+1状态预测头+匿名新增任务事件头。对于同参数`map_on`和`map_off`，只遮蔽地图信息，`hidden=48`，训练seed至少1810、1811、1812，每臂20epochs（按真实GPU smoke测得开销可在TRAIN/VALIDATION前预先冻结缩减）。

训练以真实已执行合法CP-SAT派单下的下一步公开实体状态为监督：逐实体masked MSE、任务工序互斥状态、匿名新增事件MSE，并以验证集场景宏平均状态MSE+设备可用性Brier的预定义线性组合选best。不读取BLIND TEST。保留每seed每臂best.pt、last.pt、Adam状态、RNG、指标定义、验证集每场景结果与源SHA。

比较项目：

1. `map_on` vs **同参数** `map_off`；
2. 时间历史`t−3..t` vs当前快照（现有`snapshot_map_off`入口，若做地图快照对照，保证参数量和训练预算一致）；
3. 校准：各设备on/off及设备恢复子群Brier、按故障历史/站点分组；不能只展示整体MSE；
4. 业务预测：工序完成、队列、资源占用、机器人相位、匿名新任务计数和预测概率合法性；
5. 统计：独立场景与训练seed都是层次，不能把9,046个连续相关tick视为9,046个独立环境。

如果`map_on`没有稳定好于`map_off`、设备Brier明显退步，或换验证地图后崩溃，请在正式报告明确`NO-GO`并做只读信息可观测性/校准分析，不得重新编辑VALIDATION以让结果更好。

## 5. 真正H5/H20与决策感知的递进门槛

这次交接的`pr59_gpu_train.py`只训练**真实一步teacher-forced世界状态**。它尚不能把隐藏服务计时、未来任务发布、设备故障/恢复变成自洽可执行多步事实，也没有独立在线调度收益证据。

- G2：只有一步预测有效，才设计**合法动作条件单步状态递归**：随机外生设备/任务事件必须是面向公共历史可验证概率，完整资源占用/任务工作流由原业务规则保证；再做 H5/H20滚动预测误差与分组校准，并与旧V4和合法规则世界模型对照。参考现有PETS多粒子与图网络世界模型方法，**不要从零另造潜在规划器**。
- G3：确认不同合法候选动作的H20预测分布排序与真实同状态干预事实有足够信号，再考虑共享动态参数上的决策损失，而非独立Q头。实验必须报告预测损失、设备概率校准、候选排序和独立真实仿真在线完工时间，显式消融。
- G4：模型选择和所有超参及最佳权重**先在验证集冻结到`selection.json`并提交**。之后另建受到该文件哈希保护的BLIND生成与只读评测入口，单次评估此前从未生成的横向层架`test`地图家族，不泄漏未来私有故障/任务表。没有GPU结果之前**不得把现有模型称为完成的H20世界模型**。

## 6. PR和成果交付

必须在**新Draft PR**归档：全部代码、真实动作模型计算路径、真实三seed曲线、设备校准报告、跨地图验证、模型权重best/last、Adam与CUDA RNG、pip freeze、NVIDIA driver、GPU架构与显存峰值、GPU卡时、所有原始数据/源码/checkpoint SHA、可恢复命令和失败样本。

不合并原PR、不过度训练、不中途用BLIND做选择、不给“实验预期”冒充“实际结果”。

**第一件真实工作就是按项目GPU入口跑2 epoch两arm的CUDA烟测；本ChatGPT会话没有运行GPU训练，后续全部交给你执行。**
