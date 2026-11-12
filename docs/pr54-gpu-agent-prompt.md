# 可直接下发给有GPU执行环境的Agent：PR54关系时序世界模型GPU训练与完整动态迁移

你是一名熟悉 PyTorch Geometric、世界模型、事件驱动调度和真实可复现实验的研究工程师。请在 `https://github.com/lieuyimo/wm` 工作，优先执行代码与实验，禁止仅交付技术方案。

## GPU阶段触发依据与不可更改事实（更新于2026-11-12）

- **CPU阶段已通过**：真实公开状态3360条、60个场景；[Actions #43191970049](https://github.com/lieuyimo/wm/actions/runs/43191970049)。训练和独立测试[Actions #43192730343](https://github.com/lieuyimo/wm/actions/runs/43192730343) **SUCCESS**；权重/Adam/RNG归档 [#13930509625](https://github.com/lieuyimo/wm/actions/runs/43192730343/artifacts/13930509625)。GitHub现有托管CI只有CPU。
- 原始可验证source artifact ID：`13930164510`(congestion)、`13931312132`(multi_fault)、`13931307297`(staggered_arrivals)，均来自Actions `43191970049`、source HEAD `60c951ac1affc60136f4c758a76767137a518e63`。
- 新独立12场景，快照only/4步历史GRU同参训练两seed，历史版本node MSE约**0.0102862**、快照版本约**0.0121319**，改进**15.2%**，但历史设备可用Brier **0.0098502** 比快照的 **0.0093300** 略差。
- **必须把设备Brier退步列为明确工作项目**。新GPU方案须先给出设备恢复风险概率独立校准目标、按设备类型/历史down时长的状态分组；不得仅用整体node MSE改善宣布完整世界模型已得到改进。
- 当前历史训练TEST已公开。GPU阶段改动模型结构或超参数后，必须生成完全新的独立TEST并先固定验证集模型选择，不得把原12个TEST作为新的盲测。

## 执行前提和版本安全

1. 从 **`research/pr54-temporal-relational-transition-20261112`** 开始，读取最新HEAD，不可假设某个历史 SHA 仍然最新；审查本分支的 `docs/gpu-handoff-pr54.md` 和 `experiments/world_model_integration/temporal_pr54/protocol.json`。
2. 检查PR54 CPU GitHub Actions 是否完成公开真实时序数据、模型梯度和独立验证。未通过，先在CPU或源代码修复根因并审查数据合同；**禁止直接开始昂贵的GPU扫参**。
3. 创建新的GPU训练研究分支和 Draft PR，保留原PR、原V4 checkpoint、CP-SAT候选算法、PR19工程演示，不合并，不覆盖历史训练权重或固定TEST。
4. 执行环境必须实际拥有CUDA GPU；先打印 `nvidia-smi`、`torch.cuda.get_device_name`、CUDA版本、PyTorch/PyG版本、显存。不能把CPU训练冒称GPU，也不能在本地GPU不可用时宣称已跑过。

## 任务一：恢复真实数据与基线

- 查找PR54最近一次**SUCCESS**的 `PR54 factual public history one-step transition dataset` Github Actions，下载3类场景20回合的原始Artifact，执行 `sha256sum -c SHA256SUMS`，确认三类来自同一源HEAD及协议SHA。
- 将60个全新真实场景物理拆分为train36、validation12、test12，使用源文件中的split字段，不得根据结果重新分组、删场景或改变种子。
- 每个模型输入严格限制为t-3到t的真实BusinessObserver公开观测及t时刻合法选定robot-task动作边。**所有t+1及以后状态、真实故障时长、未来到达表、尚未发布任务ID只能成为标签或真实环境，不可进入张量特征。**
- 先执行当前PR54完整的测试，核对原CPU训练选出的历史/快照模型与最新参考结果。

## 任务二：正式GPU基线复现

- 必须用CUDA版PyTorch及官方`torch-geometric==2.6.1`，如原锁是CPU torch，建立**单独GPU环境文件或容器**，不改仓库原依赖锁，不静默改变PyG、torch或数据。
- 执行命令：
  ```bash
  python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
  python -m experiments.world_model_integration.temporal_pr54.pr54_train \
    --input "${DATA_ROOT}" --out "${OUT_DIR}" --device cuda
  ```
- 保持快照only / 4步公开历史GRU两个同参模型、隐藏维度48、2个预注册训练种子、学习率和epoch不变；只依据VALIDATION选模型，selection.json落盘前不得读取TEST。
- 输出场景等权node masked MSE、工作站设备Brier、新工序数MAE，和简单状态持续不变(persistence)基线；报告CPU/GPU一致性、算力/GPU峰值显存。核查近似平分、概率取值范围，尤其不要用更高评分掩盖未来状态错误。

## 任务三：迁移成熟的动态模型并完成容量审查

在任务二基准通过后才开始，在新PR预注册新训练和新独立测试：

1. 基于现有 **PyG RGCNConv + GRU**，优先保留事件驱动业务状态和规则约束，不重写底层地图/移动算法。
2. 从**至少200个独立训练场景**扩大可观察真实状态转移数据，分别覆盖拥堵/多设备故障/分批到达、不同机器人任务负载和故障时长。与此前已公开的PR50–PR54所有测试集seed隔离，预注册TRAIN/VAL/TEST，所有真实业务轨迹从BusinessSimulator公开回调获得。
3. 做严格控制变量试验：history window 1/4/8；hidden48/96/128；两层/三层RGCN；只有当前图输入 vs 实时历史；先检查train/validation误差、设备Brier、模型容量/吞吐，再扩参数。若网络变大而验证没有改善，停止扩容。
4. 从**真实单步动态**构造递归 `predict_next`；未来未知任务到达、服务随机事件和设备条件恢复必须来自显式概率事件模型；由业务约束的独立规则层处理任务派发、工作流衔接、资源容量与通行。不要偷偷查询未来真实私有故障时间或释放表。
5. 只在合法预测状态可递归展开后，评估H5/H20及多粒子rollout误差/校准/机器人任务站点的实际状态演化。不能将独立的H20回归网络直接称为递归世界模拟器。
6. 最后用同候选和同预测horizon比较：预测only、历史关系动态、价值敏感学习、预测＋决策学习、冻结V4＋末端ranker负对照。决策损失必须在**决策损失单独反向传播时**对共享转移参数产生梯度；状态预测校准不得随意牺牲。
7. 仅在新训练/验证与模型选择完整冻结后，第一次打开新独立TEST，先统计真实相同状态候选后悔值，再接入原CP-SAT滚动决策流程，测试真实完整回合完成时间、逾期、容量/运动合法性；报告场景等权的配对置信区间与多训练seed变化。

## 任务四：资源、归档与验收

- 初始资源上限暂定**单GPU累计4卡时**，允许据实测提出下一阶段预算。第一轮配置应按设备显存试探，不得假定V100 16GB支持任意大小模型/批量。
- 保存best/latest checkpoint、模型权重、优化器与随机数状态、CUDA/驱动/torch/PyG环境、源码commit、数据协议SHA、test开放门禁、真实GPU秒数/峰值显存、每组独立场景原始结果及全部阴性对照。所有结果上传Github Actions或者新的不可变仓库归档路径，附完整SHA256及恢复命令。
- **停止条件**：数据泄漏、未来任务节点提前可见、校准/安全约束不成立、模型未能学会行动条件状态变化、验证集持续无收益、GPU预算用尽。允许明确NO-GO，不得通过删场景/改随机种子/测试调参粉饰结论。
- 输出Draft PR链接、训练权重归档、GPU实测消耗、全部消融表、独立测试结论、未来正式调度部署的接口风险。没有真实运行过的步骤直接标记为未完成，禁止提供臆造的GPU结果。

**研究核心问题**：在保持对机器人、任务、工作站、资源的未来状态预测可信和合法的条件下，学习动作条件化的关系动态及价值相关信息，是否能够稳定减少真实多机器人调度后悔值和系统完成时间？世界模型为第一优先级，不能把结果偷换成单纯CP-SAT评分网络改造。
