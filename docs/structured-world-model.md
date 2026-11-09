# 结构化世界模型：状态、架构与训练边界

本轮选择**显式规则 + 动作条件实体集合网络**作为项目的首个正式候选架构。它由 PyTorch 的共享小型 MLP 对机器人、任务和工作站编码，经掩码平均池化形成全局上下文，再由运动候选评分头、工作站服务完成头预测下一步；任务生命周期、合法分配、地图邻接与离散避碰由已有规则执行。当前实现 4,642 个参数，支持不同数量的实体，并以一次调用一个有限候选分配作为条件。它是可训练的状态转移模型，不是图神经网络、GPT、JEPA、预训练基础模型或已经验证有效的调度策略。

## 调研依据与选择

| 架构 | 匹配情况与本轮决定 |
| --- | --- |
| 固定长度 MLP | 实现简单、CPU 便宜，但直接拼接的输入通常绑定机器人和任务数量；保留为将来消融基线。 |
| 集合编码 MLP + 规则 | 基于 [Deep Sets](https://arxiv.org/abs/1703.06114) 的共享编码和顺序不敏感聚合思想；可处理可变数量实体，改造现有 PyTorch 适配器成本较低。**选择本方案**。本实现采用均值而不是论文中的特定原式，并不宣称复现论文或继承全部理论性质。 |
| 图网络/GNN + 规则 | [Graph Network-based Simulators](https://proceedings.mlr.press/v119/sanchez-gonzalez20a.html) 以节点和消息传递学习动态，有[公开实现](https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate)。它能更明确表达任务—机器人—站点的边；但当前样本小、关系定义和跨规模测试不足，暂不增加图依赖。现有集合网络只用局部路径特征及聚合表示关系，不能表达全部成对关联，这是明确限制。 |
| 时序 GNN | 能保留多步关系历史，但需要更完整的事件时序和缺测处理；当前无证据证明其额外复杂度必要。 |
| Transformer | 可对可变实体做注意力；[MAPF-World](https://arxiv.org/abs/2508.12087) 是自回归 Transformer，预测局部观测和邻居动作，并有[公开代码](https://github.com/Fibonaccirabbit/MAPF-World)。它面向局部多智能体路径行为，输入与本站点、服务、任务调度接口不同；本轮没有证据表明可直接加载其权重用于我们的状态预测。 |
| 事件时间模型 / 时序点过程 | [Neural Hawkes Process](https://papers.neurips.cc/paper_files/paper/2017/hash/6463c88460bd63bbe256e495c63aa40b-Abstract.html) 建模历史条件下的事件概率及时间。它是未来到达率或故障风险的参考，不可将未来具体事件当作已知。本轮模型仍按离散时间步推演，**未实现事件跳步或故障发生时间预测**。 |
| 基于模型的规划框架 | [TD-MPC2](https://arxiv.org/abs/2310.16828) 有[公开实现](https://github.com/nicklashansen/tdmpc2)和[预训练权重](https://www.tdmpc2.com/models)，面向连续控制中的潜空间规划；其现成任务的状态、动作和奖励与样本转运并不兼容，需新的编码器、动作适配和领域训练。本轮不能声称这些权重能直接用于本系统。 |

本次复用 PyTorch、现有训练与检查点框架及显式仿真规则，**网络权重从随机初始化训练**；没有找到经过验证、可直接接受本项目任务与工作站状态的预训练权重。没有排除未来通过适配器、迁移学习或 GNN 取得更好结果；它们需实际数据与对照实验。规模很小，理论上适合 8GB/16GB 设备，但本轮未在任何 GPU 上测显存或训练速度。

## 状态 Schema v2

旧 `Observation`/Transition JSONL 保持 v1，历史数据和实验不覆盖；`state_v2.py` 仅把按时间顺序观察到的 v1 记录派生为内存中的 v2。每条训练输入只使用该时刻及之前记录，不能从 `Simulator.history` 或场景脚本取得未释放任务和故障计划。

| 实体 | 当前字段及含义 | 观察边界 |
| --- | --- | --- |
| 机器人 | 位置、归属任务、idle/to_pickup/to_dropoff、`service_elapsed` 墙钟时间、`service_active_elapsed`、`service_paused_elapsed` | idle 表示无分配任务，不等于机械正常；机器人故障可用性 `operational_available` 明确为未知。开始服务至当前时段必须连续观察到站点可用性，才能算有效及暂停步数；否则均为 `None`。 |
| 任务 | 已释放任务、状态、起终站、机器人、释放时间、已知截止时间、未分配等待、可选优先级 | 暂无优先级业务字段，`priority=None`，不虚构临床优先级；已分配任务的历史等待不伪装成当前未分配等待。网络用截止时间已知掩码区分“未知”和“恰好到期”。 |
| 工作站 | 当前可用、机器人占位、服务会话、服务是否实际进行、已观察有效服务均值与计数、相关等待及前往任务计数 | 会话可在故障期间存在，而 `service_in_progress=False`。任务数量只是当前目标和占用推导的需求代理，并非 FIFO 实体队列；`history_complete_since_start` 标明观测是否从零时刻连续。 |

在当前仿真中，服务会话遇到可用站点时每步推进一次；因此连续观测可以恢复有效进度和已观察到的完成服务均值。在真实实验室，这一推算依赖可靠服务开始信号及持续的可用/处理状态遥测；仅有粗粒度站点状态时不能保证，需提供设备处理进度接口或保持未知。观测计数只包括能从连续状态变化识别的服务，不保证覆盖全部历史。旧 `expected_service_ticks` 的历史平均把故障暂停算入墙钟时长，因此不是有效时长真值；本模型优先使用可观察到的有效均值，缺少该统计时才回退到旧有公开先验/均值作为近似。规则基线也使用同样的可观察修正，不读取真实剩余倒计时。

## 预测目标与规则

模型的两个学习输出是：在当前地图合法候选中为机器人的下一步运动打分；对正在可用站点服务的任务预测本步是否完成。候选分配同时进入实体特征及机器人的目标路径特征。显式规则据此更新任务状态、机器人占用与可分配状态，再派生站点占用、服务和已知任务积压。多步预测递归使用自己的预测，故会累积误差。

网络不单独回归真实剩余时间，不生成新任务，也不预测未知故障恢复时刻；未来站点状态按当前状态保持直到下一次实际观测。动作是当前有效任务分配，未实现任务改派、站点切换或可预测的未来外生事件。站点队列、机器人执行延迟风险及时间分布尚无专门输出。当前推理是单步分类阈值与点估计，不是经过校准的置信区间。现实中有同一实体关系模糊或历史缺测时，可能无法从状态识别真实服务剩余时间；不能把它解释为模型推断失败的充分证据。

## 数据与本地训练协议

现有 `training_data export --profile full` 固定使用 `SPLITS_V2`：Tartu 训练种子 3000–3011、验证 4000–4003、同地图测试 9000–9003，clinic 跨地图测试 10000–10003；每个种子包含 normal/short_outage × nearest/world 的事实执行轨迹。导出器检查任务序列跨分组重复、源文件哈希和状态信息边界；终止边界按现有数据合同处理。固定仅使用 train 更新权重、validation 选检查点，两个 test 只用于报告；真实后续结果和反事实分支不作为在线输入，未执行分支也不当作事实标签。

建议首个本地 GPU **小规模工程验收**固定 `batch-size=128`、`learning-rate=0.001`、`seed=123`、10 epoch 上限，以验证集损失选择检查点，报告训练时长、显存、已有规则及修正规则、1/3/5 步条件预测覆盖和完整闭环任务指标；不得按测试集调参。此协议适合检验数据和训练流程，不等于已确定论文最终的训练方案。现有训练分布只有正常和短故障，先前短故障经常无实际服务等待，欠缺有覆盖的处理中断、积压、延迟和更多机器人数量的事实训练数据；不能据此作异常场景泛化结论。下阶段应先补充有故障曝光标记的训练/验证集，并按布局、任务生成种子和场景分组隔离，固定数据后再做故障学习效果比较。

CPU 复现（需 Python 3.11/3.12 和 `requirements-torch-cpu.txt`；各输出目录须为空）：

```bash
python -m pip install -r requirements-torch-cpu.txt
python -m unittest discover -s tests -v
bash tools/structured_cpu_acceptance.sh runs/structured-acceptance
python -m lab_transport.long_score_experiment --profile development --output runs/long-score-development
python -m lab_transport.long_score_experiment --profile test --output runs/long-score-test
```

本地 CUDA 机器需要按 [现有 PyTorch 说明](pytorch-training.md) 安装与驱动兼容的 CUDA 版 PyTorch，不应先安装 CPU-only 轮子再宣称用了 GPU。数据准备和训练示例：

```bash
python -m lab_transport.training_data export --download-sources --profile full --output runs/structured-data
python -m lab_transport.training_data validate --dataset runs/structured-data
python -m lab_transport.torch_training --dataset runs/structured-data --output runs/structured-gpu \
  --adapter lab_transport.torch_structured_adapter --device cuda --epochs 10 --batch-size 128 --seed 123
```

使用对应验证集选出的 `epoch-XXXX.pt` 运行 `lab_transport.torch_evaluate` 和 `lab_transport.torch_rollout`，完整命令参见原 PyTorch 文档；GPU 验收必须实际检查 `device` 与日志。检查点同时记录适配器、训练合同、仿真与 v2 特征代码哈希；因此旧版训练检查点应在其原代码版本加载，新版不会绕过完整性校验。公开源地图与训练数据可根据许可证在本地重建，不随本 PR 覆盖或重新分发历史数据。
