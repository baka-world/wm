# PR43：世界模型未来状态信息到调度效用的迁移验证

## 起点与研究问题

PR42 对已用过的 8 个 PR30 开发场景实际诊断：V4 数值编码的完全碰撞 0/8；完整 H20 多字段状态摘要打平 0/8；但在 shift 有 2/8 的 H20 预计完成工序数打平。说明应当先审查 **预测分布 → 调度动作评分** 的信息利用，不能直接宣称 Deep Sets 无法表示动作关系或立即改造 GNN。

**研究问题：** 若冻结同一个 V4 世界模型，是否能通过复用已有的轨迹级状态代价，改善完整在线调度，尤其是跨服务参数 shift 的迁移？

## 方法迁移，不另造算法

- Chua et al., *Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models* (NeurIPS 2018), [论文](https://proceedings.neurips.cc/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)：概率模型粒子传播与估计期望轨迹代价。这里仅复用粒子评分思想，不宣称复现 PETS 的学习集合、CEM 或控制结果。
- 仓库 PR27 已有的 `experiments/world_model_integration/decision_value/pr27_evaluate.py::_trace_cost`：在预测粒子上累计未完成任务随时间的代价、逾期加权，并在 H20 末端计算剩余行程。**直接调用该函数，权重、目标和实现不变**。
- 仓库 PR30 原有 `OnlineHorizonPolicy`：冻结 V4、H20/P3、最多4个合法CP-SAT候选、预测状态上CP-SAT续调度、CPU进程时间超预算回退到合法首候选。

## 首次实验前冻结的四个在线臂

| 臂 | 唯一变化 |
|---|---|
| CP-SAT | 原始 CP-SAT 首合法提案 |
| V4-completed | PR30 原 H20 预计完成工序数最大化 |
| V4-tie-trajectory | 保持上述完成工序数为第一目标，**仅当完成数打平**，再按 PR27 原轨迹代价、末端剩余距离排序 |
| V4-trajectory | 将 PR27 原有轨迹代价作为第一目标，以末端距离破平局 |

所有臂都通过同一 `BusinessSimulator` 实际执行从开始到终局的完整在线重调度。后续预测的工作流动作只基于**预测状态**，不能读取事实未来。

## 独立场景与评估

`pr43_protocol.json` 在首次评估前固定 24 个此前 PR30 未使用的新种子，light/medium/heavy/shift 各 6。保留相同业务参数结构；比较各策略的真实完成 tick、逾期任务、容量等待、受阻移动、胜平负、改派次数、预算回退、CPU 开销、安全违规。此 24 组是独立于 PR42 的**新合成在线评估**，不是跨地图泛化、企业设备遥测或统计定论。

不根据 PR42 shift 的两组结果调权重。PR27 中的未完成每 tick 代价和逾期权重严格保留，模型、训练数据、评估随机种子和已归档历史全部不修改。

### 必须避免的错误解释

- PR42 “完成数打平但未来状态不同”不意味着某一额外指标一定能改善真实完成时间。
- V4-trajectory 若有提升，只能证明既有状态预测在新的效用读出下被利用；不证明神经世界模型本体预测更准确。
- 若 Rule-H20 之类规则基线收益相近，不能归功于 V4 的学习型动态。
- 受 0.5 秒 **CPU process time** 限制；模型单次 rollout 无法被真正抢占，过界后回退不代表硬 wall-clock SLA。
- 各臂随真实在线决策偏移会产生不同后续状态，因此逐时刻调用次数不可直接解释为完全相同输入上的打分对照。
- 实验只用现有合成环境，并不预示现实设备故障预测可用。

## 复现入口

仓库根目录，CPU依赖，已校验 SHA256 的原 V4 检查点：

```sh
python -m unittest experiments.world_model_integration.forecast_utility.test_pr43 -v
python -m experiments.world_model_integration.forecast_utility.pr43_online --checkpoint /path/to/v4-data.pt --out /tmp/pr43-online.json
```

工作流 `.github/workflows/pr43-forecast-utility-cpu.yml` 从既有官方原冻结 V4 artifact 恢复 SHA 校验通过的文件，保存原始逐场景 JSON、测试日志、指标以及源版本 SHA。失败时不得凭推测声称结果。

**后续裁决**：只有在新独立场景里、预算合规且完成时间或逾期实际改善，才能考虑进一步验证。否则回到动作选择敏感性、预测路径和目标可识别性，而非继续增加模型训练轮次。
