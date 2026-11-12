# PR45：世界模型显式条件恢复事件与 H20 系统状态转移验证

## 项目阶段

本项目为《基于世界模型的样本转运调度系统设计与实现》。本轮沿用 PR43 冻结 V4-H20 与 PR44 已训练的公开停机恢复生存估计，不进行任何 V4 再训练；目标是将**已观测停机后的恢复事件**纳入研究性世界模型粒子预测，并检验机器人、任务、服务和资源状态的完整联动，而非另造一个调度评分器。

PR44 归档：GitHub Actions #43156302837，Artifact #13915489041。**只加载 train-only 的 station-KM 曲线**，不加载 PR44 的 heldout 真实标签来拟合或调整参数。本轮新场景种子与 PR44 的36训练、24测试均不重叠，但共享相同人为合成故障持续时间机制、同一地图，不构成真实设备或外部分布泛化。

## 依据与迁移

- 原有项目 `HybridSystemWorldModel`：V4 学习**已启动服务完成事件**，其他机器人移动、容量与工作流状态由合法规则推进；默认完全不产生未来外部故障或恢复。
- PR44：Kaplan–Meier 生存估计按站点、已知停机年龄条件化，估算 (P(T\leq a+h\mid T>a)=1-S(a+h)/S(a))。本轮仅对**调度器已经连续公开观察到**的站点停机预测恢复；未来新故障仍然不可知。
- 粒子模型预测控制（PETS 等）：将不确定动态事件按固定独立种子采样，在各粒子上继续执行相同公开策略，形成后续状态分布。本轮直接复用原 `ForecastBatch` 与 `_distribution`，不新增网络、训练损失或优化器。

## 明确分离的两个状态转移契约

1. 原 V4 的 `predict_next`、`check_business_transition` 完整执行正常 ([t,t+1)) 状态转移，服务在本 tick 仍按恢复前的设备状态处理，不得因提前恢复而在过去一个 tick 偷跑完成。
2. 在 **t+1 公开观测边界**单独采样合法外部恢复事件。只允许初始已观测为故障的设备 down→up；不得生成新故障、外部任务或未观测的恢复计划。事件应用后调用原 `capacity_view`、`rebuild` 重新计算可入站状态、站点/机器人服务关系、资源占用。新增 `check_recovery_boundary` 确认机器人、样本、clock、地图和容量等均保持原值，只有被列名的设备可用性和对应派生服务字段发生可验证变化。

**普通业务 checker 不被修改或削弱。** 异步外部恢复被显式标记为独立第二阶段合同。没有观察到停机开始时刻（例如中途接入设备）则 down age 为 Unknown，保守保持当前状态；没有生存风险集支持也不外推。

## 冻结 H20 评价

`experiments/world_model_integration/recovery_integration/pr45_protocol.json` 已在首次运行前固定：

- 新 12 个合成种子，目标站 B0、B1、C 各 4；初始真实故障由业务场景生成，故障真实结束时间**仅供真实仿真环境使用**，不得进入 V4 或恢复模型。
- 对同一事实公开状态的 tick 8/10，只在当时工作站**已观测停机且有连续停机年龄**时发起预测；连续观察不到真实未来终点，则跳过而非标为负例。
- 两臂：`original_v4_h20_no_future_faults`（原 V4 设备保持）与 `v4_h20_observed_down_km_recovery`（原 V4 + 显式恢复事件）。均为 H20 / P3；H5 同时记录。
- Factual 由原 `BusinessSimulator` + `BusinessObserver` + 最近邻合法派单提供；预测未来动作一律由**模型生成的公开状态**上重新计算最近邻，不能使用真实后续动作。
- 主端点是目标设备 H5/H20 可用性 Brier，以及系统级样本状态错误、工作站服务 Brier、资源占用误差、预计完成数和等待数误差。使用场景等权平均，多个时刻/粒子不是独立样本。
- 本次只验证预测，不在这批场景反向调条件概率、不实施线上新调度评分，也不把状态 Brier 改善包装成完整在线 makespan 收益。

## 复现

依赖原 CPU 锁文件、SHA256 核验过的 frozen V4 权重、核验过的 PR44 Artifact，CI自动执行：

```bash
python -m unittest experiments.world_model_integration.recovery_integration.test_pr45 -v
python -m experiments.world_model_integration.recovery_integration.pr45_recovery_world \
  --checkpoint /path/to/v4-data.pt \
  --pr44-result /path/to/pr44-result.json \
  --out /tmp/pr45-h20-recovery.json
```

所有训练、测试、新预注册场景、权重和恢复事件的来源在协议和报告中留档。本PR保持 Draft；不得合并到正式 RHWM 默认推理路径。

**晋升门禁：** 若 H20 系统状态预测和独立校验表现可靠，才考虑在下一阶段开展真实闭环在线调度评估（含 CPU回退、延迟和副作用）。若仅设备可用性 Brier 改善而任务/资源状态无改善，则不宣称世界模型决策价值已验证。
