# PR #46：世界模型条件恢复跨机制检验与独立闭环调度

## 研究定位与前一轮证据

本科毕业设计《基于世界模型的样本转运调度系统设计与实现》。PR #44 从在线公开的 `device_available` 序列得到**故障已发生后的条件恢复** KM 生存分布；PR #45 将其作为显式外生设备恢复事件接入冻结 V4-H20，保持原服务完成网络不变，并在同一合成机制的新种子上得到 H20 全系统状态误差改善。

PR #45 训练/测试都建立在设备故障持续时间最长不超过20 tick 的合成支持集上，长预测跨度设备可用性 Brier=0 很可能反映这个上限设定，而不是对未知维修机制的普遍准确性。因此 PR #46 将 **“跨外部停机持续时间机制”** 与 **“全程在线真实业务效果”** 分开验证，避免从前轮 Brier 自动推断在线收益。

## 冻结方案与源方法

预注册入口：`experiments/world_model_integration/recovery_shift/pr46_protocol.json`，在所有真实实验之前创建。

- 保留 PR #45 的 `ObservedRecoveryWorld`、`PublicDownClock`、`check_recovery_boundary`、独立粒子随机数和原 V4 服务转移。**不重新训练 V4，不重新拟合或修改 PR #44 的 KM 曲线**；仅复用校验哈希的原归档 TRAIN-only 站点 KM。
- 场景类别一：`within_support`，与 PR #44 相同的人工故障持续时间支持集，仅新种子。
- 场景类别二：`extended_outage`，同一 B0/B1/C 站点但将真实故障持续时间支持集从 B0:[4,6,8,10] / B1:[8,12,16,20] / C:[3,5,7,9] 扩展为 B0:[9,12,16,22] / B1:[18,24,30,36] / C:[8,12,16,24]。**这是故障持续时间机制的可控分布偏移，并非跨地图、跨企业、真实维护数据的外部泛化。**
- 两个任务使用**完全不重叠的测试种子**：预测测试24场景（每机制12，每站点4）和闭环在线24场景（每机制12，每站点4），各自不与 PR44/45 既有训练和测试种子重合。
- 故障日程仅在原 `BusinessSimulator` 场景构建阶段生成。原真实业务仿真的 `BusinessObserver` 公共可用性状态及其连续公开历史，是恢复模型唯一动态输入。预测不读取真实故障时长、结束时间、真实服务剩余时间、未发布任务或后续真值。
- 真实任务环境保留原工作流、资源容量和多机器人调度规则，每场景10个已知业务任务、最大170 tick、公开deadline=100。**这只是冻结仿真条件，未调整真实业务内核。**

### A：冻结独立 H5/H20 系统状态评测（24 场景）

两臂为同一冻结 V4-H20（设备默认保持）以及 V4+公开已知停机条件恢复；在真实公开状态 tick8/10 选取合法当前动作，在预测状态上运行原最近邻续调度，H5/H20，3粒子。主看目标设备可用性 Brier、样本/机器人任务状态、工作站服务、资源占用及任务完成/等待误差。

每种机制、每个预测跨度单独报告场景等权误差。记录恢复采样、状态合法检查及初始已观测停机年龄。所有真实后续状态仅用于离线标签。

### B：冻结完整在线闭环调度（另一批24场景）

三臂：
1. **原 CP-SAT**，仅执行第一个合法候选；
2. **原 V4-H20**，原 CP-SAT 候选、3粒子预测、完成样本数最大化；
3. **V4-H20 + 公开条件恢复**：保持第2臂完全相同的候选生成、预测状态上的 CP-SAT 续调度、H20完成样本数读出、并列时选首候选，以及0.5秒**CPU-process time** 预算和超时回退；唯一改变是显式预测已经公开的外生恢复。

策略通过同一个原 `BusinessSimulator` 真实闭环运行到结束。对照真实 makespan、逾期、容量等待、受阻移动、任务完成状态、动作变化、预测调用、CPU预算超限及安全违规。回退只执行合法原 CP-SAT 首方案，不在部分模型预测结果上排名。**CPU process-time 超时门禁不等价于有硬中断能力的 wall-clock 实时 SLA。**

### 解释门禁

- 跨机制恢复 Brier 退化，说明训练支持集不足，不能用真实结果反向修改 KM 支持、改权重或事后划子集提升论点。
- 系统状态更准确但闭环 makespan 不变，表示未来状态信息尚未稳定改变选择或业务结果；不能声称决策收益。
- 条件恢复后的 H20 误差改善仅表明冻结世界模型在相应已观测故障范围内传播了信息，不证明预测未来突发故障。
- 全部真实在线比较是新种子的同一合成地图和人为机制条件。12/机制小样本仅作工程证据，不做统计显著性和现实部署性能推断。

## 方法来源与复现

遵循 PR #44 Kaplan–Meier 右删失生存估计、PR #45 显式合法外生恢复状态转移、PR #30 原 CP-SAT-H20 决策和 PETS 风格粒子传播，没有从零训练新模型。

在仓库根目录、恢复原 CPU lock 环境、原 SHA256 检查点及原 PR44 train-only KM 后：

```bash
python -m unittest experiments.world_model_integration.recovery_integration.test_pr45 experiments.world_model_integration.recovery_shift.test_pr46 -v
python -m experiments.world_model_integration.recovery_shift.pr46_forecast --checkpoint /path/to/v4-data.pt --pr44-result /path/to/pr44-result.json --out /tmp/pr46-forecast.json
python -m experiments.world_model_integration.recovery_shift.pr46_online --checkpoint /path/to/v4-data.pt --pr44-result /path/to/pr44-result.json --out /tmp/pr46-online.json
```

CI `.github/workflows/pr46-recovery-shift-online-cpu.yml` 严格先执行 A 再执行 B，任何 A 失败或安全边界破坏会阻止 B；两批种子已在同一预注册协议中锁定，后续不得拿 A 结果修改 B 的方法。全部证据保留事件序列 SHA、逐决策数据、测试日志、依赖和源版本。

PR 继续 Draft，原主干、PR19、原 V4 默认预测器和历史训练检查点保持不变。
