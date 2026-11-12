# PR #47：公平增加 CPU 决策预算，验证恢复世界模型是否提升闭环调度

## 执行动机

根据 PR #46 真实完整在线实验和原始决策的零推理审计，恢复增强世界模型在每决策 0.5 CPU-process 秒内比原 V4-H20 多出现28次预算超限；两套策略第一次派单不同的11组场景，全部是原 V4 完成评分而恢复增强版评分不全并回退第一个合法 CP-SAT 提案。因此不能简单把这批 makespan 退步归因于恢复动态的评分偏差。

本研究经明确授权允许**适度提高 CPU 决策预算**，优先判断额外预算能否使条件恢复 V4 完成候选评分并实现真实业务收益。

已经检查已有 [PR #41](https://github.com/lieuyimo/wm/pull/41) 的事件触发可变预测跨度方法：中低负载节省 CPU，但高负载条件下并未稳定解决预算超限，也未提高完整任务完成时间。因此本轮**先直接在完全公平条件下检验额度从0.5秒提高到1.5秒的价值**，暂不加入新时域规则或改变评分。

## 预注册的真实闭环对照

预先冻结 `experiments/world_model_integration/recovery_budget/pr47_protocol.json`，24个**未曾出现在 PR44、45、46 的新场景种子**：原支持范围和延长故障机制各12个，B0/B1/C 各4个。所有场景继续原 `BusinessSimulator`、原业务工作流、资源容量和多机器人任务派发，故障日程仅在真实环境构建时决定，模型不可见其未来时刻表。

对每一个新场景严格运行**五臂**：

| 策略 | 原模型 | 预测方法 | 每决策 CPU-process 时间限额 |
|---|---|---|---:|
| CP-SAT | 不调用世界模型 | 第一个合法候选 | 不适用 |
| V4-0.5 | 冻结 V4 | 原 H20/P3 完成样本数评分 | 0.5秒 |
| 恢复版-0.5 | 冻结 V4 + PR44 公共恢复动态 | 同一个 H20/P3 评分 | 0.5秒 |
| V4-1.5 | **相同冻结 V4** | **原 H20/P3 相同评分** | **1.5秒** |
| 恢复版-1.5 | **相同冻结 V4 + PR44 恢复模型** | **相同 H20/P3 评分** | **1.5秒** |

这里没有增加任何新训练。候选分配仍通过原 `BusinessCPSAT`，预测状态上续调度仍是原 CP-SAT，原模型未重建网络、未改随机种子、未做任何新效用权重拟合；**每个CPU额度下两种V4都获得完全相同预算**，候选列表必须全部评分完成才有权改派，否则走原安全回退。

实验同一场景内 V4 四臂执行顺序按种子模4循环，以减少工作流初始化与缓存热身始终偏向单一模型的风险。每次真实完整仿真从同一原始场景/真实随机种子启动，并保存全部逐tick在线决策日志。

## 指标、解释和预算纪律

**主指标**：两个机制内、每场景配对的真实完整业务完成时间（makespan ticks，越少越好）。同时统计逾期任务、完成数、容量等待、动作改派、完整模型评分、CPU 超限和回退、神经网络推理调用、真实安全违规、CPU-process 实际开销。

严谨比较的顺序为：
1. 同预算 **V4-1.5 vs 恢复版-1.5**：新增世界模型动态有没有真实净收益；
2. **V4-1.5 vs V4-0.5** 与 **恢复版-1.5 vs 恢复版-0.5**：增加预算分别改善了多少；
3. 同预算 0.5 两臂与 CP-SAT 作为业务连续性基线。

不得把0.5秒模型与1.5秒模型的差值单独包装成“世界模型算法改进”。额外CPU是实在成本，应记录每个episode总 CPU 和峰值耗时；CPU-process time 不是可中断的硬 wall-clock SLA。

研究性统计均以**场景种子**为基本单位，预测候选和粒子不是独立试验；全程不会调新的恢复概率、CPU限额、模型权重或评分权重。不使用之前已看过的 PR46 在线组做新预算下的“独立”结果。

**任何收益都需要同时观察**：①没有安全退步；②新增预算明显降低关键评分超限；③相对**同预算**原 V4 有真实完成时间和逾期净改善；④新增计算成本在业务延迟约束范围内。否则保留原 V4 为默认研究基线。

## 复现

完整程序 `experiments/world_model_integration/recovery_budget/pr47_online.py` 重用 PR46 已验证的原在线实现，并仅替换运行时完整候选评分的 CPU-process 限额。CI 复用 PR31 原 CPU依赖，恢复原 V4 checkpoint 与 PR44 原始 KM Artifact 并验证源 SHA。

```sh
python -m unittest experiments.world_model_integration.recovery_integration.test_pr45 experiments.world_model_integration.recovery_shift.test_pr46 experiments.world_model_integration.recovery_budget.test_pr47 -v
python -m experiments.world_model_integration.recovery_budget.pr47_online --checkpoint /path/to/v4-data.pt --pr44-result /path/to/pr44-result.json --out /tmp/pr47-online.json
```

此 PR 的实验最多120条真实完整在线轨迹，预注册 CPU进程时间合计上限1350秒。全部输出逐案例 JSON、参数指纹、原始环境真实事件 SHA256、源版本、依赖哈希和工作流 logs。保持 Draft，不修改或合并先前任何 PR。
