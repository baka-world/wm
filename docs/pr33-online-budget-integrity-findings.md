# PR #33：H20 在线调度的 CPU 预算执行一致性审计

日期：2026-11-12。只读审计；源 GitHub Actions 结果已经在 PR #32 归档，不重新推理、训练或仿真。

## 审计依据

- [PR32原始在线实验 Actions #43144645020](https://github.com/lieuyimo/wm/actions/runs/43144645020) 与 [源 artifact #13911157726](https://github.com/lieuyimo/wm/actions/runs/43144645020/artifacts/13911157726)；
- [本次只读审计 Actions #43146849683](https://github.com/lieuyimo/wm/actions/runs/43146849683) **SUCCESS**，源文件 `independent-online.json` 的 SHA256 从原 artifact 校验成功；
- [审计 JSON、全场景逐次决策记录及其 hash artifact #13912356018](https://github.com/lieuyimo/wm/actions/runs/43146849683/artifacts/13912356018)；
- 预先固定单次决策预算：**0.5 CPU process seconds**。不应误称其是企业现场 wall-clock SLA。

## 发现真实预算代码缺陷

现有 `experiments/world_model_integration/online_horizon/pr30_run.py::OnlineHorizonPolicy.__call__` **仅在每次模型候选推理前**检查 `time.process_time() - start >= budget`。最后一个候选预测结束之后没有再次检查预算，就依据 `len(values)==len(plans)` 改选。

如果最后一次预测推过0.5秒，代码可能从已经超限的多个预测中选择第二方案，漏报本该回退的预算超限。且由于单次 `rollout` 不可中断，CPU时间实际上会超过0.5秒。这是**实验实现上的真实问题**，不属于模型预测误差，也不能仅通过文档解释消除。

## 完整只读量化

| 场景（各6组），策略 | 原脚本记录超限 | 实际每次决策CPU>0.5秒 | 超限后仍执行非首选提案 |
|---|---:|---:|---:|
| heavy / Rule-H20 | 6 | 13 | 0 |
| heavy / 原 V4-H20 | 6 | **17** | **4** |
| heavy / Warmstart-H20 | 5 | **14** | **1** |
| heavy / Scratch-H20 | 6 | **16** | **3** |
| medium / Scratch-H20 | 0 | **2** | **2** |
| shift / 原 V4-H20 | 0 | **10** | **3** |
| shift / Warmstart-H20 | 0 | **9** | 0 |
| shift / Scratch-H20 | 0 | **9** | 0 |
| light / 所有策略 | 0 | 0 | 0 |
| medium / CP-SAT、Rule、原V4、Warmstart | 0 | 0 | 0 |

严格意义上，“超预算后仍执行非首选方案”的实例至少为原V4 7次、Scratch 5次、Warmstart 1次；一条非首选不代表必然造成结果改善或退步，**不能直接把原来的 makespan 差值据此计算为某种已修正收益**。必须在修改后的在线执行程序里实际重跑独立测试。

## 修复验收要求（下一独立 PR）

1. **在每次候选预测完成后再次检查 CPU 预算**：如果预测返回时总CPU开销超限，禁止采用预测重排结果，安全回退CP-SAT第一合法提案，并明确计入 `budget_overrun`、`fallback`；包括**最后一个候选**完成后超限的情形。
2. 单测应使用受控假时钟复现最后一候选从0.18s跳到0.66s、原脚本错误执行非首选方案的问题，以及小于预算时正确重排的正常用例；不能靠CI随机负载触发测试。
3. 对后续真实执行，记录超时发生在哪个候选、实际CPU和回退动作；保留不可中断的预测耗时，绝不能将“回退正确”写成“物理上绝不超过0.5秒wall time”。
4. 对上游 PR30/PR32 的旧结果只保留研究归档，不改写事实；修改策略必须开新独立 PR、冻结新 seed，重新比较。用户要求的完整环境和可恢复模型断点仍沿用 PR31已归档的源码、模型SHA和包环境。
5. 修复后若决策收益消失，停止宣称H20在严格CPU预算下稳定有效；优先研究既有MPC中有限预算/自适应候选或预测跨度策略，不继续凭空加新网络层。

**当前判断：PR31后训练确实提升了独立服务事件Brier；PR32确实跑出了真实在线结果；但“0.5s严格预算下的H20收益”受执行缺陷影响，不得晋升默认策略。修复与重新评测是高于继续训练的第一优先级。**
