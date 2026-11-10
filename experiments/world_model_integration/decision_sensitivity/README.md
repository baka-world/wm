# PR #18 V4 预测到调度决策敏感性诊断

## 审计性质

- **仅开发诊断，不是新的独立确认。** 不新增训练、GPU 或样本；不调整评分、候选、调度器与运动规划。
- 源代码基线 `f9ba33515406ae077494e0cd9e5faf7c39708d3e`，PR #18 Draft。
- 独立归档 Actions run `42816481249` / artifact `13765795897`，原始 `v4-dispatch.json` 与 `v4-independent-in-data.json`。后者是**另一组 nearest_batch 事实轨迹**，不是三组固定 CP-SAT 候选的预测步明细。
- V4 权重 `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c`，训练源 run `42815634215` / artifact `13765135588`。

## 现有归档能证明什么

三个固定状态**全部在 tick 0**，候选数 2、3、4，合计 9 个。每个候选的 V4 最终评分元组与 Rule 完全相同（0/9 候选不同），三个案例排序与选择完全相同（0/3）。这是归档的真实数字，而非“神经网络没有调用”的证明。

| Case | Rule/V4 首选评分 | 第二名评分 | 首要成本差 | 排名第一个区别 | Rule/V4 选择 | 事后最优合法候选 | 两者事后损失 |
|---|---|---|---:|---|---:|---:|---:|
| 43011 tick0 | (99,0,24) | (99,1,24) | 0 | 资源争用差 1 | 0 | 1 | +5.00 |
| 43015 tick0 | (32,0,7) | (34,0,7) | 2 | 主成本差 2 | 0 | 0 | 0 |
| 43019 tick0 | (60,1,16) | (60,1,16) | 0 | 完全相同，靠候选索引破平局 | 0 | 2 | +6.75 |

43019 的首三名评分全部 `(60,1,16)`，但事实反事实流转值为 `[51,51,44.25,46.25]`；真正的选择空间存在，不能以“所有候选一样”断言 F6。43011 事实流转 `[77.5,72.5]`，但第 1 候选被次级资源争用分量压下去。43015 事实流转 `[45.25,47.25,47.25]`，当前决策本身最优。

独立归档中 V3-C 对 43015 候选 0 的主评分改变了 `-1/3` (`32 → 31.66666667`)，其余 8 个候选未变，说明保存的归档并非永远完全相等。但不能据此推断 V4 必然通过神经预测影响了排序。

## 代码级调用链

`BusinessObserver.observe` → `BusinessCPSAT.proposals` → `BusinessPolicy.evaluate` 中对每个候选每粒子 `self.model.reset_seed(1810 + tick*1009 + particle)` → `WorkflowDynamics.predict_next` → `CapacityDynamics.predict_next` → `SessionRuleDynamics.predict_next` → `eligible_sessions` → V4 Deep Sets 对可查询会话输出 sigmoid 概率 → `random()<prob` → 用阈值覆盖驱动 `ProgressAwareRule` 合法状态推进 → `release_successors` → `workflow_components` → `score_components` 的三元组 → 稳定候选索引破平局。

关键结构事实：`ServiceWorkEstimator` 只会识别 PETS 的 `_parameters` + `distribution`，不会把 V4 的事件概率直接变成终端剩余服务工时；无活动会话时它回退规则先验。因此，V4 影响最终评分必须经由**采样事件 → 状态变化 → 业务指标**的链，而不是仅调用网络就足够。这个机制是代码事实，不是已验证的三个实际状态根因。

## 逐层未决问题

原始 `v4-dispatch.json` 只保留候选总体评分、排序和事后反事实流转，**未存每步每粒子的 V4 查询概率、随机数、完成事件、预测快照或评分组件**。这些字段应标记为 `null`，不是 `0`。因此 F1（无查询）、F2（采样不变）和 F3（状态变化但评分不变）现阶段不能分类；F4/F5 在三个原始固定 5 步状态不成立，因为 V4 评分和排名均未变。F6 不能从成本并列直接推导，尤其 43011/43019 的合法候选真实后果不同。

`archive_analysis.py` 实际对 Actions 原始归档做只读检查并生成 `archive_analysis.json`；`replay_trace.py` 是尚未执行的、明确区分为**开发 CPU 确定性重放**的可审查代码。它仅在拿到精确 SHA 的 V4 原始权重和原始 JSON 后，复原相同三组状态、验证原始 5 步评分一致，再执行固定 5/10/20 窗口，记录网络真实调用、可查询事件、抽样、状态快照、业务组成项、排序以及 CPU 时间。若原始 5 步校验失败则停止，不允许挑选新状态替代。

### 只读复算

```bash
python -m experiments.world_model_integration.decision_sensitivity.archive_analysis \
  --dispatch /path/v4-dispatch.json \
  --factual /path/v4-independent-in-data.json \
  --output /tmp/v4-archive-analysis.json
```

### 最小 CPU 确定性重放（需要可运行的原仓库环境，非本次已完成结果）

```bash
python -m experiments.world_model_integration.decision_sensitivity.replay_trace \
  --dispatch /path/v4-dispatch.json \
  --v4 /path/v4-data.pt \
  --output /tmp/v4-decision-sensitivity-replay.json
```

运行前核对 SHA-256，严禁读取事实未来作为预测器输入，严禁把 replay 重新称为独立确认。5/10/20 只作为敏感性对照，不能按结果挑选最佳窗口再在已看过的原案例上声称收益。