# PR #18：世界模型信息供给与调度决策闭环（Draft）

> 基线 `cb01d0993bcaebc2e3f0a1f9941ccd93add897e5`，
> base `audit/model-inventory-pr17-20261110`，
> head `feature/world-model-decision-integration-20261110`。
> **这是一份可核验的实施记录；未执行的验收不得填成通过。**

## 0. 实现与实验边界

- **已写入可运行代码**：`business_transport/state_features.py`、`service_work.py`、
  `workflow_scoring.py`、`policy.py`、`lab_transport/business_state_learning.py`，
  新测试、pilot 和 CI 工作流。
- **PR15/16 旧对照**：`scoring='legacy'`、`scoring='workflow'`、
  `WorkflowPolicy(..., model_aware=False)` 继续调用原有模型与 _prior 评分逻辑。
  不更改 `sequence.pt`、`pets-service-cpu.pt`、PR8 历史权重记录。
- **新版显式启用**：`WorkflowPolicy(..., model_aware=True, horizon=5|10|20)`。
  `max_prediction_seconds` 可设置一个严格的单次评估墙钟时间限制，超时返回合法首候选，
  在 `fallback` 中报告 `TimeoutError`。不将故障 Oracle 注入预测。
- **训练状态**：新版业务状态模型接口和 CPU 训练器已实现，但 PR9 的原始结构化轨迹
  不等同于带完整业务字段和已隔离标签的 v1 业务状态数据集。未取得经过核验的
  v1 事实样本及独立 rollout 验收前，**不得声称新版模型已训练或上线**。
- **确认状态**：本项目的 PR15/16 数据属于开发/机制验证，不能再次宣称独立确认。
  本 PR 先做固定合成 CPU pilot；未形成新的独立地图/任务簇确认集。

## 1. 输入合同、来源和未知字段

新版 `SCHEMA_VERSION='business-observable-v1'`，
`LABEL_VERSION='business-service-event-v1'`，由
`build_business_features(observation, assignments, geometry, rules)` 输出。它**不是**
PETS v1 的 7 维输入，不是旧 Structured 的 26 维 token，不允许用于旧权重直接加载。

| 输入组 | 具体字段 | 可见来源 / 处理 |
|---|---|---|
| robot | 位置、phase、current_task、service_elapsed | 当前 RobotV2 |
| robot | service_active_elapsed、service_paused_elapsed | ObservableHistory，未知值带掩码 |
| robot | operational_available | 当前可观测属性，缺失带掩码 |
| robot→candidate | 至候选 pickup/dropoff 站点路径距离 | 当前 geometry 和公开站点位置 |
| samples/jobs | job_id、stage_index、workflow、status、assignment、release_tick、deadline、priority | 已释放业务样本；deadline/priority 可未知 |
| workflow | 已完成阶段、已知未完成后续阶段、公开阶段顺序 | 当前样本及 PublicBusinessRules.workflows；不伪造未来任务释放 |
| stations/resources | device_available、available、capacity、occupied、blocked、会话和需求 | BusinessStationV2 / StationV2 / PublicBusinessRules |
| history | 历史有效服务均值、完成计数、完整性 | ObservableHistory；缺失按 known=false 显示 |
| actions | 候选分配、站点、资源 ID、距离、现有任务关系 | 当前合法候选 |
| unknowable | 未来设备故障、真实服务剩余倒计时、未来外部任务、反事实结果 | 全部固定 `known=false,value=null` |

返回字段分为 `observed_facts`、`public_rules`、`candidate_action`、
`unknown` 和 `learnable_next_events`，同时附字段来源列表
`FIELD_SOURCES`。这些输入只由观察构建，不接受 simulator 或 oracle 对象。

注意：v1 神经 token 编码目前只是完整观测合同的轻量数值投影，
不是已经验证具有全部业务预测能力的模型。类别 ID、工作流图等存在于 JSON 合同；
当前 encoder 的部分数值投影仍待充分覆盖性测试。不能声称学习器已经利用了
所有合同字段。

## 2. 服务预测与工作流评分

`ServiceWorkEstimator` 穿透
`CapacityDynamics → RecoveryMovementModel → PETSRuleDynamics` 的只读包装：
- 检测真实已观察到的服务会话：机器人、任务、阶段、站点、位置、非空服务 elapsed 全一致，
  且有效服务进度已知；否则显式回退先验。
- 保持 `service_features()` 原 7 维编码，调用 **既有**
  `_parameters(..., transition=False)` 以及
  `ServiceDistribution.expected_remaining_steps(parameters, elapsed)`。
  每个集成成员的同一条件期望仅作离散度描述，不是校准区间。
- Structured 未有可用剩余时长分布，故只执行它原来的下一步服务完成事件预测；
  终端服务工作量回退 Rule 先验，注明
  `no_verified_remaining_duration_distribution`。
- 尚未开始的阶段、未释放但公开确定的后继阶段，全部使用公开规则先验；
  不假设未来服务已有进度，不将未来故障当成确定事件。

`workflow_components(..., estimator=None)` 原入口仍可调用。
新版传入 estimator 后，当前已在服务阶段的模型期望**替换**相应 Rule 剩余量。
其余所有阶段仍按原工作量计算，不增加第二份成本：
```
total = unfinished_job_ticks
      + remaining_transport_distance
      + sum(conditional_model_or_rule_service_steps)
```

资源竞争仍为字典序第二项，交通距离第三项。
每个候选/粒子记录 `rule_prior_service`、`model_service`、
`model_minus_prior`、`unfinished_job_ticks`、`remaining_distance`、
`resource_contention_proxy`、`score` 以及 service estimate source / fallback。

## 3. 预测与失败语义

- 每个候选从同一个只读事实观察开始；每个候选使用相同粒子种子
  `seed + tick*1009 + particle`，轨迹状态按模型预测递推。
- 当前时刻使用候选计划，后续短窗使用基于**预测状态**的 nearest action；
  后继阶段只能由预测完成触发并在下一观察边界释放；
  `check_business_transition` 逐步验证合法性。
- 已观测设备状态由 capacity_adapter 使用，不访问故障日程。
  未知后续设备故障概率继续标记为未知。
- 决策粒子记录均值、总体标准差、经验 P90；3 粒子不是 95% CI。
  在线缺省仍最小化**均值**业务成本（`risk_weight=0`）；
  可显式传入 `risk_weight`，公式为
  `mean + risk_weight * (empirical_p90 - mean)`。
  任何风险权重不得根据确认集调参。
- `max_prediction_seconds` 为可选时间预算；推理或合法性失败时按现有
  `BusinessPolicy` 回退首合法候选并记录类型/原因。
  无额外合法分配在下一步仍允许等待；日志记 `no_new_legal_assignment`。
- 首版仍预测满 5/10/20 步，不声称具备已验证的提前终止优化。

## 4. 复现入口（CPU-only）

```bash
python -m unittest discover -s experiments/world_model_integration/tests -v
python -m unittest discover -s experiments/workflow_prediction/tests -v
python -m experiments.world_model_integration.pilot --output /tmp/pr18-cpu-pilot.json
```

CI：`.github/workflows/world-model-integration.yml`，
`python3.10`、`requirements-paper-reuse.lock`，不启用 CUDA。
pilot 是固定 7×5 合成业务案例，`seed=1810`、实际仿真 `seed=1`、
`particles=3`、`budget=4`、`horizon=5`，对照：
A Rule + 旧 workflow；B PETS + 旧 workflow；
C PETS + model-aware workflow；D Structured + model-aware workflow。
并对同一事实状态使用 5/10/20 预测各模型/评分。所有结果存储在
GitHub Actions 附件中的 `decision-pilot.json`。

旧流程/权重实测与新状态模型的独立确认不是一回事。pilot 仅验证系统集成、
保留模型日志、做开发阶段现象对比；任何改进均须在新任务表和地图/种子簇、
冻结代码与阈值后独立确认。

## 5. 分阶段验收与剩余工作

| 项目 | 状态与限制 |
|---|---|
| 版本化业务观察 API | **代码已提交**；相关测试需看最终 CI |
| PETS 条件期望进入终端工作量 | **代码已提交**；受限于已观测有效服务进度的会话 |
| Structured logit 仍仅作为服务完成事件 | **代码已提交/保持原功能**；没有剩余时长分布 |
| 5/10/20 预测与成本/耗时日志 | **代码已提交**；CPU 实测以附件为准 |
| 旧业务评分与策略接口 | 保留显式 legacy 分支 |
| 新状态模型构造/训练器 | **代码已提交、训练未执行、线上不可用** |
| 决策排序因模型输出变化 | 仅可由人工替身因果测试证明接口，不可充当预测优越性 |
| Rule/PETS/Structured 真实业务闭环 | 固定 CPU pilot；不能代替独立确认 |
| 故障前瞻 | 无有效可观察预测信号；未来故障风险保持 unknown |
| CP-SAT/PIBT/LaCAM3 | 无算法/权重修改 |

不应将“PETS 在某个终端服务会话能输出条件期望”，升级成
“世界模型优于 Rule”、“完整状态信息已全部学习”或“真实业务已可靠改善”。
这三项必须分别提供数据。
