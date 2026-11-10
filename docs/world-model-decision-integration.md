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


## 6. 2026-11-10 GitHub Actions 已执行事实（非推测）

- 开发/机制回归成功工作流：[run 42777054864](https://github.com/lieuyimo/wm/actions/runs/42777054864)，
  共 7 个新增单元测试通过；旧 workflow 套件 18 项中 16 项通过、2 项因外部 motion backend 未设置跳过。
- 冻结的有限合成确认成功工作流：[run 42777561692](https://github.com/lieuyimo/wm/actions/runs/42777561692)，
  运行了 7 新测试、18 旧工作流测试、固定 7×5 CPU pilot 和 10×7、两种子确认。
  原始 JSON 和日志位于该运行生成的 Actions artifact；已执行场景不等于真实实验室业务。
- 两份原始 checkpoint 的 provenance 和 SHA-256 均继续以审计基线记录为准；本 PR 没有任何旧权重训练/替换。

### 开发 pilot 结果（7×5、2 任务、执行 seed1；仅单场景）

| 模型/评分 | 完成 | 流转时间 | 平均决策延迟 | P95 决策延迟 | 故障回退 |
|---|---:|---:|---:|---:|---:|
| A Rule+workflow | 2/2 | 16.5 | 44.31ms | 72.22ms | 0 |
| B PETS+workflow | 2/2 | 16.5 | 54.00ms | 84.23ms | 0 |
| C PETS+workflow_model | 2/2 | 16.5 | 59.35ms | 93.06ms | 0 |
| D Structured+workflow_model | 2/2 | 16.5 | 70.51ms | 108.76ms | 0 |

计时为上述一次 CPU runner 测量，各 CI 运行的宿主机和调度会造成延迟波动，不应视作稳定硬件基准。

在一个同状态 tick9 的两个合法候选中，真正执行后的反事实流转时间为
16.5 与 21.5 步。预测候选在三个窗口下均选第一个。5 步评分：
A=(15.0,23.0)，B=(17.0,23.0)，C=(19.2235,24.6214)，
D=(18.0,23.0)。C 的两候选终端服务模型增量分别为 +2.2235 与 +1.6214；
Structured 没有已验证的条件剩余时间模型，增量均为零。**信息已改变评分，实际模型
没有改变该次候选选择。** 人工替身输出改变选择的单元测试已通过，但不得用其宣称性能优势。

同状态 2 候选×3 粒子（总模型推进步数分别为30/60/120）：
C 的 5/10/20 步评估墙钟时间依次为约 0.0367s / 0.0639s / 0.1113s；
第 20 步两候选终端工作量已为零，C 的直接额外服务量增量也为零。
推理时间随预测窗口增加，终端评分的信息价值不保证单调增加。

### 冻结确认（10×7 新合成地图、预提交 3 任务表、种子 941/947）

| seed | A Rule | B PETS 原评分 | C PETS 新评分 | D Structured 新评分 |
|---|---|---|---|---|
| 941 | 3/3 完成，流转 22.667 | 3/3，22.667 | 3/3，22.667 | 3/3，22.667 |
| 947 | 3/3 完成，流转 23.667 | 3/3，23.667 | 3/3，23.667 | 3/3，23.667 |

所有组资源、离散运动和生命周期违规为零，模型回退为零。
按完整任务流转时间比较，C-A 和 C-B 在两个种子均为 0.00 步；
**没有观察到学习模型调度收益**。只有一个合成新布局、两个种子，
无法推断统计显著性，也不能宣称真实业务泛化。

### 九项结论的状态

1. 业务观察 v1 JSON 明确保留位置/阶段/有效服务时间/工作流/资源/历史/候选动作，
   当前只有新模型 feature API 接收完整 JSON；**冻结的 PETS/Structured
   仍沿各自原始特征合同接收子集，不声称它们直接接收新 schema**。
2. 过去未向 PETS 传入的站点资源池、机器人路径距离、公开工作流后续阶段和
   显式缺失掩码，现进入 v1 业务观察合同；旧 PETS 的 7 项服务特征未偷偷加维。
3. PETS 的条件服务时长分布已经在符合已观察会话条件时替换终端评分；
   旧评分仍只用规则先验。未来阶段按规则回退。
4. 更换人工测试预测可以改变实际策略选中动作；真实冻结模型在本次案例与
   两个确认种子未改变最终表现，且没有测得任务流转收益。
5. 5/10/20 步均按模型条件预测推进，未观测外生故障未知。墙钟开销
   依同状态测量增加。一步/三步/五步预测质量应另看原始 pilot JSON，
   **不应把有限窗口合法性混同于真实未来预测准确率**。
6. 原模型/改接口目前没有在开发或有限确认数据上优于 Rule，且多了 CPU 开销。
7. 新版业务 Deep Sets 模型**没有可信训练数据与独立验证，因此未训练、不可上线**。
8. 根本不可观察的未来设备异常、未知任务到达与隐藏服务倒计时不能由历史
   规则直接当成确知未来；PETS 服务时间分布不是故障概率。
9. 当前可区分的学习职责是 PETS 在观察条件下估计服务完成事件与条件剩余
   有效服务时间，以及 Structured 的服务完成分类。合法阶段释放、运动、容量、
   调度候选生成继续属于确定规则/CP-SAT/PIBT/LaCAM3，不能将其算作学习收益。


## 7. 开发案例预测误差实测（原始附件，非新确认）

从 run 42777561692 的 `decision-pilot.json` 中读取，取事实 nearest 轨迹窗口；
分别以同一个当前事实 observation 和当前实际 assignment 启动，后续预测步从
**自己预测的状态**用 nearest 分配；真实后续状态仅用于计算误差。

| 模型 | 1 步联合状态全对 | 3 步 | 5 步 | 已知服务会话 MAE（8 次） |
|---|---:|---:|---:|---:|
| Rule | 4/4 | 4/4 | 3/3 | 0.000 |
| PETS | 2/4 | 1/4 | 0/3 | 1.410 active tick |
| Structured | 2/4 | 1/4 | 0/3 | 0.000（规则回退，非模型分布） |

下一步服务完成事件统计：Rule TP=2、FP=0、FN=0；
PETS TP=0、FP=0、FN=2；Structured TP=0、FP=1、FN=2。
PETS precision 的分母为零，应报告**未定义**而非 0%；召回为0%。
Structured precision/recall 均为0%。此小样本不能作为广义基准能力断言。

5/10/20 条件窗口各模型均尝试3个且3个通过逐步合法性校验，
仅说明不变量与接口覆盖；并非都预测真实状态正确。

窗口标签 `external_windows=0`，因此**对突发外生故障的预测误差没有覆盖**。
采样窗过少且确定性强，Rule 完全正确；学习模型在该采样上预测明显更差，
这与两种子独立合成确认“调度没有改善”并不冲突。每种模型的指标和回退
都保存在 CI 原始 JSON 中，不能以单个模型替身测试替代真实统计。
