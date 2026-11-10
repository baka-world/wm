# PR #18 V3 训练修复与独立验证：最终实测结论（2026-11-10）

> **必须优先阅读：** 本节是前述 v2 实现报告的后续，不改写原始 v2 训练结果。
> 最新 V3 使用新版本输入/独立检查点，仍只允许显式实验启用，正式 online gate 未通过。
> 原始已查看的170条 Test **没有**参与 V3 选择或本次独立测试。历史 PR9/11 权重、PR15/16 结果和 PR19 演示未修改。

## 八个问题的直接回答

1. **旧模型效果差的主要原因找到没有？** 训练严重不足**已通过固定消融证明是重要原因**：
   v2 仅10次 optimizer.step()，同旧特征充分训练的 V3-A 在 Validation Brier
   从0.22856降到0.07837。但是全新场景测试仍欠佳，进一步证明训练不足并非唯一问题，
   原训练场景持续时间/扰动覆盖过窄与外推校准失败同样成立。
2. **10次更新是不是主要瓶颈？** 对原开发分布是。V3-A 做825次优化更新
   （最佳检查点第715次），验证大幅改善。V3-B 保留10次更新，即使新增特征
   Brier也只有0.22826。
3. **增加目标服务特征有独立改善吗？** 10次更新下 V3-B-v2 改善约0.00030，
   无实质证据；对齐825次更新的 epoch65，
   V3-C=0.07541、V3-A=0.07837（差约0.00296），
   对增强输入有**有限开发集支持**。两种网络初始化参数空间不同，
   单个训练seed不支持断言这是独立稳定收益。
4. **新版是否超越固定概率与经验概率？**
   开发 Validation 超过B0固定概率（0.07248 vs0.19983），
   但不及B1=0.06777/B2=0.06790；
   **全新独立测试不如B0（0.17331 vs0.15498）**，
   虽好于随训练持续时间外推失败的B1/B2。
5. **新版是否超过Rule？**
   Validation V3-C=0.07248，Rule=0.13218；
   **独立测试 V3-C=0.17331，Rule=0.10932，未超过**。
6. **是否改变真实调度选择或流转？**
   独立测试的事前门禁要求 V3 同时超过B0和B2；由于未超过B0，
   仅执行1步合法性 smoke 并停止三组闭环。没有新的真实任务分配或流转改善证据。
   原 PR18 六组开发实验已观察到模型成本不同、候选排序/动作仍相同。
7. **当前可观测信息能学什么？**
   有效服务已进行时间、公开站点服务先验、历史统计、阶段、设备当前可用性与容量争用
   都是合法条件输入；本轮尤其验证有效服务进度的开发集信息含量。
   没有公开预警的**未来外生故障**、随机服务时长隐藏抽样、
   尚未释放任务和实际反事实结果，不能由当前观测直接“已知”；
   需新可观测信号才可能改变可预测性边界。
8. **下一阶段？** **停止在当前数据上盲目延长训练。**
   归类为“开发集训练不足已修复，但独立测试仍无法超过简单概率与Rule”
   的跨机制分布外失效。优先改善**合法事实数据的服务时长/设备状态/地图任务变化覆盖**
   和按完整轨迹的评估设计；在此之前保留 Rule 作为推荐事实基线，
   不将 V3 声称为正式线上模型，不调整本次确认集或追加有利种子。

## 前置审计与版本隔离

- 冻结起点：`c16dcd143e706d4bc5e287ece73ad3b472a6374a`。
  冻结 v2 服务查询是48维；V3 增强目标输入独立版本
  `business-service-query-input-v3`，56维，包括公开目标站点服务期望、
  服务有效进度/公共期望的比值、当前墙钟进度比值和历史均值比值及缺失掩码。
  不读实际最终服务时长、未来故障计划、隐藏倒计时。B1/B2 也不读这些。
- v2 下一步标签约定：在 t 存在服务会话且有效进度已知，
  仅依据事实 `t+1` 相应 sample.status 的 pickup=>in_transit 或
  dropoff=>completed 为正；否则仍保持会话才为合法负例。
  设备暂停、未开始、未知进度、中断显式删失。
  当前实现**未覆盖首个服务步直接完成的事件**，这是明确的分母边界。
- 标签版本 `business-service-query-v2` 保留，因为事件定义未变化；
  模型权重版本升级到 `business-sets-session-v3`，
  并在检查点声明 `input_schema` 为 v2(原48维)或v3(增强56维)。
  加载器 `load_v3` 严格检查输入、观察和标签版本，
  `online_verified=False` 不允许默认正式加载。
- 旧 `business_state_learning.train_cpu()` 在 v2 schema 时明确转接
  `train_session_cpu()`；V3 训练入口是独立
  `lab_transport.business_session_v3.fit_variant()`，
  不会误调原始全局事件模型。
- `SessionRuleDynamics` 新的 `feature_version='v3'` **显式**加载新权重；
  合法会话分别输出 Bernoulli 事件，通过规则推进业务阶段，
  只在已观测活动服务会话采样；设备已暂停不能采完成事件，
  同一候选间重置粒子随机种子，临时服务阈值不会泄漏事实状态。
  CPU 自动回归含两个不同服务会话与时间边界的独立标签检查。

## 冻结优化消融（先验证，后一次性独立确认）

协议：[`v3_training_protocol.json`](../experiments/world_model_integration/v3_training_protocol.json)，
SHA-256 `efbda3f45cf30a107573fd42d28eb23c3fab69ce5420404902a47829fac62a26`。
相同 Train 338（96正）/Validation174（48正）按完整任务轨迹分组；
Adam LR0.001、seed1810、CPU1线程、每变体180秒上限、总预算600秒。
V3-A/C 小批量32、最多100epoch、Brier早停耐心10；
V3-B 全批次10epoch、严格10次更新；
固定阈值0.5，不搜索规模/网络结构；各变体从随机初始化。

| 方法 | 更新总次数 | 最佳检查点累计更新 | Validation Brier↓ |
|---|---:|---:|---:|
| 冻结V2 | 10 | 10 | 0.22856 |
| V3-A 旧特征+小批量 | 825 | 715 | 0.07837 |
| V3-B 增强特征+旧训练预算 | 10 | 10 | 0.22826 |
| V3-C 增强特征+小批量 | 1100 | 990 | **0.07248** |
| B0固定概率 | - | - | 0.19983 |
| B1进度经验风险率 | - | - | **0.06777** |
| B2进度+先验经验风险率 | - | - | 0.06790 |
| 冻结Rule | - | - | 0.13218 |

训练实际完成：[Actions #42798391472](https://github.com/lieuyimo/wm/actions/runs/42798391472)（成功）。
训练消融代码：`lab_transport/business_session_v3.py`，
经验风险率：`experiments/world_model_integration/empirical_baselines.py`。
B1是 Train 上“合法服务已生存 e 有效步”条件下一步完成率，
固定有效进度分桶、以B0为Beta平滑先验（强度4）；
B2再纳入公开服务期望/归一化进度分桶，按B1平滑（强度6）。
不从 Validation/Test 估计平滑统计。
选择前固定了按 Validation Brier 选 V3 候选，最终**选择 V3-C**；
旧170条已看 Test 无一条用于本轮选参。

检查点 SHA-256：
- `v3-a.pt`：`78de0b5f4f5b9c4d98d26fe22bdb985d38097349bc8f8ca223edf607e10de76d`
- `v3-b.pt`：`386e99dc2308553e1f764bf0a229afe4c93ec99629295155c1b2f17b402ca5d5`
- `v3-c.pt`（候选）：`65aeca1bb7a09d36af8793379a7766ea28b43fdbc6e0ef370777a2f024602696`
- 历史 v2 原始冻结权重校验：`84f426a7acdb860b283cc7f19efdc930ab6073d191bd17366949d7b7c52bb68d`，只读下载并核验，未重新训练成“v2”。

可复现训练源 JSON SHA-256：
`1b0268dd86bd4c360c29f05fbc97d9fb8263ecf6ded75e62b1ff4e4e50e34b02`。
日志和三个独立 .pt 权重由
[训练 CI artifact](https://github.com/lieuyimo/wm/actions/runs/42798391472/artifacts/13753819654)
单独存档（保留90天），没有覆盖任何旧权重。

## 事前冻结的全新独立测试（一次）

配置：[`v3_independent_config.json`](../experiments/world_model_integration/v3_independent_config.json)，
SHA-256 `4b36fdc497dc54a1a3ba158d2d675e0adf797a4951db49c1c0c237e6dca4abae`。
单次 push trigger：
`experiments/world_model_integration/v3_ONE_SHOT_CONFIRMATION_TRIGGER.txt`，
不允许再次修改触发文件以重跑确认。
[Actions #42799403483](https://github.com/lieuyimo/wm/actions/runs/42799403483) **全部通过**。

6个不同完整轨迹组，seed 33101/33103/33107/33109/33113/33117，
新地图布局3/4，4项业务任务/组，公开服务预计5/6/7步，
jitter1/2，有/无突发外生故障和不同资源容量。
6组合计事实643次服务查询：**558条有效已开始服务事件，
96正/462负；85条尚未开始已排除**。
分组间没有共享 seed/任务轨迹，也不复用曾看过的 PR18 冻结测试种子。
注意：无故障3组，有故障计划3组；本次样本排除记录没有实际服务暂停事件，
故设备暂停条件下的神经完成率**没有独立评价分母**，
不得从“有故障配置”推断真实暂停预测性能。
容量配置包含1/2，但本轮未另存实际发生的资源阻塞样本分母；
工作站/提取阶段已记录分组 Brier，
**尚未独立按 workflow 的 stage_index 细分**，属于未完成验收项。

| 独立模型 | Brier↓ | log loss↓ | 说明 |
|---|---:|---:|---|
| Rule | **0.10932** | 1.7620 | 确定0/1预测，犯错的log loss很高 |
| B0 | 0.15498 | **0.4932** | 固定Train完成率28.40%，独立实际17.20% |
| B1 | 0.26113 | 0.7934 | 训练内进度规律无法外推至更长服务 |
| B2 | 0.26552 | 0.8812 | 训练服务时长范围外，条件平滑不足 |
| 原冻结V2 | 0.21653 | 0.6259 | 独立权重不变 |
| **验证选定V3-C** | **0.17331** | 0.5949 | 优于v2但未过B0/Rule |
| PETS | 0.47984 | 2.2365 | 原冻结权重 |
| Structured | 0.49195 | 1.7287 | 原冻结权重 |

独立 V3-C：
ECE5=**0.21246**，平均预测完成概率**38.45%**，
真实完成比例**17.20%**；
精确率44.08%、召回96.875%，TP93/FP118/TN344/FN3。
过度自信且校准失衡，不能以高召回称为可靠。
完整轨迹组比较 V3-C vs Rule Brier：
仅 seed33107 为改善（差 -0.06566），
其余5组分别差 +0.00088、+0.04272、+0.07108、+0.14848、+0.12220。
任务阶段的 pickup/dropoff phase Brier 分别约0.17646/0.17028；
与真实业务工作流 stage_index 不同，不能混为一谈。

独立确认结果只运行了一次，不按测试表现修改阈值或权重。
[机器可读 artifact](https://github.com/lieuyimo/wm/actions/runs/42799403483/artifacts/13754758844)
包含全部轨迹/站点/服务进度/phase 分组统计、
冻结模型对照和 `v3-decision.json`。

## 合法状态推演与调度停止门禁

已预注册要求独立 V3-C Brier 同时低于B0、B2，才进入额外三组真实闭环。
独立结果虽然好于B2，但**不及B0**，因此触发停止：
`v3_decision.json` 中
`gate_pass=false`,
`stop_reason=independent_brier_not_better_than_B0_and_B2`，
仅使用实际训练检查点通过**1步业务预测合法性**；
本轮没有更新后的完整任务分配/决策损失/业务流转数据。
这不是负面业务效果实测，也不是证明新版不能改变动作：
只是**按协议拒绝进行耗费较大的无依据确认**。

## 证据边界与资源

- 新代码：`business_transport/session_features_v3.py`、
  `lab_transport/business_session_v3.py`、
  `experiments/world_model_integration/empirical_baselines.py`、
  `v3_training.py`、`v3_independent.py`、`v3_decision.py`、
  两个冻结配置和独立单次 CI。
- 原 PR18 Rule/Structured/PETS/PIBT/LaCAM3、CP-SAT及旧权重接口保留。
  测试合同与工作流回归在 V3 训练 CI 成功。
- CPU预热3步约0.0034秒，V3 A训练约0.709秒、
  B约0.056秒、C约0.880秒（CI所报模型训练主循环壁钟，不含依赖安装/轨迹生成），
  远小于各180秒冻结上限。**0 GPU**。
- 保存原始数据生成配置、训练选择、每epoch实际更新、损失、概率分布、
  独立轨迹级指标和检查点SHA。
- Actions artifacts 会过期，Git commit 保留可复现生成器和配置；
  若需要长期复现，必须将二进制权重与原始结果归档至长期制品库。
- 本轮最终分类：**开发分布训练瓶颈修复成立，独立泛化不达标，
  不能把模型训练成功等同于世界模型已带来调度价值。**

---

---

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


---

# PR #18 续实施：逐服务事件模型与合法事实训练（2026-11-10）

## 8. 本轮新增可执行组件与责任边界

- `business_transport/session_events.py`：目标查询
  `(robot_id,sample_id,phase,station_id)` 与显式缺失/暂停/尚未开始检查；
  对当前已经开始且有效进度已知的服务才输出合法预测查询。
- `lab_transport/business_session_learning.py`：
  `BusinessSessionEventModel` 保留 Deep Sets 40 维实体 token 全局均值上下文，
  添加独立 48 维目标会话输入和服务完成概率头。
  相同状态不同目标会话不是共享一个 logit；测试交换目标和标签，确认 ID 关联。
  旧 `business_state_learning.py` / PR9 Structured / PR11 PETS 编码及权重**不变**。
- `experiments/world_model_integration/build_dataset.py`：
  仅用实际 BusinessSimulator 的 `(observable(t),executed_actions,observable(t+1))`；
  下一真实状态仅生成完成/继续的监督标签。
  服务尚未开始、暂停、缺失进度、异常中断等被剔除并计数；
  **没有用服务未开始当负样本**。
- `business_transport/session_dynamics.py`：
  对合法逐会话查询执行新模型概率前向，以独立粒子随机数做 Bernoulli 采样，
  已有 ProgressAwareRule 决定运动/服务状态合成，
  已有 WorkflowDynamics 决定合法后继释放，
  外层容量与运动检查器仍必须通过。
  **不学习或伪造确定资源规则**。
- `business_transport/service_work.py`：
  增加可选 predicted-start PETS 受限模式，检查当前**预测状态**在目标站点的
  开始条件与冻结 PR11 训练特征分量范围。超出范围显式回退规则先验。
  该范围检查只是覆盖启发式，不代表概率已校准或真正多变量 OOD 检验。
  未来未释放阶段没有已知机器人和服务进度，不能当成已开始会话估计。

## 9. 固定数据合同与第一次 CPU 训练

训练前预提交数据配置：
[`dataset_config.json`](../experiments/world_model_integration/dataset_config.json)。
配置 SHA-256：
`e286c064edc4a11abe76ec394a3465b527907f034812b78b27f19eac8dbca3bb`。

数据生成观察（16 个独立 seed / 任务轨迹组，按整个轨迹分组而非逐行随机）：

| Split | 轨迹组 | 有效事件 | 完成 | 继续 | 明确排除 |
|---|---:|---:|---:|---:|---|
| Train | 8 | 338 | 96 | 242 | 未开始89，设备暂停12 |
| Validation | 4 | 174 | 48 | 126 | 未开始44，设备暂停8 |
| Test | 4 | 170 | 48 | 122 | 未开始44，设备暂停8 |

事实来源是新生成的**合成** BusinessSimulator 轨迹，不是 PR15/16 冻结确认集。
不同 seed / 完整任务轨迹没有交叉进入不同 split；
但地图结构模板在各 split 之间仍有复用，因此不能称为“未见地图”泛化。

训练唯一固定方案：`seed1810, Adam lr0.001, epochs10, CPU180秒`；
只用 Train 更新，Validation Brier 选 epoch，Validation F1 选固定候选阈值，
Test 单次评价且不选参。
训练已于 [Actions #42789483227](https://github.com/lieuyimo/wm/actions/runs/42789483227)
真实执行并保存模型：最佳 epoch10；选阈值0.4；
验证 Brier约0.22856；测试 Brier约0.22925；
测试 ECE5约0.16267。
权重 SHA-256：
`84f426a7acdb860b283cc7f19efdc930ab6073d191bd17366949d7b7c52bb68d`。
数据集 SHA-256：
`1b0268dd86bd4c360c29f05fbc97d9fb8263ecf6ded75e62b1ff4e4e50e34b02`。

**重要失效：** 在 Validation 选择的0.4阈值下，新模型对170条 Test
均判定完成，TP48/FP122/TN0/FN0，精确率28.24%、召回100%。
不得以100%召回率暗示良好完成事件识别。
权重元数据 `online_verified=False`，普通加载**拒绝**；
F 仅允许 `experimental=True` 在受控离线研究中调用，不能宣称生产已验证。

## 10. 同事实服务事件的模型比较（Test=170）

| 方法 | Brier↓ | 精确率 | 召回率 | 评价解释 |
|---|---:|---:|---:|---|
| Rule | 0.14118 | 0.7222 | 0.8125 | 冻结确定规则 |
| PETS 原权重 | 0.40526 | 0.3776 | 0.7708 | 原7特征条件完成概率 |
| Structured 原权重 | 0.43779 | 0.3370 | 0.6458 | 原服务 logit + sigmoid |
| Business Session v2 | 0.22925 | 0.2824 | 1.0000 | 新48维目标会话概率 |

同一组有效事实事件、同一时间边界、仅事实标签用于评估。
学习新 v2 的 Brier 好于两个旧学习基线，却明显落后 Rule；
这不是预测改善的证据。

详细的 `session-event-comparison.json`、`session-dataset.json`、
`session-v2.pt` 及训练日志由 CI 作为 60 天保留期的
Actions artifact 存档。**该 artifact 不是永久 Git 归档**，
需要长期保存必须另行纳入机构制品库；不可声称二进制权重已提交 Git。

## 11. 六组真实合成仿真闭环与模型作用

第一次、在任何结果前提交的六组对照配置：
[`session_confirmation_config.json` 初次提交](https://github.com/lieuyimo/wm/blob/d1eb751de016731a6ca7acba7ac8f03eeff765cf/experiments/world_model_integration/session_confirmation_config.json)。
成功执行：[CI #42790810770](https://github.com/lieuyimo/wm/actions/runs/42790810770)。
两执行 seed × 三模式，四任务/案例，共24任务；
相同 CP-SAT 候选、运动、预算、外生配置，F 用离线实验加载。

| 方案 | 总完成 | 平均限制流转时间（步） | 平均决策耗时（ms） | 安全违规 |
|---|---:|---:|---:|---:|
| A Rule+workflow | 24/24 | 43.667 | 16.58 | 0 |
| B PETS+原完整工作流评分 | 24/24 | 44.000 | 19.68 | 0 |
| C PETS+活动条件剩余服务评分 | 24/24 | 44.000 | 22.40 | 0 |
| D PETS+受限 predicted-start 评分 | 24/24 | 44.000 | 22.93 | 0 |
| E Structured+相同评分 | 24/24 | 44.000 | 26.16 | 0 |
| F Session v2（仅离线实验） | 24/24 | 43.833 | 24.04 | 0 |

六个同状态固定候选点：B–F 相对 A **选择改变0/6、排名改变0/6**；
模型候选分数可以不同但没有越过排名边界。
六组同选动作的完整事实反事实平均决策损失1.375步，
且无需看到不可观测未来才完成选择。不能据此主张学习模型改善调度。

后续发现原配置不能确保“同时服务”的充分覆盖，
于是**在看到初次结果之后**额外改变了合成资源并行容量，
并增加1/3/5/10/20步预测误差统计。
该操作已不是预注册独立确认，**必须明确标为事后机制/鲁棒性诊断，
不能取代或覆盖第一次对照**。
[补充诊断 CI #42791585030](https://github.com/lieuyimo/wm/actions/runs/42791585030)
6案例的结果：A平均42.500步，F平均41.750步，C/D均42.583步；
F在新配置下略优于A，但这是事后修改配置的探索数据，**不算独立学习收益**。

在这份补充诊断中，事实有18至23个并发服务观察步/案例、
无故障模式0个设备故障观察步、有突发故障模式各6步。
1/3/5/10/20步预测的完整联合状态 exact（不同步长分母不同）：
Rule为16/24、7/24、5/24、4/24、3/20；
F为10/24、2/24、0/24、0/24、0/20。
所有模型的这些窗口均通过合法性检查，但**合法覆盖不等于预测正确率**。
未知故障与不含外生事件窗口分别计入原始结果，不偷用 Oracle。
在这种较长期场景，学习 F 的状态预测依然不优于规则。

## 12. 官方运动分支与限制

- [CI #42791585030](https://github.com/lieuyimo/wm/actions/runs/42791585030)
  实际安装已有固定来源 PIBT 与 LaCAM3，完成旧工作流回归与
  新的 `tests/test_session_motion.py` 运动集成测试。
  新模型 query 确实被调用；预测分支重置随机种子，
  与真实恢复快照对照重复计算，不修改事实恢复缓存和站点容量。
  官方规划算法/源码未改。
- 既有 PR15/16 结果和 PR9/PR11 检查点未改；
  原 PR15/16 原始同状态数据在本阶段没有重新完整跑一遍，
  新增事实轨迹与本次 CPU 对照是独立于这些旧数据的新合成证据。
- 用户要求的“新模型超越 Rule”不是工程完成条件；
  **本轮功能打通，但学习预测、决策排序、完整工作流价值均未验证优势**。
  更可能同时存在有限数据/标签可预测性、跨轨迹泛化不足、
  候选对短窗服务误差的排名不敏感、未知外生事件不可预测等瓶颈。
  现阶段应停止在这些确认/测试样本上追求有利阈值。


## 13. 真实工作站容量争用与 PETS 预测起始支持（探索性机制回归）

[官方运动 + 训练 + 容量 stress CI #42792492268](https://github.com/lieuyimo/wm/actions/runs/42792492268)
全部步骤成功。该 stress **发生在初次确认结果之后**，因此不算独立确认。

使用现有、未更改的 BusinessConstraints 单容量资源合成案例（2业务任务）。
事实 nearest 基线发生3个资源等待 tick；六组 CP-SAT 策略执行中
各出现1个资源等待 tick，全部完成2/2任务、限制流转16.5步，
0容量/离散运动/生命周期违规，没有模型回退。

新评分读取的逐阶段服务估计来源（候选×粒子×预测窗口中的次数，
不是独立轨迹样本）：

| 模式 | PETS 活动条件剩余估计 | PETS 合法预测起始估计 | 规则先验 |
|---|---:|---:|---:|
| C：仅活动会话 | 14 | 0 | 105 |
| D：允许受限预测起始 | 14 | **4** | 101 |

D 的4次 predicted-start 被清楚标注为 **预测未来状态中的合规起始**，
并通过冻结 PR11 训练特征分量范围检查，不等于已经完成独立校准。
在本案例中新增模型估计未改变完成数量、流转时间或选择收益。
没有任何证据支持将未来设备故障率从 PETS 时长分布推断出来。

此实测表明：实现不仅新增了一个开关，也**真正调用到了冻结 PETS
的合法预测起始服务估计**；但截至本轮，无法证明未来起始分布估计
在真实事实数据上更准确或改善调度。
