# PR29 补充：条件触发的 V4 后训练与多步误差研究计划

日期：2026-11-12
研究阶段：**计划修订，不是已执行的新训练。** 本文约束 PR29 后续可能的 PR30+ 实验，不修改任何已发表/已归档模型、PR29 事实组或业务规则。

## 1. 原始 V4 到底训练到什么程度：先确认事实

证据来自仓库 `experiments/world_model_integration/v4_final_evidence_manifest.json`、`v4_frozen_config.json`、`train_v4.py` 和 PR29 归档；**不能仅因 H20 不稳定就断言欠拟合**：

| 项目 | 已知事实 |
|---|---|
| 原训练 | 32 个独立合成场景组、2,784 条合法服务事件监督样本（560 positive） |
| 原验证 | 8组、639条监督事件（127 positive） |
| 参数和实体覆盖 | 原训练每组 2 或 4 jobs，1～3 robots，A 容量 1/2，配置服务 ticks 3～8，jitter 0～2 |
| 训练优化 | Adam、BCEWithLogits、batch32、LR 0.001，最大100 epoch、patience10 |
| 实际选择 | 最佳验证 Brier 出现在 **epoch16**，共执行26 epoch，最佳梯度更新1,392，实际总更新2,262 |
| 原模型类别 | Deep Sets + 每个 active service 的目标查询，预测**下一时间单位服务完成事件概率**；规则转移推进完整系统 |
| PR29 新条件 | medium 3 jobs；heavy 5 jobs；`heavy_service_shift` 服务额外+3 ticks、jitter4，超出原训练 jitter 范围 |
| PR29 结论 | medium 的 V4-H20 首步 regret 减少，4胜2平2负，bootstrap区间跨零；其它三类场景无选择收益 |

**解释：** 原训练因 validation 早停，且从第16到第26 epoch 未更新最佳模型。没有证据表明原训练只是梯度步数不足；更直接的假说是**训练场景覆盖/负载组合不足**、输入表示对实体竞争关系不充分、以及滚动误差和决策目标不一致。它们需要区分。第一个训练组仅含2/4 jobs，新测试出现3/5 jobs；这是**场景覆盖差异，不自动等于神经网络容量欠拟合**。

注意原始 `torch.save` 只保存 `model_state`、metadata，没有保存旧 Adam optimizer moments：若从 V4 checkpoint 恢复参数并新建 Adam，是**weight warm-start fine-tuning**，不是“与第26 epoch 无缝接续完全相同优化器状态”；必须按这一准确表述记录。

## 2. 已验证且可借鉴的相邻工作

| 论文 / 代码 | 可迁移的思想 | 本项目不得混淆 |
|---|---|---|
| Janner et al., **When to Trust Your Model: MBPO**, NeurIPS 2019；[paper](https://jannerm.github.io/mbpo-www/)；[代码](https://github.com/jannerm/mbpo) | 评估模型在哪些状态与规划深度值得信任，短真实状态起点滚动能降低长时误差累积 | MBPO是model-based RL，不是 V4 服务事件模型现成的完整后训练器，不能不经适配声称复现 |
| Chua et al., **PETS**, NeurIPS 2018；[论文](https://proceedings.nips.cc/paper_files/paper/2018/hash/3de568f8597b94bda53149c7d7f5958c-Abstract.html)、[mbrl-lib](https://github.com/facebookresearch/mbrl-lib) | 概率模型不确定性、多个独立模型/粒子与有限预算规划的对照 | PR29 三粒子是预测随机性的模拟，不等于已建立 ensemble epistemic uncertainty；多网络要另计CPU |
| Venkatraman et al., **Improving Multi-step Prediction of Learned Time Series Models**, 2015；[作者稿](https://publications.ri.cmu.edu/storage/publications/pub_files/2015/1/Venkatraman.pdf) | 对自回归部署产生的输入分布偏移，使用真实历史对预测状态分布做数据聚合/纠正 | 不直接套 DAgger 的 imitation-policy 更新到已有事件概率网络 |
| Somalwar et al., **Learning with Imperfect Models: When Multi-step Prediction Mitigates Compounding Error**, CDC 2025；[论文](https://arxiv.org/abs/2504.01766) | 单步准确率、直接多步建模、单步网络的多步损失之间的取舍；部分观测/设定偏差时多步目标可能更合适 | 论文分析以线性动力系统为主，不能直接推出离散业务事件 + 不可微规则仿真必然收益 |
| Bengio et al., **Scheduled Sampling**, NeurIPS 2015；[论文](https://proceedings.neurips.cc/paper/2015/hash/e995f98d56967d946471af29d7bf99f1-Abstract.html) | teacher-forcing 与部署时自回归输入分布差异的概念参考 | V4不是自回归 RNN，不能简单按 scheduled sampling 改已有损失；须注意理论批评与可识别性 |

## 3. 决策门槛：必须先辨别哪种不稳定

| 观察证据 | 首要解释 | 先做什么，不做什么 |
|---|---|---|
| Train 与新独立同机制 validation 的一步事件 Brier / NLL 均高；训练曲线尚未收敛 | **可能容量不足或优化未充分** | 先核对原最佳 epoch、训练样本密度及模型偏差；可做**固定训练数据**上 V4 warm-start 对照，但原训练早停使“纯追加epoch”缺乏支持 |
| 原 Train Brier 很低而 held-out high-load / jitter-shift 明显更差，相关特征/事件组在 Train 稀少或缺失 | **分布偏移 / 数据覆盖不足** | 先从现有业务仿真机制生成新的 Train/Validation **新 seed**，覆盖3/5 jobs、jitter4 和服务较长条件；保留原先的训练混合以防灾难遗忘；不改模型结构 |
| 一步事件 Brier 稳定但 H10/H20 误差快速增长，使用模型自生成状态时更差 | **rollout distribution mismatch / 累积误差** | 参照 MBPO/PETS 优先对比短预测跨度或更频繁用真实观测重新锚定；若仍需训练，先找可忠实实施的多步监督或数据聚合论文方案；不能将不可微工作流强行端到端反传 |
| 状态预测与事件预测合理，但终局 makespan 排序不稳定或 CP-SAT 已最佳 | **决策目标失配 / 无改善空间** | 优先检验候选评分、任务关键路径与是否实际存在不同结果，**不启动后训练** |

原则：上述条件应在**新 Train/Validation 数据、原档训练记录和不涉及未见 Test 结果调参的诊断**下判定。PR29 的32组现在已被读过，只能作为研究参考，不再作为后训练选择模型的最终盲测。

## 4. 若满足触发条件，如何实际开展有参考的 post-training

训练优先采用**沿用现成实现，不自行设计算法**的三臂对照：

- **Arm A：冻结原 V4**，无新训练。
- **Arm B：同一 V4 架构、从原已保存的 `model_state` warm-start**，使用原训练的 BCEWithLogits / Adam / train-only factual service event query 数据，Validation-only early stopping 与 best-checkpoint 选择；将继续训练的 optimizer 重置如实标记。
- **Arm C：同一 V4 架构，随机初始化**，在与 B **完全相同的扩充训练数据**上训练同一预算，判断收益来自已有权重还是仅仅更多真实训练覆盖。

**默认不立即实施 Arm D（多步训练新损失）**。只有先证明 H20 的误差主要来自 teacher-forced 一步与 rollout 不匹配，并找到与 V4 条件二元事件及离散合法转移相容的正式论文实现时，才制定单独的预注册实验；不能为了提升某个测试集而自造“多步BCE”并冒称复现。

### 数据纪律与验证设计

1. 原 V4 权重及其 SHA256 `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c` 永久冻结，任何新权重使用新文件名、新 SHA、新 model_version，不覆盖。
2. 从现有 `build_dataset_v4.py` 的**公开状态观测、合法会话监督标签**扩充独立 Train/Val 场景；不能从离线事后更优方案或隐藏 `service_remaining` 生成训练输入。保持 train/val 按 scene seed 分组。
3. **预先冻结新 Test 组**，与原 PR21～PR29 的全部已见独立实验 seed 分离；不得将 PR29 的 heavy cases 纳入 Test 选择/超参优化后又当作泛化验证。若采用 PR29 的观察设计新 Train 覆盖，必须在论文说明这是后训练开发动机，盲测只能来自真正新 seed / 新机制组合。
4. 一步监督报告 train/val/novel-Test Brier、NLL、ECE、按 station/phase/elapsed/服务预计工时/job count/robot count 分层；完整系统单独报告 H5/H10/H20、已知终止覆盖、robot/resource/sample state errors、动作排序与**有限候选真实后悔值**；训练 loss 改善本身不作为成功。
5. 使用与原 V4 相同的架构和优化超参作为首轮**参考基线**。其他训练配置若有调整，必须说明文献来源、唯一变化因素、新验证数据选择准则，停止在同一 Test 上反复挑选。
6. 优先 CPU（原 V4 纯训练约 4.4 CPU 秒），先做小规模完整数据流水线验证后再定总预算。无需预先占用 V100。冻结训练/验证与盲测各自 CPU 开销、训练 group 数、总梯度更新、最佳 epoch。
7. **候选晋升条件**：新模型在**新独立 Test**上具有更好的 H20 完整状态预测或决策 regret，且在原覆盖范围内不明显退化，同时不超过已定 CPU 预算；至少报告各场景胜/平/负和区间。若未达到，保留 V4，不因为“再多训一点”修改 Test 协议。

## 5. 下一阶段顺序

1. **先执行计划中的独立 H20 在线调度实验**，报告完整 CPU/数据开销与负面结果，不能被后训练分支自动替代；
2. 若仍不稳定，参考本文件定位优化不足、场景覆盖不足、rollout 累积误差、或动作价值失配；这一步不依赖重复运行旧测试组；
3. **证据指向模型可改善的不足**时，再开新 Draft PR 执行有来源的 V4 warm-start 与 scratch 对照，并冻结新 Train/Validation/Test；
4. 如果新 Test 的状态和决策指标均未改善，停止后训练，不再不停更换网络刷结果。

最终研究表述：**“将模型适用范围与继续训练视为两个独立研究对象：首先在未见场景验证预测/决策泛化，再在有诊断证据时使用原开源/已验证算法及新增独立数据进行后训练。”**

本文仅是研究方案与事实门槛，不包含任何已完成后训练的宣称。
