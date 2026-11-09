# 指定论文源码复用：CP-SAT + PETS 第一轮系统集成

本轮在 PR #10 的 `bea358cdb0b7729655123aa02014d5ff88a64aea` 上完成可运行的 CPU 原型：OR-Tools CP-SAT 生成当前合法分配候选，MBRL-Lib 概率集成预测有效服务时长，现有规则负责状态转移，五步推演评分选择候选。它是合成场景的系统集成验收，不是自动化实验室的临床或生产性能证明。[冻结协议](paper_reuse_protocol.md)记录了开发、求解器预算修正和独立确认种子。

## 来源、许可及实际复用范围

| 来源 | 固定版本与许可 | 实际使用 | 未声称复现的内容 |
|---|---|---|---|
| Zhou 等，2025，[A multi-robot–multi-task scheduling system for autonomous chemistry laboratories](https://doi.org/10.1039/D4DD00313F) | 论文为业务问题参考 | 实验室多机器人任务与资源调度的背景 | 没有复制论文实验数据，未声称复现其完整实验调度系统 |
| Google [OR-Tools flexible_job_shop_sat.py](https://github.com/google/or-tools/blob/100f66e6242ab8bf8d32feb8f3bf086db66ae2b5/examples/python/flexible_job_shop_sat.py) | 源码 SHA `100f66e6242ab8bf8d32feb8f3bf086db66ae2b5`；Python 包 `ortools==9.14.6206`；[Apache-2.0](licenses/or-tools-Apache-2.0.txt) | 原样运行官方示例；以 `CpModel`、可选区间、`AddNoOverlap`、先后依赖与 makespan 建立当前分配模型 | 官方示例是固定实例，不直接处理本项目 Observation；项目约束由薄适配层映射，CP-SAT 不是世界模型 |
| Chua 等，2018，[PETS 论文](https://arxiv.org/abs/1805.12114) 与 Meta [MBRL-Lib](https://github.com/facebookresearch/mbrl-lib/tree/3f93cccfc8d635f74e335a2f07aab6e9a48fc021) | SHA `3f93cccfc8d635f74e335a2f07aab6e9a48fc021`；`mbrl==0.2.0`；[MIT](licenses/mbrl-lib-MIT.txt) | 未修改源码的 `GaussianMLP`、`OneDTransitionRewardModel`、`ModelTrainer`、`ReplayBuffer` 和 `BootstrapIterator` | 未使用 PETS 连续控制入口、MPC/CEM 策略或原论文实验结果，也没有另写集成网络内部结构 |

Python **3.10**、CPU PyTorch **2.5.1+cpu**、NumPy **1.26.4**、OR-Tools **9.14.6206**、Hydra **1.0.3**、OmegaConf **2.0.6**、Gymnasium **0.26.3**。这些保持 MBRL-Lib 的原始依赖范围；完整版本见 [requirements-paper-reuse.lock](../requirements-paper-reuse.lock)。Python 3.10 是上游声明支持的版本，不需要改写上游代码以兼容新 Python。

两项独立上游核验实际完成：官方 OR-Tools 文件哈希校验后原样执行，`status: OPTIMAL`，示例 makespan 为 **6**；MBRL-Lib 的 5 项原生模型、训练器和回放缓冲测试通过。另以上游 `GaussianMLP`、`ReplayBuffer` 和 `ModelTrainer` 对玩具数据运行 3 个 CPU epoch，验证 3 成员输出形状、有限损失及参数更新。该玩具测试仅验证上游接口，不是本项目的训练或性能结果。源码检出后 `git status --porcelain` 为空。完整命令、哈希和输出在[证据目录](../examples/evidence/paper-reuse-integration-20261109/README.md)。

## 接口与信息边界

[CP-SAT 适配器](../lab_transport/paper_reuse/ortools_dispatch.py)只接受当前 `Observation`、已有静态 `Geometry` 和模拟器当前合法 `candidates`。每个任务是同一机器人执行的“前往取样及取样”“前往送样及送样”两阶段；距离用 `Geometry.distance()`，工序严格先后、每个任务和机器人至多选一次，机器人可选区间使用 `AddNoOverlap`。目标先最大化当前分配数，再最小化完成时间上界；没有额外工作站容量或虚构工序。它最多枚举 4 个当前匹配，每次求解限制 0.5 秒墙钟和 0.05 确定性时间、单线程。不可达/忙碌/无效候选被排除；无方案或求解失败使用确定性合法回退并记录原因，仍由仿真器检查路径冲突与动作。

[PETS 服务模块](../lab_transport/paper_reuse/pets_service.py)使用上游 `GaussianMLP` 的 **5 成员、2 层、32 隐藏单元**，输出对数有效服务时长的均值和方差。概率分布经生存概率计算 `P(T≤e+1 | T>e)`，其中 `e` 为已观测有效服务进度；固定种子随机采样完成事件，还可计算服务剩余工作量的条件期望。每个未来粒子的预测输入只来自当前或已预测可观察状态；未观察到服务进度时明确按零处理，不读取隐藏倒计时。未知新任务和故障不会成为预测事实。

为了复用上游 `OneDTransitionRewardModel` 和 `ReplayBuffer`，缓冲的一个常零 observation 是占位符，action 槽存放**当前服务协变量**，next-observation 槽是对数时长标签；这里没有把协变量宣称为真实机器人动作。模型只输出服务分布；[PETSRuleDynamics](../lab_transport/paper_reuse/pets_service.py)通过已有 `ProgressAwareRule.predict_next()` 执行移动、任务生命周期、站点可用性与状态关系，网络不直接输出任意任务状态。额外合同检查机器人与样本 ID、邻接移动、独占、任务状态合法转移和机器人/样本关联。

[RolloutDispatch](../lab_transport/paper_reuse/rollout_dispatch.py)对相同 CP-SAT 候选使用 PR #10 的五步 `W + R, C, D` 工作量评分与 3 个粒子：规则组用历史时长估计 R，PETS 组用概率模型的条件期望估计 R，并在服务开始时采样。所有组维持相同已知任务、候选预算和按任务键固定的外生扰动。后续动作在各自预测状态上执行已有最近优先批量分配；不读取真实轨迹的未来动作。仿真器持续负责实际离散安全检查。最近优先基线只预测自己的候选，因此比 CP-SAT 多候选组计算更少。

## 训练数据与删失

直接从 PR #9 已生成的事实训练/验证轨迹提取服务样本，没有为训练重生成大批场景：**352 条训练、176 条验证，共 528 条完整服务**，其中 **24 条**有故障暂停；暂停步数只用于离线审计，不是模型输入或有效时长标签。轨迹中的服务已开始但未完成的样本排除数为 **0**，服务开始不可见而排除数为 **0**；单测另外构造了未完成服务，确认其被标记为删失而不是当作完整标签。提取程序只在服务实际完成后形成标签，特征来自服务开始时记录的历史和当前状态。PR #9 的 test 轨迹未用于训练。

归档保存约 5 KB 的派生训练/验证样本、源 `manifest.json` SHA-256 和 33 KB CPU 检查点；原始约 70 MB 的历史轨迹没有重复提交。派生样本可直接重训；如需从头独立审计每条原始转移的提取，仍需取得 PR #9 的原始轨迹或其历史运行工件。固定种子重新训练时，模型张量与归档检查点逐项相同。

## 独立确认结果

固定方案后使用 Tartu `49002/49003`、clinic `50000/50001`，各运行正常、集中到达、处理中断、到站不可用四类场景；任务签名与 PR #8–#10 和本轮已查看的开发/试验场景均不重叠。每种策略的 16 场景含 **96 个任务**。下表流转时间单位为仿真步，CPU 时间为本地每次调度平均耗时：

| 组合 | 完成任务 | 平均流转 | 每次决策 CPU | 每次决策 PETS 前向 | 回退 / 离散违规 |
|---|---:|---:|---:|---:|---:|
| 最近优先 + 规则 | 96/96 | 104.667 | 0.88 ms | 0 | 0 / 0 |
| CP-SAT + 规则 | 96/96 | 101.823 | 9.63 ms | 0 | 0 / 0 |
| CP-SAT + PETS | 96/96 | 101.844 | 21.12 ms | 49.95 | 0 / 0 |

PETS 组实际执行 **4,196 次**网络前向，其中 **620 次**直接用于服务状态转移、**3,576 次**用于估计剩余服务工作；推演中采样出 99 次服务完成事件，这些是预测事件而非真实服务记录。三组均无重复分配，CP-SAT 枚举无部分截断。场景没有业务来源支持的样本截止时间，因此超时数量为 **null（未定义）**，不能报告为零超时；全部任务均在仿真运行上限内完成。

故障确实作用于执行：确认集处理中断的 4 个场景合计记录 80 个站点不可用等待步，到站不可用的 4 个场景合计 76 步；三组数字相同，正常与集中到达场景均为 0。这是实际不可用等待计数，不代表故障造成的全部额外流转时间。

按完整场景配对，CP-SAT+规则相对最近优先平均流转 **−2.844 步**，16 场景为 10 好、5 差、1 相同；PETS 相对 CP-SAT+规则为 **+0.021 步**，1 好、2 差、13 相同。每组只有 **4 个独立地图/种子簇**；对应的分层 bootstrap 探索性区间分别为 [−3.667, −2.021] 和 [−0.271, 0.313]，不足以证明对其他地图或真实实验室泛化。相对 CP-SAT+规则，PETS 增加约 **11.5 ms/决策**，未在本组证明额外调度收益。

### 预测指标

在相同最近优先事实轨迹上按场景平均，确认集有 192 条已完成服务、525 个可用站点的服务完成判断机会：

| 指标 | 规则 | 原 PR #9 结构化服务头 | PETS |
|---|---:|---:|---:|
| 服务有效时长 MAE（步） | 0.638 | 不适用 | 0.607 |
| 时长连续密度 NLL | 不适用 | 不适用 | 0.253 |
| 标称 90% 时长区间覆盖率 | 不适用 | 不适用 | 95.3% |
| 服务完成事件 Brier，越低越好 | 0.178 | 0.139 | 0.165 |

规则和既有完成头没有同口径的服务**时长概率分布**，故其 NLL、区间覆盖率和结构化头的时长 MAE 不适用；没有为填表虚构分布。PETS 的 NLL 是对连续对数正态混合密度的评价，服务时长在仿真中为离散步数。覆盖率与 Brier 仅在本批合成场景描述，未证明跨场景校准。

已知实体联合状态正确率（先逐场景求比例，再对场景平均）：

| 预测窗口 | 规则 | 原结构化模型 | 原结构化服务 + 规则移动 | PETS + 规则移动 |
|---|---:|---:|---:|---:|
| 1 步 | 88.3% | 70.1% | 89.0% | 88.0% |
| 3 步 | 78.8% | 52.4% | 79.2% | 77.7% |
| 5 步 | 70.2% | 43.5% | 71.9% | 70.6% |

预测窗口按固定规则每 5 个合格窗口抽 1 个，并排除了窗口内有未来新分配、新任务或站点可用性变化的记录；因此不是所有异常时段的总体正确率。它们是同一场景内相关窗口，不是独立统计样本。PETS 一步联合正确率尚未达到 99% 的长期工程目标。

## 实际问题及复现

原始上游实现无需修改。首个上游 pytest 命令误写了测试名，未运行任何测试，随后使用仓库中真实存在的测试名重跑并全部通过。首个小型闭环检查发现未定义截止时间时汇总器不能把 `null` 相加，已修复为保留 `null`。首次完整试验用 0.05 秒墙钟求解时，在同一开发场景一次追加候选得到 `UNKNOWN`，导致两次运行候选数量不同；已记录原始诊断，增加确定性预算及截断检查，再更换独立确认种子。修复后两次完整运行的候选评分、分配动作及结果逐项一致，耗时除外；墙钟上限在严重负载下仍可能触发回退，日志会显式显示，不能保证任意硬件上的时间相同。

在已有 `uv` 与 Python 3.10 的 CPU 环境：

```bash
uv venv --python 3.10 .venv-paper
uv pip install --python .venv-paper/bin/python --index-strategy unsafe-best-match -r requirements-paper-reuse.lock
source .venv-paper/bin/activate
bash tools/paper_reuse/verify_upstreams.sh /tmp/wm-upstreams-new
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/service_value/tests -v
python -m unittest discover -s experiments/paper_reuse/tests -v
bash tools/paper_reuse/run_acceptance.sh /tmp/wm-paper-acceptance-new
```

脚本拒绝覆盖非空输出目录；会在 CPU 上用归档的既有标签重训小模型作功能检查，以冻结检查点开展两组闭环与预测评价，再各复跑一次比较候选、分数、动作和结果。输出完整场景配置、机器可读原始结果和分析。工作流 [.github/workflows/paper-reuse-integration.yml](../.github/workflows/paper-reuse-integration.yml)使用标准 CPU runner；上游原始文件和许可证在独立临时目录取回并校验，不修改第三方仓库。
