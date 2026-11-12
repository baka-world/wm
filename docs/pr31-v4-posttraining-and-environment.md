# PR31：V4 同架构热启动/从头训练、独立测试与完整可恢复环境

**执行日期：2026-11-12，状态：训练、验证、盲测、断点恢复及永久环境归档全部成功。**

## 1. 实际复用的训练方法

- 不更换现有 V4 模型架构：`BusinessSessionEventModel(query_width=56)`（Deep Sets）。
- 不自行设计新的训练损失：复用原 V4 的 PyTorch `binary_cross_entropy_with_logits`、`torch.optim.Adam(lr=0.001)`、batch32、以 Validation Brier 选 best checkpoint、patience10 早停。
- 三臂公平对照：**原始冻结V4**、**从原V4权重 warm-start 并新建Adam**、**相同扩充数据从头训练**。
- 原始 V4 训练 32 组、2784条服务事件、最佳验证epoch16、训练到epoch26；未保存原Adam，因此本轮 warm-start 不宣称是严格恢复原优化器。
- 训练触发来自 PR29/PR30 的高负载和长时跨度跨场景不稳定：PR30 真正在线 H20 在中负载和高负载改善完成时间，但高负载 Rule-H20 已取得相近收益，且单次CPU预算超限；这提供了进一步训练数据覆盖的明确研究动机。

## 2. 预注册数据与独立性

`experiments/world_model_integration/posttraining/pr31_protocol.json` 在训练前固定所有组及新旧权重；所有监督标签来自现有 `build_dataset_v4.build_one` 的原始事实模拟器状态与合法服务完成事件，非Oracle隐藏倒计时。

| 用途 | 组数 | 实际服务事件记录 | 正例 |
|---|---:|---:|---:|
| 原 V4 Train 事实回放 | 32 | 包含于Train | 包含于Train |
| 新扩充 Train | 48 | 包含于Train | 包含于Train |
| Train 合计 | **80** | **9,940** | **1,280** |
| 新 Validation | **12** | **1,810** | **180** |
| 新 Independent Test | **16** | **2,411** | **240** |
| 独立调度 Test | **16** | 16个新场景，真实对照两套首步合法方案 | 不用于模型拟合 |

新增 Train/Val/Test 覆盖2/3/5个任务及不同资源/服务长度/jitter 组合，原V4 Train回放以保留旧分布。此前 PR #29/#30 的测试组没有进入新模型盲测，源组、协议和代码版本均有 SHA256。

**独立测试数据只在训练与 Validation 选权重结束后生成。**

## 3. 已执行的完整训练和测试结果

[GitHub Actions #43143690680](https://github.com/lieuyimo/wm/actions/runs/43143690680) **SUCCESS**，[90天即时 Actions Artifact #13911001730](https://github.com/lieuyimo/wm/actions/runs/43143690680/artifacts/13911001730)。

| 原版/新训练结果 | 原冻结 V4 | Warm-start V4 | Scratch V4 |
|---|---:|---:|---:|
| 新 Validation Brier | 0.071378 | 0.049939 | **0.049863** |
| **独立 Test Brier** | 0.067313 | **0.049349** | 0.050490 |
| 独立 Test LogLoss | 0.215462 | **0.152441** | 0.156950 |
| 主训练 CPU（秒） | 不训练 | 21.28 | 14.38 |
| 新的完整 Adam+RNG 可恢复状态 | 原版无 | 已保存并验证 | 已保存并验证 |

注意：**按预先固定的 Validation Brier，选中的是 Scratch V4**；独立测试中 Warm-start 反而略好。不能事后将最优 Test 值改写为事先的选型结果，也不能把一次测试优势描述为普适优势。

### 16个全新场景的 H20 受限首步候选决策

各臂使用完全相同的候选首步方案真实完成结果，后续策略是 CP-SAT，预测模型使用同样的 H20/P3 及事先固定的“完成工序数多者优先，平局选原CP-SAT第一方案”：

| 每个负载4组，平均候选 Makespan Regret（ticks） | 原 V4 | Warm-start | Scratch |
|---|---:|---:|---:|
| 轻负载 | 0 | 0 | 0 |
| 中负载 | 0 | 0 | 0 |
| 高负载 | 1.25 | 1.25 | 1.25 |
| 高负载服务参数变化 | 0.25 | 0.25 | 0.25 |

**结论：扩充数据对服务完成事件概率建模产生了明显改进，但在这16组独立首步候选排序中暂未改善调度后悔值。** 下游动作评分也可能是瓶颈；若继续，应先验证真正在线调度而非继续追加同一任务上的训练轮次。

## 4. 用户要求的环境与状态保存：已实际实施

训练执行环境（GitHub Actions runner Ubuntu 24.04、Python3.10、Torch2.5.1+cpu、NumPy1.26.4、单CPU线程，0 GPU）在运行前捕获：

1. `environment/environment.json`：Python/OS/CPU、Torch及NumPy版本、git commit及worktree状态、CPU配置、白名单训练环境变量。
2. `environment/pip-freeze.txt`：包含全部已安装 Python 包及精确版本；`environment/requirements-paper-reuse.lock`：现有锁定依赖文件拷贝。
3. `environment/source_manifest.json`：训练程序、原始V4网络、输入编码、事实数据生成器、状态/事件协议等关键源文件 SHA256。
4. `development.json`、`independent-test.json`：冻结事实输入数据、独立源组、源协议哈希；`trained/validation-selection.json`：只基于 Validation 确定的模型选择和训练曲线。
5. `trained/warmstart_v4-best.pt` 和 `scratch_v4-best.pt`：可直接由既有 `load_v4` 加载的最佳推理权重，原始预训练 V4 亦单独保留。
6. 每个新臂 `best-resume.pt` 与 `latest-resume.pt`：保存对应**完整模型权重、Adam状态（含一阶/二阶矩）、Python/NumPy/torch RNG、epoch/step计数、最佳状态、早停计数、训练曲线、训练数据和环境哈希**。
7. **4/4 次断点恢复校验通过**：热启动及从头训练分别对最佳/最新两种断点做两份独立恢复，并进行同一个后续优化步，参数SHA256完全一致。训练恢复是可执行行为而非只保存了一份optimizer名称。
8. `SHA256SUMS`、完整训练日志、测试和评测日志，以及源训练原始权重，用于重建和审计。

### 持久归档

不仅保存 90 天的 GitHub Actions artifact，还创建了一个**GitHub Draft Release**：

[PR31 完整可恢复环境与权重的 Draft Release](https://github.com/lieuyimo/wm/releases/tag/untagged-871b60f7bf013400cb45)

Release tag：`rhwm-pr31-repro-43143690680`，包括 `full-training-and-environment.tar.gz`（含数据、训练权重、optimizer+RNG断点、环境快照、原始权重和所有日志）与 `archive.sha256`。Release为Draft，仓库管理者可以查看并决定是否正式发布；原 PR 继续 Draft，不自动合并、不修改 PR #19。

代码实际执行 Git SHA：`708099c4445eb5c0f072602881bd846e2318747f`；环境精确版本及来源文件的hash已在该次归档中保留。

## 5. 阶段判断与下一项

**后训练工程与事件预测提升：GO。端到端H20调度后悔值提升：这批独立首步场景未观察到。** 下一步应让**Validation选中的 Scratch V4**与原冻结 V4/Rule 进行新预注册场景的完整在线对比，并记录完整模型加载时间、候选翻转、CPU预算、业务完成时间和逾期，才能判断是否值得替换默认策略。Warm-start 可以作为评测对照，但不能用独立 Test 的结果倒过来覆盖 Validation 时做出的选择。

本轮没有使用GPU，也没有修改原模型源代码、原V4训练历史、任何已归档PR29/30实测结果。
