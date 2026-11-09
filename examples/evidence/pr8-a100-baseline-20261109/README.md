# A100 GPU 训练与效果验证（2026-11-09）

本目录保存结构化世界模型在单张 NVIDIA A100-SXM4-80GB 上的精简实验证据。完整报告见
[`docs/structured-world-model-a100-gpu-20261109.md`](../../../../docs/structured-world-model-a100-gpu-20261109.md)。

## 结论摘要

- GPU 训练验收通过：10/10 epoch，墙钟 60.41 s，PyTorch 分配器峰值 65.8 MiB；验证损失 0.4961→0.3021，最优 epoch 10。
- 完整训练模型明显优于此前 CPU smoke 模型（相同 200 条测试前缀：Tartu 一步 0.655→0.775，5 步 0.462→0.5449；clinic 一步 0.655→0.800，5 步 0.279→0.3770）。
- 仍明显落后修正规则（全量测试一步联合准确率 Tartu 0.7787 vs 0.9713；clinic 0.8031 vs 0.9333；5 步 −31.35/−24.30 个百分点）。
- 调度闭环节奏没有改善：16 个场景全部 6/6 完成，平均流转时间 `nearest` 75.864、修正规则 76.760、神经 77.344 步；工作站服务等待/最大积压/最大等待完全相同。神经比修正规则差 0.583 步（0 胜/4 负/12 平）。
- 主要限制：训练集故障覆盖极低（11,848 条训练转移只有 10 个真实服务暂停 tick）、没有失败 episode、车队只有 2 个机器人；一步状态准确率与闭环流转时间没有直接优化关系；均值池化集合网络缺少成对关系。

## 文件

| 文件 | 内容 |
|---|---|
| `loss-curve.csv` | 10 个 epoch 的 train/validation loss 和耗时 |
| `training-report.json` | 训练配置摘要、每轮报告、峰值显存、最佳检查点 |
| `dataset-splits.csv` | 各 split 的 episode/transition 数 |
| `dataset-fault-coverage.csv` | 按 split/regime 的真实服务暂停覆盖 |
| `dataset-audit.json` | 逐 episode 的故障、服务、实体数量审计 |
| `eval-one-step.csv` | 全量测试集一步联合/机器人/任务/服务完成指标 |
| `eval-multistep.csv` | 3/5 步条件预测、覆盖率和指标 |
| `eval-summary.json` | 全量评估汇总（含 by-regime） |
| `eval-run-200.json` | 与 CPU smoke 相同的 200 条测试前缀结果 |
| `predict-bench.json` | 单步延迟和批量前向吞吐 |
| `closed-loop-policy-means.csv` | 5 个策略在 16 场景的均值指标 |
| `closed-loop-paired.csv` | 神经 − 修正规则的配对差值、bootstrap 区间和胜负平 |
| `closed-loop-summary.json` | 闭环配对汇总 |
| `closed-loop-cases.json` | 逐场景、逐策略的完整指标 |
| `gpu-monitor-training.csv` | 训练期间 1 s 采样的 GPU 利用率/显存/功耗/温度 |
| `gpu-monitor-eval.csv` | 全量评估期间 1 s 采样 |
| `gpu-stats.json` | 上述采样的汇总和分配器峰值 |
| `SHA256SUMS` | 本目录证据文件校验和 |

## 复现与边界

- 代码：`main` commit `0364a63447b7ecc67ea1872ba938afb3536fefc1`；本 PR 不修改模型/训练/评估代码。
- 检查点：`epoch-0010.pt`，SHA256 `3572c6c3697c4ca76944519ff0a612432bd65d922fd339dc0835036c6c75dfdf`；检查点、完整数据集和原始日志不提交 Git。
- 评估/闭环入口会校验数据 manifest、适配器源码和共享代码哈希；本轮重新拉取同一 commit 后 8 个哈希逐项一致。
- 全部为公开地图上的合成仿真，不是临床或真实设备记录；不能据此宣称故障泛化或调度改善。
- 闭环中 `nearest/world/rule` 是原生 Simulator 策略，可能同一 tick 分配多个任务；两个外部 RolloutPolicy 每次分配 1 个任务。神经 vs 修正规则是严格同预算配对，其他比较只作参考。
- 本批场景无 deadline，`late_count` 不适用。
