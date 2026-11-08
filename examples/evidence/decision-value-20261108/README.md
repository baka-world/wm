# 决策价值 CPU 运行证据

`pilot-summary.json` / `confirm-summary.json` 是两组预先固定种子的本地汇总；`*-decisions.json.gz` 包含场景假设、在线策略决策、离线候选分支以及预测与参考的 1、5、20 步状态。位置仅保留稳定散列键，不包含原始路网坐标或源地图。任务均为合成，离线参考包含事后信息，不得作为在线输入。

`ci-` 前缀是 [GitHub Actions 42523304631](https://github.com/lieuyimo/wm/actions/runs/42523304631) 的实际产物；除 CPU 计时及环境信息，云端与本地逐项结果一致。GitHub 产物的临时保存期为 14 天，仓库内压缩副本长期随此分支保留。

运行口径、局限和复现命令见 [实验报告](../../../docs/decision-value-experiment.md)。对应本地运行日志位于 `validation/logs/decision-value-*-local-20261108.log`；CPU 时间取决于运行环境，汇总不是独立同分布统计检验。
