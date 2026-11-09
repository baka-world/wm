# 服务动力学 CPU 消融证据

详见[实验报告](../../../docs/service-dynamics-results.md)和[固定协议](../../../docs/service-dynamics-protocol.md)。所有数据来自公开地图上的合成仿真，不是医疗或设备记录。PR #8 的原 GPU 汇总另保存在 `../pr8-a100-baseline-20261109/`。

- `dataset-manifest.json`：拆分、种子、源地图哈希、事实轨迹哈希；`dataset-coverage.json.gz` 记录事后故障覆盖与场景参数。场景元数据不是模型输入。
- `training.json`：三个变体的配置、类别权重、验证阈值、逐轮损失、数据和源码哈希；三个 `.pt` 是本轮小规模 CPU 训练的实验检查点，不是 PR #8 的 GPU 检查点或生产模型。
- `prediction-metrics.csv`、`closed-loop-metrics.csv`：主要测试与配对闭环结果；`evaluation-summary.json.gz` 和 `prediction-cases.json.gz` 保留各模式与逐轨迹计数。
- `same-state-redacted.json.gz`、`closed-loop-cases-redacted.json.gz`：候选、评分与结果；为避免重新分发地图衍生坐标，预测位置替换为稳定位置哈希。完整轨迹可用脚本和公开源重新生成。
- `initial-evaluation-summary.json.gz`：增加同一三步模型 0.50 阈值诊断之前的首轮汇总，保留分析时间顺序。该补充没有重新训练或根据测试集选择阈值。
- `SHA256SUMS`：上述归档内容校验；本轮训练、数据、实验和单测完整命令输出另见 `validation/logs/service-*.log`。

在当前目录执行 `sha256sum -c SHA256SUMS` 可检查全部证据。训练和数据导出拒绝覆盖非空目录；检查点与数据 manifest 及关键模型源码哈希绑定。
