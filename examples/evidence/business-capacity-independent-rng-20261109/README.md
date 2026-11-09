# 多阶段与资源容量：独立随机流确认记录

本目录保存最终的 CPU 开发与独立确认批次。实验设定、限制和结果见 [报告](../../../docs/business-constraints-results.md) 及 [冻结协议](../../../docs/business-constraints-protocol.md)。地图为公开通用基准；工作站、容量、流程、服务时间和故障均是可见的合成仿真假设，不代表真实实验室。

* `scenario-config.json`：冻结配置；包括独立的策略/模型随机种子及与先前批次不同的环境种子。
* `development-raw.json.gz`、`confirmation-raw.json.gz`：完整场景结果、候选、求解状态、模型调用、墙钟耗时及预定同状态分支；`*-summary.json` 为汇总。
* `full-state-confirmation.json.gz`：在新确认集运行前固定、从全部四种事实来源提取的同状态诊断，动作不同的候选从相同前缀独立重放。该分析依据历史批次提出，和本次确认共用场景，不是额外独立试验；`full-state-summary.json` 为摘要。
* `logs/final-reproduction.log.gz`：开发及确认各复跑、补充诊断复跑和最终核验的完整日志；`logs/regression-*.log.gz`：四组回归测试输出。
* `provenance.json`：代码、依赖、地图、模型、策略种子和配置哈希；`SHA256SUMS` 覆盖本目录其余文件。

最初缺少显式容量、以及随机流隔离前的试验均留在[历史证据目录](../business-capacity-20261109/README.md)，其原始数据未被覆盖。完整复现命令见报告；脚本拒绝覆盖非空输出目录。压缩文件可用 `gzip -dc <文件名>` 检查，切换到本目录后执行 `sha256sum -c SHA256SUMS` 验证归档。
