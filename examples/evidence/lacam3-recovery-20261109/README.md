# LaCAM3 恢复实验和上游证据

[报告](../../../docs/lacam3-recovery-results.md)解释指标口径、实现、结果、限制及复现命令；[冻结协议](../../../docs/lacam3-recovery-protocol.md)记录预先确定的触发门槛和预算。任务、站点配置和服务时间是仿真假设。

- `clinic-gate.json`：五个从 PR #13 轨迹提取的原始状态、全部官方完整路径与墙钟/CPU/峰值内存。`clinic-gate-initial.*` 保留首次执行证据。
- `paired.json.gz`：22 个完整场景、每场景三个后端的全部逐步位置、目标、锁定、事件、恢复计划与计时，含未完成任务。`summary.json` 为每场景摘要和完整场景级分组汇总。
- `ablation.json.gz`：事后“只执行第一步”机制诊断；不是独立确认集。`model-alignment.json` 为事实转移、缓存递归和三模型 CP-SAT 小闭环接口记录。
- `pilot-before-strict-retry-*` 和 `initial-capacity-ghost-error.log.gz` 保存第一次配对及初次模型接口错误，没有覆盖：收紧重试键和清除临时容量状态后，按同一配置重新完成正式配对。
- `logs/` 包含官方原始构建、6 项原生测试、README 示例及输出、桥接构建、165 项 Python 回归、两次完整配对及重复核对。实际机器墙钟时延随负载变化。
- `benchmark-config.json`、`provenance.json` 固定场景协议、源提交和公开资源哈希。切换到本目录执行 `sha256sum -c SHA256SUMS` 验证文件，`gzip -dc 文件` 查看压缩记录。
