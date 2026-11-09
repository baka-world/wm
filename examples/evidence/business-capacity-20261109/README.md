# 多阶段与资源容量对照：完整证据

**历史批次：**本目录保存隔离策略与环境随机数之前的结果；代码未使用种子读取未来，但为避免随机流关联，最终确认改用独立随机流和新的环境种子。本目录既有原始数据未被覆盖，[历史报告](../../../docs/business-constraints-shared-rng-pilot.md)记录当时的结果；`SHA256SUMS.pre-rng-isolation` 保留原有哈希清单。

实验设定、限制及结果见 [报告](../../../docs/business-constraints-results.md) 和 [冻结协议](../../../docs/business-constraints-protocol.md)。配置中的所有设备容量、流程、服务时间与故障均为合成仿真假设。原始轨迹不包含可以在线使用的未来事实；反事实分支及其离线结果只能用于事后诊断。

* `scenario-config.json`：最终固定配置；与源码配置一致。
* `development-raw.json.gz`、`confirmation-raw.json.gz`：完整场景结果、候选、求解状态、模型计数、时间和预先抽样的独立分支；相应 `*-summary.json` 为汇总。
* `full-state-confirmation.json.gz`：确认结果公布后追加的探索性全部决策点诊断，含四种事实来源、同状态分数和动作不同的独立分支；`full-state-summary.json` 为其摘要。相关状态不能看作独立样本。
* `logs/final-reproduction.log.gz`：开发及确认各复跑、补充诊断复跑和最终核验的完整输出；`logs/regression-*.log.gz`：四组实际回归测试记录。
* `pre-explicit-capacity-*`：修复所有站点必须显式给出容量之前的原始配置和首次试验记录；这些已看过的种子不用于最终独立确认。
* `public-hospital-three-robot-pilot.log.gz`：原有医院地图三机器人未完成的开发试验，保留失败及实际限制。
* `provenance.json`：基准代码、固定依赖、地图、模型和当前配置哈希；`SHA256SUMS` 覆盖本目录所有其他文件。

从仓库根目录运行 `sha256sum -c` 时需先切换到此目录；完整复现命令见报告。压缩文件可使用 `gzip -dc <文件名>` 检查。实验脚本会拒绝覆盖非空输出目录。
