# CP-SAT 与 PETS 集成：可复现证据

[实验报告](../../../docs/paper_reuse_integration.md)和[冻结协议](../../../docs/paper_reuse_protocol.md)说明方法、结果、上游来源与局限。两个上游源码不在本仓库内修改或重新发布，准确 SHA、许可证及被使用模块见 `upstream-provenance.json`。

- `existing-service-labels.json.gz`：仅来自 PR #9 历史 train/validation 事实轨迹的服务开始时特征及事后完成的有效时长标签；包含来源清单哈希、删失计数，不含测试分组。`pets-service-cpu.pt` 是本轮由这些数据训练的冻结小型 CPU 检查点。
- `scenario-config.json`：开发和独立确认的 32 个合成场景、随机种子和外生故障。配置含未来故障/任务，仅供线下复现，不能输入在线模型。
- `development-raw.json.gz`、`confirmation-raw.json.gz`：每个完整场景三种策略的样本结局、逐决策候选和评分、CP-SAT 状态、模型调用及预测指标。`*-summary.json` 和 `*-analysis.json` 分别汇总闭环和以地图/种子簇配对的统计。
- `training-metadata.json`：冻结检查点对应的训练/验证数量、上游 SHA、数据哈希和训练损失；使用 `[0]` observation 加协变量 action 槽适配上游 ReplayBuffer，并非真实机器人动作。
- `pre-budget-fix-*.json`：修正求解器墙钟预算之前的探索性结果和同种子枚举截断诊断；这些已查看的试验种子不再当作独立确认。
- `logs/`：两家上游原始示例/测试、训练、完整双次复跑、历史与新增回归、初期失败及修复记录。最终 GitHub Actions 也在 PR 检查项中。

所有数据均为合成仿真；模型调用不得读取本目录的未来事件配置或事后标签。可在本目录执行 `sha256sum -c SHA256SUMS` 验证文件；重复执行时请使用新的空输出目录。
