# PR #16 冻结离线诊断证据目录

设计和原始来源：`experiments/oracle_value/config.json`、PR #15 HEAD `d7d29923c77a026ed3f4ec5fb3e2934bc4385200`。

执行顺序：历史 16 决策点对照 > 公平性验收 > 冻结的 24 场景 / 8 地图种子簇确认。

成功的 CPU GitHub Actions 将在本目录下生成 `history/`、`confirmation/`、`SHA256SUMS`，并自动按原始确认集更新 `docs/world-model-value-ceiling.md`；如失败，只能从该 Actions 运行附件获取部分日志与失败证据，不能将空目录误报为完整结果。

重要限制：单一随机实现的完整结果 Oracle 是事后机会空间，不是稳健在线收益；在任何可能的额外随机实现前必须验证同事实前缀、隐藏事件信息边界和后续策略保持不变。
