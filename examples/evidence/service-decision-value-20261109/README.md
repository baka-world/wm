# 本轮服务预测决策价值：原始证据

结果解读、统计口径和复现命令见[实验结果](../../../docs/service-decision-results.md)，事前协议及后续新增的探索性检查见[实验协议](../../../docs/service-decision-protocol.md)。本轮使用 PR #9 既有 CPU 检查点，并校验其哈希和训练合同；没有复制检查点、训练数据或历史结果。

- `scenario-config.json`：开发/确认完整仿真参数和时间表，**仅供离线场景复现**；其中的故障和未来任务计划不得送给在线模型。
- `counterfactual-{development,confirmation}-raw.json.gz`：每个初始状态、候选动作的模型预测、单独反事实重放、短期服务事件和完整结果。位置以哈希记录；其他条件固定后逐候选独立重放。
- `counterfactual-*-oracle-audit.json.gz`：第二次独立重放复核的逐候选服务事件、服务开始及结束、有效时长；含特权未来信息，只能离线使用。
- `aligned-*-delta.json.gz`：后续分配一致性的探索性补充结果，通过布局、种子、场景、决策 tick 及动作关联到对应 counterfactual 原始记录；避免重复保存真实轨迹。
- `online-*-raw.json.gz`：四种可在线使用配置的闭环指标、逐决策耗时和动作；场景配置引用同目录 JSON。
- `*-summary.json`、`*-analysis.json`、`online-*-paired.json`：原始记录派生的窗口、事件、排序、配对及闭环汇总。
- `initial-original-horizon-development-summary.json`：冻结新评分前的开发集原评分窗口诊断结果；不是确认集。
- `logs/`：本地实际执行的完整实验、回归测试和逐候选复核日志，gzip 压缩。GitHub Actions 的运行和日志也在对应 PR 的检查项中。

`SHA256SUMS` 记录归档文件内容，可在本目录执行 `sha256sum -c SHA256SUMS` 校验。所有场景为合成实验假设；离线参考包含未来信息，不是可部署调度器。
