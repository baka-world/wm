# PR25: 参考 AGV 运输不确定性工作的 pgmpy 贝叶斯延迟风险迁移

## 方法来源与边界

- Qi Tang, Huan Wang, *Data-driven automated job shop scheduling optimization considering AGV obstacle avoidance*, Scientific Reports (2025), DOI https://doi.org/10.1038/s41598-024-82870-1 。论文使用历史数据和贝叶斯网络推断路径障碍/运输不确定性。
- **本 PR 不是原论文结构或实验的精确复现**：本仿真器没有路径障碍概率标签，我们根据实际公开到站事件预测的是**运输段耗时超过已知路径时间基线的概率**，不是障碍发生概率。不能套用原论文的 4.4% 数据。
- 使用开源 [pgmpy 1.0.0](https://github.com/pgmpy/pgmpy)（MIT）的 `DiscreteBayesianNetwork` + `BayesianEstimator(BDeu)` + `VariableElimination` 完成参数学习与后验概率查询。图结构只使用公开距离类别、目标站附近其他机器人、运输阶段 → 延迟标签，**图结构是本项目的工程适配，不是原论文官方代码**。
- 只将 `positive_delay_ticks`（训练集观察的正延迟平均值）用于预报时长的辅助尝试；不把它混同为直接拟合的完整时间分布。

## 冻结训练/测试与真实运行

沿用 PR24 已在实验前冻结的 24 train + 8 validation + 12 holdout 独立场景；实际训练仅用 240 条 train 运输段，其中 32 条发生正延迟。

- [Actions #43131582471](https://github.com/lieuyimo/wm/actions/runs/43131582471)：SUCCESS，6 项交通事实/pgmpy 官方运行测试全部 PASS。
- [原始分组 JSON、已拟合 CPD 模型、日志及校验和 artifact #13905800894](https://github.com/lieuyimo/wm/actions/runs/43131582471/artifacts/13905800894)。
- 0 GPU，没有训练神经网络，不修改路径规则、事实时间生成机制或旧 PR。

| 数据分组 | 场景 / legs | 延迟 Brier：假定永不延迟 | 延迟 Brier：训练阳性率常数 | 延迟 Brier：pgmpy | 时长 MAE：几何基线 | 时长 MAE：pgmpy 期望增量 |
|---|---|---:|---:|---:|---:|---:|
| Train | 24 / 240 | 0.12326 | 0.10817 | **0.09773** | **0.80729** | 1.08740 |
| Validation | 8 / 80 | 0.09896 | 0.09035 | **0.08912** | **0.67187** | 1.01513 |
| Holdout | 12 / 120 | 0.14583 | 0.12472 | **0.10918** | **0.78819** | 1.04452 |

## 裁决

**Bayesian 延迟风险预测作为辅助通道 GO，绝不声称直接取代运输时长预测基线。**

Holdout 延迟风险 Brier 0.10918 优于常数0.12472，说明当前三个公开特征在这批独立合成场景里有可识别的延迟风险信息。但是使用预估平均正延迟来修正到站时间，MAE 1.04452 比原几何基线0.78819更差。预测事件风险的提升尚未证明完整未来状态预测改进。

下一阶段 PR26 可以把这条风险通道并联 V4 的服务完成概率与 RHWM 系统状态预测。任何把风险直接转换为随机停顿的方式都必须独立验证且保留原规则的合法转移，不得默认为更好；如果失败，则只保留风险输出而不注入运动动力学。
