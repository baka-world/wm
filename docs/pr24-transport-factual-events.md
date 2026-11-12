# PR24：运输到站与工作站准入事实采集

**目的**：为已有论文关于 AGV 运输时间偏差的预测方法提供可观察的事实标签。**本阶段只有采集和几何距离基线，没有训练运输模型。**

## 实现与安全边界

- `business_transport/transport_observability.py`：可选的 `ObservableTransportSimulator`，仅在第一次看到机器人实际到达该阶段目标站点时增加 `transport_arrived` 事件；调用原父类执行服务、资源和运动逻辑。不改变通用 `BusinessSimulator` 的旧数据与结果。
- `transport_events/benchmark.py`：将真实 `assigned → transport_arrived` 作为第一段运输，`picked_up → transport_arrived` 作为第二段，`service_started` 仅作为资源准入事件，单独计算到站后的等待。初始输入仅使用 `BusinessObserver` 在起运 tick 的公开机器人/站点位置与资源状态；真实未来时间戳只能用于结果标签。
- 路径基线 `geometry.distance`；第二段补偿完成提取后不可立即移动的执行 tick。该对齐是公开仿真规则，不是拟合 test 数据。
- `transport_protocol.json`：预注册 24 Train、8 Validation、12 Holdout 场景；另封存 12 个未来系统 H1/H5/H10 评测及 12 个 PR27 决策评测种子，不能回头刷种子。

## 真实 GitHub Actions 结果

[CI #43131007044](https://github.com/lieuyimo/wm/actions/runs/43131007044) success，[原始 events/rows/摘要/校验和 artifact #13904619250](https://github.com/lieuyimo/wm/actions/runs/43131007044/artifacts/13904619250)。3 个测试通过。

| 组 | 场景数 | 完整运输段 | 删失 | 路径基线 MAE (tick) | 超路径基线延迟率 | 到站后资源等待段 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 24 | 240 | 0 | 0.8073 | 12.33% | 18 |
| Validation | 8 | 80 | 0 | 0.6719 | 9.90% | 6 |
| Holdout | 12 | 120 | 0 | 0.7882 | 14.58% | 9 |

**结论**：已经存在正负标签，可将“超过规划器/几何时间基线的延迟概率”交给 Tang & Wang (2025) 所借鉴的已有 Bayesian 方法作为下一项**可证伪的参考实现**。不能把这里的排队规则视作未知设备故障，不能把代码中的路径冲突误报成企业实测障碍，也不能把基线 MAE 当作模型误差下降。

**局限**：这批场景来自小型、无真实传感器数据的业务仿真。由于样本量低，运输 Bayesian 网络必须保持低维、明确的参考方法，不用实验后再选择最有利的测试分桶。
