# 长期调度价值可学习性与 ADP 方法选型

独立、CPU-only、PR18 Draft 实验。实验脚本采用 tick0 的真实 BusinessSimulator 完整重启，首步执行相同合法候选，之后固定 P1 nearest。不同候选使用同一随机扰动 seed，随机服务时长最多四组、确定性仅一组。尚未验证非零 tick 条件未来重采样，禁止使用中途状态做主实验。

`protocol.json` 冻结 48/12/16 个独立场景组、三组不重叠的地图 layout、服务时间、释放配置、随机 seed 和完整测试协议。即使某场景只有一个候选或零真实差距，也必须保留在报告中，不按 Oracle 优势筛选。

比较 B0 Rule 5×3，B1 低复杂度确定性启发式，B2 当前可观测特征 OLS/Ridge，B3 V4 5×3，B4 Rule 5×3 终点摘要加同类 OLS/Ridge，B5 V4 5×3 终点摘要加同类 OLS/Ridge。B6 的经验风险模型不具有同等完整状态转移接口，因此不强行适配。三种价值回归均拟合同状态中心化完整成本标签，特征标准化只用 Train，OLS 与 5 种 Ridge alpha 只通过 Validation 排序损失选择。Test 仅在模型文件和选择 SHA 写入 test_lock.json 后执行一次。

真实成本仅用于离线标签、后悔值、排序评估；模型和启发式只读取当前合法观察、公共规则、geometry、候选动作，以及适用的 Rule/V4 自身预测。测试数据没有用于选择特征或参数。多候选、多随机未来按**独立场景组**汇总，不当独立样本扩张显著性。所有组和原始各候选结果归档在 evidence 中。

`python -m unittest discover -s experiments/world_model_integration/value_learnability/tests -v`

`python -m experiments.world_model_integration.value_learnability.run --v4 /path/to/v4-data.pt --out /tmp/value-preflight --phase preflight`

`python -m experiments.world_model_integration.value_learnability.run --v4 /path/to/v4-data.pt --out /tmp/value-full --phase full`

预检需确认 CPU 总预算可覆盖全组后再执行正式 run。实验墙钟预算 1800 秒，满额停止；未完成测试须 BLOCKED，不得报道部分测试为独立确认。数值结果依赖真正执行，禁止空想填表。V4 需 SHA-256 匹配指定冻结权重，禁止替代。`test_lock.json` 是独立测试前封存的选择记录。

局限：纯仿真、仅 tick0，无当前活动服务会话；不能外推任意滚动重调度，更不能声称已实现完整 ADP 策略改进。版本中的模型只是近似策略评价实验。
