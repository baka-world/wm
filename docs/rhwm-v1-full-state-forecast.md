# RHWM-v1: 完整可观测系统状态预测（PR #20）

## 当前阶段边界

本 PR 从 PR #18 原目标分支合并提交 `d8fa4660490d11b4fe5f48aaeaf34d8ddb3c5b65` 建立。PR #18 已合并到 `audit/model-inventory-pr17-20261110`，尚未合并主分支 `main`。PR #19 工程演示不属于本 PR，禁止修改。

**P0 是世界模型，不是 UI/服务端管理功能。**

本次可执行的 `RHWM-v1` 是完整业务系统的**混合状态世界模型**：

- 输入：当前 `ObservationV2`、当前合法 `Assignment` 集合、未来明确声明的公开状态续调度策略。
- 概率源：冻结 V4 对可观测且已开始的服务会话给出 `P(finish at t+1 | state, action)`；或使用规则模型作为严格对照。
- 状态推进：已有 `WorkflowDynamics`、`CapacityDynamics`、`ProgressAwareRule`、公开工作流后继释放。
- 输出：每个粒子每步的**完整**任务、机器人、工作站、资源公开状态；以及按时刻汇总的样本状态分布、机器人任务分布、站点服务概率、共享资源平均占用和等待/运输数量。
- 记录 V4 **这次真正前向推理得到的**服务事件概率及抽样结果；无额外二次调用，不使用模拟器私有剩余时间。
- 校验：每步必须由 `check_business_transition` 通过；未知未来新任务、设备故障和条件复检不被伪装成可预测事实。
- 预算：最多20步/8粒子；不使用GPU，不训练任何模型。

系统状态的机器人位置、工序合法性、站点资源容量并**非**通过 V4 学习获得；它们依赖显式规则，论文需如实表述。这仍符合“世界模型预测完整系统状态”的操作要求，但不是“端到端神经预测完整状态”。

## 代码

- `business_transport/world_model/contracts.py`：事件、单步、单粒子、分布的版本化公开接口；
- `business_transport/world_model/hybrid.py`：基于现有规则/冻结 V4 的合法状态滚动预测；
- `business_transport/session_dynamics.py`：暴露一次调用中的实际服务事件概率，预测结果不变；
- `experiments/world_model_integration/system_world_model/tests/test_rhwm_v1.py`：测试实时合规、H5滚动、多动作对照、确定性、异常配置与有真实检查点的 V4 推理；
- `experiments/world_model_integration/system_world_model/smoke.py`：同一个公开业务状态、两种不同合法首步调度动作的完整未来状态预测 JSON。

## 使用

```bash
python -m experiments.world_model_integration.system_world_model.smoke \
  --kind rule --horizon 5 --particles 3 --output /tmp/rhwm-rule.json

# 冻结 V4 检查点必须 SHA256 校验且显式标记实验用途
python -m experiments.world_model_integration.system_world_model.smoke \
  --kind v4 --checkpoint /path/to/frozen-v4.pt \
  --horizon 5 --particles 3 --output /tmp/rhwm-v4.json
```

此 smoke 是**合成样本上的预测演示**，输出不可误称为真实未来；它不衡量调度收益，也不直接充当独立测试集。

## 本次不做

1. 新神经图网络/重训练，留给 RHWM-v2 专门 PR；
2. 旧 PR #18 已知五个状态的微观调参；
3. 隐藏真值、真实医院轨迹、预知复检/故障/未来到达任务；
4. 修改正式 CP-SAT、运动规划器和 PR #19；
5. 从本接口成功运行便推断 V4 在业务调度上显著优于规则。

## 下一阶段门槛

RHWM-v1 的 CPU 合约与真实冻结 V4 推理在 GitHub Actions 上通过后，才能启动下一 PR 的**系统级 H1/H5/H10 预测误差测量**，再判断是否训练 RHWM-v2 关系网络。没有模型状态接口的验证，就不应先执行图网络训练。
