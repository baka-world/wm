# RHWM-v2 参考方法迁移：pycox Logistic-Hazard（2026-11-12）

## 一、事实结果与阶段决定

已经完成 **pycox 官方 LogisticHazard 的真实 CPU 训练、验证、独立合成测试**，不只是规划或接口预留。

**判定：方法迁移完成；暂不晋升为 RHWM 默认概率动力学组件。** 在 24 个常规参数、新 seed 场景的服务事件 Brier 上好于 V4，但 H10 完整系统状态误差更高；在 12 个参数变化场景中，服务事件与 H1/H5/H10 完整状态预测均明显不如 V4。不得把事件级 Brier 的局部改善表述为完整世界模型性能普遍提高。

本 PR 从仍为 Draft 的 #22 HEAD `5e54ca8aad0584b552dd430289a4131a0e9f2242` 建立。**不修改 PR #19、PR #21、PR #22，主分支不变。**

## 二、出处、实现和许可证

- 理论论文：Kvamme & Borgan (2019), *Continuous and Discrete-Time Survival Prediction with Neural Networks*, [arXiv:1910.06724](https://arxiv.org/abs/1910.06724)。
- 官方实现：[pycox.models.LogisticHazard](https://github.com/havakv/pycox/blob/master/pycox/models/logistic_hazard.py)，BSD-2-Clause，固定 `pycox==0.3.0`。
- 官方训练与网络：[torchtuples](https://github.com/havakv/torchtuples)，BSD-2-Clause，`torchtuples==0.2.2`，直接使用 `practical.MLPVanilla`、`optim.Adam`、`callbacks.BestWeights`、`callbacks.EarlyStopping`。
- 上游 `LogisticHazard.label_transform` 负责离散时间/删失标签，`NLLLogistiHazardLoss` 负责损失，`predict_hazard` 负责事件条件概率。没有新设计生存函数、危险率损失、图编码器或自定义网络训练循环。
- 固定输入：工作站 ID one-hot、取样/送达阶段、**服务开始时的公开预计工时**、**资源容量**；推理时仅额外使用已观察的有效服务历时。时间刻度 `[0..max(train duration)+2]` 仅由训练集确定。超过训练支持范围明确回退已知规则。
- `business_transport/world_model/kaplan_meier.py` 原共用合法转移部分提炼成 `SurvivalEventRuleDynamics`，保留原 `KaplanMeierRuleDynamics` 别名，V4/Rule 行为不改。
- 独立依赖文件 `requirements-rhwm-method-transfer.txt`，不全局升级历史 CPU/PyTorch 锁；实际运行 Python3.10、Torch2.5.1+cpu。

## 三、冻结实验协议

`experiments/world_model_integration/method_transfer/pycox_protocol.json` 在训练前提交，规定：

| 数据 | 事先分组数 | 用途 |
|---|---:|---|
| Train | 48 | 拟合 pycox 的时间分箱、标准网络权重；同时拟合公平 KM 基线 |
| Validation | 12 | 仅用于上游 BestWeights / EarlyStopping 的验证损失 |
| Test-IID | 24 | 完全新 seed 的合成参数常规场景 |
| Test-shift | 12 | 完全新 seed 的合成服务时间/抖动参数变化 |

- 96 组 seed 不重复，也不与上一轮 KM 及 RHWM-v1 H1/H5/H10 场景重合。
- 训练实际服务 480 条，验证 120 条。用事实日志 `service_started` / `service_finished` 对齐事件历时；未完成则标记右删失；训练特征只来自事件开始时 `BusinessObserver` 的可观测公开状态，绝不读取隐藏剩余服务倒计时或真实未来派单。
- 标准 MLP 两层 32/32，Adam 0.01，batch size 128，最大 40 epoch，最优参数选自 validation loss；实际训练 40 epochs。
- 独立组比较 Rule、重新拟合 KM、冻结 V4、pycox：同一事实轨迹，公开最近批量调度策略，固定首步候选，未来按各自预测状态续调度，三粒子，H1/H5/H10。按场景/seed 等权汇总，不能把粒子、事件或 anchor 当独立重复。
- 无 GPU、无真实企业日志、无自定义动态训练损失、无正式 CP-SAT/PIBT/LaCAM3 改动。预先规定的 CPU 时间预算 480 秒，实际统计主进程 CPU 27.48 秒（不含依赖安装和 Actions 启动）。

## 四、真实独立结果（越低越好）

### 24 个 Test-IID 场景组

| 目标 | Rule | Kaplan–Meier | 冻结 V4 | pycox |
|---|---:|---:|---:|---:|
| 服务事件 Brier | 不统计 | 0.20522 | 0.18498 | **0.16517** |
| H1 样本状态不一致率 | **0.03177** | 0.05540 | 0.03555 | 0.04435 |
| H5 样本状态不一致率 | **0.07905** | 0.23092 | 0.16925 | 0.18690 |
| H10 样本状态不一致率 | **0.17274** | 0.27178 | 0.19448 | 0.24360 |
| H10 站点忙闲 Brier | 0.13264 | 0.12693 | **0.11242** | 0.11991 |
| H10 资源占用 MAE | **0.21181** | 0.26350 | 0.22569 | 0.24074 |

### 12 个 Test-shift 参数变化场景组

| 目标 | Rule | Kaplan–Meier | 冻结 V4 | pycox |
|---|---:|---:|---:|---:|
| 服务事件 Brier | 不统计 | 0.39009 | **0.17007** | 0.25913 |
| H1 样本状态不一致率 | 0.17042 | 0.23685 | **0.12595** | 0.21023 |
| H5 样本状态不一致率 | **0.08903** | 0.31590 | 0.13634 | 0.19345 |
| H10 样本状态不一致率 | 0.22581 | 0.44660 | **0.18681** | 0.31974 |
| H10 站点忙闲 Brier | 0.17917 | 0.21080 | **0.14426** | 0.20929 |
| H10 资源占用 MAE | **0.24306** | 0.40556 | 0.24691 | 0.36096 |

**重要解释：** 此次独立测试的 IID 组也并非各指标上 V4 始终优于规则模型。原 PR #21 上 V4 的优势只是当时冻结的六个合成场景的观察，不应替代更大测试的新事实。测试结果反映当前公开状态预测模型对场景种子和时长机制具有敏感性，不能据此声称真实业务稳定提升或方法普适泛化。

## 五、可复现证据、代码和动作

- [GitHub Actions #43127431601](https://github.com/lieuyimo/wm/actions/runs/43127431601) **success**
- [原始 JSON、训练出来的 `.pt` 权重、CPU 单测与独立评测日志、SHA256 artifact #13903880552](https://github.com/lieuyimo/wm/actions/runs/43127431601/artifacts/13903880552)；保存 90 天
- 9 项方法迁移/兼容测试 + 9 项历史 RHWM 回归 = **18 项测试通过**；V4 冻结 checkpoint SHA256 `60c8a6c3d832da3e6519072688ea099f64ae120ba19d3bd2ec705f7f1a11ac0c` 校验成功。
- `business_transport/world_model/pycox_survival.py`：官方训练 API 与事件风险查询适配。
- `experiments/world_model_integration/method_transfer/{pycox_protocol.json,pycox_run.py,tests/test_pycox_transfer.py}`：冻结数据协议、事实事件提取、训练和全状态指标。
- `.github/workflows/rhwm-pycox-logistic-hazard-cpu.yml`：实际训练并归档。

## 六、下一阶段取舍

**不为获得好成绩重新调整已看过的 IID/shift 测试集，不默认追加 GNN、不创造新的 hazard loss。**

服务时间建模已存在三个可复用概率方案：V4、KM、Logistic-Hazard；后两者完成方法迁移但无充分证据取代 V4。下一项优先研究**运输动态/到站时间**，它是世界模型当前尚未学习的一类动力学：

1. 先查 Tang & Wang (2025) AGV 运输延误贝叶斯网络模型及其引用的公开实现，复核是否确有可以复用的源代码、许可证和所需路径/障碍时间标注。
2. 只在既有事实运输事件中能找到与路径、拥堵、机器人相对位置相关的延迟变量时，迁移 `pgmpy` 等已有概率图方法并进行同预算评测；不能把随机障碍伪装成企业现场事实。
3. 仍以完整系统状态 H1/H5/H10、动作条件、真实合法调度指标和 CPU 成本为总验收，不单凭一个事件指标晋升默认模型。

**方法出处与工程贡献必须分别记录，毕业论文不能把开源 Logistic-Hazard 或 KM 算法当原创。**
