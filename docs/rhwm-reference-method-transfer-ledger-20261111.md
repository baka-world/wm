# RHWM 方法迁移台账：来源 → 源码 → 数据契约 → 验收

**纪律：先找已有方法及其维护中的公开实现，再作数据/接口适配。禁止把自行设计的算法、手搓分布拟合器或为优化指标改写仿真机制作成论文贡献。**

本记录基于 2026-11-11 核对的论文、上游 README、PyPI 发布记录、GitHub 许可证和现有代码。与 RHWM 相关的术语“世界模型”指**动作条件下预测完整业务系统状态**；学习的概率事件与遵守物理/业务合法性的显式状态转移，必须分开标记。

## 方法迁移目标及公开出处

| 原论文及方法 | 可复用现成实现 | 许可证、版本/环境 | 本项目迁移对象 | 不迁移的东西 | 当前状态 |
|---|---|---|---|---|---|
| Kaplan & Meier, *Nonparametric Estimation from Incomplete Observations*, JASA (1958), DOI [10.1080/01621459.1958.10501452](https://doi.org/10.1080/01621459.1958.10501452)；product-limit 生存估计 | [lifelines.KaplanMeierFitter](https://lifelines.readthedocs.io/en/latest/fitters/univariate/KaplanMeierFitter.html), [上游源代码](https://github.com/CamDavidsonPilon/lifelines/blob/master/lifelines/fitters/kaplan_meier_fitter.py) | **MIT；锁定 0.30.0**（支持项目 Python 3.10；0.30.3 已要求 Python ≥3.11） | 以服务开始/完成日志拟合各站点生存曲线 S(t)，从 `S(t+1)/S(t)` 形成**当前已服务 t 步后下一步完成的条件概率**，调用既有合法状态转移模块生成完整未来状态 | 不重写 KM 拟合器，不声称它是神经网络、GNN、原创世界模型，不反复调学习目标选成绩 | **已接入代码，独立 CPU 验证工作流待完成** |
| Reed, Löfstrand & Andrews, *Modelling stochastic behaviour in simulation digital twins through neural nets*, J Simulation (2021), DOI [10.1080/17477778.2021.1874844](https://doi.org/10.1080/17477778.2021.1874844) | 论文描述 ANN/MDN 条件随机事件分布耦合 DES，未确认论文官方可直接运行的代码包 | **论文方法引用，不宣称已获得官方代码** | “学习随机业务事件 → 显式转移系统状态”的概念依据；V4 + 规则及外部 KM 都遵循其范式 | 不复制论文网络、自创 MDN 或声称直接复现原实验 | 概念迁移，现有 RHWM-v1 已涵盖 |
| Kvamme et al. 相关离散时间生存预测模型、Logistic-Hazard | [pycox LogisticHazard](https://github.com/havakv/pycox/blob/master/pycox/models/logistic_hazard.py)，[pycox](https://github.com/havakv/pycox) | **BSD-2-Clause，PyPI 0.3.0**；依赖 Torch、torchtuples | 当 KM 作为严格外部基线建立后，使用上游 Logistic-Hazard 进行公开特征条件化的服务时长预测，对照同一独立数据的 V4/Rule/KM | 不自定义“新 hazard loss”；不在当前已用的 test group 上调参 | **候选，未安装或训练** |
| Tang & Wang, *Data-driven automated job shop scheduling optimization considering AGV obstacle avoidance*, Sci Reports (2025), DOI [10.1038/s41598-024-82870-1](https://doi.org/10.1038/s41598-024-82870-1) | 论文给出 Bayesian 网络/障碍风险建模；**未确认其官方可运行代码仓库** | 论文开放阅读；未经作者允许不直接复制图表或内容 | 将**运输延误/障碍风险**作为已有研究支持的预测目标，数据来自实际运输事件记录或明确假设的仿真事实 | 不复制粒子群调度器；已有 CP-SAT/MAPF 保持不变；不能把“随机阻塞”编造为已发生的企业事件 | 下一方法迁移候选，缺实际 AGV 障碍标注时不得训练或宣称验证 |
| Zhou et al., *A multi-robot–multi-task scheduling system for autonomous chemistry laboratories*, Digital Discovery (2025), DOI [10.1039/D4DD00313F](https://doi.org/10.1039/D4DD00313F) | [multi-bot-coordinator](https://github.com/pic-ai-robotic-chemistry/multi-bot-coordinator) | **MIT；仓库含调度和 MAPF 实现** | 实验室多机器人任务协同与已知业务约束背景，作为已有调度研究的现实参考 | 不把已有任务分配/路径规划算法宣称原创、不覆盖现有 CP-SAT/PB/LaCAM3 | 已确定参考，当前不额外引入其运行时 |
| Chua et al., *Deep RL in a Handful of Trials using Probabilistic Dynamics Models*, NeurIPS (2018), PETS | [mbrl-lib](https://github.com/facebookresearch/mbrl-lib) | **MIT；上游已在 2024 年归档**，本仓 `requirements-paper-reuse.lock` 已固定该依赖的 Git commit | 有限粒子多步不确定性传播；已有 RHWM 使用多粒子合法系统状态预测 | 不迁移连续 MuJoCo/MPC 的训练流程，不为改名引入新损失函数 | 已借鉴，仅用现有 RHWM 机制 |

## 已落实的第一个可执行方法迁移：上游 lifelines 生存估计

**训练标签来自真实执行过的公开业务事件**：`service_started` / `service_finished` 对齐 robot/sample/phase/station；使用 `event.tick` 计算历时（结束前还在运行的条目标记右删失）。训练数据不得读取隐藏 `service_remaining`、未来设备故障日程或测试组服务事件。

**上游计算**：`lifelines.KaplanMeierFitter().fit(durations, event_observed)`。程序没有自己实现 product-limit 估计。

**查询接口**：对已可观察的活跃服务，计算
```
p_finish_next = 1 - S(active_elapsed + 1) / S(active_elapsed)
```
其中 S(t) 由上游 `survival_function_at_times` 查询。若当前工作站没有任何训练事件或查询时间超过数据支持范围，退回明确的公开服务时间规则，不声称由 KM 可靠外推。

**系统推进**：把服务事件抽样送进已存在的 `ProgressAwareRule → WorkflowDynamics → check_business_transition`。这是必要的业务代码适配，不是新预测算法。原始 V4/Rule 权重和检查点保持不变。

**审计文件**：
- `requirements-rhwm-method-transfer.txt`：仅独立增加固定版本 lifelines，不更改主依赖锁；
- `business_transport/world_model/kaplan_meier.py`：外部 KM 模型及合法状态转移适配；
- `business_transport/world_model/hybrid.py`：将 KM 作为第三种明确来源的预测事件模型；
- `experiments/world_model_integration/method_transfer/km_reference_protocol.json`：先冻结 train/test/new seed、不借用 PR21 之前已见测试集；
- `experiments/world_model_integration/method_transfer/compare_km_v4.py`：真正由模拟器生成事实事件，按组拟合、按组验证；
- `experiments/world_model_integration/method_transfer/tests/test_km_reference.py`：验证实际使用上游类、右删失、事实状态完整性；
- `.github/workflows/rhwm-km-method-transfer-cpu.yml`：0 GPU / 0 新神经训练 CPU 验收。

## 怎样解读结果

- 即使 KM 的某个指标优于 V4，也只是证明此种**固定合成时长机制**下的现成统计基线很强，不等于实验室长期运行规律已学习到。
- 即使 KM 未赢 V4，项目也完成了合法的参考方法迁移和一条新的强基线；之后是否进入 `pycox` 需按业务时间可观察、删失率、特征与数据量判断。
- 禁止用已见测试组继续对 KM 按站点分组/阶段分组反复选最优策略；禁止把非神经 KM 写成“全神经世界模型”，也禁止因为文献介绍 AGV 障碍而新增未依据的障碍相关训练标签。
- **有限预算的根本含义**仍是对输入公开状态及候选动作预测完整业务任务/机器人/资源状态，少算力预测，有限预算下反馈调度；单步概率只是其一种内部组件。

**本文档是方法出处、许可、输入输出与工程边界的来源记录，不是原创方法声明。**
