# 本科毕业设计：基于世界模型的样本转运调度系统设计与实现

本项目的主线是完成毕业论文及配套系统：在仿真环境中设计并实现实验室样本转运调度，构建能预测候选调度结果的世界模型模块，并通过运行与实验说明系统如何工作、效果如何。文献查新和算法对比为系统设计与论文论证提供依据，不是另起一个以追求算法创新为目标的项目。

仓库已有一个基于合成实验室场景的 CPU 原型，可运行样本任务、机器人调度、轻量预测和离线回放；完整毕业设计系统仍需结合实际需求扩展。[原型说明](docs/prototype.md)包含运行方式和限制，[毕设实施主线](docs/thesis-project-plan.md)记录后续工作。

准备在本地拉取、运行或开发模型，请从[本地开发与交接指南](docs/local-development.md)开始：包含 Windows / Linux 命令、数据与训练流程、接口说明、GPU 接入条件和常见问题。

## 当前资料

- [交接背景和边界](docs/handoff.md)
- [毕设系统与论文实施主线](docs/thesis-project-plan.md)
- [样本转运原型与运行说明](docs/prototype.md)
- [有论文与公开导航图来源的医院转运场景](docs/source-backed-hospital-scenario.md)
- [样本规则与公开数据复用说明](docs/specimen-rules-and-public-data.md)
- [有来源的时间估计与规则辅助调度](docs/timing-and-rule-aware-dispatch.md)
- [学习型预测器与仿真对照实验](docs/world-model-experiment.md)
- [多取送点与未见地图的对照实验](docs/multiple-locations-experiment.md)
- [多步世界模型与毕设准备情况检查](docs/world-model-readiness.md)
- [CPU 状态转移模型和留出场景实验](docs/state-transition-experiment.md)
- [状态转移模型第二轮验证方案](docs/state-transition-revision.md)
- [预测对样本转运决策的价值：CPU 反事实验证](docs/decision-value-experiment.md)
- [第二轮：故障覆盖、候选差距及评分对齐](docs/decision-value-round2-report.md)
- [结构化世界模型、状态边界与本地训练协议](docs/structured-world-model.md)
- [新版评分与最小模型 CPU 验收结果](docs/structured-world-model-results.md)
- [PR #8 GPU 基线冻结说明](docs/pr8-baseline-provenance.md)
- [服务动力学故障覆盖与 CPU 消融结果](docs/service-dynamics-results.md)
- [服务预测的调度价值：方案与实验结果](docs/service-decision-results.md)
- [指定源码复用：CP-SAT 与 PETS CPU 闭环集成](docs/paper_reuse_integration.md)
- [多阶段转运、工作站容量与五组调度对照](docs/business-constraints-results.md)
- [本地 GPU 训练接入准备和分工](docs/local-gpu-readiness.md)
- [本地开发与交接指南](docs/local-development.md)
- [可替换的 PyTorch 训练、评估与仿真入口](docs/pytorch-training.md)
- [CPU 神经模型流程验收记录](docs/pytorch-smoke-acceptance.md)
- [系统设计论文材料草稿](docs/thesis-system-design-draft.md)
- [示例仿真结果和离线回放](examples/evidence/)
- [框架验收与结果](docs/framework-assessment.md)
- [相邻论文和阶段零实验设计](docs/related-work-and-stage-zero.md)
- [RGTA 与静态预检工作流](.github/workflows/mapd-cloud-validation.yml)
- [Lay2Graph 多机器人闭环工作流](.github/workflows/lay2graph-integration.yml)及其[验收驱动](validation/lay2graph/run_smoke.py)
- `validation/logs/` 保存已核实的运行日志，较大的完整日志使用 gzip 压缩。

RGTA 小规模单测和分配实验已通过。Lay2Graph 在上游仓储场景中完成了两台机器人、6 个订单的真实仿真闭环；证据、限制和原始日志见框架验收记录。这些结果用于判断系统实现基础，并不代表毕业设计系统已经完成。

运行原型（Python 3.11+，不需要第三方 Python 包）：

```bash
python -m unittest discover -s tests -v
python -m lab_transport compare --seed 42 --output runs/demo-42
```

运行后打开 `runs/demo-42/world/report.html` 查看示例回放。示例使用合成数据，原型目前不连接 Lay2Graph 或真实机器人。

未安装 PyTorch 时，神经训练相关测试会跳过；完整测试及模型训练环境请参考[本地开发指南](docs/local-development.md)。

另可按[场景来源说明](docs/source-backed-hospital-scenario.md)运行基于公开医院 ICU—实验室路网的适配。路网来自医院项目，双机器人、到达时刻和执行步长是明确标注的功能测试设置；不填没有依据的临床时限或温度。

## 约束

当前开发和 GitHub Actions 使用 CPU 及标准 GitHub-hosted Linux runner；不启用云端 GPU 或付费云资源，不修改上游仓库。后续在用户授权接入的本地 GPU 上验收模型训练。变更在功能分支和 Pull Request 中审阅，未经用户确认不合并。上游版本固定在工作流中，日志和报告应与对应提交及运行关联。
