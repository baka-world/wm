# 本科毕业设计：基于世界模型的样本转运调度系统设计与实现

本项目的主线是完成毕业论文及配套系统：在仿真环境中设计并实现实验室样本转运调度，构建能预测候选调度结果的世界模型模块，并通过运行与实验说明系统如何工作、效果如何。文献查新和算法对比为系统设计与论文论证提供依据，不是另起一个以追求算法创新为目标的项目。

仓库当前完成的是公开框架验收和资料整理；样本转运系统本身尚未实现。[毕设实施主线](docs/thesis-project-plan.md)列出了候选系统模块、当前状态和接下来的工作。

## 当前资料

- [交接背景和边界](docs/handoff.md)
- [毕设系统与论文实施主线](docs/thesis-project-plan.md)
- [框架验收与结果](docs/framework-assessment.md)
- [相邻论文和阶段零实验设计](docs/related-work-and-stage-zero.md)
- [RGTA 与静态预检工作流](.github/workflows/mapd-cloud-validation.yml)
- [Lay2Graph 多机器人闭环工作流](.github/workflows/lay2graph-integration.yml)及其[验收驱动](validation/lay2graph/run_smoke.py)
- `validation/logs/` 保存已核实的运行日志，较大的完整日志使用 gzip 压缩。

RGTA 小规模单测和分配实验已通过。Lay2Graph 在上游仓储场景中完成了两台机器人、6 个订单的真实仿真闭环；证据、限制和原始日志见框架验收记录。这些结果用于判断系统实现基础，并不代表毕业设计系统已经完成。

## 约束

使用 CPU 和公开仓库的标准 GitHub-hosted Linux runner；不启用 GPU 或付费云资源，不修改上游仓库。变更在功能分支和 Pull Request 中审阅，未经用户确认不合并。上游版本固定在工作流中，日志和报告应与对应提交及运行关联。
