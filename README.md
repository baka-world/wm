# 基于世界模型的样本转运调度系统：研究与验收

这是本科毕业设计的研究和阶段零验收仓库。目标是在实验室样本转运仿真中，检验动作条件预测能否在有限 CPU 预算下改善异常调度恢复；该假说与创新性仍待验证。

## 当前资料

- [交接背景和边界](docs/handoff.md)
- [框架验收与结果](docs/framework-assessment.md)
- [相邻论文和阶段零实验设计](docs/related-work-and-stage-zero.md)
- [RGTA 与静态预检工作流](.github/workflows/mapd-cloud-validation.yml)
- [Lay2Graph 多机器人闭环工作流](.github/workflows/lay2graph-integration.yml)及其[验收驱动](validation/lay2graph/run_smoke.py)
- `validation/logs/` 保存已核实的运行日志，较大的完整日志使用 gzip 压缩。

RGTA 小规模单测和分配实验已通过。Lay2Graph 的静态配置检查已通过；真实仿真必须以集成工作流的实际运行和订单完成状态为准，不能根据配置或服务启动推断通过。均不等于完成了实验室异常恢复研究。

## 约束

使用 CPU 和公开仓库的标准 GitHub-hosted Linux runner；不启用 GPU 或付费云资源，不修改上游仓库。变更在功能分支和 Pull Request 中审阅，未经用户确认不合并。上游版本固定在工作流中，日志和报告应与对应提交及运行关联。
