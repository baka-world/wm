# 公开 MAPD 框架验收记录

截至 2026-11-06；本项目不修改上游仓库，不使用 GPU。CPU workflow 使用公开仓库的 GitHub 标准托管 Linux runner；[GitHub 文档](https://docs.github.com/en/actions/concepts/billing-and-usage)说明此类 runner 在公开仓库中免费。此记录区分运行事实与计划。

## RGTA：小规模分配预检通过

- 上游：<https://github.com/YiBangWon/rgta-mtmapd>，固定提交 `bd1ad9710eb5672942767b04efff223eb784e24c`。上游 [README](https://github.com/YiBangWon/rgta-mtmapd/blob/bd1ad9710eb5672942767b04efff223eb784e24c/README.md) 说明 Python 标准库即可运行，事件式模拟使用路线成本近似，另有 PIBT 路径验证脚本；本次没有运行后者。
- 原始 [运行 #1](https://github.com/lieuyimo/wm/actions/runs/42199855975) 的 RGTA 测试发现失败，日志为 `ImportError: Start directory is not importable`。仓库主分支移除了不适用的 `-t .`，后续 [运行 #2](https://github.com/lieuyimo/wm/actions/runs/42200322139) 成功。
- 功能分支 [运行 #3](https://github.com/lieuyimo/wm/actions/runs/42207296079) 对固定提交再次通过：3 个单元测试，Kiva `6x6`、两种方法、两个种子共 4 组，每组完成 60 个任务。工作流额外要求至少执行 1 个测试，避免零测试时误报通过。原始失败和成功日志位于 `validation/logs/`。
- 这只证明相应测试和小实验在指定版本上完成；并未证明碰撞安全、实验室工作站容量处理、异常恢复、论文主要规模结果或世界模型效果。

## Lay2Graph：静态预检通过，真实闭环以集成运行结果为准

- 上游：<https://github.com/juknierim/Lay2Graph-MAPD-Framework>，固定提交 `f04d860374ab636ddd6199c5643e3b78b5f5505b`，MIT 许可。其 [README](https://github.com/juknierim/Lay2Graph-MAPD-Framework/blob/f04d860374ab636ddd6199c5643e3b78b5f5505b/README.md) 描述任务分配、路径规划、订单管理、机器人通信、状态管理、事件式机器人仿真与场景评估服务。
- [预检 #3](https://github.com/lieuyimo/wm/actions/runs/42207296079) 验证 Compose 配置与所需服务名；未启动服务，不是功能仿真。
- `validation/lay2graph/run_smoke.py` 和 `.github/workflows/lay2graph-integration.yml` 是本仓库新增的验收驱动：在临时目录从上游现成仓库场景截取 2 台机器人及 6 个订单，启动上游 CPU 服务，加载地图与机器人状态，启用碰撞检查，并执行有限步数/时间的仿真。脚本要求全部订单完成、订单管理器报告 `FINISHED`，且至少 2 台机器人承担已完成订单；它保存步骤、订单结果、服务日志和 fixture。
- 上述脚本的存在不等于验收通过。即使该仓储场景通过，也仅说明小规模上游功能闭环，不是针对实验室任务、独立碰撞验证或异常恢复的完整验收。
- 第一次集成 [运行 #1](https://github.com/lieuyimo/wm/actions/runs/42207840908) 在进入 runner 前失败：工作流 job 级环境变量错误引用 `runner.temp`；GitHub 注释指出 `Unrecognized named-value: 'runner'`。后续已改用 `github.workspace` 并重新运行。
- [运行 #2](https://github.com/lieuyimo/wm/actions/runs/42207941051) 成功构建服务，但 AMR 通信服务的 3003 端口未就绪。上游 AMR 通信及仿真服务的 FastAPI 生命周期即使关闭 MQTT 传输也尝试连接 broker；验收工作流最初未启动该服务。对应完整 job 日志压缩保存于 `validation/logs/`。
- [运行 #3](https://github.com/lieuyimo/wm/actions/runs/42208615769) 补上 broker 后 7 个服务端口均就绪，验收进入初始化；路径规划接口对缺失的 focal heuristic 字段返回 422。验收驱动已改为使用上游示例的非空配置；修正后的 [运行 #4](https://github.com/lieuyimo/wm/actions/runs/42209255973) 正在执行，不能预先宣称通过。

## 适用性和下一步

RGTA 更适合快速考察任务分组和分配策略；其事件式代价代理不能替代逐步路径执行和碰撞验证。Lay2Graph 的服务和仿真链条更接近真实系统，但构建及集成成本更高，其现有仓储案例并不自动包含实验室工作站资源约束或所需异常模型。

两者都应作为候选平台或基线，而不是项目阶段零完成的证据。阶段零仍需定义样本、站点容量、截止时间、异常和恢复判据；再在同一场景输入、随机扰动与 CPU 预算下比较传统方案。详见 `docs/related-work-and-stage-zero.md`。
