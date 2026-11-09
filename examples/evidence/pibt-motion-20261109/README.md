# PIBT 运动后端：实验与上游运行证据

结果、假设和约束详见 [报告](../../../docs/pibt-motion-results.md) 和[冻结协议](../../../docs/pibt-motion-protocol.md)。该批数据是仿真结果；设备、容量、任务和新增机器人位置不是医院实测参数。

* `pr12-original-pilot.py.txt` 是找回的 PR #12 原临时试验脚本，SHA-256 为 `3ce74d2b31028c068de6f49253acc5a82838cb52cf2a7a40ae59d8c5bca30b24`。其中旧导入路径和未显式声明的单站点容量反映当时的代码；当前兼容复现见 `hospital_reproduction.py`。
* `hospital-legacy.json.gz`、`hospital-pibt.json.gz` 含原有相同初始配置、公开源哈希、有向边、工作站、完整逐步机器人位置与目标、任务阶段、服务和等待、实际事件及安全审计。真实内部倒计时只存在于离线诊断历史，未输入调度器。
* `grid-benchmark.json.gz` 含 16 个 PR #12 旧场景及 4 个五机器人场景，每组两种运动后端的全部任务、路径、耗时、完成与违规结果；`model-alignment.json` 记录三种模型的共享运动和小规模 CP-SAT 闭环验收。
* `joint-diagnostic.json` 给出从两张医院图卡住状态出发的有向联合移动见证；它是离线检验，不是线上规划器。`topology-audit.json` 的无向投影只用来计桥，执行不改变道路方向。
* `logs/upstream-readme-build.log.gz` 保存未修改上游源码时的首次构建失败；`logs/upstream-compatible-build.log.gz` 和 `logs/upstream-tests.log.gz` 保存仅加标准头编译参数后的构建及原生测试。`upstream-mapf-result.txt.gz`、`upstream-mapd-result.txt.gz` 和示例日志是作者原生示例输出。
* `logs/upstream-first-wrong-cwd-test.log.gz` 保存首次自动化脚本把原生测试放在错误工作目录、找不到上游相对路径测试数据的失败。现已按上游要求在源码下的构建目录运行；无需修改上游代码。
* `logs/final-acceptance.log.gz` 包含两次冻结实验、逐项复现核对和诊断；`logs/regression-*.log.gz` 记录历史及新测试。
* `provenance.json` 固定上游和子模块提交、配置、已有模型和相关源码哈希；`SHA256SUMS` 覆盖此目录其他文件。

公开源图和设备配置由复现脚本从已固定的 commit 下载并校验 SHA-256；不修改上游仓库。切换到本目录后运行 `sha256sum -c SHA256SUMS` 可验证文件；解压使用 `gzip -dc <文件>`。
