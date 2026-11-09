# 官方 PIBT 运动协调接入与医院失败案例复现

基于 PR #12 的 `d261caf0ad553d7d6090886a5368bcfe504bec44`，保留原运动后端、CP-SAT 调度、服务容量和三种既有世界模型。新增后端在不改动作者算法源码的情况下调用其 C++ PIBT；没有使用上游 MAPD 的任务分配，也没有训练模型。完整场景、原始逐步轨迹、上游运行日志和复现脚本见[证据目录](../examples/evidence/pibt-motion-20261109/README.md)，冻结实验设计见[协议](pibt-motion-protocol.md)。

## 论文、源码和构建

方法来自 Okumura 等 [Priority Inheritance with Backtracking for Iterative Multi-agent Path Finding](https://doi.org/10.1016/j.artint.2022.103752)。官方 [Kei18/pibt2 的 public 分支](https://github.com/Kei18/pibt2/tree/faab5b916649549f1cd563df8dbf6e4f6382f631)固定在 `faab5b916649549f1cd563df8dbf6e4f6382f631`；项目调用其 `pibt2/include/pibt.hpp`、`pibt2/src/pibt.cpp` 中的 `PIBT`、`PIBT::run` 和 `PIBT::funcPIBT`。原生 MAPF/MAPD 示例分别运行作者的 `mapf.cpp`、`mapd.cpp` 及 `pibt_mapd.cpp`，但 MAPD 只用于验证上游示例，不进入项目任务分配。

官方依赖 `grid-pathfinding` 固定提交 `f444df84459258d7b4d8ceffdbbf2e201b042a68`，Google Test 固定 `aefb45469ee7e6bde0cd1d2c18412046c30e7bb6`；`grid-pathfinding` 自身的 Google Test 子模块是 `609281088cfefc76f9d0ce82e1ff6c30cc3591e5`。前两项代码使用 MIT 许可证，Google Test 为 BSD 许可证，副本分别见 [PIBT](licenses/pibt2-MIT.txt)、[grid-pathfinding](licenses/grid-pathfinding-MIT.txt) 和 [Google Test](licenses/googletest-BSD-3-Clause.txt)。可视化所需 openFrameworks 未用于本次构建。

本地 C++ 编译器为 GCC 14.2、CMake 3.31.10。严格按上游 README 的 `cmake .. && make` 构建时，固定的 `grid-pathfinding` 因使用 `std::memset` 却没有显式包含 `<cstring>` 而失败；完整原始错误已保留。随后仅用编译选项 `-include cstring` 注入标准头文件，未修改上游源码：**22/22 原生测试通过**，MAPF 示例 100 台、MAPD 示例 50 台均输出 `solved=1`。该兼容处理和所有命令均可复现，不能把第一次失败说成原样构建成功。

## 两张医院图的原问题

找回了 PR #12 当时的临时脚本，并逐项复现其地图、三机器人、任务、释放时刻、服务和 800 步上限。旧版对未列出的站点隐式给容量 1；新 schema 要求显式列出相同容量，未改变业务数值。原后端的全部汇总字段与 PR #12 压缩日志一致：两张图均为 **4/9 阶段、2/5 完整任务**，运行至 800 步。

卡住时没有设备不可用或机器人运动暂停。Tartu 最后 R1 在 `(178,0)` 和 R3 在 `(170,0)` 等待被 R2 占据的取样点 `(208,0)`，而携带样本的 R2 也因其他机器人占位无法前往 `(204,0)`；clinic 最后 R2/R3 等待 R1 占据的取样点，R1 也无法完成前往目标的局部移动。原 `_move` 将其他机器人的当前格视为阻塞，三机器人时又不运行其只支持两台机器人与有向图的联合恢复。两图起点与站点之间静态可达。

进一步对**卡住的状态**做离线有限状态联合搜索：保持有向边、禁止同格和交换边，并保守地将到达自己任务目标的其他机器人固定。Tartu 的载样机器人存在一条 **33 步**到目标的合法联合移动见证，clinic 存在 **5 步**见证。因此，这两个冻结状态不能简单解释为“没有合法解围路径”；当前证据支持局部逐机器人避让和互相占用造成停滞，而不是设备故障。该离线搜索只用于诊断，不作为在线规划器，也不证明任意任务都可完成。

## 医院图：相同任务、不同运动后端

每行包含 5 个业务任务和 9 个阶段。流转时间为原始释放至最终阶段完成；**受限平均流转**对未完成任务按 800 步截断后计入，不能冒充这些任务最终的真实流转时间。路径耗时是本地 CPU 上每仿真步运动规划实际墙钟均值/P95，包含对应后端的适配和校验开销。

| 地图 / 后端 | 完整任务 | 阶段 | 仿真结束步 | 受限平均流转 | 最长连续无机器人移动 | 规划均值 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| Tartu / 原后端 | 2/5 | 4/9 | 800 | 527.2 步 | 640 步 | 0.390 / 1.118 ms |
| Tartu / PIBT | 5/5 | 9/9 | 175 | 113.0 步 | 3 步 | 0.324 / 1.229 ms |
| clinic / 原后端 | 2/5 | 4/9 | 800 | 501.8 步 | 722 步 | 0.336 / 0.762 ms |
| clinic / PIBT | 2/5 | 4/9 | 800 | 487.2 步 | 2 步 | 0.226 / 0.721 ms |

Tartu 的 PIBT 组 **175 次**调用了原生求解器，完成全部 5 个任务。clinic 的 PIBT 组调用 **798 次**，仍有 3 个任务未完成：最后一个阶段在第 37 步完成，此后机器人持续移动但未继续交付，最后 100 步只出现 3 种联合位置。最长无移动时间变短不意味着任务恢复；这是一次实际观察到的**活锁**。两个医院场景均无记录到的顶点碰撞、交换边、非法边、超容量、重复任务分配或非法生命周期。

## PR #12 场景与五机器人补充

完整重跑 PR #12 的 16 个独立确认场景（三台机器人）以及预先固定的 4 个五机器人场景。两后端均使用相同最近优先调度、任务、容量和外生扰动。两张公开地图是通用 MAPF 网格，站点用途、工作时间和新增机器人家点均为仿真假设。

| 机器人 / 后端 | 场景 | 完整任务 | 阶段 | 平均流转 | 最长连续无移动 | 规划均值 / P95 |
|---|---:|---:|---:|---:|---:|---:|
| 3 / 原后端 | 16 | 96/96 | 135/135 | 40.667 步 | 11 步 | 0.319 / 0.844 ms |
| 3 / PIBT | 16 | 96/96 | 135/135 | 43.906 步 | 14 步 | 1.266 / 4.242 ms |
| 5 / 原后端 | 4 | 24/24 | 30/30 | 31.125 步 | 3 步 | 0.417 / 1.415 ms |
| 5 / PIBT | 4 | 24/24 | 30/30 | 34.333 步 | 4 步 | 1.927 / 6.151 ms |

上述 20 个完整场景中没有发现顶点碰撞、交换边、非法移动、超容量或非法生命周期；PIBT 组分别实际调用作者的原生求解器 1,151 次和 243 次。20 个场景的 PIBT 平均流转均高于各自的原后端配对结果；其中同一地图/种子的场景相关，不能把它们当作 20 个独立地图样本。旧后端记录的被阻拦移动尝试与 PIBT 内部回溯次数口径不同；后者上游接口未暴露，原始数据中标记为 `null`，不是 0。

## 适用条件与预测口径

作者论文和[作者网页](https://kei18.github.io/pibt2/)给出的活性条件要求每对相邻顶点属于一个简单环。静态 `empty8` 是双向连通图、无桥边；`random32` 有 20 条桥边，clinic 有 25 条桥边。Tartu 有 50 条单向边；对其忽略方向的**离线统计投影**有 21 条桥边，实际执行始终保留原有方向。服务和暂停锁定会进一步改变可通行图。除静态 `empty8` 的无桥性质外，这些地图/动态占位均不应被宣称满足原论文的活性条件。

官方实现没有公开可持续更新的单步接口，适配器每步构造一次实例、调用官方 PIBT 并取第一步；因此上游内部的优先级年龄每步重置。这种滚动调用仍执行官方回溯代码，但**不等同于论文中保留年龄的连续执行，也不能继承其活性保证**；clinic 的活锁证实此局限值得进一步处理，尚不能断言重置是唯一原因。若目标位置被服务中的机器人锁住，或动态剩余图无法到达目标，本步临时保留当前位置并记录，绝不穿过锁定位置或添加反向道路。

现有规则、PETS、结构化服务头分别与相同 PIBT 运动适配器组合。在一个含当前运动暂停的合成小场景中，三者均在 **29/29 个事实单步**输出与实际执行相同的机器人联合位置，暂停状态出现 4 次；三者分别与 CP-SAT 的五步 Rollout 跑通 **4/4 阶段**，各自进行了 120 次预测侧原生调用。PETS 和结构化头分别有 93 次网络前向与 87 次服务头调用。结果证明运动部分复用同一规则，不证明服务预测或多步世界状态必然准确；未来任务、故障和暂停变化仍不可预知。

## 复现、验证与后续限制

需要 Python 3.10、[PR #11 固定 CPU 依赖](../requirements-paper-reuse.lock)、C++17、CMake ≥3.16、Git 和网络访问固定公共源码。以下路径均必须为空或不存在，脚本不会覆盖已有结果：

```bash
uv venv --python 3.10 .venv-paper
uv pip install --python .venv-paper/bin/python --index-strategy unsafe-best-match -r requirements-paper-reuse.lock
source .venv-paper/bin/activate
bash tools/pibt_motion/prepare_upstream.sh /tmp/wm-pibt-upstream-new
export WM_PIBT_LIBRARY=/tmp/wm-pibt-upstream-new/bridge/libwm_pibt_bridge.so
python -m unittest discover -s tests -v
python -m unittest discover -s experiments/service_value/tests -v
python -m unittest discover -s experiments/paper_reuse/tests -v
python -m unittest discover -s experiments/business_constraints/tests -v
python -m unittest discover -s experiments/pibt_motion/tests -v
python -m tools.pibt_motion.prepare_sources --output /tmp/wm-pibt-hospital-sources-new
bash tools/pibt_motion/run_acceptance.sh "$WM_PIBT_LIBRARY" /tmp/wm-pibt-hospital-sources-new /tmp/wm-pibt-acceptance-new
```

验收脚本独立运行两次医院及网格对照、比较除墙钟耗时外全部结果，再执行离线见证与安全审计。本地原有和新增 Python 测试分别为 **112、11、11、14、7 项**。CPU 工作流见 [pibt-motion.yml](../.github/workflows/pibt-motion.yml)。

真实部署仍需要：真实实验室布局及单向规则和设备容量；定位、控制与连续空间避障；机器人和工作站状态实时接口；对设备停机/网络失联和异常恢复的验证；以及针对动态锁定、桥边和三台以上机器人的长期活性保障。这个仿真结果不是临床实测，也不能证明所有医院路网可通行。
