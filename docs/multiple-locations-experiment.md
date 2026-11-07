# 多取送点与未见地图对照实验

## 运行前设定

上一轮六项任务都使用同一个取点和送点，学习预测与原预测的分配序列在 30/30 个场景相同。为了检查候选地点变化后是否仍如此，本轮在运行前固定以下方案；本实验用于系统验证，不预设学习策略必须胜出。

| 地图 | 来源和任务映射 | 使用边界 |
| --- | --- | --- |
| 塔尔图医院 L2 | 继续使用 [医院公开地图](https://github.com/scafld/covsg24_fleet_backend/blob/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml) 中 `icu1_pickup_location` 与 `lab_dropoff_location`；把名为 `icu_2` 的路点额外**假设为第二取样点**，目的地仍为实验室交付点 | 公开地图显示 `icu_2` 路点，但未见证据表明该点配置了取样交接设备；第二取样点及任务流是本实验假设，不是医院实测流程 |
| Open-RMF 诊所 L1 | 固定 [Open-RMF clinic 演示地图](https://github.com/open-rmf/rmf_demos/blob/672a67a1643ac12d89fc21565fa92f33f5c2e0fc/rmf_demos_maps/maps/clinic/clinic.building.yaml)的 graph 0，保留交通方向、比例尺及两台配送机器人的原始停靠点。将左右诊疗室映射为两个模拟取样点、左右护理站映射为两个模拟交接点 | Open-RMF 将其称为诊所演示地图，不能当成真实医院平面；原数据并未声称这些点是标本采集或检验科接收点。这里只检验未参与训练的另一张公开仿真地图，不模拟电梯或另一楼层 |

Open-RMF 示例仓库以 [Apache-2.0](https://github.com/open-rmf/rmf_demos/blob/672a67a1643ac12d89fc21565fa92f33f5c2e0fc/LICENSE) 发布；本项目运行时读取固定版本并校验 SHA-256，不修改上游。诊所地图的比例尺为其 L1 配置中的 0.7 米参考线，速度取 [clinic 配送机器人配置](https://github.com/open-rmf/rmf_demos/blob/672a67a1643ac12d89fc21565fa92f33f5c2e0fc/rmf_demos/config/clinic/deliveryRobot_config.yaml)中的 0.7 米/秒，均为仿真配置而非实测。两张地图都以 10 秒离散步估计路径时间；装载和交接的跨医院参考及假设沿用[时间模型说明](timing-and-rule-aware-dispatch.md)。

固定诊所地图 SHA-256：`780bfabff950c4fb3132dd1933994255afe3b93cbc25a8265e0214adb62ff7f0`；配送机器人配置 SHA-256：`e63fd0f2e74eac14c4651ad6ee04d20d9e97b8a0a13cc415791ce013b6d9c316`。原始 graph 0 为 54 个相关点、110 条有向边；按标称速度离散后为 62 个点、126 条有向边。诊疗室、护理站和两个机器人出生点名称可直接在固定源文件中核对；用途映射仍是本实验假设。

### 固定实验划分

- 不重新训练或调整上一轮的学习模型；使用仓库中在塔尔图单取送点合成场景上训练的[固定参数](../examples/evidence/world-model/learned-parameters.json)。诊所地图和两个取点的组合对该模型均未参与训练。
- 塔尔图使用种子 1000–1004，诊所使用种子 2000–2004；每张地图分别运行无中断与短暂站点中断两种条件，共 20 个场景。两种条件复用种子，以便配对。
- 每场有两台机器人和 6 个人工任务，两个取点各承担 3 个任务，顺序由种子确定，释放间隔取 4、8、12、16 个仿真步。塔尔图任务均送实验室；诊所示例的两个模拟护理交接点各承担 3 个目的地，分配顺序由种子确定。没有引入无依据的临床时限。
- 短暂中断从场景涉及的站点中按种子选取，起始步取 10–55、持续 6–14 步；均为仿真扰动而非故障统计。每场最多 350 步。

### 对照与预算

比较最近取样点 `nearest`、原统计预测 `world`、固定学习预测 `learned`，以及新增的传统参考 `rollout`。后者对当前可执行候选做有限步前视，依据**当前**任务、机器人位置、站点状态和已完成服务的均值，估计其它机器人与候选任务的执行；每候选最多 120 个仿真步，不读取未来任务、未来中断或实际剩余服务时间。预测中假设当前中断再持续一个步长、后续服务时间为已观测均值向上取整；其他已知等待任务用 `world` 分配。预测期内未完成的候选将截止时间作为排序下界，不声称这是完成时间或不可行证明，也不构成对截止时间后的优化保证。`rollout` 是实现范围明确的传统参考，不是 CBS/LaCAM 之类具有相应保证的强最优规划器。

各策略使用同一组任务、站点中断和按样本键固定的服务时间抽样，路线和离散冲突检查相同。`rollout` 的每候选计算量高于单次预测，因此单独报告各策略 CPU 时间和候选评估次数；本轮**没有**实施所有方法相同的墙钟计算上限，不能据此声称公平地比较了相同计算预算。

记录完成率、超时场次、截断平均任务耗时、每场 CPU 时间和候选评估次数；在 `world` 的同一批已执行决策上比较预测误差，仅对已完成任务计算并报告缺失数。只有两张地图、每组少量种子，以及人工站点用途和扰动，均不支持泛化或显著性结论；如 `learned`、`rollout` 表现更差或相同，也照实保留。

## 本轮结果（2026-11-07）

[完整逐场任务、调度和指标](../examples/evidence/multisite/experiment-results.json)来自本地 CPU 运行；[独立复跑日志](../validation/logs/multisite-cpu-2026-11-07.log)保存逐场输出，自动工作流另行重新下载并校验地图、复算结果。两个地图各 10 场，四个策略分别完成 120/120 个任务，均无场次超出仿真期限；记录的离散状态中未发现同点或对向换边，**这不证明连续空间或真实机器人安全**。所有候选在 120 步前视内完成，没有触发前视截止。

平均任务耗时为从释放到交付的仿真步，1 步名义上为 10 秒。未完成任务按第 350 步截断；本轮没有未完成任务。

| 地图 / 条件 | 最近取样点 | 原预测 `world` | 学习预测 `learned` | 有限前视 `rollout` |
| --- | ---: | ---: | ---: | ---: |
| 塔尔图 / 正常 | 119.633 | 120.700 | 119.633 | 119.433 |
| 塔尔图 / 短中断 | 120.933 | 122.133 | 120.933 | 121.367 |
| 诊所 / 正常 | 33.800 | 33.800 | 33.800 | 36.100 |
| 诊所 / 短中断 | 34.400 | 34.400 | 34.400 | 36.767 |

与原预测相比，学习策略在塔尔图每种条件下有 2/5 场出现不同的完整分配序列；诊所每种条件均为 0/5。对同一批 `world` 已执行、已完成任务，交付时间平均绝对预测误差如下，括号内为误差样本数，均无删失：

| 地图 / 条件 | `world`（步） | `learned`（步） |
| --- | ---: | ---: |
| 塔尔图 / 正常（30） | 10.967 | 7.834 |
| 塔尔图 / 短中断（30） | 11.103 | 8.298 |
| 诊所 / 正常（30） | 4.833 | 6.006 |
| 诊所 / 短中断（30） | 5.433 | 6.554 |

这一组小样本中，固定学习模型对训练地图的多取点测试有所帮助，但在未见诊所地图的同决策预测上误差更大，闭环分配没有变化。前视对照在两张地图上也并非始终最好，且本地每场平均 CPU 时间约为 `world` 的 3.1–4.7 倍；时间受机器环境影响，并未实施相同计算预算。不能据此宣称模型具有稳定优势或跨地图泛化能力；也不能把诊所护理交接当作检验科实际接收。

### 复现

使用 Python 3.11+ 和 CPU，分别从上述固定提交下载两张地图及对应机器人配置，原始文件只在本地或 CI 临时目录使用。将路径替换为本地文件：

```bash
python -m lab_transport.multisite_experiment \
  --hospital-map /path/to/hospital.building.yaml --tiago-config /path/to/tiago.launch.xml \
  --clinic-map /path/to/clinic.building.yaml --clinic-config /path/to/deliveryRobot_config.yaml \
  --model examples/evidence/world-model/learned-parameters.json --output /tmp/multisite-result
python tests/check_multisite_experiment.py /tmp/multisite-result examples/evidence/multisite
```

检查器核对配对任务、站点分布、训练与测试种子、扰动范围、完成情况和离散安全记录；并将结果与仓库记录对比，CPU 时间除外。CI 具体下载地址与执行命令见 [sample-transport 工作流](../.github/workflows/sample-transport.yml)。[GitHub Actions 运行 42323467519](https://github.com/lieuyimo/wm/actions/runs/42323467519)验证了提交 `920f574ea6d374071a346666b3dda4a0ddde4700` 的 40 项单元测试、20 场复算和无 GPU 的诊所回放浏览器检查；[完整运行日志](../validation/logs/multisite-action-42323467519.log)已归档。

也可生成可播放的双机器人路线和任务状态页面（诊所地图、默认种子 2000）：

```bash
python -m lab_transport.clinic_demo --clinic-map /path/to/clinic.building.yaml \
  --clinic-config /path/to/deliveryRobot_config.yaml --policy learned \
  --model examples/evidence/world-model/learned-parameters.json --output /tmp/clinic-demo
```

打开 `/tmp/clinic-demo/report.html`，拖动进度条或点击播放。它还输出逐事件 JSONL、任务 CSV 和完整快照；其中站点用途和任务均为仿真设置。也可使用 `--policy world` 或 `--policy rollout`。CI 对三种回放运行脚本检查，并在无 GPU 的无头浏览器中检查学习策略页面。
