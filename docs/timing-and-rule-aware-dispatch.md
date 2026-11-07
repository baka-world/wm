# 时间估计与规则辅助调度

## 时间参数的来源

| 参数 | 仿真中的用法 | 依据与限制 |
| --- | --- | --- |
| 路网比例 | 原图一段 67.2 米的参考线除以对应像素距离，得到约 0.032140407 米/像素 | [作者固定版本的医院地图](https://github.com/scafld/covsg24_fleet_backend/blob/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml)含 `measurements`；[RMF Traffic Editor 说明](https://index.ros.org/p/rmf_traffic_editor/)将参考线的 `distance` 定义为米。该比例尺来自公开地图配置，不等于我们在现场复测过尺寸 |
| 机器人速度 | 0.5 米/秒，用于估计无额外等待的路段行驶时间 | 原项目的 [TIAGo fleet adapter](https://github.com/scafld/covsg24_fleet_backend/blob/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_fleet_server/launch/fleet_adapters/tiago_fleet_adapter.launch.xml)将 `linear_velocity` 标为标称值。配置值不是每次运输的实测速度 |
| 取样点装载 | 原论文模型为每份样本 30–60 秒；以 10 秒一步时，仿真均匀选取 30、40、50、60 秒 | [Rossetti、Felder 和 Kumar 的医院物流模拟研究](https://cpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/7/795/files/2016/08/mobile_robot_simulation_hcm.pdf)第 9 页说明该范围依据观察建模。这是另一家医院的历史研究，只能作为可追溯的跨场景参数，不能称为塔尔图医院实测 |
| 实验室交付操作 | 示例暂按 1 步，即 10 秒 | **本项目的测试假设**；尚未找到塔尔图案例中可用的人工交接实测分布 |
| 步长 | 10 秒用于示例离散化 | **计算设置**，不是任何论文给出的运输时长。单条原始路段向上取整可能增加不到一个仿真步的离散误差 |

对原始导航边，估计路段步数为 `max(1, ceil(像素距离 × 米/像素 ÷ 标称速度 ÷ 每步秒数))`。程序将路段拆成对应数量的共享中间节点，保留原始单向或双向通行方向；中间节点是计算插值，不是上游地图新增的真实设施。示例得到 76 个离散节点、112 条有向连接。世界模型在该图上计算以步数表示的预测路程。

这个时间模型忽略加减速、转弯、行人、门打开等待、拥堵和网络延迟，且样本装载时间来自另一家医院，因此只能称为**有来源参数的估计模型**，不能称为经实测校准的塔尔图机器人时间，也不能用于推断真实安全距离或临床达标情况。

## 规则如何参与调度

`rule` 策略只在用户明确提供适用样本类型、规则、采集时间、仿真起始时刻以及上述时间模型时使用规则。示例使用 [Michigan Medicine 院内儿科血气手册](https://www.pathology.med.umich.edu/handbook/details?print=1&testID=1088)所载采集至实验室接收时间作为参考：将接收期限换算为仿真中的**交付代理上限**。这里把仿真交付完成近似为接收时刻仅是测试假设；实际接收可能发生在另一时刻，按时交付不能证明按时接收，更不能证明临床合规。手册中的分析时限没有被当成运输时限；仿真也没有建模检验分析。

规则策略先比较任务的交付代理上限，较早者优先；对于同一上限，用世界模型预测选择机器人。没有已知代理上限的任务排在已知上限任务之后，再按预测结果排序。该方法是不可抢占的调度启发式，不保证全局最优、任务一定按时或真实环境安全。

原 `world` 策略仅按当前候选任务自身的预计完成时间与迟到惩罚评分。有些任务即使立刻执行也预计迟到时，这一分数可能很高，因而先选择时限宽松任务；新增策略显式优先较早代理上限。测试中也检查了已超过上限的任务不会因这一原因被排在后面。

若没有采集时间，系统不推断截止时间；明确指定的规则与样本类型不匹配时拒绝应用。规则来源、适用机构、时间模型和跨机构用途保存在结果元数据中。包装与温度仍仅检查录入值，不作为自动拦截或实体条件验证。

## 人工诊断场景和运行方式

[样本示例](../examples/rule-dispatch-synthetic.json)明确标明：塔尔图路网与 Michigan Medicine 的儿科血气规则组合，只用于人工构造的功能测试。两份样本同时可调度，`S02` 采集更早，因而代理上限更早。一个机器人比较最近取样点、原 `world` 和 `rule` 三种策略；两个样本都最终完成。

本地一次固定种子的测试中，最近策略与原 `world` 均先分配 `S01`，各有一个任务在仿真中超过交付代理上限；`rule` 先分配 `S02`，无任务超过该代理上限。这个例子是为了验证规则确实影响选择而设计的单一诊断场景，**不是算法普遍更优或临床合规的证据**。

[GitHub Actions 42315794163](https://github.com/lieuyimo/wm/actions/runs/42315794163) 对提交 `dc3f683ac68040557667f0c0bad70e40384c78cc` 通过 29 项 Python 测试、带来源时间模型的公开数据回放、人工规则调度比较和无 GPU 浏览器检查。公开数据回放中，两种既有策略均完成 4/4 个映射任务；[完整日志](../validation/logs/time-rule-dispatch-action-42315794163.log)已归档。工作流不上传原始临床文件或逐条衍生数据。

```bash
mkdir -p /tmp/wm-time
curl --fail --location 'https://raw.githubusercontent.com/scafld/covsg24_fleet_backend/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml' -o /tmp/wm-time/hospital.building.yaml
curl --fail --location 'https://raw.githubusercontent.com/scafld/covsg24_fleet_backend/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_fleet_server/launch/fleet_adapters/tiago_fleet_adapter.launch.xml' -o /tmp/wm-time/tiago.launch.xml
python -m lab_transport.rule_dispatch --map-file /tmp/wm-time/hospital.building.yaml --tiago-config /tmp/wm-time/tiago.launch.xml --seconds-per-tick 10 --profiles scenarios/specimen-rule-profiles.json --samples examples/rule-dispatch-synthetic.json --output /tmp/wm-time/run
python tests/check_rule_run.py /tmp/wm-time/run
node tests/test_graph_playback.js /tmp/wm-time/run/rule/report.html rule
```

程序校验两份上游文件的固定 SHA-256，结果目录不覆盖现有文件。需要重新运行时请更换输出目录。目录中的 `rule/report.html` 可离线查看调度过程和代理上限提示。

带来源时间模型也可与 MIMIC 单次 ICU stay 的相对采样时序组合：在[数据说明](specimen-rules-and-public-data.md)的示例命令中，把 `--seconds-per-tick 60` 改为 `--seconds-per-tick 10 --tiago-config /tmp/wm-time/tiago.launch.xml`。该时序只标识一般血液样本，没有证据对应 Michigan 儿科血气规则，因此不会自动套用规则或临床截止时间。
