# 可追溯的医院样本转运场景

## 选择依据和边界

Valner 等于 2022 年发表的 [《Scalable and heterogenous mobile robot fleet-based task automation in crowded hospital environments—a field test》](https://doi.org/10.3389/frobt.2022.922835) 研究塔尔图大学医院 ICU 到实验室的样本转运，文中举例为血液样本。论文第 2.1 节和图 1 描述了医院实地平面、走廊、ICU 与实验室；机器人途经两道半自动门。第 2.4 节和表 1 描述医护人员发起任务并确认装载、实验室人员取样并确认交付。作者在第 3 节报告了 TIAGo 的实际转运演示。**多机器人在测试环境验证；论文明确指出当时尚未在医院使用多台配送机器人。**

论文的数据声明提供 [SCAFLD 项目](https://github.com/scafld)和 [Zenodo 复现包](https://doi.org/10.5281/zenodo.6467038)。我们检查了作者公开的 [`hospital_2nd_floor.building.yaml`](https://github.com/scafld/covsg24_fleet_backend/blob/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml)：其路网有 37 条连接，26 条双向、11 条单向；连接涉及 32 个路点。文件中有明确命名的 [ICU 取样点](https://github.com/scafld/covsg24_fleet_backend/blob/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml#L341)、[实验室交付点](https://github.com/scafld/covsg24_fleet_backend/blob/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml#L337)。

这是一例**院内 ICU 到实验室的转运**，不是实验室内部不同仪器之间的完整流程；它是论文场景的可引用来源，不能推断所有医院或本项目使用者的实际场地相同。

## 逐项来源与适配

| 内容 | 在本原型中的处理 | 来源与限制 |
| --- | --- | --- |
| 路网和坐标 | 运行时读取上游固定版本的路点坐标与车道方向，32 个路点、63 条有向边 | 对应论文公开的医院 RMF 路网；只读取原图导航图，不复制上游底图、墙体或家具，不是完整楼层或实验室精确数字孪生 |
| 取放地点 | 使用上游 `icu1_pickup_location` 和 `lab_dropoff_location` | 来源于上述固定版本配置；模型只表达取、送，不实现检验处理 |
| 样本流程 | 请求、取样确认、运输、交付确认的状态抽象 | 依据论文第 2.4 节表 1 的人工装载与取走流程；模拟中的一步服务不等于真实操作耗时 |
| 机器人数量及初始位置 | 两台；初始位置使用上游路点 `charging_point` 和 `icu_1_test_location_1` | **本项目功能测试设置**。原论文没有声称这两台机器人以此方式在医院实测 |
| 任务数量及到达时刻 | 4 个任务，释放于第 0、12、24、36 个抽象步，均为 ICU 到实验室 | **本项目确定性功能测试工作负载**，不是医院真实订单、到达率或样本统计；尚未在已核查论文和仓库文件中找到此类原始记录，Zenodo 复现包在当前环境未能打开 |
| 行驶和取放时间 | 每经过一条导航边和每次取放各占 1 个抽象步；无随机波动或预设故障 | **仿真简化**。图边长不一，不能把这些步数换算为秒、分钟或真实通行能力 |
| 时限、温度和样本类型 | 此测试不填写时限或温度；仅保留一般样本任务 | [WHO 实验室质量工具](https://extranet.who.int/lqsi/checklist/2/13)要求实验室的样本收集 SOP 包含运输时限、温度范围等，并对紧急样本有处理程序；该页面未给出适用于所有样本的统一数值。没有指定检验项目、医院 SOP 或样本稳定性依据时，不能虚构临床阈值。未设置时限显示为“未设置”，不计为零超时的合规证明 |
| 门、人群和安全 | 路网拓扑不含动态门控制、行人或实体尺寸，检查离散节点重合和对向交换 | 不能据此声称真实碰撞安全、障碍物避让能力或完成了医院部署验收 |

可以进一步借鉴具体、可核查的样本规则，但必须保留其适用范围：

- [Michigan Medicine 儿科血气检验手册](https://www.pathology.med.umich.edu/handbook/details?print=1&testID=1088)针对其院内儿科血气样本，要求室温、密封生物危险袋，并在采集后 30 分钟内送达实验室；手册同时提到分析时限。采集到交付与采集到分析是不同的时间区间，不能把 30 分钟全部当作机器人运输预算。
- [Leeds Teaching Hospitals 血气检验页面](https://www.leedsth.nhs.uk/services/pathology/tests/blood-gases/)要求其样本在采集后 10 分钟内由人送达，并先联系实验室。这是另一机构、另一流程的要求，也不能直接解释成允许机器人送样。
- [美国 CDC 成人血培养采集指导](https://www.cdc.gov/lab-quality/php/preventing-adult-blood-culture-contamination/collect.html)要求记录具体采集日期、时间和地点，并立即送实验室。它没有在该步骤给出可以直接使用的统一数值截止时间。

这些真实规则可指导后续的样本类型、采集时间、包装方式和时限字段设计；目前仿真只有抽象步，也不知道塔尔图案例中样本的具体检验项目和采用的 SOP，因此没有把上述其他机构的数值套到医院案例上。对实际样本仍应以使用单位现行 SOP 为准。

论文文章采用 CC BY；**文章许可不自动等于代码库中各文件的许可**。我们未在该后端仓库根目录找到明确许可证，也无法从当前环境核实 Zenodo 包内的具体许可。因此本仓库不分发上游地图文件或导出的地图数据；运行时从上游固定版本获取文件，校验 SHA-256，并在本地生成适配结果。需要再分发上游地图及衍生数据时应先核实其适用许可。我们没有修改上游仓库。

## 运行与验收

运行场景需要 Python 3 和 `curl`；附加的回放脚本测试需要 Node.js。不需要 GPU、ROS 或付费服务：

```bash
mkdir -p /tmp/wm-hospital
curl --fail --location 'https://raw.githubusercontent.com/scafld/covsg24_fleet_backend/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml' --output /tmp/wm-hospital/hospital.building.yaml
python -m lab_transport.hospital_case --map-file /tmp/wm-hospital/hospital.building.yaml --output /tmp/wm-hospital/run
python tests/check_hospital_run.py /tmp/wm-hospital/hospital.building.yaml /tmp/wm-hospital/run
node tests/test_graph_playback.js /tmp/wm-hospital/run/world/report.html
```

固定输入 SHA-256 为 `8c2ebd01b49818f1d40f32e849b2c41ac800861f0b4ce82498d4298c6f2ad058`；输入变化时适配器拒绝继续。适配器是针对该固定文件格式的窄范围读取器，不声称支持任意 RMF YAML。`report.html` 的箭头来自原路网方向；页面的坐标仅用于显示。输出目录不可覆盖，可更换目录重跑。

初次按上述路网执行时，原型在两个任务交会后卡住，两种策略在 150 个仿真步内均只完成 1/4 个任务。检查事件后，为该两机器人路网仿真增加了有限的联合节点搜索：完全停顿时优先为已装载机器人找到一段可走路径。重新执行后，两种策略均完成 4/4，均为 87 个抽象步；核对各 113 次节点移动均遵循上游单/双向连接、无节点重合和对向交换，均出现一次 13 步恢复计划。详见 [本地验收日志](../validation/logs/tartu-hospital-smoke-20261106.txt)。

[GitHub Actions 验收运行](https://github.com/lieuyimo/wm/actions/runs/42255557949)验证了 16 项 Python 测试、固定来源路网、两种策略、方向及节点检查，并通过无 GPU 的 Chrome 页面检查；[完整 job 日志](../validation/logs/sample-transport-hospital-action-42255557949.log)已归档，对应提交 `def7c5032909c5081e6831b4b7b02d77faecf4e8`。该运行的自动化检查是功能测试，不能替代真实场地测试。

上述只是一组人工设置任务的闭环功能测试，不是世界模型效果的统计比较，也不证明恢复算法对任意图、任意机器人数量或所有停滞都有效。原型的 `world` 仍是基于已观测服务时间和静态路径的轻量预测基线，尚未训练神经世界模型，尚未接入 Lay2Graph。下一步应以导师确认的具体样本类别和真实 SOP 为准引入时限、温度等业务条件；在取得有许可的任务记录或公开数据后再构造有依据的工作负载，并进一步校验门控制与通行约束。
