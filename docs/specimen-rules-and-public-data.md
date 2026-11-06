# 可追溯的样本规则与公开数据

## 本次实现了什么

样本任务现在可携带 `specimen_type`、`collected_at`、`packaging`、`temperature_condition` 和 `rule_profile`；这些字段为可选值，缺失不被当作满足要求。规则配置保存在 [specimen-rule-profiles.json](../scenarios/specimen-rule-profiles.json)，每条规则须注明适用机构、样本类型、适用范围、来源网址和访问日期。

当前参考配置来自 [Michigan Medicine 院内儿科血气手册](https://www.pathology.med.umich.edu/handbook/details?print=1&testID=1088)：室温、密封生物危险袋、采集后 30 分钟内送达；手册还提及采集后 30 分钟内分析。规则检查器分别比较**采集至实验室接收**和**采集至分析**，它们不是同一时间段。来源适用的是该院儿科血气检验，不能直接套用到所有血样、塔尔图医院或 MIMIC 记录。

以下命令用**人工构造的测试记录**演示规则输入和结果：

```bash
python -m lab_transport.specimen_rules \
  --profiles scenarios/specimen-rule-profiles.json \
  --samples examples/specimen-rules-synthetic.json
```

检查器只比较显式录入的字段：缺少收样、分析、包装或温度记录时返回 `unknown`；样本类型不匹配时返回 `not_applicable`；未知规则和无时区的真实时间输入会被拒绝。录入的包装或温度只是声明，不能证明实体条件。尤其是，仿真中的机器人 `delivered` 不会自动当作实验室实际 `received_at`。

当前规则检查是独立的数据处理功能，**尚不把医院 SOP 或真实分钟数自动用于调度器**：已有路网的仿真步数未经过物理时间标定，也没有真实实验室接收和分析事件。不能据此宣称临床合规或用于实际接收判定。扩展规则前需核实目标检验项目和适用机构的现行 SOP。

## 公开数据评估

| 来源 | 可以借鉴 | 不能推断或限制 |
| --- | --- | --- |
| [MIMIC-IV Clinical Database Demo 2.2](https://physionet.org/content/mimic-iv-demo/2.2/)，[DOI](https://doi.org/10.13026/dp1f-ex47)，开放访问、ODbL-1.0 | 100 名患者的脱敏临床演示数据；`labevents.specimen_id` 可将同一样本的多条检测记录分组；`d_labitems.fluid` 可识别血液检测；ICU stay 表可限定同一人的住院区间 | `charttime` **通常**接近采样时间，不等于运输请求；`storetime` 是检验信息在系统中可用的时间，不是实验室收件时间。它没有机器人路线或真实运输耗时 |
| MIMIC 日期处理，[原始说明](https://physionet.org/content/mimiciv/2.2/) | 同一患者内的时间间隔保持一致 | 日期对每名患者独立平移，不能把不同患者的绝对时间拼成真实同时到达的全院任务流 |
| [Clinical laboratory orders, specimens and results](https://databasesample.com/datasets/clinical-laboratory-orders-specimens-and-results-dataset)，CC0-1.0 | 有 95 条 `collected_at` / `received_at` 样本记录，可作为数据接口或展示的测试夹具 | 发布方明确表示每条均为生成数据，分布并非实测；不能作为真实医院任务流、实测运输时间或评估代表性的证据 |

[MIMIC 检验字段说明](https://mimic.mit.edu/docs/iv/modules/hosp/labevents.html)和 [项目数据字典](https://mimic.mit.edu/docs/iv/modules/hosp/d_labitems.html)是上述字段解释的来源。演示集用于检查流程和接口，患者样本量小，也不构成代表性医院数据。

## 从公开数据生成一段可复现的测试时序

适配器从固定版本的 `d_labitems`、`labevents`、`icustays` 读取数据并校验 PhysioNet 发布的 SHA-256。它按匿名标识确定性选择第一个符合条件的单次 ICU stay：在该 stay 中，血液类检测的采样时刻须落在入出 ICU 区间；按 `specimen_id` 将多项检验合并，每组取最早 `charttime`，取从首个样本起 24 小时内的前 4 个样本。输出仅有重新编号的样本、相对秒数和来源说明，不输出患者标识、绝对日期、检测结果或临床时限。

这些仍是**采样记录的时序代理**：数据本身没有标识样本是否由机器人送出、是否经由 ICU 取样点或某个实验室窗口。将其映射到塔尔图医院地图，是明确的跨机构仿真假设，不是重演某个医院的真实任务，也不代表全院负载。数据本身是 ODbL-1.0，注意 PhysioNet 对衍生数据的说明；本仓库不分发原始临床记录或生成的逐条时序。

在可访问互联网的本机或免费 GitHub runner 上：

```bash
mkdir -p /tmp/wm-mimic
curl --fail --location 'https://physionet.org/files/mimic-iv-demo/2.2/hosp/d_labitems.csv.gz' -o /tmp/wm-mimic/d_labitems.csv.gz
curl --fail --location 'https://physionet.org/files/mimic-iv-demo/2.2/hosp/labevents.csv.gz' -o /tmp/wm-mimic/labevents.csv.gz
curl --fail --location 'https://physionet.org/files/mimic-iv-demo/2.2/icu/icustays.csv.gz' -o /tmp/wm-mimic/icustays.csv.gz
curl --fail --location 'https://raw.githubusercontent.com/scafld/covsg24_fleet_backend/b9e32eeebb3a568af0e41c6e34500076cd2f6cad/covsg24_environments/environments/hospital_2nd_floor/hospital_2nd_floor.building.yaml' -o /tmp/wm-mimic/hospital.building.yaml
python -m lab_transport.mimic_workload --labitems /tmp/wm-mimic/d_labitems.csv.gz --labevents /tmp/wm-mimic/labevents.csv.gz --icustays /tmp/wm-mimic/icustays.csv.gz --output /tmp/wm-mimic/timeline.json
python -m lab_transport.hospital_case --map-file /tmp/wm-mimic/hospital.building.yaml --workload /tmp/wm-mimic/timeline.json --seconds-per-tick 60 --output /tmp/wm-mimic/replay
python tests/check_mimic_run.py /tmp/wm-mimic/timeline.json /tmp/wm-mimic/replay 60
```

这里的 `--seconds-per-tick 60` 是显式的**测试假设**，用于将相对采样秒数向上取整映射到仿真释放步，不是机器人速度或每条路段的实测值；每条边仍占 1 个仿真步。此数值未经过标定，不能用于实际运输时间或临床时限判断。要重新运行，请换用新的输出路径。

## 验收结果

[GitHub Actions 运行 42274589682](https://github.com/lieuyimo/wm/actions/runs/42274589682) 对提交 `c4e771a51fa93c5e23c39a3cc91664cad7201432` 从 PhysioNet 下载并校验三个公开演示文件，成功提取一个 ICU stay 的 4 个样本相对时序；两种调度策略各完成 4/4 个映射后的测试任务。工作流还通过了 24 项 Python 测试、路网方向和节点检查、规则示例检查，以及无 GPU Chrome 的回放与来源提示检查。[完整工作流日志](../validation/logs/mimic-rule-replay-action-42274589682.log)已归档；运行器未把原始或逐条衍生数据打包上传。

这次验收说明数据处理和仿真链路能够执行；它使用一段采样时序、跨机构地图和假设的步长，不能推断真实运输性能或证明预测策略优于基线。

回放仍不模拟人的移动、门的开关、空间碰撞或临床接收流程；当前世界模型是轻量预测基线，不是已训练的神经网络，也没有接入 Lay2Graph。程序和结果仅服务于本科毕设的仿真系统设计和测试。
