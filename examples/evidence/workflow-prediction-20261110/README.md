# 完整工作流预测：实验和复现证据

[实验报告](../../../docs/workflow-prediction-results.md)给出可观测边界、指标定义、完整结果及局限；[冻结协议](../../../docs/workflow-prediction-protocol.md)记录预先固定的配置、评分和采样方法。所有任务、站点、容量与故障为仿真假设。

- `config.json` 和 `config-audit.json` 固定开发/确认场景及任务表重复检查；`historical-map-audit.json` 复用 PR #12 的既往地图与任务表审计。确认集是在**已见地图**上的新任务表与种子。
- `development/`、`confirmation/` 各含完整可观察事实轨迹、模型预测、逐粒子评分组件、固定候选、独立反事实及闭环结果的 `raw.json.gz`；摘要、工作流窗口覆盖和评分诊断分别存为 `summary.json`、`coverage.json`、`decision-diagnostics.json`。事实轨迹仅作离线标签，不含隐藏服务剩余倒计时。
- `edge-diagnostics.json` 给出确定性三阶段递归的全部 16 个预测/事实步骤，以及容量拒绝时旧、新站点服务关系的实际记录；该控制实验不用于声称神经模型准确。
- `pilot/` 保留首次试验、站点关系修复前后的试验及一次评估脚本变量名错误；`logs/pre-sort-repeat-failure.log.gz` 与 `pilot/pre-sort-development-*.json.gz` 保留排序稳定性修复前的首次重复失败。修复只固定计算顺序，没有改变分数公式或场景。
- `logs/final-acceptance.log.gz` 记录开发和确认集双次运行与比较；`logs/post-change-codecheck.log.gz` 核对补充属性处理后的场景输出，`logs/full-regression.log.gz`、`logs/latest-workflow-tests.log.gz`、`logs/motion-regression.log.gz` 分别记录原有回归、新增业务测试及 PR #14 运动历史场景回归。
- `provenance.json` 固定 PR #14、试验代码、公开地图哈希和既有模型检查点哈希；`SHA256SUMS` 覆盖此目录其他文件。`gzip -dc 文件` 可查看压缩内容。

在已安装 `requirements-paper-reuse.lock` 的 Linux / Python 3.10 CPU 环境，先按 PR #14 的 [PIBT](../../../tools/pibt_motion/prepare_upstream.sh) 和 [LaCAM3](../../../tools/lacam3_recovery/prepare_upstream.sh) 脚本构建官方组件，然后执行：

```bash
export WM_PIBT_LIBRARY=/path/to/pibt/bridge/libwm_pibt_bridge.so
export WM_LACAM3_WORKER=/path/to/lacam3/bridge/wm_lacam3_worker
python -m tools.pibt_motion.prepare_sources --output /tmp/wm-workflow-hospital-sources
export WM_HOSPITAL_SOURCES=/tmp/wm-workflow-hospital-sources
python -m unittest discover -s experiments/workflow_prediction/tests -v
bash tools/workflow_prediction/run_acceptance.sh "$WM_PIBT_LIBRARY" "$WM_LACAM3_WORKER" /tmp/wm-workflow-acceptance
```

输出目录必须不存在或为空，脚本拒绝覆盖。它对开发和确认场景各运行两遍，仅排除实测时间后逐项核对，再生成覆盖与评分诊断。完整 CPU 自动化见 [workflow-prediction.yml](../../../.github/workflows/workflow-prediction.yml)；原始、PIBT 和 LaCAM3 后端的历史场景在继承的 [LaCAM3 工作流](../../../.github/workflows/lacam3-recovery.yml)中重跑。
