# 示例运行记录

这些文件来自 `scenarios/laboratory_demo.json`、种子 42 的一次合成数据运行。`seed42-comparison.json` 是两种策略的汇总；`seed42-world/report.html` 可离线回放预测调度过程，同目录还保存了事件、样本和汇总记录。`seed42-nearest/` 保存最近取样点策略的样本及汇总。

可在仓库根目录重新运行：

```bash
python -m lab_transport compare --seed 42 --output runs/demo-42
```

这是单场景、单种子的功能演示，时间单位是仿真步，不代表实际实验室时间，也不能用于证明普遍性能。
