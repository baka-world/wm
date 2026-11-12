# PR60 → A100 GPU 世界模型训练结果（2026-11-12）

基准：PR60 head `e32c402ecbf67d19788c0a5c99e5282d6b5017e5`。  
范围：真实 A100 上训练 PR59/PR60 的 teacher-forced `t→t+1` 公开状态模型；**不读取/生成 BLIND TEST，不宣称 H20 递归或在线调度收益**。

## 1. 数据与环境核验

- A100-SXM4-80GB，设备 0/1/6/7；GPU 2–5 被其他用户 vLLM 占用，未触碰。
- Python 3.11.15，torch 2.5.1+cu124，torch-geometric 2.6.1，numpy 1.26.4，驱动 595.71.05。
- TRAIN 60 独立场景 / **9,046** 条真实 t→t+1 公共状态转移；VALIDATION 12 独立场景 / **1,517** 条；BLIND TEST 0 生成/读取。
- 所有原始 artifacts `sha256sum -c SHA256SUMS` 通过；PR59 source head `48acd54b19805c4582bd616bb93fc81c7c92b10d`；PR60 new source head `a1b004f1f615165b8d6ce550b64c4f83f076ec88`；quality verdict PASS。
- map family：TRAIN `vertical_blocked_route_training`；VALIDATION `diagonal_staggered_validation`；无重叠。

## 2. GPU G0 smoke

hidden 48，2 epochs，1 seed 1810，batch 16，map_off + map_on：

| 项目 | 结果 |
|---|---|
| 墙钟 | 127 s（含数据加载/验证） |
| 参数量 | 52,953（两臂相同） |
| PyTorch 分配器峰值 | 39.48 MB |
| nvidia-smi 整卡峰值 | 601 MiB |
| GPU 利用率均值/峰值 | 9.14% / 14% |
| NaN/非有限 | 无 |
| map_on vs map_off（2 epoch val） | node MSE 0.017343 vs 0.017642；device Brier 0.010370 vs 0.010341；selection 0.019936 vs 0.020228 |

GPU 梯度契约（真实 batch）：loss 有限；`map_conv1` 梯度 L1=0.01178>0；`relation_step1` 梯度 L1=0.63693>0；峰值分配 25.27 MiB。

说明：`test_pr59.py` 完整 unittest 需要可选 `mbrl` 依赖，当前 GPU 环境未安装；GPU entrypoint 不依赖该模块。CPU CI 已运行原测试，本轮用真实 CUDA 梯度检查补足 map 分支合同。

## 3. 正式训练设置

- 6 个独立运行：`map_off`/`map_on` × seeds 1810/1811/1812；20 epochs；hidden 48；batch 16；lr 3e-4；weight decay 1e-4；device Brier selection weight 0.25。
- 4 张空闲 GPU 并行：GPU 0 顺序 `map_off`；GPU 1/6/7 分别 `map_on` 1810/1811/1812。
- 每 run 独立输出目录，`--output` 不覆盖；模型选择只用 VALIDATION macro node masked MSE + 0.25 device Brier；BLIND TEST 未读取。
- 每 run 保存 best/last checkpoint（含 Adam、CPU/CUDA RNG、source SHA、validation metrics）和 `selection.json`。

## 4. 正式结果

### 4.1 每 seed

| arm | seed | wall (s) | best epoch | node MSE | device Brier | new-task MAE | peak alloc (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| map_off | 1810 | 506.6 | 19 | 0.005236 | 0.004397 | 0.232941 | 37.65 |
| map_off | 1811 | 481.9 | 17 | 0.006151 | 0.004325 | 0.188078 | 37.79 |
| map_off | 1812 | 508.5 | 19 | 0.006214 | 0.004451 | 0.158438 | 37.86 |
| map_on | 1810 | 535.9 | 19 | 0.004196 | 0.004457 | 0.163585 | 37.77 |
| map_on | 1811 | 543.7 | 17 | 0.005582 | 0.004355 | 0.155641 | 37.91 |
| map_on | 1812 | 331.3 | 19 | 0.005623 | 0.004318 | 0.212555 | 37.98 |

### 4.2 聚合（3 seeds）

| arm | params | node MSE mean ± sd | device Brier mean ± sd | new-task MAE mean |
|---|---:|---:|---:|---:|
| map_off | 52,953 | 0.005867 ± 0.000447 | 0.0043910 ± 0.0000515 | 0.193152 |
| map_on | 52,953 | **0.005134 ± 0.000663** | **0.0043768 ± 0.0000588** | **0.177260** |
| 相对变化 | — | **−12.50%** | −0.32% | −8.23% |

配对差值（map_on − map_off）：

| seed | Δ node MSE | Δ device Brier | Δ new-task MAE |
|---:|---:|---:|---:|
| 1810 | −0.001040 | +0.000061 | −0.069355 |
| 1811 | −0.000569 | +0.000029 | −0.032437 |
| 1812 | −0.000592 | −0.000133 | +0.054117 |

- node MSE：map_on 在 **3/3 seeds、3/3 regimes** 更好。
- device Brier：map_on 均值略好，但只有 1/3 seeds 更好；差异量级 ~3×10⁻⁵–1.3×10⁻⁴（相对 ~0.7–3%），没有明显退步，也不是干净的校准胜出。
- new-task MAE：map_on 2/3 seeds 更好，均值 −8.23%。

### 4.3 分 regime

| regime | node MSE map_off/on | device Brier map_off/on |
|---|---:|---:|
| congestion | 0.005800 / **0.005272** | 0.002826 / **0.002755** |
| multi_fault | 0.006773 / **0.006112** | 0.006679 / **0.006677** |
| staggered_arrivals | 0.005029 / **0.004017** | **0.003668** / 0.003699 |

- map_on 在三个 regime 的 node MSE 全部更好。
- Brier 在 congestion/multi_fault 略好，staggered 略差（+0.000031，约 +0.84%）。

## 5. 资源

| 项目 | 结果 |
|---|---|
| 6 个 run 的 summed GPU/card time | **2,907.85 s = 0.8077 GPU-hour**（低于 4 GPU-hour 首轮预算） |
| 实际并行墙钟 | 约 27 min（4 张 GPU） |
| 参数 | 52,953（两臂相同） |
| PyTorch 分配器峰值 | 37.65–37.98 MB |
| nvidia-smi 整卡峰值 | 601 MiB |
| GPU 利用率均值/峰值（0/1/6/7） | 10.36/26%；3.98/18%；3.99/22%；3.95/27% |
| 功耗/温度峰值 | 65.6–74.5 W / 40°C |

低利用率说明该小模型在 A100 上仍是 CPU/data-loading 主导；没有资源异常或越界。

## 6. 检查点与可恢复产物

- 每个 arm/seed 的 `best.pt`、`last.pt` 都包含 model state、Adam state、CPU/CUDA RNG、source SHA、validation metrics；`train_report.json` 与 `selection.json` 同时保存。
- 6 个 run 的 best/last + train_report + selection 已打包到本地：
  `pr60-gpu-checkpoints-20261112.tar.gz`（6,661,520 B，SHA256 `919a0b625659e492811a74bc5cdf76a60ccbf639acb32c33a420b7db30a74e14`）。
- `checkpoint_manifest.csv` 记录每个 best/last 的 SHA256/大小和 selection SHA；`resource.json` 记录 GPU 使用与累计卡时。
- 原始 60/12 数据、checkpoint、Adam/RNG、源 artifacts 未提交 Git；仅提交紧凑 evidence。

## 7. 门禁裁决

**GO（条件性）**：map 信息在 one-step teacher-forced 状态预测上通过了稳定性检查——node MSE 在 3/3 seeds、3/3 regimes 更好，均值 **−12.50%**；参数数相同；没有 NaN/崩溃；没有 BLIND 泄漏。

**NO-GO（当前阶段）**：

- device Brier 只在 1/3 seeds 更好，staggered 子群略差；校准还不是干净胜出。
- 这是 teacher-forced one-step；没有 H5/H20 递归、没有外生事件自洽、没有在线调度收益。
- 不晋升新检查点、不启动 BLIND TEST、不把本结果称为完整世界模型或调度改善。

下一步若继续：先做 device on/off/recovery 分组校准与阈值分析；只有校准和一阶转移都可信后，才设计合法递归 H5/H20；最后才做决策感知与独立 BLIND（selection 先冻结）。

## 8. 产物

- `examples/evidence/pr60-gpu-training-20261112/`：summary、per-seed/per-epoch/per-scenario CSV、checkpoint manifest、GPU usage、aggregate.py、SHA256SUMS。
- 原始 GPU run 目录：`/data/wm-a100-pr60-gpu-20261112/runs/pr60-<arm>-<seed>-20ep-e32c402`。
- 本地 checkpoint archive：`/data/wm-a100-pr60-gpu-20261112/pr60-gpu-checkpoints-20261112.tar.gz`。

BLIND_TEST_FILES_READ=0；BLIND_TEST_EPISODES_GENERATED=0。未修改 PR60 代码，未合并 main。
