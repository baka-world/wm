# 可替换的 PyTorch 训练与仿真入口

本文的网络是为验证数据、训练、恢复、评估和仿真接口而准备的小型示例，不是已经确定的论文最终模型。已核实的项目交接材料没有给出此前提到的分层网络的完整结构，不能将本示例称为该网络的复现。所有样本任务与扰动均由仿真合成，公开地图上的站点角色也包含明确说明的实验假设；不是临床实测。

## CPU 验证

以下从本仓库根目录运行，使用 Linux/macOS 风格的 shell；Windows 上虚拟环境解释器是 `runs\torch-cpu-venv\Scripts\python.exe`。本项目其余标准库原型无需安装 PyTorch；这些命令的环境依赖另行安装。CPU 文件固定 PyTorch 2.5.1+cpu 和 NumPy 1.26.4，间接依赖没有全部锁定。

```bash
python -m venv runs/torch-cpu-venv
runs/torch-cpu-venv/bin/python -m pip install -r requirements-torch-cpu.txt
runs/torch-cpu-venv/bin/python -m lab_transport.training_data export \
  --download-sources --profile full --output runs/local-gpu-data
runs/torch-cpu-venv/bin/python -m lab_transport.torch_training \
  --dataset runs/local-gpu-data --output runs/cpu-training-smoke --device cpu \
  --epochs 1 --train-limit 500 --validation-limit 250
runs/torch-cpu-venv/bin/python -m lab_transport.torch_training \
  --dataset runs/local-gpu-data --output runs/cpu-training-smoke --device cpu \
  --epochs 2 --train-limit 500 --validation-limit 250 \
  --resume runs/cpu-training-smoke/epoch-0001.pt
runs/torch-cpu-venv/bin/python -m lab_transport.torch_evaluate \
  --dataset runs/local-gpu-data --checkpoint runs/cpu-training-smoke/epoch-0002.pt \
  --output runs/cpu-evaluation-smoke --device cpu --limit-per-split 150
runs/torch-cpu-venv/bin/python -m lab_transport.torch_rollout \
  --checkpoint runs/cpu-training-smoke/epoch-0002.pt \
  --output runs/cpu-rollout-smoke --device cpu --download-sources
```

这些命令会在 `runs/` 创建新目录；输出目录非空时会拒绝覆盖。已有数据集可直接引用，不需重复生成。导出器与 `--download-sources` 会下载并校验固定的公开源文件，不会把数据上传。示例的训练限制按数据顺序截取，目的是快速验收，不能当作有代表性的研究样本。完整训练可去掉两个 `--*-limit`，训练集和验证集分别使用；测试集不会用于反向传播或选择检查点。指定测试集后，应先固定配置，不得通过反复查看测试结果调整方案。数据集和地图衍生物只存本地，传播前需按来源核查许可。

训练每轮写一个 `epoch-NNNN.pt` 和对应运行报告；报告记录验证损失最低的轮次。继续训练时 `--epochs` 是包含已完成轮次的目标总数，`--resume` 必须指向同一输出目录，数据清单、适配器代码、共享特征代码、设备类型与名称、Python/PyTorch、批量大小、学习率、种子和样本限制必须匹配。模型、优化器、批次顺序以及相应随机数状态一同保存。CPU 下已用自动测试比较连续两轮和中断后恢复，模型参数与优化器状态逐项一致；不能据此保证不同 GPU 或软件版本逐位一致。CPU 检查点不能作为跨设备**继续训练**的检查点，但可在兼容代码和 GPU 环境中用于推理。请只加载可信的检查点和适配器代码。

## 接入用户本地 GPU 时

首先确定实际操作系统、驱动、GPU 显存和 CUDA 支持；使用 [PyTorch 官方版本安装说明](https://pytorch.org/get-started/previous-versions/#v251)安装与设备兼容的 **PyTorch 2.5.1 CUDA 构建**，不要在 GPU 环境安装上面的 CPU 专用依赖文件。尚未知道用户的具体设备，不能假设某一 CUDA wheel 适用。NumPy 可另行安装 `numpy==1.26.4`。

保持上面数据准备命令和参数，使用已安装 CUDA 构建的虚拟环境执行同样的训练、恢复、评估和闭环命令，把每处 `--device cpu` 改成 `--device cuda`，并使用新的输出目录。先保留两项 `--*-limit` 做少量数据验收，再依据显存与模型规模决定完整训练。显式 `--device cuda` 不可用会报错，不会静默改用 CPU；`--device auto` 会选择当前可用设备，报告中必须检查实际设备。报告记录设备名、PyTorch CUDA 构建和 PyTorch 分配器峰值显存（不等于整机显存占用）。目前这条 CUDA 路径尚未在用户设备上运行，不能视作 GPU 验收通过。

## 扩展点及信息边界

[训练入口](../lab_transport/torch_training.py)从已验证的分组轨迹读取当前观测、真实已执行分配及下一观测，按 `--adapter` 指定 Python 模块。适配器提供 `ADAPTER_VERSION`、`build_model()`、`encode_transition(geometry, observation, assignments, next_observation)`、`collate(items)`、`loss(outputs, labels)` 和 `build_dynamics(geometry, network, device)`。批次目前使用由张量组成的扁平字典，包含变长候选项掩码；更复杂图结构、序列、优化器、损失或分布式训练如有需要仍应扩展训练框架。`model.forward()` 只收到输入字典；适配器用下一观测构造标签时，必须自行确保没有将未来信息写入输入。本仓库示例有相应测试，不能自动证明第三方适配器无泄漏。

[示例适配器](../lab_transport/torch_smoke_adapter.py)的两个小网络预测当前已知机器人的单步移动与工作站服务完成，候选路径使用当前可见占用和公开静态路网构造；下一时刻的移动、后续任务、站点中断、真实服务倒计时、场次种子和完整任务时间表不进入特征。ID 仅用于匹配实体，不作为数值特征。预测时按自己已经预测的前序机器人动作约束后续候选；训练时候选由当前状态构造，不用真实未来动作形成候选掩码。它既不预测新任务，也不预测未来站点可用性变化。多步预测目前是重复调用单步模型，训练器尚不提供序列窗口训练。

[评估入口](../lab_transport/torch_evaluate.py)把固定检查点与规则预测、保持原状态预测比较。单步“联合准确率”仅覆盖起始时已知的样本状态和机器人位置，不包含站点是否变化、真实时限、轨迹距离或未知新任务；多步指标只包含未来没有新任务、站点变化和后续分配的窗口，报告保留排除数量与覆盖率，不可称为所有场景的多步准确率。当前入口要求评估数据清单与训练时相同，以便检查划分；换一套外部数据时，还需实现跨数据集任务时间表去重验证。

[调度入口](../lab_transport/torch_rollout.py)按有限候选预算和固定步数预测候选分配结果，在现有仿真器的外部分配接口中执行；事件日志记录预测、选择与调用次数。模型无效输出或预测约束检查失败时，使用第一个当前可用候选并记录回退，仿真器仍会检查分配及其已实现的离散移动约束；这不是活性、物理碰撞或医疗安全保证。模型预测以外的传统规划机制、置信度校准、完整等预算对照和真实设备联调仍需完成。闭环产物的完整回放属于仿真事后调试信息，可能包含模型不应使用的隐藏状态，不能当作模型输入。
