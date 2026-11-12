# PR54 GPU 任务交接清单

状态：**CPU 结构与因果数据门禁阶段。GPU 暂未启动。**

项目：`lieuyimo/wm`，本科毕业论文《基于世界模型的样本转运调度系统设计与实现》，方法目标 Decision-Aware Relational World Models for Dynamic Multi-Robot Scheduling。

工作分支：`research/pr54-temporal-relational-transition-20261112`。**保持 Draft；不合并任何历史 PR，不修改 PR19 工程演示、不覆盖 V4 或既有训练检查点。**

## 现阶段已经交付的代码

- `experiments/world_model_integration/temporal_pr54/protocol.json`：60个新独立真实BusinessSimulator回合，train36/validation12/test12，三种压力负载，历史窗口4，one-step真实状态转移，已有测试完全不复用。
- `pr54_public_sequence.py`：从官方 `on_transition` 真实公开状态回调生成 `(s_{t-3:t},a_t,s_{t+1})`；t+1只出现在监督标签，未知的未来发布、设备故障时刻只存在于事实仿真器。
- `pr54_model.py`：**已有官方 PyG RGCNConv ×2 + 每个公开实体4步GRU**，预测下一步的机器人、工序、工作站、资源可观测状态及匿名新任务事件。不训练 Q 网络，不从未来任务表创建当前图节点。
- `pr54_train.py`：train/validation-only最优epoch及结构选择；先写入 selection.json，再打开物理隔离TEST；CPU或显式`--device cuda`，训练权重、Adam优化器、Torch RNG、per-episode评价和来源环境完整保存。

**重要范围边界**：本轮PR54是 teacher-forced one-step 准确性与历史信息价值验证，**还不是可部署的全递归H20世界模拟器**。仅将代码在GPU上复跑，不等于完成模型迁移或证明在线收益。

## GPU 启动门槛

进入以下任一情形后，正式开新PR进行 GPU 工作，而不是继续增加 CPU 推理预算：

- 真实 one-step 公开时序模型与历史遮蔽对照的CPU数据门禁已经通过，并计划将训练集扩展到**至少200个独立场景**、跨工作站负载/设备故障时长及任务到达机制；或者
- 需要 **H5/H20真正递归的、带合法外生事件采样的世界状态rollout**，单次迭代需多粒子、多候选、多个训练seed与多步梯度传播；
- CPU已经成为模型学习及rollout验证吞吐瓶颈，而不是仅影响CP-SAT合法候选枚举。

若one-step历史建模在独立场景不优于无历史对照，应先完成模型训练/信息充分性审计，不允许凭猜测直接加深网络或无限制扩容。

## GPU 任务清单

- [ ] **G0 真实资源确认**：记录GPU名称、CUDA driver/runtime、PyTorch CUDA可用性、可用显存和启动空载显存。初步适配目标V100 16GB；其他GPU均以实测OOM和吞吐为准。不预先宣称GPU可用。
- [ ] **G1 原始来源校验**：拉取PR54数据GitHub Actions成功运行的**具体Artifact ID**，逐组验证SHA256SUMS/source_head/protocol哈希；train/validation/test分为三个物理JSON文件，不能让训练/模型选择代码访问test。
- [ ] **G2 GPU等价性预检**：安装仓库原锁依赖及官方`torch-geometric==2.6.1`。实际CUDA设备`torch.cuda.is_available()==True`，针对同一固定真实公共输入比较CPU/GPU预测、状态概率、合法候选排序和极小分差案例。容差在测试开始前锁定，报告排序翻转，不可直接修改浮点tiebreak掩盖翻转。
- [ ] **G3 GPU one-step复现**：以完全相同的预注册PR54数据运行下列GPU命令；保持数据拆分、两个对照arm与模型seed不变，记录train/val/test实体masked MSE、设备Brier、新工序预测MAE及best epoch，严禁根据TEST改结构。
- [ ] **G4 结构与容量试验**：确认G3等价后，**新PR、新冻结协议、新未见TEST**；对hidden48/96/128、关系传播2/3步及历史窗口1/4/8进行控制变量消融。训练大小、训练epoch、参数量、峰值显存、GPU时间、预测与校准分开统计。若不同规模的验证误差不改善，则不继续堆参数。
- [ ] **G5 真正世界状态递归**：复用真实业务公开状态合同和CP-SAT合法候选，预测(t+1)状态后构造下一合法公共状态，由**规则转移**处理工作流和容量约束，并以显式概率事件处理未来任务发布/设备恢复/服务；不从真实仿真器调用未来私有故障/任务表。完成H5/H20闭环rollout以及事件预测Brier、逐实体状态误差和物理合法性测试。
- [ ] **G6 关系与决策感知消融**：同一训练数据/模型规模/候选/随机seed比较 (a) rule/V4，(b) 纯预测关系模型，(c) history关系模型，(d) decision/value-aware关系模型，(e) frozen V4+ranker负对照。**决策损失单独反传必须触及共享关系状态转移参数**，同时报告概率校准；不能只换评分头。
- [ ] **G7 独立测试与在线真实闭环**：模型与lambda/容差冻结后第一次读取新独立TEST；先做H5/H20真实预测、场景级candidate regret，再接回完整BusinessSimulator的CP-SAT合法候选在线闭环，报告makespan、late jobs、设备事件与资源违规，按独立场景和训练seed给出配对不确定区间。
- [ ] **G8 可恢复归档**：保存best/latest权重、Adam、AMP GradScaler(若使用)、CUDA/Torch/Python/NumPy随机数状态、调度器及数据协议、代码SHA、环境锁、pip freeze、数据SHA、CLI参数、逐轮history、GPU名称/峰值显存/总GPU卡时。结束后释放GPU。

## 当前PR54 CPU/GPU可复现训练命令

在校验源Artifact并物理划分数据到 `$DATA_ROOT/{train,validation,test}.json` 后：

```bash
python -m pip install 'uv==0.13.5'
uv pip install --system --python "$(command -v python)" --index-strategy unsafe-best-match -r requirements-paper-reuse.lock
python -m pip install 'torch-geometric==2.6.1'

# 必须使用真正具备CUDA的PyTorch构建；原CPU锁依赖在GPU机器上
# 可能需要安装与驱动兼容的CUDA版torch，再重新核验torch/PyG。
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"

python -m experiments.world_model_integration.temporal_pr54.pr54_train \
  --input "$DATA_ROOT" \
  --out "$OUT_DIR" \
  --device cuda
```

注意：CPU runner使用的 `requirements-paper-reuse.lock` 历史环境可能锁定CPU torch，**在GPU环境必须显式核验CUDA torch而不是盲信安装命令**。真实GPU训练应在新PR写出可重现的GPU环境锁/镜像，绝不能静默修改仓库原始锁文件。

## 资源建议（预算，尚非实测）

- 初步使用 **一张V100 16GB或显存更大GPU** 跑小batch时序关系模型；先试`hidden=48/96`、少量训练step，记录最大显存，再决定`hidden=128`与ensemble大小。
- 第一批授权上限建议 **4 GPU卡时**（含启动预检、训练和受控消融）。达到上限或无法获得校准/泛化增益应停止并提交实测材料，不假定所有配置都能在4卡时完成。
- 每轮至少记录真实 wall seconds/GPU seconds/峰值显存、完整排除训练样本数量、损失和验证集指标；G5/G7正式H20多粒子在线预测预算需要依据G3/G4实测重新立项，而不是虚报当前GPU可行性。

## 可直接交给执行Agent的Prompt

见 [pr54-gpu-agent-prompt.md](pr54-gpu-agent-prompt.md)。仅当满足上面的启动门槛、实际具备GPU并得到明确的GPU执行预算时下发。**当前不宣称已经调用GPU。**
