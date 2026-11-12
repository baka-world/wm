# PR60 → GPU Agent：世界模型真实训练数据与复现实验交接

> 截至2026-11-12，**CPU高质量数据与PyG地图模型前向/梯度门禁已完成；GPU训练由用户委托其他Agent执行。**
> 当前没有实际GPU训练、CUDA性能测量、H20递归模型或真实在线调度收益。

## 1. 交接项目与不可变边界

项目仓库：<https://github.com/lieuyimo/wm>。

本科毕业论文：《基于世界模型的样本转运调度系统设计与实现》；研究论文方法为关系感知、动作条件化、面向决策的系统世界状态转移学习。真正优先事项是**学到合法且有决策价值的系统动态**，不是继续改调度评分或仅训练Q网络。

- PR59（基线）：<https://github.com/lieuyimo/wm/pull/59>，含**可微分公开地图 SAGEConv + 时序GRU + 原RGCN**的单步状态模型和参数一致的map-on/map-off对照；PR59测试使用原真实`BusinessSimulator`第一步观察，证明跨PyG图的map/entity两个ID空间正确、地图分支梯度可到达状态转移。
- PR60（当前数据/交接）：<https://github.com/lieuyimo/wm/pull/60>，从PR59继承全部代码，在CPU新增36个非重叠训练场景；原24个不修改，验证12个不修改。
- PR54历史 CPU 训练：历史 GRU+RGCN 在固定地图真实独立TEST上下一步节点MSE优于快照约15.2%，但设备可用性Brier轻微退步。需要特别验证设备故障/恢复校准，不能把节点MSE改善说成全面世界模型改善。

**禁止**合并旧PR、覆盖旧权重、改原V4或CP-SAT、偷偷访问独立BLIND测试结果、用未来隐藏的故障结束时间/未发布任务表当模型输入。源数据是真实**合成仿真**而非企业现场数据。

## 2. 本轮已经实际通过的CPU门禁

| 数据／结果 | 原始GitHub Actions | 原始Artifact ID | 审计事实 |
|---|---|---|---|
| PR59原始TRAIN 24 | [Run 43234653164](https://github.com/lieuyimo/wm/actions/runs/43234653164) | **13950899958** | 24场景、3,627连续真实公开下一步标签；21完成/3截尾、H20动作效应24/24 |
| PR59原始VAL 12 | 同上 | **13950504487** | 12场景、1,517连续真实下一步标签；12完成/0截尾、H20动作效应12/12 |
| PR59独立只读CPU质量审计 | [Run 43235876892](https://github.com/lieuyimo/wm/actions/runs/43235876892) | **13953285875** | `PASS_QUALITY_TO_HANDOFF_GRAPH_TRAINING`，所有预注册门槛PASS，TEST未生成 |
| PR60新TRAIN拥堵12 | [Run 43237354137](https://github.com/lieuyimo/wm/actions/runs/43237354137) | **13952952294** | 12个真实完整源，0GPU |
| PR60新TRAIN双故障12 | 同上 | **13952912183** | 12个真实完整源，0GPU |
| PR60新TRAIN分批到达12 | 同上 | **13952323097** | 12个真实完整源，0GPU |
| PR60增量新颖度、H20幅度及完整审计 | 同上 | **13953931113** | `PASS_60_TRAIN_TRUE_DYNAMICS_READY_FOR_GPU_AGENT`，所有预注册门槛PASS |

- **最终完整TRAIN = 60独立场景，9,046条连续t→t+1公共状态转移**（原24的3,627 + 新36的5,419）。
- **VALIDATION = 12独立场景，1,517条连续状态转移**。验证地图是与训练地图完全不同的**同尺寸7×5障碍图家族**，不等于不同尺寸地图泛化。
- **BLIND TEST = 12个仅在`experiments/world_model_integration/sealed_pr59/protocol.json`预注册的横向层架地图场景**，目前未生成任何真实标签文件。**不能在训练/模型选择前运行。**
- 新增36场景：31完成、5真实右截尾；H20动作差异36/36，且36/36最大候选H20平均公共状态L1≥0.01；48次down/48次recovery；新数据原始语义去重后唯一率75.38%，相对原24个训练场景的逐样本边际新颖率**87.14%**。
- 整个60个训练场景有8个真实右截尾；这不会否定有意义的真实一步状态标签与在真实观察范围内的H5/H20标签，但**最终完成时间不可伪造为260tick，也不可作为完整排序监督**。
- 训练场景有6种真实障碍布局×6种服务/容量/机器人/任务源配置，且3类负载均覆盖；验证/测试地图家族与训练图家族严格不重叠。以上均来自合成环境。

### 官方下载与校验命令（在有GitHub权限的GPU主机执行）

```bash
git clone https://github.com/lieuyimo/wm.git
cd wm
git fetch origin research/pr60-marginal-high-information-cpu-data-gpu-handoff-20261112
git checkout research/pr60-marginal-high-information-cpu-data-gpu-handoff-20261112
git rev-parse HEAD

mkdir -p /tmp/wm-gpu-source/pr59-train /tmp/wm-gpu-source/pr59-validation
mkdir -p /tmp/wm-gpu-source/pr60 /tmp/wm-gpu-source/quality/pr59 /tmp/wm-gpu-source/quality/pr60

gh run download 43234653164 --repo lieuyimo/wm \
  --name pr59-real-public-world-43234653164-train \
  --dir /tmp/wm-gpu-source/pr59-train

gh run download 43234653164 --repo lieuyimo/wm \
  --name pr59-real-public-world-43234653164-validation \
  --dir /tmp/wm-gpu-source/pr59-validation

gh run download 43237354137 --repo lieuyimo/wm \
  --pattern 'pr60-true-new-training-43237354137-*' \
  --dir /tmp/wm-gpu-source/pr60

gh run download 43235876892 --repo lieuyimo/wm \
  --name pr59-original-natural-train-val-readonly-quality-43235876892 \
  --dir /tmp/wm-gpu-source/quality/pr59

gh run download 43237354137 --repo lieuyimo/wm \
  --name pr60-real-60train-12val-marginal-quality-43237354137 \
  --dir /tmp/wm-gpu-source/quality/pr60

for folder in /tmp/wm-gpu-source/pr59-train /tmp/wm-gpu-source/pr59-validation \
  /tmp/wm-gpu-source/pr60/* /tmp/wm-gpu-source/quality/pr59 \
  /tmp/wm-gpu-source/quality/pr60; do
  (cd "$folder" && sha256sum -c SHA256SUMS)
done
```

进一步核验：PR59原始`source_head.txt`必须是 **`48acd54b19805c4582bd616bb93fc81c7c92b10d`**。PR60新原始源的`source_head.txt`必须与对应[Run 43237354137](https://github.com/lieuyimo/wm/actions/runs/43237354137)的`head_sha`完全一致。原始`quality.json`必须分别是上表两个PASS裁决，且`GPU_hours=0`和`BLIND_TEST_source_scenes_generated=0`。

## 3. GPU开始前的可执行结构门禁

参考源码：

- `experiments/world_model_integration/sealed_pr59/pr59_dual_model.py`，类型`PublicTopologyTemporalData`自定义`Data.__inc__`，**map-edge索引按地图cell数量增加**，entity-to-map二部链接的实体行、地图行分别增加各自节点数；此合同必须在真实GPU上再次验证。
- `PublicMapTemporalRelationalWorld`在t−3..t公开GRU信息+地图 SAGEConv两次消息传播后融入现有RGCN，最后**输出真正公共t+1逐实体6字段与新任务匿名事件概率**，不是单独Q值预测器。
- `experiments/world_model_integration/sealed_pr59/test_pr59.py`：含第一步真实仿真观察、两地图混合batch、带map与mask地图同参数、地图信息影响预测的测试和梯度非零。
- `experiments/world_model_integration/sealed_pr59/pr59_gpu_train.py`：**CUDA-only入口**，拒绝无GPU的自动CPU回退；用完整自然源训练、按独立场景宏平均VAL node MSE + 0.25设备Brier进行训练阶段epoch选择；保存每臂每seed的best/last权重+Adam优化器+CPU/CUDA RNG、原始source_sha和结果。
- **该GPU入口虽在CPU上做过语法和`--help`预检，尚未在GPU上实际训练或度量显存/训练时间。** 先做真实GPU烟测，检查是否出现任何CUDA算子、PyG batching或CUDA显存问题。

### 资源预算和阶段停机

第一批以**单张V100 16GB或同级GPU、最多4 GPU卡时**为初始研究预算，必须先做2 epoch、1训练seed、map_on vs map_off的烟测并记录真实峰值显存、epoch时间。若不存在适配的CUDA/PyG环境或烟测超过资源限制，提交可复现原因与所需最小额外资源，不要虚构预算已经足够。

通过后建议以`hidden=48`、`map_off/map_on`、3个训练seed先完成**同参数公平对照**；96/128宽度仅在验证集证实48维拟合瓶颈且预算允许时按单变量增加。优先检验：设备down/recovery Brier、真实独立地图验证集状态MSE、任务完成/队列的误差、map信息梯度与消融、单步预测合法性。既有PR54设备Brier退步，**不得单凭一般node MSE下降就宣布PASS**。

### 建议起始命令

```bash
python -m pip install 'uv==0.13.5'
uv pip install --system --index-strategy unsafe-best-match -r requirements-paper-reuse.lock
python -m pip install 'torch-geometric==2.6.1'
python - <<'PY'
import torch, torch_geometric
print('torch',torch.__version__)
print('CUDA available',torch.cuda.is_available())
print('GPU',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
assert torch.cuda.is_available(), 'Real GPU required'
PY

mkdir -p /tmp/wm-gpu-artifacts
python -m experiments.world_model_integration.sealed_pr59.pr59_gpu_train \
  --train /tmp/wm-gpu-source/pr59-train/data.json \
          /tmp/wm-gpu-source/pr60/pr60-true-new-training-43237354137-congestion/data.json \
          /tmp/wm-gpu-source/pr60/pr60-true-new-training-43237354137-multi_fault/data.json \
          /tmp/wm-gpu-source/pr60/pr60-true-new-training-43237354137-staggered_arrivals/data.json \
  --validation /tmp/wm-gpu-source/pr59-validation/data.json \
  --output /tmp/wm-gpu-artifacts/pr59-smoke \
  --git-head "$(git rev-parse HEAD)" \
  --hidden 48 --epochs 2 --batch-size 16 \
  --seeds 1810 --arms map_off,map_on
```

请注意：每次运行`--output`目录必须是不存在的新路径，不能覆盖原始训练检查点。通过真实烟测后，在新路径下执行20epoch×3seed两臂，训练前固定随机种子列表/epoch/模型选择；不得根据BLIND TEST调整。命令示例只负责**真正的一步系统状态预测训练**，不等于H20递归世界模型训练。

## 4. 模型验证与后续方法研究门禁

**GPU G0：** 验证GPU软件、source SHA、模型数值有限、PyG map/entity batch索引与map分支确实有非零梯度；真实保存best/last/Adam/RNG，统计GPU卡时和显存。

**GPU G1：** 三训练seed相同条件下分别训练map ON/OFF，模型选择只用VALIDATION；报告逐场景自然分布的entity masked MSE、站点健康Brier、新可见任务数MAE、业务完成/队列变化误差。训练集按场景等权报告置信区间，不能把9,046个相关tick冒充9,046个独立世界。

**GPU G2：** 如地图作用和设备概率质量可信，设计可递归的单步`t→t+1`公开状态转移，将**未知未来到达、服务完成、设备故障恢复**建模为有校准概率的外生事件，同时用原业务规则投影到合法任务生命周期与资源容量。区分训练模型输出和真实仿真未来私有状态。真正测H5/H20滚动误差、校准、物理合法性，**不能把teacher-forced one-step说成完整H20**。

**GPU G3：** 再按相邻研究移植决策敏感loss（若验证显著改善，不可损害校准），从可信的公共状态转移反事实多步预测评估合法候选调度，跟原CP-SAT和原V4同初始状态闭环比较makespan/逾期/安全，按独立场景配对统计，不允许只在已开发地图上报告最终收益。

**GPU G4：** 仅在**所有验证模型决策与超参、checkpoints、GPU预算**被写入不可变`selection.json`后，才运行完全未被观察的`test`地图族。原`pr59_source.py`对`--split test`刻意直接拒绝：需另建有明确“selection存在且SHA固定”保护的独立TEST生成与评测入口，生成后只允许一次最终报告，不得根据TEST继续修改模型并宣称相同TEST仍盲测。

## 5. 交付材料与开发约束

GPU Agent需要提交新的**Draft PR**，基础必须是PR60最新HEAD，禁止覆盖/合并PR54/55/57/58/59/60历史PR。必须保留以下完整可恢复产出：

- 可复现实验环境：`pip freeze`、torch/PyG/CUDA版本、driver/GPU型号、CPU/GPU RNG、源码git SHA、每一源文件SHA256；
- 权重：每臂每seed每配置 best/last（含Adam和随机数状态）；训练/验证曲线、选择文件、指标定义；
- 资源：实际峰值显存、GPU卡时/epoch与总GPU卡时、推理吞吐、失败/回退记录；
- 测试：相同参数map-on/off、snapshot基线、逐场景验证、校准、合法动作差异幅度、必要的H5/H20滚动长期误差；
- 如果当前源数据不足、时序转移不可识别或地图编码没有收益，明确`NO-GO`并停止浪费GPU，不通过修改BLIND结果辩护。

**结论：本阶段进入“GPU世界模型训练前可交付的数据与代码状态”，而不是“新模型已训练完成”。下一次实际GPU训练与后续真盲测由用户授权的独立GPU Agent执行。**
