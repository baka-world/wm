# PR #16 世界模型调度价值上限与瓶颈归因（CPU 离线确认集）

该文档由原始确认集结果自动生成，不包含历史 16 决策点的独立统计。
它只讨论固定 CP-SAT 候选、统一 nearest_batch 后续策略、固定随机实现的事后机会空间。
Full-outcome Oracle 并不是可在线执行的世界模型，不存在算法优势的先验保证。

## 冻结范围与样本

- 场景 24，地图/种子簇 8，采样点 48，候选总数 174。
- 有效决策点 48，失败或不可比较点 0（不静默删除）。
- 任务表与先前数据重复 0 次；跨新种子重复 0 次。
- 同一地图/种子的 serial/parallel/outage 共用任务表是预先定义的配对控制，不是三次独立试验。
- 进程 CPU 时间 82.5218 秒；墙钟时间 82.5353 秒（不含环境准备）。

## 六种方法的有限候选决策损失

只有所有候选都完成全部业务任务，才将受限平均流转差称为 step regret。
未完成任务采用完成数优先的排序，并单独记录流转时间差，绝不合并成任意损失标量。
先按地图/种子聚合，再等权平均各簇；决策点不是独立样本。

| 方法 | 平均簇决策损失（步） | 纳入簇数 | 命中事后最优 | 与 Rule 相同选择 | 更好/更差/无变化 |
|---|---:|---:|---:|---:|---:|
| Full-outcome Oracle | 0.0000 | 8 | 48/48 | 21 | 27/0/0 |
| PETS-5 | 1.7639 | 8 | 21/48 | 48 | 0/0/0 |
| Perfect-State-20 | 0.2812 | 8 | 34/48 | 26 | 17/5/0 |
| Perfect-State-5 | 1.2257 | 8 | 24/48 | 42 | 4/2/0 |
| Rule-5 | 1.7639 | 8 | 21/48 | 48 | 0/0/0 |
| Structured-5 | 1.7153 | 8 | 22/48 | 47 | 1/0/0 |

Oracle 相对 Rule-5 的有限候选事后机会均值： 1.7639 步。
Oracle regret 为零只是定义中的取最小值，不能当作真实算法增益。

## Q1 候选结果是否有真实差异

- 完全相同 1/48；差距不足 1 步 10/48。
- 差距至少 1 步 38/48；至少 3 步 25/48。
- 候选完成数量不完全相同或有超时 0；最大完整流转差 12.1667 步。
- 簇均候选最大差 3.6667 步。

## Q2–Q4 真实未来、窗口和预测误差

Perfect-State-5 和 Perfect-State-20 都直接取每条候选反事实轨迹的真实后续状态，
但严格沿用原 workflow_score 排序；它们与直接用完整结果选择的 Oracle 不同。
先比较 Rule-5 与 Perfect-State-5，再比较 Perfect-State-5 与 Perfect-State-20；
真实 20 步依然无法排序时，必须讨论评分函数对最终业务流转目标的偏差。

- 真未来长度不足或不可比较：5 步 0，20 步 0。
- 方法错误：{}
- 完整反事实的事件总量：{"resource_wait": 1029, "service_finished": 3606, "service_started": 3606, "stage_ready": 759, "station_unavailable_wait": 0}
- 触发联合路径恢复次数：33（并非服务预测误差）。
- 原始文件保留逐候选五步服务、阶段释放、机器人位置与站点占用差异；
  这些误差既可能来自服务事件，也可能来自联合路径或后续派工，不得直接归因神经网络。

## Q5 候选覆盖

- 原候选方案没有覆盖的当前合法单边（robot-task）共 24 条。
- 单边合法不构成另一整个分配方案可执行或更优的证明，不能单凭此宣称候选生成是主瓶颈。

## 数值反例与正例

- empty8/81013/workflow_outage tick=0: 候选 4，最大差 12.1667；Rule regret 10.8333，Perfect-5 10.8333，Perfect-20 0.0000，PETS 10.8333，Structured 10.8333。
- empty8/81012/workflow_outage tick=0: 候选 4，最大差 9.6667；Rule regret 6.0000，Perfect-5 6.0000，Perfect-20 0.0000，PETS 6.0000，Structured 6.0000。
- random32/82013/workflow_outage tick=0: 候选 4，最大差 9.6667；Rule regret 9.5000，Perfect-5 0.0000，Perfect-20 0.0000，PETS 9.5000，Structured 9.5000。
- random32/82012/workflow_outage tick=0: 候选 4，最大差 8.6667；Rule regret 8.3333，Perfect-5 0.0000，Perfect-20 0.0000，PETS 8.3333，Structured 8.3333。
- empty8/81011/workflow_outage tick=0: 候选 4，最大差 8.3333；Rule regret 8.3333，Perfect-5 0.0000，Perfect-20 0.0000，PETS 8.3333，Structured 8.3333。
- empty8/81013/workflow_serial tick=0: 候选 4，最大差 7.5000；Rule regret 3.6667，Perfect-5 3.6667，Perfect-20 0.0000，PETS 3.6667，Structured 3.6667。
- empty8/81014/workflow_outage tick=0: 候选 4，最大差 7.1667；Rule regret 7.1667，Perfect-5 7.1667，Perfect-20 0.0000，PETS 7.1667，Structured 7.1667。
- empty8/81012/workflow_serial tick=0: 候选 4，最大差 6.5000；Rule regret 0.3333，Perfect-5 0.3333，Perfect-20 0.3333，PETS 0.3333，Structured 0.3333。

## 失败与反事实公平

- 反事实执行失败 0 条；所有失败在原始记录中有原因。
- 历史机制验收结果：0 个不一致。
- 每个状态第一个候选被再次独立重放；事实策略在全部分支后重新运行，检查完整历史和事件不变。
- 每条分支的恢复快照/路径缓存重新构造；按样本、阶段、站点键控的服务随机扰动保证同次实现公平。
- 现有单次随机实现无法排除事后幸运分支的影响。若差距大，需要冻结并验证同事实前缀的额外随机性分析。

## 研究判断与停止条件

**E: multiple bottlenecks; 20-step true states outperform 5-step true states, and 5-step truth also beats the predictors**。

这个标签仅是基于明确数值的工作诊断，不自动证明某个瓶颈是唯一因果因素。
当前任务上是否继续训练学习预测器，必须以规则基线与真未来参考的实际差距为前提。
本轮停止后不自动训练、改架构、改候选算法或合并 PR。

## 完整重现命令

需要预先准备官方 PIBT 共享库与 LaCAM3 worker，再运行：

    python -m unittest discover -s experiments/oracle_value/tests -v
    python -m experiments.oracle_value.evaluate --profile history --pibt-library PATH --lacam-worker PATH --output /tmp/oracle-history
    python -m experiments.oracle_value.evaluate --profile confirmation --pibt-library PATH --lacam-worker PATH --output /tmp/oracle-confirm

GitHub Actions 的 CPU 工作流负责构建现有官方后端，并保存完整控制台日志与运行产物。
