"""Render a results report only after the full confirmation run succeeds."""
import gzip
import json
from collections import Counter
from pathlib import Path
import sys


def f(x):
    return "undefined" if x is None else (f"{x:.4f}" if isinstance(x,(int,float)) else str(x))


def report(summary_path, raw_path, out):
    s=json.loads(Path(summary_path).read_text())
    with gzip.open(raw_path,"rt",encoding="utf-8") as h:
        data=json.load(h)
    states=s["states"]
    valid=[d for d in states if d.get("valid")]
    out_rows=[]
    out_rows.extend([
        "# PR #16 世界模型调度价值上限与瓶颈归因（CPU 离线确认集）",
        "",
        "该文档由原始确认集结果自动生成，不包含历史 16 决策点的独立统计。",
        "它只讨论固定 CP-SAT 候选、统一 nearest_batch 后续策略、固定随机实现的事后机会空间。",
        "Full-outcome Oracle 并不是可在线执行的世界模型，不存在算法优势的先验保证。",
        "",
        "## 冻结范围与样本",
        "",
        f"- 场景 {s['scenarios']}，地图/种子簇 {s['clusters']}，采样点 {s['actual_decisions']}，候选总数 {s['candidate_total']}。",
        f"- 有效决策点 {s['valid_decisions']}，失败或不可比较点 {s['failed_decisions']}（不静默删除）。",
        f"- 任务表与先前数据重复 {data['preflight']['prior_duplicates']} 次；跨新种子重复 {data['preflight']['cross_seed_duplicates']} 次。",
        "- 同一地图/种子的 serial/parallel/outage 共用任务表是预先定义的配对控制，不是三次独立试验。",
        f"- 进程 CPU 时间 {f(data['cpu_seconds'])} 秒；墙钟时间 {f(data['wall_seconds'])} 秒（不含环境准备）。",
        "",
        "## 六种方法的有限候选决策损失",
        "",
        "只有所有候选都完成全部业务任务，才将受限平均流转差称为 step regret。",
        "未完成任务采用完成数优先的排序，并单独记录流转时间差，绝不合并成任意损失标量。",
        "先按地图/种子聚合，再等权平均各簇；决策点不是独立样本。",
        "",
        "| 方法 | 平均簇决策损失（步） | 纳入簇数 | 命中事后最优 | 与 Rule 相同选择 | 更好/更差/无变化 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name,entry in s["methods"].items():
        reg=entry["regret"]
        out_rows.append(f"| {name} | {f(reg['mean_of_cluster_means'])} | {reg['n_clusters']} | "
                        f"{entry['oracle_best_hits']}/{len(valid)} | {entry['same_choice_as_rule']} | "
                        f"{entry['different_better']}/{entry['different_worse']}/{entry['different_equal']} |")
    gaps=s["candidate_gaps"]
    out_rows.extend([
        "",
        "Oracle 相对 Rule-5 的有限候选事后机会均值：" +
            f" {f(s['finite_candidate_rule_opportunity']['mean_of_cluster_means'])} 步。",
        "Oracle regret 为零只是定义中的取最小值，不能当作真实算法增益。",
        "",
        "## Q1 候选结果是否有真实差异",
        "",
        f"- 完全相同 {gaps['identical_complete_outcomes']}/{len(valid)}；差距不足 1 步 {gaps['gap_lt_1']}/{len(valid)}。",
        f"- 差距至少 1 步 {gaps['gap_ge_1']}/{len(valid)}；至少 3 步 {gaps['gap_ge_3']}/{len(valid)}。",
        f"- 候选完成数量不完全相同或有超时 {gaps['mixed_completion_candidates']}；最大完整流转差 {f(gaps['largest_complete_gap'])} 步。",
        f"- 簇均候选最大差 {f(gaps['cluster_spread']['mean_of_cluster_means'])} 步。",
        "",
        "## Q2–Q4 真实未来、窗口和预测误差",
        "",
        "Perfect-State-5 和 Perfect-State-20 都直接取每条候选反事实轨迹的真实后续状态，",
        "但严格沿用原 workflow_score 排序；它们与直接用完整结果选择的 Oracle 不同。",
        "先比较 Rule-5 与 Perfect-State-5，再比较 Perfect-State-5 与 Perfect-State-20；",
        "真实 20 步依然无法排序时，必须讨论评分函数对最终业务流转目标的偏差。",
    ])
    error_counts=Counter()
    events=Counter()
    errors=[]
    missing_5=missing_20=0
    uncovered=0
    recovery=0
    for case in data["cases"]:
        for dec in case["decisions"]:
            uncovered+=len(dec["candidate_diversity"]["unrepresented_eligible_pairs"])
            for method,record in dec["methods"].items():
                if record.get("error"):error_counts[method]+=1
            missing_5+=bool(dec["methods"]["Perfect-State-5"].get("error"))
            missing_20+=bool(dec["methods"]["Perfect-State-20"].get("error"))
            for branch in dec["branches"]:
                if "error" in branch:
                    errors.append((case["layout"],case["seed"],case["regime"],
                                   dec["tick"],branch["error"]))
                else:
                    events.update(branch["event_counts"])
                    recovery+=branch["motion_recovery"]["recovery_triggers"]
    out_rows.extend([
        "",
        f"- 真未来长度不足或不可比较：5 步 {missing_5}，20 步 {missing_20}。",
        "- 方法错误：" + json.dumps(dict(error_counts),ensure_ascii=False),
        "- 完整反事实的事件总量：" + json.dumps(dict(events),ensure_ascii=False),
        f"- 触发联合路径恢复次数：{recovery}（并非服务预测误差）。",
        "- 原始文件保留逐候选五步服务、阶段释放、机器人位置与站点占用差异；",
        "  这些误差既可能来自服务事件，也可能来自联合路径或后续派工，不得直接归因神经网络。",
        "",
        "## Q5 候选覆盖",
        "",
        f"- 原候选方案没有覆盖的当前合法单边（robot-task）共 {uncovered} 条。",
        "- 单边合法不构成另一整个分配方案可执行或更优的证明，不能单凭此宣称候选生成是主瓶颈。",
        "",
        "## 数值反例与正例",
        "",
    ])
    for d in sorted((x for x in valid if x["flow_spread"] is not None),
                    key=lambda x:x["flow_spread"],reverse=True)[:8]:
        m=d["methods"]
        out_rows.append(
            f"- {d['layout']}/{d['seed']}/{d['regime']} tick={d['tick']}: "
            f"候选 {d['candidate_count']}，最大差 {f(d['flow_spread'])}；"
            f"Rule regret {f(m['Rule-5']['regret_steps'])}，"
            f"Perfect-5 {f(m['Perfect-State-5']['regret_steps'])}，"
            f"Perfect-20 {f(m['Perfect-State-20']['regret_steps'])}，"
            f"PETS {f(m['PETS-5']['regret_steps'])}，"
            f"Structured {f(m['Structured-5']['regret_steps'])}。"
        )
    out_rows.extend([
        "",
        "## 失败与反事实公平",
        "",
        f"- 反事实执行失败 {len(errors)} 条；所有失败在原始记录中有原因。",
        f"- 历史机制验收结果：{len(data['history_validation_issues'])} 个不一致。",
        "- 每个状态第一个候选被再次独立重放；事实策略在全部分支后重新运行，检查完整历史和事件不变。",
        "- 每条分支的恢复快照/路径缓存重新构造；按样本、阶段、站点键控的服务随机扰动保证同次实现公平。",
        "- 现有单次随机实现无法排除事后幸运分支的影响。若差距大，需要冻结并验证同事实前缀的额外随机性分析。",
        "",
        "## 研究判断与停止条件",
        "",
        "**" + s["diagnostic_not_final_causal_judgement"] + "**。",
        "",
        "这个标签仅是基于明确数值的工作诊断，不自动证明某个瓶颈是唯一因果因素。",
        "当前任务上是否继续训练学习预测器，必须以规则基线与真未来参考的实际差距为前提。",
        "本轮停止后不自动训练、改架构、改候选算法或合并 PR。",
        "",
        "## 完整重现命令",
        "",
        "需要预先准备官方 PIBT 共享库与 LaCAM3 worker，再运行：",
        "",
        "    python -m unittest discover -s experiments/oracle_value/tests -v",
        "    python -m experiments.oracle_value.evaluate --profile history --pibt-library PATH --lacam-worker PATH --output /tmp/oracle-history",
        "    python -m experiments.oracle_value.evaluate --profile confirmation --pibt-library PATH --lacam-worker PATH --output /tmp/oracle-confirm",
        "",
        "GitHub Actions 的 CPU 工作流负责构建现有官方后端，并保存完整控制台日志与运行产物。",
    ])
    Path(out).write_text("\n".join(out_rows)+"\n",encoding="utf-8")


if __name__ == "__main__":
    report(*sys.argv[1:4])
