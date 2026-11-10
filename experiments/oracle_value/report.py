"""Render a results report only after the full confirmation run succeeds."""
import gzip
import json
from collections import Counter
from pathlib import Path
import sys


def f(x):
    return "undefined" if x is None else (f"{x:.4f}" if isinstance(x,(int,float)) else str(x))


def report(summary_path, raw_path, out, robustness_summary=None, attribution_summary=None):
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
    if attribution_summary:
        attribution=json.loads(Path(attribution_summary).read_text(encoding="utf-8"))
        out_rows.extend([
            "",
            "## 五步窗口模型预测的决策相关误差（第一粒子，仅离线归因）",
            "",
            "| 模型 | 服务真/预测/匹配/漏报/误报 | 服务匹配事件绝对时间误差合计 | 后继释放真/预测/匹配/漏报/误报 | 机器人差异步 | 资源状态差异步 |",
            "|---|---|---:|---|---:|---:|",
        ])
        for model,x in attribution['models'].items():
            service="/".join(str(x.get('service_'+key+'_events',0))
                             for key in ('true','predicted','matched','missed','spurious'))
            successor="/".join(str(x.get('stage_release_'+key+'_events',0))
                               for key in ('true','predicted','matched','missed','spurious'))
            out_rows.append(f"| {model} | {service} | {x.get('service_timing_abs_error_ticks',0)} | "
                            f"{successor} | {x.get('robot_position_task_disagreement',0)} | "
                            f"{x.get('station_capacity_disagreement',0)} |")
        out_rows.extend([
            "",
            "上述事件匹配绝对时间误差仅覆盖同时出现在五步窗内的事件；漏报单列，不将其强行转成时间误差。",
            "后继释放也按相同规则处理，不能因服务完成事件不同就认定全部结果来自服务头。",
            "完整反事实中发生在第 5 步之后、第 20 步以内的服务完成/后继释放/资源等待事件数量："+
                json.dumps(attribution['outside_5_inside_20_events'],ensure_ascii=False),
            "",
        ])
    if robustness_summary:
        robust=json.loads(Path(robustness_summary).read_text(encoding="utf-8"))
        out_rows.extend([
            "",
            "## 预先冻结的多随机实现稳健性检验（仅候选差距 ≥3 步）",
            "",
            f"- 达标的状态 {robust['qualifying_decisions']}，成功 {robust['valid_robust_decisions']}，失败 {robust['failed_replays']}。",
            f"- 固定额外服务扰动种子 {robust['extra_noise_seeds']}；额外反事实候选重放 {robust['candidate_replays']} 条。",
            f"- 原单次 Oracle 的候选在额外实现平均后仍是最佳：{robust['nominal_oracle_remains_mean_best']}/{robust['valid_robust_decisions']}。",
            f"- 达标状态的额外实现下平均 Rule 到均值最佳机会：{f(robust['mean_robust_rule_opportunity_qualifying_states'])} 步。",
            f"- 原单次 Oracle 选中的候选在额外实现下对 Rule 的平均收益：{f(robust['mean_robust_nominal_oracle_gain_qualifying_states'])} 步。",
            f"- 条件选中簇的等权均值最佳机会：{f(robust['mean_of_qualifying_cluster_means'])} 步。",
            f"- 稳健性重放墙钟耗时 {f(robust['wall_seconds'])} 秒。",
            "- **只对首轮大差距状态重放**，因此上述均值不能与全体 48 状态的 Rule 损失均值直接相减。",
            "- 服务扰动仅改变决策时刻之后新开服务会话的按任务/阶段键控随机数，"
            "事实前缀、未来故障/释放和运动算法保持不变。",
            "- 额外种子用于事后稳健性，不向 Rule/PETS/Structured 的预测过程泄露。",
        ])
    out_rows.extend([
        "",
        "## 失败与反事实公平",
        "",
        f"- 反事实执行失败 {len(errors)} 条；所有失败在原始记录中有原因。",
        f"- 历史机制验收结果：{len(data['history_validation_issues'])} 个不一致。",
        "- 每个状态第一个候选被再次独立重放；事实策略在全部分支后重新运行，检查完整历史和事件不变。",
        "- 每条分支的恢复快照/路径缓存重新构造；按样本、阶段、站点键控的服务随机扰动保证同次实现公平。",
        ("- 已单独归档同事实前缀、3 个固定额外未来服务时长种子的初步稳健性检验；"\
         "仍不能把事后候选均值最优视为可在线使用的策略。" if robustness_summary else\
         "- 现有单次随机实现无法排除事后幸运分支的影响。"),
        "",
        "## 研究判断与停止条件",
        "",
        "**" + s["diagnostic_not_final_causal_judgement"] + "**。",
        "",
        "这个标签仅是基于明确数值的工作诊断，不自动证明某个瓶颈是唯一因果因素。",
        f"- A 候选机会不足：不成立，38/48 个状态差距至少 1 步，25/48 个至少 3 步。",
        f"- B 预测误差：存在次要证据，准确 5 步使决策损失从 "
        f"{f(s['methods']['Rule-5']['regret']['mean_of_cluster_means'])} 降到 "
        f"{f(s['methods']['Perfect-State-5']['regret']['mean_of_cluster_means'])} 步；"
        "但 PETS 的 48 次选择全部与规则相同，结构化头仅 1 次改善。",
        f"- C 窗口/评分：更突出的瓶颈，准确 20 步将损失继续降至 "
        f"{f(s['methods']['Perfect-State-20']['regret']['mean_of_cluster_means'])} 步，"
        "仍有非零残余和错误排序。预测窗口与评分终端估计的贡献不能仅凭此次数据完全分离。",
        "- D 候选集：存在未覆盖的单条合法 robot-task 配对，但没有补充完整可行批次的更优反事实证据，不能定为 D。",
        "**最终选择 E（多瓶颈，当前以预测窗口/评分转换为主，预测误差亦存在）。**",
        "**研究建议：不要在当前 5 步 workflow_score 不变的条件下继续盲目训练 PETS/结构化模型；"
        "优先研究如何把更长时域的可信信息转化成更好决策。**",
        "该建议不构成开发下一代模型、重设评分或扩大候选搜索的自动授权。",
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
    report(*sys.argv[1:])
