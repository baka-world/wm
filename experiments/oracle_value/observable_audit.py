"""PR17: read-only, CPU-only observable-vs-hindsight audit of frozen PR16 evidence.

Never uses simulator internals as online predictors. The only future labels are
the already archived twenty actual post-decision observations per fixed plan.
No rerun, retraining, scenario reconstruction, or external imports.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
from statistics import mean

METHODS = ("Rule-5", "Perfect-State-5", "Perfect-State-20", "Full-outcome Oracle")
ROOT = Path("examples/evidence/oracle-value-20261110")
REPORT = Path("docs/observable-information-audit.md")
ROWS = ROOT / "observability-audit.jsonl"
SUMMARY = ROOT / "observability-audit-summary.json"


def station_status(observation):
    stations = observation.get("stations")
    if not isinstance(stations, list) or not stations:
        raise ValueError("observation has no stations")
    result = {}
    for station in stations:
        if "device_available" not in station:
            raise ValueError("device_available must be explicit; station.available is capacity-confounded")
        sid = station["id"]
        if sid in result:
            raise ValueError("duplicate station ID")
        result[sid] = bool(station["device_available"])
    return result


def fault_timeline(observation, states, horizon=20):
    """Detect real observed ONSETS, not an outage regime label or capacity waits.

    Each trace represents ONE original candidate's actual future. Change from
    device_available true -> false at the next boundary is an onset. Current
    unavailable is reported separately. A second false->true->false gives two
    independent onsets. All offsets are measured from the decision tick.
    """
    if len(states) < horizon:
        raise ValueError("truncated true trace; cannot infer no fault")
    base_tick = observation["tick"]
    previous = station_status(observation)
    initial_down = sorted(k for k, v in previous.items() if not v)
    onsets, recoveries, down_steps = [], [], []
    previous_tick = base_tick
    for state in states[:horizon]:
        tick = state.get("tick")
        if tick != previous_tick + 1:
            raise ValueError(f"nonconsecutive true observations: {previous_tick} -> {tick}")
        current = station_status(state)
        if set(current) != set(previous):
            raise ValueError("station set changed within true trace")
        offset = tick - base_tick
        for sid in sorted(current):
            if previous[sid] and not current[sid]:
                onsets.append({"station_id": sid, "absolute_tick": tick, "offset": offset})
            if not previous[sid] and current[sid]:
                recoveries.append({"station_id": sid, "absolute_tick": tick, "offset": offset})
            if not current[sid]:
                down_steps.append({"station_id": sid, "absolute_tick": tick, "offset": offset})
        previous, previous_tick = current, tick
    return {"initial_down": initial_down, "onsets": onsets,
            "recoveries": recoveries, "down_steps": down_steps}


def group_for(horizon, timeline):
    if timeline["initial_down"]:
        return "1_currently_unavailable"
    if any(e["offset"] <= horizon for e in timeline["onsets"]):
        return "2_future_fault_from_currently_normal"
    return "3_currently_normal_no_fault_within_horizon"


def visible_summary(observation):
    stations = observation["stations"]
    robots = observation.get("robots", [])
    samples = observation.get("samples", [])
    return {
        "observation_top_level_fields": sorted(observation),
        "station_fields": sorted(stations[0]) if stations else [],
        "robot_fields": sorted(robots[0]) if robots else [],
        "sample_fields": sorted(samples[0]) if samples else [],
        "currently_unavailable_stations": sorted(s["id"] for s in stations if not s["device_available"]),
        "capacity_blocked_stations": sorted(s["id"] for s in stations if s.get("capacity_blocked") is True),
        "occupied_capacity_total": sum(s.get("resource_occupied", 0) for s in stations),
        "occupied_capacity_by_resource": {
            s.get("resource_id", s["id"]): s.get("resource_occupied", 0)
            for s in sorted(stations, key=lambda x: x["id"])
        },
        "active_service_robots": sum(r.get("service_elapsed") is not None for r in robots),
        "observed_effective_service_count_total": sum(
            s.get("observed_effective_service_count", 0) or 0 for s in stations),
        "station_service_history": [
            {"station_id": s["id"],
             "observed_effective_service_count": s.get("observed_effective_service_count"),
             "observed_effective_service_mean": s.get("observed_effective_service_mean"),
             "expected_service_ticks": s.get("expected_service_ticks"),
             "history_complete_since_start": s.get("history_complete_since_start"),
             "service_session_active": s.get("service_session_active"),
             "service_in_progress": s.get("service_in_progress")}
            for s in sorted(stations, key=lambda x: x["id"])
        ],
        "recovery_snapshot_fields": sorted(
            (observation.get("recovery_snapshot") or {}).keys())
            if isinstance(observation.get("recovery_snapshot"), dict) else [],
        "service_elapsed": [{"robot_id": r["id"], "service_elapsed": r.get("service_elapsed"),
                             "phase": r.get("phase"), "task_id": r.get("task_id")}
                            for r in robots if r.get("service_elapsed") is not None],
        "robot_count": len(robots),
        "visible_sample_count": len(samples),
        "visible_pending_sample_count": sum(s.get("status") not in ("completed", "delivered")
                                            for s in samples),
        "explicit_history_fields": sorted(k for k in observation if "history" in k.lower()
                                          or "event" in k.lower() or "log" in k.lower()),
    }


def inspect_state(case, decision, summary_state, robust_map):
    if not summary_state["valid"]:
        raise ValueError("invalid previous decision included; cannot silently omit")
    observed = decision["observable_state"]
    if observed["tick"] != decision["tick"]:
        raise ValueError("decision tick mismatch")
    if len(decision["plans"]) != len(decision["branches"]):
        raise ValueError("candidate count mismatch")
    traces = [fault_timeline(observed, b["trace_first20"])
              for b in decision["branches"]]
    if not traces:
        raise ValueError("no true candidate traces")
    ref = traces[0]
    same_future_faults = all(
        t["initial_down"] == ref["initial_down"] and
        t["onsets"] == ref["onsets"] and t["recoveries"] == ref["recoveries"]
        for t in traces
    )
    if not same_future_faults:
        raise ValueError("candidate-dependent outage timeline: cannot assign a single future label")
    if tuple((case[k] for k in ("layout", "seed", "regime"))) != tuple(
            summary_state[k] for k in ("layout", "seed", "regime")):
        raise ValueError("summary state identity mismatch")
    if decision["tick"] != summary_state["tick"]:
        raise ValueError("summary tick mismatch")
    id_key = (case["layout"], case["seed"], case["regime"], decision["tick"])
    methods = summary_state["methods"]
    performance = {k: {
        "choice": methods[k]["choice"],
        "regret_steps": methods[k]["regret_steps"],
        "completion_deficit": methods[k]["completion_deficit"],
    } for k in METHODS}
    if any(x["regret_steps"] is None for x in performance.values()):
        raise ValueError("uncensored regret is unavailable; do not substitute a number")
    if methods["Full-outcome Oracle"]["regret_steps"] != 0:
        raise ValueError("full-outcome oracle is no longer zero; input changed")
    future = {}
    for horizon in (5, 20):
        events = [x for x in ref["onsets"] if x["offset"] <= horizon]
        down = [x for x in ref["down_steps"] if x["offset"] <= horizon]
        future[str(horizon)] = {
            "group": group_for(horizon, ref),
            "new_fault_onset": bool(events),
            "any_device_unavailable": bool(ref["initial_down"]) or bool(down),
            "first_onset_offset": min((x["offset"] for x in events), default=None),
            "first_onset_tick": min((x["absolute_tick"] for x in events), default=None),
            "onsets": events,
            "fault_affected_steps": down,
        }
    visible = visible_summary(observed)
    return {
        "layout": case["layout"], "seed": case["seed"], "regime": case["regime"],
        "cluster": case["layout"] + "/" + str(case["seed"]),
        "tick": decision["tick"], "id": "/".join(map(str, id_key)),
        "candidate_count": len(decision["plans"]),
        "available_online": visible,
        "public_deterministic": {
            "station_positions": {s["id"]: s.get("position") for s in observed["stations"]},
            "station_resource_capacity": {
                s.get("resource_id", s["id"]): s.get("resource_capacity")
                for s in observed["stations"]
            },
            "workflow_templates_not_embedded_in_observation": True,
            "map_and_workflow_templates_source": "business_transport/scenario.py",
        },
        "future_offline_label_only": {
            **future, "all_20_step_onsets": ref["onsets"],
            "all_20_step_recoveries": ref["recoveries"],
            "current_down_stations": ref["initial_down"],
            "identical_outage_timeline_across_all_fixed_candidates": same_future_faults,
        },
        "method_regrets": performance,
        "oracle_opportunity_vs_rule": performance["Rule-5"]["regret_steps"],
        "candidate_flow_spread": summary_state["flow_spread"],
        "candidate_outcome_tied": summary_state["outcome_tied"],
        "candidate_all_completed": summary_state["all_candidates_complete"],
        "robustness_additional_service_seeds_available": id_key in robust_map,
        "robustness_original_pr16_if_available": robust_map.get(id_key),
        "audit_note": "Future fault labels and per-candidate true traces are never online predictors.",
    }


def cluster_group_statistics(records, horizon):
    groups = ("1_currently_unavailable", "2_future_fault_from_currently_normal",
              "3_currently_normal_no_fault_within_horizon")
    output = {}
    for name in groups:
        chosen = [r for r in records if r["future_offline_label_only"][str(horizon)]["group"] == name]
        by_cluster = defaultdict(list)
        for row in chosen:
            by_cluster[row["cluster"]].append(row)
        def stats(field):
            cm = [mean(field(x) for x in rows) for rows in by_cluster.values()]
            return {"state_mean_descriptive": mean(field(x) for x in chosen) if chosen else None,
                    "cluster_mean_equal_weight": mean(cm) if cm else None,
                    "cluster_count": len(cm)}
        output[name] = {
            "decisions": len(chosen),
            "current_observation_descriptive": {
                "mean_occupied_capacity": stats(
                    lambda r: r["available_online"]["occupied_capacity_total"]),
                "mean_active_service_robots": stats(
                    lambda r: r["available_online"]["active_service_robots"]),
                "mean_historical_effective_service_count": stats(
                    lambda r: r["available_online"]["observed_effective_service_count_total"]),
            },
            "clusters_with_group": len(by_cluster),
            "mean_candidate_spread": stats(lambda r: r["candidate_flow_spread"]),
            "mean_oracle_opportunity_vs_rule": stats(lambda r: r["oracle_opportunity_vs_rule"]),
            "methods": {method: stats(lambda r: r["method_regrets"][method]["regret_steps"])
                        for method in METHODS},
        }
    return output


def load_and_analyze(raw, previous, robustness):
    if raw.get("profile") != "confirmation" or raw.get("limited_pilot"):
        raise ValueError("expected complete frozen PR16 confirmation, not pilot/history")
    cases = raw["cases"]
    if len(cases) != 24 or len({(c["layout"], c["seed"]) for c in cases}) != 8:
        raise ValueError("expected 24 scenes and 8 clusters; frozen evidence differs")
    if previous["actual_decisions"] != 48 or previous["valid_decisions"] != 48:
        raise ValueError("expected all 48 PR16 valid decisions")
    old = {(s["layout"], s["seed"], s["regime"], s["tick"]): s
           for s in previous["states"]}
    if len(old) != 48:
        raise ValueError("duplicate or missing PR16 summary identities")
    robust_map = {(x["layout"], x["seed"], x["regime"], x["tick"]): x["result"]
                  for x in robustness["states"]}
    records = []
    for case in cases:
        for decision in case["decisions"]:
            key = case["layout"], case["seed"], case["regime"], decision["tick"]
            if key not in old:
                raise ValueError(f"no PR16 summary row for {key}")
            records.append(inspect_state(case, decision, old[key], robust_map))
    if len(records) != 48 or len({r["id"] for r in records}) != 48:
        raise ValueError("48 unique decisions required")
    if {tuple(r[k] for k in ("layout", "seed", "regime", "tick"))
            for r in records} != set(old):
        raise ValueError("raw and summary decision identities differ")
    records.sort(key=lambda r: (r["layout"], r["seed"], r["regime"], r["tick"]))
    totals = {
        "decisions": len(records), "scenes": len(cases),
        "clusters": len({r["cluster"] for r in records}),
        "candidates": sum(r["candidate_count"] for r in records),
        "robustness_overlap": sum(r["robustness_additional_service_seeds_available"]
                                  for r in records),
        "pre_fault_20_states": [
            {"id": r["id"], "tick": r["tick"],
             "first_fault_offset": r["future_offline_label_only"]["20"]["first_onset_offset"],
             "occupied_capacity_total": r["available_online"]["occupied_capacity_total"],
             "active_service_robots": r["available_online"]["active_service_robots"],
             "observed_effective_service_count_total":
                 r["available_online"]["observed_effective_service_count_total"]}
            for r in records
            if r["future_offline_label_only"]["20"]["group"] ==
               "2_future_fault_from_currently_normal"
        ],
        "outage_timeline_consistent_across_all_candidates": all(
            r["future_offline_label_only"]["identical_outage_timeline_across_all_fixed_candidates"]
            for r in records),
        "onset_within_5": sum(r["future_offline_label_only"]["5"]["new_fault_onset"]
                              for r in records),
        "onset_within_20": sum(r["future_offline_label_only"]["20"]["new_fault_onset"]
                               for r in records),
        "currently_device_unavailable": sum(bool(r["future_offline_label_only"]["current_down_stations"])
                                             for r in records),
        "share_of_rule_to_perfect20_improvement_from_real_pre_fault_onset_20": (
            sum(r["method_regrets"]["Rule-5"]["regret_steps"] -
                r["method_regrets"]["Perfect-State-20"]["regret_steps"]
                for r in records if r["future_offline_label_only"]["20"]["group"] ==
                    "2_future_fault_from_currently_normal") /
            sum(r["method_regrets"]["Rule-5"]["regret_steps"] -
                r["method_regrets"]["Perfect-State-20"]["regret_steps"]
                for r in records)
        ),
        "pre_fault_robustness": {
            "count": sum(r["robustness_original_pr16_if_available"] is not None
                         for r in records if r["future_offline_label_only"]["20"]["group"] ==
                             "2_future_fault_from_currently_normal"),
            "nominal_best_stays_best": sum(
                r["robustness_original_pr16_if_available"]["nominal_best_still_robust_best"]
                for r in records if r["future_offline_label_only"]["20"]["group"] ==
                    "2_future_fault_from_currently_normal"
                and r["robustness_original_pr16_if_available"] is not None),
            "mean_robust_rule_opportunity": mean([
                r["robustness_original_pr16_if_available"]["robust_rule_opportunity"]
                for r in records if r["future_offline_label_only"]["20"]["group"] ==
                    "2_future_fault_from_currently_normal"
                and r["robustness_original_pr16_if_available"] is not None
            ]) if any(r["future_offline_label_only"]["20"]["group"] ==
                     "2_future_fault_from_currently_normal"
                     and r["robustness_original_pr16_if_available"] is not None
                     for r in records) else None,
        },
    }
    if totals["candidates"] != 174:
        raise ValueError("candidate count diverges from PR16")
    return records, {
        "input_protocol": "PR16 frozen 24 scenes / 8 map-seed clusters / 48 decisions",
        "online_observable_fields": records[0]["available_online"]["observation_top_level_fields"],
        "station_fields": records[0]["available_online"]["station_fields"],
        "robot_fields": records[0]["available_online"]["robot_fields"],
        "sample_fields": records[0]["available_online"]["sample_fields"],
        "recovery_snapshot_fields": records[0]["available_online"]["recovery_snapshot_fields"],
        "counts": totals,
        "by_5_step_fault_visibility": cluster_group_statistics(records, 5),
        "by_20_step_fault_visibility": cluster_group_statistics(records, 20),
        "observability_warning": (
            "Fault onset labels derived solely from realized post-action state changes. "
            "Neither future fault schedule nor realized stochastic service times are "
            "inputs to the currently deployed models. Association is not prognosis."
        ),
    }


def fmt(value):
    return "无" if value is None else f"{value:.4f}"


def build_report(records, summary):
    counts = summary["counts"]
    headings = [
        "# PR #17：世界模型可观测信息与故障预测价值审计",
        "",
        "## 证据范围与边界",
        "",
        f"只读取 PR #16 的 {counts['scenes']} 个确认场景、{counts['clusters']} 个地图/种子簇、"
        f"{counts['decisions']} 个决策、{counts['candidates']} 个原始 CP-SAT 候选，"
        f"以及原有 {counts['robustness_overlap']} 个额外服务随机性状态。",
        "不训练、不重放、不生成新实验。未来真实状态仅用于事后标注。"
        "完全相同的故障发生/恢复时刻在所有候选间一致的检验结果："
        + ("通过。" if counts["outage_timeline_consistent_across_all_candidates"] else "不通过。"),
        "",
        "## 在线信息 A、公开规则 B 与未知未来 C",
        "",
        "**A：当前可见。** 机器人位置、任务/阶段/服务进行状态，"
        "工作站 device_available、资源容量占用、capacity_blocked、"
        "当前 tick 与已公开的工作流状态。当前设备可用性必须读取 "
        "device_available，不能用受资源满载影响的 available 替代。"
        "状态文件实际字段名见机器可读 summary。",
        "",
        "**B：公开确定。** 地图/工作站位置、资源容量及分组、工作流的"
        "多阶段路径和已经公开的任务元数据。公开工作流模板不代表能知道未来外部任务的到达时刻。",
        "",
        "**C：事后才知道。** 未发生的设备故障和恢复、真实服务持续时间、"
        "尚未公开的外部任务；不得从 case assumptions、后台故障日程、"
        "trace_first20 或反事实结果提取在线预测特征。",
        "",
        "事件口径：对每个决策的当前站点状态与每条原始候选后续 20 个连续真实状态比较，"
        "仅 device_available 从 true 转为 false 才计作新故障；"
        "已经不可用的设备单列。记录故障绝对 tick 与距离决策的 offset。"
        "分别按 5 和 20 步窗口分组；故障场景名称不是状态标签。",
        "",
        f"48 个状态中，决策时已有设备不可用 {counts['currently_device_unavailable']} 个；"
        f"未来 5 步内新发生故障 {counts['onset_within_5']} 个，"
        f"未来 20 步内新发生故障 {counts['onset_within_20']} 个。",
        "这 8 个真实故障前决策均发生在 tick=0，故障在 tick=10 开始（距离决策 10 步）。"
        if len(counts["pre_fault_20_states"]) == 8 and
           all(x["tick"] == 0 and x["first_fault_offset"] == 10
               for x in counts["pre_fault_20_states"])
        else "故障时刻与决策距离详见逐决策记录。",
    ]
    for horizon in (5, 20):
        headings += [
            "", f"## 真实未来 {horizon} 步分层（描述性，非独立样本）", "",
            "| 当前状态/真实事件类别 | 决策数 | 涉及簇 | Rule-5 | Perfect-State-5 | Perfect-State-20 | Oracle 机会空间 | 候选差距 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for group, item in summary[f"by_{horizon}_step_fault_visibility"].items():
            headings.append(
                f"| {group} | {item['decisions']} | {item['clusters_with_group']} | "
                f"{fmt(item['methods']['Rule-5']['cluster_mean_equal_weight'])} | "
                f"{fmt(item['methods']['Perfect-State-5']['cluster_mean_equal_weight'])} | "
                f"{fmt(item['methods']['Perfect-State-20']['cluster_mean_equal_weight'])} | "
                f"{fmt(item['mean_oracle_opportunity_vs_rule']['cluster_mean_equal_weight'])} | "
                f"{fmt(item['mean_candidate_spread']['cluster_mean_equal_weight'])} |"
            )
        headings += [
            "",
            "表中数值为先在每个地图/种子簇内对属于该组的状态求均值，"
            "再对出现该组的簇等权平均。各组覆盖簇数可能不同，"
            "不能将不同组的均值当作随机化因果估计。"
            "Full-outcome Oracle 的决策损失按定义为 0，"
            "表中 Oracle 机会空间是 Rule-5 减去该 Oracle 的损失。"
        ]
    headings += [
        "", "## 事后机会由哪些真实事件贡献", "",
        f"按实际故障跃迁而不是场景标签分层，20 步内将发生故障的 8 个状态贡献了 "
        f"{counts['share_of_rule_to_perfect20_improvement_from_real_pre_fault_onset_20']*100:.2f}% "
        "的 Rule-5 到 Perfect-State-20 总损失下降。"
        "这个贡献说明事后未来故障信息对候选排序影响大，"
        "绝不能把该百分比解读为可提前预测的模型收益。",
        f"这些状态在原 PR #16 三个额外未来服务扰动种子的稳健性记录中覆盖 "
        f"{counts['pre_fault_robustness']['count']} 个，"
        f"事后最优候选保持最优 "
        f"{counts['pre_fault_robustness']['nominal_best_stays_best']} 个；"
        "条件子集的平均可得机会为 "
        f"{fmt(counts['pre_fault_robustness']['mean_robust_rule_opportunity'])} 步。"
        "该稳健性仅改变后续服务随机数，不改变外生故障日程；"
        "因此既不验证故障预测能力，也不代表全 48 个状态的随机重复均值。",
    ]
    observed_histories = sorted({f for r in records for f in
                                 r["available_online"]["explicit_history_fields"]})
    station_history_names = [
        "observed_effective_service_count", "observed_effective_service_mean",
        "history_complete_since_start", "expected_service_ticks"
    ]
    pre_fault = counts["pre_fault_20_states"]
    empty_pre_fault = bool(pre_fault) and all(
        x["active_service_robots"] == 0 and x["occupied_capacity_total"] == 0 and
        x["observed_effective_service_count_total"] == 0 for x in pre_fault
    )
    headings += [
        "", "## 能否从现有在线信息预测未来设备故障？", "",
        "**故障发生前的设备状态变化：** 当前观察字段只有设备可用/不可用状态，"
        "没有预设的健康分数、错误码、预警事件或故障概率。"
        "未来状态中出现的设备状态跃迁，只能在其真正发生时被识别，"
        "不能回填到之前的决策时刻。",
        "",
        "**服务历史与进度：** 当前机器人可见 service_elapsed、service_active_elapsed"
        "和 service_paused_elapsed；工作站还暴露 observed_effective_service_count、"
        "observed_effective_service_mean、history_complete_since_start 与"
        " expected_service_ticks。这些是历史服务统计/预期，**并非设备健康预警**。"
        f"顶层含 history/event/log 字样的字段：{', '.join(observed_histories) if observed_histories else '无'}。",
        ("8 个真正发生未来故障的决策均位于 tick=0，"
         "此时服务历史计数、活动服务机器人、资源占用全为 0，"
         "不能从已有服务历史推断十步后的设备故障。"
         if empty_pre_fault else
         "故障前观测包含的服务历史见逐决策记录；没有证据确认其可预测外生故障。"),
        "当前状态中的服务历史统计可以帮助分析一般服务时间，但现有材料"
        "没有设备健康退化与未来故障的标注对应关系，不能由服务时间变长"
        "直接推定未来故障。",
        "",
        "**资源占用与负载：** resource_occupied 和 capacity_blocked"
        "表征当下调度/队列拥堵，与未来设备故障预兆不是同一变量。"
        + ("8 个故障前 tick=0 的状态资源占用全部为 0。"
           if empty_pre_fault else "") +
        "即便有限样本出现相关性，也可能来自固定场景/时刻配置，"
        "不能称为可部署的故障风险模型证据。",
        "",
        "**故障来源与信息泄漏：** PR #16 记录的是仿真中外部故障日程触发的真实状态，"
        "未发生事件对线上规则与模型不公开。"
        "Perfect-State-20 直接使用已经实现的未来，因此其优势不能等价为"
        "当前观测可推断的未来。不能把后台 outage 时间表、"
        "未来服务随机数或事后标签当成学习特征。",
        "",
        "样本只有 8 个独立地图/种子簇，按同一种子配对的三个业务场景不是独立故障史；"
        "既没有真实设备随时间退化数据，也没有观测-故障的前瞻性验证。"
        "因此本轮不计算伪造的故障分类准确率、AUC 或置信风险分数。",
        "",
        "## 最终裁决：B（没有足够可观测的提前预警信号）", "",
        "现有世界模型能够依据当前设备是否已经不可用、服务是否正在进行、"
        "资源是否拥堵等已发生事实调整调度，"
        "但**不能凭已有观测可靠地提前判断尚未发生的外生设备故障**。"
        "服务时长及后续任务的不确定性同样不能从事后 Oracle 优势直接倒推为在线可预测。",
        "",
        "若需要在真实实验室建立可验证的预警任务，最小新增数据应有："
        "设备 ID、带时间戳的可用/故障/恢复状态切换与错误码；"
        "设备运行时长、维护与告警记录；"
        "工位开始/结束服务的样本/设备/时间戳；"
        "当前队列长度、资源占用、任务到达时间及公开时刻；"
        "以及明确区分事件产生时间与调度器实际接收时间的日志。"
        "有条件再采集设备特定健康遥测（温度、电流、振动等），"
        "需按设备故障机理选择，不能虚构为现有可用字段。"
        "训练/验证须按时间和设备分组，并与只依靠设备当前状态的基线比较。",
        "",
        "**下一阶段：停止继续训练现有世界模型，不启动 PR #18，"
        "不修改故障情景，不重跑实验；只保留本次只读审计与数据需求。**",
        "",
        "## 原始审计记录", "",
        "逐决策 JSON Lines：examples/evidence/oracle-value-20261110/observability-audit.jsonl。",
        "机器可读分层汇总：examples/evidence/oracle-value-20261110/observability-audit-summary.json。",
        "可复查运行：python -m unittest discover -s experiments/oracle_value/tests -p 'test_observable_audit.py' -v "
        "&& python -m experiments.oracle_value.observable_audit。",
        "",
    ]
    return "\n".join(headings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "confirmation/raw.json.gz")
    parser.add_argument("--prior-summary", type=Path,
                        default=ROOT / "confirmation/summary.json")
    parser.add_argument("--robustness", type=Path,
                        default=ROOT / "robustness/raw.json")
    parser.add_argument("--rows", type=Path, default=ROWS)
    parser.add_argument("--summary", type=Path, default=SUMMARY)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args()
    with gzip.open(args.raw, "rt", encoding="utf-8") as file:
        raw = json.load(file)
    previous = json.loads(args.prior_summary.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness.read_text(encoding="utf-8"))
    records, summary = load_and_analyze(raw, previous, robustness)
    args.rows.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.rows.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                                 for r in records), encoding="utf-8")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False,
                                       sort_keys=True, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(build_report(records, summary), encoding="utf-8")
    print("OBSERVABILITY_AUDIT", json.dumps(summary["counts"], ensure_ascii=False))
    for horizon in (5, 20):
        print("GROUPS", horizon, json.dumps(
            {k: v["decisions"] for k, v in summary[f"by_{horizon}_step_fault_visibility"].items()},
            ensure_ascii=False))


if __name__ == "__main__":
    main()
