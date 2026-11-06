#!/usr/bin/env python3
"""Exercise upstream Lay2Graph services with a small, bounded MAPD scenario.

The upstream checkout is read-only from this harness: fixture subsets and all
reports are written to the supplied output directory. Run after the services
have been started by the repository's CI workflow.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def subset(source: Path, destination: Path, count: int) -> list:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) < count:
        raise ValueError(f"not enough records in {source}")
    result = data[:count]
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-seconds", type=int, default=600)
    parser.add_argument("--max-steps", type=int, default=1500)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = source / "ScenarioEvaluationServices/scenario_evaluation"
    scenario = evaluation / "evaluation_mapd/Warehouse_35x21"
    fixture = output / "fixtures"
    fixture.mkdir(exist_ok=True)
    base = fixture / "base-data.json"
    state = fixture / "system-state.json"
    orders = fixture / "orders.json"
    robots = subset(scenario / "base_data_files/base_data_file_warehouse_21_35_10_amr.json", base, 2)
    states = subset(scenario / "system_state_files/system_state_file_warehouse_35x21_grid_1_10_amr.json", state, 2)
    tasks = subset(scenario / "order_files/orders_scenario_warehouse_35x21_grid_1_20_seed_1_fq_1.json", orders, 6)
    if [r["amr_id"] for r in robots] != [s["amr_id"] for s in states]:
        raise AssertionError("robot base and initial state identifiers do not match")

    # The framework uses relative imports and paths from this directory.
    os.chdir(evaluation)
    sys.path.insert(0, str(evaluation))
    import requests

    # Upstream helper functions do not consistently check HTTP status or set
    # timeouts. Turn errors into explicit acceptance failures in this process.
    original_request = requests.sessions.Session.request

    def checked_request(session, method, url, **kwargs):
        kwargs.setdefault("timeout", (5, 90))
        response = original_request(session, method, url, **kwargs)
        response.raise_for_status()
        return response

    requests.sessions.Session.request = checked_request
    from api.api_config import set_service_urls
    from api import client_api as api
    from api.client_api_models import (
        CBSImprovementRequest, CollisionDetection, DurationEdgeRequest,
        FocalHeuristicRequest, HeuristicRequest, OrderInfoRequest,
        PathPlanningStrategy, SetParkingNodesRequest, StatisticFileNames,
        TaskAssignmentStrategy, WaitingTimeRequest,
    )
    from data.enums import PathPlanningHeuristic, PathPlanningStrategy as Planning, TaskAssignmentStrategy as Assignment
    from microservice_initialization.init_path_planning import set_lif_object_from_file_in_travel_time_service
    from microservice_initialization.init_scenario_orders import init_scenario_with_orders
    from microservice_initialization.init_system_and_base_state_management import (
        create_system_state_and_base_state_object_from_file, send_system_state_objects,
    )

    spec = importlib.util.spec_from_file_location("smoke_scenario_config", scenario / "evaluation_config.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load upstream scenario config")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    config.GRAPH, config.DISTANCE = "grid", 1
    config.NUMBER_AMR, config.NUMBER_OF_ORDERS = len(robots), len(tasks)
    config.get_base_data_file = lambda: str(base)
    config.get_system_state_file = lambda: str(state)
    config.get_orders_file = lambda: str(orders)
    set_service_urls()

    for reset in (api.reset_order_management, api.reset_time_measurement,
                  api.reset_task_assignment, api.reset_base_data_management,
                  api.reset_amr_simulation, api.reset_system_state_management):
        reset()
    api.set_name_of_statistic_files(StatisticFileNames(
        AMRStatisticDataFileName="/statistic_data/amr.csv",
        GraphStatisticDataFileName="/statistic_data/graph.csv",
    ))
    layout = set_lif_object_from_file_in_travel_time_service(config)
    system_objects, base_objects = create_system_state_and_base_state_object_from_file(config)
    if len(system_objects) != 2 or len(base_objects) != 2:
        raise AssertionError("expected exactly two simulated robots")
    send_system_state_objects(system_objects, base_objects)
    assignment = TaskAssignmentStrategy(dispatchingStrategy=Assignment.central)
    api.set_task_assignment_strategy_api(assignment)
    api.set_dispatching_strategy_in_system_state_management(assignment)
    api.set_dispatching_strategy_in_amr_simulation(assignment)
    api.set_collision_detection_api(CollisionDetection(collisionDetection=True))
    api.set_path_planning_strategy_api(PathPlanningStrategy(pathPlanningStrategy=Planning.cbs, boundary=1.0))
    # The upstream client's exclude_none serialization omits a None field,
    # while the upstream API requires the field to be present even for CBS.
    api.set_focal_heuristic(FocalHeuristicRequest(heuristic=config.FOCAL_HEURISTIC))
    api.set_heuristic_for_a_star(HeuristicRequest(heuristic=PathPlanningHeuristic.euclidean))
    api.set_cbs_improvement_cost_function(CBSImprovementRequest(useAdvancedCostFunction=False))
    api.set_waiting_time_in_simulation(WaitingTimeRequest(waitingTime=1))
    api.set_duration_edge_in_path_planning(DurationEdgeRequest(duration=1))
    next_id = int(init_scenario_with_orders(config))
    if next_id != len(tasks) + 1:
        raise AssertionError(f"unexpected next order ID: {next_id}")
    api.set_parking_nodes_for_layout(SetParkingNodesRequest(
        layoutId=layout.layoutId, parkingNodes=config.get_parking_nodes(config.GRAPH, config.DISTANCE),
    ))

    deadline = time.monotonic() + args.max_seconds
    last = None
    with (output / "steps.jsonl").open("w", encoding="utf-8") as step_log:
        for step in range(1, args.max_steps + 1):
            if time.monotonic() > deadline:
                raise TimeoutError(f"simulation exceeded {args.max_seconds}s")
            last = api.run_simulation_step().json()
            record = {"step": step, **last}
            step_log.write(json.dumps(record, ensure_ascii=False) + "\n")
            step_log.flush()
            if step == 1 or step % 20 == 0 or last.get("simulationFinish"):
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if last.get("simulationFinish"):
                break
    if not last or last.get("simulationFinish") is not True or last.get("percentageFinish") != 100:
        raise AssertionError(f"simulation did not complete all orders: {last}")

    order_ids = [str(i) for i in range(1, next_id)]
    order_results = api.get_order_info(OrderInfoRequest(orderIds=order_ids)).json()
    (output / "orders-result.json").write_text(json.dumps(order_results, indent=2), encoding="utf-8")
    by_id = {str(row["orderId"]): row for row in order_results}
    if set(by_id) != set(order_ids):
        raise AssertionError(f"unexpected order IDs: {sorted(by_id)}")
    if any(row.get("orderStatus") != "FINISHED" for row in by_id.values()):
        raise AssertionError(f"not all order-management states are FINISHED: {order_results}")
    used_robots = sorted({str(row.get("amrId")) for row in by_id.values() if row.get("amrId")})
    if len(used_robots) < 2:
        raise AssertionError(f"fewer than two robots completed orders: {used_robots}")
    driving = api.get_statistic_information_from_amr_simulation().json()
    summary = {"robots": [r["amr_id"] for r in robots], "order_count": len(tasks),
               "completed_robots": used_robots, "last_step": last, "driving_times": driving,
               "collision_detection_enabled": True,
               "scope": "small upstream warehouse scenario; no independent collision proof or laboratory validation"}
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("LAY2GRAPH_E2E_ACCEPTED " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
