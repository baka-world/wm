from dataclasses import asdict, replace
import json
from pathlib import Path
import unittest

from lab_transport.decision_value import (BUDGET, HORIZON, ModelPolicy, batch_candidates,
    candidates_for, factual_run, nearest_batch, replay_branch, trace_score, truth_trace)
from lab_transport.domain import DirectedGraph, Grid, RobotSpec, Sample, Scenario, Station, load_scenario
from lab_transport.dynamics_model import Geometry, HybridDynamicsModel
from lab_transport.multisite_experiment import discrete_safety_violations
from lab_transport.simulation import Simulator

ROOT = Path(__file__).resolve().parents[1]


class DecisionValueTests(unittest.TestCase):
    def setUp(self):
        self.case = load_scenario(ROOT / 'scenarios/laboratory_demo.json')
        self.geo = Geometry(self.case.grid)

    def test_batches_include_wait_and_obey_budget_and_disjointness(self):
        rows = []
        Simulator(self.case, 'world', 42).run(on_transition=rows.append)
        obs = rows[0].observation
        batches, total = batch_candidates(obs, candidates_for(obs), self.geo)
        self.assertEqual(batches[0], ())
        self.assertLessEqual(len(batches), BUDGET)
        self.assertGreaterEqual(total, len(batches))
        self.assertEqual(len(batches[1]), 2)
        for batch in batches:
            self.assertEqual(len(batch), len({a.robot_id for a in batch}))
            self.assertEqual(len(batch), len({a.sample_id for a in batch}))
        self.assertEqual(nearest_batch(obs, (), self.geo), ())

    def test_replay_prefix_and_independent_candidates_are_reproducible(self):
        original, rows = factual_run(self.case, 42, {})
        row = next(r for r in rows if r.observation.tick > 0 and candidates_for(r.observation))
        batches, _ = batch_candidates(row.observation, candidates_for(row.observation), self.geo)
        first, transitions = replay_branch(self.case, 42, {}, original, rows, row.observation.tick, batches[0])
        again, _ = replay_branch(self.case, 42, {}, original, rows, row.observation.tick, batches[0])
        self.assertEqual(first['history'], again['history'])
        self.assertEqual(first['history'][:row.observation.tick + 1], original['history'][:row.observation.tick + 1])
        trace, valid = truth_trace(row.observation, transitions)
        self.assertTrue(valid)
        self.assertEqual(len(trace), HORIZON)
        self.assertEqual(len(trace_score(row.observation, trace, self.geo)), 2)
        self.assertEqual(discrete_safety_violations(first), 0)
        if len(batches) > 1:
            other, _ = replay_branch(self.case, 42, {}, original, rows, row.observation.tick, batches[1])
            self.assertNotEqual(first['history'], other['history'])

    def test_future_information_not_given_to_model_policy(self):
        observations = []
        for service, future, outage in ((3, 'FUTURE_A', (100, 110)), (9, 'FUTURE_B', (200, 210))):
            case = Scenario('hidden', Grid(5, 2, frozenset()), {
                'A': Station('A', (0, 0), service, 5.0, (outage,), 0),
                'B': Station('B', (4, 0), 1, 1.0, (), 0),
            }, (RobotSpec('R1', (0, 0), (0, 1)), RobotSpec('R2', (2, 1), (2, 1))),
               (Sample('S', 'A', 'B', 0, None), Sample('S2', 'A', 'B', 1, None),
                Sample(future, 'A', 'B', 20, None)), 0, 4)
            rows = []
            Simulator(case, 'world', 7).run(on_transition=rows.append)
            observations.append(rows[1].observation)
        self.assertEqual(observations[0], observations[1])
        serialized = json.dumps(asdict(observations[0]))
        self.assertNotIn('service_remaining', serialized)
        self.assertNotIn('FUTURE_', serialized)
        self.assertTrue(candidates_for(observations[0]))
        geometry = Geometry(Grid(5, 2, frozenset()))
        decisions = []
        for obs in observations:
            policy = ModelPolicy(HybridDynamicsModel(geometry, mode='heuristic'), geometry)
            decisions.append(policy(obs, candidates_for(obs)))
        self.assertEqual(decisions[0], decisions[1])

    def test_pause_blocks_movement_and_default_is_unchanged(self):
        baseline = Simulator(self.case, 'world', 42).run()
        default = Simulator(self.case, 'world', 42, motion_pauses={}).run()
        self.assertEqual(baseline, default)
        paused = Simulator(self.case, 'world', 42, motion_pauses={'R1': ((0, 5),)}, collect_metrics=True).run()
        positions = [{r['id']: r['position'] for r in h['robots']}['R1'] for h in paused['history'][:6]]
        self.assertTrue(all(p == positions[0] for p in positions))
        self.assertGreater(paused['summary']['motion_pause_ticks'], 0)
        self.assertEqual(discrete_safety_violations(paused), 0)
        with self.assertRaises(ValueError):
            Simulator(self.case, motion_pauses={'R999': ((0, 3),)})
        with self.assertRaises(ValueError):
            Simulator(self.case, motion_pauses={'R1': ((5, 5),)})

    def test_recovery_cannot_move_a_paused_robot(self):
        nodes = {(i, 0): (float(i), 0.0) for i in range(3)}
        graph = DirectedGraph(nodes, (((0, 0), (1, 0)), ((1, 0), (0, 0)),
                                      ((1, 0), (2, 0)), ((2, 0), (1, 0))))
        case = Scenario('recovery pause', graph,
                        {'A': Station('A', (0, 0), 1, 1, ()),
                         'B': Station('B', (2, 0), 1, 1, ())},
                        (RobotSpec('R1', (1, 0), (0, 0)), RobotSpec('R2', (2, 0), (2, 0))),
                        (Sample('S', 'A', 'B', 0, None),), 0, 5)
        sim = Simulator(case, motion_pauses={'R1': ((0, 3),)}, collect_metrics=True)
        sim.samples['S'].status, sim.samples['S'].robot_id = 'assigned', 'R2'
        sim.robots['R2'].task_id, sim.robots['R2'].phase = 'S', 'to_pickup'
        for tick in range(3):
            sim._move(tick, set())
            self.assertEqual(sim.robots['R1'].position, (1, 0))
        self.assertGreater(sim.metrics['motion_pause_ticks'], 0)

    def test_censored_branch_is_not_treated_as_completed(self):
        case = replace(self.case, max_ticks=2)
        original, rows = factual_run(case, 42, {})
        actual, branch_rows = replay_branch(case, 42, {}, original, rows, 0, ())
        trace, valid = truth_trace(rows[0].observation, branch_rows)
        self.assertFalse(valid)
        self.assertLess(len(trace), HORIZON)
        self.assertEqual(actual['summary']['status'], 'timed_out')


if __name__ == '__main__':
    unittest.main()
