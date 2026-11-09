from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
import unittest

from lab_transport.decision_value import replay_branch
from lab_transport.dynamics_model import Geometry
from lab_transport.simulation import Simulator
from lab_transport.state_v2 import ObservableHistory
from lab_transport.transition_schema import Assignment
from test_structured_world_model import outage_case

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch optional')
class CounterfactualServiceTests(unittest.TestCase):
    def setup_case(self):
        case = outage_case()
        rows = []
        result = Simulator(case, 'nearest', 1, collect_metrics=True).run(on_transition=rows.append)
        history = ObservableHistory()
        rich = {r.observation.tick:history.observe(r.observation) for r in rows}
        return case, rows, result, rich

    def test_duration_is_derived_from_each_branch_and_excludes_paused_ticks(self):
        from experiments.service_value.counterfactual import actual_durations
        case, rows, result, _ = self.setup_case()
        result, branch = replay_branch(case, 1, {}, result, rows, 0, (Assignment('R1','S1'),))
        durations, missing = actual_durations(result, branch, {'S1'})
        self.assertEqual(missing, [])
        self.assertEqual(durations[('S1','to_pickup','A')], 4)
        self.assertEqual(durations[('S1','to_dropoff','B')], 1)
        self.assertNotIn(('future','to_pickup','A'), durations)

    def test_two_candidates_have_independently_replayed_service_events(self):
        from lab_transport.multisite_experiment import make_case
        from experiments.service_value.counterfactual import actual_durations
        from test_training_data import grid_fixture
        case = make_case(grid_fixture(), 'tartu', 25000, 'normal')
        rows = []
        factual = Simulator(case, 'world', 25000, collect_metrics=True).run(on_transition=rows.append)
        sample = rows[0].observation.samples[0].id
        results = []
        for robot in ('R1','R2'):
            result, branch = replay_branch(case, 25000, {}, factual, rows, 0, (Assignment(robot,sample),))
            starts = [e for e in result['events'] if e['type']=='service_started' and e['sample_id']==sample
                      and e['phase']=='to_pickup']
            self.assertEqual(starts[0]['robot_id'], robot)
            durations, missing = actual_durations(result, branch, {sample})
            self.assertEqual(missing, [])
            results.append(durations)
        self.assertEqual(results[0], results[1])  # keyed randomness, still separately observed branches

    def test_oracle_uses_actual_duration_but_does_not_publish_it_in_state(self):
        from experiments.service_value.counterfactual import OfflineServiceOracle, OracleUnavailable
        case, rows, _, rich = self.setup_case()
        row = next(r for r in rows if r.observation.tick == 5)
        obs = rich[5]
        actual = OfflineServiceOracle(Geometry(case.grid), {('S1','to_pickup','A'):4})
        predicted = actual.predict_next(obs, row.assignments)
        self.assertEqual(predicted.samples[0].status, 'assigned')
        self.assertEqual(predicted.stations[0].expected_service_ticks, obs.stations[0].expected_service_ticks)
        with self.assertRaises(OracleUnavailable):
            OfflineServiceOracle(Geometry(case.grid), {}).predict_next(obs, row.assignments)

    def test_original_three_step_score_matches_existing_policy(self):
        from experiments.service_value.counterfactual import original_score, rollout
        from lab_transport.torch_rollout import RolloutPolicy
        from lab_transport.torch_structured_adapter import ProgressAwareRule
        case, rows, _, rich = self.setup_case()
        initial = rich[0]
        action = Assignment('R1','S1')
        geo = Geometry(case.grid)
        trace, _ = rollout(ProgressAwareRule(geo), initial, action, geo)
        policy = RolloutPolicy(ProgressAwareRule(geo), geo, horizon=3, budget=8)
        policy(initial, (action,))
        self.assertEqual(original_score(initial, action, trace[2], geo), policy.log[0]['candidates_evaluated'][0]['score'])

    def test_pr9_checkpoint_is_compatible_without_retraining(self):
        from experiments.service_value.counterfactual import load_pr9_model
        model, info = load_pr9_model()
        self.assertEqual(info['threshold'], .6)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 4642)

    def test_workload_score_counts_past_time_once_and_ignores_future_tasks(self):
        from experiments.service_value.scoring import workload_components, workload_score
        case, _, _, rich = self.setup_case()
        initial = rich[0]
        trace = [replace(initial, tick=t) for t in (1,2,3)]
        geo = Geometry(case.grid)
        parts = workload_components(initial, trace, geo)
        self.assertEqual(parts['known_unfinished_ticks'], 3)
        self.assertEqual(parts['remaining_distance'], 3)
        self.assertEqual(parts['estimated_remaining_service'], 5)
        self.assertEqual(workload_score(initial, trace, geo)[0], 11)
        future = replace(trace[-1], samples=trace[-1].samples +
                         (replace(initial.samples[0], id='future', release_tick=1),))
        self.assertEqual(workload_score(initial, trace, geo), workload_score(initial, trace[:-1] + [future], geo))

    def test_workload_subtracts_observed_effective_service_progress(self):
        from experiments.service_value.scoring import workload_components
        from lab_transport.torch_structured_adapter import ProgressAwareRule
        case, rows, _, rich = self.setup_case()
        row = next(r for r in rows if r.observation.tick == 5)
        geo = Geometry(case.grid)
        predicted = ProgressAwareRule(geo).predict_next(rich[5], row.assignments)
        self.assertEqual(predicted.robots[0].service_active_elapsed, 2)
        parts = workload_components(rich[5], [predicted], geo)
        self.assertEqual(parts['estimated_remaining_service'], 3)
        self.assertEqual(parts['known_unfinished_ticks'], 1)

    def test_oracle_cannot_be_inserted_into_online_controller(self):
        from experiments.service_value.counterfactual import OfflineServiceOracle
        from experiments.service_value.online import OnlinePolicy
        case, _, _, _ = self.setup_case()
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            OnlinePolicy(OfflineServiceOracle(Geometry(case.grid), {}), Geometry(case.grid),
                         ObservableHistory(), 'workload')

    def test_online_original_score_matches_pr9_at_same_horizon(self):
        from experiments.service_value.online import OnlinePolicy
        from lab_transport.torch_rollout import RolloutPolicy
        from lab_transport.torch_structured_adapter import ProgressAwareRule
        case, rows, _, _ = self.setup_case()
        geo = Geometry(case.grid)
        action = Assignment('R1', 'S1')
        current = rows[0].observation
        a = OnlinePolicy(ProgressAwareRule(geo), geo, ObservableHistory(), 'original')
        b = RolloutPolicy(ProgressAwareRule(geo), geo, horizon=20, budget=8, observer=ObservableHistory())
        self.assertEqual(a(current, (action,)), b(current, (action,)))
        self.assertEqual(a.decisions[0]['evaluated'][0]['score'], b.log[0]['candidates_evaluated'][0]['score'])

    def test_aligned_rollout_uses_only_predicted_state_for_later_assignments(self):
        from lab_transport.decision_value import candidates_for, nearest_batch
        from lab_transport.decision_value_round2 import burst_case
        from experiments.service_value.aligned import aligned_rollout
        from lab_transport.torch_structured_adapter import ProgressAwareRule
        from test_training_data import grid_fixture
        case = burst_case(grid_fixture(),'tartu',25000)
        rows = []
        Simulator(case,'world',25000).run(on_transition=rows.append)
        initial = ObservableHistory().observe(rows[0].observation)
        geo = Geometry(case.grid)
        class Recording:
            def __init__(self):
                self.inner = ProgressAwareRule(geo)
                self.later = []
            def predict_next(self, observation, assignments):
                if observation.tick > initial.tick:
                    self.later.append(assignments)
                    self.assert_equal(assignments, nearest_batch(observation,candidates_for(observation),geo))
                return self.inner.predict_next(observation,assignments)
            def assert_equal(self, left, right):
                if left != right:
                    raise AssertionError('future assignments did not use the predicted state')
        recorder = Recording()
        aligned_rollout(recorder,initial,Assignment('R1',initial.samples[0].id),geo)
        self.assertTrue(any(recorder.later))

    def test_future_arrival_and_outage_do_not_change_online_score(self):
        from lab_transport.domain import Sample
        from experiments.service_value.online import OnlinePolicy
        from lab_transport.torch_structured_adapter import ProgressAwareRule
        case = outage_case()
        stations = dict(case.stations)
        stations['B'] = replace(stations['B'],outages=((10,20),))
        changed = replace(case,stations=stations,samples=case.samples + (Sample('later','A','B',10,None),))
        first,second = Simulator(case,'world',1),Simulator(changed,'world',1)
        first.samples['S1'].status = second.samples['S1'].status = 'waiting'
        obs1,obs2 = first._observable_state(0),second._observable_state(0)
        self.assertEqual(obs1,obs2)
        geo = Geometry(case.grid)
        a = OnlinePolicy(ProgressAwareRule(geo),geo,ObservableHistory(),'workload')
        b = OnlinePolicy(ProgressAwareRule(geo),geo,ObservableHistory(),'workload')
        action = (Assignment('R1','S1'),)
        self.assertEqual(a(obs1,action),b(obs2,action))
        self.assertEqual(a.decisions[0]['evaluated'][0]['score'],b.decisions[0]['evaluated'][0]['score'])


if __name__ == '__main__':
    unittest.main()
