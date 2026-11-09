from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station
from lab_transport.dynamics_model import Geometry
from lab_transport.simulation import Simulator
from lab_transport.state_v2 import ObservableHistory

try:
    import torch
except ModuleNotFoundError:
    torch = None


def outage_case():
    return Scenario('unit', Grid(5, 2, frozenset()),
        {'A': Station('A', (0, 0), 4, 4, ((1, 5),), 0),
         'B': Station('B', (3, 0), 1, 1, (), 0)},
        (RobotSpec('R1', (0, 0), (0, 0)),), (Sample('S1', 'A', 'B', 0, None),), 0, 80)


class ObservableStateTests(unittest.TestCase):
    def test_fault_pause_is_separate_from_active_and_completed_history(self):
        rows = []
        Simulator(outage_case(), 'nearest', 1).run(on_transition=rows.append)
        tracker = ObservableHistory()
        states = [tracker.observe(row.observation) for row in rows]
        state = next(x for x in states if x.tick == 5)
        robot = state.robots[0]
        self.assertEqual((robot.service_elapsed, robot.service_active_elapsed, robot.service_paused_elapsed), (5, 1, 4))
        self.assertEqual(state.stations[0].service_task_id, 'S1')
        paused = next(x for x in states if x.tick == 3)
        self.assertTrue(paused.stations[0].service_session_active)
        self.assertFalse(paused.stations[0].service_in_progress)
        after = next(x for x in states if x.tick == 8)
        self.assertEqual(after.stations[0].observed_effective_service_count, 1)
        self.assertEqual(after.stations[0].observed_effective_service_mean, 4)
        self.assertEqual(after.stations[0].expected_service_ticks, 8)  # legacy wall-time prior
        self.assertNotIn('service_remaining', json.dumps(asdict(state)))

    def test_history_gap_never_fabricates_progress(self):
        rows = []
        Simulator(outage_case(), 'nearest', 1).run(on_transition=rows.append)
        row = next(x for x in rows if x.observation.tick == 5)
        tracker = ObservableHistory()
        rich = tracker.observe(row.observation)
        self.assertIsNone(rich.robots[0].service_active_elapsed)
        self.assertIsNone(rich.robots[0].service_paused_elapsed)
        self.assertIsNone(rich.robots[0].operational_available)
        self.assertFalse(rich.stations[0].history_complete_since_start)
        self.assertEqual(rich, tracker.observe(row.observation))
        with self.assertRaises(ValueError):
            tracker.observe(replace(row.observation, tick=4))

    def test_model_observes_every_tick_without_changing_dispatch(self):
        normal = Simulator(outage_case(), 'nearest', 1).run()
        seen = []
        observed = Simulator(outage_case(), 'nearest', 1).run(on_observation=seen.append)
        self.assertEqual(normal['summary'], observed['summary'])
        self.assertEqual(normal['events'], observed['events'])
        self.assertEqual([x.tick for x in seen], list(range(len(seen))))


@unittest.skipIf(torch is None, 'PyTorch optional')
class StructuredModelTests(unittest.TestCase):
    def test_progress_aware_rule_does_not_finish_during_paused_time(self):
        from lab_transport.torch_structured_adapter import ProgressAwareRule
        rows = []
        case = outage_case()
        Simulator(case, 'nearest', 1).run(on_transition=rows.append)
        tracker = ObservableHistory()
        states = {row.observation.tick: tracker.observe(row.observation) for row in rows}
        row = next(x for x in rows if x.observation.tick == 5)
        predicted = ProgressAwareRule(Geometry(case.grid)).predict_next(states[5], row.assignments)
        self.assertEqual(predicted.samples[0].status, 'assigned')
        self.assertEqual((predicted.robots[0].service_elapsed, predicted.robots[0].service_active_elapsed,
                          predicted.robots[0].service_paused_elapsed), (6, 2, 4))

    def test_future_labels_do_not_enter_features_and_actions_do(self):
        from lab_transport.torch_structured_adapter import encode_transition, entity_tokens
        from lab_transport.transition_schema import Assignment
        rows = []
        case = outage_case()
        Simulator(case, 'nearest', 1).run(on_transition=rows.append)
        row = rows[0]
        obs = ObservableHistory().observe(row.observation)
        altered = replace(row.next_observation,
            samples=row.next_observation.samples + (replace(obs.samples[0], id='future-only',
                                                            release_tick=row.next_observation.tick),),
            stations=tuple(replace(s, available=not s.available) for s in row.next_observation.stations))
        first = encode_transition(Geometry(case.grid), obs, row.assignments, row.next_observation)
        second = encode_transition(Geometry(case.grid), obs, row.assignments, altered)
        self.assertEqual([(i['kind'], i['input'], i['entities']) for i in first],
                         [(i['kind'], i['input'], i['entities']) for i in second])
        self.assertNotEqual(entity_tokens(obs, ()), entity_tokens(obs, (Assignment('R1', 'S1'),)))

    def test_variable_entity_padding_and_permutation_invariance(self):
        from lab_transport.torch_structured_adapter import build_model, collate, encode_transition
        rows = []
        case = outage_case()
        Simulator(case, 'nearest', 1).run(on_transition=rows.append)
        row = rows[0]
        obs = ObservableHistory().observe(row.observation)
        one = encode_transition(Geometry(case.grid), obs, row.assignments, row.next_observation)[0]
        two = dict(one, entities=list(reversed(one['entities'])) + [one['entities'][0]])
        inputs, _ = collate([one, two])
        network = build_model().eval()
        context = network.context(inputs['motion_entities'], inputs['motion_entities_mask']) if one['kind'] == 'motion' else network.context(inputs['service_entities'], inputs['service_entities_mask'])
        permuted = dict(one, entities=list(reversed(one['entities'])))
        x, _ = collate([one, permuted])
        kind = one['kind']
        contexts = network.context(x[kind + '_entities'], x[kind + '_entities_mask'])
        self.assertTrue(torch.allclose(contexts[0], contexts[1], atol=1e-6))
        self.assertEqual(context.shape[0], 2)

    def test_cpu_training_resume_evaluation_and_closed_loop(self):
        from test_training_data import grid_fixture
        from lab_transport.multisite_experiment import make_case
        from lab_transport.torch_training import train, load_checkpoint
        from lab_transport.torch_evaluate import evaluate
        from lab_transport.torch_rollout import run
        from lab_transport.training_data import export_dataset
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = grid_fixture()
            dataset = root / 'data'
            export_dataset({'tartu': base}, {'fixture': 'synthetic'}, dataset,
                {'train': ('tartu', (3000,)), 'validation': ('tartu', (4000,)), 'test_tartu': ('tartu', (9000,))})
            out = root / 'train'
            options = dict(dataset=dataset, output=out, adapter_name='lab_transport.torch_structured_adapter',
                           device_name='cpu', batch_size=32, train_limit=120, validation_limit=90)
            train(epochs=1, **options)
            before = load_checkpoint(out / 'epoch-0001.pt')
            train(epochs=2, resume=out / 'epoch-0001.pt', **options)
            after = load_checkpoint(out / 'epoch-0002.pt')
            self.assertTrue(any(not torch.equal(before['model_state'][k], after['model_state'][k])
                                for k in before['model_state']))
            straight_out = root / 'straight'
            train(epochs=2, **dict(options, output=straight_out))
            straight = load_checkpoint(straight_out / 'epoch-0002.pt')
            self.assertTrue(all(torch.equal(straight['model_state'][k], after['model_state'][k])
                                for k in after['model_state']))
            result = evaluate(dataset, out / 'epoch-0002.pt', root / 'eval', splits=('test_tartu',), limit_per_split=25)
            self.assertEqual(result['transitions_per_split'], {'test_tartu': 25})
            self.assertIn('progress_rule', result['summaries']['test_tartu']['one_step'])
            case = make_case(base, 'tartu', 11000, 'short_outage')
            rollout = run(out / 'epoch-0002.pt', root / 'rollout', case, seed=11000, budget=2)
            self.assertGreater(rollout['model_steps'], 0)
            self.assertEqual(rollout['fallbacks'], 0)
            self.assertEqual(rollout['discrete_safety_violations'], 0)

    def test_long_score_formula_ignores_future_arrivals(self):
        from lab_transport.long_score_experiment import long_score
        case = outage_case()
        simulator = Simulator(case, 'nearest', 1)
        simulator.samples['S1'].status = 'waiting'
        initial = simulator._observable_state(0)
        future = replace(initial, tick=1,
            samples=initial.samples + (replace(initial.samples[0], id='future-only', release_tick=1),))
        # 1 unfinished known task-tick + 3 travel steps + 4+1 service steps.
        self.assertEqual(long_score(initial, [future], Geometry(case.grid)), (9.0, 3))


if __name__ == '__main__':
    unittest.main()
