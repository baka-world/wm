from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lab_transport.dynamics_model import Geometry
from lab_transport.simulation import Simulator
from lab_transport.state_v2 import ObservableHistory
from lab_transport.transition_schema import Transition
from test_structured_world_model import outage_case

try:
    import torch
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch optional')
class ServiceDynamicsTests(unittest.TestCase):
    def rows(self):
        raw = []
        case = outage_case()
        Simulator(case, 'nearest', 1).run(on_transition=raw.append)
        observer = ObservableHistory()
        rows = [Transition(observer.observe(r.observation), r.assignments,
                           observer.observe(r.next_observation), r.next_boundary, r.termination) for r in raw]
        return case, rows

    def test_three_step_label_censors_future_outage_and_counts_completion(self):
        from lab_transport.service_learning import sequence_target, service_keys
        _, rows = self.rows()
        self.assertIsNone(sequence_target(rows, 0, service_keys(rows[0])[0]))
        row = next(i for i, r in enumerate(rows) if r.observation.tick == 5)
        self.assertEqual(sequence_target(rows, row, service_keys(rows[row])[0]), 1)
        self.assertIsNone(sequence_target(rows[:row+1], row, service_keys(rows[row])[0]))

    def test_weight_and_auxiliary_loss_update_shared_network(self):
        from lab_transport.service_learning import loss_value, sequence_collate
        from lab_transport.torch_structured_adapter import build_model, encode_transition
        case, rows = self.rows()
        row = next(r for r in rows if r.observation.tick == 5)
        items = encode_transition(Geometry(case.grid), row.observation, row.assignments, row.next_observation)
        for item in items:
            if item['kind'] == 'service':
                item['sequence_target'] = 1
        inputs, labels = sequence_collate(items)
        model = build_model()
        loss, count = loss_value(model, inputs, labels, positive_weight=3, sequence_weight=.3)
        self.assertGreater(count, 0)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertTrue(any(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))

    def test_rule_motion_hybrid_uses_same_deterministic_positions(self):
        from lab_transport.service_dynamics import RuleMotionNeuralService
        from lab_transport.torch_structured_adapter import ProgressAwareRule, build_model
        case, rows = self.rows()
        geometry = Geometry(case.grid)
        hybrid = RuleMotionNeuralService(geometry, build_model(), torch.device('cpu'), .35)
        rule = ProgressAwareRule(geometry)
        for row in rows:
            a = hybrid.predict_next(row.observation, row.assignments)
            b = rule.predict_next(row.observation, row.assignments)
            self.assertEqual([r.position for r in a.robots], [r.position for r in b.robots])

    def test_hybrid_does_not_call_the_neural_movement_head(self):
        from lab_transport.service_dynamics import RuleMotionNeuralService
        from lab_transport.torch_structured_adapter import build_model
        case, rows = self.rows()
        network = build_model()
        class UnusedMotion(torch.nn.Module):
            def forward(self, value):
                raise AssertionError('neural movement was called')
        network.motion = UnusedMotion()
        hybrid = RuleMotionNeuralService(Geometry(case.grid), network, torch.device('cpu'))
        for row in rows:
            hybrid.predict_next(row.observation, row.assignments)

    def test_calibration_is_validation_only_and_ties_use_fixed_rule(self):
        from lab_transport.service_learning import calibrate
        result = calibrate([.1, .2, .8, .9], [0, 0, 1, 1])
        self.assertEqual(result['threshold'], .5)
        with self.assertRaisesRegex(ValueError, 'positive and negative'):
            calibrate([.1, .2], [0, 0])

    def test_future_service_labels_do_not_change_current_network_inputs(self):
        from lab_transport.torch_structured_adapter import encode_transition
        case, rows = self.rows()
        row = next(r for r in rows if r.observation.tick == 5)
        future = replace(row.next_observation,
            stations=tuple(replace(s, available=not s.available) for s in row.next_observation.stations),
            samples=row.next_observation.samples + (replace(row.observation.samples[0], id='future-new',
                                                             release_tick=row.observation.tick+1),))
        a = encode_transition(Geometry(case.grid), row.observation, row.assignments, row.next_observation)
        b = encode_transition(Geometry(case.grid), row.observation, row.assignments, future)
        self.assertEqual([(i['kind'], i['input'], i['entities']) for i in a],
                         [(i['kind'], i['input'], i['entities']) for i in b])

    def test_fault_export_separates_whole_scenarios_and_measures_actual_wait(self):
        from lab_transport.service_fault_data import export, select_seeds
        from lab_transport.training_data import validate_dataset
        from test_training_data import grid_fixture
        pools = {'train':('tartu',21000,1), 'validation':('tartu',23000,1),
                 'test_tartu':('tartu',25000,1)}
        bases = {'tartu':grid_fixture()}
        with patch('lab_transport.service_fault_data.POOLS', pools), \
             patch('lab_transport.service_fault_data.SPLITS_V2', {'old':('tartu',(3000,))}), \
             tempfile.TemporaryDirectory() as tmp:
            selected, _ = select_seeds(bases)
            self.assertEqual(len({seed for _, seeds in selected.values() for seed in seeds}), 3)
            root = Path(tmp)/'dataset'
            report = export(root, bases, {'fixture':'synthetic'})
            self.assertEqual(validate_dataset(root)['episodes'], 24)
            outages = [r for r in report['coverage'] if r['regime'] == 'outage_processing']
            self.assertTrue(all(r['service_wait_ticks'] > 0 for r in outages))
            self.assertTrue(all(r['pending_at_recovery'] > 0 for r in outages))
            self.assertTrue(all(r['service_wait_ticks'] == 0 for r in report['coverage'] if r['regime'] == 'normal'))
            with self.assertRaises(FileExistsError):
                export(root, bases, {'fixture':'synthetic'})


if __name__ == '__main__':
    unittest.main()
