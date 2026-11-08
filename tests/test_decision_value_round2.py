import unittest

from lab_transport.decision_value_round2 import (bootstrap_paired, contention_proxy, paired_incident,
                                                   pairwise_loss, select_index, service_sessions, spearman)
from lab_transport.domain import Grid, RobotSpec, Sample, Scenario, Station
from lab_transport.simulation import Simulator
from lab_transport.transition_schema import Assignment


class Round2Tests(unittest.TestCase):
    def make_case(self, outage_a=(), outage_b=(), robots=1):
        grid = Grid(5, 2, frozenset())
        stations = {'A': Station('A', (0, 0), 4, 4, outage_a, 0),
                    'B': Station('B', (3, 0), 1, 1, outage_b, 0)}
        fleet = (RobotSpec('R1', (0, 0), (0, 0)),)
        samples = (Sample('S1', 'A', 'B', 0, None),)
        if robots == 2:
            fleet += (RobotSpec('R2', (1, 0), (1, 0)),)
            samples += (Sample('S2', 'A', 'B', 0, None),)
        return Scenario('unit', grid, stations, fleet, samples, 0, 80)

    def test_outage_during_service_counts_only_unavailable_ticks(self):
        fault = self.make_case(outage_a=((1, 4),))
        normal = self.make_case()
        a = Simulator(fault, 'nearest', 1, collect_metrics=True).run()
        b = Simulator(normal, 'nearest', 1, collect_metrics=True).run()
        incident = {'kind': 'station_outage', 'station_id': 'A', 'start': 1, 'end': 4}
        sessions = service_sessions(a, fault, incident)
        self.assertEqual(sum(x['unavailable_wait_ticks'] for x in sessions), 3)
        self.assertEqual(sum(x['interrupted_after_service_start'] for x in sessions), 1)
        matched = paired_incident(a, b, fault, incident)
        self.assertEqual(matched['affected_service_sample_count'], 1)
        self.assertEqual(matched['fault_vs_no_fault_completion_deltas'][0]['completion_delta_ticks'], 3)
        self.assertEqual(matched['service_wait_ticks'], 3)

    def test_arrival_during_outage_is_distinct_from_ordinary_service(self):
        case = self.make_case(outage_b=((6, 10),))
        result = Simulator(case, 'nearest', 1, collect_metrics=True).run()
        incident = {'kind': 'station_outage', 'station_id': 'B', 'start': 6, 'end': 10}
        sessions = service_sessions(result, case, incident)
        target = next(x for x in sessions if x['station_id'] == 'B')
        self.assertTrue(target['dropoff_arrived_during_outage'])
        self.assertGreater(target['unavailable_wait_ticks'], 0)
        self.assertFalse(target['interrupted_after_service_start'])

    def test_station_competition_is_an_occupancy_proxy(self):
        case = self.make_case(outage_a=((1, 4),), robots=2)
        result = Simulator(case, 'nearest', 1, collect_metrics=True).run()
        proxy = contention_proxy(result, case)
        self.assertGreater(proxy['occupied_target_robot_ticks'], 0)
        self.assertGreater(proxy['occupied_target_nonmoving_ticks'], 0)
        self.assertGreater(proxy['tasks_seeking_occupied_target'], 0)
        self.assertIn('not measured FIFO', proxy['scope'])

    def test_rank_correlations_and_ties(self):
        self.assertAlmostEqual(spearman([1,2,3], [3,2,1]), -1)
        self.assertAlmostEqual(spearman([1,2,3], [1,2,3]), 1)
        self.assertIsNone(spearman([1,1,1], [1,2,3]))
        self.assertEqual(pairwise_loss([1,1], [1,2])['rate'], .5)
        self.assertEqual(pairwise_loss([2,1], [1,2])['rate'], 1)
        self.assertIsNone(pairwise_loss([1,2], [4,4])['rate'])

    def test_bootstrap_treats_seed_as_a_cluster(self):
        values = [('map1', 1, 0), ('map1', 1, 10), ('map1', 2, 1), ('map2', 3, 3)]
        a = bootstrap_paired(values)
        self.assertEqual(a['clusters'], 3)
        self.assertAlmostEqual(a['mean'], 3)
        self.assertEqual(a, bootstrap_paired(values))
        self.assertIsNone(a['ci95'])  # one map stratum has only one independent seed
        extended = values + [('map2', 4, 4)]
        self.assertLessEqual(bootstrap_paired(extended)['ci95'][0], bootstrap_paired(extended)['ci95'][1])

    def test_tied_scores_prefer_assignment_but_strictly_better_wait_is_allowed(self):
        batches = [(), (Assignment('R1', 'S1'),)]
        self.assertEqual(select_index([(2, 0), (2, 0)], batches), 1)
        self.assertEqual(select_index([(1, 0), (2, 0)], batches), 0)


if __name__ == '__main__': unittest.main()
