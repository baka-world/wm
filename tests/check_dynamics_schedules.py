"""Audit nominal schedule overlap on the two fixed source-backed maps."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_transport.clinic_map import make_clinic_scenario
from lab_transport.dynamics_experiment import SPLITS, audit_task_schedules
from lab_transport.dynamics_revision import SPLITS_V2
from lab_transport.hospital_case import make_scenario
from lab_transport.multisite_experiment import make_case, prepare_tartu
from lab_transport.simulation import Simulator

hospital, tiago, clinic, clinic_config = (Path(path).read_bytes() for path in sys.argv[1:5])
bases = {"tartu": prepare_tartu(make_scenario(hospital, seconds_per_tick=10, tiago_config=tiago)),
         "clinic": make_clinic_scenario(clinic, clinic_config)}
pilot = audit_task_schedules(bases, SPLITS)
revision = audit_task_schedules(bases, SPLITS_V2)
assert pilot["cross_split_duplicate_groups"] == 1
assert pilot["cross_split_task_duplicate_groups"] == 1
assert {(c["split"], c["seed"], c["regime"]) for c in pilot["cross_split_duplicates"][0]} == {
    ("train", 3001, "normal"), ("test_tartu", 5001, "normal")}
assert revision["cross_split_duplicate_groups"] == 0
assert revision["cross_split_task_duplicate_groups"] == 0 and revision["unique_task_patterns"] == 24
assert revision["scenario_patterns"] == revision["unique_patterns"] == 48
draws = []
for seed in (3001, 5001):
    scenario = make_case(bases["tartu"], "tartu", seed, "normal")
    simulator = Simulator(scenario, "world", seed)
    draws.append([(simulator._service_duration(s.id, "to_pickup", s.pickup),
                   simulator._service_duration(s.id, "to_dropoff", s.dropoff)) for s in scenario.samples])
assert draws[0] != draws[1]
assert sum(a == b for p, q in zip(*draws) for a, b in zip(p, q)) == 8
print('Schedule audit verified: pilot has one cross-split pattern; revised 48 patterns are disjoint')
