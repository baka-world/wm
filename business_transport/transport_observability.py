"""Transport arrival and service admission *factual* event instrumentation.

Opt-in subclass: legacy BusinessSimulator traces and historical experiment
hashes stay untouched. Does not alter _move(), _service_duration(), CP-SAT,
robot states or hidden future. Arrival is first observed at the station before
admission; service admission is the existing service_started event.
"""
from __future__ import annotations

from business_transport.simulation import BusinessSimulator


class ObservableTransportSimulator(BusinessSimulator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._observed_arrivals = set()

    def _handle_station(self, tick, robot):
        if self.business is not None and robot.task_id is not None:
            sample = self.samples[robot.task_id]
            phase = robot.phase
            target = sample.spec.pickup if phase == "to_pickup" else sample.spec.dropoff
            key = (robot.id, robot.task_id, phase)
            if robot.position == self._target(robot) and key not in self._observed_arrivals:
                self._event(tick, "transport_arrived", robot_id=robot.id,
                            sample_id=robot.task_id, phase=phase,
                            station_id=target, observational=True)
                self._observed_arrivals.add(key)
        return super()._handle_station(tick, robot)
