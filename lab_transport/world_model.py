"""Action-conditioned, CPU-only forecasts from observations available so far."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .domain import Grid, Position, Sample


@dataclass(frozen=True)
class PublicStation:
    """Information available to the predictor, excluding future outages/noise."""
    position: Position
    prior_ticks: float


class WorldModel:
    """Predict sample pickup/completion and update service estimates online.

    This is a small interpretable baseline, not a trained neural world model.
    It models static shortest paths and an online station-duration mean; it does
    not simulate future traffic or know future disruptions.
    """

    def __init__(self, grid: Grid, stations: Mapping[str, PublicStation]):
        self.grid = grid
        self.stations = dict(stations)
        self._observed: dict[str, list[int]] = {station_id: [] for station_id in stations}

    def observe_service(self, station_id: str, duration: int) -> None:
        if station_id not in self._observed or duration < 1:
            raise ValueError("invalid station service observation")
        self._observed[station_id].append(duration)

    def expected_service(self, station_id: str) -> float:
        values = self._observed[station_id]
        return sum(values) / len(values) if values else self.stations[station_id].prior_ticks

    def observations(self) -> dict[str, list[int]]:
        return {station_id: list(values) for station_id, values in self._observed.items()}

    def predict(self, tick: int, robot_id: str, position: Position, sample: Sample,
                station_busy: Mapping[str, int] | None = None) -> dict:
        """Forecast consequences of assigning one available robot to one sample."""
        pickup, dropoff = self.stations[sample.pickup], self.stations[sample.dropoff]
        to_pickup = self.grid.shortest_path(position, pickup.position)
        to_dropoff = self.grid.shortest_path(pickup.position, dropoff.position)
        if to_pickup is None or to_dropoff is None:
            raise ValueError("candidate is unreachable on the static map")
        occupied = station_busy or {}
        # Occupancy is a current observation, not knowledge of future arrivals.
        pickup_wait = max(0.0, occupied.get(sample.pickup, 0) - len(to_pickup))
        pickup_at = tick + len(to_pickup) + pickup_wait + self.expected_service(sample.pickup)
        drop_wait = max(0.0, occupied.get(sample.dropoff, 0) - (pickup_at - tick + len(to_dropoff)))
        completion = pickup_at + len(to_dropoff) + drop_wait + self.expected_service(sample.dropoff)
        late = max(0.0, completion - sample.deadline_tick)
        return {
            "robot_id": robot_id, "sample_id": sample.id,
            "expected_pickup_tick": round(pickup_at, 3),
            "expected_completion_tick": round(completion, 3),
            "expected_pickup_wait": round(pickup_wait, 3),
            "expected_dropoff_wait": round(drop_wait, 3),
            "predicted_final_robot_position": list(dropoff.position),
            "predicted_final_sample_status": "completed",
            "predicted_lateness": round(late, 3),
            "score": round(completion + 4 * late, 3),
        }
