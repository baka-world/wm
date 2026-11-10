"""Compare repeated fixed experiments, excluding measured time and memory."""
import json
from pathlib import Path
import sys

MEASUREMENT_KEYS = {
    'wall_ms', 'cpu_ms', 'peak_rss_kb', 'planning_wall_ms', 'planning_cpu_ms',
    'planning_peak_rss_kb', 'planning_wall_mean_ms', 'planning_wall_p95_ms',
    'planning_total_wall_ms', 'planning_cpu_ms_known', 'planner_peak_rss_kb',
    'parent_process_peak_rss_kb_cumulative', 'parent_process_cpu_ms',
    'simulation_wall_ms', 'motion_total_wall_ms', 'motion_wall_mean_ms',
    'motion_wall_p95_ms', 'motion_wall_ms',
}


def stable(value):
    if isinstance(value, dict):
        return {key: stable(item) for key, item in value.items() if key not in MEASUREMENT_KEYS}
    if isinstance(value, list):
        return [stable(item) for item in value]
    return value


if __name__ == '__main__':
    first, second = (json.loads(Path(name).read_text()) for name in sys.argv[1:3])
    if stable(first) != stable(second):
        raise ValueError('repeated experiment differs beyond measured timing and memory')
    print(f'{Path(sys.argv[1]).name}: repeated outputs match except measured timing and memory')
