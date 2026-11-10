"""Sample per-scene process-tree CPU/RSS separately from timed comparisons.

Percentages are interval averages normalized to one CPU core. Kernel clock
quantization and short-lived children can over- or underestimate their peak.
Official worker VmHWM remains separately available in paired.json. This never
changes the benchmark's policy or inputs.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
import time
import traceback

import psutil

from experiments.lacam3_recovery.benchmark import run_one, scenarios

INTERVAL_SECONDS = 0.02


def _run_case(pipe, business, seed, backend, pibt_library, lacam_worker):
    try:
        result = run_one(business, seed, backend, pibt_library, lacam_worker)
        pipe.send({'jobs_completed': result['jobs_completed'], 'jobs': result['jobs'],
                   'stages_completed': result['stages_completed'], 'stages': result['stages'],
                   'ticks': result['ticks'], 'status': result['status']})
    except BaseException:
        pipe.send({'error': traceback.format_exc()})
    finally:
        pipe.close()


def _sample(process):
    try:
        processes = [process] + process.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None
    cpu = 0.0
    rss = 0
    living = 0
    for member in processes:
        try:
            times = member.cpu_times()
            cpu += times.user + times.system
            # Account for finished, reaped grandchildren as well as living
            # ones; otherwise the 3–5 ms LaCAM worker can disappear unseen.
            cpu += getattr(times, 'children_user', 0) + getattr(times, 'children_system', 0)
            rss += member.memory_info().rss
            living += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return (cpu, rss, living) if living else None


def profile_case(context, business, seed, backend, pibt_library, lacam_worker):
    receive, send = context.Pipe(duplex=False)
    child = context.Process(target=_run_case, args=(send, business, seed, backend, pibt_library, lacam_worker))
    child.start()
    send.close()
    process = psutil.Process(child.pid)
    previous = None
    samples = 0
    peak_cpu = None
    peak_rss = 0
    started = time.perf_counter()
    while True:
        now = time.perf_counter()
        current = _sample(process)
        if current is not None:
            cpu, rss, _ = current
            peak_rss = max(peak_rss, rss)
            samples += 1
            if previous is not None and now > previous[0] and cpu >= previous[1]:
                usage = 100 * (cpu - previous[1]) / (now - previous[0])
                peak_cpu = usage if peak_cpu is None else max(peak_cpu, usage)
            previous = (now, cpu)
        if not child.is_alive():
            break
        if now - started > 120:
            child.terminate()
            child.join()
            raise TimeoutError('single resource-profile case exceeded 120 seconds')
        child.join(INTERVAL_SECONDS)
    child.join()
    if not receive.poll():
        raise RuntimeError(f'profile child exited {child.exitcode} without result')
    outcome = receive.recv()
    receive.close()
    if child.exitcode != 0 or 'error' in outcome:
        raise RuntimeError(outcome.get('error', f'profile child exited {child.exitcode}'))
    return {'outcome': outcome, 'sample_count': samples,
            'peak_sampled_cpu_percent_one_core': peak_cpu,
            'peak_sampled_process_tree_rss_kb': peak_rss / 1024 if peak_rss else None}


def run(source_dir, pibt_library, lacam_worker, paired_file, output):
    if output.exists():
        raise FileExistsError('refusing to overwrite resource-profile evidence')
    cases, _ = scenarios(source_dir)
    paired = json.loads(paired_file.read_text())['cases']
    context = multiprocessing.get_context('fork')
    records = []
    for layout, seed, regime, robots, business, _ in cases:
        baseline = next(c for c in paired if (c['layout'], c['seed'], c['regime'], c['robots']) ==
                        (layout, seed, regime, robots))
        methods = {}
        for backend in ('legacy', 'pibt', 'hybrid'):
            measured = profile_case(context, business, seed, backend, pibt_library, lacam_worker)
            expected = baseline['methods'][backend]
            if any(measured['outcome'][key] != expected[key] for key in measured['outcome']):
                raise ValueError(f'profile rerun changed outcome: {layout}, {seed}, {regime}, {backend}')
            methods[backend] = measured
        records.append({'layout': layout, 'seed': seed, 'regime': regime, 'robots': robots, 'methods': methods})
        print('profiled', layout, seed, regime, robots, flush=True)
    output.write_text(json.dumps({'sampling_interval_seconds': INTERVAL_SECONDS,
        'cpu_definition': 'maximum sampled interval mean of whole child process tree, 100% = one core; quantization and short-lived processes can over- or underestimate',
        'memory_definition': 'maximum sampled sum of process RSS; shared pages may be counted more than once',
        'purpose': 'separate replay for resource sampling; timed paired.json is unmodified',
        'cases': records}, ensure_ascii=False, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--pibt-library', type=Path, required=True)
    parser.add_argument('--lacam-worker', type=Path, required=True)
    parser.add_argument('--paired', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.source_dir, args.pibt_library, args.lacam_worker, args.paired, args.output)


if __name__ == '__main__':
    main()
