import argparse
import csv
import os
import random
import statistics
import sys
import time
import types

import numpy as np


try:
    import tqdm  # noqa: F401
except ModuleNotFoundError:
    tqdm_stub = types.ModuleType('tqdm')
    tqdm_stub.tqdm = lambda iterable, *args, **kwargs: iterable
    sys.modules['tqdm'] = tqdm_stub

from GA_SEAD_process import GA_SEAD as OriginalGA, InformationOfUAVs
from GA_SEAD_chaos_process import GA_SEAD as ChaosGA


class CountingOriginalGA(OriginalGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_count = 0

    def crossover_operator(self, wheel, population):
        self.generation_count += 1
        return super().crossover_operator(wheel, population)


class CountingChaosGA(ChaosGA):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generation_count = 0

    def crossover_operator(self, wheel, population):
        self.generation_count += 1
        return super().crossover_operator(wheel, population)


def build_sample_case():
    targets = [
        [3100, 2200], [500, 3700], [2300, 2500], [2000, 3900],
        [4450, 3600], [4630, 4780], [1400, 4500],
    ]
    uav_id = [1, 2, 3, 4, 5, 6]
    uav_type = [1, 2, 3, 1, 3, 2]
    cruise_speed = [70, 80, 90, 60, 100, 80]
    turning_radii = [200, 250, 300, 180, 300, 260]
    uav_state = [
        [1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3],
        [1800, 400, -20 * np.pi / 180], [2200, 280, 45 * np.pi / 180],
        [4740, 300, 140 * np.pi / 180],
    ]
    base = [
        [0, 0, -np.pi / 2], [0, 0, -np.pi / 2], [1000, 6000, np.pi / 2],
        [1000, 6000, np.pi / 2], [4000, 5500, np.pi / 3], [4000, 5500, np.pi / 3],
    ]
    return targets, InformationOfUAVs(uav_id, uav_type, uav_state, cruise_speed, turning_radii, base)


def evaluate_solution(ga, solution):
    fitness, mission_time, total_distance, penalty = ga.objectives_evaluation(solution)
    return {
        'fitness': float(fitness),
        'mission_time': float(mission_time),
        'total_distance': float(total_distance),
        'penalty': float(penalty),
    }


def run_once(kind, mode, args, seed):
    random.seed(seed)
    np.random.seed(seed)
    targets, uav_info = build_sample_case()

    if kind == 'original':
        ga = CountingOriginalGA(targets, population_size=args.population)
    else:
        ga = CountingChaosGA(
            targets,
            population_size=args.population,
            chaos_seed=(seed * 0.137) % 1.0,
            chaos_map=args.chaos_map,
            chaos_maps=args.chaos_maps.split(',') if args.chaos_maps else None,
            chaos_early_stop=not args.no_early_stop,
            early_stop_min_generations=args.early_stop_min_generations,
            early_stop_patience=args.early_stop_patience,
            early_stop_rel_tol=args.early_stop_rel_tol,
            early_stop_min_time_ratio=args.early_stop_min_time_ratio,
            filter_duplicates=args.filter_duplicates,
        )

    start = time.perf_counter()
    if mode == 'time':
        solution, _ = ga.run_GA_time_period_version(
            args.time_interval, uav_info, None, True, distributed=args.distributed
        )
    else:
        solution, _, convergence = ga.run_GA(args.iterations, uav_info, None, distributed=args.distributed)
        ga.generation_count = max(ga.generation_count, max(0, len(convergence) - 1))
    elapsed = time.perf_counter() - start

    metrics = evaluate_solution(ga, solution)
    return {
        'kind': kind,
        'mode': mode,
        'seed': seed,
        'elapsed_sec': elapsed,
        'generations': int(getattr(ga, 'generation_count', 0)),
        **metrics,
    }


def summarize(rows, mode):
    print(f'\n==== GA SPEED BENCHMARK: {mode.upper()} MODE ====')
    by_kind = {}
    for row in rows:
        by_kind.setdefault(row['kind'], []).append(row)

    for kind in ('original', 'chaos'):
        values = by_kind.get(kind, [])
        if not values:
            continue
        elapsed = [v['elapsed_sec'] for v in values]
        generations = [v['generations'] for v in values]
        mission_time = [v['mission_time'] for v in values]
        fitness = [v['fitness'] for v in values]
        print(
            f'{kind:8s} elapsed mean={statistics.mean(elapsed):.4f}s, '
            f'gen mean={statistics.mean(generations):.1f}, '
            f'mission_time mean={statistics.mean(mission_time):.3f}, '
            f'fitness mean={statistics.mean(fitness):.8f}, n={len(values)}'
        )

    if 'original' in by_kind and 'chaos' in by_kind:
        pairs = zip(by_kind['original'], by_kind['chaos'])
        speedups = []
        mission_delta = []
        for original, chaos in pairs:
            if chaos['elapsed_sec'] > 0:
                speedups.append(original['elapsed_sec'] / chaos['elapsed_sec'])
            mission_delta.append(chaos['mission_time'] - original['mission_time'])
        if speedups:
            print(f'paired speedup original/chaos mean={statistics.mean(speedups):.3f}x')
        if mission_delta:
            print(f'paired mission_time delta chaos-original mean={statistics.mean(mission_delta):.3f}s')


def write_csv(rows, output_path):
    if not rows:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCSV saved to: {output_path}')


def parse_args():
    parser = argparse.ArgumentParser(description='Compare original GA_SEAD and chaos GA_SEAD runtime.')
    parser.add_argument('--mode', choices=['time', 'iter', 'both'], default='both')
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=20260603)
    parser.add_argument('--population', type=int, default=120)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--time-interval', type=float, default=0.5)
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--chaos-map', default='piecewise')
    parser.add_argument('--chaos-maps', default='')
    parser.add_argument('--no-early-stop', action='store_true')
    parser.add_argument('--early-stop-min-generations', type=int, default=5)
    parser.add_argument('--early-stop-patience', type=int, default=8)
    parser.add_argument('--early-stop-rel-tol', type=float, default=1e-4)
    parser.add_argument('--early-stop-min-time-ratio', type=float, default=0.20)
    parser.add_argument('--filter-duplicates', action='store_true')
    parser.add_argument('--output', default=os.path.join('paperCode', 'ga_speed_benchmark_results.csv'))
    return parser.parse_args()


def main():
    args = parse_args()
    modes = ['time', 'iter'] if args.mode == 'both' else [args.mode]
    all_rows = []

    for mode in modes:
        mode_rows = []
        for i in range(args.runs):
            seed = args.seed + i
            original = run_once('original', mode, args, seed)
            chaos = run_once('chaos', mode, args, seed)
            mode_rows.extend([original, chaos])
            print(
                f'run={i + 1:02d} mode={mode} '
                f'original={original["elapsed_sec"]:.4f}s/{original["generations"]}gen '
                f'chaos={chaos["elapsed_sec"]:.4f}s/{chaos["generations"]}gen '
                f'speedup={original["elapsed_sec"] / max(chaos["elapsed_sec"], 1e-12):.3f}x '
                f'mission_delta={chaos["mission_time"] - original["mission_time"]:.3f}s'
            )
        summarize(mode_rows, mode)
        all_rows.extend(mode_rows)

    write_csv(all_rows, args.output)


if __name__ == '__main__':
    main()
