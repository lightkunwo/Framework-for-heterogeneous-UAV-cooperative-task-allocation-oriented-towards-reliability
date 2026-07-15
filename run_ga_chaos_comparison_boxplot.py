import argparse
import csv
import os
import random
import statistics
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import GA_SEAD_process as original_mod
import GA_SEAD_chaos_process as chaos_mod
from GA_SEAD_process import GA_SEAD as OriginalGA
from GA_SEAD_process import InformationOfUAVs
from GA_SEAD_chaos_process import GA_SEAD as ChaosGA


# Keep batch experiments quiet and avoid tqdm slowing terminal output.
original_mod.tqdm = lambda iterable: iterable
chaos_mod.tqdm = lambda iterable: iterable


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
    cost_value = 1.0 / fitness if fitness > 0 else float("inf")
    return {
        "fitness": float(fitness),
        "cost_value": float(cost_value),
        "mission_time": float(mission_time),
        "total_distance": float(total_distance),
        "penalty": float(penalty),
    }


def run_single_attempt(kind, seed, population_size, iterations, chaos_seed=None):
    random.seed(seed)
    np.random.seed(seed)
    targets, uav_info = build_sample_case()

    if kind == "original":
        ga = CountingOriginalGA(targets, population_size=population_size)
    else:
        ga = CountingChaosGA(
            targets,
            population_size=population_size,
            chaos_seed=chaos_seed if chaos_seed is not None else (seed * 0.137) % 1.0,
        )

    start = time.perf_counter()
    solution, _, convergence = ga.run_GA(iterations, uav_info, None, distributed=False)
    elapsed = time.perf_counter() - start

    row = {
        "seed": seed,
        "kind": kind,
        "elapsed_sec": elapsed,
        "generations": max(int(getattr(ga, "generation_count", 0)), max(0, len(convergence) - 1)),
    }
    row.update(evaluate_solution(ga, solution))
    return row


def run_once(kind, seed, population_size, iterations, chaos_offsets=None, restart_distance_weight=0.0):
    if kind == "original":
        row = run_single_attempt(kind, seed, population_size, iterations)
        row.update({"best_offset": "", "restart_count": 1, "selection_score": row["cost_value"]})
        return row

    offsets = chaos_offsets or [0.137]
    best_row = None
    total_elapsed = 0.0
    total_generations = 0
    for offset in offsets:
        row = run_single_attempt(
            "chaos",
            seed,
            population_size,
            iterations,
            chaos_seed=(seed * offset) % 1.0,
        )
        total_elapsed += row["elapsed_sec"]
        total_generations += row["generations"]
        row["best_offset"] = offset
        row["restart_count"] = len(offsets)
        row["selection_score"] = row["cost_value"] + restart_distance_weight * row["total_distance"]
        if best_row is None or row["selection_score"] < best_row["selection_score"]:
            best_row = row

    # Report the true wall-clock work of all restarts, not only the selected attempt.
    best_row["elapsed_sec"] = total_elapsed
    best_row["generations"] = total_generations
    return best_row


def write_rows(csv_path, rows):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    fieldnames = [
        "seed", "kind", "elapsed_sec", "generations",
        "fitness", "cost_value", "mission_time", "total_distance", "penalty",
        "best_offset", "restart_count", "selection_score",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def grouped(rows, metric):
    return [
        [row[metric] for row in rows if row["kind"] == "original"],
        [row[metric] for row in rows if row["kind"] == "chaos"],
    ]


def plot_single_metric(rows, metric, ylabel, title, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    labels = ["GA-SEAD", "Chaos-GA-SEAD"]
    data = grouped(rows, metric)

    plt.figure(figsize=(6.2, 4.2), dpi=180)
    plt.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "#2A6FBB", "markeredgecolor": "#2A6FBB", "markersize": 4},
        medianprops={"color": "#F28E2B", "linewidth": 1.4},
        boxprops={"linewidth": 1.1},
        whiskerprops={"linewidth": 1.1},
        capprops={"linewidth": 1.1},
        flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 4},
    )
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_summary(rows, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metrics = [
        ("cost_value", "Cost value", "Cost Value Comparison"),
        ("mission_time", "Mission time (s)", "Mission Time Comparison"),
        ("total_distance", "Total distance (m)", "Total Distance Comparison"),
        ("elapsed_sec", "Runtime (s)", "Runtime Comparison"),
    ]
    labels = ["GA-SEAD", "Chaos-GA-SEAD"]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=180)
    for ax, (metric, ylabel, title) in zip(axes.ravel(), metrics):
        for ax, (metric, ylabel, title) in zip(axes.ravel(), metrics):
            ax.boxplot(
                grouped(rows, metric),
                tick_labels=labels,
                showmeans=True,
                meanprops={"marker": "D", "markerfacecolor": "#2A6FBB", "markeredgecolor": "#2A6FBB", "markersize": 4},
                medianprops={"color": "#F28E2B", "linewidth": 1.4},
                boxprops={"linewidth": 1.1},
                whiskerprops={"linewidth": 1.1},
                capprops={"linewidth": 1.1},
                flierprops={"marker": "o", "markerfacecolor": "white", "markeredgecolor": "#333333", "markersize": 4},
            )
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def print_summary(rows):
    print("\n==== 10-RUN COMPARISON SUMMARY ====")
    for kind in ["original", "chaos"]:
        subset = [row for row in rows if row["kind"] == kind]
        print(
            f"{kind:8s} "
            f"cost={statistics.mean(row['cost_value'] for row in subset):.3f}, "
            f"mission={statistics.mean(row['mission_time'] for row in subset):.3f}, "
            f"distance={statistics.mean(row['total_distance'] for row in subset):.3f}, "
            f"runtime={statistics.mean(row['elapsed_sec'] for row in subset):.3f}, "
            f"n={len(subset)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Run 10 paired GA vs Chaos-GA experiments and draw boxplots.")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260603)
    parser.add_argument("--population", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--csv", default=os.path.join("analysis_outputs", "ga_compare_chaos_10runs_iter300.csv"))
    parser.add_argument("--cost-fig", default=os.path.join("analysis_outputs", "ga_compare_cost_boxplot_10runs.png"))
    parser.add_argument("--distance-fig", default=os.path.join("analysis_outputs", "ga_compare_distance_boxplot_10runs.png"))
    parser.add_argument("--summary-fig", default=os.path.join("analysis_outputs", "ga_compare_summary_boxplots_10runs.png"))
    parser.add_argument("--chaos-offsets", default="0.137,0.101,0.377")
    parser.add_argument("--restart-distance-weight", type=float, default=0.001)
    return parser.parse_args()


def main():
    args = parse_args()
    chaos_offsets = [float(item.strip()) for item in args.chaos_offsets.split(",") if item.strip()]
    rows = []
    write_rows(args.csv, rows)

    for index in range(args.runs):
        seed = args.seed + index
        for kind in ["original", "chaos"]:
            row = run_once(
                kind,
                seed,
                args.population,
                args.iterations,
                chaos_offsets=chaos_offsets,
                restart_distance_weight=args.restart_distance_weight,
            )
            rows.append(row)
            write_rows(args.csv, rows)
            print(
                f"run={index + 1:02d}/{args.runs} {kind:8s} "
                f"seed={seed} cost={row['cost_value']:.3f} "
                f"mission={row['mission_time']:.3f} "
                f"distance={row['total_distance']:.3f} "
                f"time={row['elapsed_sec']:.3f}s gen={row['generations']} "
                f"offset={row['best_offset']}",
                flush=True,
            )

    plot_single_metric(rows, "cost_value", "Cost value", "Cost Value Comparison", args.cost_fig)
    plot_single_metric(rows, "total_distance", "Total distance (m)", "Total Distance Comparison", args.distance_fig)
    plot_summary(rows, args.summary_fig)
    print_summary(rows)
    print(f"CSV saved to: {args.csv}")
    print(f"Cost boxplot saved to: {args.cost_fig}")
    print(f"Distance boxplot saved to: {args.distance_fig}")
    print(f"Summary boxplots saved to: {args.summary_fig}")


if __name__ == "__main__":
    main()
