"""DPGA meta-knowledge-transfer ablation experiment.

Code version: v4.0

This experiment disables reliability degradation, reinforcement-learning
decisions, subsystem selection, failure-triggered replanning, and flight-process
simulation. It compares plain DPGA island evolution with DPGA island evolution
that receives meta-knowledge injections between generations.
"""

import argparse
import random
import sys
import time
import types
from types import SimpleNamespace

import numpy as np


def install_dubins_fallback_if_needed():
    try:
        import dubins  # noqa: F401
        return
    except ModuleNotFoundError:
        from dubins_path import Dubins

    planner = Dubins()

    class LocalDubinsPath:
        def __init__(self, start, goal, turning_radius):
            self.start = [float(value) for value in start]
            self.goal = [float(value) for value in goal]
            self.turning_radius = max(1e-6, float(turning_radius))
            self.kappa = 1.0 / self.turning_radius

        def path_length(self):
            normalized_length = planner.path_length(self.start, self.goal, self.kappa)
            return float(normalized_length / self.kappa)

        def sample_many(self, step_size):
            path, _, _ = planner.plan(self.start, self.goal, self.kappa)
            if not path:
                configurations = [tuple(self.start), tuple(self.goal)]
            else:
                configurations = [tuple(self.start)] + [tuple(point) for point in path]
                configurations.append(tuple(self.goal))
            distances = [0.0]
            for index in range(1, len(configurations)):
                dx = configurations[index][0] - configurations[index - 1][0]
                dy = configurations[index][1] - configurations[index - 1][1]
                distances.append(distances[-1] + float(np.hypot(dx, dy)))
            return configurations, distances

    def shortest_path(start, goal, turning_radius):
        return LocalDubinsPath(start, goal, turning_radius)

    sys.modules["dubins"] = types.SimpleNamespace(shortest_path=shortest_path)


install_dubins_fallback_if_needed()

import secondDemo as sd


EXPERIMENT_VERSION = "v4.0"

TARGETS_SITES = [
    [6734, 1453], [2233, 10], [5530, 1424], [401, 841], [3082, 1644], [7608, 4458],
    [7573, 3716], [7265, 1268], [6898, 1885], [1112, 2049], [5468, 2606], [5989, 2873],
    [4706, 2674], [4612, 2035], [6347, 2683], [6107, 669], [7611, 5184], [7462, 3590],
    [7732, 4723], [5900, 3561], [4483, 3369], [6101, 1110], [5199, 2182], [1633, 2809],
    [4307, 2322], [675, 1006], [7555, 4819], [7541, 3981], [3177, 756], [7352, 4506],
    [7545, 2801], [3245, 3305], [6426, 3173], [4608, 1198], [23, 2216], [7248, 3779],
    [7762, 4595], [7392, 2244], [3484, 2829], [6271, 2135], [4985, 140], [1916, 1569],
    [7280, 4899], [7509, 3239], [10, 2676], [6807, 2993], [5185, 3258], [3023, 1942]
]


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))


def build_default_scenario():
    # Current experiment scale: 13 UAVs, type1=5, type2=4, type3=4.
    uav_id = list(range(1, 14))
    uav_type = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    cruise_speed = [70, 72, 74, 76, 78, 82, 84, 86, 88, 90, 92, 94, 96]
    turning_radii = [200, 205, 210, 215, 220, 245, 250, 255, 260, 290, 295, 300, 305]
    base_locations = [[2500.0, 4000.0, np.pi / 2] for _ in uav_id]
    initial_states = [state[:] for state in base_locations]
    return {
        "targets_sites": [target[:] for target in TARGETS_SITES],
        "uav_id": uav_id,
        "uav_type": uav_type,
        "cruise_speed": cruise_speed,
        "turning_radii": turning_radii,
        "initial_states": initial_states,
        "base_locations": base_locations,
    }


def build_source_states(initial_states, offset_radius):
    count = max(1, len(initial_states))
    source_states = []
    for index, state in enumerate(initial_states):
        angle = 2.0 * np.pi * index / count
        source_states.append([
            float(state[0]) + float(offset_radius) * np.cos(angle),
            float(state[1]) + float(offset_radius) * np.sin(angle),
            float(state[2]),
        ])
    return source_states


def build_uav_information(scenario, states):
    return sd.InformationOfUAVs(
        scenario["uav_id"],
        scenario["uav_type"],
        [state[:] for state in states],
        scenario["cruise_speed"],
        scenario["turning_radii"],
        [base[:] for base in scenario["base_locations"]],
        uav_best_solution=[],
        new_targets=[],
        tasks_completed=[],
    )


def configure_population_size(mission, population_size):
    population_size = int(population_size)
    if population_size <= mission.elitism_num:
        raise ValueError("population size must be larger than elitism_num")
    mission.population_size = population_size
    available = population_size - mission.elitism_num
    crossover_num = int(round(available * mission.crossover_prob))
    crossover_num = min(available, max(0, crossover_num))
    if crossover_num % 2 == 1:
        crossover_num -= 1
    mission.crossover_num = crossover_num
    mission.mutation_num = population_size - mission.elitism_num - mission.crossover_num


def clone_chromosome(chromosome):
    return [list(row) for row in chromosome]


def clone_population(mission, population):
    return [
        mission.Chromosome(clone_chromosome(individual.chromosome))
        for individual in population
    ]


def clone_islands(mission, islands):
    return [clone_population(mission, island) for island in islands]


def repair_transferred_headings(mission, population):
    max_heading = max(0, int(mission.heading_discretization) - 1)
    for individual in population:
        chromosome = getattr(individual, "chromosome", None)
        if not chromosome or len(chromosome) != 5:
            continue
        chromosome[4] = [
            int(np.clip(int(round(value)), 0, max_heading))
            for value in chromosome[4]
        ]


def evolve_one_generation(mission, population):
    if not population:
        raise ValueError("population must not be empty")
    mission.fitness_evaluation(population)
    wheel = mission.get_roulette_wheel(population)
    next_population = []
    next_population.extend(mission.elitism_operator(population))
    next_population.extend(mission.crossover_operator(wheel, population))
    next_population.extend(mission.mutation_operator(wheel, population))
    mission.fitness_evaluation(next_population)
    return next_population


def evolve_fixed_generations(mission, population, generations):
    mission.fitness_evaluation(population)
    for _ in range(int(generations)):
        population = evolve_one_generation(mission, population)
    return mission.find_best_solution(population), population


def ranked_population(population):
    return sorted(population, key=lambda individual: individual.fitness_value, reverse=True)


def dpga_ring_exchange(mission, islands, exchange_elites):
    if exchange_elites <= 0 or len(islands) <= 1:
        return 0

    for island in islands:
        mission.fitness_evaluation(island)

    elite_packets = []
    for island in islands:
        elite_packets.append([
            clone_chromosome(ind.chromosome)
            for ind in ranked_population(island)[:exchange_elites]
        ])

    replaced = 0
    for receiver_index, island in enumerate(islands):
        incoming = elite_packets[(receiver_index - 1) % len(islands)]
        worst_indices = sorted(range(len(island)), key=lambda idx: island[idx].fitness_value)
        for offset, chromosome in enumerate(incoming[:len(worst_indices)]):
            island[worst_indices[offset]] = mission.Chromosome(clone_chromosome(chromosome))
            replaced += 1
    return replaced


def solution_metrics(mission, solution):
    fitness, mission_time, total_distance, penalty = mission.objectives_evaluation(solution)
    return {
        "fitness": float(fitness),
        "objective_cost": float(1.0 / fitness),
        "mission_time_s": float(mission_time),
        "total_distance_m": float(total_distance),
        "sequence_penalty": float(penalty),
    }


def build_initial_islands(mission, args, seed):
    configure_population_size(mission, args.island_population_size)
    islands = []
    for island_index in range(args.island_count):
        set_seed(seed + 10_000 * (island_index + 1))
        population = mission.generate_population()
        mission.fitness_evaluation(population)
        islands.append(population)
    return islands


def prepare_meta_elites(mission, scenario, args):
    source_states = build_source_states(scenario["initial_states"], args.source_offset_radius)
    source_info = build_uav_information(scenario, source_states)
    mission.information_setting(source_info, None, distributed=False)

    configure_population_size(mission, args.meta_population_size)
    set_seed(args.meta_seed)
    source_population = mission.generate_population()
    _, source_population = evolve_fixed_generations(
        mission, source_population, args.source_generations
    )

    context = SimpleNamespace(uavs=[scenario["uav_id"], scenario["uav_type"]])
    transfer = sd.KnowledgeTransferManager(context, mission)
    return transfer.extract_elites(source_population, k=args.elite_count)


def inject_meta_knowledge(mission, islands, meta_elites, target_info, args, seed, generation):
    if not meta_elites or args.kt_ratio <= 0:
        return 0

    injected = 0
    target_info.scope_uav_ids = list(target_info.uav_id)
    target_info.subsystem_id = "DPGA-META-KT"
    context = SimpleNamespace(uavs=[target_info.uav_id, target_info.uav_type])

    for island_index, island in enumerate(islands):
        set_seed(seed + 500_000 + generation * 1000 + island_index)
        transfer = sd.KnowledgeTransferManager(context, mission)
        mission.fitness_evaluation(island)
        islands[island_index], applied, _ = transfer.inject_elites(
            island,
            meta_elites,
            args.kt_ratio,
            tuple(target_info.uav_id),
            "meta",
            target_info,
        )
        repair_transferred_headings(mission, islands[island_index])
        injected += int(applied)
    return injected


def run_dpga_branch(mission, initial_islands, meta_elites, target_info, seed, enable_kt, args):
    islands = clone_islands(mission, initial_islands)
    configure_population_size(mission, args.island_population_size)

    injected_total = 0
    exchange_total = 0
    start = time.perf_counter()

    for generation in range(1, args.generations + 1):
        if enable_kt and args.kt_interval > 0 and generation % args.kt_interval == 0:
            injected_total += inject_meta_knowledge(
                mission, islands, meta_elites, target_info, args, seed, generation
            )

        for island_index, island in enumerate(islands):
            set_seed(seed + 700_000 + generation * 10_000 + island_index)
            islands[island_index] = evolve_one_generation(mission, island)

        if args.exchange_interval > 0 and generation % args.exchange_interval == 0:
            exchange_total += dpga_ring_exchange(mission, islands, args.exchange_elites)

    for island in islands:
        mission.fitness_evaluation(island)
    best_solution = max(
        (individual for island in islands for individual in island),
        key=lambda individual: individual.fitness_value,
    )
    optimization_time = time.perf_counter() - start

    result = solution_metrics(mission, best_solution)
    result.update({
        "seed": int(seed),
        "kt_enabled": bool(enable_kt),
        "optimization_time_s": float(optimization_time),
        "injected_elites": int(injected_total),
        "dpga_exchanged_elites": int(exchange_total),
    })
    return result


def run_paired_trial(mission, scenario, target_info, meta_elites, seed, run_index, args):
    mission.information_setting(target_info, None, distributed=False)
    initial_islands = build_initial_islands(mission, args, seed)

    results = {}
    branch_order = [False, True] if run_index % 2 else [True, False]
    for enable_kt in branch_order:
        results[enable_kt] = run_dpga_branch(
            mission, initial_islands, meta_elites, target_info, seed, enable_kt, args
        )
    return results[False], results[True]


def finite_stats(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    std = np.std(values, ddof=1) if values.size > 1 else 0.0
    return float(np.mean(values)), float(std)


def paired_delta(rows, key):
    return [float(row_on[key] - row_off[key]) for row_off, row_on in rows]


def print_results(rows):
    print("\n==== DPGA META-KT PAIRED RESULTS (ON - OFF) ====")
    print(
        "seed | opt_off(s) | opt_on(s) | delta(s) | "
        "mission_off(s) | mission_on(s) | dist_off(m) | dist_on(m) | delta_dist(m) | injected | exchanged"
    )
    print("-" * 152)

    for row_off, row_on in rows:
        print(
            f"{row_off['seed']:8d} | "
            f"{row_off['optimization_time_s']:10.4f} | {row_on['optimization_time_s']:9.4f} | "
            f"{row_on['optimization_time_s'] - row_off['optimization_time_s']:8.4f} | "
            f"{row_off['mission_time_s']:14.3f} | {row_on['mission_time_s']:13.3f} | "
            f"{row_off['total_distance_m']:11.2f} | {row_on['total_distance_m']:10.2f} | "
            f"{row_on['total_distance_m'] - row_off['total_distance_m']:13.2f} | "
            f"{row_on['injected_elites']:8d} | {row_on['dpga_exchanged_elites']:9d}"
        )

    print("\n==== AGGREGATE SUMMARY ====")
    metrics = (
        ("optimization_time_s", "Optimization time", "s", 4),
        ("mission_time_s", "Predicted mission time", "s", 3),
        ("total_distance_m", "Total flight distance", "m", 2),
        ("objective_cost", "Objective cost", "", 3),
    )
    for key, label, unit, digits in metrics:
        off_mean, off_std = finite_stats([off[key] for off, _ in rows])
        on_mean, on_std = finite_stats([on[key] for _, on in rows])
        delta_mean, delta_std = finite_stats(paired_delta(rows, key))
        suffix = f" {unit}" if unit else ""
        print(
            f"{label}: OFF={off_mean:.{digits}f}±{off_std:.{digits}f}{suffix}, "
            f"ON={on_mean:.{digits}f}±{on_std:.{digits}f}{suffix}, "
            f"paired Δ={delta_mean:.{digits}f}±{delta_std:.{digits}f}{suffix}"
        )

    dist_delta = np.asarray(paired_delta(rows, "total_distance_m"), dtype=float)
    mission_delta = np.asarray(paired_delta(rows, "mission_time_s"), dtype=float)
    print(f"KT reduced distance in {int(np.count_nonzero(dist_delta < 0))}/{len(rows)} runs.")
    print(f"KT reduced predicted mission time in {int(np.count_nonzero(mission_delta < 0))}/{len(rows)} runs.")


def build_parser():
    parser = argparse.ArgumentParser(
        description="DPGA meta-KT ablation without reliability, RL, subsystem, or replanning triggers."
    )
    # Experiment parameters: ten paired runs by default.
    parser.add_argument("--runs", type=int, default=10, help="Number of paired KT OFF/ON runs.")
    parser.add_argument("--base-seed", type=int, default=20260715, help="First target-domain seed.")
    parser.add_argument("--meta-seed", type=int, default=20260714, help="Source-domain meta-training seed.")
    parser.add_argument("--island-count", type=int, default=13, help="Number of DPGA island populations.")
    parser.add_argument("--island-population-size", type=int, default=24, help="Population size per DPGA island.")
    parser.add_argument("--generations", type=int, default=20, help="DPGA generations in the target domain.")
    parser.add_argument("--exchange-interval", type=int, default=5, help="Generation interval for DPGA ring exchange.")
    parser.add_argument("--exchange-elites", type=int, default=2, help="Elites sent from each island during exchange.")
    parser.add_argument("--meta-population-size", type=int, default=300, help="Source meta-bank population size.")
    parser.add_argument("--source-generations", type=int, default=20, help="Source-domain meta-training generations.")
    parser.add_argument("--elite-count", type=int, default=8, help="Number of source elites retained.")
    parser.add_argument("--kt-interval", type=int, default=5, help="Generation interval for meta-KT injection.")
    parser.add_argument("--kt-ratio", type=float, default=0.15, help="Fraction of each island replaced by meta-KT.")
    parser.add_argument("--source-offset-radius", type=float, default=400.0,
                        help="Source-domain UAV position offset in meters.")
    return parser


def validate_args(args):
    if args.runs <= 0:
        raise ValueError("--runs must be greater than 0")
    if args.island_count <= 0:
        raise ValueError("--island-count must be greater than 0")
    if args.island_population_size < 4:
        raise ValueError("--island-population-size must be at least 4")
    if args.meta_population_size < 4:
        raise ValueError("--meta-population-size must be at least 4")
    if args.generations < 0 or args.source_generations < 0:
        raise ValueError("generation counts must not be negative")
    if args.exchange_interval < 0 or args.kt_interval < 0:
        raise ValueError("intervals must not be negative")
    if args.exchange_elites < 0:
        raise ValueError("--exchange-elites must not be negative")
    if args.elite_count <= 0:
        raise ValueError("--elite-count must be greater than 0")
    if not 0.0 < args.kt_ratio <= 1.0:
        raise ValueError("--kt-ratio must be in (0, 1]")


def main():
    args = build_parser().parse_args()
    validate_args(args)
    scenario = build_default_scenario()

    print(f"DPGA meta-KT ablation experiment {EXPERIMENT_VERSION}")
    print(
        f"Runs={args.runs}, islands={args.island_count}, island_pop={args.island_population_size}, "
        f"generations={args.generations}, exchange_every={args.exchange_interval}, "
        f"KT_every={args.kt_interval}, KT_ratio={args.kt_ratio:.2f}"
    )
    print("Reliability model=OFF, RL decision=OFF, subsystem=OFF, failure-triggered replanning=OFF")

    mission = sd.GA_SEAD(scenario["targets_sites"], args.meta_population_size)
    meta_elites = prepare_meta_elites(mission, scenario, args)
    if not meta_elites:
        raise RuntimeError("meta-elite bank is empty")
    print(f"Meta-elite bank ready: {len(meta_elites)} elites (source cost excluded from both groups).")

    target_info = build_uav_information(scenario, scenario["initial_states"])
    rows = []
    for run_index in range(1, args.runs + 1):
        seed = args.base_seed + run_index - 1
        row_off, row_on = run_paired_trial(
            mission, scenario, target_info, meta_elites, seed, run_index, args
        )
        rows.append((row_off, row_on))

    print_results(rows)


if __name__ == "__main__":
    main()
