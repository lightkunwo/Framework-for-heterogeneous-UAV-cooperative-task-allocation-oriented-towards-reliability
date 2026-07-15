import random
import time
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import dubins


class GA_SEAD(object):
    def __init__(self, targets, population_size=300, crossover_prob=0.67, elitism_num=2,
                 heading_discretization=10, use_chaos=True, chaos_mu=4.0,
                 chaos_seed=None, chaos_map='piecewise', chaos_maps=None,
                 chaos_map_params=None, chaos_transform=None, chaos_in_operators=False,
                 chaos_seed_ratio=0.05, guided_seed_ratio=0.10, guided_distance_weight=2e-4,
                 filter_duplicates=False, stagnation_chaos=False,
                 stagnation_chaos_patience=4, stagnation_replace_ratio=0.20,
                 stagnation_chaos_max_rounds=1,
                 near_duplicate_threshold=None, duplicate_mutation_rounds=2,
                 chaos_early_stop=True, early_stop_min_generations=5,
                 early_stop_patience=8, early_stop_rel_tol=1e-4,
                 early_stop_min_time_ratio=0.20,
                 local_refine=True, local_refine_passes=2,
                 multi_start_offsets=(0.137, 0.101, 0.377),
                 multi_start_distance_weight=0.001):
        self.targets = targets
        # Global message
        self.uav_id = []
        self.uav_type = []
        self.uav_velocity = []
        self.uav_turning_radius = []
        self.uav_state = []
        self.uav_base = []
        self.uav_num, self.target_num = 0, 0
        # GA parameters
        self.total_population_size = population_size
        self.population_size = population_size
        self.elitism_num = elitism_num
        self.crossover_num = round((self.population_size - self.elitism_num) * crossover_prob)
        self.mutation_num = self.population_size - self.crossover_num - self.elitism_num
        self.crossover_prob = crossover_prob
        self.mutation_operators_prob = [0.25, 0.25, 0.25, 0.25]
        self.crossover_operators_prob = [0.5, 0.5]
        # Coefficients of objective function
        self.lambda_1 = 0
        self.lambda_2 = 10
        # The precomputed matrix for optimization
        self.cost_graph = []
        self.uavType_for_missions = []
        self.tasks_status = [3 for _ in range(len(self.targets))]
        self.heading_discretization = heading_discretization
        self.discrete_integer_heading = [theta for theta in range(heading_discretization)]
        self.heading_multiplier = 2 * np.pi / heading_discretization
        self.remaining_targets = []
        self.task_amount_array = []
        self.task_index_array = []
        self.target_sequence = []
        self.target_index_array = []
        # Chaos-GA parameters. The public GA API is kept compatible with GA_SEAD_process.py.
        self.use_chaos = use_chaos
        self.chaos_mu = chaos_mu
        self.chaos_map = chaos_map
        self.chaos_maps = tuple(chaos_maps) if chaos_maps else self._default_chaos_maps(chaos_map)
        self.chaos_step = 0
        self.chaos_map_params = {
            'circle_a': 0.5,
            'circle_b': 0.2,
            'chebyshev_k': 4,
            'piecewise_p': 0.4,
            'tent_alpha': 0.5,
            'sine_a': 4.0,
            'iterative_a': 0.7,
            'singer_mu': 1.07,
            'sinusoidal_a': 2.3,
        }
        if chaos_map_params:
            self.chaos_map_params.update(chaos_map_params)
        self.chaos_transform = chaos_transform
        self.chaos_state = self._init_chaos_state(chaos_seed)
        self.chaos_in_operators = chaos_in_operators
        self.chaos_seed_ratio = float(np.clip(chaos_seed_ratio, 0.0, 1.0))
        self.guided_seed_ratio = float(np.clip(guided_seed_ratio, 0.0, 1.0))
        self.guided_distance_weight = float(max(0.0, guided_distance_weight))
        self._force_chaos_random = False
        self.filter_duplicates = filter_duplicates
        self.stagnation_chaos = stagnation_chaos
        self.stagnation_chaos_patience = stagnation_chaos_patience
        self.stagnation_replace_ratio = stagnation_replace_ratio
        self.stagnation_chaos_max_rounds = stagnation_chaos_max_rounds
        self.near_duplicate_threshold = near_duplicate_threshold
        self.duplicate_mutation_rounds = duplicate_mutation_rounds
        self.chaos_early_stop = chaos_early_stop
        self.early_stop_min_generations = early_stop_min_generations
        self.early_stop_patience = early_stop_patience
        self.early_stop_rel_tol = early_stop_rel_tol
        self.early_stop_min_time_ratio = early_stop_min_time_ratio
        self.local_refine = local_refine
        self.local_refine_passes = max(0, int(local_refine_passes))
        self.multi_start_offsets = tuple(multi_start_offsets) if multi_start_offsets else (0.137,)
        self.multi_start_distance_weight = float(max(0.0, multi_start_distance_weight))
        self.multi_start_info = None

    @staticmethod
    def _default_chaos_maps(chaos_map):
        if str(chaos_map).lower() in ('mixed', 'mixed_fast'):
            return ('piecewise', 'tent', 'sine', 'circle')
        return (chaos_map,)

    @staticmethod
    def _init_chaos_state(seed):
        if seed is None:
            state = random.random()
        else:
            # A fractional seed keeps the chaotic state in (0, 1).
            state = abs(float(seed)) % 1.0
        forbidden = (0.0, 0.25, 0.5, 0.75, 1.0)
        if state <= 1e-8 or state >= 1 - 1e-8 or any(abs(state - x) <= 1e-8 for x in forbidden):
            state = 0.6180339887498949
        return state

    @staticmethod
    def _safe_unit(value, fallback=0.6180339887498949):
        if not np.isfinite(value):
            return fallback
        value = float(value) % 1.0
        if value <= 1e-12 or value >= 1.0 - 1e-12:
            return fallback
        return min(max(value, 1e-12), 1.0 - 1e-12)

    def _active_chaos_map(self):
        name = str(self.chaos_maps[self.chaos_step % len(self.chaos_maps)]).lower()
        self.chaos_step += 1
        return name

    def _chaos_next(self):
        """Generate the next chaotic variable in (0, 1)."""
        x = min(max(self.chaos_state, 1e-12), 1.0 - 1e-12)
        map_name = self._active_chaos_map()

        if map_name == 'logistic':
            next_x = self.chaos_mu * x * (1.0 - x)
        elif map_name == 'tent':
            alpha = min(max(float(self.chaos_map_params.get('tent_alpha', 0.5)), 1e-6), 1.0 - 1e-6)
            next_x = x / alpha if x < alpha else (1.0 - x) / (1.0 - alpha)
        elif map_name == 'piecewise':
            p = min(max(float(self.chaos_map_params.get('piecewise_p', 0.4)), 1e-6), 0.499999)
            if x < p:
                next_x = x / p
            elif x < 0.5:
                next_x = (x - p) / (0.5 - p)
            elif x < 1.0 - p:
                next_x = (1.0 - p - x) / (0.5 - p)
            else:
                next_x = (1.0 - x) / p
        elif map_name == 'sine':
            a = min(max(float(self.chaos_map_params.get('sine_a', 4.0)), 1e-6), 4.0)
            next_x = (a / 4.0) * np.sin(np.pi * x)
        elif map_name == 'circle':
            a = float(self.chaos_map_params.get('circle_a', 0.5))
            b = float(self.chaos_map_params.get('circle_b', 0.2))
            next_x = x + b - (a / (2.0 * np.pi)) * np.sin(2.0 * np.pi * x)
        elif map_name == 'chebyshev':
            k = max(2, int(self.chaos_map_params.get('chebyshev_k', 4)))
            z = 2.0 * x - 1.0
            next_x = (np.cos(k * np.arccos(np.clip(z, -1.0, 1.0))) + 1.0) / 2.0
        elif map_name in ('gauss', 'mouse'):
            next_x = (1.0 / x) % 1.0
        elif map_name == 'iterative':
            a = min(max(float(self.chaos_map_params.get('iterative_a', 0.7)), 1e-6), 1.0)
            next_x = abs(np.sin(a * np.pi / x))
        elif map_name == 'singer':
            mu = min(max(float(self.chaos_map_params.get('singer_mu', 1.07)), 0.9), 1.08)
            next_x = mu * (7.86 * x - 23.31 * x ** 2 + 28.75 * x ** 3 - 13.302875 * x ** 4)
        elif map_name == 'sinusoidal':
            a = float(self.chaos_map_params.get('sinusoidal_a', 2.3))
            next_x = a * x ** 2 * np.sin(np.pi * x)
        else:
            next_x = self.chaos_mu * x * (1.0 - x)

        self.chaos_state = self._safe_unit(next_x, fallback=self._init_chaos_state(None))
        if self.chaos_state <= 1e-12 or self.chaos_state >= 1 - 1e-12:
            self.chaos_state = self._init_chaos_state(None)
        return self.chaos_state

    def _chaos_uniform(self):
        """Map a chaotic variable to a U(0, 1)-like random number."""
        y = min(max(self._chaos_next(), 1e-12), 1.0 - 1e-12)
        if self.chaos_transform == 'an':
            # The distribution transform described by the An chaotic random generator text.
            y = (np.log(y + 0.5) + np.log(2.0)) / np.log(3.0)
        elif self.chaos_transform == 'logistic_cdf':
            y = (2.0 / np.pi) * np.arcsin(np.sqrt(y))
        return float(min(max(y, 0.0), 1.0 - 1e-12))

    def _rand_uniform(self):
        use_chaos_random = self.use_chaos and (self.chaos_in_operators or self._force_chaos_random)
        return self._chaos_uniform() if use_chaos_random else random.random()

    def _rand_int(self, low, high):
        """Return an integer in [low, high)."""
        if high <= low:
            return low
        value = low + int(self._rand_uniform() * (high - low))
        return min(max(value, low), high - 1)

    def _rand_choice(self, choices):
        choices = list(choices)
        if not choices:
            raise IndexError('Cannot choose from an empty sequence.')
        return choices[self._rand_int(0, len(choices))]

    def _rand_sample(self, choices, k):
        pool = list(choices)
        result = []
        for _ in range(min(k, len(pool))):
            index = self._rand_int(0, len(pool))
            result.append(pool.pop(index))
        return result

    def _rand_shuffle(self, values):
        values = list(values)
        for i in range(len(values) - 1, 0, -1):
            j = self._rand_int(0, i + 1)
            values[i], values[j] = values[j], values[i]
        return values

    def _weighted_choice(self, choices, probs):
        choices = list(choices)
        probs = np.array(probs, dtype=float)
        total = np.sum(probs)
        if total <= 0:
            return self._rand_choice(choices)
        cumulative = np.cumsum(probs / total)
        r = self._rand_uniform()
        index = int(np.searchsorted(cumulative, r, side='right'))
        return choices[min(index, len(choices) - 1)]

    def _is_meaningful_improvement(self, new_fitness, best_fitness):
        if best_fitness <= 0:
            return new_fitness > best_fitness
        return new_fitness > best_fitness * (1.0 + self.early_stop_rel_tol)

    def _should_stop_early(self, generation, stale_generations, start_time=None, time_interval=None):
        if not self.use_chaos or not self.chaos_early_stop:
            return False
        if generation < self.early_stop_min_generations:
            return False
        if stale_generations < self.early_stop_patience:
            return False
        if start_time is not None and time_interval is not None and time_interval > 0:
            elapsed = time.time() - start_time
            if elapsed < time_interval * self.early_stop_min_time_ratio:
                return False
        return True

    def _should_apply_stagnation_chaos(self, stale_generations, applied_rounds=0):
        return (
            self.use_chaos
            and self.stagnation_chaos
            and self.stagnation_chaos_patience > 0
            and applied_rounds < self.stagnation_chaos_max_rounds
            and stale_generations >= self.stagnation_chaos_patience
        )

    def apply_stagnation_chaos(self, population):
        if not population or not self.use_chaos or not self.stagnation_chaos:
            return population

        new_population = list(population)
        elite_count = min(max(1, self.elitism_num), len(new_population))
        replace_pool = max(0, len(new_population) - elite_count)
        if replace_pool <= 0:
            return new_population

        replace_count = max(1, int(round(replace_pool * self.stagnation_replace_ratio)))
        fitness_ranking = sorted(range(len(new_population)), key=lambda idx: new_population[idx].fitness_value)
        target_indices = fitness_ranking[:replace_count]
        for idx in target_indices:
            new_population[idx] = self._large_scope_chaos_mutation(new_population[idx].chromosome)
        return new_population

    class Chromosome:
        def __init__(self, chromosome):
            self.chromosome = chromosome
            self.fitness_value = 0

        def copy_chromosome(self):
            chromosome_len, gene_len = len(self.chromosome), len(self.chromosome[0])
            duplicate_chromosome = [[0 for _ in range(gene_len)] for _ in range(chromosome_len)]
            for i in range(chromosome_len):
                for j in range(gene_len):
                    duplicate_chromosome[i][j] = self.chromosome[i][j]
            return duplicate_chromosome

    @staticmethod
    def order2target_bundle(chromosome):
        zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2], chromosome[3], chromosome[4])]
        return sorted(sorted(zipped_gene, key=lambda u: u[2]), key=lambda u: u[1])

    @staticmethod
    def order2task_bundle(chromosome):
        zipped_gene = [list(g) for g in zip(chromosome[0], chromosome[1], chromosome[2], chromosome[3], chromosome[4])]
        return sorted(zipped_gene, key=lambda u: u[2])

    @staticmethod
    def turn2order_based(chromosome):
        order_based_gene = np.array(sorted(chromosome, key=lambda u: u[0]))
        return [[g[i] for g in order_based_gene] for i in range(5)]

    @staticmethod
    def get_roulette_wheel(population):
        fitness_list = np.array([c.fitness_value for c in population])
        return fitness_list / np.sum(fitness_list)

    def objectives_evaluation(self, chromosome):
        cost = [0 for _ in range(self.uav_num)]
        task_sequence_time, time_list = [[] for _ in range(self.uav_num)], []  # time
        pre_site, pre_heading = [0 for _ in range(self.uav_num)], [0 for _ in range(self.uav_num)]
        for j, _ in enumerate(chromosome.chromosome[0]):
            assign_uav = self.uav_id.index(chromosome.chromosome[3][j])
            assign_target = chromosome.chromosome[1][j]
            assign_heading = chromosome.chromosome[4][j]
            cost[assign_uav] += self.cost_graph[assign_uav][pre_site[assign_uav]][pre_heading[assign_uav]][assign_target][assign_heading]
            task_sequence_time[assign_uav].append([assign_target, chromosome.chromosome[2][j],
                                                   cost[assign_uav] / self.uav_velocity[assign_uav]])
            pre_site[assign_uav], pre_heading[assign_uav] = assign_target, assign_heading
        for j in range(self.uav_num):
            cost[j] += self.cost_graph[j][pre_site[j]][pre_heading[j]][0][0]
        for sequence in task_sequence_time:
            time_list.extend(sequence)
        time_list.sort()
        # time sequence penalty
        penalty, j = 0, 0
        for task_num in self.tasks_status:
            if task_num >= 2:
                for k in range(1, task_num):
                    penalty += max(0, time_list[j + k - 1][2] - time_list[j + k][2])
            j += task_num
        # Calculate objective value
        mission_time = np.max(np.divide(cost, self.uav_velocity))
        total_distance = np.sum(cost)
        fitness = 1 / (mission_time + self.lambda_1 * total_distance + self.lambda_2 * penalty)
        return fitness, mission_time, total_distance, penalty

    def fitness_evaluation(self, population):
        for chromosome in population:
            fitness, _, _, _ = self.objectives_evaluation(chromosome)
            # Update fitness value
            chromosome.fitness_value = fitness

    @staticmethod
    def _metrics_cost_value(metrics):
        fitness = metrics[0]
        return float('inf') if fitness <= 0 else 1.0 / fitness

    def _is_local_refine_better(self, candidate_metrics, best_metrics):
        candidate_cost = self._metrics_cost_value(candidate_metrics)
        best_cost = self._metrics_cost_value(best_metrics)
        tol = 1e-9
        if candidate_cost < best_cost - tol:
            return True
        if abs(candidate_cost - best_cost) > tol:
            return False
        if candidate_metrics[1] < best_metrics[1] - tol:
            return True
        return candidate_metrics[1] <= best_metrics[1] + tol and candidate_metrics[2] < best_metrics[2] - tol

    def local_refine_solution(self, chromosome):
        """Greedily polish the best chromosome without changing the GA objective."""
        if not self.local_refine or self.local_refine_passes <= 0 or not hasattr(chromosome, 'chromosome'):
            return chromosome
        if not chromosome.chromosome or len(chromosome.chromosome) < 5 or not chromosome.chromosome[0]:
            return chromosome

        best_gene = chromosome.copy_chromosome()
        best_chromosome = self.Chromosome(best_gene)
        best_metrics = self.objectives_evaluation(best_chromosome)
        gene_count = len(best_gene[0])

        for _ in range(self.local_refine_passes):
            improved = False
            # Task-order repair lets the local search escape a good assignment with a poor execution order.
            for left in range(gene_count - 1):
                for right in range(left + 1, gene_count):
                    candidate_gene = [row[:] for row in best_gene]
                    for row in range(1, 5):
                        candidate_gene[row][left], candidate_gene[row][right] = \
                            candidate_gene[row][right], candidate_gene[row][left]
                    candidate = self.Chromosome(candidate_gene)
                    candidate_metrics = self.objectives_evaluation(candidate)
                    if self._is_local_refine_better(candidate_metrics, best_metrics):
                        best_gene = candidate_gene
                        best_chromosome = candidate
                        best_metrics = candidate_metrics
                        improved = True

            # Insertion repair handles cases where one task is good but appears too early or too late.
            for source in range(gene_count):
                for target in range(gene_count):
                    if source == target:
                        continue
                    candidate_gene = [row[:] for row in best_gene]
                    for row in range(1, 5):
                        value = candidate_gene[row].pop(source)
                        candidate_gene[row].insert(target, value)
                    candidate = self.Chromosome(candidate_gene)
                    candidate_metrics = self.objectives_evaluation(candidate)
                    if self._is_local_refine_better(candidate_metrics, best_metrics):
                        best_gene = candidate_gene
                        best_chromosome = candidate
                        best_metrics = candidate_metrics
                        improved = True

            for point in range(gene_count):
                task_type = int(best_gene[2][point])
                capable_uavs = self.uavType_for_missions[task_type - 1] \
                    if 1 <= task_type <= len(self.uavType_for_missions) else [best_gene[3][point]]

                # Heading repair is cheap and directly reduces Dubins transition cost.
                for heading in self.discrete_integer_heading:
                    if heading == best_gene[4][point]:
                        continue
                    candidate_gene = [row[:] for row in best_gene]
                    candidate_gene[4][point] = heading
                    candidate = self.Chromosome(candidate_gene)
                    candidate_metrics = self.objectives_evaluation(candidate)
                    if self._is_local_refine_better(candidate_metrics, best_metrics):
                        best_gene = candidate_gene
                        best_chromosome = candidate
                        best_metrics = candidate_metrics
                        improved = True

                # Agent repair is more expensive, so it is evaluated after heading repair.
                for uav_id in capable_uavs:
                    if uav_id == best_gene[3][point]:
                        continue
                    for heading in self.discrete_integer_heading:
                        candidate_gene = [row[:] for row in best_gene]
                        candidate_gene[3][point] = uav_id
                        candidate_gene[4][point] = heading
                        candidate = self.Chromosome(candidate_gene)
                        candidate_metrics = self.objectives_evaluation(candidate)
                        if self._is_local_refine_better(candidate_metrics, best_metrics):
                            best_gene = candidate_gene
                            best_chromosome = candidate
                            best_metrics = candidate_metrics
                            improved = True
            if not improved:
                break

        best_chromosome.fitness_value = best_metrics[0]
        return best_chromosome

    def generate_guided_chromosome(self):
        """Build a cost-aware seed chromosome using chaos-randomized target order."""
        genes = []
        order = 1
        costs = [0.0 for _ in range(self.uav_num)]
        pre_site = [0 for _ in range(self.uav_num)]
        pre_heading = [0 for _ in range(self.uav_num)]
        finish_time_by_task = {}
        id_to_index = {uid: idx for idx, uid in enumerate(self.uav_id)}

        remaining_targets = [target_id for target_id in range(1, self.target_num + 1)
                             if self.tasks_status[target_id - 1] > 0]
        target_order = self._rand_shuffle(remaining_targets)

        for target_id in target_order:
            task_types = [n + 1 for n in range(3 - self.tasks_status[target_id - 1], 3)]
            for task_type in task_types:
                best = None
                for uav_id in self.uavType_for_missions[task_type - 1]:
                    if uav_id not in id_to_index:
                        continue
                    u = id_to_index[uav_id]
                    for heading in self.discrete_integer_heading:
                        leg = self.cost_graph[u][pre_site[u]][pre_heading[u]][target_id][heading]
                        next_costs = list(costs)
                        next_costs[u] += leg
                        completion_time = next_costs[u] / self.uav_velocity[u]
                        prev_finish = finish_time_by_task.get((target_id, task_type - 1), 0.0)
                        precedence_penalty = max(0.0, prev_finish - completion_time)
                        mission_time = float(np.max(np.divide(next_costs, self.uav_velocity)))
                        total_distance = float(np.sum(next_costs))
                        # Tiny chaotic jitter breaks ties without overpowering the cost-aware score.
                        jitter = 1e-6 * self._rand_uniform() if self.use_chaos else 0.0
                        distance_term = max(self.lambda_1, self.guided_distance_weight) * total_distance
                        score = mission_time + distance_term + self.lambda_2 * precedence_penalty + jitter
                        if best is None or score < best[0]:
                            best = (score, u, uav_id, heading, leg, completion_time)
                if best is None:
                    # Fallback should be rare; keep feasibility if a capability list is empty.
                    capable_uavs = self.uavType_for_missions[task_type - 1]
                    uav_id = self._rand_choice(capable_uavs)
                    u = id_to_index[uav_id]
                    heading = self._rand_choice(self.discrete_integer_heading)
                    leg = self.cost_graph[u][pre_site[u]][pre_heading[u]][target_id][heading]
                    completion_time = (costs[u] + leg) / self.uav_velocity[u]
                else:
                    _, u, uav_id, heading, leg, completion_time = best
                genes.append([order, target_id, task_type, uav_id, heading])
                order += 1
                costs[u] += leg
                pre_site[u] = target_id
                pre_heading[u] = heading
                finish_time_by_task[(target_id, task_type)] = completion_time

        return self.Chromosome(self.turn2order_based(genes)) if genes else self.Chromosome([[] for _ in range(5)])

    def generate_population(self):
        def generate_chromosome(use_chaos_random=False):
            chromosome = np.zeros((5, sum(self.tasks_status)), dtype=int)
            for i in range(chromosome.shape[1]):
                chromosome[0][i] = i + 1  # order
                candidates = [n for n in range(1, self.target_num + 1)
                              if np.count_nonzero(chromosome[1] == n) < self.tasks_status[n - 1]]
                chromosome[1][i] = self._rand_choice(candidates) if use_chaos_random \
                    else random.choice(candidates)  # target id
            # turn to target-based
            target_bundle_chromosome = self.order2target_bundle(chromosome)
            for i in range(len(target_bundle_chromosome)):
                target_bundle_chromosome[i][2] = mission_type_list[i]  # mission type
                capable_uavs = self.uavType_for_missions[target_bundle_chromosome[i][2] - 1]
                target_bundle_chromosome[i][3] = self._rand_choice(capable_uavs) if use_chaos_random \
                    else random.choice(capable_uavs)  # uav id
                target_bundle_chromosome[i][4] = self._rand_choice(self.discrete_integer_heading) \
                    if use_chaos_random else random.choice(self.discrete_integer_heading)  # heading angle
            return self.Chromosome(self.turn2order_based(target_bundle_chromosome))  # back to order-based
        mission_type_list = []
        for tasks in self.tasks_status:
            mission_type_list.extend([n + 1 for n in range(3 - tasks, 3)])
        previous_force = self._force_chaos_random
        chaos_count = int(round(self.population_size * self.chaos_seed_ratio)) if self.use_chaos else 0
        guided_count = int(round(self.population_size * self.guided_seed_ratio)) if self.use_chaos else 0
        chaos_count = min(chaos_count, self.population_size)
        population = []
        try:
            for i in range(self.population_size):
                # Use chaos only for a seed subset. The rest keeps the original fast random initialization.
                self._force_chaos_random = i < chaos_count
                population.append(generate_chromosome(use_chaos_random=i < chaos_count))
            self._force_chaos_random = self.use_chaos
            for _ in range(guided_count):
                population.append(self.generate_guided_chromosome())
            if len(population) > self.population_size:
                self.fitness_evaluation(population)
                population = sorted(population, key=lambda chromosome: chromosome.fitness_value, reverse=True)[:self.population_size]
            random.shuffle(population)
            return population
        finally:
            self._force_chaos_random = previous_force

    def selection(self, roulette_wheel, num):
        if not (self.use_chaos and self.chaos_in_operators):
            return np.random.choice(np.arange(len(roulette_wheel)), size=num, replace=False, p=roulette_wheel)

        selected = []
        available = list(range(len(roulette_wheel)))
        weights = np.array(roulette_wheel, dtype=float)
        for _ in range(min(num, len(available))):
            local_weights = weights[available]
            total = np.sum(local_weights)
            if total <= 0:
                local_index = self._rand_int(0, len(available))
            else:
                cumulative = np.cumsum(local_weights / total)
                local_index = int(np.searchsorted(cumulative, self._rand_uniform(), side='right'))
                local_index = min(local_index, len(available) - 1)
            selected.append(available.pop(local_index))
        return np.array(selected, dtype=int)

    def crossover_operator(self, wheel, population):
        use_chaos_ops = self.use_chaos and self.chaos_in_operators

        def two_point_crossover(parent_1, parent_2):
            # turn to target-based
            target_based_gene = [self.order2target_bundle(parent_1.chromosome),
                                 self.order2target_bundle(parent_2.chromosome)]
            # choose cut point
            if use_chaos_ops:
                cut_point_1, cut_point_2 = sorted(self._rand_sample(range(len(parent_1.chromosome[0])), 2))
            else:
                cut_point_1, cut_point_2 = sorted(random.sample(range(len(parent_1.chromosome[0])), 2))
            cut_len = cut_point_2 - cut_point_1
            target_based_gene[0][cut_point_1:cut_point_2], target_based_gene[1][cut_point_1:cut_point_2] = \
                [target_based_gene[0][cut_point_1 + i][:3] + target_based_gene[1][cut_point_1 + i][3:] for i in
                 range(cut_len)], \
                [target_based_gene[1][cut_point_1 + i][:3] + target_based_gene[0][cut_point_1 + i][3:] for i in
                 range(cut_len)]
            # back to order-based
            child_1 = self.turn2order_based(target_based_gene[0])
            child_2 = self.turn2order_based(target_based_gene[1])
            return [self.Chromosome(child_1), self.Chromosome(child_2)]

        def target_bundle_crossover(parent_1, parent_2):
            # turn to target-based
            target_based_gene = [self.order2target_bundle(parent_1.chromosome),
                                 self.order2target_bundle(parent_2.chromosome)]
            # select targets to exchange
            if use_chaos_ops:
                targets_exchanged = self._rand_sample(
                    self.remaining_targets, self._rand_int(1, len(self.remaining_targets) + 1)
                )
            else:
                targets_exchanged = random.sample(
                    self.remaining_targets, random.randint(1, len(self.remaining_targets))
                )
            for target in targets_exchanged:
                start_index = sum(self.tasks_status[:target - 1])
                for i in range(start_index, start_index + self.tasks_status[target - 1]):
                    target_based_gene[0][i], target_based_gene[1][i] = \
                        target_based_gene[0][i][:3] + target_based_gene[1][i][3:], \
                        target_based_gene[1][i][:3] + target_based_gene[0][i][3:]
            # back to order-based
            child_1 = self.turn2order_based(target_based_gene[0])
            child_2 = self.turn2order_based(target_based_gene[1])
            return [self.Chromosome(child_1), self.Chromosome(child_2)]

        children = []
        for k in range(0, self.crossover_num, 2):
            p_1, p_2 = self.selection(wheel, 2)
            if use_chaos_ops:
                operator = self._weighted_choice([two_point_crossover, target_bundle_crossover],
                                                 self.crossover_operators_prob)
            else:
                operator = np.random.choice([two_point_crossover, target_bundle_crossover],
                                            p=self.crossover_operators_prob)
            children.extend(operator(population[p_1], population[p_2]))
        return children

    def mutation_operator(self, wheel, population):
        use_chaos_ops = self.use_chaos and self.chaos_in_operators

        def point_agent_mutation(chromosome):
            # choose a point to mutate
            mut_point = self._rand_int(0, len(chromosome.chromosome[0])) if use_chaos_ops \
                else np.random.randint(0, len(chromosome.chromosome[0]))
            new_gene = chromosome.copy_chromosome()
            # mutate assign UAV
            candidates = [i for i in self.uavType_for_missions[new_gene[2][mut_point] - 1]]
            new_gene[3][mut_point] = self._rand_choice(candidates) if use_chaos_ops else random.choice(candidates)
            return self.Chromosome(new_gene)

        def point_heading_mutation(chromosome):
            # choose a point to mutate
            mut_point = self._rand_int(0, len(chromosome.chromosome[0])) if use_chaos_ops \
                else np.random.randint(0, len(chromosome.chromosome[0]))
            new_gene = chromosome.copy_chromosome()
            # mutate assign heading
            candidates = [i for i in self.discrete_integer_heading if i != chromosome.chromosome[4][mut_point]]
            new_gene[4][mut_point] = self._rand_choice(candidates) if use_chaos_ops else random.choice(candidates)
            return self.Chromosome(new_gene)

        def target_bundle_mutation(chromosome):
            target_based_gene = self.order2target_bundle(chromosome.chromosome)
            for i, task_type in enumerate(self.target_sequence):
                if use_chaos_ops:
                    self.target_sequence[i] = self._rand_shuffle(task_type)
                else:
                    random.shuffle(task_type)
            shuffle_sequence = self.target_sequence[0] + self.target_sequence[1] + self.target_sequence[2]
            mutate_target_based = [[] for _ in range(len(target_based_gene))]
            j = 0
            for sequence in shuffle_sequence:
                mutate_target_based[j:j + self.tasks_status[sequence]] = \
                    [[b[:1] for b in target_based_gene[j:j + self.tasks_status[sequence]]][i] +
                     [a[1:] for a in target_based_gene[self.target_index_array[sequence]:self.target_index_array[sequence + 1]]]
                     [i] for i in range(self.tasks_status[sequence])]
                j += self.tasks_status[sequence]
            return self.Chromosome(self.turn2order_based(mutate_target_based))

        def task_bundle_mutation(chromosome):
            # turn to target-based
            task_based_gene = self.order2task_bundle(chromosome.chromosome)
            # choose a task to mutate
            mut_task = self._rand_int(0, 3) if use_chaos_ops else np.random.randint(0, 3)
            # shuffle the state
            task_sequence = list(range(self.task_amount_array[mut_task]))
            if use_chaos_ops:
                task_sequence = self._rand_shuffle(task_sequence)
            else:
                random.shuffle(task_sequence)
            # copy
            chromosome_len, gene_len = len(task_based_gene), len(task_based_gene[0])
            mutate_task_based = [[0 for _ in range(gene_len)] for _ in range(chromosome_len)]
            for i in range(chromosome_len):
                for j in range(gene_len):
                    mutate_task_based[i][j] = task_based_gene[i][j]
            # task mutation
            for i, sequence in enumerate(task_sequence):
                mutate_task_based[self.task_index_array[mut_task] + i][3:] = \
                    task_based_gene[self.task_index_array[mut_task] + sequence][3:]
            return self.Chromosome(self.turn2order_based(mutate_task_based))

        mutation_operators = [point_agent_mutation, point_heading_mutation, target_bundle_mutation, task_bundle_mutation]
        if use_chaos_ops:
            return [self._weighted_choice(mutation_operators, self.mutation_operators_prob)
                    (population[self.selection(wheel, 1)[0]]) for _ in range(self.mutation_num)]
        return [np.random.choice(mutation_operators, p=self.mutation_operators_prob)
                (population[self.selection(wheel, 1)[0]]) for _ in range(self.mutation_num)]

    @staticmethod
    def _flatten_decision_gene(chromosome):
        if not chromosome or len(chromosome) < 5 or not chromosome[0]:
            return tuple()
        flat = []
        for row in chromosome[1:5]:
            flat.extend(row)
        return tuple(flat)

    def _is_similar_chromosome(self, chrom_a, chrom_b):
        gene_a = self._flatten_decision_gene(chrom_a)
        gene_b = self._flatten_decision_gene(chrom_b)
        if not gene_a or len(gene_a) != len(gene_b):
            return False
        if gene_a == gene_b:
            return True
        if self.near_duplicate_threshold is None:
            return False
        match_ratio = sum(a == b for a, b in zip(gene_a, gene_b)) / len(gene_a)
        return match_ratio >= self.near_duplicate_threshold

    def _large_scope_chaos_mutation(self, chromosome):
        previous_force = self._force_chaos_random
        self._force_chaos_random = True
        try:
            new_gene = [list(row) for row in chromosome]
            if len(new_gene) < 5 or not new_gene[0]:
                return self.Chromosome(new_gene)

            gene_count = len(new_gene[0])
            mutation_ratio = 0.25 + 0.25 * self._rand_uniform()
            mutation_count = max(1, int(np.ceil(gene_count * mutation_ratio)))
            for mut_point in self._rand_sample(range(gene_count), mutation_count):
                task_type = int(new_gene[2][mut_point])
                capable_uavs = self.uavType_for_missions[task_type - 1] \
                    if 1 <= task_type <= len(self.uavType_for_missions) else self.uav_id
                if capable_uavs:
                    new_gene[3][mut_point] = self._rand_choice(capable_uavs)
                if self.discrete_integer_heading:
                    new_gene[4][mut_point] = self._rand_choice(self.discrete_integer_heading)
            return self.Chromosome(new_gene)
        finally:
            self._force_chaos_random = previous_force

    def chaotic_filter_population(self, population):
        if not self.use_chaos or not self.filter_duplicates or len(population) <= 1:
            return population

        filtered = []
        if self.near_duplicate_threshold is None:
            seen = set()
            for chromosome in population:
                candidate = chromosome
                attempts = 0
                signature = self._flatten_decision_gene(candidate.chromosome)
                while signature in seen and attempts < self.duplicate_mutation_rounds:
                    candidate = self._large_scope_chaos_mutation(candidate.chromosome)
                    signature = self._flatten_decision_gene(candidate.chromosome)
                    attempts += 1
                seen.add(signature)
                filtered.append(candidate)
            return filtered

        for chromosome in population:
            candidate = chromosome
            attempts = 0
            while any(self._is_similar_chromosome(candidate.chromosome, kept.chromosome) for kept in filtered) \
                    and attempts < self.duplicate_mutation_rounds:
                candidate = self._large_scope_chaos_mutation(candidate.chromosome)
                attempts += 1
            filtered.append(candidate)
        return filtered

    def elitism_operator(self, population):
        fitness_ranking = sorted(range(len(population)), key=lambda u: population[u].fitness_value, reverse=True)
        return [population[_] for _ in fitness_ranking[:self.elitism_num]]

    def information_setting(self, information, population, distributed=False):
        lost_agent, build_graph = False, False
        terminated_tasks, new_target = \
            sorted(information.tasks_completed, key=lambda u: u[1]), sorted(information.new_targets)
        clear_task, new_task = [], []
        if terminated_tasks:  # check terminated tasks
            for task in terminated_tasks:
                if self.tasks_status[task[0] - 1] == 3 - task[1] + 1:
                    self.tasks_status[task[0] - 1] -= 1
                    clear_task.append(task)
        if new_target:  # check new targets
            for target in new_target:
                if target not in self.targets:
                    self.targets.append(target)
                    self.tasks_status.append(3)
                    new_task.append(self.targets.index(target)+1)
            build_graph = True
        if not set(self.uav_id) == set(information.uav_id):  # check agents
            build_graph = True
            lost_agent = True
        # Clear the information
        self.uav_id = information.uav_id
        self.uav_type = information.uav_type
        self.uav_velocity = information.cruising_speed
        self.uav_turning_radius = information.turning_radii
        self.uav_state = information.uav_states
        self.uav_base = information.base
        self.uavType_for_missions = [[] for _ in range(3)]
        self.uav_num, self.target_num = len(self.uav_id), len(self.targets)
        # Classify capable UAVs to the missions
        # [surveillance[1,3],attack[1,2,3],munition[2]], [surveillance[s,a],attack[a,m],verification[s]]
        for i, agent in enumerate(self.uav_type):
            if agent == 1:  # surveillance
                self.uavType_for_missions[0].append(self.uav_id[i])
                self.uavType_for_missions[2].append(self.uav_id[i])
            elif agent == 2:  # attack, combat
                self.uavType_for_missions[0].append(self.uav_id[i])
                self.uavType_for_missions[1].append(self.uav_id[i])
            elif agent == 3:  # munition
                self.uavType_for_missions[1].append(self.uav_id[i])
        # COST TABLE (graph) -------------------------------------------------------------------------------------
        if build_graph:
            self.cost_graph = [[[[[0 for a in range(self.heading_discretization)] for b in range(self.target_num + 1)]
                                 for c in range(self.heading_discretization)] for d in range(self.target_num + 1)]
                               for u in range(self.uav_num)]
            for a in range(1, self.target_num + 1):
                for b in self.discrete_integer_heading:
                    for c in range(1, self.target_num + 1):
                        for d in self.discrete_integer_heading:
                            source_node = self.targets[a - 1] + [self.heading_multiplier * b]
                            end_node = self.targets[c - 1] + [self.heading_multiplier * d]
                            if source_node == end_node:
                                end_node[-1] += 1e-5
                            for u in range(self.uav_num):
                                distance = dubins.shortest_path(source_node, end_node,
                                                                self.uav_turning_radius[u]).path_length()
                                self.cost_graph[u][a][b][c][d] = distance
        # Cost of UAVs to targets or back to base (update real time information in graph)
        for a in range(1, len(self.targets) + 1):
            for b in self.discrete_integer_heading:
                node = self.targets[a - 1] + [self.heading_multiplier * b]
                for u in range(len(self.uav_id)):
                    distance = dubins.shortest_path(self.uav_state[u], node,
                                                    self.uav_turning_radius[u]).path_length()
                    self.cost_graph[u][0][0][a][b] = distance
                    distance = dubins.shortest_path(node, self.uav_base[u], self.uav_turning_radius[u]).path_length()
                    self.cost_graph[u][a][b][0][0] = distance
        for u in range(len(self.uav_id)):
            distance = dubins.shortest_path(self.uav_state[u], self.uav_base[u],
                                            self.uav_turning_radius[u]).path_length()
            self.cost_graph[u][0][0][0][0] = distance

        # GA parameters update
        self.population_size = round(self.total_population_size / len(self.uav_id)) \
            if distributed else self.total_population_size
        self.crossover_num = round((self.population_size - self.elitism_num) * self.crossover_prob)
        self.mutation_num = self.population_size - self.crossover_num - self.elitism_num
        self.lambda_1 = 1 / (sum(self.uav_velocity))

        # Predefined matrix
        self.remaining_targets = [target_id for target_id in range(1, len(self.targets) + 1) if
                                  not self.tasks_status[target_id - 1] == 0]
        self.task_amount_array = [np.count_nonzero(np.array(self.tasks_status) >= 3 - t) for t in range(3)]
        self.task_index_array = [0, self.task_amount_array[0], self.task_amount_array[0] + self.task_amount_array[1]]
        self.target_sequence = [[index for (index, value) in enumerate(self.tasks_status) if value == task_num]
                                for task_num in range(1, 4)]
        self.target_index_array = [0]
        for k, times in enumerate(self.tasks_status):
            self.target_index_array.append(self.target_index_array[k] + times)

        # Modify population
        if population:
            for elite in information.elite_chromosomes:
                for task in clear_task:
                    for site in range(len(elite[0])):
                        if elite[1][site] == task[0] and elite[2][site] == task[1]:
                            for row in elite:
                                row.pop(site)
                            elite[0] = [sequence for sequence in range(1, len(elite[0])+1)]
                            break

            if lost_agent:
                for elite in information.elite_chromosomes:
                    for index, task_type in enumerate(elite[2]):
                        if elite[3][index] not in self.uav_id:
                            elite[3][index] = self._rand_choice(self.uavType_for_missions[task_type - 1])

            if new_target:
                for elite in information.elite_chromosomes:
                    for target in new_task:
                        insert_index = sorted([self._rand_int(0, len(elite[0]) + 1) for _ in range(3)])
                        insert_index = [insert_index[i] + i for i in range(3)]
                        task_type = 1
                        for point in insert_index:
                            elite[1].insert(point, target)
                            elite[2].insert(point, task_type)
                            elite[3].insert(point, self._rand_choice(self.uavType_for_missions[task_type-1]))
                            elite[4].insert(point, self._rand_choice(self.discrete_integer_heading))
                            task_type += 1
                    elite[0] = [sequence for sequence in range(1, len(elite[1]) + 1)]

            if new_target or clear_task or lost_agent:
                population = self.generate_population()
            population.extend([self.Chromosome(elite) for elite in information.elite_chromosomes
                               if len(elite[0]) == sum(self.tasks_status)])
        return population

    def run_GA(self, iteration, uav_message, population=None, distributed=False):
        fitness_convergence = []
        population = self.information_setting(uav_message, population, distributed)
        residual_tasks = sum(self.tasks_status)
        if residual_tasks != 0:
            self.crossover_operators_prob = [0, 1] if residual_tasks <= 1 else [0.5, 0.5]
            if not population:
                try:
                    population = self.generate_population()
                except IndexError:
                    return [[] for _ in range(5)], 1e5, [], 0
                iteration -= 1
            if self.filter_duplicates:
                population = self.chaotic_filter_population(population)
            self.fitness_evaluation(population)
            wheel = self.get_roulette_wheel(population)
            best_fitness = max([_.fitness_value for _ in population])
            stale_generations = 0
            chaos_perturbations = 0
            fitness_convergence.append(1 / best_fitness)
            for generation in tqdm(range(iteration)):
                new_population = []
                new_population.extend(self.elitism_operator(population))
                new_population.extend(self.crossover_operator(wheel, population))
                new_population.extend(self.mutation_operator(wheel, population))
                if self.filter_duplicates:
                    new_population = self.chaotic_filter_population(new_population)
                self.fitness_evaluation(new_population)
                wheel = self.get_roulette_wheel(new_population)
                population = new_population
                current_best = max([_.fitness_value for _ in population])
                if self._is_meaningful_improvement(current_best, best_fitness):
                    best_fitness = current_best
                    stale_generations = 0
                else:
                    stale_generations += 1
                if self.stagnation_chaos and self._should_apply_stagnation_chaos(stale_generations, chaos_perturbations):
                    population = self.apply_stagnation_chaos(population)
                    self.fitness_evaluation(population)
                    wheel = self.get_roulette_wheel(population)
                    current_best = max([_.fitness_value for _ in population])
                    if self._is_meaningful_improvement(current_best, best_fitness):
                        best_fitness = current_best
                    stale_generations = 0
                    chaos_perturbations += 1
                fitness_convergence.append(1 / current_best)
                if self._should_stop_early(generation + 1, stale_generations):
                    break
            return self.local_refine_solution(self.find_best_solution(population)), population, fitness_convergence
        else:
            return [[] for _ in range(5)], 0, [], 0

    def _clone_for_restart(self, chaos_seed):
        return GA_SEAD(
            [list(target) for target in self.targets],
            population_size=self.total_population_size,
            crossover_prob=self.crossover_prob,
            elitism_num=self.elitism_num,
            heading_discretization=self.heading_discretization,
            use_chaos=self.use_chaos,
            chaos_mu=self.chaos_mu,
            chaos_seed=chaos_seed,
            chaos_map=self.chaos_map,
            chaos_maps=self.chaos_maps,
            chaos_map_params=dict(self.chaos_map_params),
            chaos_transform=self.chaos_transform,
            chaos_in_operators=self.chaos_in_operators,
            chaos_seed_ratio=self.chaos_seed_ratio,
            guided_seed_ratio=self.guided_seed_ratio,
            guided_distance_weight=self.guided_distance_weight,
            filter_duplicates=self.filter_duplicates,
            stagnation_chaos=self.stagnation_chaos,
            stagnation_chaos_patience=self.stagnation_chaos_patience,
            stagnation_replace_ratio=self.stagnation_replace_ratio,
            stagnation_chaos_max_rounds=self.stagnation_chaos_max_rounds,
            near_duplicate_threshold=self.near_duplicate_threshold,
            duplicate_mutation_rounds=self.duplicate_mutation_rounds,
            chaos_early_stop=self.chaos_early_stop,
            early_stop_min_generations=self.early_stop_min_generations,
            early_stop_patience=self.early_stop_patience,
            early_stop_rel_tol=self.early_stop_rel_tol,
            early_stop_min_time_ratio=self.early_stop_min_time_ratio,
            local_refine=self.local_refine,
            local_refine_passes=self.local_refine_passes,
            multi_start_offsets=self.multi_start_offsets,
            multi_start_distance_weight=self.multi_start_distance_weight,
        )

    def run_GA_multi_start(self, iteration, uav_message, population=None, distributed=False,
                           base_seed=None, chaos_offsets=None, restart_distance_weight=None):
        """Run several chaotic initial states and keep the best distance-aware result."""
        offsets = tuple(chaos_offsets) if chaos_offsets else self.multi_start_offsets
        distance_weight = self.multi_start_distance_weight if restart_distance_weight is None \
            else float(max(0.0, restart_distance_weight))
        if not self.use_chaos or len(offsets) <= 1:
            result = self.run_GA(iteration, uav_message, population, distributed)
            solution = result[0]
            fitness, mission_time, total_distance, penalty = self.objectives_evaluation(solution)
            self.multi_start_info = {
                'best_offset': offsets[0] if offsets else None,
                'restart_count': 1,
                'cost_value': 1.0 / fitness if fitness > 0 else float('inf'),
                'mission_time': mission_time,
                'total_distance': total_distance,
                'penalty': penalty,
                'selection_score': 1.0 / fitness if fitness > 0 else float('inf'),
                'attempts': [],
            }
            return result

        best_record = None
        attempts = []
        for index, offset in enumerate(offsets):
            if base_seed is None:
                chaos_seed = self._safe_unit(self.chaos_state + float(offset) + index * 0.173)
            else:
                random.seed(int(base_seed))
                np.random.seed(int(base_seed) % (2 ** 32 - 1))
                chaos_seed = (float(base_seed) * float(offset)) % 1.0
            restart_ga = self._clone_for_restart(chaos_seed)
            start_time = time.time()
            solution, restart_population, convergence = restart_ga.run_GA(
                iteration, uav_message, population=None, distributed=distributed
            )
            elapsed = time.time() - start_time
            fitness, mission_time, total_distance, penalty = restart_ga.objectives_evaluation(solution)
            cost_value = 1.0 / fitness if fitness > 0 else float('inf')
            selection_score = cost_value + distance_weight * total_distance
            record = {
                'ga': restart_ga,
                'solution': solution,
                'population': restart_population,
                'convergence': convergence,
                'offset': offset,
                'chaos_seed': chaos_seed,
                'elapsed_sec': elapsed,
                'generations': max(0, len(convergence) - 1),
                'fitness': fitness,
                'cost_value': cost_value,
                'mission_time': mission_time,
                'total_distance': total_distance,
                'penalty': penalty,
                'selection_score': selection_score,
            }
            attempts.append({key: value for key, value in record.items()
                             if key not in ('ga', 'solution', 'population', 'convergence')})
            if best_record is None or selection_score < best_record['selection_score']:
                best_record = record

        best_ga = best_record['ga']
        self.__dict__.update(best_ga.__dict__)
        self.multi_start_info = {
            'best_offset': best_record['offset'],
            'best_chaos_seed': best_record['chaos_seed'],
            'restart_count': len(offsets),
            'cost_value': best_record['cost_value'],
            'mission_time': best_record['mission_time'],
            'total_distance': best_record['total_distance'],
            'penalty': best_record['penalty'],
            'selection_score': best_record['selection_score'],
            'total_elapsed_sec': sum(attempt['elapsed_sec'] for attempt in attempts),
            'total_generations': sum(attempt['generations'] for attempt in attempts),
            'attempts': attempts,
        }
        return best_record['solution'], best_record['population'], best_record['convergence']

    def run_GA_time_period_version(self, time_interval, uav_message, population=None, update=True, distributed=False):
        iteration = 0
        start_time = time.time()
        if update:
            population = self.information_setting(uav_message, population, distributed)
        residual_tasks = sum(self.tasks_status)
        if residual_tasks != 0:
            self.crossover_operators_prob = [0, 1] if residual_tasks <= 1 else [0.5, 0.5]
            if not population:
                population = self.generate_population()
            if self.filter_duplicates:
                population = self.chaotic_filter_population(population)
            self.fitness_evaluation(population)
            wheel = self.get_roulette_wheel(population)
            best_fitness = max([_.fitness_value for _ in population])
            stale_generations = 0
            chaos_perturbations = 0
            while time.time() - start_time <= time_interval:
                iteration += 1
                new_population = []
                new_population.extend(self.elitism_operator(population))
                new_population.extend(self.crossover_operator(wheel, population))
                new_population.extend(self.mutation_operator(wheel, population))
                if self.filter_duplicates:
                    new_population = self.chaotic_filter_population(new_population)
                self.fitness_evaluation(new_population)
                wheel = self.get_roulette_wheel(new_population)
                population = new_population
                current_best = max([_.fitness_value for _ in population])
                if self._is_meaningful_improvement(current_best, best_fitness):
                    best_fitness = current_best
                    stale_generations = 0
                else:
                    stale_generations += 1
                if self.stagnation_chaos and self._should_apply_stagnation_chaos(stale_generations, chaos_perturbations):
                    population = self.apply_stagnation_chaos(population)
                    self.fitness_evaluation(population)
                    wheel = self.get_roulette_wheel(population)
                    current_best = max([_.fitness_value for _ in population])
                    if self._is_meaningful_improvement(current_best, best_fitness):
                        best_fitness = current_best
                    stale_generations = 0
                    chaos_perturbations += 1
                if self._should_stop_early(iteration, stale_generations, start_time, time_interval):
                    break
            best_solution = self.find_best_solution(population)
            if time.time() - start_time <= time_interval:
                best_solution = self.local_refine_solution(best_solution)
            return best_solution, population
        else:
            chromosome = self.Chromosome([[] for _ in range(5)])
            self.fitness_evaluation([chromosome])
            return chromosome, []

    def run_RS(self, iteration, uav_message, population=None):
        a = []
        population = self.generate_population()
        self.fitness_evaluation(population)
        fitness = [_.fitness_value for _ in population]
        a.append(1/max(fitness))
        iteration -= 1
        for _ in range(iteration):
            new_population = self.generate_population()
            self.fitness_evaluation(new_population)
            new_fitness = [_.fitness_value for _ in population]
            for i, chromosome in enumerate(new_population):
                if new_fitness[i] > min(fitness):
                    fitness[fitness.index(min(fitness))] = new_fitness[i]
                    population[i] = chromosome
            a.append(1/max(fitness))
        return self.find_best_solution(population), population, a

    @staticmethod
    def find_best_solution(population):
        fitness = 0
        index = 0
        for i, chromosome in enumerate(population):
            if chromosome.fitness_value > fitness:
                fitness = chromosome.fitness_value
                index = i
        return population[index]

    def plot_result(self, best_solution, curve=None):
        def dubins_plot(state_list, c, time_):
            distance = 0
            route_ = [[] for _ in range(2)]
            arrow_ = []
            for a in range(1, len(state_list)-1):
                state_list[a][2] *= np.pi / 180
            for a in range(len(state_list) - 1):
                sp = state_list[a]
                gp = state_list[a + 1] if state_list[a] != state_list[a + 1] \
                    else [state_list[a + 1][0], state_list[a + 1][1], state_list[a + 1][2] - 1e-5]
                dubins_path = dubins.shortest_path(sp, gp, self.uav_turning_radius[c])
                path, _ = dubins_path.sample_many(.1)
                route_[0].extend([b[0] for b in path])
                route_[1].extend([b[1] for b in path])
                distance += dubins_path.path_length()
                try:
                    time_[a].append(distance / self.uav_velocity[c])
                except IndexError:
                    pass
            try:
                arrow_.extend(
                    [[route_[0][arr], route_[1][arr], route_[0][arr + 100], route_[1][arr + 100]]
                     for arr in range(0, len(route_[0]), 15000)])
            except IndexError:
                pass
            return distance, route_, arrow_

        fitness, mission_time, total_distance, penalty = self.objectives_evaluation(best_solution)
        best_solution = best_solution.chromosome
        print(f'Chromosome: \n{np.array(best_solution)}')
        print("==============================================================================")
        uav_num = len(self.uav_id)
        dist = np.zeros(uav_num)
        task_sequence_state = [[] for _ in range(uav_num)]
        task_route = [[] for _ in range(uav_num)]
        route_state = [[] for _ in range(uav_num)]
        arrow_state = [[] for _ in range(uav_num)]
        for j in range(len(best_solution[0])):
            assign_uav = self.uav_id.index(best_solution[3][j])
            assign_target = best_solution[1][j]
            assign_heading = best_solution[4][j] * self.heading_multiplier * 180 / np.pi
            task_sequence_state[assign_uav].append([
                self.targets[assign_target - 1][0], self.targets[assign_target - 1][1],
                assign_heading])
            task_route[assign_uav].extend([[assign_target, best_solution[2][j]]])
        for j in range(uav_num):
            task_sequence_state[j] = [self.uav_state[j]] + task_sequence_state[j] + [self.uav_base[j]]
            dist[j], route_state[j], arrow_state[j] = dubins_plot(task_sequence_state[j], j, task_route[j])

        task_type = ["classify", "attack", "verify"]
        uav_type = ["Surveillance UAV", "Attack UAV", "Munition UAV"]
        print("Tasks assigned: ")
        for j in range(len(task_route)):
            print(f'\nUAV{self.uav_id[j]} ({uav_type[self.uav_type[j] - 1]}): ')
            for k in range(len(task_route[j])):
                print(f'Target{task_route[j][k][0]} {task_type[task_route[j][k][1] - 1]} task')

        print("==============================================================================")
        print("Results: ")
        print(f'Mission time: {np.round(mission_time, 3)} (sec)')
        print(f'Total distance: {np.round(total_distance, 3)} (m)')
        print(f'Cost value: {np.round(1 / fitness, 3)}')
        print(f'Penalty for task sequence constraints: {np.round(penalty, 3)}')
        print("==============================================================================")

        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#06C2AC', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#C875C4']
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
        font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}
        if curve:
            plt.subplot(122)
            plt.plot([b for b in range(1, len(curve) + 1)], curve, '-')
            plt.grid()
            plt.title("Convergence", font0)
            plt.xlabel("Iteration", font0)
            plt.ylabel("Cost", font0)
            plt.subplot(121)
        else:
            fig, ax = plt.subplots()
            labels = ax.get_xticklabels() + ax.get_yticklabels()
            [label.set_fontname('Times New Roman') for label in labels]
        for i in range(uav_num):
            plt.plot(route_state[i][0], route_state[i][1], '-', linewidth=0.8, color=color_style[i], label=f'UAV {self.uav_id[i]}')
            plt.text(self.uav_state[i][0]-100, self.uav_state[i][1]-200, f'UAV {self.uav_id[i]}', font)
            plt.axis("equal")
            for arrow in arrow_state[i]:
                plt.arrow(arrow[0], arrow[1], arrow[2] - arrow[0], arrow[3] - arrow[1], width=16, color=color_style[i])
        plt.plot([x[0] for x in self.uav_state], [x[1] for x in self.uav_state], 'k^', markerfacecolor='none', markersize=8)
        plt.plot([b[0] for b in self.targets], [b[1] for b in self.targets], 'ms', label='Target position',
                 markerfacecolor='none', markersize=6)
        plt.plot([x[0] for x in self.uav_base], [x[1] for x in self.uav_base], 'r*', markerfacecolor='none', markersize=10, label='Base')
        for t in self.targets:
            plt.text(t[0]+100, t[1]+100, f'Target {self.targets.index(t)+1}', font1)
        for b in self.uav_base:
            plt.text(b[0]-100, b[1]-200, f'Base', font2)
        plt.legend(loc='upper right', prop=font)
        plt.title("Routes", font0)
        plt.xlabel('East, m', font0)
        plt.ylabel('North, m', font0)
        plt.show()


class InformationOfUAVs(object):
    def __init__(self, uav_id, uav_type, uav_states, cruising_velocities, minimum_tuning_radii, base_configurations,
                 uav_best_solution=None, new_targets=None, tasks_completed=None):
        self.uav_id = uav_id
        self.uav_type = uav_type
        self.uav_states = uav_states
        self.cruising_speed = cruising_velocities
        self.turning_radii = minimum_tuning_radii
        self.base = base_configurations
        self.new_targets = new_targets if new_targets else []
        self.tasks_completed = tasks_completed if tasks_completed else []
        self.elite_chromosomes = uav_best_solution if uav_best_solution else []


if __name__ == "__main__":
    # targets

    targets = [[3100, 2200], [500, 3700], [2300, 2500], [2000, 3900], [4450, 3600], [4630, 4780], [1400, 4500]]
    # UAVs
    UAV_ID = [1, 2, 3, 4, 5, 6]
    UAV_type = [1, 2, 3, 1, 3, 2]  # 1: surveillance, 2: attack, 3: munition
    cruising_speed = [70, 80, 90, 60, 100, 80]  # (m/s)
    minimum_turning_radii = [200, 250, 300, 180, 300, 260]
    UAV_state = [[1000, 300, -np.pi], [1500, 700, np.pi / 2], [3000, 0, np.pi / 3],
                 [1800, 400, -20 * np.pi / 180], [2200, 280, 45 * np.pi / 180],
                 [4740, 300, 140 * np.pi / 180]]  # [East(m), North(m), heading angle(rad)]
    base_configuration = [[0, 0, -np.pi / 2], [0, 0, -np.pi / 2], [1000, 6000, np.pi / 2],
                          [1000, 6000, np.pi / 2], [4000, 5500, np.pi / 3],
                          [4000, 5500, np.pi / 3]]  # [East(m), North(m), runway direction(rad)]
    uav_info = InformationOfUAVs(UAV_ID, UAV_type, UAV_state, cruising_speed, minimum_turning_radii, base_configuration)

    population_size = 300
    iteration = 300
    print("iteratiosjhghjgjhada空间看你能拿,nsn",iteration);
    ga = GA_SEAD(targets, population_size)
    solution, ga_population, convergence = ga.run_GA_multi_start(iteration, uav_info, base_seed=20260603)
    print(f"Multi-start info: {ga.multi_start_info}")
    ga.plot_result(solution, convergence)
