import time
import numpy as np
from math import cos, sin, hypot, exp
from matplotlib import pyplot as plt
import multiprocessing as mp
import queue
import dubins
from dubins_model import angle_between, step_pid
from GA_SEAD_process import GA_SEAD, InformationOfUAVs

import os
from datetime import datetime
from subsystem import SubsystemSelector
from swarmReliability import ConsecutiveKOutOfN
from second_dpga import run_task_allocation_process_replan, run_main_process
from second_plotting import generate_time_flow_plot as plotting_generate_time_flow_plot
from second_plotting import finalize_simulation_outputs
from second_migration import (
    TORCH_AVAILABLE,
    initialize_transfer_policy,
    record_policy_point,
    build_subsystem_state,
    update_actor_critic,
    decide_replan_action,
)
#zhongwen
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'


class UAV(object):
    def __init__(self, uav_id, uav_type, uav_velocity, uav_Rmin, initial_position, depot, reliability_alpha=0.0001, reliability_beta=0.0):
        self.id = uav_id
        self.type = uav_type
        self.velocity = uav_velocity
        self.Rmin = uav_Rmin
        self.omega_max = self.velocity / self.Rmin
        self.x0 = initial_position[0]
        self.y0 = initial_position[1]
        self.theta0 = initial_position[2]
        self.depot = depot
        self.reliability_alpha = reliability_alpha
        self.reliability_beta = reliability_beta
        self.start_time = None
        self.is_failed = False
        self.failure_time = None

    def get_reliability(self, current_time, cumulative_distance=0.0):
        if self.start_time is None:
            self.start_time = current_time
        elapsed_time = current_time - self.start_time
        effective_distance = max(0.0, float(cumulative_distance))
        reliability = exp(-(self.reliability_alpha * elapsed_time + self.reliability_beta * effective_distance))
        return max(0.0, min(1.0, reliability))

    def check_failure(self, current_time, threshold=0.98, cumulative_distance=0.0):
        if self.is_failed:
            return True
        current_reliability = self.get_reliability(current_time, cumulative_distance)
        if current_reliability < threshold:
            self.is_failed = True
            self.failure_time = current_time
            return True
        return False


class DynamicSEADMissionSimulator(object):

    def __init__(self, targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations,
                 reliability_alpha_dict=None, reliability_beta_dict=None, failure_threshold=0.98, save_dir='./simulation_results',
                 enable_subsystem=True, consecutive_k=2, min_subsystem_reliability=0.95,
                 enable_assist_reallocation=True, assist_replan_cooldown=2.0):
        self.targets_sites = targets_sites
        self.uavs = [uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations, [], [], [], []]

        self.reliability_alpha_dict = reliability_alpha_dict if reliability_alpha_dict else {
            1: 0.0001, 2: 0.00015, 3: 0.00017
        }
        self.reliability_beta_dict = reliability_beta_dict if reliability_beta_dict else {
            1: 1.0e-06, 2: 1.5e-06, 3: 2.2e-06
        }
        self.failure_threshold = failure_threshold
        self.save_dir = save_dir

        self.enable_subsystem = enable_subsystem
        self.consecutive_k = consecutive_k
        self.min_subsystem_reliability = min_subsystem_reliability
        self.enable_assist_reallocation = enable_assist_reallocation
        self.assist_replan_cooldown = assist_replan_cooldown

        # State-driven reallocation policy parameters
        self.rebuild_threshold_high = 0.65
        self.rebuild_threshold_low = 0.35
        self.external_migrate_ratio = 0.40
        self.internal_migrate_ratio = 0.15

        # Actor-Critic for migration-policy decision is initialized inside second_migration.py
        initialize_transfer_policy(self)

        # For normalizing distance-style spatial indicators
        _norm_points = []
        _norm_points.extend([[float(t[0]), float(t[1])] for t in self.targets_sites])
        _norm_points.extend([[float(s[0]), float(s[1])] for s in initial_states])
        _norm_points.extend([[float(b[0]), float(b[1])] for b in base_locations])
        if len(_norm_points) >= 2:
            _arr = np.array(_norm_points, dtype=float)
            self.map_diag = float(np.linalg.norm(_arr.max(axis=0) - _arr.min(axis=0)))
            if self.map_diag <= 1e-6:
                self.map_diag = 1.0
        else:
            self.map_diag = 1.0

        if self.enable_subsystem:
            self.subsystem_selector = SubsystemSelector(
                consecutive_k=self.consecutive_k,
                min_subsystem_reliability=self.min_subsystem_reliability
            )
            print(f"✅ Subsystem Selector Enabled (k={consecutive_k}, R_min={min_subsystem_reliability})")
            if self.use_torch_actor_critic:
                print("🧠 Torch Actor-Critic Enabled for subsystem decision-making")
            elif self.use_actor_critic:
                print("⚠️  Torch unavailable. Fallback to heuristic decision policy.")
        else:
            self.subsystem_selector = None
            print("⚠️  Subsystem Selector Disabled")

        self.subsystem_events = []
        self.subsystem_reliability_history = []
        self.task_completion_log = []
        self.policy_history = []
        self.initial_assignment_log = {}

        print(f"🤝 Assist Reallocation: {'Enabled' if self.enable_assist_reallocation else 'Disabled'} "
              f"(cooldown={self.assist_replan_cooldown}s)")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.epoch = 0

    def _record_policy_point(self, event_time, subsystem_id, trigger, decision,
                             alpha_rebuild, beta_internal, beta_external,
                             migrate_mode, migrate_ratio, scope=None):
        return record_policy_point(
            self, event_time, subsystem_id, trigger, decision,
            alpha_rebuild, beta_internal, beta_external,
            migrate_mode, migrate_ratio, scope
        )

    @staticmethod
    def _empty_chromosome_5rows():
        return [[] for _ in range(5)]

    @staticmethod
    def _normalize_chromosome_5rows(chrom):
        if chrom is None:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()

        try:
            import numpy as _np
            if isinstance(chrom, _np.ndarray):
                chrom = chrom.tolist()
        except Exception:
            pass

        if isinstance(chrom, (list, tuple)) and len(chrom) == 0:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()

        if isinstance(chrom, (list, tuple)) and len(chrom) == 5 and all(isinstance(r, (list, tuple)) for r in chrom):
            return [list(r) for r in chrom]

        return DynamicSEADMissionSimulator._empty_chromosome_5rows()

    @staticmethod
    def _uav_msg_chromosome_5rows(uav_msg):
        if not isinstance(uav_msg, (list, tuple)) or len(uav_msg) <= 7:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()
        return DynamicSEADMissionSimulator._normalize_chromosome_5rows(uav_msg[7])

    @staticmethod
    def _project_chromosome_to_scope(chrom, scope_uav_ids):
        norm = DynamicSEADMissionSimulator._normalize_chromosome_5rows(chrom)
        if not isinstance(scope_uav_ids, (list, tuple, set)) or len(scope_uav_ids) == 0:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()
        allowed = set(int(uid) for uid in scope_uav_ids)
        if len(norm) != 5:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()
        lens = [len(r) for r in norm if isinstance(r, list)]
        if len(lens) != 5 or min(lens) <= 0:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()
        L = min(lens)
        keep = []
        for i in range(L):
            try:
                uid = int(norm[3][i])
            except Exception:
                continue
            if uid in allowed:
                keep.append(i)
        if len(keep) == 0:
            return DynamicSEADMissionSimulator._empty_chromosome_5rows()
        projected = [[norm[r][i] for i in keep] for r in range(5)]
        projected[0] = [i + 1 for i in range(len(keep))]
        return projected

    def _build_initial_assignment_log(self, chromosome):
        norm = self._normalize_chromosome_5rows(chromosome)
        log = {uid: [] for uid in self.uavs[0]}
        if len(norm) != 5 or len(norm[0]) == 0:
            return log
        task_type_names = {1: 'reconnaissance', 2: 'attack', 3: 'verification'}
        gene_num = min(len(norm[0]), len(norm[1]), len(norm[2]), len(norm[3]), len(norm[4]))
        ordered_indices = sorted(range(gene_num), key=lambda i: int(norm[0][i]))
        for i in ordered_indices:
            try:
                target_id = int(norm[1][i])
                task_type = int(norm[2][i])
                uav_id = int(norm[3][i])
            except Exception:
                continue
            if uav_id not in log:
                log[uav_id] = []
            target_xy = self.targets_sites[target_id - 1] if 1 <= target_id <= len(self.targets_sites) else [None, None]
            log[uav_id].append({
                'target_id': target_id,
                'task_type': task_type,
                'task_name': task_type_names.get(task_type, str(task_type)),
                'coord': (target_xy[0], target_xy[1]),
            })
        return log

    # -------- 集中式初次分配（单次求解）--------
    def run_initial_allocation_once(self):
        mission = GA_SEAD(self.targets_sites, 100)
        uavs_info = InformationOfUAVs(
            self.uavs[0], self.uavs[1], self.uavs[4], self.uavs[2], self.uavs[3], self.uavs[5],
            uav_best_solution=[self._empty_chromosome_5rows() for _ in self.uavs[0]]
        )
        solution, _ = mission.run_GA_time_period_version(1.0, uavs_info, None, True, distributed=True)
        return solution.fitness_value, solution.chromosome

    def _encode_type_code(self, subsystem_uav_ids, alive_state):
        counts = {1: 0, 2: 0, 3: 0}
        for uid in subsystem_uav_ids:
            if uid not in self.uavs[0]:
                continue
            idx = self.uavs[0].index(uid)
            if alive_state[idx] != 1:
                continue
            uav_type = self.uavs[1][idx]
            if uav_type in counts:
                counts[uav_type] += 1

        bits = ''.join('1' if counts[t] > 0 else '0' for t in (1, 2, 3))
        nums = ''.join(str(min(9, counts[t])) for t in (1, 2, 3))
        type_code = bits + nums
        presence_count = sum(1 for t in (1, 2, 3) if counts[t] > 0)
        return type_code, presence_count, counts

    def _unfinished_task_coords(self, unfinished_tasks):
        coords = []
        for task in unfinished_tasks:
            if not task:
                continue
            tid = int(task[0]) - 1
            if 0 <= tid < len(self.targets_sites):
                coords.append([float(self.targets_sites[tid][0]), float(self.targets_sites[tid][1])])
        return coords

    def _compute_task_spread(self, unfinished_tasks):
        coords = self._unfinished_task_coords(unfinished_tasks)
        if len(coords) <= 1:
            return 0.0
        pts = np.array(coords, dtype=float)
        center = pts.mean(axis=0)
        dists = np.linalg.norm(pts - center, axis=1)
        return float(dists.mean())

    def _compute_uav_position_metrics(self, alive_members, uav_positions, unfinished_tasks):
        if not alive_members:
            return 0.0, 0.0, 0.0

        uav_xy = []
        for uid in alive_members:
            pos = uav_positions.get(uid, None) if isinstance(uav_positions, dict) else None
            if pos is not None and len(pos) >= 2:
                uav_xy.append([float(pos[0]), float(pos[1])])
            else:
                if uid in self.uavs[0]:
                    idx = self.uavs[0].index(uid)
                    init_pos = self.uavs[4][idx]
                    uav_xy.append([float(init_pos[0]), float(init_pos[1])])
        if len(uav_xy) == 0:
            return 0.0, 0.0, 0.0

        uav_arr = np.array(uav_xy, dtype=float)
        if len(uav_arr) <= 1:
            d_uav_dispersion = 0.0
        else:
            center = uav_arr.mean(axis=0)
            d_uav_dispersion = float(np.linalg.norm(uav_arr - center, axis=1).mean())

        task_coords = self._unfinished_task_coords(unfinished_tasks)
        if len(task_coords) == 0:
            d_uav_task = 0.0
            d_uav_task_std = 0.0
        else:
            task_arr = np.array(task_coords, dtype=float)
            dist_matrix = np.linalg.norm(uav_arr[:, None, :] - task_arr[None, :, :], axis=2)
            nearest = dist_matrix.min(axis=1)
            d_uav_task = float(nearest.mean())
            d_uav_task_std = float(nearest.std())

        return d_uav_task, d_uav_task_std, d_uav_dispersion

    def _build_subsystem_state(self, subsystem_uav_ids, reliability_history, alive_state, all_completed_tasks,
                               idle_flags, uav_positions=None):
        return build_subsystem_state(
            self, subsystem_uav_ids, reliability_history, alive_state, all_completed_tasks, idle_flags, uav_positions
        )

    @staticmethod
    def _sigmoid(x):
        x = float(np.clip(x, -20.0, 20.0))
        return float(1.0 / (1.0 + np.exp(-x)))

    def _decode_type_presence(self, subsystem_state):
        if "type_presence_count" in subsystem_state:
            return int(subsystem_state.get("type_presence_count", 0))
        type_code = str(subsystem_state.get("TypeCode", "000000"))
        bits = type_code[:3]
        try:
            return int(sum(1 for b in bits if int(b) > 0))
        except Exception:
            return 0

    def _state_to_vector(self, subsystem_state):
        n = max(1, int(subsystem_state.get("N", 0)))
        n_total = max(1, len(self.uavs[0]))
        total_tasks = max(1, len(self.targets_sites) * 3)
        map_diag = max(1e-6, self.map_diag)

        r_mean = float(np.clip(subsystem_state.get("R_mean", 0.0), 0.0, 1.0))
        r_min = float(np.clip(subsystem_state.get("R_min", 0.0), 0.0, 1.0))
        type_presence = float(np.clip(self._decode_type_presence(subsystem_state) / 3.0, 0.0, 1.0))
        n_norm = float(np.clip(int(subsystem_state.get("N", 0)) / n_total, 0.0, 1.0))
        n_remain = float(np.clip(int(subsystem_state.get("N_remain", 0)) / total_tasks, 0.0, 1.0))
        d_spread = float(np.clip(float(subsystem_state.get("D_spread", 0.0)) / map_diag, 0.0, 1.0))
        d_uav_task = float(np.clip(float(subsystem_state.get("D_uav_task", 0.0)) / map_diag, 0.0, 1.0))
        d_uav_task_std = float(np.clip(float(subsystem_state.get("D_uav_task_std", 0.0)) / max(1e-6, 0.5 * map_diag), 0.0, 1.0))
        d_uav_dispersion = float(np.clip(float(subsystem_state.get("D_uav_dispersion", 0.0)) / map_diag, 0.0, 1.0))
        idle_ratio = float(np.clip(int(subsystem_state.get("N_idle", 0)) / n, 0.0, 1.0))

        return np.array([
            r_mean,
            r_min,
            n_norm,
            type_presence,
            n_remain,
            d_spread,
            d_uav_task,
            d_uav_task_std,
            d_uav_dispersion,
            idle_ratio
        ], dtype=float)

    def _actor_rebuild_prob(self, state_vec):
        if not self.use_torch_actor_critic:
            return 0.5
        with torch.no_grad():
            s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=self.ac_device)
            probs = self.ac_actor_alpha(s)
            p = float(probs[0, 1].item())
        return float(np.clip(p, 0.05, 0.95))

    def _actor_internal_migrate_prob(self, state_vec):
        if not self.use_torch_actor_critic:
            return 0.5
        with torch.no_grad():
            s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=self.ac_device)
            probs = self.ac_actor_beta(s)
            p = float(probs[0, 1].item())
        return float(np.clip(p, 0.05, 0.95))

    def _critic_value(self, state_vec):
        if not self.use_torch_actor_critic:
            return 0.0
        with torch.no_grad():
            s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=self.ac_device)
            v = self.ac_critic(s)
        return float(v.item())

    def _compute_policy_reward(self, prev_state, next_state, decision):
        total_tasks = max(1, len(self.targets_sites) * 3)
        map_diag = max(1e-6, self.map_diag)

        prev_remain = float(prev_state.get("N_remain", 0))
        next_remain = float(next_state.get("N_remain", 0))
        progress = float(np.clip((prev_remain - next_remain) / total_tasks, -1.0, 1.0))

        prev_r = float(prev_state.get("R_mean", 0.0))
        next_r = float(next_state.get("R_mean", 0.0))
        rel_gain = float(np.clip(next_r - prev_r, -1.0, 1.0))

        prev_type = float(self._decode_type_presence(prev_state))
        next_type = float(self._decode_type_presence(next_state))
        type_gain = float(np.clip((next_type - prev_type) / 3.0, -1.0, 1.0))

        prev_spread = float(prev_state.get("D_spread", 0.0))
        next_spread = float(next_state.get("D_spread", 0.0))
        spread_gain = float(np.clip((prev_spread - next_spread) / map_diag, -1.0, 1.0))

        prev_disp = float(prev_state.get("D_uav_dispersion", 0.0))
        next_disp = float(next_state.get("D_uav_dispersion", 0.0))
        cohesion_gain = float(np.clip((prev_disp - next_disp) / map_diag, -1.0, 1.0))

        prev_n = max(1.0, float(prev_state.get("N", 0)))
        next_n = max(1.0, float(next_state.get("N", 0)))
        prev_idle = float(prev_state.get("N_idle", 0)) / prev_n
        next_idle = float(next_state.get("N_idle", 0)) / next_n
        idle_relief = float(np.clip(next_idle - prev_idle, -1.0, 1.0))

        rebuild_cost = 0.08 if decision == "rebuild" else 0.0
        reward = (
            1.20 * progress
            + 0.60 * rel_gain
            + 0.30 * type_gain
            + 0.25 * spread_gain
            + 0.15 * cohesion_gain
            + 0.10 * idle_relief
            - rebuild_cost
        )
        return float(reward)

    def _update_actor_critic(self, prev_state, next_state, decision, migrate_mode, terminal=False):
        return update_actor_critic(self, prev_state, next_state, decision, migrate_mode, terminal)

    def _decide_replan_action(self, subsystem_state):
        return decide_replan_action(self, subsystem_state)

    def task_allocation_process_replan(self, ga2control_queue, control2ga_queue, output_interval=0.5):
        return run_task_allocation_process_replan(self, ga2control_queue, control2ga_queue, output_interval)

    def main_process(self, uav, u2u_communication, gcs_init_solution_queue,
                     ga2replan_queue, replan2ga_queue, u2g, uav_failure=None, subsystem_queue=None):
        return run_main_process(self, uav, u2u_communication, gcs_init_solution_queue,
                                ga2replan_queue, replan2ga_queue, u2g, uav_failure, subsystem_queue)

    def print_task_route_summary_task_style(self):
        uav_type = ['reconnaissance UAV', 'attack UAV', 'munition UAV']
        task_type = ['reconnaissance', 'attack', 'verification']

        uav_ids = self.uavs[0]
        uav_types = self.uavs[1]

        task_route = [[] for _ in range(len(uav_ids))]
        id2idx = {uid: i for i, uid in enumerate(uav_ids)}

        for log in sorted(self.task_completion_log, key=lambda x: x['time']):
            uid = log['uav_id']
            if uid in id2idx:
                task_route[id2idx[uid]].append(log)

        print("\n" + "=" * 70)
        print("MISSION SUMMARY (TASK ROUTE STYLE)")
        print("=" * 70)

        for j in range(len(task_route)):
            print(f'\nUAV{uav_ids[j]} ({uav_type[uav_types[j] - 1]}): ')
            if len(task_route[j]) == 0:
                print("No task executed")
            else:
                for k in range(len(task_route[j])):
                    log = task_route[j][k]
                    target_xy = self.targets_sites[log['target_id'] - 1] if 1 <= log['target_id'] <= len(self.targets_sites) else [None, None]
                    msg = (f"Target{log['target_id']} {task_type[log['task_type'] - 1]} task | "
                           f"coord=({target_xy[0]}, {target_xy[1]}) | "
                           f"distance={log.get('distance', 0.0):.2f} m | "
                           f"time={log.get('time', 0.0):.2f} s")
                    if self.enable_subsystem and log.get('subsystem_id'):
                        msg += f" | subsystem={log.get('subsystem_id')}"
                    print(msg)

        print("\n" + "-" * 70)
        print("SUBSYSTEM EVENTS")
        print("-" * 70)
        if len(self.subsystem_events) == 0:
            print("No subsystem event.")
        else:
            for i, ev in enumerate(self.subsystem_events, 1):
                failed = ev.get('failed_uav')
                members = ev.get('selected_uavs', [])
                t = ev.get('time', 0.0)
                if members:
                    print(f'Event {i}: UAV{failed} failed at t={t:.2f}s -> Subsystem: {members}')
                else:
                    print(f'Event {i}: UAV{failed} failed at t={t:.2f}s -> Subsystem selection failed')
        print("=" * 70 + "\n")

    def generate_time_flow_plot(self, position, x, y, targets_sites, UAVs, failures,
                                active_subsystem, color_style, start_time, save_path):
        return plotting_generate_time_flow_plot(
            self, position, x, y, targets_sites, UAVs, failures,
            active_subsystem, color_style, start_time, save_path
        )

    def start_simulation(self, realtime_plot=False, uav_failure=None):
        uav_num = len(self.uavs[0])
        target_num = len(self.targets_sites)

        u2u_nodes = [mp.Queue() for _ in range(uav_num)]
        GCS = mp.Queue()
        subsystem_queues = [mp.Queue() for _ in range(uav_num)]
        uav_failure = [None for _ in range(uav_num)] if not uav_failure else uav_failure

        # 分布式重规划GA队列（仅REPLAN用）
        replan2ga = mp.Queue()
        ga2replan = mp.Queue()

        # GCS -> 每个UAV 的初始解下发队列
        gcs_init_solution_queues = [mp.Queue() for _ in range(uav_num)]

        UAVs = [UAV(self.uavs[0][n], self.uavs[1][n], self.uavs[2][n], self.uavs[3][n],
                    self.uavs[4][n], self.uavs[5][n], self.reliability_alpha_dict[self.uavs[1][n]],
                    self.reliability_beta_dict[self.uavs[1][n]])
                for n in range(uav_num)]

        replan_process = mp.Process(
            target=self.task_allocation_process_replan,
            args=(ga2replan, replan2ga)
        )

        main_processes = [mp.Process(target=self.main_process, args=(
            UAVs[n], u2u_nodes, gcs_init_solution_queues[n],
            ga2replan, replan2ga, GCS, uav_failure[n], subsystem_queues[n]
        )) for n in range(uav_num)]

        position = [[[UAVs[_].x0, UAVs[_].y0, 0]] for _ in range(uav_num)]
        x, y, yaw = [[UAVs[_].x0] for _ in range(uav_num)], [[UAVs[_].y0] for _ in range(uav_num)], [UAVs[_].theta0 for
                                                                                                     _ in
                                                                                                     range(uav_num)]
        distance = [0 for _ in range(uav_num)]
        state, completed = [1 for _ in range(uav_num)], [0 for _ in range(uav_num)]
        failures = [0 for _ in range(uav_num)]
        reliability_history = {i: [1.0] for i in range(uav_num)}
        time_history = {i: [0.0] for i in range(uav_num)}
        active_subsystem = []
        all_completed_tasks = []
        idle_flags = {uid: False for uid in self.uavs[0]}

        # 子系统轮次标识
        subsystem_epoch = 0

        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#06C2AC']
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 8}
        font0 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 10}
        font1 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
        font2 = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}

        origin_plot_list = [[p[0] for p in self.uavs[4]], [p[1] for p in self.uavs[4]]]
        targets_plot_list = [[p[0] for p in self.targets_sites], [p[1] for p in self.targets_sites]]
        base_plot_list = [[p[0] for p in self.uavs[5]], [p[1] for p in self.uavs[5]]]
        failure_uav_list = []

        theta = np.arange(0, 2 * np.pi, 0.1)
        fuselage = np.array([150 * np.cos(theta), 40 * np.sin(theta)])
        wing = 40 * np.array([[-0.5, -0.5, 0.5, 0.5, -0.5], [-6, 6, 6, -6, -6]])
        yaw_list = [[UAVs[_].theta0] for _ in range(uav_num)]
        rotation_translation = lambda angle, pos, bias_x, bias_y: np.array(
            [np.add(pos[0] * np.cos(angle) + pos[1] * np.sin(angle), bias_x),
             np.add(-pos[0] * np.sin(angle) + pos[1] * np.cos(angle), bias_y)])

        if realtime_plot:
            plt.ion()
            fig = plt.figure(figsize=(21, 6))
            ax_trajectory = plt.subplot(1, 3, 1)
            ax_reliability = plt.subplot(1, 3, 2)
            ax_policy = plt.subplot(1, 3, 3)
        else:
            plt.ioff()

        # 先启动UAV进程（它们阻塞等待初始解）
        for p in main_processes:
            p.start()

        # 启动重规划GA进程（等replan输入）
        replan_process.start()

        # GCS做一次集中式初始分配，并下发到每个UAV
        init_fit, init_solution = self.run_initial_allocation_once()
        self.initial_assignment_log = self._build_initial_assignment_log(init_solution)
        for q in gcs_init_solution_queues:
            q.put([init_fit, init_solution])

        start_time = time.time()
        print("🚀 Mission start!!")
        print(f"故障阈值: {self.failure_threshold}")
        print("各UAV类型的可靠度衰减系数(α):")
        for uav_type, alpha in self.reliability_alpha_dict.items():
            uav_type_name = ["", "侦察型", "攻击型", "弹药型"][uav_type]
            print(f"  类型{uav_type} ({uav_type_name}): α = {alpha}")
        print(f"子系统选择: {'启用' if self.enable_subsystem else '禁用'}")
        if self.enable_subsystem:
            print(f"  Consecutive-k: {self.consecutive_k}")
            print(f"  Min Reliability: {self.min_subsystem_reliability}")
        print("=" * 70)

        while state != completed:
            try:
                surveillance = GCS.get(timeout=1.0)
            except queue.Empty:
                for i, proc in enumerate(main_processes):
                    if state[i] == 1 and (not proc.is_alive()):
                        state[i] = 0
                        print(f"⚠️  [GCS] UAV {self.uavs[0][i]} process exited without [44], force mark shutdown.")
                continue

            if surveillance[0] == 100:
                self.task_completion_log.append({
                    'uav_id': surveillance[1],
                    'target_id': surveillance[2],
                    'task_type': surveillance[3],
                    'time': surveillance[4] - start_time,
                    'distance': float(surveillance[5]) if len(surveillance) > 5 else 0.0,
                    'subsystem_id': surveillance[6] if len(surveillance) > 6 else None
                })
                all_completed_tasks.append((surveillance[2], surveillance[3]))
                all_completed_tasks = list(set(all_completed_tasks))

                # Any subsystem state change must trigger migration-driven replanning.
                if self.enable_subsystem and active_subsystem and surveillance[1] in active_subsystem:
                    completed_uav_id = surveillance[1]
                    available_subsystem = [uid for uid in active_subsystem if uid in self.uavs[0]
                                           and state[self.uavs[0].index(uid)] == 1]
                    if len(available_subsystem) > 0:
                        uav_positions, uav_reliabilities = self.collect_uav_states(position, reliability_history, UAVs)
                        all_completed_tasks = list(set([tuple(t) if isinstance(t, list) else t for t in all_completed_tasks]))
                        unfinished_tasks = self._extract_unfinished_tasks(all_completed_tasks)

                        subsystem_state = self._build_subsystem_state(
                            available_subsystem, reliability_history, state, all_completed_tasks, idle_flags,
                            uav_positions=uav_positions
                        )
                        policy = self._decide_replan_action(subsystem_state)
                        decision = policy["decision"]
                        p_rebuild = policy["p_rebuild"]
                        migrate_mode = policy["migrate_mode"]
                        migrate_ratio = policy["migrate_ratio"]
                        beta_internal = policy["beta_internal"]
                        beta_external = policy["beta_external"]

                        if decision == "rebuild":
                            selected_uavs, subsystem_info = self.subsystem_selector.select_subsystem(
                                failed_uav_id=completed_uav_id,
                                all_uavs=UAVs,
                                uav_positions=uav_positions,
                                uav_reliabilities=uav_reliabilities,
                                unfinished_tasks=unfinished_tasks,
                                current_time=time.time() - start_time,
                                max_distance=2000
                            )
                            if completed_uav_id in available_subsystem and completed_uav_id not in selected_uavs:
                                selected_uavs.append(completed_uav_id)
                        else:
                            selected_uavs = available_subsystem[:]

                        selected_uavs = list(dict.fromkeys([
                            uid for uid in selected_uavs
                            if uid in self.uavs[0] and state[self.uavs[0].index(uid)] == 1
                        ]))
                        if not selected_uavs:
                            selected_uavs = available_subsystem[:]
                            decision = "internal"
                            migrate_mode = "internal"
                            migrate_ratio = self.internal_migrate_ratio
                            p_rebuild = 0.05
                            beta_internal = 0.95
                            beta_external = 0.05

                        # State must be refreshed after action execution (internal/rebuild).
                        updated_subsystem_state = self._build_subsystem_state(
                            selected_uavs, reliability_history, state, all_completed_tasks, idle_flags,
                            uav_positions=uav_positions
                        )
                        self._update_actor_critic(subsystem_state, updated_subsystem_state, decision, migrate_mode)

                        subsystem_epoch += 1
                        subsystem_id = f"SS-{subsystem_epoch}"
                        active_subsystem = selected_uavs
                        self._record_policy_point(
                            event_time=time.time() - start_time,
                            subsystem_id=subsystem_id,
                            trigger="task_complete",
                            decision=decision,
                            alpha_rebuild=p_rebuild,
                            beta_internal=beta_internal,
                            beta_external=beta_external,
                            migrate_mode=migrate_mode,
                            migrate_ratio=migrate_ratio,
                            scope=active_subsystem
                        )
                        print(f"\n🔁 [GCS] Task-completion reallocation by UAV {completed_uav_id}. "
                              f"Subsystem {active_subsystem} (id={subsystem_id}, epoch={subsystem_epoch})")
                        print(f"   policy: decision={decision}, alpha={p_rebuild:.3f}, "
                              f"beta={beta_internal:.3f}/{beta_external:.3f}, migrate_ratio={migrate_ratio:.2f}")
                        for u in range(uav_num):
                            subsystem_queues[u].put([
                                999, active_subsystem, subsystem_epoch,
                                decision, p_rebuild, migrate_mode, migrate_ratio, updated_subsystem_state, subsystem_id,
                                beta_internal, beta_external
                            ])

            elif surveillance[0] == 223:

                failed_uav_id = surveillance[1]
                failure_pos = [surveillance[2], surveillance[3]]
                failure_reliability = surveillance[4]
                failed_terminated_tasks = surveillance[5] if len(surveillance) > 5 else []

                failures[self.uavs[0].index(failed_uav_id)] = len(position[position.index(max(position, key=len))])
                failure_uav_list.append(failure_pos)


                print(f"UAV {failed_uav_id} FAILED at t={np.round(time.time() - start_time, 2)}s")
                print(f"Reliability: {failure_reliability:.4f}")
                print(f"Completed tasks: {len(failed_terminated_tasks)}")
                print(f"{'💥' * 30}")

                if self.enable_subsystem:
                    print(f"\n🔧 Triggering subsystem selection...")
                    uav_positions, uav_reliabilities = self.collect_uav_states(position, reliability_history, UAVs)
                    uav_positions[failed_uav_id] = failure_pos + [0]
                    uav_reliabilities[failed_uav_id] = failure_reliability

                    all_completed_tasks.extend(failed_terminated_tasks)
                    all_completed_tasks = list(set([tuple(t) if isinstance(t, list) else t for t in all_completed_tasks]))
                    unfinished_tasks = self._extract_unfinished_tasks(all_completed_tasks)

                    available_all = [
                        uid for uid in self.uavs[0]
                        if uid != failed_uav_id and state[self.uavs[0].index(uid)] == 1
                    ]
                    subsystem_state = self._build_subsystem_state(
                        available_all, reliability_history, state, all_completed_tasks, idle_flags, uav_positions=uav_positions
                    )
                    policy = self._decide_replan_action(subsystem_state)
                    decision = policy["decision"]
                    p_rebuild = policy["p_rebuild"]
                    migrate_mode = policy["migrate_mode"]
                    migrate_ratio = policy["migrate_ratio"]
                    beta_internal = policy["beta_internal"]
                    beta_external = policy["beta_external"]

                    selected_uavs = []
                    subsystem_info = {}
                    internal_candidates = [
                        uid for uid in active_subsystem
                        if uid != failed_uav_id and uid in self.uavs[0] and state[self.uavs[0].index(uid)] == 1
                    ]

                    if decision in ("internal", "light") and internal_candidates:
                        selected_uavs = internal_candidates
                    else:
                        selected_uavs, subsystem_info = self.subsystem_selector.select_subsystem(
                            failed_uav_id=failed_uav_id,
                            all_uavs=UAVs,
                            uav_positions=uav_positions,
                            uav_reliabilities=uav_reliabilities,
                            unfinished_tasks=unfinished_tasks,
                            current_time=time.time() - start_time,
                            max_distance=2000
                        )
                        if (not selected_uavs) and internal_candidates:
                            selected_uavs = internal_candidates
                            decision = "internal"
                            migrate_mode = "internal"
                            migrate_ratio = self.internal_migrate_ratio
                            p_rebuild = 0.05
                            beta_internal = 0.95
                            beta_external = 0.05

                    selected_uavs = list(dict.fromkeys([
                        uid for uid in selected_uavs
                        if uid != failed_uav_id and uid in self.uavs[0] and state[self.uavs[0].index(uid)] == 1
                    ]))

                    if selected_uavs:
                        type_distribution = {}
                        for uid in selected_uavs:
                            idx = self.uavs[0].index(uid)
                            utype = self.uavs[1][idx]
                            type_distribution[utype] = type_distribution.get(utype, 0) + 1
                        if not subsystem_info:
                            subsystem_info = {}
                        if "reliability" not in subsystem_info:
                            rel_values = [uav_reliabilities.get(uid, 1.0) for uid in selected_uavs]
                            subsystem_info["reliability"] = float(np.mean(rel_values)) if rel_values else 0.0
                        if "type_distribution" not in subsystem_info:
                            subsystem_info["type_distribution"] = type_distribution
                    else:
                        subsystem_info = {"reliability": 0.0, "type_distribution": {}}

                    # State must be refreshed after action execution (internal/rebuild).
                    updated_subsystem_state = self._build_subsystem_state(
                        selected_uavs, reliability_history, state, all_completed_tasks, idle_flags,
                        uav_positions=uav_positions
                    )
                    self._update_actor_critic(subsystem_state, updated_subsystem_state, decision, migrate_mode)

                    self.subsystem_events.append({
                        'time': time.time() - start_time,
                        'failed_uav': failed_uav_id,
                        'selected_uavs': selected_uavs,
                        'subsystem_reliability': subsystem_info.get('reliability', 0),
                        'type_distribution': subsystem_info.get('type_distribution', {}),
                        'decision': decision,
                        'p_rebuild': p_rebuild,
                        'migrate_mode': migrate_mode,
                        'migrate_ratio': migrate_ratio,
                        'beta_internal': beta_internal,
                        'beta_external': beta_external,
                        'state_snapshot': updated_subsystem_state
                    })
                    active_subsystem = selected_uavs

                    # 新轮次：先广播999，再广播998(故障机剔除)，都带epoch
                    subsystem_epoch += 1
                    subsystem_id = f"SS-{subsystem_epoch}"
                    self._record_policy_point(
                        event_time=time.time() - start_time,
                        subsystem_id=subsystem_id,
                        trigger="uav_failed",
                        decision=decision,
                        alpha_rebuild=p_rebuild,
                        beta_internal=beta_internal,
                        beta_external=beta_external,
                        migrate_mode=migrate_mode,
                        migrate_ratio=migrate_ratio,
                        scope=selected_uavs
                    )
                    print(f"\n Broadcasting subsystem selection result to all UAVs... (id={subsystem_id}, epoch={subsystem_epoch})")
                    print(f"   policy: decision={decision}, alpha={p_rebuild:.3f}, "
                          f"beta={beta_internal:.3f}/{beta_external:.3f}, migrate_ratio={migrate_ratio:.2f}")
                    for u in range(uav_num):
                        subsystem_queues[u].put([
                            999, selected_uavs, subsystem_epoch,
                            decision, p_rebuild, migrate_mode, migrate_ratio, updated_subsystem_state, subsystem_id,
                            beta_internal, beta_external
                        ])

                    for u in range(uav_num):
                        subsystem_queues[u].put([998, failed_uav_id, subsystem_epoch])

                    if selected_uavs:

                        print(f"Subsystem Activated: {selected_uavs}")
                        print(f"Subsystem Reliability: {subsystem_info['reliability']:.6f}")
                        print(f"Type Distribution: {subsystem_info['type_distribution']}")

                    else:
                        print(f"\n  Subsystem selection failed, all UAVs continue current tasks\n")
                else:
                    print("⚠️  Subsystem selection disabled")

            elif surveillance[0] == 44:
                shutdown_uav_id = surveillance[1]
                state[self.uavs[0].index(shutdown_uav_id)] = 0
                print(f"🛑[UAV {shutdown_uav_id}] Shut down at t={np.round(time.time() - start_time, 2)}s")

                # Keep subsystem scope synced when a member shuts down naturally.
                if self.enable_subsystem and shutdown_uav_id in active_subsystem:
                    alive_subsystem = [
                        uid for uid in active_subsystem
                        if uid in self.uavs[0] and state[self.uavs[0].index(uid)] == 1
                    ]
                    if alive_subsystem != active_subsystem:
                        active_subsystem = alive_subsystem
                        if active_subsystem:
                            uav_positions, _ = self.collect_uav_states(position, reliability_history, UAVs)
                            subsystem_state = self._build_subsystem_state(
                                active_subsystem, reliability_history, state,
                                all_completed_tasks, idle_flags, uav_positions=uav_positions
                            )
                            policy = self._decide_replan_action(subsystem_state)
                            decision = policy["decision"]
                            p_rebuild = policy["p_rebuild"]
                            migrate_mode = policy["migrate_mode"]
                            migrate_ratio = policy["migrate_ratio"]
                            beta_internal = policy["beta_internal"]
                            beta_external = policy["beta_external"]
                            subsystem_epoch += 1
                            subsystem_id = f"SS-{subsystem_epoch}"
                            self._record_policy_point(
                                event_time=time.time() - start_time,
                                subsystem_id=subsystem_id,
                                trigger="shutdown_sync",
                                decision=decision,
                                alpha_rebuild=p_rebuild,
                                beta_internal=beta_internal,
                                beta_external=beta_external,
                                migrate_mode=migrate_mode,
                                migrate_ratio=migrate_ratio,
                                scope=active_subsystem
                            )
                            print(f"🔄 [GCS] Sync subsystem after shutdown -> {active_subsystem} "
                                  f"(id={subsystem_id}, epoch={subsystem_epoch})")
                            for u in range(uav_num):
                                subsystem_queues[u].put([
                                    999, active_subsystem, subsystem_epoch,
                                    decision, p_rebuild, migrate_mode, migrate_ratio, subsystem_state, subsystem_id,
                                    beta_internal, beta_external
                                ])

            elif surveillance[0] == 224:
                requester_uav_id = surveillance[1]
                req_epoch = surveillance[5] if len(surveillance) > 5 else -1

                if not self.enable_subsystem:
                    continue

                # 仅接受当前子系统轮次内的请求
                if req_epoch != subsystem_epoch:
                    print(f"️  [GCS] Ignore stale assist-reallocation request from UAV {requester_uav_id} "
                          f"(req_epoch={req_epoch}, current_epoch={subsystem_epoch})")
                    continue

                # 仅对子系统成员发起的请求进行处理
                if requester_uav_id not in active_subsystem:
                    print(f"ℹ️  [GCS] Ignore assist request from UAV {requester_uav_id} not in active subsystem.")
                    continue

                available_subsystem = [uid for uid in active_subsystem if uid in self.uavs[0]
                                       and state[self.uavs[0].index(uid)] == 1]

                if len(available_subsystem) == 0:
                    print("ℹ️  [GCS] No available subsystem members for assist-reallocation.")
                    continue

                uav_positions, uav_reliabilities = self.collect_uav_states(position, reliability_history, UAVs)
                all_completed_tasks = list(set([tuple(t) if isinstance(t, list) else t for t in all_completed_tasks]))
                unfinished_tasks = self._extract_unfinished_tasks(all_completed_tasks)

                subsystem_state = self._build_subsystem_state(
                    available_subsystem, reliability_history, state, all_completed_tasks, idle_flags,
                    uav_positions=uav_positions
                )
                policy = self._decide_replan_action(subsystem_state)
                decision = policy["decision"]
                p_rebuild = policy["p_rebuild"]
                migrate_mode = policy["migrate_mode"]
                migrate_ratio = policy["migrate_ratio"]
                beta_internal = policy["beta_internal"]
                beta_external = policy["beta_external"]

                if decision == "rebuild":
                    selected_uavs, subsystem_info = self.subsystem_selector.select_subsystem(
                        failed_uav_id=requester_uav_id,
                        all_uavs=UAVs,
                        uav_positions=uav_positions,
                        uav_reliabilities=uav_reliabilities,
                        unfinished_tasks=unfinished_tasks,
                        current_time=time.time() - start_time,
                        max_distance=2000
                    )
                    if requester_uav_id in available_subsystem and requester_uav_id not in selected_uavs:
                        selected_uavs.append(requester_uav_id)
                else:
                    selected_uavs = available_subsystem[:]
                    subsystem_info = {}

                selected_uavs = list(dict.fromkeys([
                    uid for uid in selected_uavs
                    if uid in self.uavs[0] and state[self.uavs[0].index(uid)] == 1
                ]))

                if not selected_uavs:
                    selected_uavs = available_subsystem[:]
                    decision = "internal"
                    migrate_mode = "internal"
                    migrate_ratio = self.internal_migrate_ratio
                    p_rebuild = 0.05
                    beta_internal = 0.95
                    beta_external = 0.05

                type_distribution = {}
                for uid in selected_uavs:
                    idx = self.uavs[0].index(uid)
                    utype = self.uavs[1][idx]
                    type_distribution[utype] = type_distribution.get(utype, 0) + 1
                if not subsystem_info:
                    subsystem_info = {}
                if "reliability" not in subsystem_info:
                    rel_values = [uav_reliabilities.get(uid, 1.0) for uid in selected_uavs]
                    subsystem_info["reliability"] = float(np.mean(rel_values)) if rel_values else 0.0
                if "type_distribution" not in subsystem_info:
                    subsystem_info["type_distribution"] = type_distribution

                # State must be refreshed after action execution (internal/rebuild).
                updated_subsystem_state = self._build_subsystem_state(
                    selected_uavs, reliability_history, state, all_completed_tasks, idle_flags,
                    uav_positions=uav_positions
                )
                self._update_actor_critic(subsystem_state, updated_subsystem_state, decision, migrate_mode)

                subsystem_epoch += 1
                subsystem_id = f"SS-{subsystem_epoch}"
                active_subsystem = selected_uavs
                self._record_policy_point(
                    event_time=time.time() - start_time,
                    subsystem_id=subsystem_id,
                    trigger="assist_reallocation",
                    decision=decision,
                    alpha_rebuild=p_rebuild,
                    beta_internal=beta_internal,
                    beta_external=beta_external,
                    migrate_mode=migrate_mode,
                    migrate_ratio=migrate_ratio,
                    scope=active_subsystem
                )
                print(f"\n [GCS] Assist-reallocation triggered by UAV {requester_uav_id}. "
                      f"Broadcast replanning for subsystem {active_subsystem} (id={subsystem_id}, epoch={subsystem_epoch})")
                print(f"   policy: decision={decision}, alpha={p_rebuild:.3f}, "
                      f"beta={beta_internal:.3f}/{beta_external:.3f}, migrate_ratio={migrate_ratio:.2f}")

                for u in range(uav_num):
                    subsystem_queues[u].put([
                        999, active_subsystem, subsystem_epoch,
                        decision, p_rebuild, migrate_mode, migrate_ratio, updated_subsystem_state, subsystem_id,
                        beta_internal, beta_external
                    ])

            else:
                index = self.uavs[0].index(surveillance[1])
                position[index].append([surveillance[2], surveillance[3], surveillance[4]])
                x[index].append(surveillance[2])
                y[index].append(surveillance[3])
                yaw[index] = surveillance[5]
                yaw_list[index].append(surveillance[5])
                reliability = surveillance[6] if len(surveillance) > 6 else 1.0
                is_idle = int(surveillance[7]) if len(surveillance) > 7 else 0
                idle_flags[surveillance[1]] = bool(is_idle)
                reliability_history[index].append(reliability)
                time_history[index].append(surveillance[4] - start_time)

                if realtime_plot:
                    ax_trajectory.cla()
                    ax_reliability.cla()
                    ax_policy.cla()

                    ax_trajectory.plot(origin_plot_list[0], origin_plot_list[1], 'k^', markerfacecolor='none',
                                       markersize=8)
                    ax_trajectory.plot(targets_plot_list[0], targets_plot_list[1], 'ms', markerfacecolor='none',
                                       markersize=6)
                    ax_trajectory.plot(base_plot_list[0], base_plot_list[1], 'r*', markerfacecolor='none',
                                       markersize=10)

                    subsystem_label = f"Subsystem: {active_subsystem}" if active_subsystem else "No Subsystem"
                    ax_trajectory.set_title(
                        f"t = {np.round(time.time() - start_time, 3)}s | UAV {surveillance[1]} R = {reliability:.4f}\n{subsystem_label}",
                        font0)

                    for u in range(uav_num):
                        line_style = '--' if u + 1 not in active_subsystem and active_subsystem else '-'
                        line_width = 1.5 if u + 1 in active_subsystem else 1.0
                        ax_trajectory.plot(x[u], y[u], line_style, linewidth=line_width, color=color_style[u])
                        ax_trajectory.text(x[u][-1] + 100, y[u][-1] - 120, f'UAV {self.uavs[0][u]}', fontsize='8')
                        if state[u]:
                            yaw_angle = -yaw[u]
                            uav_fuselage = rotation_translation(yaw_angle, fuselage, x[u][-1], y[u][-1])
                            uav_wing = rotation_translation(yaw_angle, wing, x[u][-1], y[u][-1])
                            ax_trajectory.plot(uav_fuselage[0], uav_fuselage[1], 'k-', linewidth=1)
                            ax_trajectory.fill_between(uav_wing[0], uav_wing[1], facecolor="black")

                    for t in range(target_num):
                        ax_trajectory.text(self.targets_sites[t][0] + 100, self.targets_sites[t][1] + 100,
                                           f'Target {t + 1}', font1)
                    for fail in failure_uav_list:
                        ax_trajectory.plot(fail[0], fail[1], 'kx', markersize=12)

                    ax_trajectory.axis("equal")
                    ax_trajectory.set_xlabel('East, m', font0)
                    ax_trajectory.set_ylabel('North, m', font0)
                    ax_trajectory.grid(alpha=0.3)

                    uav_type_labels = ["", "侦察型", "攻击型", "弹药型"]
                    for u in range(uav_num):
                        if len(time_history[u]) > 1:
                            uav_type_name = uav_type_labels[self.uavs[1][u]]
                            line_style = '-' if u + 1 in active_subsystem or not active_subsystem else '--'
                            alpha_val = 1.0 if u + 1 in active_subsystem or not active_subsystem else 0.3
                            ax_reliability.plot(time_history[u], reliability_history[u],
                                                line_style, linewidth=2, markersize=3, alpha=alpha_val,
                                                label=f'UAV {self.uavs[0][u]} ({uav_type_name})', color=color_style[u])

                    ax_reliability.axhline(y=self.failure_threshold, color='r', linestyle='--',
                                           linewidth=2, label=f'故障阈值 ({self.failure_threshold})')
                    ax_reliability.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
                    ax_reliability.set_ylabel('Reliability R(t)', fontsize=11, fontweight='bold')
                    ax_reliability.set_title('实时可靠度演化曲线', fontsize=12, fontweight='bold')
                    ax_reliability.legend(loc='upper right', fontsize=8)
                    ax_reliability.grid(True, alpha=0.3)
                    ax_reliability.set_ylim([0.95, 1.01])

                    if self.policy_history:
                        policy_sorted = sorted(self.policy_history, key=lambda e: e["time"])
                        pt = [e["time"] for e in policy_sorted]
                        pa = [e["alpha_rebuild"] for e in policy_sorted]
                        pb = [e["beta_internal"] for e in policy_sorted]
                        ax_policy.plot(pt, pa, '-o', linewidth=2, markersize=4, color='tab:red',
                                       label='alpha (rebuild)')
                        ax_policy.plot(pt, pb, '-s', linewidth=2, markersize=4, color='tab:blue',
                                       label='beta (internal)')
                        last_event = policy_sorted[-1]
                        ax_policy.set_title(
                            f"策略曲线 | {last_event.get('subsystem_id', 'SS-NA')}",
                            fontsize=12, fontweight='bold'
                        )
                    else:
                        ax_policy.set_title('策略曲线 (alpha / beta)', fontsize=12, fontweight='bold')
                        ax_policy.text(0.5, 0.5, 'No policy event yet', transform=ax_policy.transAxes,
                                       ha='center', va='center', fontsize=10, alpha=0.7)
                    ax_policy.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
                    ax_policy.set_ylabel('Probability', fontsize=11, fontweight='bold')
                    ax_policy.set_ylim([0.0, 1.0])
                    ax_policy.grid(True, alpha=0.3)
                    handles, labels = ax_policy.get_legend_handles_labels()
                    if handles:
                        ax_policy.legend(loc='upper right', fontsize=8)

                    plt.tight_layout()
                    plt.pause(1e-5)

        finalize_simulation_outputs(
            self, realtime_plot, replan2ga, start_time, position, distance,
            active_subsystem, color_style, font, font0, font1, font2, base_plot_list,
            uav_num, reliability_history, failures, target_num, UAVs, time_history, x, y
        )
