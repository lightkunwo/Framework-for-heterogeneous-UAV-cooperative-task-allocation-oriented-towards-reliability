import queue
import time
from math import cos, sin, hypot

import numpy as np
import dubins
from GA_SEAD_process import GA_SEAD, InformationOfUAVs
from dubins_model import angle_between, step_pid
from second_migration import KnowledgeTransferManager


def run_task_allocation_process_replan(simulator, ga2control_queue, control2ga_queue, output_interval=0.5):
    self = simulator
    population, update = None, True
    mission = GA_SEAD(self.targets_sites, 100)
    transfer = KnowledgeTransferManager(self, mission)

    while True:
        uavs = control2ga_queue.get()

        if uavs is None or uavs == [44]:
            break

        if not isinstance(uavs, InformationOfUAVs):
            print(f"[WARN][REPLAN_GA] invalid input type: {type(uavs)}, skip")
            update = False
            continue

        population, migrate_meta = transfer.maybe_apply_event_injection(population, uavs)
        if migrate_meta['attempted']:
            print(
                f"[MIGRATE][{migrate_meta['subsystem_id']}] mode={migrate_meta['migrate_mode']}, "
                f"beta={migrate_meta['beta_internal']:.3f}/{migrate_meta['beta_external']:.3f}, "
                f"ratio={migrate_meta['migrate_ratio']:.2f}, injected={migrate_meta['applied']}"
            )

        solution, population = mission.run_GA_time_period_version(
            output_interval, uavs, population, update, distributed=True
        )

        transfer.update_banks(population, migrate_meta['scope_key'], k=8)
        ga2control_queue.put([solution.fitness_value, solution.chromosome, 'REPLAN'])
        update = False


def run_main_process(simulator, uav, u2u_communication, gcs_init_solution_queue, ga2replan_queue, replan2ga_queue, u2g, uav_failure=None, subsystem_queue=None):
    self = simulator
    targets_sites = self.targets_sites[:]
    x_n = uav.x0
    y_n = uav.y0
    cumulative_distance = 0.0
    e_previous, v, headingRate = None, 0, 0
    theta_n = uav.theta0
    pos = 0

    broadcast_list = [i for i in range(len(u2u_communication)) if not i + 1 == uav.id]
    receive_confirm = False
    interval = 2

    recede_horizon = 1
    path_window = 50
    desire_point_index = 0
    proj_value = 10
    waypoint_radius = 80
    previous_time, previous_broadcast_time, previous_u2g_time = 0, 0, 0
    last_repair_log_time = 0.0

    terminated_tasks, new_targets = [], []
    packets, path, target = [], [], None
    fitness, best_solution = 0, []
    task_confirm = False
    update = False
    into = False
    AT, NT = [], []
    back_to_base, return_pub = False, False
    failure = False
    task_type = ["reconnaissance task", "attack task", "verification task"]

    current_reliability = 1.0
    in_subsystem = False
    subsystem_members = []
    scope_uav_ids = self.uavs[0][:]

    # 子系统epoch，防止旧剔除消息污染新轮次
    local_subsystem_epoch = 0
    assist_replan_requested = False
    last_assist_replan_time = -1e9
    current_replan_decision = "internal"
    current_rebuild_prob = 0.0
    current_migrate_mode = "internal"
    current_migrate_ratio = self.internal_migrate_ratio
    current_beta_internal = 0.50
    current_beta_external = 0.50
    current_subsystem_state = {}
    current_subsystem_id = "SS-0"

    # INIT_EXEC：执行集中式初始解；REPLAN_DIST：分布式重规划
    mode = "INIT_EXEC"

    def pack_broadcast_packet(fit, chromosome, position):
        scoped_chrom = self._project_chromosome_to_scope(chromosome, scope_uav_ids)
        return [uav.id, uav.type, uav.velocity, uav.Rmin, position, uav.depot, fit, scoped_chrom,
                terminated_tasks, new_targets, int(task_confirm), scope_uav_ids[:]], position

    def repack_packets2ga_thread(msg):
        latest_by_id = {}
        for uav_msg in msg:
            latest_by_id[uav_msg[0]] = uav_msg

        ordered_ids = sorted(latest_by_id.keys())
        ordered_msgs = [latest_by_id[uid] for uid in ordered_ids]

        repack_packets = [[] for _ in range(10)]
        fixx = [m[10] for m in ordered_msgs]
        task_accomplished, new_target = [], []

        for uav_msg in ordered_msgs:
            for i in range(7):
                repack_packets[i].append(uav_msg[i])

            repack_packets[7].append(self._uav_msg_chromosome_5rows(uav_msg))

            if uav_msg[8]:
                repack_packets[8].extend(uav_msg[8])
            if uav_msg[9]:
                repack_packets[9].extend(uav_msg[9])

            task_accomplished.extend(uav_msg[8])
            new_target.extend(uav_msg[9])

        uav_info_struct = InformationOfUAVs(
            repack_packets[0], repack_packets[1], repack_packets[4],
            repack_packets[2], repack_packets[3], repack_packets[5],
            uav_best_solution=repack_packets[7]
        )
        uav_info_struct.scope_uav_ids = scope_uav_ids[:]
        uav_info_struct.replan_decision = current_replan_decision
        uav_info_struct.rebuild_prob = current_rebuild_prob
        uav_info_struct.migrate_mode = current_migrate_mode
        uav_info_struct.migrate_ratio = current_migrate_ratio
        uav_info_struct.beta_internal = current_beta_internal
        uav_info_struct.beta_external = current_beta_external
        uav_info_struct.subsystem_state = current_subsystem_state
        uav_info_struct.subsystem_epoch = local_subsystem_epoch
        uav_info_struct.subsystem_id = current_subsystem_id

        task_accomplished = list(dict.fromkeys(tuple(t) if isinstance(t, list) else t for t in task_accomplished))
        new_target = list(dict.fromkeys(tuple(t) if isinstance(t, list) else t for t in new_target))

        return uav_info_struct, fixx, task_accomplished, new_target

    def generate_path(chromosome, position, path, targ, index):
        if chromosome and not back_to_base:
            path_route, task_sequence_state = [], []
            for p in range(len(chromosome[0])):
                if chromosome[3][p] == uav.id:
                    assign_target = chromosome[1][p]
                    assign_heading = chromosome[4][p] * 10
                    task_sequence_state.append([targets_sites[assign_target - 1][0],
                                                targets_sites[assign_target - 1][1], assign_heading,
                                                assign_target, chromosome[2][p]])
            if len(task_sequence_state) == 0:
                return [], [], 0

            task_sequence_state.append(uav.depot)
            for state in task_sequence_state[:-1]:
                state[2] *= np.pi / 180

            dubins_path = dubins.shortest_path(position, task_sequence_state[0][:3], uav.Rmin)
            path_route.extend(dubins_path.sample_many(uav.velocity / 10)[0])
            for p in range(len(task_sequence_state) - 1):
                sp = task_sequence_state[p][:3]
                gp = task_sequence_state[p + 1][:3] if task_sequence_state[p][:3] != task_sequence_state[p + 1][
                                                                                     :3] else \
                    [task_sequence_state[p + 1][0], task_sequence_state[p + 1][1],
                     task_sequence_state[p + 1][2] - 1e-3]
                dubins_path = dubins.shortest_path(sp, gp, uav.Rmin)
                path_route.extend(dubins_path.sample_many(uav.velocity / 10)[0][1:])
            return path_route, task_sequence_state, 0
        elif back_to_base:
            return path, targ, index
        else:
            return [], [], 0

    # 等待GCS下发集中式初始解
    init_packet = gcs_init_solution_queue.get()
    if isinstance(init_packet, (list, tuple)) and len(init_packet) >= 2:
        fitness, best_solution = init_packet[0], init_packet[1]
        path, target, desire_point_index = generate_path(best_solution, [x_n, y_n, theta_n], path, target,
                                                         desire_point_index)

        # 防止无任务机瞬间退出：仅标记返航，不立即44
        if not path or not target:
            back_to_base = True

    start_time = time.time()
    while True:
        current_reliability = uav.get_reliability(time.time() - start_time, cumulative_distance)

        # 子系统消息：只有在成员内才转入分布式重规划
        if subsystem_queue and not subsystem_queue.empty():
            try:
                subsystem_msg = subsystem_queue.get_nowait()

                # [999, selected_uavs, epoch]
                if subsystem_msg[0] == 999:
                    subsystem_members = subsystem_msg[1]
                    msg_epoch = subsystem_msg[2] if len(subsystem_msg) > 2 else 0
                    current_replan_decision = subsystem_msg[3] if len(subsystem_msg) > 3 else "internal"
                    current_rebuild_prob = float(subsystem_msg[4]) if len(subsystem_msg) > 4 else 0.0
                    current_migrate_mode = subsystem_msg[5] if len(subsystem_msg) > 5 else "internal"
                    current_migrate_ratio = float(subsystem_msg[6]) if len(subsystem_msg) > 6 else self.internal_migrate_ratio
                    current_subsystem_state = subsystem_msg[7] if len(subsystem_msg) > 7 and isinstance(
                        subsystem_msg[7], dict) else {}
                    current_subsystem_id = subsystem_msg[8] if len(subsystem_msg) > 8 else f"SS-{msg_epoch}"
                    current_beta_internal = float(subsystem_msg[9]) if len(subsystem_msg) > 9 else 0.50
                    current_beta_external = float(subsystem_msg[10]) if len(subsystem_msg) > 10 else float(
                        np.clip(1.0 - current_beta_internal, 0.05, 0.95))
                    if msg_epoch >= local_subsystem_epoch:
                        local_subsystem_epoch = msg_epoch
                        if isinstance(subsystem_members, list) and uav.id in subsystem_members:
                            in_subsystem = True
                            mode = "REPLAN_DIST"
                            scope_uav_ids = subsystem_members[:]
                            assist_replan_requested = False
                            print(
                                f"✅ [UAV {uav.id}] Joined subsystem(id={current_subsystem_id}, epoch={local_subsystem_epoch}): {subsystem_members}, triggering distributed replanning...")
                            print(
                                f"   policy: decision={current_replan_decision}, alpha={current_rebuild_prob:.3f}, "
                                f"beta={current_beta_internal:.3f}/{current_beta_external:.3f}, migrate_ratio={current_migrate_ratio:.2f}")
                            path, target = [], []
                            # Prevent stale global elites from blocking the first subsystem GA cycle
                            best_solution = self._empty_chromosome_5rows()
                            fitness = 0
                            back_to_base = False
                            update = True
                        else:
                            in_subsystem = False
                            if mode == "REPLAN_DIST":
                                mode = "INIT_EXEC"
                                scope_uav_ids = self.uavs[0][:]
                                receive_confirm = False
                                packets.clear()
                                assist_replan_requested = False
                            print(f"ℹ️  [UAV {uav.id}] Not in subsystem, keep executing current plan.")

                # [998, remove_id, epoch]
                elif subsystem_msg[0] == 998:
                    remove_id = subsystem_msg[1]
                    msg_epoch = subsystem_msg[2] if len(subsystem_msg) > 2 else -1
                    if msg_epoch == local_subsystem_epoch and remove_id in scope_uav_ids:
                        scope_uav_ids.remove(remove_id)
                        print(f"[PRUNE][UAV {uav.id}] remove UAV {remove_id}, new_scope={scope_uav_ids}")

            except Exception as e:
                print(f"⚠️  [UAV {uav.id}] Error processing subsystem message: {e}")

        if uav.check_failure(time.time() - start_time, self.failure_threshold, cumulative_distance):
            print(f'💥 [UAV {uav.id}] FAILED at t={np.round(time.time() - start_time, 3)}s')
            print(f'   Reliability: {current_reliability:.4f} < {self.failure_threshold}')
            uav.is_failed = True
            uav.failure_time = time.time() - start_time
            u2g.put([223, uav.id, x_n, y_n, current_reliability, terminated_tasks])
            print(f"💀 [UAV {uav.id}] Exiting immediately...")
            u2g.put([44, uav.id])
            break

        # 仅REPLAN_DIST模式消费分布式GA结果
        if mode == "REPLAN_DIST":
            while not ga2replan_queue.empty():
                item = ga2replan_queue.get()
                fitness, best_solution = item[0], item[1]
                if best_solution:
                    path, target, desire_point_index = generate_path(best_solution, [x_n, y_n, theta_n],
                                                                     path, target, desire_point_index)
                    if path and target:
                        assist_replan_requested = False

        # 仅REPLAN_DIST模式运行分布式广播/组包/送GA
        if mode == "REPLAN_DIST":
            if int(time.time() * 10) % int(
                    interval * 10) == 0 and time.time() - previous_broadcast_time >= interval / 1.01:
                previous_broadcast_time = time.time()
                current_best_packet, pos = pack_broadcast_packet(fitness, best_solution, [x_n, y_n, theta_n])
                packets.append(current_best_packet)
                for q in broadcast_list:
                    u2u_communication[q].put(current_best_packet)
                terminated_tasks, new_targets = [], []
                receive_confirm = True

            if int(time.time() * 10 % 10) == 3 and receive_confirm:
                while not u2u_communication[uav.id - 1].empty():
                    packets.append(u2u_communication[uav.id - 1].get(timeout=1e-5))

                to_ga_message, fix_target, at, nt = repack_packets2ga_thread(packets)

                expected_scope = set(scope_uav_ids)
                recv_scope = set([uid for uid in to_ga_message.uav_id if uid in expected_scope])
                if len(recv_scope) != len(to_ga_message.uav_id):
                    keep_idx = [i for i, uid in enumerate(to_ga_message.uav_id) if uid in expected_scope]
                    to_ga_message.uav_id = [to_ga_message.uav_id[i] for i in keep_idx]
                    to_ga_message.uav_type = [to_ga_message.uav_type[i] for i in keep_idx]
                    to_ga_message.uav_states = [to_ga_message.uav_states[i] for i in keep_idx]
                    to_ga_message.cruising_speed = [to_ga_message.cruising_speed[i] for i in keep_idx]
                    to_ga_message.turning_radii = [to_ga_message.turning_radii[i] for i in keep_idx]
                    to_ga_message.base = [to_ga_message.base[i] for i in keep_idx]
                    to_ga_message.elite_chromosomes = [to_ga_message.elite_chromosomes[i] for i in keep_idx]
                if uav.id not in expected_scope:
                    packets.clear()
                    receive_confirm = False
                    update = False
                    continue

                if recv_scope != expected_scope:
                    if not recv_scope or (uav.id not in recv_scope):
                        print(
                            f"[DBG][DROP][{current_subsystem_id}] UAV={uav.id} scope empty/not-self. "
                            f"recv={sorted(recv_scope)}, scope={sorted(expected_scope)}")
                        packets.clear()
                        receive_confirm = False
                        update = False
                        continue

                    # Repair-and-push: continue replanning with currently reachable members
                    active_scope = sorted(recv_scope)
                    keep_idx = [i for i, uid in enumerate(to_ga_message.uav_id) if uid in recv_scope]
                    to_ga_message.uav_id = [to_ga_message.uav_id[i] for i in keep_idx]
                    to_ga_message.uav_type = [to_ga_message.uav_type[i] for i in keep_idx]
                    to_ga_message.uav_states = [to_ga_message.uav_states[i] for i in keep_idx]
                    to_ga_message.cruising_speed = [to_ga_message.cruising_speed[i] for i in keep_idx]
                    to_ga_message.turning_radii = [to_ga_message.turning_radii[i] for i in keep_idx]
                    to_ga_message.base = [to_ga_message.base[i] for i in keep_idx]
                    to_ga_message.elite_chromosomes = [to_ga_message.elite_chromosomes[i] for i in keep_idx]
                    to_ga_message.scope_uav_ids = active_scope[:]
                    scope_uav_ids = active_scope[:]
                    print(
                        f"[DBG][REPAIR-PUSH][{current_subsystem_id}] UAV={uav.id} partial scope accepted. "
                        f"recv={active_scope}, expected={sorted(expected_scope)}")

                repaired = 0
                for i, e in enumerate(to_ga_message.elite_chromosomes):
                    if isinstance(e, (list, tuple)) and len(e) == 5:
                        try:
                            is_consistent = set(int(uid) for uid in e[3]).issubset(recv_scope)
                        except Exception:
                            is_consistent = False
                        if not is_consistent:
                            to_ga_message.elite_chromosomes[i] = self._project_chromosome_to_scope(e, recv_scope)
                            repaired += 1
                if repaired > 0:
                    now_ts = time.time()
                    if now_ts - last_repair_log_time >= 2.0:
                        print(
                            f"[DBG][REPAIR][{current_subsystem_id}] UAV={uav.id} repaired {repaired} inconsistent elites. uav_ids={sorted(recv_scope)}")
                        last_repair_log_time = now_ts

                AT.extend(at)
                NT.extend(nt)
                to_ga_message.tasks_completed, to_ga_message.new_targets = AT, NT

                # 不再依赖sum(fix_target)==0，避免卡派发
                if len(to_ga_message.uav_id) > 0:
                    replan2ga_queue.put(to_ga_message)
                    AT, NT = [], []
                    for task in terminated_tasks[:]:
                        if task in to_ga_message.tasks_completed:
                            terminated_tasks.remove(task)
                    for task in new_targets[:]:
                        if task in to_ga_message.new_targets:
                            new_targets.remove(task)
                    update = True
                else:
                    update = False

                packets.clear()
                receive_confirm = False

        # 路径跟踪（所有模式都执行）
        if path and len(target) > 0 and time.time() - previous_time >= .1:
            if hypot(uav.depot[0] - x_n, uav.depot[1] - y_n) <= waypoint_radius and back_to_base and not return_pub:
                print(f'✈️  [UAV {uav.id}] Mission completed: {np.round(time.time() - start_time, 3)}s')
                return_pub = True

            if hypot(target[0][0] - x_n, target[0][1] - y_n) <= waypoint_radius and not into:
                into = True

            if into and hypot(target[0][0] - x_n, target[0][1] - y_n) >= waypoint_radius:
                if target[0][3:]:
                    task_info = target[0][3:]
                    terminated_tasks.append(task_info)
                    event_subsystem_id = current_subsystem_id if (self.enable_subsystem and in_subsystem) else None
                    u2g.put([100, uav.id, task_info[0], task_info[1], time.time(), cumulative_distance,
                             event_subsystem_id])

                    subsystem_tag = "[Subsystem]" if in_subsystem else ""
                    print(f"✅ [UAV {uav.id}] {subsystem_tag} Target {task_info[0]} "
                          f"{task_type[task_info[1] - 1]} finished: {np.round(time.time() - start_time, 3)}s")
                    del target[0]
                    task_confirm = False
                    into = False

            if len(target) > 0 and hypot(target[0][0] - x_n, target[0][1] - y_n) <= 2 * uav.Rmin:
                if target[0][3:]:
                    task_confirm = True
                else:
                    back_to_base = True
            elif len(target) == 0:
                if self.enable_assist_reallocation and in_subsystem and mode == "REPLAN_DIST":
                    remain_tasks = int(current_subsystem_state.get("N_remain", 1)) if isinstance(
                        current_subsystem_state, dict) else 1
                    if remain_tasks <= 0:
                        in_subsystem = False
                        mode = "INIT_EXEC"
                        scope_uav_ids = self.uavs[0][:]
                        assist_replan_requested = False
                        back_to_base = True
                        print(f"🏁 [UAV {uav.id}] No unfinished task remaining, leave subsystem and return to base.")
                    else:
                        back_to_base = False
                        now = time.time()
                        if (not assist_replan_requested) and (
                                now - last_assist_replan_time >= self.assist_replan_cooldown):
                            u2g.put([224, uav.id, x_n, y_n, now, local_subsystem_epoch])
                            assist_replan_requested = True
                            last_assist_replan_time = now
                            print(f"🤝 [UAV {uav.id}] No pending task in subsystem, request assist-reallocation "
                                  f"(epoch={local_subsystem_epoch})")
                else:
                    back_to_base = True

            # 关键修复：防止t=0.0s直接关机
            elapsed = time.time() - start_time
            at_base = hypot(uav.depot[0] - x_n, uav.depot[1] - y_n) <= waypoint_radius
            if (back_to_base and at_base and v <= 0.01 and elapsed > 2.0) and not failure:
                u2g.put([44, uav.id])
                break

            if len(target) > 0:
                future_point = np.array([x_n + uav.velocity * cos(theta_n) * recede_horizon,
                                         y_n + uav.velocity * sin(theta_n) * recede_horizon])
                world_record = 1e10
                desire_point = 0
                start = desire_point_index
                for i in range(start, start + path_window):
                    try:
                        a = np.array([path[i][0], path[i][1]])
                        b = np.array([path[i + 1][0], path[i + 1][1]])
                    except IndexError:
                        a = np.array([path[-2][0], path[-2][1]])
                        b = np.array([path[-1][0], path[-1][1]])
                        i = -2
                    va = future_point - a
                    vb = b - a
                    if np.dot(vb, vb) < 1e-9:
                        continue
                    projection = np.dot(va, vb) / np.dot(vb, vb) * vb
                    normal_point = a + projection
                    if max(path[i][0], path[i + 1][0]) > normal_point[0] > min(path[i][0], path[i + 1][0]) and \
                            max(path[i][1], path[i + 1][1]) > normal_point[1] > min(path[i][1], path[i + 1][1]):
                        normal_point = normal_point[:]
                    else:
                        normal_point = b
                    d = np.linalg.norm(va - (normal_point - a))
                    if d < world_record:
                        world_record = d
                        desire_point = normal_point
                        proj_value = np.dot(va, desire_point - np.array([x_n, y_n]))
                        desire_point_index = i

                angle_between_two_points = angle_between((x_n, y_n), desire_point)
                relative_angle = angle_between_two_points - theta_n
                error_of_heading = relative_angle if abs(relative_angle) <= np.pi else \
                    (-relative_angle / abs(relative_angle)) * (relative_angle + 2 * np.pi)
                difference = error_of_heading - e_previous if e_previous else 0
                u = 3 * error_of_heading + 10 * difference
                if u > v / uav.Rmin:
                    u = v / uav.Rmin
                elif u < -v / uav.Rmin:
                    u = -v / uav.Rmin
                e_previous = error_of_heading

                if proj_value <= 0 and back_to_base:
                    v_command, u = 0, 0
                else:
                    v_command = uav.velocity
                dt = time.time() - previous_time if previous_time != 0 else .1
                previous_time = time.time()
                prev_x_n, prev_y_n = x_n, y_n
                x_n, y_n, theta_n, v, headingRate = step_pid(v, headingRate, x_n, y_n, theta_n, u, v_command, dt)
                cumulative_distance += hypot(x_n - prev_x_n, y_n - prev_y_n)

        if time.time() - previous_u2g_time >= 0.5:
            is_idle = int(len(target) == 0)
            u2g.put([0, uav.id, x_n, y_n, time.time(), theta_n, current_reliability, is_idle])
            previous_u2g_time = time.time()

        if uav_failure:
            if time.time() - start_time >= uav_failure:
                print(f'🔥 [UAV {uav.id}] Manual failure injection at t={np.round(time.time() - start_time, 3)}s')
                uav.is_failed = True
                uav.failure_time = time.time() - start_time
                u2g.put([223, uav.id, x_n, y_n, current_reliability, terminated_tasks])
                print(f"💀 [UAV {uav.id}] Exiting immediately...")
                u2g.put([44, uav.id])
                break
