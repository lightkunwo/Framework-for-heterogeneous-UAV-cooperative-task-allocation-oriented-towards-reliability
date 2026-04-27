import os
import time

import numpy as np
from matplotlib import pyplot as plt


def generate_time_flow_plot(simulator, position, x, y, targets_sites, UAVs, failures,
                            active_subsystem, color_style, start_time, save_path):
    self = simulator
    print("✓ 生成时间流图...")

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 9}
    font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 11}
    font_target = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
    font_base = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}

    max_length = len(max(position, key=len))
    snapshot_indices = []
    time_points = []
    longest_idx = position.index(max(position, key=len))

    for i in range(1, 7):
        index = max(0, min(int(max_length * i / 6) - 1, max_length - 1))
        snapshot_indices.append(index)
        if index < len(position[longest_idx]):
            time_stamp = position[longest_idx][index][2] - start_time
        else:
            time_stamp = position[longest_idx][-1][2] - start_time
        time_points.append(time_stamp)

    print(f"   Time snapshots: {[f'{t:.2f}s' for t in time_points]}")
    task_colors = {1: 'cyan', 2: 'orange', 3: 'green'}

    for ax, snap_idx, time_stamp in zip(axes, snapshot_indices, time_points):
        ax.set_title(f't = {time_stamp:.2f}s', fontdict=font_title)

        for uav_idx in range(len(position)):
            if snap_idx < len(position[uav_idx]):
                trajectory_x = [p[0] for p in position[uav_idx][:snap_idx + 1]]
                trajectory_y = [p[1] for p in position[uav_idx][:snap_idx + 1]]

                if len(trajectory_x) > 1:
                    line_style = '-' if UAVs[uav_idx].id in active_subsystem or not active_subsystem else '--'
                    ax.plot(
                        trajectory_x, trajectory_y, line_style,
                        linewidth=1.5, color=color_style[uav_idx], alpha=0.7
                    )

                if trajectory_x and trajectory_y:
                    ax.plot(
                        trajectory_x[-1], trajectory_y[-1], '^',
                        markersize=10, color=color_style[uav_idx],
                        markeredgecolor='black', markeredgewidth=1.5
                    )
                    ax.text(
                        trajectory_x[-1] + 100, trajectory_y[-1] - 150,
                        f'UAV {UAVs[uav_idx].id}', fontsize=8, color='black', fontweight='bold'
                    )

                if failures[uav_idx] and snap_idx >= failures[uav_idx]:
                    failure_pos = position[uav_idx][min(failures[uav_idx], len(position[uav_idx]) - 1)]
                    ax.plot(failure_pos[0], failure_pos[1], 'x', markersize=15, color='black', linewidth=3)

        for target_idx, tgt in enumerate(targets_sites):
            ax.plot(tgt[0], tgt[1], 's', markersize=10, color='magenta',
                    markerfacecolor='none', markeredgewidth=2)
            ax.text(tgt[0] + 150, tgt[1] + 150, f'Target {target_idx + 1}', fontdict=font_target)

            for t_type in [1, 2, 3]:
                completed = [
                    log for log in self.task_completion_log
                    if log['target_id'] == target_idx + 1 and
                    log['task_type'] == t_type and
                    log['time'] <= time_stamp
                ]
                executing = [
                    log for log in self.task_completion_log
                    if log['target_id'] == target_idx + 1 and
                    log['task_type'] == t_type and
                    abs(log['time'] - time_stamp) < 3.0
                ]

                angle = (t_type - 1) * 2 * np.pi / 3
                marker_x = tgt[0] + 400 * np.cos(angle)
                marker_y = tgt[1] + 400 * np.sin(angle)

                if completed:
                    ax.plot(marker_x, marker_y, 'x', markersize=12, color=task_colors[t_type], linewidth=3)
                elif executing:
                    ax.plot(marker_x, marker_y, 'o', markersize=14, color=task_colors[t_type],
                            markerfacecolor='none', linewidth=2.5)

        if UAVs:
            ax.plot(UAVs[0].depot[0], UAVs[0].depot[1], '*', markersize=15, color='red',
                    markerfacecolor='none', markeredgewidth=2)
            ax.text(UAVs[0].depot[0] - 150, UAVs[0].depot[1] - 250, 'Base', fontdict=font_base)

        ax.set_xlabel('East, m', fontdict=font)
        ax.set_ylabel('North, m', fontdict=font)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(save_path)
    print(f"✓ 时间流图已保存: {save_path}")
    plt.close()


def finalize_simulation_outputs(simulator, realtime_plot, replan2ga, start_time, position, distance,
                                active_subsystem, color_style, font, font0, font1, font2, base_plot_list,
                                uav_num, reliability_history, failures, target_num, UAVs, time_history, x, y):
    self = simulator
    mission_time = np.round(time.time() - start_time, 3)
    print(" Mission complete!!")

    replan2ga.put([44])

    if realtime_plot:
        plt.close('all')

    print("正在生成图表...\n")

    print(" <<<<<<<<<< Numerical results >>>>>>>>>>> fhh")
    for u, uav in enumerate(position):
        for p in range(1, len(uav) - 1):
            distance[u] += np.linalg.norm(np.subtract(uav[p][:2], uav[p - 1][:2]))
    objectives = mission_time + np.sum(distance) / np.sum(self.uavs[2])
    print(f"Mission time: {mission_time} sec\n"
          f"Total distance: {np.sum(distance):.2f} m\n"
          f"Objectives: {objectives:.3f}\n"
          f"Distance per UAV: {distance}\n")

    print("可靠度统计:")
    uav_type_names = ["", "侦察型", "攻击型", "弹药型"]
    for u in range(uav_num):
        if reliability_history[u]:
            uav_type_name = uav_type_names[self.uavs[1][u]]
            print(
                f"  UAV {self.uavs[0][u]} ({uav_type_name}, α={UAVs[u].reliability_alpha}, β={UAVs[u].reliability_beta}): "
                f"初始={reliability_history[u][0]:.4f}, 末期={reliability_history[u][-1]:.4f}, "
                f"故障={'是' if failures[u] else '否'}"
            )
    print()

    self.print_task_route_summary_task_style()

    print("✓ 生成最终轨迹图...")
    fig, ax = plt.subplots(figsize=(12, 10))
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    for i in range(len(position)):
        line_style = '-' if self.uavs[0][i] in active_subsystem or not active_subsystem else '--'
        line_width = 2.0 if self.uavs[0][i] in active_subsystem else 1.5
        plt.plot([p[0] for p in position[i]], [p[1] for p in position[i]], line_style,
                 linewidth=line_width, color=color_style[i],
                 label=f'UAV {self.uavs[0][i]}' + (' [Subsystem]' if self.uavs[0][i] in active_subsystem else ''))

    for h in range(uav_num):
        plt.text(self.uavs[4][h][0] - 100, self.uavs[4][h][1] - 200, f'UAV {self.uavs[0][h]}', fontsize='8')
        plt.plot(self.uavs[5][h][0], self.uavs[5][h][1], 'r*', markerfacecolor='none', markersize=10)
        if failures[h]:
            plt.plot(position[h][-1][0], position[h][-1][1], 'x', markersize=10, color='black', linewidth=3)

    plt.axis('equal')
    plt.plot([xx[0] for xx in self.uavs[4]], [xx[1] for xx in self.uavs[4]], 'k^', markerfacecolor='none', markersize=8)
    plt.plot([b[0] for b in self.targets_sites], [b[1] for b in self.targets_sites], 'ms',
             label='Target position', markerfacecolor='none', markersize=6)

    for idx, t in enumerate(self.targets_sites, 1):
        plt.text(t[0] + 100, t[1] + 100, f'Target {idx}', font1)
    for info in UAVs:
        plt.text(info.depot[0] - 100, info.depot[1] - 200, 'Base', font2)

    plt.plot(base_plot_list[0], base_plot_list[1], 'r*', markerfacecolor='none', markersize=10, label='Base')
    plt.legend(loc='upper right', prop=font)
    plt.title('UAV Trajectories' + (' [With Subsystem Selection]' if self.enable_subsystem else ''),
              font0, fontsize=14, fontweight='bold')
    plt.xlabel('East, m', font0)
    plt.ylabel('North, m', font0)
    plt.grid(alpha=0.3)

    save_path_1 = os.path.join(self.save_dir, f'01_trajectories_{self.timestamp}.png')
    plt.savefig(save_path_1, dpi=150, bbox_inches='tight')
    print(f"✓ 轨迹图已保存: {save_path_1}")
    plt.close()

    print('✓=> 生成各 UAV 单独路线图...')
    individual_dir = os.path.join(self.save_dir, f'02_individual_routes_{self.timestamp}')
    if not os.path.exists(individual_dir):
        os.makedirs(individual_dir)

    for i in range(uav_num):
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]

        ax.plot([p[0] for p in position[i]], [p[1] for p in position[i]], '-', linewidth=2.0,
                color=color_style[i], label=f'UAV {self.uavs[0][i]}')
        ax.plot(self.uavs[4][i][0], self.uavs[4][i][1], 'k^', markerfacecolor='none', markersize=8)
        ax.plot(self.uavs[5][i][0], self.uavs[5][i][1], 'r*', markerfacecolor='none', markersize=10)
        ax.text(self.uavs[4][i][0] - 100, self.uavs[4][i][1] - 200, f'UAV {self.uavs[0][i]}', fontsize=8)
        ax.text(self.uavs[5][i][0] - 100, self.uavs[5][i][1] - 200, 'Base', font2)

        if failures[i]:
            ax.plot(position[i][-1][0], position[i][-1][1], 'x', markersize=10, color='black', linewidth=3)

        ax.plot([t[0] for t in self.targets_sites], [t[1] for t in self.targets_sites], 'ms',
                markerfacecolor='none', markersize=5, alpha=0.75)
        for idx, t in enumerate(self.targets_sites, 1):
            ax.text(t[0] + 60, t[1] + 60, f'T{idx}', fontsize=6, color='m', alpha=0.8)

        ax.axis('equal')
        ax.set_title(f'UAV {self.uavs[0][i]} Route', fontsize=12, fontweight='bold')
        ax.set_xlabel('East, m', font0)
        ax.set_ylabel('North, m', font0)
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        single_path = os.path.join(individual_dir, f'uav_{self.uavs[0][i]}_route.png')
        fig.savefig(single_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    print(f'✓ 各 UAV 单独路线图已保存到目录: {individual_dir}')

    save_path_3 = os.path.join(self.save_dir, f'03_time_flow_{self.timestamp}.png')
    generate_time_flow_plot(self, position, x, y, self.targets_sites, UAVs, failures,
                            active_subsystem, color_style, start_time, save_path_3)

    print('✓=> 生成可靠度曲线图...')
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(uav_num):
        if len(time_history[i]) > 1:
            uav_type_name = uav_type_names[self.uavs[1][i]]
            line_style = '-' if self.uavs[0][i] in active_subsystem or not active_subsystem else '--'
            ax.plot(
                time_history[i], reliability_history[i], line_style, linewidth=2, markersize=4,
                label=f'UAV {self.uavs[0][i]} ({uav_type_name}, α={UAVs[i].reliability_alpha}, β={UAVs[i].reliability_beta})',
                color=color_style[i]
            )

    ax.axhline(y=self.failure_threshold, color='r', linestyle='--', linewidth=2,
               label=f'故障阈值 ({self.failure_threshold})')
    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reliability R(t)', fontsize=12, fontweight='bold')
    ax.set_title('UAV Reliability Evolution Over Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.95, 1.01])

    save_path_4 = os.path.join(self.save_dir, f'04_reliability_curve_{self.timestamp}.png')
    plt.savefig(save_path_4, dpi=150, bbox_inches='tight')
    print(f'✓ 可靠度曲线已保存: {save_path_4}')
    plt.close()

    print('✓=> 生成策略曲线图 (alpha / beta)...')
    fig, ax = plt.subplots(figsize=(12, 6))
    if self.policy_history:
        policy_sorted = sorted(self.policy_history, key=lambda e: e['time'])
        pt = [e['time'] for e in policy_sorted]
        pa = [e['alpha_rebuild'] for e in policy_sorted]
        pb = [e['beta_internal'] for e in policy_sorted]
        ax.plot(pt, pa, '-o', linewidth=2, markersize=5, color='tab:red', label='alpha (rebuild)')
        ax.plot(pt, pb, '-s', linewidth=2, markersize=5, color='tab:blue', label='beta (internal)')
        for event in policy_sorted:
            if event.get('decision') == 'rebuild':
                ax.scatter(event['time'], event['alpha_rebuild'], color='tab:red', s=45, marker='^', alpha=0.9)
            if event.get('migrate_mode') == 'external':
                ax.scatter(event['time'], event['beta_internal'], color='tab:blue', s=45, marker='v', alpha=0.9)
    else:
        ax.text(0.5, 0.5, 'No policy events recorded', transform=ax.transAxes,
                ha='center', va='center', fontsize=11, alpha=0.75)

    ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
    ax.set_title('Policy Curves: alpha (rebuild) and beta (internal migration)', fontsize=14, fontweight='bold')
    ax.set_ylim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper right', fontsize=10)

    save_path_5 = os.path.join(self.save_dir, f'05_policy_curve_{self.timestamp}.png')
    plt.savefig(save_path_5, dpi=150, bbox_inches='tight')
    print(f'✓ 策略曲线已保存: {save_path_5}')
    plt.close()

    print('✓ 生成统计文件...')
    stats_path = os.path.join(self.save_dir, f'06_mission_stats_{self.timestamp}.txt')
    print('统计文件路径', stats_path)
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write('=' * 70 + '\n')
        f.write('UAV SWARM SEAD MISSION SIMULATION REPORT\n')
        f.write('=' * 70 + '\n\n')
        f.write(f'Simulation Timestamp: {self.timestamp}\n')
        f.write(f'Mission Duration: {mission_time} seconds\n')
        f.write(f'Number of UAVs: {uav_num}\n')
        f.write(f'Number of Targets: {target_num}\n\n')

        if self.subsystem_events:
            f.write('SUBSYSTEM SELECTION EVENTS:\n')
            f.write('-' * 70 + '\n')
            for i, event in enumerate(self.subsystem_events, 1):
                f.write(f'Event {i}:\n')
                f.write(f"  Time: {event['time']:.2f}s\n")
                f.write(f"  Failed UAV: {event['failed_uav']}\n")
                f.write(f"  Selected UAVs: {event['selected_uavs']}\n")
                f.write(f"  Subsystem Reliability: {event['subsystem_reliability']:.6f}\n\n")

        f.write('INITIAL ASSIGNMENT SUMMARY:\n')
        f.write('-' * 70 + '\n')
        uav_type = ['reconnaissance UAV', 'attack UAV', 'munition UAV']
        uav_ids = self.uavs[0]
        uav_types = self.uavs[1]
        init_log = self.initial_assignment_log if hasattr(self, 'initial_assignment_log') else {}
        for j in range(len(uav_ids)):
            uid = uav_ids[j]
            f.write(f'\nUAV{uid} ({uav_type[uav_types[j] - 1]}):\n')
            assigned_tasks = init_log.get(uid, [])
            if len(assigned_tasks) == 0:
                f.write('  No task assigned in initial allocation\n')
            else:
                for item in assigned_tasks:
                    f.write(
                        f"  Target{item['target_id']} {item['task_name']} task | "
                        f"coord=({item['coord'][0]}, {item['coord'][1]})\n"
                    )

        f.write('TASK ROUTE SUMMARY:\n')
        f.write('-' * 70 + '\n')
        task_type = ['reconnaissance', 'attack', 'verification']
        task_route = [[] for _ in range(len(uav_ids))]
        id2idx = {uid: i for i, uid in enumerate(uav_ids)}
        for log in sorted(self.task_completion_log, key=lambda x: x['time']):
            uid = log['uav_id']
            if uid in id2idx:
                task_route[id2idx[uid]].append(log)

        for j in range(len(task_route)):
            f.write(f'\nUAV{uav_ids[j]} ({uav_type[uav_types[j] - 1]}):\n')
            if len(task_route[j]) == 0:
                f.write('  No task executed\n')
            else:
                for log in task_route[j]:
                    target_xy = self.targets_sites[log['target_id'] - 1] if 1 <= log['target_id'] <= len(self.targets_sites) else [None, None]
                    msg = (f"  Target{log['target_id']} {task_type[log['task_type'] - 1]} task | "
                           f"coord=({target_xy[0]}, {target_xy[1]}) | "
                           f"distance={log.get('distance', 0.0):.2f} m | "
                           f"time={log.get('time', 0.0):.2f} s")
                    if self.enable_subsystem and log.get('subsystem_id'):
                        msg += f" | subsystem={log.get('subsystem_id')}"
                    f.write(msg + '\n')

    print(f'✓ 统计文件已保存: {stats_path}\n')
    print(f'所有结果已保存到: {self.save_dir}\n')
