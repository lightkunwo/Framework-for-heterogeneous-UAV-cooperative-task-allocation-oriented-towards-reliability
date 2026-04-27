import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    TORCH_AVAILABLE = True
except Exception:
    torch = None
    nn = None
    F = None
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class PolicyNet(nn.Module):
        def __init__(self, n_states, n_hiddens, n_actions):
            super(PolicyNet, self).__init__()
            self.fc1 = nn.Linear(n_states, n_hiddens)
            self.fc2 = nn.Linear(n_hiddens, n_actions)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            return F.softmax(x, dim=1)


    class ValueNet(nn.Module):
        def __init__(self, n_states, n_hiddens):
            super(ValueNet, self).__init__()
            self.fc1 = nn.Linear(n_states, n_hiddens)
            self.fc2 = nn.Linear(n_hiddens, 1)

        def forward(self, x):
            x = self.fc1(x)
            x = F.relu(x)
            return self.fc2(x)
else:
    class PolicyNet:
        pass

    class ValueNet:
        pass


def initialize_transfer_policy(simulator):
    simulator.use_actor_critic = True
    simulator.ac_state_dim = 10
    simulator.ac_hidden_dim = 64
    simulator.ac_gamma = 0.95
    simulator.ac_actor_lr = 0.001
    simulator.ac_critic_lr = 0.002
    simulator.use_torch_actor_critic = bool(simulator.use_actor_critic and TORCH_AVAILABLE)
    simulator.ac_device = None
    simulator.ac_actor_alpha = None
    simulator.ac_actor_beta = None
    simulator.ac_critic = None
    simulator.ac_actor_optimizer = None
    simulator.ac_critic_optimizer = None
    if simulator.use_torch_actor_critic:
        simulator.ac_device = torch.device("cpu")
        simulator.ac_actor_alpha = PolicyNet(simulator.ac_state_dim, simulator.ac_hidden_dim, 2).to(simulator.ac_device)
        simulator.ac_actor_beta = PolicyNet(simulator.ac_state_dim, simulator.ac_hidden_dim, 2).to(simulator.ac_device)
        simulator.ac_critic = ValueNet(simulator.ac_state_dim, simulator.ac_hidden_dim).to(simulator.ac_device)
        simulator.ac_actor_optimizer = torch.optim.Adam(
            list(simulator.ac_actor_alpha.parameters()) + list(simulator.ac_actor_beta.parameters()),
            lr=simulator.ac_actor_lr
        )
        simulator.ac_critic_optimizer = torch.optim.Adam(simulator.ac_critic.parameters(), lr=simulator.ac_critic_lr)
    simulator.ac_update_steps = 0


def record_policy_point(simulator, event_time, subsystem_id, trigger, decision,
                        alpha_rebuild, beta_internal, beta_external,
                        migrate_mode, migrate_ratio, scope=None):
    try:
        scope_list = list(scope) if scope else []
    except Exception:
        scope_list = []
    simulator.policy_history.append({
        "time": float(event_time),
        "subsystem_id": str(subsystem_id) if subsystem_id else "SS-NA",
        "trigger": str(trigger),
        "decision": str(decision),
        "alpha_rebuild": float(np.clip(alpha_rebuild, 0.0, 1.0)),
        "beta_internal": float(np.clip(beta_internal, 0.0, 1.0)),
        "beta_external": float(np.clip(beta_external, 0.0, 1.0)),
        "migrate_mode": str(migrate_mode),
        "migrate_ratio": float(max(0.0, migrate_ratio)),
        "scope": scope_list
    })


def _encode_type_code(simulator, subsystem_uav_ids, alive_state):
    counts = {1: 0, 2: 0, 3: 0}
    for uid in subsystem_uav_ids:
        if uid not in simulator.uavs[0]:
            continue
        idx = simulator.uavs[0].index(uid)
        if alive_state[idx] != 1:
            continue
        uav_type = simulator.uavs[1][idx]
        if uav_type in counts:
            counts[uav_type] += 1
    bits = ''.join('1' if counts[t] > 0 else '0' for t in (1, 2, 3))
    nums = ''.join(str(min(9, counts[t])) for t in (1, 2, 3))
    return bits + nums, sum(1 for t in (1, 2, 3) if counts[t] > 0), counts


def _unfinished_task_coords(simulator, unfinished_tasks):
    coords = []
    for task in unfinished_tasks:
        if not task:
            continue
        tid = int(task[0]) - 1
        if 0 <= tid < len(simulator.targets_sites):
            coords.append([float(simulator.targets_sites[tid][0]), float(simulator.targets_sites[tid][1])])
    return coords


def _compute_task_spread(simulator, unfinished_tasks):
    coords = _unfinished_task_coords(simulator, unfinished_tasks)
    if len(coords) <= 1:
        return 0.0
    pts = np.array(coords, dtype=float)
    center = pts.mean(axis=0)
    dists = np.linalg.norm(pts - center, axis=1)
    return float(dists.mean())


def _compute_uav_position_metrics(simulator, alive_members, uav_positions, unfinished_tasks):
    if not alive_members:
        return 0.0, 0.0, 0.0
    uav_xy = []
    for uid in alive_members:
        pos = uav_positions.get(uid, None) if isinstance(uav_positions, dict) else None
        if pos is not None and len(pos) >= 2:
            uav_xy.append([float(pos[0]), float(pos[1])])
        elif uid in simulator.uavs[0]:
            idx = simulator.uavs[0].index(uid)
            init_pos = simulator.uavs[4][idx]
            uav_xy.append([float(init_pos[0]), float(init_pos[1])])
    if len(uav_xy) == 0:
        return 0.0, 0.0, 0.0
    uav_arr = np.array(uav_xy, dtype=float)
    if len(uav_arr) <= 1:
        d_uav_dispersion = 0.0
    else:
        center = uav_arr.mean(axis=0)
        d_uav_dispersion = float(np.linalg.norm(uav_arr - center, axis=1).mean())
    task_coords = _unfinished_task_coords(simulator, unfinished_tasks)
    if len(task_coords) == 0:
        return 0.0, 0.0, d_uav_dispersion
    task_arr = np.array(task_coords, dtype=float)
    dist_matrix = np.linalg.norm(uav_arr[:, None, :] - task_arr[None, :, :], axis=2)
    nearest = dist_matrix.min(axis=1)
    return float(nearest.mean()), float(nearest.std()), d_uav_dispersion


def build_subsystem_state(simulator, subsystem_uav_ids, reliability_history, alive_state, all_completed_tasks,
                          idle_flags, uav_positions=None):
    alive_members = []
    rel_values = []
    for uid in subsystem_uav_ids:
        if uid not in simulator.uavs[0]:
            continue
        idx = simulator.uavs[0].index(uid)
        if alive_state[idx] != 1:
            continue
        alive_members.append(uid)
        rel_hist = reliability_history.get(idx, [1.0])
        rel_values.append(rel_hist[-1] if rel_hist else 1.0)
    if rel_values:
        r_min = float(min(rel_values))
        r_mean = float(sum(rel_values) / len(rel_values))
    else:
        r_min, r_mean = 0.0, 0.0
    type_code, type_presence_count, type_counts = _encode_type_code(simulator, alive_members, alive_state)
    unfinished_tasks = simulator._extract_unfinished_tasks(all_completed_tasks)
    n_remain = len(unfinished_tasks)
    d_spread = _compute_task_spread(simulator, unfinished_tasks)
    d_uav_task, d_uav_task_std, d_uav_dispersion = _compute_uav_position_metrics(
        simulator, alive_members, uav_positions if uav_positions is not None else {}, unfinished_tasks
    )
    n_idle = sum(1 for uid in alive_members if idle_flags.get(uid, False))
    return {
        "R_min": r_min,
        "R_mean": r_mean,
        "N": len(alive_members),
        "TypeCode": type_code,
        "type_presence_count": type_presence_count,
        "type_counts": type_counts,
        "N_remain": n_remain,
        "D_spread": float(d_spread),
        "D_uav_task": float(d_uav_task),
        "D_uav_task_std": float(d_uav_task_std),
        "D_uav_dispersion": float(d_uav_dispersion),
        "N_idle": int(n_idle),
        "members": alive_members,
    }


def _decode_type_presence(subsystem_state):
    if "type_presence_count" in subsystem_state:
        return int(subsystem_state.get("type_presence_count", 0))
    type_code = str(subsystem_state.get("TypeCode", "000000"))
    bits = type_code[:3]
    try:
        return int(sum(1 for b in bits if int(b) > 0))
    except Exception:
        return 0


def _state_to_vector(simulator, subsystem_state):
    n = max(1, int(subsystem_state.get("N", 0)))
    n_total = max(1, len(simulator.uavs[0]))
    total_tasks = max(1, len(simulator.targets_sites) * 3)
    map_diag = max(1e-6, simulator.map_diag)
    r_mean = float(np.clip(subsystem_state.get("R_mean", 0.0), 0.0, 1.0))
    r_min = float(np.clip(subsystem_state.get("R_min", 0.0), 0.0, 1.0))
    type_presence = float(np.clip(_decode_type_presence(subsystem_state) / 3.0, 0.0, 1.0))
    n_norm = float(np.clip(int(subsystem_state.get("N", 0)) / n_total, 0.0, 1.0))
    n_remain = float(np.clip(int(subsystem_state.get("N_remain", 0)) / total_tasks, 0.0, 1.0))
    d_spread = float(np.clip(float(subsystem_state.get("D_spread", 0.0)) / map_diag, 0.0, 1.0))
    d_uav_task = float(np.clip(float(subsystem_state.get("D_uav_task", 0.0)) / map_diag, 0.0, 1.0))
    d_uav_task_std = float(np.clip(float(subsystem_state.get("D_uav_task_std", 0.0)) / max(1e-6, 0.5 * map_diag), 0.0, 1.0))
    d_uav_dispersion = float(np.clip(float(subsystem_state.get("D_uav_dispersion", 0.0)) / map_diag, 0.0, 1.0))
    idle_ratio = float(np.clip(int(subsystem_state.get("N_idle", 0)) / n, 0.0, 1.0))
    return np.array([r_mean, r_min, n_norm, type_presence, n_remain, d_spread,
                     d_uav_task, d_uav_task_std, d_uav_dispersion, idle_ratio], dtype=float)


def _actor_rebuild_prob(simulator, state_vec):
    if not simulator.use_torch_actor_critic:
        return 0.5
    with torch.no_grad():
        s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
        probs = simulator.ac_actor_alpha(s)
        p = float(probs[0, 1].item())
    return float(np.clip(p, 0.05, 0.95))


def _actor_internal_migrate_prob(simulator, state_vec):
    if not simulator.use_torch_actor_critic:
        return 0.5
    with torch.no_grad():
        s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
        probs = simulator.ac_actor_beta(s)
        p = float(probs[0, 1].item())
    return float(np.clip(p, 0.05, 0.95))


def _compute_policy_reward(simulator, prev_state, next_state, decision):
    total_tasks = max(1, len(simulator.targets_sites) * 3)
    map_diag = max(1e-6, simulator.map_diag)
    prev_remain = float(prev_state.get("N_remain", 0))
    next_remain = float(next_state.get("N_remain", 0))
    progress = float(np.clip((prev_remain - next_remain) / total_tasks, -1.0, 1.0))
    prev_r = float(prev_state.get("R_mean", 0.0))
    next_r = float(next_state.get("R_mean", 0.0))
    rel_gain = float(np.clip(next_r - prev_r, -1.0, 1.0))
    prev_type = float(_decode_type_presence(prev_state))
    next_type = float(_decode_type_presence(next_state))
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
    reward = 1.20 * progress + 0.60 * rel_gain + 0.30 * type_gain + 0.25 * spread_gain + 0.15 * cohesion_gain + 0.10 * idle_relief - rebuild_cost
    return float(reward)


def update_actor_critic(simulator, prev_state, next_state, decision, migrate_mode, terminal=False):
    if not simulator.use_torch_actor_critic:
        return
    s = _state_to_vector(simulator, prev_state)
    s_next = _state_to_vector(simulator, next_state)
    action_alpha = 1 if decision == 'rebuild' else 0
    action_beta = 1 if migrate_mode == 'internal' else 0
    state_t = torch.tensor(s[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
    next_state_t = torch.tensor(s_next[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
    action_alpha_t = torch.tensor([[action_alpha]], dtype=torch.long, device=simulator.ac_device)
    action_beta_t = torch.tensor([[action_beta]], dtype=torch.long, device=simulator.ac_device)
    reward = _compute_policy_reward(simulator, prev_state, next_state, decision)
    reward_t = torch.tensor([[reward]], dtype=torch.float32, device=simulator.ac_device)
    with torch.no_grad():
        next_v = torch.zeros((1, 1), dtype=torch.float32, device=simulator.ac_device)
        if not terminal:
            next_v = simulator.ac_critic(next_state_t)
        td_target = reward_t + simulator.ac_gamma * next_v
    value_s = simulator.ac_critic(state_t)
    td_delta = td_target - value_s
    alpha_probs = simulator.ac_actor_alpha(state_t)
    beta_probs = simulator.ac_actor_beta(state_t)
    alpha_log_prob = torch.log(alpha_probs.gather(1, action_alpha_t).clamp(min=1e-8))
    beta_log_prob = torch.log(beta_probs.gather(1, action_beta_t).clamp(min=1e-8))
    actor_loss = -torch.mean((alpha_log_prob + beta_log_prob) * td_delta.detach())
    critic_loss = torch.mean(F.mse_loss(value_s, td_target.detach()))
    simulator.ac_actor_optimizer.zero_grad()
    simulator.ac_critic_optimizer.zero_grad()
    actor_loss.backward()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(simulator.ac_actor_alpha.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(simulator.ac_actor_beta.parameters(), max_norm=5.0)
    torch.nn.utils.clip_grad_norm_(simulator.ac_critic.parameters(), max_norm=5.0)
    simulator.ac_actor_optimizer.step()
    simulator.ac_critic_optimizer.step()
    prob_rebuild = float(alpha_probs[0, 1].detach().cpu().item())
    prob_internal = float(beta_probs[0, 1].detach().cpu().item())
    td_error = float(td_delta.detach().cpu().item())
    simulator.ac_update_steps += 1
    if simulator.ac_update_steps % 25 == 0:
        print(f"[AC] steps={simulator.ac_update_steps}, reward={reward:.3f}, td={td_error:.3f}, alpha={prob_rebuild:.3f}, beta={prob_internal:.3f}")


def decide_replan_action(simulator, subsystem_state):
    n = max(1, int(subsystem_state.get('N', 0)))
    r_mean = float(subsystem_state.get('R_mean', 0.0))
    type_presence_count = _decode_type_presence(subsystem_state)
    n_remain = int(subsystem_state.get('N_remain', 0))
    d_spread = float(subsystem_state.get('D_spread', 0.0))
    d_uav_task = float(subsystem_state.get('D_uav_task', 0.0))
    d_uav_task_std = float(subsystem_state.get('D_uav_task_std', 0.0))
    d_uav_dispersion = float(subsystem_state.get('D_uav_dispersion', 0.0))
    n_idle = int(subsystem_state.get('N_idle', 0))
    rel_risk = float(np.clip(1.0 - r_mean, 0.0, 1.0))
    type_risk = float(np.clip((3 - type_presence_count) / 3.0, 0.0, 1.0))
    remain_pressure = float(np.clip(n_remain / max(1.0, len(simulator.targets_sites) * 3.0), 0.0, 1.0))
    spread_pressure = float(np.clip(d_spread / max(1e-6, simulator.map_diag), 0.0, 1.0))
    uav_task_pressure = float(np.clip(d_uav_task / max(1e-6, simulator.map_diag), 0.0, 1.0))
    uav_task_imbalance = float(np.clip(d_uav_task_std / max(1e-6, 0.5 * simulator.map_diag), 0.0, 1.0))
    formation_pressure = float(np.clip(d_uav_dispersion / max(1e-6, simulator.map_diag), 0.0, 1.0))
    idle_relief = float(np.clip(n_idle / n, 0.0, 1.0))
    if simulator.use_torch_actor_critic:
        state_vec = _state_to_vector(simulator, subsystem_state)
        alpha_rebuild = _actor_rebuild_prob(simulator, state_vec)
        beta_internal = _actor_internal_migrate_prob(simulator, state_vec)
    else:
        score_alpha = 0.10 + 0.28 * rel_risk + 0.20 * type_risk + 0.12 * remain_pressure + 0.08 * spread_pressure + 0.12 * uav_task_pressure + 0.05 * uav_task_imbalance + 0.05 * formation_pressure - 0.10 * idle_relief
        alpha_rebuild = float(np.clip(score_alpha, 0.05, 0.95))
        score_beta = 0.55 + 0.20 * idle_relief + 0.10 * (1.0 - remain_pressure) + 0.05 * (1.0 - spread_pressure) - 0.20 * type_risk - 0.10 * uav_task_pressure - 0.05 * formation_pressure
        beta_internal = float(np.clip(score_beta, 0.05, 0.95))
    beta_external = float(1.0 - beta_internal)
    decision = 'rebuild' if np.random.rand() < alpha_rebuild else 'internal'
    migrate_mode = 'internal' if np.random.rand() < beta_internal else 'external'
    migrate_ratio = simulator.internal_migrate_ratio if migrate_mode == 'internal' else simulator.external_migrate_ratio
    return {
        'alpha_rebuild': alpha_rebuild,
        'p_rebuild': alpha_rebuild,
        'decision': decision,
        'migrate_mode': migrate_mode,
        'migrate_ratio': float(migrate_ratio),
        'beta_internal': beta_internal,
        'beta_external': beta_external,
    }


@dataclass
class KnowledgeTransferState:
    global_elite_bank: list = field(default_factory=list)
    local_elite_bank: dict = field(default_factory=dict)
    prev_transfer_distance: dict = field(default_factory=dict)
    last_migrate_epoch: Optional[int] = None


class KnowledgeTransferManager:
    def __init__(self, simulator, mission):
        self.simulator = simulator
        self.mission = mission
        self.state = KnowledgeTransferState()
        self.task_type_to_uav_type = {
            1: {1, 2},
            2: {2, 3},
            3: {1, 2},
        }

    @staticmethod
    def clone_chromosome(chrom):
        return [list(row) for row in chrom]

    def extract_elites(self, pop, k=8):
        if not pop:
            return []
        ranked = sorted(pop, key=lambda c: getattr(c, 'fitness_value', 0.0), reverse=True)
        elites = []
        for ind in ranked[:k]:
            try:
                if ind.chromosome and len(ind.chromosome) == 5:
                    elites.append(self.clone_chromosome(ind.chromosome))
            except Exception:
                continue
        return elites

    def build_tasktype_uav_ids(self, uav_info):
        mapping = {1: [], 2: [], 3: []}
        uav_ids = list(getattr(uav_info, 'uav_id', []))
        uav_types = list(getattr(uav_info, 'uav_type', []))
        for uid, utype in zip(uav_ids, uav_types):
            for task_kind, allowed_types in self.task_type_to_uav_type.items():
                if utype in allowed_types:
                    mapping[task_kind].append(int(uid))
        for task_kind in mapping:
            mapping[task_kind] = sorted(set(mapping[task_kind]))
        return mapping

    @staticmethod
    def nearest_valid(value, candidates, fallback):
        pool = candidates if candidates else fallback
        if not pool:
            return int(round(value))
        return int(min(pool, key=lambda x: abs(x - value)))

    @staticmethod
    def safe_centers(chrom_list, length):
        if not chrom_list:
            return None
        uav_values = []
        heading_values = []
        for chrom in chrom_list:
            if not chrom or len(chrom) != 5:
                continue
            if len(chrom[0]) != length or len(chrom[3]) != length or len(chrom[4]) != length:
                continue
            uav_values.extend(chrom[3])
            heading_values.extend(chrom[4])
        if not uav_values or not heading_values:
            return None
        return float(np.mean(uav_values)), float(np.mean(heading_values))

    @staticmethod
    def pick_focus_uav(candidate_uavs, target_chrom):
        if not candidate_uavs:
            return None
        if not target_chrom or len(target_chrom) != 5 or len(target_chrom[3]) == 0:
            return int(candidate_uavs[np.random.randint(0, len(candidate_uavs))])
        counter = {int(uid): 0 for uid in candidate_uavs}
        for uid in target_chrom[3]:
            uid_int = int(uid)
            if uid_int in counter:
                counter[uid_int] += 1
        present = [uid for uid, c in counter.items() if c > 0]
        if present:
            return int(max(present, key=lambda uid: counter[uid]))
        return int(candidate_uavs[np.random.randint(0, len(candidate_uavs))])

    @staticmethod
    def safe_centers_for_uav(chrom_list, length, focus_uav):
        if not chrom_list:
            return None
        heading_values = []
        for chrom in chrom_list:
            if not chrom or len(chrom) != 5:
                continue
            if len(chrom[0]) != length or len(chrom[3]) != length or len(chrom[4]) != length:
                continue
            for idx in range(length):
                if int(chrom[3][idx]) == int(focus_uav):
                    heading_values.append(chrom[4][idx])
        if not heading_values:
            return None
        return float(focus_uav), float(np.mean(heading_values))

    def adaptive_radius(self, center_distance, key):
        prev_dist = self.state.prev_transfer_distance.get(key, None)
        if prev_dist is None:
            radius = 1.0
        elif center_distance > prev_dist:
            radius = center_distance * (1.0 + np.random.rand())
        elif center_distance < prev_dist:
            radius = max(0.1, center_distance * np.random.rand())
        else:
            radius = 1.0
        self.state.prev_transfer_distance[key] = center_distance
        if not np.isfinite(radius) or radius <= 0:
            radius = 1.0
        return float(np.clip(radius, 0.1, 2.5))

    def center_map_source_to_target(self, source_chrom, target_chrom, radius,
                                    source_center, target_center,
                                    task_uav_ids, fallback_uav_ids, focus_uav):
        new_chrom = self.clone_chromosome(target_chrom)
        length = len(new_chrom[0])
        src_len = len(source_chrom[0])
        src_center_uav, src_center_heading = source_center
        tgt_center_uav, tgt_center_heading = target_center

        source_indices = [idx for idx in range(src_len) if int(source_chrom[3][idx]) == int(focus_uav)]
        if not source_indices:
            source_indices = list(range(src_len))
        cursor = 0
        changed = 0

        for g in range(length):
            if int(target_chrom[3][g]) != int(focus_uav):
                continue
            src_g = source_indices[cursor % len(source_indices)]
            cursor += 1
            task_kind = int(new_chrom[2][g]) if g < len(new_chrom[2]) else 1
            allowed_uavs = task_uav_ids.get(task_kind, [])
            if int(focus_uav) in allowed_uavs:
                new_chrom[3][g] = int(focus_uav)
            else:
                knowledge_uav = source_chrom[3][src_g]
                mapped_uav = int(round(radius * (knowledge_uav - src_center_uav) + tgt_center_uav))
                new_chrom[3][g] = self.nearest_valid(mapped_uav, allowed_uavs, fallback_uav_ids)

            knowledge_heading = source_chrom[4][src_g]
            mapped_heading = int(round(radius * (knowledge_heading - src_center_heading) + tgt_center_heading))
            mapped_heading = max(0, min(35, mapped_heading))
            new_chrom[4][g] = mapped_heading
            changed += 1

        if changed == 0:
            for g in range(length):
                task_kind = int(new_chrom[2][g]) if g < len(new_chrom[2]) else 1
                allowed_uavs = task_uav_ids.get(task_kind, [])
                if int(focus_uav) in allowed_uavs:
                    src_g = source_indices[0]
                    new_chrom[3][g] = int(focus_uav)
                    knowledge_heading = source_chrom[4][src_g]
                    mapped_heading = int(round(radius * (knowledge_heading - src_center_heading) + tgt_center_heading))
                    mapped_heading = max(0, min(35, mapped_heading))
                    new_chrom[4][g] = mapped_heading
                    changed = 1
                    break

        new_chrom[0] = [idx for idx in range(1, length + 1)]
        return new_chrom, changed

    def inject_elites(self, pop, elites, ratio, scope_key, migrate_mode, uav_info):
        subsystem_id = str(getattr(uav_info, 'subsystem_id', 'SS-NA')) if uav_info is not None else 'SS-NA'
        if not pop or not elites or ratio <= 0:
            return pop, 0, subsystem_id
        inject_num = max(1, int(len(pop) * ratio))
        worst_indices = sorted(range(len(pop)), key=lambda idx: pop[idx].fitness_value)
        task_uav_ids = self.build_tasktype_uav_ids(uav_info)
        fallback_uav_ids = sorted(set(getattr(uav_info, 'uav_id', [])))
        if not fallback_uav_ids:
            fallback_uav_ids = self.simulator.uavs[0][:]
        applied = 0

        for i in range(min(inject_num, len(worst_indices))):
            target_idx = worst_indices[i]
            try:
                target_chrom = self.clone_chromosome(pop[target_idx].chromosome)
            except Exception:
                continue
            if not target_chrom or len(target_chrom) != 5 or len(target_chrom[0]) == 0:
                continue

            gene_len = len(target_chrom[0])
            compatible_sources = [
                e for e in elites
                if isinstance(e, (list, tuple)) and len(e) == 5 and len(e[0]) == gene_len
            ]
            if not compatible_sources:
                continue

            source_chrom = self.clone_chromosome(compatible_sources[i % len(compatible_sources)])
            focus_uav = self.pick_focus_uav(fallback_uav_ids, target_chrom)
            if focus_uav is None:
                continue
            source_center = self.safe_centers_for_uav(compatible_sources, gene_len, focus_uav)
            if source_center is None:
                source_center = self.safe_centers(compatible_sources, gene_len)

            target_pool = []
            for ind in pop:
                try:
                    chrom = ind.chromosome
                    if chrom and len(chrom) == 5 and len(chrom[0]) == gene_len:
                        target_pool.append(self.clone_chromosome(chrom))
                except Exception:
                    continue
            target_center = self.safe_centers_for_uav(target_pool, gene_len, focus_uav)
            if target_center is None:
                target_center = self.safe_centers(target_pool, gene_len)

            if source_center is None or target_center is None:
                pop[target_idx] = self.mission.Chromosome(source_chrom)
                continue

            if len(fallback_uav_ids) >= 2:
                uav_range = float(max(fallback_uav_ids) - min(fallback_uav_ids))
                if uav_range <= 0:
                    uav_range = 1.0
            else:
                uav_range = 1.0
            heading_range = 35.0
            center_distance = float(np.sqrt(
                ((source_center[0] - target_center[0]) / uav_range) ** 2 +
                ((source_center[1] - target_center[1]) / heading_range) ** 2
            ))
            radius_key = (migrate_mode, scope_key, gene_len, int(focus_uav))
            radius = self.adaptive_radius(center_distance, radius_key)
            migrated_chrom, migrated_genes = self.center_map_source_to_target(
                source_chrom, target_chrom, radius,
                source_center, target_center,
                task_uav_ids, fallback_uav_ids, focus_uav
            )
            if migrated_genes > 0:
                pop[target_idx] = self.mission.Chromosome(migrated_chrom)
                applied += 1
        return pop, applied, subsystem_id

    def maybe_apply_event_injection(self, population, uavs):
        scope_key = tuple(sorted(getattr(uavs, 'scope_uav_ids', [])))
        migrate_mode = getattr(uavs, 'migrate_mode', 'internal')
        migrate_ratio = float(getattr(uavs, 'migrate_ratio', 0.0))
        beta_internal = float(np.clip(float(getattr(uavs, 'beta_internal', 0.5)), 0.05, 0.95))
        beta_external = float(1.0 - beta_internal)
        current_epoch = int(getattr(uavs, 'subsystem_epoch', -1))
        allow = current_epoch != self.state.last_migrate_epoch
        attempted = False
        applied = 0
        subsystem_id = str(getattr(uavs, 'subsystem_id', 'SS-NA')) if uavs is not None else 'SS-NA'

        if population is not None and migrate_ratio > 0 and allow:
            if migrate_mode == 'external':
                source_elites = self.state.global_elite_bank
            elif migrate_mode == 'internal':
                source_elites = self.state.local_elite_bank.get(scope_key, [])
            else:
                source_elites = self.state.local_elite_bank.get(scope_key, []) if np.random.rand() < beta_internal else self.state.global_elite_bank
            if source_elites:
                attempted = True
                population, applied, subsystem_id = self.inject_elites(population, source_elites, migrate_ratio, scope_key, migrate_mode, uavs)
                self.state.last_migrate_epoch = current_epoch

        return population, {
            'scope_key': scope_key,
            'migrate_mode': migrate_mode,
            'migrate_ratio': migrate_ratio,
            'beta_internal': beta_internal,
            'beta_external': beta_external,
            'attempted': attempted,
            'applied': applied,
            'subsystem_id': subsystem_id,
            'allow': allow,
        }

    def update_banks(self, population, scope_key, k=8):
        current_elites = self.extract_elites(population, k=k)
        if current_elites:
            self.state.local_elite_bank[scope_key] = current_elites
            self.state.global_elite_bank = (current_elites + self.state.global_elite_bank)[:24]
