import os
import queue
import random
import time
import builtins
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime
from math import cos, sin, hypot
from typing import Optional

import dubins
import multiprocessing as mp
import numpy as np
from matplotlib import pyplot as plt

from dubins_model import angle_between, step_pid
from GA_SEAD_process import GA_SEAD, InformationOfUAVs
from subsystem import SubsystemSelector
from swarmReliability import ConsecutiveKOutOfN

try:
    from scipy.special import gammainc
    SCIPY_AVAILABLE = True
except Exception:
    gammainc = None
    SCIPY_AVAILABLE = False

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

plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'

DEFAULT_GAMMA_PARAMETER_DICT = {
    1: {"shape_rate": 2.55, "scale": 1.00, "limit": 100.0},
    2: {"shape_rate": 2.85, "scale": 1.00, "limit": 100.0},
    3: {"shape_rate": 2.65, "scale": 1.00, "limit": 100.0},
}

# -------------------- Console Capture (main process) --------------------
_GLOBAL_CONSOLE_LOG = []
_CONSOLE_LOG_SHARED = None
_CONSOLE_CAPTURE_INSTALLED = False
_ORIGINAL_PRINT = builtins.print


def _set_shared_console_log(shared_list):
    global _CONSOLE_LOG_SHARED
    _CONSOLE_LOG_SHARED = shared_list


def _enable_console_capture():
    """Tee print output to in-memory buffer for mission stats."""
    global _CONSOLE_CAPTURE_INSTALLED, _ORIGINAL_PRINT
    if _CONSOLE_CAPTURE_INSTALLED:
        return

    _ORIGINAL_PRINT = builtins.print

    def _tee_print(*args, **kwargs):
        sep = kwargs.get('sep', ' ')
        end = kwargs.get('end', '\n')
        try:
            text = sep.join(str(a) for a in args)
        except Exception:
            text = ' '.join(repr(a) for a in args)
        if end is None:
            end = ''
        if isinstance(end, str) and end not in ('', '\n'):
            text = text + end
        if _CONSOLE_LOG_SHARED is not None:
            try:
                _CONSOLE_LOG_SHARED.append(text)
            except Exception:
                _GLOBAL_CONSOLE_LOG.append(text)
        else:
            _GLOBAL_CONSOLE_LOG.append(text)
            # keep buffer bounded to avoid uncontrolled growth
            if len(_GLOBAL_CONSOLE_LOG) > 50000:
                del _GLOBAL_CONSOLE_LOG[:10000]
        _ORIGINAL_PRINT(*args, **kwargs)

    builtins.print = _tee_print
    _CONSOLE_CAPTURE_INSTALLED = True


def _get_console_log_snapshot():
    if _CONSOLE_LOG_SHARED is not None:
        try:
            return list(_CONSOLE_LOG_SHARED)
        except Exception:
            pass
    return list(_GLOBAL_CONSOLE_LOG)


if TORCH_AVAILABLE:
    class PolicyNet(nn.Module):
        def __init__(self, n_states, n_hiddens, n_actions):
            print("torch available")
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


def initialize_transfer_policy(simulator, enable_rl_decision=True):
    simulator.use_actor_critic = bool(enable_rl_decision)
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
    simulator.use_numpy_actor_critic = bool(simulator.use_actor_critic and (not TORCH_AVAILABLE))
    simulator.ac_np_alpha_w = None
    simulator.ac_np_alpha_b = None
    simulator.ac_np_beta_w = None
    simulator.ac_np_beta_b = None
    simulator.ac_np_critic_w = None
    simulator.ac_np_critic_b = 0.0
    if simulator.use_torch_actor_critic:
        print("useAc")
        simulator.ac_device = torch.device("cpu")
        simulator.ac_actor_alpha = PolicyNet(simulator.ac_state_dim, simulator.ac_hidden_dim, 2).to(simulator.ac_device)
        simulator.ac_actor_beta = PolicyNet(simulator.ac_state_dim, simulator.ac_hidden_dim, 2).to(simulator.ac_device)
        simulator.ac_critic = ValueNet(simulator.ac_state_dim, simulator.ac_hidden_dim).to(simulator.ac_device)
        simulator.ac_actor_optimizer = torch.optim.Adam(
            list(simulator.ac_actor_alpha.parameters()) + list(simulator.ac_actor_beta.parameters()),
            lr=simulator.ac_actor_lr
        )
        simulator.ac_critic_optimizer = torch.optim.Adam(simulator.ac_critic.parameters(), lr=simulator.ac_critic_lr)
    # elif simulator.use_numpy_actor_critic:
    #     rng = np.random.default_rng(42)
    #     scale = 0.05
    #     simulator.ac_np_alpha_w = rng.normal(0.0, scale, size=(simulator.ac_state_dim, 2))
    #     simulator.ac_np_alpha_b = np.zeros(2, dtype=float)
    #     simulator.ac_np_beta_w = rng.normal(0.0, scale, size=(simulator.ac_state_dim, 2))
    #     simulator.ac_np_beta_b = np.zeros(2, dtype=float)
    #     simulator.ac_np_critic_w = rng.normal(0.0, scale, size=(simulator.ac_state_dim,))
    #     simulator.ac_np_critic_b = 0.0
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
    latest_reliability = getattr(simulator, "latest_reliability", None)
    for uid in subsystem_uav_ids:
        if uid not in simulator.uavs[0]:
            continue
        idx = simulator.uavs[0].index(uid)
        if alive_state[idx] != 1:
            continue
        alive_members.append(uid)
        if latest_reliability is not None and len(latest_reliability) > idx:
            rel = float(latest_reliability[idx])
        else:
            rel_hist = reliability_history.get(idx, [1.0])
            rel = rel_hist[-1] if rel_hist else 1.0
        if not np.isfinite(rel):
            print(f"⚠ Invalid reliability for UAV {uid} in subsystem input; clipped to 0.0")
            rel = 0.0
        rel_values.append(float(np.clip(rel, 0.0, 1.0)))
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


def _np_softmax(logits):
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    return ez / max(1e-12, float(np.sum(ez)))


def _np_actor_probs(simulator, state_vec, actor_name):
    s = np.asarray(state_vec, dtype=float)
    if actor_name == "alpha":
        logits = s @ simulator.ac_np_alpha_w + simulator.ac_np_alpha_b
    else:
        logits = s @ simulator.ac_np_beta_w + simulator.ac_np_beta_b
    return _np_softmax(logits)


def _actor_rebuild_prob(simulator, state_vec):
    if simulator.use_torch_actor_critic:
        with torch.no_grad():
            s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
            probs = simulator.ac_actor_alpha(s)
            p = float(probs[0, 1].item())
        return float(np.clip(p, 0.05, 0.95))
    if getattr(simulator, 'use_numpy_actor_critic', False):
        probs = _np_actor_probs(simulator, state_vec, "alpha")
        return float(np.clip(float(probs[1]), 0.05, 0.95))
    else:
        return 0.5


def _actor_internal_migrate_prob(simulator, state_vec):
    if simulator.use_torch_actor_critic:
        with torch.no_grad():
            s = torch.tensor(state_vec[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
            probs = simulator.ac_actor_beta(s)
            p = float(probs[0, 1].item())
        return float(np.clip(p, 0.05, 0.95))
    if getattr(simulator, 'use_numpy_actor_critic', False):
        probs = _np_actor_probs(simulator, state_vec, "beta")
        return float(np.clip(float(probs[1]), 0.05, 0.95))
    else:
        return 0.5


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
    reward = 1.20 * progress + 0.60 * rel_gain + 0.30 * type_gain + 0.25 * spread_gain + 0.15 * cohesion_gain + 0.10 * idle_relief
    return float(reward)


def update_actor_critic(simulator, prev_state, next_state, decision, migrate_mode, terminal=False):
    if not simulator.use_torch_actor_critic and not getattr(simulator, 'use_numpy_actor_critic', False):
        return
    s = _state_to_vector(simulator, prev_state)
    s_next = _state_to_vector(simulator, next_state)
    action_alpha = 1 if decision == 'rebuild' else 0
    action_beta = 1 if migrate_mode == 'internal' else 0
    reward = _compute_policy_reward(simulator, prev_state, next_state, decision)

    if getattr(simulator, 'use_numpy_actor_critic', False):
        # Critic: linear value function V(s)=w^T s + b
        value_s = float(np.dot(simulator.ac_np_critic_w, s) + simulator.ac_np_critic_b)
        next_v = 0.0 if terminal else float(np.dot(simulator.ac_np_critic_w, s_next) + simulator.ac_np_critic_b)
        td_target = reward + simulator.ac_gamma * next_v
        td_delta = float(td_target - value_s)

        # Critic SGD update (MSE gradient)
        simulator.ac_np_critic_w += simulator.ac_critic_lr * td_delta * s
        simulator.ac_np_critic_b += simulator.ac_critic_lr * td_delta

        # Actor policy-gradient update with TD advantage
        alpha_probs = _np_actor_probs(simulator, s, "alpha")
        beta_probs = _np_actor_probs(simulator, s, "beta")
        alpha_onehot = np.zeros(2, dtype=float)
        beta_onehot = np.zeros(2, dtype=float)
        alpha_onehot[action_alpha] = 1.0
        beta_onehot[action_beta] = 1.0

        grad_alpha_logits = alpha_onehot - alpha_probs
        grad_beta_logits = beta_onehot - beta_probs
        simulator.ac_np_alpha_w += simulator.ac_actor_lr * td_delta * np.outer(s, grad_alpha_logits)
        simulator.ac_np_alpha_b += simulator.ac_actor_lr * td_delta * grad_alpha_logits
        simulator.ac_np_beta_w += simulator.ac_actor_lr * td_delta * np.outer(s, grad_beta_logits)
        simulator.ac_np_beta_b += simulator.ac_actor_lr * td_delta * grad_beta_logits

        simulator.ac_update_steps += 1
        if simulator.ac_update_steps % 25 == 0:
            print(
                f"[AC-NP] steps={simulator.ac_update_steps}, reward={reward:.3f}, td={td_delta:.3f}, "
                f"alpha={float(alpha_probs[1]):.3f}, beta={float(beta_probs[1]):.3f}"
            )
        return

    # Torch branch
    state_t = torch.tensor(s[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
    next_state_t = torch.tensor(s_next[np.newaxis, :], dtype=torch.float32, device=simulator.ac_device)
    action_alpha_t = torch.tensor([[action_alpha]], dtype=torch.long, device=simulator.ac_device)
    action_beta_t = torch.tensor([[action_beta]], dtype=torch.long, device=simulator.ac_device)
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
    if simulator.use_torch_actor_critic or getattr(simulator, 'use_numpy_actor_critic', False):
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
    if getattr(simulator, 'enable_knowledge_transfer', True):
        migrate_mode = 'internal' if np.random.rand() < beta_internal else 'external'
        migrate_ratio = simulator.internal_migrate_ratio if migrate_mode == 'internal' else simulator.external_migrate_ratio
    else:
        migrate_mode = 'disabled'
        migrate_ratio = 0.0
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
            3: {1},
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

def run_task_allocation_process_replan(simulator, ga2control_queue, control2ga_queue, output_interval=0.5):
    self = simulator
    _set_shared_console_log(getattr(self, 'console_log_shared', None))
    _enable_console_capture()
    population, update = None, True
    print("replan:",self.targets_sites)#重规划目标
    mission = GA_SEAD(self.targets_sites, 100)
    transfer = KnowledgeTransferManager(self, mission) if self.enable_knowledge_transfer else None
    last_subsystem_epoch = None
    current_scope_uav_ids = []
    current_task_pool = set()
    current_seed = self._empty_chromosome_5rows()

    def _is_nonempty_chromosome_5rows(chrom):
        norm = self._normalize_chromosome_5rows(chrom)
        return len(norm) == 5 and len(norm[0]) > 0

    def _build_completed_task_set(uav_info):
        completed = set()
        snapshot = getattr(uav_info, "subsystem_state", {}) if isinstance(getattr(uav_info, "subsystem_state", {}), dict) else {}
        merged = []
        merged.extend(snapshot.get("completed_tasks_snapshot", []))
        merged.extend(getattr(uav_info, "tasks_completed", []) or [])
        for t in merged:
            if isinstance(t, (list, tuple)) and len(t) >= 2:
                try:
                    completed.add((int(t[0]), int(t[1])))
                except Exception:
                    continue
        return completed

    def _compatible_uav_candidates(task_type, scope_ids, uav_info):
        allowed_types = self.task_type_to_uav_types.get(int(task_type), set())
        id2type = {int(uid): int(ut) for uid, ut in zip(self.uavs[0], self.uavs[1])}
        active = set(int(uid) for uid in getattr(uav_info, "uav_id", []))
        return [uid for uid in scope_ids if uid in active and id2type.get(uid) in allowed_types]

    def _pick_uav_for_task(task_type, target_id, scope_ids, uav_info):
        cands = _compatible_uav_candidates(task_type, scope_ids, uav_info)
        if not cands:
            return None
        target_xy = self.targets_sites[int(target_id) - 1] if 1 <= int(target_id) <= len(self.targets_sites) else [0.0, 0.0]
        state_map = {}
        for uid, st in zip(getattr(uav_info, "uav_id", []), getattr(uav_info, "uav_states", [])):
            try:
                state_map[int(uid)] = st
            except Exception:
                continue
        best_uid = cands[0]
        best_d = float("inf")
        for uid in cands:
            st = state_map.get(uid, [0.0, 0.0, 0.0])
            dx = float(st[0]) - float(target_xy[0])
            dy = float(st[1]) - float(target_xy[1])
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_uid = uid
        return best_uid

    def _build_pool_seed_and_status(uav_info):
        init_chrom = self._normalize_chromosome_5rows(getattr(self, "initial_solution_chromosome", None))
        if len(init_chrom) != 5 or len(init_chrom[0]) == 0:
            return self._empty_chromosome_5rows(), set(), [0 for _ in range(len(self.targets_sites))], []

        scope_ids = getattr(uav_info, "scope_uav_ids", []) or getattr(uav_info, "uav_id", [])
        scope_ids = [int(uid) for uid in scope_ids]
        scope_set = set(scope_ids)

        subsystem_state = getattr(uav_info, "subsystem_state", {})
        failed_uav_id = None
        if isinstance(subsystem_state, dict):
            fv = subsystem_state.get("failed_uav_id", None)
            if fv is not None:
                try:
                    failed_uav_id = int(fv)
                except Exception:
                    failed_uav_id = None

        completed_set = _build_completed_task_set(uav_info)

        rows = [[] for _ in range(5)]
        task_pool = set()
        L = min(len(init_chrom[0]), len(init_chrom[1]), len(init_chrom[2]), len(init_chrom[3]), len(init_chrom[4]))
        ordered_idx = sorted(range(L), key=lambda i: int(init_chrom[0][i]))
        for i in ordered_idx:
            try:
                tid = int(init_chrom[1][i])
                ttype = int(init_chrom[2][i])
                uid = int(init_chrom[3][i])
                heading = int(init_chrom[4][i])
            except Exception:
                continue

            if (tid, ttype) in completed_set:
                continue

            belongs_to_pool = (uid in scope_set) or (failed_uav_id is not None and uid == failed_uav_id)
            if not belongs_to_pool:
                continue

            assign_uid = uid
            if assign_uid not in scope_set:
                reassigned = _pick_uav_for_task(ttype, tid, scope_ids, uav_info)
                if reassigned is None:
                    continue
                assign_uid = reassigned

            rows[0].append(len(rows[0]) + 1)
            rows[1].append(tid)
            rows[2].append(ttype)
            rows[3].append(assign_uid)
            rows[4].append(heading)
            task_pool.add((tid, ttype))

        seed = rows if len(rows[0]) > 0 else self._empty_chromosome_5rows()
        status = [0 for _ in range(len(self.targets_sites))]
        types_by_target = {}
        for tid, ttype in task_pool:
            types_by_target.setdefault(int(tid), set()).add(int(ttype))
        for tid, tset in types_by_target.items():
            if not tset:
                continue
            min_t = min(tset)
            status[tid - 1] = max(status[tid - 1], 4 - min_t)

        return seed, task_pool, status, scope_ids

    def _refresh_task_index_cache():
        mission.remaining_targets = [
            target_id for target_id in range(1, len(mission.targets) + 1)
            if not mission.tasks_status[target_id - 1] == 0
        ]
        mission.task_amount_array = [np.count_nonzero(np.array(mission.tasks_status) >= 3 - t) for t in range(3)]
        mission.task_index_array = [
            0,
            mission.task_amount_array[0],
            mission.task_amount_array[0] + mission.task_amount_array[1],
        ]
        mission.target_sequence = [
            [index for (index, value) in enumerate(mission.tasks_status) if value == task_num]
            for task_num in range(1, 4)
        ]
        mission.target_index_array = [0]
        for k, times in enumerate(mission.tasks_status):
            mission.target_index_array.append(mission.target_index_array[k] + times)

    def _filter_chromosome_to_pool(chrom, pool, scope_ids, uav_info):
        norm = self._normalize_chromosome_5rows(chrom)
        if len(norm) != 5 or len(norm[0]) == 0 or not pool:
            return self._empty_chromosome_5rows()
        pool = set((int(t[0]), int(t[1])) for t in pool)
        scope_set = set(int(uid) for uid in scope_ids)
        rows = [[] for _ in range(5)]
        L = min(len(norm[0]), len(norm[1]), len(norm[2]), len(norm[3]), len(norm[4]))
        ordered_idx = sorted(range(L), key=lambda i: int(norm[0][i]))
        for i in ordered_idx:
            try:
                tid = int(norm[1][i])
                ttype = int(norm[2][i])
                uid = int(norm[3][i])
                heading = int(norm[4][i])
            except Exception:
                continue
            if (tid, ttype) not in pool:
                continue
            assign_uid = uid
            if assign_uid not in scope_set:
                assign_uid = _pick_uav_for_task(ttype, tid, scope_ids, uav_info)
                if assign_uid is None:
                    continue
            rows[0].append(len(rows[0]) + 1)
            rows[1].append(tid)
            rows[2].append(ttype)
            rows[3].append(assign_uid)
            rows[4].append(heading)
        return rows if len(rows[0]) > 0 else self._empty_chromosome_5rows()

    def _filter_population_to_pool(pop, pool, scope_ids, uav_info):
        if not pop or not pool:
            return pop
        expected_len = int(sum(mission.tasks_status))
        filtered = []
        for ind in pop:
            c = getattr(ind, "chromosome", None)
            fc = _filter_chromosome_to_pool(c, pool, scope_ids, uav_info)
            if _is_nonempty_chromosome_5rows(fc) and len(fc[0]) == expected_len:
                filtered.append(mission.Chromosome(fc))
        return filtered

    while True:
        uavs = control2ga_queue.get()

        if uavs is None or uavs == [44]:
            break

        if not isinstance(uavs, InformationOfUAVs):
            print(f"[WARN][REPLAN_GA] invalid input type: {type(uavs)}, skip")
            update = False
            continue

        epoch = int(getattr(uavs, "subsystem_epoch", 0) or 0)
        epoch_changed = (epoch != last_subsystem_epoch)

        if epoch_changed:
            mission = GA_SEAD(self.targets_sites, 100)
            transfer = KnowledgeTransferManager(self, mission) if self.enable_knowledge_transfer else None
            try:
                mission.information_setting(uavs, None, distributed=True)
            except Exception as e:
                print(f"[WARN][REPLAN_GA] init information_setting failed at epoch={epoch}: {e}")

            seed, pool, status, scope_ids = _build_pool_seed_and_status(uavs)
            current_seed = seed
            current_task_pool = pool
            current_scope_uav_ids = scope_ids
            mission.tasks_status = list(status)
            _refresh_task_index_cache()

            expected_genes = int(sum(mission.tasks_status))
            if _is_nonempty_chromosome_5rows(current_seed) and len(current_seed[0]) == expected_genes:
                population = [mission.Chromosome([list(r) for r in current_seed])]
                uavs.elite_chromosomes = [current_seed]
                print(f"[WARM-START] epoch={epoch}, scope={current_scope_uav_ids}, pool_tasks={len(current_task_pool)}, genes={len(current_seed[0])}")
            else:
                population = None
                print(f"[WARM-START] epoch={epoch}, no valid seed (expected_genes={expected_genes}), fallback to GA population.")
            update = False
            last_subsystem_epoch = epoch

        if current_task_pool:
            population = _filter_population_to_pool(population, current_task_pool, current_scope_uav_ids, uavs)
            if (not population) and _is_nonempty_chromosome_5rows(current_seed):
                population = [mission.Chromosome([list(r) for r in current_seed])]

        if transfer is not None:
            population, migrate_meta = transfer.maybe_apply_event_injection(population, uavs)
            if current_task_pool:
                population = _filter_population_to_pool(population, current_task_pool, current_scope_uav_ids, uavs)
                if (not population) and _is_nonempty_chromosome_5rows(current_seed):
                    population = [mission.Chromosome([list(r) for r in current_seed])]
        else:
            migrate_meta = {
                'scope_key': tuple(sorted(getattr(uavs, 'scope_uav_ids', []))),
                'migrate_mode': 'disabled',
                'migrate_ratio': 0.0,
                'beta_internal': 0.5,
                'beta_external': 0.5,
                'attempted': False,
                'applied': 0,
                'subsystem_id': str(getattr(uavs, 'subsystem_id', 'SS-NA')),
                'allow': False,
            }
        if migrate_meta['attempted']:
            print(
                f"[MIGRATE][{migrate_meta['subsystem_id']}] mode={migrate_meta['migrate_mode']}, "
                f"beta={migrate_meta['beta_internal']:.3f}/{migrate_meta['beta_external']:.3f}, "
                f"ratio={migrate_meta['migrate_ratio']:.2f}, injected={migrate_meta['applied']}"
            )

        solution, population = mission.run_GA_time_period_version(
            output_interval, uavs, population, update, distributed=True
        )

        if transfer is not None:
            transfer.update_banks(population, migrate_meta['scope_key'], k=8)
        ga2control_queue.put([solution.fitness_value, solution.chromosome, 'REPLAN'])
        update = False


def run_main_process(simulator, uav, u2u_communication, gcs_init_solution_queue, ga2replan_queue, replan2ga_queue, u2g, uav_failure=None, subsystem_queue=None):
    self = simulator
    _set_shared_console_log(getattr(self, 'console_log_shared', None))
    _enable_console_capture()
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
    current_degradation = 0.0
    current_health = 100.0
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
    waiting_for_replan = False

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
        sim_elapsed = time.time() - start_time
        current_degradation = uav.update_degradation(sim_elapsed)
        current_reliability = uav.get_reliability(sim_elapsed, cumulative_distance)
        current_health = uav.get_health_percentage()
        if not (np.isfinite(current_reliability) and np.isfinite(current_degradation) and np.isfinite(current_health)):
            raise RuntimeError(
                f"UAV {uav.id} invalid Gamma state: "
                f"R={current_reliability}, X={current_degradation}, H={current_health}"
            )

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
                                f" [UAV {uav.id}] Joined subsystem(id={current_subsystem_id}, epoch={local_subsystem_epoch}): {subsystem_members}, triggering distributed replanning...")
                            print(
                                f"   policy: decision={current_replan_decision}, alpha={current_rebuild_prob:.3f}, "
                                f"beta={current_beta_internal:.3f}/{current_beta_external:.3f}, migrate_ratio={current_migrate_ratio:.2f}")
                            path, target = [], []
                            # Prevent stale global elites from blocking the first subsystem GA cycle
                            best_solution = self._empty_chromosome_5rows()
                            fitness = 0
                            back_to_base = False
                            update = True
                            waiting_for_replan = True
                        else:
                            in_subsystem = False
                            if mode == "REPLAN_DIST":
                                mode = "INIT_EXEC"
                                scope_uav_ids = self.uavs[0][:]
                                receive_confirm = False
                                packets.clear()
                                assist_replan_requested = False
                                waiting_for_replan = False
                            print(f"ℹ  [UAV {uav.id}] Not in subsystem, keep executing current plan.")

                # [998, remove_id, epoch]
                elif subsystem_msg[0] == 998:
                    remove_id = subsystem_msg[1]
                    msg_epoch = subsystem_msg[2] if len(subsystem_msg) > 2 else -1
                    if msg_epoch == local_subsystem_epoch and remove_id in scope_uav_ids:
                        scope_uav_ids.remove(remove_id)
                        print(f"[PRUNE][UAV {uav.id}] remove UAV {remove_id}, new_scope={scope_uav_ids}")

            except Exception as e:
                print(f"⚠  [UAV {uav.id}] Error processing subsystem message: {e}")

        if uav.check_failure(sim_elapsed, self.failure_threshold, cumulative_distance,
                             current_reliability=current_reliability):
            print(f'💥 [UAV {uav.id}] FAILED at t={np.round(sim_elapsed, 3)}s')
            print(f'   Reliability: {current_reliability:.4f} < {self.failure_threshold}')
            print(f'   Degradation: {current_degradation:.4f}, Health: {current_health:.2f}%')
            uav.is_failed = True
            uav.failure_time = sim_elapsed
            u2g.put([223, uav.id, x_n, y_n, current_reliability, terminated_tasks,
                     current_degradation, current_health])
            print(f"💀 [UAV {uav.id}] Exiting immediately...")
            u2g.put([44, uav.id])
            break

        # 仅在分布式重规划模式且等待重规划结果时，才处理来自GA的重规划方案
        if mode == "REPLAN_DIST":
            while not ga2replan_queue.empty():
                item = ga2replan_queue.get()
                fitness, best_solution = item[0], item[1]
                if best_solution:
                    path, target, desire_point_index = generate_path(best_solution, [x_n, y_n, theta_n],
                                                                     path, target, desire_point_index)
                    if path and target:
                        assist_replan_requested = False
                    waiting_for_replan = False

        if mode == "REPLAN_DIST" and waiting_for_replan:
            if time.time() - previous_u2g_time >= interval / 1.01:
                previous_u2g_time = time.time()
                # Keep GCS packet format consistent: [msg_type, uav_id, x, y, t, yaw, reliability, is_idle, ...]
                u2g.put([0, uav.id, x_n, y_n, time.time(), theta_n, current_reliability, 1,
                         local_subsystem_epoch, current_degradation, current_health])
            time.sleep(0.02)
            continue

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
            u2g.put([0, uav.id, x_n, y_n, time.time(), theta_n, current_reliability, is_idle,
                     local_subsystem_epoch, current_degradation, current_health])
            previous_u2g_time = time.time()

        if uav_failure:
            if time.time() - start_time >= uav_failure:
                manual_elapsed = time.time() - start_time
                print(f'🔥 [UAV {uav.id}] Manual failure injection at t={np.round(manual_elapsed, 3)}s')
                uav.is_failed = True
                uav.failure_time = manual_elapsed
                u2g.put([223, uav.id, x_n, y_n, current_reliability, terminated_tasks,
                         current_degradation, current_health])
                print(f"💀 [UAV {uav.id}] Exiting immediately...")
                u2g.put([44, uav.id])
                break

def generate_time_flow_plot(simulator, position, x, y, targets_sites, UAVs, failures,
                            active_subsystem, color_style, start_time, save_path):
    self = simulator
    print("✓ 生成时间流图...")

    if not position or all(len(track) == 0 for track in position):
        print("⚠️  时间流图跳过：无可用轨迹数据")
        return

    fig, axes = plt.subplots(3, 2, figsize=(16, 20))
    axes = axes.flatten()

    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 9}
    font_title = {'family': 'Times New Roman', 'weight': 'bold', 'size': 11}
    font_target = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'm', 'size': 8}
    font_base = {'family': 'Times New Roman', 'weight': 'normal', 'color': 'r', 'size': 8}

    longest_track = max(position, key=len)
    max_length = len(longest_track)
    if max_length <= 0:
        print("⚠️  时间流图跳过：轨迹点为空")
        plt.close(fig)
        return
    snapshot_indices = []
    time_points = []
    longest_idx = position.index(longest_track)

    for i in range(1, 7):
        index = max(0, min(int(max_length * i / 6) - 1, max_length - 1))
        snapshot_indices.append(index)
        if index < len(position[longest_idx]):
            raw_t = float(position[longest_idx][index][2])
        else:
            raw_t = float(position[longest_idx][-1][2])
        if raw_t < 1e6:
            time_stamp = max(0.0, raw_t)
        else:
            time_stamp = max(0.0, raw_t - start_time)
        time_points.append(time_stamp)

    print(f"   Time snapshots: {[f'{t:.2f}s' for t in time_points]}")
    task_colors = {1: 'cyan', 2: 'orange', 3: 'green'}

    # 预计算每个 (target, task_type) 的完成时间序列，避免在每个子图里重复全表扫描。
    completion_times = {}
    for log in self.task_completion_log:
        try:
            key = (int(log['target_id']), int(log['task_type']))
            completion_times.setdefault(key, []).append(float(log['time']))
        except Exception:
            continue
    for key in completion_times:
        completion_times[key].sort()

    for panel_idx, (ax, snap_idx, time_stamp) in enumerate(zip(axes, snapshot_indices, time_points), start=1):
        print(f"   rendering snapshot {panel_idx}/6 ...")
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
                times = completion_times.get((target_idx + 1, t_type), [])
                if times:
                    completed = bisect_right(times, time_stamp) > 0
                    left = bisect_left(times, time_stamp - 3.0)
                    executing = left < len(times) and abs(times[left] - time_stamp) < 3.0
                else:
                    completed = False
                    executing = False

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

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(save_path)
    print(f"✓ 时间流图已保存: {save_path}")
    plt.close(fig)


def generate_task_sequence_plot(simulator, save_path):
    self = simulator
    print("✓ 生成任务时序图...")

    if not self.task_completion_log:
        print("⚠️  任务时序图跳过：暂无任务完成记录")
        return

    logs = sorted(self.task_completion_log, key=lambda x: float(x.get('time', 0.0)))
    uav_ids = list(self.uavs[0])
    target_num = len(self.targets_sites)
    prev_finish_time = {uid: 0.0 for uid in uav_ids}
    cmap = plt.get_cmap('tab20')
    uav_colors = {uid: cmap(i % 20) for i, uid in enumerate(uav_ids)}

    fig_h = max(10, int(target_num * 0.35))
    fig, ax = plt.subplots(figsize=(18, fig_h))

    for log in logs:
        try:
            uid = int(log['uav_id'])
            tid = int(log['target_id'])
            t_end = max(0.0, float(log.get('time', 0.0)))
        except Exception:
            continue

        t_start = min(prev_finish_time.get(uid, 0.0), t_end)
        width = max(0.5, t_end - t_start)
        y0 = tid - 0.4
        ax.broken_barh([(t_start, width)], (y0, 0.8),
                       facecolors=uav_colors.get(uid, 'tab:blue'),
                       edgecolors='black', linewidth=0.5, alpha=0.9)
        ax.text(t_start + min(1.0, width * 0.15), tid, f'U{uid}',
                fontsize=6, va='center', ha='left', color='black')
        prev_finish_time[uid] = max(prev_finish_time.get(uid, 0.0), t_end)

    ax.set_title('Task Sequence Timeline', fontsize=14, fontweight='bold')
    ax.set_xlabel('Mission Time (s)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Target', fontsize=12, fontweight='bold')
    ax.set_yticks(range(1, target_num + 1))
    ax.set_yticklabels([f'T{i}' for i in range(1, target_num + 1)], fontsize=8)
    ax.grid(True, axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✓ 任务时序图已保存: {save_path}")
    plt.close(fig)


def finalize_simulation_outputs(simulator, realtime_plot, replan2ga, start_time, position, distance,
                                active_subsystem, color_style, font, font0, font1, font2, base_plot_list,
                                uav_num, reliability_history, failures, target_num, UAVs, time_history, x, y,
                                degradation_history=None, health_history=None, reliability_model_time_history=None):
    self = simulator
    mission_time = np.round(time.time() - start_time, 3)
    if reliability_model_time_history is None:
        reliability_model_time_history = time_history
    if degradation_history is None:
        degradation_history = {i: [0.0 for _ in reliability_history.get(i, [])] for i in range(uav_num)}
    if health_history is None:
        health_history = {i: [100.0 for _ in reliability_history.get(i, [])] for i in range(uav_num)}
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
            health_text = f"健康={health_history[u][-1]:.2f}% " if health_history.get(u) else ""
            print(
                f"  UAV {self.uavs[0][u]} ({uav_type_name}, "
                f"shape_rate={UAVs[u].gamma_shape_rate:.4f}, scale={UAVs[u].gamma_scale:.4f}, "
                f"limit={UAVs[u].degradation_limit:.1f}): "
                f"初始={reliability_history[u][0]:.4f}, 末期={reliability_history[u][-1]:.4f}, "
                f"{health_text}"
                f"故障={'是' if failures[u] else '否'}"
            )
    print()

    if self.final_unfinished_summary is None:
        completed_pairs = [(int(log['target_id']), int(log['task_type'])) for log in self.task_completion_log]
        alive_state = [0 if getattr(UAVs[i], 'is_failed', False) else 1 for i in range(uav_num)]
        self.final_unfinished_summary = self._summarize_unfinished_tasks(completed_pairs, alive_state)
    mission_status = self.final_unfinished_summary
    if mission_status and mission_status.get('unfinished_total', 0) > 0:
        print("未完成任务诊断:")
        print(f"  剩余任务总数: {mission_status['unfinished_total']}")
        print(f"  各类型剩余: {mission_status['unfinished_by_type']}")
        print(f"  各类型不可执行数量: {mission_status['blocked_by_type']}")
        print(f"  当前存活UAV类型: {mission_status['alive_uav_types']}")
        if mission_status.get('sample_blocked_tasks'):
            print(f"  示例不可执行任务: {mission_status['sample_blocked_tasks']}")
        print()

    self.print_task_route_summary_task_style()

    save_path_0 = os.path.join(self.save_dir, f'00_task_sequence_{self.timestamp}.png')
    generate_task_sequence_plot(self, save_path_0)

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

    print('✓=> 生成无人机剩余健康状态图...')
    fig, ax = plt.subplots(figsize=(12, 6))
    health_xmax = 0.0
    for i in range(uav_num):
        if len(reliability_model_time_history[i]) > 1 and len(health_history[i]) > 1:
            line_style = '-' if self.uavs[0][i] in active_subsystem or not active_subsystem else '--'
            n = min(len(reliability_model_time_history[i]), len(health_history[i]))
            plot_t = list(reliability_model_time_history[i][:n])
            plot_h = list(health_history[i][:n])

            # The mission may finish before a UAV reaches zero health. Extend
            # only the saved degradation plot to the expected failure point so
            # the curve shows the full lifetime trend without changing simulation data.
            if plot_t and plot_h and plot_h[-1] > 0.0:
                last_t = float(plot_t[-1])
                if degradation_history and i in degradation_history and len(degradation_history[i]) > 0:
                    last_degradation = float(degradation_history[i][-1])
                else:
                    last_degradation = UAVs[i].degradation_limit * (1.0 - float(plot_h[-1]) / 100.0)
                expected_rate = max(
                    float(UAVs[i].gamma_shape_rate) * float(UAVs[i].gamma_scale),
                    1e-9
                )
                remaining_degradation = max(0.0, float(UAVs[i].degradation_limit) - last_degradation)
                if remaining_degradation > 0.0:
                    plot_t.append(last_t + remaining_degradation / expected_rate)
                    plot_h.append(0.0)

            if plot_t:
                health_xmax = max(health_xmax, max(plot_t))
            ax.plot(
                plot_t, plot_h, line_style,
                linewidth=2, markersize=4, label=f'UAV {self.uavs[0][i]}', color=color_style[i]
            )
    ax.set_xlabel('Model time / h', fontsize=12, fontweight='bold')
    ax.set_ylabel('Remaining health / %', fontsize=12, fontweight='bold')
    ax.set_title('Gamma Degradation Paths of UAVs', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2 if uav_num > 8 else 1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 100.0])
    ax.set_xlim(0.0, max(1.0, health_xmax))
    ax.margins(x=0.0)

    save_path_4 = os.path.join(self.save_dir, f'04_uav_health_degradation_{self.timestamp}.png')
    plt.savefig(save_path_4, dpi=150, bbox_inches='tight')
    print(f'✓ 无人机剩余健康状态图已保存: {save_path_4}')
    plt.close()

    print('✓=> 生成 Gamma 可靠度曲线图...')
    fig, ax = plt.subplots(figsize=(12, 6))
    reliability_xmax = 0.0
    for i in range(uav_num):
        if len(reliability_model_time_history[i]) > 1:
            line_style = '-' if self.uavs[0][i] in active_subsystem or not active_subsystem else '--'
            n = min(len(reliability_model_time_history[i]), len(reliability_history[i]))
            plot_t = list(reliability_model_time_history[i][:n])
            plot_r = list(reliability_history[i][:n])

            # Saved reliability figures should show the full lifetime trend,
            # even when the mission ends before R(t) approaches zero.
            if plot_t and plot_r and plot_r[-1] > 1e-3:
                def _rel_at(model_t):
                    if model_t <= 0.0:
                        return 1.0
                    shape = float(UAVs[i].gamma_shape_rate) * float(model_t)
                    threshold_argument = float(UAVs[i].degradation_limit) / float(UAVs[i].gamma_scale)
                    return float(np.clip(gammainc(shape, threshold_argument), 0.0, 1.0))

                lo = float(plot_t[-1])
                hi = max(lo + 1.0, lo * 1.2 + 1.0)
                for _ in range(80):
                    if _rel_at(hi) <= 1e-3:
                        break
                    hi = hi * 1.5 + 1.0
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if _rel_at(mid) <= 1e-3:
                        hi = mid
                    else:
                        lo = mid
                plot_t.append(hi)
                plot_r.append(0.0)

            if plot_t:
                reliability_xmax = max(reliability_xmax, max(plot_t))
            mark_every = max(1, n // 45)
            ax.plot(
                plot_t, plot_r, line_style,
                linewidth=2, marker='*', markevery=mark_every, markersize=5,
                label=f'UAV {self.uavs[0][i]}',
                color=color_style[i]
            )

    ax.axhline(y=self.failure_threshold, color='r', linestyle='--', linewidth=2,
               label=f'故障阈值 ({self.failure_threshold})')
    ax.set_xlabel('Model time / h', fontsize=12, fontweight='bold')
    ax.set_ylabel('Reliability R(t)', fontsize=12, fontweight='bold')
    ax.set_title('UAV Gamma Reliability Evolution Over Model Time', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2 if uav_num > 8 else 1)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.0, 1.01])
    ax.set_xlim(0.0, max(1.0, reliability_xmax))
    ax.margins(x=0.0)

    save_path_5 = os.path.join(self.save_dir, f'05_uav_gamma_reliability_{self.timestamp}.png')
    plt.savefig(save_path_5, dpi=150, bbox_inches='tight')
    print(f'✓ Gamma 可靠度曲线已保存: {save_path_5}')
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
        f.write(f'Code File: {os.path.basename(__file__)}\n')
        f.write(f'Runtime: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
        f.write(f'Mission Duration: {mission_time} seconds\n')
        f.write(f'Number of UAVs: {uav_num}\n')
        f.write(f'Number of Targets: {target_num}\n\n')

        if self.final_unfinished_summary:
            ms = self.final_unfinished_summary
            f.write('MISSION FEASIBILITY SUMMARY:\n')
            f.write('-' * 70 + '\n')
            f.write(f"Unfinished Total: {ms.get('unfinished_total', 0)}\n")
            f.write(f"Unfinished By Type: {ms.get('unfinished_by_type', {})}\n")
            f.write(f"Blocked By Type: {ms.get('blocked_by_type', {})}\n")
            f.write(f"Feasible Count: {ms.get('feasible_count', 0)}\n")
            f.write(f"Alive UAV Types: {ms.get('alive_uav_types', [])}\n")
            f.write(f"Sample Blocked Tasks: {ms.get('sample_blocked_tasks', [])}\n\n")

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
            if getattr(UAVs[j], 'failure_time', None) is not None:
                f.write(f"  Failure Time: {float(UAVs[j].failure_time):.2f} s\n")
            else:
                f.write("  Failure Time: None (survived)\n")
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

        f.write('\n' + '-' * 70 + '\n')
        f.write('CONSOLE OUTPUT (MAIN PROCESS)\n')
        f.write('-' * 70 + '\n')
        console_lines = _get_console_log_snapshot()
        if console_lines:
            for line in console_lines:
                f.write(str(line) + '\n')
        else:
            f.write('No console output captured.\n')

    print(f'✓ 统计文件已保存: {stats_path}\n')
    print(f'所有结果已保存到: {self.save_dir}\n')

#zhongwen
plt.rcParams['font.family'] = 'WenQuanYi Micro Hei'


class UAV(object):
    def __init__(self, uav_id, uav_type, uav_velocity, uav_Rmin, initial_position, depot,
                 reliability_alpha=0.0001, gamma_shape_rate=None, gamma_scale=None,
                 degradation_limit=None, simulation_seconds_per_model_hour=1.0,
                 degradation_seed=None, individual_factor=1.0):
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Gamma reliability model requires scipy.special.gammainc.")
        self.id = uav_id
        self.type = uav_type
        self.velocity = uav_velocity
        self.Rmin = uav_Rmin
        self.omega_max = self.velocity / self.Rmin
        self.x0 = initial_position[0]
        self.y0 = initial_position[1]
        self.theta0 = initial_position[2]
        self.depot = depot
        # Compatibility only. New reliability calculation uses the Gamma degradation model below.
        self.reliability_alpha = reliability_alpha
        self.gamma_shape_rate = float(gamma_shape_rate if gamma_shape_rate is not None else 2.55)
        self.gamma_scale = float(gamma_scale if gamma_scale is not None else 1.0)
        self.degradation_limit = float(degradation_limit if degradation_limit is not None else 100.0)
        self.simulation_seconds_per_model_hour = float(simulation_seconds_per_model_hour)
        self.degradation_seed = degradation_seed
        self.individual_factor = float(individual_factor)
        self._validate_gamma_parameters()
        self.degradation_state = 0.0
        self.last_degradation_update_time = None
        self.rng = np.random.default_rng(degradation_seed)
        self.is_failed = False
        self.failure_time = None

    def _validate_gamma_parameters(self):
        params = {
            "gamma_shape_rate": self.gamma_shape_rate,
            "gamma_scale": self.gamma_scale,
            "degradation_limit": self.degradation_limit,
            "simulation_seconds_per_model_hour": self.simulation_seconds_per_model_hour,
        }
        for name, value in params.items():
            if (not np.isfinite(value)) or value <= 0.0:
                raise ValueError(f"UAV {self.id} invalid Gamma parameter {name}={value}")

    def get_model_time(self, current_time):
        elapsed_seconds = max(0.0, float(current_time))
        return elapsed_seconds / self.simulation_seconds_per_model_hour

    def update_degradation(self, current_time):
        model_time = self.get_model_time(current_time)
        if self.last_degradation_update_time is None:
            self.last_degradation_update_time = model_time
            return self.degradation_state

        delta_time = max(0.0, model_time - self.last_degradation_update_time)
        self.last_degradation_update_time = model_time
        if delta_time <= 0.0:
            return self.degradation_state

        shape_increment = self.gamma_shape_rate * delta_time
        if shape_increment > 0.0:
            increment = self.rng.gamma(shape=shape_increment, scale=self.gamma_scale)
            if (not np.isfinite(increment)) or increment < 0.0:
                raise RuntimeError(f"UAV {self.id} invalid Gamma degradation increment: {increment}")
            self.degradation_state += float(increment)

        if (not np.isfinite(self.degradation_state)) or self.degradation_state < 0.0:
            raise RuntimeError(f"UAV {self.id} invalid degradation_state={self.degradation_state}")
        return self.degradation_state

    def get_reliability(self, current_time, cumulative_distance=0.0):
        model_time = self.get_model_time(current_time)
        if model_time <= 0.0:
            return 1.0
        shape = self.gamma_shape_rate * model_time
        threshold_argument = self.degradation_limit / self.gamma_scale
        reliability = gammainc(shape, threshold_argument)
        if not np.isfinite(reliability):
            raise RuntimeError(
                f"UAV {self.id} invalid Gamma reliability: "
                f"shape={shape}, limit_over_scale={threshold_argument}, value={reliability}"
            )
        return float(np.clip(reliability, 0.0, 1.0))

    def get_health_percentage(self):
        health = 100.0 * max(0.0, 1.0 - self.degradation_state / self.degradation_limit)
        return float(np.clip(health, 0.0, 100.0))

    def check_failure(self, current_time, threshold=0.98, cumulative_distance=0.0, current_reliability=None):
        if self.is_failed:
            return True
        if current_reliability is None:
            current_reliability = self.get_reliability(current_time, cumulative_distance)
        if not np.isfinite(current_reliability):
            raise RuntimeError(f"UAV {self.id} invalid reliability for failure check: {current_reliability}")
        if current_reliability < threshold:
            self.is_failed = True
            self.failure_time = current_time
            return True
        return False


class DynamicSEADMissionSimulator(object):

    def __init__(self, targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations,
                 reliability_alpha_dict=None, failure_threshold=0.98, save_dir='./simulation_results',
                 enable_subsystem=True, min_member_count=2, min_subsystem_reliability=0.95,
                 enable_assist_reallocation=True, assist_replan_cooldown=2.0,
                 enable_knowledge_transfer=True, enable_rl_decision=True,
                 gamma_parameter_dict=None, simulation_seconds_per_model_hour=1.0,
                 gamma_global_seed=2026, gamma_individual_factor_range=0.04):
        if not SCIPY_AVAILABLE:
            raise RuntimeError("Gamma reliability model requires scipy.special.gammainc.")
        self.targets_sites = targets_sites
        self.uavs = [uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations, [], [], [], []]

        self.reliability_alpha_dict = reliability_alpha_dict or {1: 0.00002, 2: 0.00009, 3: 0.00003}
        self.gamma_parameter_dict = gamma_parameter_dict or DEFAULT_GAMMA_PARAMETER_DICT
        self.simulation_seconds_per_model_hour = float(simulation_seconds_per_model_hour)
        self.gamma_global_seed = int(gamma_global_seed)
        self.gamma_individual_factor_range = float(gamma_individual_factor_range)
        if (not np.isfinite(self.simulation_seconds_per_model_hour)) or self.simulation_seconds_per_model_hour <= 0.0:
            raise ValueError(
                f"simulation_seconds_per_model_hour must be > 0, got {self.simulation_seconds_per_model_hour}"
            )
        if (not np.isfinite(self.gamma_individual_factor_range)) or self.gamma_individual_factor_range < 0.0:
            raise ValueError(
                f"gamma_individual_factor_range must be >= 0, got {self.gamma_individual_factor_range}"
            )
        self.gamma_uav_parameters = self._build_gamma_uav_parameters()
        self.latest_reliability = None
        self.latest_degradation = None
        self.latest_health = None
        self.failure_threshold = failure_threshold
        self.save_dir = save_dir

        self.enable_subsystem = enable_subsystem
        self.min_member_count = max(2, int(min_member_count))
        self.min_subsystem_reliability = min_subsystem_reliability
        self.enable_assist_reallocation = enable_assist_reallocation
        self.assist_replan_cooldown = assist_replan_cooldown
        self.enable_knowledge_transfer = bool(enable_knowledge_transfer)
        self.enable_rl_decision = bool(enable_rl_decision)
        # Task type -> UAV type capability mapping
        # 1: reconnaissance, 2: attack, 3: verification
        self.task_type_to_uav_types = {
            1: {1, 2},
            2: {2, 3},
            3: {1},
        }

        # State-driven reallocation policy parameters
        self.rebuild_threshold_high = 0.65
        self.rebuild_threshold_low = 0.35
        self.external_migrate_ratio = 0.40
        self.internal_migrate_ratio = 0.15

        # Actor-Critic for migration-policy decision (can be disabled by switch)
        initialize_transfer_policy(self, enable_rl_decision=self.enable_rl_decision)

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
                min_subsystem_reliability=self.min_subsystem_reliability,
                min_member_count=self.min_member_count,
                strict_min_member_count=True
            )
            print(f"✅ Subsystem Selector Enabled (type-coverage for type selection, "
                  f"min_members={self.min_member_count}(hard), R_min={min_subsystem_reliability})")
            if self.use_torch_actor_critic:
                print(" Torch Actor-Critic Enabled for subsystem decision-making")
            elif self.use_actor_critic:
                print("Torch unavailable. Fallback to heuristic decision policy.")
            else:
                print("RL Decision Disabled. Using heuristic subsystem decision policy.")
        else:
            self.subsystem_selector = None
            print("子系统未启动")

        self.subsystem_events = []
        self.subsystem_reliability_history = []
        self.task_completion_log = []
        self.policy_history = []
        self.console_log_shared = None
        self.initial_assignment_log = {}
        self.initial_solution_chromosome = self._empty_chromosome_5rows()
        self.final_unfinished_summary = None

        print(f"🤝 Assist Reallocation: {'Enabled' if self.enable_assist_reallocation else 'Disabled'} "
              f"(cooldown={self.assist_replan_cooldown}s)")
        print(f"🔄 Knowledge Transfer: {'Enabled' if self.enable_knowledge_transfer else 'Disabled'}")
        print(f"🧠 RL Decision: {'Enabled' if self.enable_rl_decision else 'Disabled'}")

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.epoch = 0

    def _build_gamma_uav_parameters(self):
        params_by_id = {}
        required_keys = ("shape_rate", "scale", "limit")
        for idx, (uid, utype) in enumerate(zip(self.uavs[0], self.uavs[1])):
            if utype not in self.gamma_parameter_dict:
                raise ValueError(f"Missing Gamma parameters for UAV type {utype} (UAV {uid})")
            base = self.gamma_parameter_dict[utype]
            missing = [k for k in required_keys if k not in base]
            if missing:
                raise ValueError(f"Gamma parameter dict for UAV type {utype} missing keys: {missing}")

            seed_uid = int(uid) if isinstance(uid, (int, np.integer)) else idx + 1
            rng = np.random.default_rng(self.gamma_global_seed + seed_uid)
            if self.gamma_individual_factor_range > 0.0:
                individual_factor = float(rng.uniform(
                    1.0 - self.gamma_individual_factor_range,
                    1.0 + self.gamma_individual_factor_range
                ))
            else:
                individual_factor = 1.0
            shape_rate = float(base["shape_rate"]) * individual_factor
            gamma_scale = float(base["scale"])
            degradation_limit = float(base["limit"])
            degradation_seed = int(self.gamma_global_seed + 10000 + seed_uid)

            for name, value in (
                ("shape_rate", shape_rate),
                ("scale", gamma_scale),
                ("limit", degradation_limit),
            ):
                if (not np.isfinite(value)) or value <= 0.0:
                    raise ValueError(f"UAV {uid} invalid Gamma {name}={value}")

            params_by_id[uid] = {
                "shape_rate": shape_rate,
                "scale": gamma_scale,
                "limit": degradation_limit,
                "simulation_seconds_per_model_hour": self.simulation_seconds_per_model_hour,
                "degradation_seed": degradation_seed,
                "individual_factor": individual_factor,
            }
        return params_by_id

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
        mission = GA_SEAD(self.targets_sites, 300)
        uavs_info = InformationOfUAVs(
            self.uavs[0], self.uavs[1], self.uavs[4], self.uavs[2], self.uavs[3], self.uavs[5],
            uav_best_solution=[self._empty_chromosome_5rows() for _ in self.uavs[0]]
        )
        solution, _ = mission.run_GA_time_period_version(0.5, uavs_info, None, True, distributed=True)
        return solution.fitness_value, solution.chromosome

    def collect_uav_states(self, position, reliability_history, UAVs):
        uav_positions = {}
        uav_reliabilities = {}
        latest_reliability = getattr(self, "latest_reliability", None)
        for i, uav in enumerate(UAVs):
            if position[i]:
                last_pos = position[i][-1]
                uav_positions[uav.id] = last_pos[:3] if len(last_pos) >= 3 else [last_pos[0], last_pos[1], 0]
            else:
                uav_positions[uav.id] = [uav.x0, uav.y0, uav.theta0]

            if latest_reliability is not None and len(latest_reliability) > i:
                rel = float(latest_reliability[i])
            elif reliability_history[i]:
                rel = reliability_history[i][-1]
            else:
                rel = 1.0
            if getattr(uav, "is_failed", False):
                rel = 0.0
            if not np.isfinite(rel):
                print(f"⚠ Invalid latest reliability for UAV {uav.id}; clipped to 0.0")
                rel = 0.0
            uav_reliabilities[uav.id] = float(np.clip(rel, 0.0, 1.0))

        return uav_positions, uav_reliabilities

    def _extract_unfinished_tasks(self, completed_tasks_list):
        unfinished = []
        completed_set = set(completed_tasks_list)
        for i in range(len(self.targets_sites)):
            for task_type in [1, 2, 3]:
                task_tuple = (i + 1, task_type)
                if task_tuple not in completed_set:
                    unfinished.append(task_tuple)
        return unfinished

    def _alive_uav_types(self, alive_state):
        alive_types = set()
        for i, uid in enumerate(self.uavs[0]):
            if i < len(alive_state) and alive_state[i] == 1:
                alive_types.add(int(self.uavs[1][i]))
        return alive_types

    def _summarize_unfinished_tasks(self, completed_tasks_list, alive_state):
        unfinished = self._extract_unfinished_tasks(completed_tasks_list)
        summary = {
            'unfinished_total': len(unfinished),
            'unfinished_by_type': {1: 0, 2: 0, 3: 0},
            'blocked_by_type': {1: 0, 2: 0, 3: 0},
            'feasible_count': 0,
            'alive_uav_types': sorted(self._alive_uav_types(alive_state)),
            'sample_blocked_tasks': [],
        }
        alive_types = set(summary['alive_uav_types'])
        for task in unfinished:
            tid, ttype = int(task[0]), int(task[1])
            summary['unfinished_by_type'][ttype] = summary['unfinished_by_type'].get(ttype, 0) + 1
            allowed_types = self.task_type_to_uav_types.get(ttype, set())
            if allowed_types & alive_types:
                summary['feasible_count'] += 1
            else:
                summary['blocked_by_type'][ttype] = summary['blocked_by_type'].get(ttype, 0) + 1
                if len(summary['sample_blocked_tasks']) < 8:
                    summary['sample_blocked_tasks'].append((tid, ttype))
        return summary

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
        reward = (
            1.20 * progress
            + 0.60 * rel_gain
            + 0.30 * type_gain
            + 0.25 * spread_gain
            + 0.15 * cohesion_gain
            + 0.10 * idle_relief
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
        return generate_time_flow_plot(
            self, position, x, y, targets_sites, UAVs, failures,
            active_subsystem, color_style, start_time, save_path
        )

    def start_simulation(self, realtime_plot=False, uav_failure=None):
        _enable_console_capture()
        uav_num = len(self.uavs[0])
        target_num = len(self.targets_sites)
        console_manager = mp.Manager()
        self.console_log_shared = console_manager.list()
        _set_shared_console_log(self.console_log_shared)

        u2u_nodes = [mp.Queue() for _ in range(uav_num)]
        GCS = mp.Queue()
        subsystem_queues = [mp.Queue() for _ in range(uav_num)]
        uav_failure = [None for _ in range(uav_num)] if not uav_failure else uav_failure

        # 分布式重规划GA队列（仅REPLAN用）
        replan2ga = mp.Queue()
        ga2replan = mp.Queue()

        # GCS -> 每个UAV 的初始解下发队列
        gcs_init_solution_queues = [mp.Queue() for _ in range(uav_num)]

        UAVs = []
        for n in range(uav_num):
            uid = self.uavs[0][n]
            params = self.gamma_uav_parameters[uid]
            UAVs.append(UAV(
                uid, self.uavs[1][n], self.uavs[2][n], self.uavs[3][n],
                self.uavs[4][n], self.uavs[5][n],
                reliability_alpha=self.reliability_alpha_dict.get(self.uavs[1][n], 0.0),
                gamma_shape_rate=params["shape_rate"],
                gamma_scale=params["scale"],
                degradation_limit=params["limit"],
                simulation_seconds_per_model_hour=params["simulation_seconds_per_model_hour"],
                degradation_seed=params["degradation_seed"],
                individual_factor=params["individual_factor"],
            ))

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
        self.latest_reliability = np.ones(uav_num, dtype=float)
        self.latest_degradation = np.zeros(uav_num, dtype=float)
        self.latest_health = np.full(uav_num, 100.0, dtype=float)
        reliability_history = {i: [1.0] for i in range(uav_num)}
        degradation_history = {i: [0.0] for i in range(uav_num)}
        health_history = {i: [100.0] for i in range(uav_num)}
        time_history = {i: [0.0] for i in range(uav_num)}
        reliability_model_time_history = {i: [0.0] for i in range(uav_num)}
        active_subsystem = []
        all_completed_tasks = []
        idle_flags = {uid: False for uid in self.uavs[0]}
        gamma_debug_printed = False

        # 子系统轮次标识
        subsystem_epoch = 0

        color_style = ['tab:blue', 'tab:green', 'tab:orange', '#DC143C', '#808080', '#030764', '#C875C4', '#008080',
                       '#DAA520', '#580F41', '#7BC8F6', '#06C2AC', '#2E8B57', '#FF8C00', '#8A2BE2', '#708090']
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
        self.initial_solution_chromosome = self._normalize_chromosome_5rows(init_solution)
        self.initial_assignment_log = self._build_initial_assignment_log(init_solution)
        for q in gcs_init_solution_queues:
            q.put([init_fit, init_solution])

        start_time = time.time()
        print("模拟开始!")
        print(f"故障阈值: {self.failure_threshold}")
        print("Gamma 单机可靠度模型参数:")
        print(f"  simulation_seconds_per_model_hour = {self.simulation_seconds_per_model_hour}")
        for uav in UAVs:
            uav_type_name = ["", "侦察型", "攻击型", "弹药型"][uav.type]
            print(
                f"  UAV {uav.id} | type={uav.type}({uav_type_name}) | "
                f"gamma_shape_rate={uav.gamma_shape_rate:.6f} | "
                f"gamma_scale={uav.gamma_scale:.6f} | "
                f"degradation_limit={uav.degradation_limit:.6f} | "
                f"seed={uav.degradation_seed} | factor={uav.individual_factor:.5f}"
            )
        print(f"子系统选择: {'启用' if self.enable_subsystem else '禁用'}")
        if self.enable_subsystem:
            print(f"  Min Members: {self.min_member_count} (hard)")
            print(f"  Min Reliability: {self.min_subsystem_reliability}")
        print(f"知识迁移: {'启用' if self.enable_knowledge_transfer else '禁用'}")
        print(f"强化学习决策: {'启用' if self.enable_rl_decision else '禁用'}")
        print("=" * 70)

        while state != completed:
            try:
                surveillance = GCS.get(timeout=1.0)
            except queue.Empty:
                for i, proc in enumerate(main_processes):
                    if state[i] == 1 and (not proc.is_alive()):
                        state[i] = 0
                        if self.latest_reliability is not None and len(self.latest_reliability) > i:
                            self.latest_reliability[i] = 0.0
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
                        updated_subsystem_state["failed_uav_id"] = None
                        updated_subsystem_state["completed_tasks_snapshot"] = [
                            [int(t[0]), int(t[1])] if isinstance(t, (list, tuple)) and len(t) >= 2 else t
                            for t in all_completed_tasks
                        ]
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
                failure_reliability = float(surveillance[4])
                failed_terminated_tasks = surveillance[5] if len(surveillance) > 5 else []
                failure_degradation = float(surveillance[6]) if len(surveillance) > 6 else 0.0
                failure_health = float(surveillance[7]) if len(surveillance) > 7 else 100.0

                failed_idx = self.uavs[0].index(failed_uav_id)
                failure_rel_time = time.time() - start_time
                failure_model_time = failure_rel_time / self.simulation_seconds_per_model_hour
                state[failed_idx] = 0
                reliability_history[failed_idx].append(float(np.clip(failure_reliability, 0.0, 1.0)))
                degradation_history[failed_idx].append(float(max(0.0, failure_degradation)))
                health_history[failed_idx].append(float(np.clip(failure_health, 0.0, 100.0)))
                time_history[failed_idx].append(failure_rel_time)
                reliability_model_time_history[failed_idx].append(failure_model_time)
                self.latest_reliability[failed_idx] = 0.0
                self.latest_degradation[failed_idx] = float(max(0.0, failure_degradation))
                self.latest_health[failed_idx] = float(np.clip(failure_health, 0.0, 100.0))
                # Sync failure state in GCS-side UAV objects immediately.
                if 0 <= failed_idx < len(UAVs):
                    UAVs[failed_idx].is_failed = True
                    UAVs[failed_idx].failure_time = failure_rel_time
                failures[failed_idx] = max(0, len(position[failed_idx]) - 1)
                failure_uav_list.append(failure_pos)


                print(f"UAV {failed_uav_id} FAILED at t={np.round(failure_rel_time, 2)}s")
                print(f"Reliability: {failure_reliability:.4f}")
                print(f"Gamma degradation: {failure_degradation:.4f}, health: {failure_health:.2f}%")
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
                    updated_subsystem_state["failed_uav_id"] = int(failed_uav_id)
                    updated_subsystem_state["completed_tasks_snapshot"] = [
                        [int(t[0]), int(t[1])] if isinstance(t, (list, tuple)) and len(t) >= 2 else t
                        for t in all_completed_tasks
                    ]
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
                shutdown_idx = self.uavs[0].index(shutdown_uav_id)
                state[shutdown_idx] = 0
                if self.latest_reliability is not None and len(self.latest_reliability) > shutdown_idx:
                    self.latest_reliability[shutdown_idx] = 0.0
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
                updated_subsystem_state["failed_uav_id"] = None
                updated_subsystem_state["completed_tasks_snapshot"] = [
                    [int(t[0]), int(t[1])] if isinstance(t, (list, tuple)) and len(t) >= 2 else t
                    for t in all_completed_tasks
                ]
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
                raw_t = float(surveillance[4]) if len(surveillance) > 4 else 0.0
                # Normalize mixed time sources:
                # - absolute epoch time (time.time())
                # - relative mission time (time.time()-start_time)
                if raw_t < 1e6:
                    rel_t = max(0.0, raw_t)
                    abs_t = start_time + rel_t
                else:
                    abs_t = raw_t
                    rel_t = max(0.0, raw_t - start_time)

                position[index].append([surveillance[2], surveillance[3], abs_t])
                x[index].append(surveillance[2])
                y[index].append(surveillance[3])
                yaw[index] = surveillance[5]
                yaw_list[index].append(surveillance[5])
                reliability = float(surveillance[6]) if len(surveillance) > 6 else 1.0
                is_idle = int(surveillance[7]) if len(surveillance) > 7 else 0
                degradation = float(surveillance[9]) if len(surveillance) > 9 else 0.0
                health = float(surveillance[10]) if len(surveillance) > 10 else 100.0
                if not np.isfinite(reliability):
                    print(f"⚠ Invalid reliability packet from UAV {surveillance[1]}; clipped to 0.0")
                    reliability = 0.0
                if not np.isfinite(degradation):
                    print(f"⚠ Invalid degradation packet from UAV {surveillance[1]}; clipped to 0.0")
                    degradation = 0.0
                if not np.isfinite(health):
                    print(f"⚠ Invalid health packet from UAV {surveillance[1]}; clipped to 100.0")
                    health = 100.0
                reliability = float(np.clip(reliability, 0.0, 1.0))
                degradation = float(max(0.0, degradation))
                health = float(np.clip(health, 0.0, 100.0))
                model_t = rel_t / self.simulation_seconds_per_model_hour
                idle_flags[surveillance[1]] = bool(is_idle)
                self.latest_reliability[index] = reliability if state[index] == 1 else 0.0
                self.latest_degradation[index] = degradation
                self.latest_health[index] = health
                reliability_history[index].append(reliability)
                degradation_history[index].append(degradation)
                health_history[index].append(health)
                time_history[index].append(rel_t)
                reliability_model_time_history[index].append(model_t)

                if not gamma_debug_printed:
                    print(
                        "[GammaFlow] "
                        f"UAV ID={surveillance[1]}, current reliability={reliability:.6f}, "
                        f"current degradation={degradation:.6f}, current health={health:.2f}%, "
                        f"current subsystem/swarm reliability input={self.latest_reliability[index]:.6f}"
                    )
                    gamma_debug_printed = True

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
                        if len(reliability_model_time_history[u]) > 1:
                            uav_type_name = uav_type_labels[self.uavs[1][u]]
                            line_style = '-' if u + 1 in active_subsystem or not active_subsystem else '--'
                            alpha_val = 1.0 if u + 1 in active_subsystem or not active_subsystem else 0.3
                            ax_reliability.plot(reliability_model_time_history[u], reliability_history[u],
                                                line_style, linewidth=2, markersize=3, alpha=alpha_val,
                                                label=f'UAV {self.uavs[0][u]} ({uav_type_name})', color=color_style[u])

                    ax_reliability.axhline(y=self.failure_threshold, color='r', linestyle='--',
                                           linewidth=2, label=f'故障阈值 ({self.failure_threshold})')
                    ax_reliability.set_xlabel('Model time / h', fontsize=11, fontweight='bold')
                    ax_reliability.set_ylabel('Reliability R(t)', fontsize=11, fontweight='bold')
                    ax_reliability.set_title('实时 Gamma 可靠度演化曲线', fontsize=12, fontweight='bold')
                    ax_reliability.legend(loc='upper right', fontsize=8)
                    ax_reliability.grid(True, alpha=0.3)
                    ax_reliability.set_ylim([0.0, 1.01])
                    ax_reliability.set_xlim(left=0.0)
                    ax_reliability.margins(x=0.0)

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

            # Capability-aware deadlock check:
            # If unfinished tasks remain but none can be executed by currently alive UAV types,
            # terminate simulation early with explicit reason instead of waiting for all UAVs to fail.
            mission_status = self._summarize_unfinished_tasks(all_completed_tasks, state)
            if mission_status['unfinished_total'] > 0 and mission_status['feasible_count'] == 0:
                self.final_unfinished_summary = mission_status
                print("⚠️  Mission terminated early: remaining tasks are infeasible for alive UAV types.")
                print(f"   alive_uav_types={mission_status['alive_uav_types']}, "
                      f"unfinished_total={mission_status['unfinished_total']}, "
                      f"unfinished_by_type={mission_status['unfinished_by_type']}, "
                      f"blocked_by_type={mission_status['blocked_by_type']}")
                if mission_status['sample_blocked_tasks']:
                    print(f"   sample_blocked_tasks={mission_status['sample_blocked_tasks']}")
                for i, proc in enumerate(main_processes):
                    if proc.is_alive():
                        proc.terminate()
                    state[i] = 0
                break

        finalize_simulation_outputs(
            self, realtime_plot, replan2ga, start_time, position, distance,
            active_subsystem, color_style, font, font0, font1, font2, base_plot_list,
            uav_num, reliability_history, failures, target_num, UAVs, time_history, x, y,
            degradation_history=degradation_history,
            health_history=health_history,
            reliability_model_time_history=reliability_model_time_history
        )
        try:
            console_manager.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    print("aaaaaa")
    print(dubins.__file__)

    _enable_console_capture()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    targets_sites = [
        [6734, 1453], [2233, 10], [5530, 1424], [401, 841], [3082, 1644], [7608, 4458],
        [7573, 3716], [7265, 1268], [6898, 1885], [1112, 2049], [5468, 2606], [5989, 2873],
        [4706, 2674], [4612, 2035], [6347, 2683], [6107, 669], [7611, 5184], [7462, 3590],
        [7732, 4723], [5900, 3561], [4483, 3369], [6101, 1110], [5199, 2182], [1633, 2809],
        [4307, 2322], [675, 1006], [7555, 4819], [7541, 3981], [3177, 756], [7352, 4506],
        [7545, 2801], [3245, 3305], [6426, 3173], [4608, 1198], [23, 2216], [7248, 3779],
        [7762, 4595], [7392, 2244], [3484, 2829], [6271, 2135], [4985, 140], [1916, 1569],
        [7280, 4899], [7509, 3239], [10, 2676], [6807, 2993], [5185, 3258], [3023, 1942]
    ]

    uav_id = list(range(1, 14))
    # 13架：type1=5, type2=4, type3=4
    uav_type = [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    cruise_speed = [70, 72, 74, 76, 78, 82, 84, 86, 88, 90, 92, 94, 96]
    turning_radii = [200, 205, 210, 215, 220, 245, 250, 255, 260, 290, 295, 300, 305]

    base_locations = [[2500, 4000, np.pi / 2] for _ in range(len(uav_id))]
    initial_states = [[float(base[0]), float(base[1]), float(base[2])] for base in base_locations]
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("=" * 70)
    print("=" * 70)
    print("after modify replan")
    print(f"withh13UAV数量: {len(uav_id)}")
    print(f"目标数量: {len(targets_sites)}")
    print(f"总任务数CZCCCzCd: {len(targets_sites) * 3} (侦察+攻击+验证)")
    print("UAV初始位置: 全部从njjnkjknknj基地位置出发")
    print("=" * 70 + "\n")

    dynamic_SEAD_mission = DynamicSEADMissionSimulator(
        targets_sites, uav_id, uav_type, cruise_speed, turning_radii, initial_states, base_locations,
        reliability_alpha_dict={1: 0.00002, 2: 0.00009, 3: 0.00003},
        failure_threshold=0.97,
        save_dir=os.path.join(script_dir, 'result_extended7'),
        enable_subsystem=True,
        min_member_count=2,
        min_subsystem_reliability=0.95,
        enable_knowledge_transfer=True,
        enable_rl_decision=True,
        gamma_parameter_dict=DEFAULT_GAMMA_PARAMETER_DICT,
        simulation_seconds_per_model_hour=1.0,
        gamma_global_seed=2026
    )
    
    manual_failures = [None for _ in range(len(uav_id))]

    dynamic_SEAD_mission.start_simulation(
        realtime_plot=False,
        uav_failure=manual_failures
    )
