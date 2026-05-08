import numpy as np
from math import hypot
from collections import defaultdict


class SubsystemSelector:

    def __init__(self, min_subsystem_reliability=0.95, min_member_count=2, strict_min_member_count=False):
        self.min_subsystem_reliability = min_subsystem_reliability
        self.min_member_count = max(1, int(min_member_count))
        # 默认不把人数阈值作为硬约束：以“任务类型覆盖+可靠度”优先
        self.strict_min_member_count = bool(strict_min_member_count)
        self.max_member_count = max(6, self.min_member_count)
        self.search_radius_step = 1000
        self.max_search_distance = 5000

        # Task type to UAV type mapping
        self.task_type_to_uav = {
            1: [1, 2],  # Reconnaissance task -> Surveillance or Attack UAV
            2: [2, 3],  # Attack task -> Attack or Munition UAV
            3: [1]      # Verification task -> Surveillance UAV only
        }

    def select_subsystem(self, failed_uav_id, all_uavs, uav_positions,
                         uav_reliabilities, unfinished_tasks, current_time,
                         max_distance=2000):

        print(f"\n{'=' * 70}")
        print(f" Building Replanning Subsystem - Failed UAV: {failed_uav_id}")
        print(f"{'=' * 70}")

        failed_uav = next((uav for uav in all_uavs if uav.id == failed_uav_id), None)
        if not failed_uav:
            print(f" Warning: Failed UAV {failed_uav_id} not found")
            return [], {}

        failed_pos = uav_positions.get(failed_uav_id, [0, 0, 0])

        requirement_profile = self._analyze_task_requirements(unfinished_tasks, all_uavs, failed_uav_id)
        required_types = requirement_profile['required_types']
        required_task_types = requirement_profile.get('required_task_types', [])
        task_type_counts = requirement_profile.get('task_type_counts', {})
        print(f" Required task types: {required_task_types}")
        print(f" Remaining tasks by type: {task_type_counts}")
        print(f" Compatible UAV types: {required_types}")

        candidates = []
        search_radius_used = max_distance
        search_radius = max_distance
        while search_radius <= self.max_search_distance:
            candidates = self._find_candidates(
                all_uavs, failed_uav_id, failed_pos, uav_positions,
                uav_reliabilities, requirement_profile, search_radius
            )
            if candidates:
                search_radius_used = search_radius
                break
            if search_radius < self.max_search_distance:
                print(f"⚠ No candidates within {search_radius}m, expanding search radius...")
            search_radius += self.search_radius_step

        if not candidates:
            print(f"⚠ No candidates found within adaptive range (max={self.max_search_distance}m)")
            return [], {}

        print(f"Candidate search radius used: {search_radius_used}m")

        candidates.sort(key=lambda x: x['score'], reverse=True)
        self._print_candidates(candidates)

        selected = self._greedy_select(candidates, requirement_profile)
        selected_uavs, subsystem_info = self._validate_subsystem(
            selected, requirement_profile, current_time
        )
        subsystem_info['search_radius_used'] = search_radius_used

        print(f"{'=' * 70}\n")
        return selected_uavs, subsystem_info

    def _analyze_task_requirements(self, unfinished_tasks, all_uavs, failed_uav_id):
        # remaining task-type coverage requirement (核心约束)
        required_task_types = set()
        task_type_counts = defaultdict(int)
        task_support_counts = defaultdict(int)
        available_type_counts = defaultdict(int)
        required_types = set()

        for uav in all_uavs:
            if uav.id == failed_uav_id or uav.is_failed:
                continue
            available_type_counts[uav.type] += 1

        for task in unfinished_tasks:
            task_type = int(task[1]) if len(task) > 1 else int(task[0])
            required_task_types.add(task_type)
            task_type_counts[task_type] += 1
            for uav_type in self.task_type_to_uav.get(task_type, []):
                task_support_counts[uav_type] += 1
                if available_type_counts.get(uav_type, 0) > 0:
                    required_types.add(uav_type)

        return {
            'required_task_types': sorted(required_task_types),
            'task_type_counts': dict(task_type_counts),
            'required_types': sorted(required_types),  # for ranking emphasis only
            'task_support_counts': dict(task_support_counts),
            'available_type_counts': dict(available_type_counts),
        }

    def _find_candidates(self, all_uavs, failed_uav_id, failed_pos,
                         uav_positions, uav_reliabilities, requirement_profile,
                         max_distance):

        candidates = []
        required_types = set(requirement_profile['required_types'])
        task_support_counts = requirement_profile['task_support_counts']

        for uav in all_uavs:
            if uav.id == failed_uav_id or uav.is_failed:
                continue

            uav_pos = uav_positions.get(uav.id, [0, 0, 0])
            distance = hypot(failed_pos[0] - uav_pos[0], failed_pos[1] - uav_pos[1])
            if distance > max_distance:
                continue

            reliability = uav_reliabilities.get(uav.id, 1.0)
            urgency = task_support_counts.get(uav.type, 0)
            score = self._calculate_score(
                uav_type=uav.type,
                distance=distance,
                reliability=reliability,
                required_types=required_types,
                max_distance=max_distance,
                urgency=urgency
            )

            candidates.append({
                'uav': uav,
                'uav_id': uav.id,
                'distance': distance,
                'reliability': reliability,
                'type': uav.type,
                'score': score
            })

        return candidates

    def _calculate_score(self, uav_type, distance, reliability,
                         required_types, max_distance, urgency):
        distance_score = max(0.0, 1.0 - (distance / max_distance))
        reliability_score = float(reliability)
        type_match_score = 1.0 if uav_type in required_types else 0.2
        urgency_score = min(1.0, urgency / 3.0) if urgency > 0 else 0.0

        w_distance = 0.30
        w_reliability = 0.45
        w_urgency = 0.25
        total_score = (w_distance * distance_score +
                       w_reliability * reliability_score +
                       w_urgency * urgency_score) * type_match_score
        return total_score

    def _greedy_select(self, candidates, requirement_profile):
        # Coverage-first greedy selection:
        # 1) first satisfy task-type coverage with minimal high-score set
        # 2) only when strict_min_member_count=True, fill to min_member_count
        selected = []
        covered_task_types = set()
        required_task_types = set(requirement_profile.get('required_task_types', []))
        if not required_task_types:
            return selected

        ranked = sorted(candidates, key=lambda x: x['score'], reverse=True)
        while (required_task_types - covered_task_types) and len(selected) < self.max_member_count:
            best = None
            best_gain = -1
            best_score = -1.0
            for cand in ranked:
                if cand in selected:
                    continue
                can_cover = set()
                for task_type in (required_task_types - covered_task_types):
                    if cand['type'] in self.task_type_to_uav.get(task_type, []):
                        can_cover.add(task_type)
                gain = len(can_cover)
                if gain > best_gain or (gain == best_gain and cand['score'] > best_score):
                    best = cand
                    best_gain = gain
                    best_score = cand['score']
            if best is None or best_gain <= 0:
                break
            selected.append(best)
            for task_type in required_task_types:
                if best['type'] in self.task_type_to_uav.get(task_type, []):
                    covered_task_types.add(task_type)

        # 仅在 strict 模式下补足人数；默认不追求高 desired counts
        if self.strict_min_member_count:
            for cand in ranked:
                if len(selected) >= self.min_member_count:
                    break
                if cand not in selected and len(selected) < self.max_member_count:
                    selected.append(cand)

        return selected

    def _evaluate_reliability(self, selected, requirement_profile):
        if not selected:
            return 0.0, False, 0.0

        reliabilities = [max(1e-9, float(s['reliability'])) for s in selected]
        base_reliability = float(np.exp(np.mean(np.log(reliabilities))))

        type_distribution = self._get_type_distribution(selected)
        required_task_types = requirement_profile.get('required_task_types', [])
        coverage_ok = True
        for task_type in required_task_types:
            compatible = self.task_type_to_uav.get(task_type, [])
            if not any(type_distribution.get(t, 0) > 0 for t in compatible):
                coverage_ok = False
                break

        required_task_types_set = set(required_task_types)
        covered_count = 0
        for task_type in required_task_types_set:
            compatible = self.task_type_to_uav.get(task_type, [])
            if any(type_distribution.get(t, 0) > 0 for t in compatible):
                covered_count += 1
        redundancy_score = float(covered_count / max(1, len(required_task_types_set)))
        subsystem_reliability = float(max(0.0, min(1.0, base_reliability)))
        return subsystem_reliability, coverage_ok, redundancy_score

    def _validate_subsystem(self, selected, requirement_profile, current_time):
        selected_uavs = [s['uav_id'] for s in selected]
        subsystem_reliability, coverage_ok, redundancy_score = self._evaluate_reliability(selected, requirement_profile)
        type_distribution = self._get_type_distribution(selected)

        print(f"\n✓ Final Selected Subsystem:")
        print(f"  UAV Count: {len(selected_uavs)}")
        print(f"  UAV List: {selected_uavs}")
        print(f"  Required Task Types: {requirement_profile.get('required_task_types', [])}")
        print(f"  Min Member Count: {self.min_member_count}")
        print(f"  Strict Min Member Count: {self.strict_min_member_count}")
        print(f"  Subsystem Reliability: {subsystem_reliability:.6f}")
        print(f"  Type Coverage OK: {coverage_ok}")
        print(f"  Type Redundancy Score: {redundancy_score:.3f}")

        subsystem_info = {
            'selected_uavs': selected_uavs,
            'reliability': subsystem_reliability,
            'avg_distance': np.mean([s['distance'] for s in selected]) if selected else 0,
            'type_distribution': type_distribution,
            'task_coverage': coverage_ok,
            'type_redundancy_score': redundancy_score,
            'required_types': requirement_profile['required_types'],
            'required_task_types': requirement_profile.get('required_task_types', []),
            'task_type_counts': requirement_profile.get('task_type_counts', {}),
            'min_member_count': self.min_member_count,
            'selection_time': current_time
        }

        size_ok = (len(selected_uavs) >= self.min_member_count) if self.strict_min_member_count else (len(selected_uavs) >= 1)

        if (not selected_uavs) or \
                (not size_ok) or \
                (not coverage_ok) or \
                (subsystem_reliability < self.min_subsystem_reliability):
            print("⚠ Selected subsystem does not satisfy constraints "
                  f"(coverage={coverage_ok}, size={len(selected_uavs)}/{self.min_member_count}"
                  f"{' strict' if self.strict_min_member_count else ' soft'}, "
                  f"R={subsystem_reliability:.4f}/{self.min_subsystem_reliability:.4f})")
            return [], subsystem_info

        return selected_uavs, subsystem_info

    def _get_type_distribution(self, selected):
        distribution = defaultdict(int)
        for s in selected:
            distribution[s['type']] += 1
        return dict(distribution)

    def _print_candidates(self, candidates):
        print(f"\n Candidate UAV Ranking:")
        for i, cand in enumerate(candidates[:10], 1):
            print(f"{i}.UAV {cand['uav_id']} (Type {cand['type']}) - "
                  f"Distance: {cand['distance']:.0f}m, "
                  f"Reliability: {cand['reliability']:.4f}, "
                  f"Score: {cand['score']:.4f}")
