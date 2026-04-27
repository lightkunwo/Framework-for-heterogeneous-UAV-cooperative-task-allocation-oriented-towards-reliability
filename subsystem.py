import numpy as np
from math import hypot
from collections import defaultdict


class SubsystemSelector:

    def __init__(self, min_subsystem_reliability=0.95, min_member_count=2):
        print("init")
        self.min_subsystem_reliability = min_subsystem_reliability
        self.min_member_count = max(2, int(min_member_count))
        self.max_member_count = 6

        # Task type to UAV type mapping
        self.task_type_to_uav = {
            1: [1, 2],  # Reconnaissance task -> Surveillance or Attack UAV
            2: [2, 3],  # Attack task -> Attack or Munition UAV
            3: [1, 2]   # Verification task -> Surveillance or Attack UAV
        }

    def select_subsystem(self, failed_uav_id, all_uavs, uav_positions,
                         uav_reliabilities, unfinished_tasks, current_time,
                         max_distance=2000):

        print(f"\n{'=' * 70}")
        print(f"🔧 Building Replanning Subsystem - Failed UAV: {failed_uav_id}")
        print(f"{'=' * 70}")

        failed_uav = next((uav for uav in all_uavs if uav.id == failed_uav_id), None)
        if not failed_uav:
            print(f"⚠ Warning: Failed UAV {failed_uav_id} not found")
            return [], {}

        failed_pos = uav_positions.get(failed_uav_id, [0, 0, 0])

        requirement_profile = self._analyze_task_requirements(unfinished_tasks, all_uavs, failed_uav_id)
        required_types = requirement_profile['required_types']
        desired_counts = requirement_profile['desired_counts']
        print(f"📋 Required UAV types: {required_types}")
        print(f"📋 Desired type counts: {desired_counts}")

        candidates = self._find_candidates(
            all_uavs, failed_uav_id, failed_pos, uav_positions,
            uav_reliabilities, requirement_profile, max_distance
        )

        if not candidates:
            print("⚠ No candidates found within range")
            return [], {}

        candidates.sort(key=lambda x: x['score'], reverse=True)
        self._print_candidates(candidates)

        selected = self._greedy_select(candidates, requirement_profile)
        selected_uavs, subsystem_info = self._validate_subsystem(
            selected, requirement_profile, current_time
        )

        print(f"{'=' * 70}\n")
        return selected_uavs, subsystem_info

    def _analyze_task_requirements(self, unfinished_tasks, all_uavs, failed_uav_id):
        task_support_counts = defaultdict(int)
        required_types = set()

        available_type_counts = defaultdict(int)
        for uav in all_uavs:
            if uav.id == failed_uav_id or uav.is_failed:
                continue
            available_type_counts[uav.type] += 1

        for task in unfinished_tasks:
            task_type = task[1] if len(task) > 1 else task[0]
            supported_types = self.task_type_to_uav.get(task_type, [])
            for uav_type in supported_types:
                task_support_counts[uav_type] += 1
                if available_type_counts.get(uav_type, 0) > 0:
                    required_types.add(uav_type)

        desired_counts = {}
        for uav_type in sorted(required_types):
            desired = 1
            if uav_type == 2 and task_support_counts.get(uav_type, 0) > 0:
                desired = 2
            elif task_support_counts.get(uav_type, 0) >= 3:
                desired = 2
            desired_counts[uav_type] = min(desired, max(1, available_type_counts.get(uav_type, 1)))

        return {
            'required_types': sorted(required_types),
            'desired_counts': desired_counts,
            'task_support_counts': dict(task_support_counts),
            'available_type_counts': dict(available_type_counts)
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
        selected = []
        type_counts = defaultdict(int)
        desired_counts = requirement_profile['desired_counts']

        # Step 1: satisfy type-aware desired counts first
        for uav_type in desired_counts:
            same_type_candidates = [c for c in candidates if c['type'] == uav_type and c not in selected]
            same_type_candidates.sort(key=lambda x: x['score'], reverse=True)
            for cand in same_type_candidates:
                if type_counts[uav_type] >= desired_counts[uav_type]:
                    break
                selected.append(cand)
                type_counts[uav_type] += 1

        # Step 2: minimal subsystem size fallback
        for cand in candidates:
            if cand not in selected and len(selected) < self.min_member_count:
                selected.append(cand)
                type_counts[cand['type']] += 1

        # Step 3: add optional redundancy if it improves reliability and stays compact
        for cand in candidates:
            if cand in selected or len(selected) >= self.max_member_count:
                continue
            temp_selected = selected + [cand]
            temp_reliability, _, _ = self._evaluate_reliability(temp_selected, requirement_profile)
            current_reliability, _, _ = self._evaluate_reliability(selected, requirement_profile)
            if temp_reliability >= current_reliability:
                selected.append(cand)
                type_counts[cand['type']] += 1

        return selected

    def _evaluate_reliability(self, selected, requirement_profile):
        if not selected:
            return 0.0, False, 0.0

        reliabilities = [max(1e-9, float(s['reliability'])) for s in selected]
        base_reliability = float(np.exp(np.mean(np.log(reliabilities))))

        type_distribution = self._get_type_distribution(selected)
        required_types = requirement_profile['required_types']
        desired_counts = requirement_profile['desired_counts']

        coverage_ok = all(type_distribution.get(t, 0) >= 1 for t in required_types) if required_types else True

        if desired_counts:
            redundancy_terms = [
                min(type_distribution.get(t, 0), desired_counts[t]) / float(desired_counts[t])
                for t in desired_counts
            ]
            redundancy_score = float(np.mean(redundancy_terms))
        else:
            redundancy_score = 1.0

        size_score = min(1.0, len(selected) / float(max(1, self.min_member_count)))
        subsystem_reliability = base_reliability * (0.65 + 0.20 * redundancy_score + 0.15 * size_score)
        subsystem_reliability = float(max(0.0, min(1.0, subsystem_reliability)))
        return subsystem_reliability, coverage_ok, redundancy_score

    def _validate_subsystem(self, selected, requirement_profile, current_time):
        selected_uavs = [s['uav_id'] for s in selected]
        subsystem_reliability, coverage_ok, redundancy_score = self._evaluate_reliability(selected, requirement_profile)
        type_distribution = self._get_type_distribution(selected)

        print(f"\n✓ Final Selected Subsystem:")
        print(f"  UAV Count: {len(selected_uavs)}")
        print(f"  UAV List: {selected_uavs}")
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
            'desired_counts': requirement_profile['desired_counts'],
            'selection_time': current_time
        }

        if (not coverage_ok) or len(selected_uavs) < self.min_member_count or subsystem_reliability < self.min_subsystem_reliability:
            print("⚠ Selected subsystem does not satisfy type-aware reliability constraints")
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
