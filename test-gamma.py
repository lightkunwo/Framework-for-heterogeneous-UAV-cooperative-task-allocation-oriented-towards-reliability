import os
import inspect
from datetime import datetime

import numpy as np

import secondDemo
from secondDemo import DynamicSEADMissionSimulator


def _validate_plotting_chain():
    source = inspect.getsource(secondDemo.finalize_simulation_outputs)
    required_tokens = {
        "health_xmax": "health plot x-axis starts at 0 and uses full extended range",
        "remaining_degradation": "health plot extends unfinished degradation curves to 0%",
        "reliability_xmax": "reliability plot x-axis starts at 0 and uses full extended range",
        "plot_r.append(0.0)": "reliability plot extends unfinished reliability curves to the bottom",
        "ax.set_xlim(0.0": "saved Gamma plots force the x-axis to start from 0",
    }
    missing = [desc for token, desc in required_tokens.items() if token not in source]
    if missing:
        raise RuntimeError(
            "The imported secondDemo.py is not the updated plotting version. "
            f"Imported path: {secondDemo.__file__}. Missing fixes: {missing}"
        )


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    _validate_plotting_chain()

    # Small DPGA allocation scenario: three targets, twelve UAVs, matching the
    # paper-style reliability figure size while keeping the allocation small.
    targets_sites = [
        [900, 650],
        [1250, 900],
        [1650, 700],
    ]
    uav_id = list(range(1, 13))
    uav_type = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    cruise_speed = [90 + 2 * (i % 4) for i in range(len(uav_id))]
    turning_radii = [180 + 10 * (i % 4) for i in range(len(uav_id))]
    base_locations = [[0.0, 0.0, np.pi / 4] for _ in uav_id]
    initial_states = [[float(b[0]), float(b[1]), float(b[2])] for b in base_locations]

    # Paper-style setting: Gamma degradation increments, UAV reliability
    # threshold R_Pf=0.6, inspection interval 1 model hour.
    # The thesis text reports R_f=0.6 as the reliability failure threshold; it
    # does not provide a separate numeric degradation limit L, so L=100 keeps
    # the same limit used by the current Gamma implementation.
    paper_gamma_params = {
        1: {"shape_rate": 1.1, "scale": 0.9, "limit": 100.0},
        2: {"shape_rate": 1.1, "scale": 0.9, "limit": 100.0},
        3: {"shape_rate": 1.1, "scale": 0.9, "limit": 100.0},
    }
    individual_factor_range = 0.08
    shape_rates = sorted({params["shape_rate"] for params in paper_gamma_params.values()})
    scales = sorted({params["scale"] for params in paper_gamma_params.values()})
    limits = sorted({params["limit"] for params in paper_gamma_params.values()})
    
    save_dir = os.path.join(script_dir, "result_paper_gamma_small_dpga")
    print("=" * 70)
    print("Small DPGA Gamma Reliability Experiment")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Imported secondDemo.py: {secondDemo.__file__}")
    print(f"Targets: {len(targets_sites)}; UAVs: {len(uav_id)}")
    print("Subsystem: Disabled; Actor-Critic/RL: Disabled; Knowledge Transfer: Disabled")
    print(f"Gamma base shape_rate={shape_rates}, scale={scales}, X_f(limit)={limits}, R_Pf=0.6")
    print(f"Paper-style layering: per-UAV shape_rate is perturbed by +/-{individual_factor_range * 100:.1f}%.")
    print("simulation_seconds_per_model_hour=0.5, so the reliability curve reaches model time faster.")
    print("=" * 70)

    simulator_kwargs = {
        "reliability_alpha_dict": {1: 0.0, 2: 0.0, 3: 0.0},
        "failure_threshold": 0.6,
        "save_dir": save_dir,
        "enable_subsystem": False,
        "min_member_count": 2,
        "min_subsystem_reliability": 0.95,
        "enable_assist_reallocation": False,
        "enable_knowledge_transfer": False,
        "enable_rl_decision": False,
        "gamma_parameter_dict": paper_gamma_params,
        "simulation_seconds_per_model_hour": 0.5,
        "gamma_global_seed": 2026,
        # Newer secondDemo.py supports this switch. It keeps each UAV type's
        # configured base shape_rate but gives each UAV a small manufacturing/
        # degradation difference, so the 12 reliability curves are not identical.
        # Older server copies do not,
        # so the kwargs are filtered by the actual constructor signature below.
        "gamma_individual_factor_range": individual_factor_range,
    }
    init_signature = inspect.signature(DynamicSEADMissionSimulator.__init__)
    supported_kwargs = {
        key: value for key, value in simulator_kwargs.items()
        if key in init_signature.parameters
    }
    skipped_kwargs = sorted(set(simulator_kwargs) - set(supported_kwargs))
    if skipped_kwargs:
        print(f"Skipped unsupported simulator options: {skipped_kwargs}")
    required_options = {
        "gamma_parameter_dict",
        "simulation_seconds_per_model_hour",
        "failure_threshold",
        "gamma_individual_factor_range",
    }
    missing_required = sorted(required_options - set(supported_kwargs))
    if missing_required:
        raise RuntimeError(
            "The imported secondDemo.py does not support the Gamma reliability experiment options: "
            f"{missing_required}. Please run this script with the updated secondDemo.py."
        )

    simulator = DynamicSEADMissionSimulator(
        targets_sites,
        uav_id,
        uav_type,
        cruise_speed,
        turning_radii,
        initial_states,
        base_locations,
        **supported_kwargs,
    )

    simulator.start_simulation(
        realtime_plot=False,
        uav_failure=[None for _ in uav_id],
    )

    print("=" * 70)
    print(f"Experiment finished. Results saved to: {save_dir}")
    print("Key plots:")
    print("  04_uav_health_degradation_<timestamp>.png")
    print("  05_uav_gamma_reliability_<timestamp>.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
