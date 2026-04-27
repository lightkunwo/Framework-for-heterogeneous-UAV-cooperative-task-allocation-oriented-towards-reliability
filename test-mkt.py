import numpy as np
from tqdm import tqdm
from types import SimpleNamespace
from GA_SEAD_process import GA_SEAD

def run_comparative_test(num_runs=10):

    targets = [
        [500, 4500], [1200, 4300], [800, 3800], [1500, 3500], [500, 3000],
        [3500, 4500], [4200, 4000], [3800, 3500], [4500, 3200], [3200, 3800],
        [800, 1500], [1500, 1200], [500, 800], [1200, 500], [1800, 800],
        [3500, 1500], [4200, 1200], [3800, 800], [4500, 500], [3200, 1000]
    ]


    uav_message = SimpleNamespace(
        uav_id=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        uav_type=[1, 2, 3, 1, 2, 3, 2, 2, 1, 3],
        cruising_speed=[70, 80, 95, 75, 85, 90, 82, 88, 72, 92],
        turning_radii=[200, 250, 300, 210, 260, 310, 245, 255, 190, 295],
        uav_states=[
            [700, 5500, -np.pi / 2], [1500, 5500, -np.pi / 2],
            [5500, 2500, np.pi], [5500, 3500, np.pi],
            [2500, -500, np.pi / 2], [3500, -500, np.pi / 2],
            [6500, 2500, 0], [500, 1500, 0],
            [0, 5000, -np.pi / 4], [5000, 0, 3 * np.pi / 4]
        ],
        base=[[2500, 4000, np.pi / 2] for _ in range(10)],
        tasks_completed=[],
        new_targets=[],
        elite_chromosomes=[]
    )

    summary = {
        'With_MKT': {'time': [], 'dist': [], 'cost': [], 'penalty': []},
        'Without_MKT': {'time': [], 'dist': [], 'cost': [], 'penalty': []}
    }

    # --- 2. 执行对比实验 ---
    for mode in ['With_MKT', 'Without_MKT']:
        print(f"\n>> 正在执行方案: {mode} (100次随机运行)...")
        mkt_flag = True if mode == 'With_MKT' else False

        for _ in tqdm(range(num_runs)):

            ga = GA_SEAD(targets, population_size=300)
            ga.enable_mkt3 = mkt_flag


            best_sol, _, _ = ga.run_GA(iteration=100, uav_message=uav_message)

            # 获取各项性能指标
            fitness, m_time, dist, penalty = ga.objectives_evaluation(best_sol)

            summary[mode]['time'].append(m_time)
            summary[mode]['dist'].append(dist)
            summary[mode]['penalty'].append(penalty)
            # Cost 为适应度分母部分（任务总代价）
            summary[mode]['cost'].append(1.0 / fitness if fitness > 0 else 1e9)

    # --- 3. 数据汇总输出 ---
    print("\n" + "=" * 75)
    print(f"{'性能指标 (10遍均值)':<22} | {'带 MKT (知识迁移型)':<22} | {'普通 GA (不带 MKT)'}")
    print("-" * 75)

    metrics_map = [
        ('Mission Time (s)', 'time'),
        ('Total Distance (m)', 'dist'),
        ('Total Cost Value', 'cost'),
        ('Penalty Value', 'penalty')
    ]

    for label, key in metrics_map:
        avg_mkt = np.mean(summary['With_MKT'][key])
        avg_norm = np.mean(summary['Without_MKT'][key])
        improvement = ((avg_norm - avg_mkt) / avg_norm * 100) if avg_norm != 0 else 0
        print(f"{label:<22} | {avg_mkt:<22.4f} | {avg_norm:.4f}")

    print("=" * 75)
    print("测试完成。MKT 通过跨无人机种群的知识共享，理论上能有效降低 Total Cost。")

if __name__ == "__main__":
    run_comparative_test(10)