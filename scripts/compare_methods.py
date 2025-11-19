"""
Compare PID Tuning Methods: Ziegler-Nichols vs Reinforcement Learning

This script loads ZN tuned PIDs and RL trained models,
then compares their performance on various metrics.

Usage:
    python scripts/compare_methods.py --maneuver circular --rl_model path/to/model.zip
    python scripts/compare_methods.py --all
"""

import argparse
import numpy as np
import sys
import os
import json
from tabulate import tabulate
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.missile import Missile
from src.target import Target


# Ziegler-Nichols tuned PID values (from tune_ziegler_nichols.py)
# These will be populated after running ZN tuning
ZN_TUNED_PIDS = {
    'circular': {'Kp': 2000.0, 'Ki': 10.0, 'Kd': 15.0},  # Placeholder
    'straight': {'Kp': 1500.0, 'Ki': 8.0, 'Kd': 12.0},   # Placeholder
    'zigzag': {'Kp': 2500.0, 'Ki': 12.0, 'Kd': 18.0},    # Placeholder
    'evasive': {'Kp': 3000.0, 'Ki': 15.0, 'Kd': 20.0},   # Placeholder
}


def run_pid_test(kp, ki, kd, maneuver='circular', n_test=50, max_steps=500, dt=0.01):
    """
    Test PID parameters and collect detailed metrics

    Returns:
        metrics: dict of performance metrics
    """
    hits = 0
    hit_times = []
    distances = []
    trajectories = []
    overshoots = []

    for run in range(n_test):
        # Random initialization (same as RL env)
        missile_x = np.random.uniform(0, 2000)
        missile_y = np.random.uniform(2000, 8000)
        missile_vx = np.random.uniform(800, 900)

        missile = Missile(
            x=missile_x, y=missile_y,
            vx=missile_vx, vy=0.0,
            max_speed=1000.0, max_accel=1000.0,
            kp=kp, ki=ki, kd=kd
        )

        target_x = np.random.uniform(6000, 9000)
        target_y = np.random.uniform(3000, 7000)
        target_heading = np.random.uniform(0, 2 * np.pi)

        target = Target(
            x=target_x, y=target_y,
            speed=1000.0,
            maneuver=maneuver
        )
        target.heading = target_heading

        # Run simulation
        distance_history = []
        hit = False
        hit_time = max_steps
        min_distance = float('inf')

        for step in range(max_steps):
            missile.update(target.position, dt)
            target.update(dt, missile_pos=missile.position)

            distance = np.linalg.norm(target.position - missile.position)
            distance_history.append(distance)
            min_distance = min(min_distance, distance)

            # Check hit
            if distance < 50.0:
                hit = True
                hit_time = step
                break

            # Check out of bounds
            if (missile.x < -1000 or missile.x > 11000 or
                missile.y < -1000 or missile.y > 11000):
                break

        if hit:
            hits += 1
            hit_times.append(hit_time)

        distances.append(distance_history[-1] if distance_history else 10000)
        trajectories.append(distance_history)

        # Calculate overshoot (how much missile passes optimal distance)
        if len(distance_history) > 10:
            overshoot = max(0, min_distance - 50.0) if min_distance < 50.0 else 0
            overshoots.append(overshoot)

    # Calculate metrics
    hit_rate = (hits / n_test) * 100
    avg_hit_time = np.mean(hit_times) if hit_times else max_steps
    avg_final_distance = np.mean(distances)
    avg_overshoot = np.mean(overshoots) if overshoots else 0

    # Trajectory smoothness (average jerk)
    jerks = []
    for traj in trajectories:
        if len(traj) > 2:
            velocity = np.diff(traj)
            accel = np.diff(velocity)
            jerk = np.diff(accel)
            jerks.append(np.mean(np.abs(jerk)))
    avg_jerk = np.mean(jerks) if jerks else 0

    metrics = {
        'hit_rate': hit_rate,
        'avg_hit_time': avg_hit_time,
        'avg_final_distance': avg_final_distance,
        'avg_overshoot': avg_overshoot,
        'avg_jerk': avg_jerk,
        'trajectories': trajectories
    }

    return metrics


def load_rl_pid(model_path, maneuver='circular'):
    """
    Load RL trained model and extract PID parameters

    Note: This is a placeholder. In practice, you'd load the model
    and run inference to get PID values.

    Returns:
        dict of PID parameters
    """
    # Placeholder - replace with actual model loading
    # from sb3_contrib import RecurrentPPO
    # model = RecurrentPPO.load(model_path)

    # For now, return placeholder values
    # These should be replaced with actual RL-trained values
    rl_pids = {
        'circular': {'Kp': 2300.0, 'Ki': 15.0, 'Kd': 20.0},
        'straight': {'Kp': 1800.0, 'Ki': 10.0, 'Kd': 15.0},
        'zigzag': {'Kp': 2800.0, 'Ki': 18.0, 'Kd': 25.0},
        'evasive': {'Kp': 3500.0, 'Ki': 20.0, 'Kd': 30.0},
    }

    return rl_pids.get(maneuver, {'Kp': 2000.0, 'Ki': 10.0, 'Kd': 15.0})


def compare_methods(maneuver='circular', rl_model_path=None, n_test=50):
    """
    Compare ZN and RL tuning methods
    """
    print(f"\n{'='*60}")
    print(f"COMPARING METHODS: {maneuver.upper()} TARGET")
    print(f"{'='*60}\n")

    # Get ZN tuned PID
    zn_pid = ZN_TUNED_PIDS.get(maneuver, {'Kp': 2000.0, 'Ki': 10.0, 'Kd': 15.0})
    print(f"Ziegler-Nichols PID:")
    print(f"  Kp = {zn_pid['Kp']:.2f}, Ki = {zn_pid['Ki']:.3f}, Kd = {zn_pid['Kd']:.3f}")

    # Get RL tuned PID
    if rl_model_path:
        rl_pid = load_rl_pid(rl_model_path, maneuver)
    else:
        print("\n⚠️  No RL model provided, using placeholder values")
        rl_pid = load_rl_pid(None, maneuver)

    print(f"\nRL Tuned PID:")
    print(f"  Kp = {rl_pid['Kp']:.2f}, Ki = {rl_pid['Ki']:.3f}, Kd = {rl_pid['Kd']:.3f}")

    # Test ZN
    print(f"\n\nTesting Ziegler-Nichols ({n_test} episodes)...")
    zn_metrics = run_pid_test(
        zn_pid['Kp'], zn_pid['Ki'], zn_pid['Kd'],
        maneuver=maneuver, n_test=n_test
    )

    # Test RL
    print(f"Testing RL ({n_test} episodes)...")
    rl_metrics = run_pid_test(
        rl_pid['Kp'], rl_pid['Ki'], rl_pid['Kd'],
        maneuver=maneuver, n_test=n_test
    )

    # Create comparison table
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}\n")

    table_data = [
        ['Metric', 'Ziegler-Nichols', 'RL', 'Improvement'],
        ['Hit Rate (%)', f"{zn_metrics['hit_rate']:.1f}",
         f"{rl_metrics['hit_rate']:.1f}",
         f"+{rl_metrics['hit_rate'] - zn_metrics['hit_rate']:.1f}"],
        ['Avg Hit Time (steps)', f"{zn_metrics['avg_hit_time']:.1f}",
         f"{rl_metrics['avg_hit_time']:.1f}",
         f"{rl_metrics['avg_hit_time'] - zn_metrics['avg_hit_time']:.1f}"],
        ['Avg Final Distance (m)', f"{zn_metrics['avg_final_distance']:.1f}",
         f"{rl_metrics['avg_final_distance']:.1f}",
         f"{rl_metrics['avg_final_distance'] - zn_metrics['avg_final_distance']:.1f}"],
        ['Avg Overshoot (m)', f"{zn_metrics['avg_overshoot']:.1f}",
         f"{rl_metrics['avg_overshoot']:.1f}",
         f"{rl_metrics['avg_overshoot'] - zn_metrics['avg_overshoot']:.1f}"],
        ['Avg Jerk (smoothness)', f"{zn_metrics['avg_jerk']:.2f}",
         f"{rl_metrics['avg_jerk']:.2f}",
         f"{rl_metrics['avg_jerk'] - zn_metrics['avg_jerk']:.2f}"],
    ]

    print(tabulate(table_data, headers='firstrow', tablefmt='grid'))

    # Save results
    results = {
        'maneuver': maneuver,
        'zn_pid': zn_pid,
        'rl_pid': rl_pid,
        'zn_metrics': {k: v for k, v in zn_metrics.items() if k != 'trajectories'},
        'rl_metrics': {k: v for k, v in rl_metrics.items() if k != 'trajectories'}
    }

    os.makedirs('results/comparison', exist_ok=True)
    with open(f'results/comparison/{maneuver}_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to results/comparison/{maneuver}_comparison.json")

    # Plot trajectories
    plot_comparison(zn_metrics, rl_metrics, maneuver)

    return results


def plot_comparison(zn_metrics, rl_metrics, maneuver):
    """
    Plot comparison of trajectory distances
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot sample trajectories
    for i in range(min(5, len(zn_metrics['trajectories']))):
        ax1.plot(zn_metrics['trajectories'][i], alpha=0.5, color='blue')
    ax1.set_title(f'Ziegler-Nichols: {maneuver.capitalize()} Target')
    ax1.set_xlabel('Time (steps)')
    ax1.set_ylabel('Distance to Target (m)')
    ax1.axhline(y=50, color='r', linestyle='--', label='Hit Radius')
    ax1.legend()
    ax1.grid(True)

    for i in range(min(5, len(rl_metrics['trajectories']))):
        ax2.plot(rl_metrics['trajectories'][i], alpha=0.5, color='green')
    ax2.set_title(f'RL Tuned: {maneuver.capitalize()} Target')
    ax2.set_xlabel('Time (steps)')
    ax2.set_ylabel('Distance to Target (m)')
    ax2.axhline(y=50, color='r', linestyle='--', label='Hit Radius')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    os.makedirs('results/comparison/plots', exist_ok=True)
    plt.savefig(f'results/comparison/plots/{maneuver}_trajectories.png', dpi=150)
    print(f"✅ Plot saved to results/comparison/plots/{maneuver}_trajectories.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare PID Tuning Methods')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--rl_model', type=str, default=None,
                       help='Path to trained RL model (.zip)')
    parser.add_argument('--n_test', type=int, default=50,
                       help='Number of test episodes')
    parser.add_argument('--all', action='store_true',
                       help='Compare all maneuver types')

    args = parser.parse_args()

    if args.all:
        maneuvers = ['straight', 'circular', 'zigzag', 'evasive']
    else:
        maneuvers = [args.maneuver]

    all_results = {}

    for maneuver in maneuvers:
        results = compare_methods(
            maneuver=maneuver,
            rl_model_path=args.rl_model,
            n_test=args.n_test
        )
        all_results[maneuver] = results

    # Overall summary
    if args.all:
        print(f"\n\n{'#'*60}")
        print("# OVERALL SUMMARY")
        print(f"{'#'*60}\n")

        summary_table = [['Maneuver', 'ZN Hit Rate', 'RL Hit Rate', 'Improvement']]
        for maneuver, res in all_results.items():
            summary_table.append([
                maneuver.capitalize(),
                f"{res['zn_metrics']['hit_rate']:.1f}%",
                f"{res['rl_metrics']['hit_rate']:.1f}%",
                f"+{res['rl_metrics']['hit_rate'] - res['zn_metrics']['hit_rate']:.1f}%"
            ])

        print(tabulate(summary_table, headers='firstrow', tablefmt='grid'))


if __name__ == "__main__":
    main()
