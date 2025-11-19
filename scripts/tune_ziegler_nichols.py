"""
Ziegler-Nichols PID Tuning Method
Automated tuning to find optimal PID parameters for comparison with RL

Method:
1. Find Ku (ultimate gain): Increase Kp until sustained oscillation
2. Measure Tu (ultimate period): Period of oscillation
3. Calculate PID parameters using ZN formulas

Usage:
    python scripts/tune_ziegler_nichols.py --maneuver circular
    python scripts/tune_ziegler_nichols.py --all
"""

import argparse
import numpy as np
import sys
import os
from scipy import signal
from scipy.fft import fft, fftfreq

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.missile import Missile
from src.target import Target


def run_simulation(kp, ki, kd, maneuver='circular', n_runs=5, max_steps=500, dt=0.01):
    """
    Run simulation with given PID parameters

    Returns:
        avg_distance_history: Average distance over time
        hit_rate: Percentage of successful hits
        avg_hit_time: Average time to hit
    """
    distance_histories = []
    hits = 0
    hit_times = []

    for run in range(n_runs):
        # Random initialization
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

        for step in range(max_steps):
            missile.update(target.position, dt)
            target.update(dt, missile_pos=missile.position)

            distance = np.linalg.norm(target.position - missile.position)
            distance_history.append(distance)

            # Check hit
            if distance < 50.0:
                hit = True
                hit_time = step
                hits += 1
                break

            # Check out of bounds
            if (missile.x < -1000 or missile.x > 11000 or
                missile.y < -1000 or missile.y > 11000):
                break

        # Pad if needed
        while len(distance_history) < max_steps:
            distance_history.append(distance_history[-1] if distance_history else 10000)

        distance_histories.append(distance_history)
        if hit:
            hit_times.append(hit_time)

    # Average distance history
    avg_distance_history = np.mean(distance_histories, axis=0)
    hit_rate = (hits / n_runs) * 100
    avg_hit_time = np.mean(hit_times) if hit_times else max_steps

    return avg_distance_history, hit_rate, avg_hit_time


def detect_oscillation(distance_history, threshold=0.1):
    """
    Detect if distance shows sustained oscillation

    Returns:
        is_oscillating: bool
        period: oscillation period (in steps), or None
    """
    # Remove trend (detrend)
    detrended = signal.detrend(distance_history)

    # Check variance (oscillation has high variance)
    variance = np.var(detrended)

    if variance < 100:  # Too small variance, no oscillation
        return False, None

    # FFT to find dominant frequency
    n = len(detrended)
    yf = fft(detrended)
    xf = fftfreq(n, d=1)[:n//2]

    # Find peak frequency (excluding DC component)
    power = 2.0/n * np.abs(yf[1:n//2])

    if len(power) == 0:
        return False, None

    peak_idx = np.argmax(power)
    peak_freq = xf[1:][peak_idx]

    # Period in steps
    if peak_freq > 0:
        period = 1.0 / peak_freq
    else:
        return False, None

    # Check if oscillation is sustained (power is significant)
    max_power = np.max(power)

    if max_power > 50:  # Threshold for sustained oscillation
        return True, period
    else:
        return False, None


def find_ultimate_gain(maneuver='circular', max_iterations=20):
    """
    Binary search to find Ku (ultimate gain) for sustained oscillation

    Returns:
        Ku: Ultimate gain
        Tu: Ultimate period (in seconds)
    """
    print(f"\nFinding ultimate gain (Ku) for '{maneuver}' target...")
    print("=" * 60)

    # Binary search bounds
    kp_low = 100.0
    kp_high = 10000.0

    best_kp = None
    best_period = None

    for iteration in range(max_iterations):
        kp_mid = (kp_low + kp_high) / 2.0

        print(f"\nIteration {iteration + 1}: Testing Kp = {kp_mid:.1f}")

        # Test with P-only controller
        distance_history, hit_rate, _ = run_simulation(
            kp=kp_mid, ki=0.0, kd=0.0,
            maneuver=maneuver,
            n_runs=3
        )

        # Check for oscillation
        is_oscillating, period = detect_oscillation(distance_history)

        if is_oscillating:
            print(f"  ✓ Sustained oscillation detected! Period = {period:.1f} steps")
            best_kp = kp_mid
            best_period = period

            # Try higher gain
            kp_low = kp_mid
        else:
            print(f"  ✗ No sustained oscillation")
            # Try lower gain
            kp_high = kp_mid

        # Convergence check
        if (kp_high - kp_low) < 50:
            break

    if best_kp is None:
        print("\n⚠️  Warning: Could not find sustained oscillation")
        print("   Using last tested Kp as approximation")
        best_kp = kp_mid
        best_period = 100  # Approximation

    # Convert period from steps to seconds
    Tu = best_period * 0.01  # dt = 0.01

    print(f"\n{'='*60}")
    print(f"ULTIMATE PARAMETERS FOUND:")
    print(f"  Ku (Ultimate Gain) = {best_kp:.1f}")
    print(f"  Tu (Ultimate Period) = {Tu:.3f} seconds")
    print(f"{'='*60}\n")

    return best_kp, Tu


def calculate_ziegler_nichols_pid(Ku, Tu):
    """
    Calculate PID parameters using Ziegler-Nichols formulas

    Returns:
        dict of PID variants
    """
    results = {}

    # Classic PID (some overshoot)
    results['classic'] = {
        'Kp': 0.6 * Ku,
        'Ki': (0.6 * Ku) * 2 / Tu,
        'Kd': (0.6 * Ku) * Tu / 8,
        'description': 'Classic ZN (some overshoot)'
    }

    # Pessen Integral Rule (some overshoot)
    results['pessen'] = {
        'Kp': 0.7 * Ku,
        'Ki': (0.7 * Ku) * 2.5 / Tu,
        'Kd': (0.7 * Ku) * Tu * 3 / 20,
        'description': 'Pessen Integral Rule (some overshoot)'
    }

    # No overshoot
    results['no_overshoot'] = {
        'Kp': 0.2 * Ku,
        'Ki': (0.2 * Ku) * 2 / Tu,
        'Kd': (0.2 * Ku) * Tu / 3,
        'description': 'No Overshoot'
    }

    return results


def test_zn_variants(zn_variants, maneuver='circular', n_test=20):
    """
    Test all ZN variants and compare performance
    """
    print(f"\nTesting Ziegler-Nichols variants on '{maneuver}' target...")
    print("=" * 60)

    results = {}

    for variant_name, params in zn_variants.items():
        print(f"\nTesting {params['description']}:")
        print(f"  Kp = {params['Kp']:.2f}, Ki = {params['Ki']:.3f}, Kd = {params['Kd']:.3f}")

        distance_history, hit_rate, avg_hit_time = run_simulation(
            kp=params['Kp'],
            ki=params['Ki'],
            kd=params['Kd'],
            maneuver=maneuver,
            n_runs=n_test
        )

        # Calculate metrics
        avg_distance = np.mean(distance_history)
        final_distance = distance_history[-1]

        print(f"  Hit Rate: {hit_rate:.1f}%")
        print(f"  Avg Hit Time: {avg_hit_time:.1f} steps")
        print(f"  Avg Distance: {avg_distance:.1f}m")
        print(f"  Final Distance: {final_distance:.1f}m")

        results[variant_name] = {
            'params': params,
            'hit_rate': hit_rate,
            'avg_hit_time': avg_hit_time,
            'avg_distance': avg_distance,
            'final_distance': final_distance
        }

    return results


def save_results(results, maneuver):
    """Save results to file"""
    os.makedirs('results/ziegler_nichols', exist_ok=True)
    filename = f'results/ziegler_nichols/{maneuver}_zn_tuning.txt'

    with open(filename, 'w') as f:
        f.write(f"Ziegler-Nichols Tuning Results for '{maneuver}' Target\n")
        f.write("=" * 60 + "\n\n")

        for variant_name, data in results.items():
            f.write(f"{data['params']['description']}\n")
            f.write(f"  Kp = {data['params']['Kp']:.2f}\n")
            f.write(f"  Ki = {data['params']['Ki']:.3f}\n")
            f.write(f"  Kd = {data['params']['Kd']:.3f}\n")
            f.write(f"  Hit Rate: {data['hit_rate']:.1f}%\n")
            f.write(f"  Avg Hit Time: {data['avg_hit_time']:.1f} steps\n")
            f.write(f"  Avg Distance: {data['avg_distance']:.1f}m\n")
            f.write(f"  Final Distance: {data['final_distance']:.1f}m\n")
            f.write("\n")

    print(f"\n✅ Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Ziegler-Nichols PID Tuning')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--all', action='store_true',
                       help='Tune for all maneuver types')

    args = parser.parse_args()

    if args.all:
        maneuvers = ['straight', 'circular', 'zigzag', 'evasive']
    else:
        maneuvers = [args.maneuver]

    all_results = {}

    for maneuver in maneuvers:
        print(f"\n\n{'#'*60}")
        print(f"# ZIEGLER-NICHOLS TUNING: {maneuver.upper()} TARGET")
        print(f"{'#'*60}\n")

        # Step 1: Find ultimate gain and period
        Ku, Tu = find_ultimate_gain(maneuver=maneuver)

        # Step 2: Calculate ZN variants
        zn_variants = calculate_ziegler_nichols_pid(Ku, Tu)

        print("\nCalculated PID Parameters:")
        print("=" * 60)
        for variant_name, params in zn_variants.items():
            print(f"\n{params['description']}:")
            print(f"  Kp = {params['Kp']:.2f}")
            print(f"  Ki = {params['Ki']:.3f}")
            print(f"  Kd = {params['Kd']:.3f}")

        # Step 3: Test all variants
        results = test_zn_variants(zn_variants, maneuver=maneuver, n_test=20)

        # Step 4: Find best variant
        best_variant = max(results.items(), key=lambda x: x[1]['hit_rate'])

        print(f"\n{'='*60}")
        print(f"BEST VARIANT: {best_variant[1]['params']['description']}")
        print(f"  Kp = {best_variant[1]['params']['Kp']:.2f}")
        print(f"  Ki = {best_variant[1]['params']['Ki']:.3f}")
        print(f"  Kd = {best_variant[1]['params']['Kd']:.3f}")
        print(f"  Hit Rate: {best_variant[1]['hit_rate']:.1f}%")
        print(f"{'='*60}")

        # Save results
        save_results(results, maneuver)

        all_results[maneuver] = {
            'Ku': Ku,
            'Tu': Tu,
            'variants': results,
            'best': best_variant
        }

    # Summary
    print(f"\n\n{'#'*60}")
    print("# SUMMARY: ALL MANEUVERS")
    print(f"{'#'*60}\n")

    for maneuver, data in all_results.items():
        best = data['best']
        print(f"\n{maneuver.capitalize()}:")
        print(f"  Best: {best[1]['params']['description']}")
        print(f"  Kp={best[1]['params']['Kp']:.2f}, "
              f"Ki={best[1]['params']['Ki']:.3f}, "
              f"Kd={best[1]['params']['Kd']:.3f}")
        print(f"  Hit Rate: {best[1]['hit_rate']:.1f}%")


if __name__ == "__main__":
    main()
