"""
Demo: Basic PID Missile vs Moving Target
Real-time visualization with Pygame
Configurable PID parameters
"""
import sys
import os
import numpy as np
import yaml
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from warsim.simulator.missile import Missile
from warsim.simulator.target import Target
from warsim.visualization.pygame_renderer import PygameRenderer


def load_config(config_path: str = "config_pid.yaml") -> dict:
    """Load PID configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_basic_pid_demo(target_maneuver: str = "circular",
                       pid_config: dict = None,
                       max_steps: int = 500):
    """
    Run basic PID demonstration

    Args:
        target_maneuver: Target maneuver type
        pid_config: PID configuration dict
        max_steps: Maximum simulation steps
    """
    # Load config
    config = load_config()

    # Get PID parameters
    if pid_config is None:
        pid_params = config['default_pid']
    else:
        pid_params = pid_config

    kp = pid_params['kp']
    ki = pid_params['ki']
    kd = pid_params['kd']

    # Simulation parameters
    dt = config['simulation']['dt']
    map_size = config['simulation']['map_size']
    hit_radius = config['simulation']['hit_radius']

    print("=" * 60)
    print("Basic PID Missile vs Moving Target")
    print("=" * 60)
    print(f"Target Maneuver: {target_maneuver}")
    print(f"PID Parameters: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    print(f"Map Size: {map_size}m x {map_size}m")
    print("=" * 60)
    print("\nPress ESC or Q to quit\n")

    # Random initial positions
    missile_x = np.random.uniform(0, map_size * 0.3)
    missile_y = np.random.uniform(0, map_size * 0.3)
    target_x = np.random.uniform(map_size * 0.6, map_size)
    target_y = np.random.uniform(map_size * 0.6, map_size)

    # Initialize missile
    missile_heading = np.random.uniform(0, 360)
    missile_speed = config['missile']['initial_speed']
    missile_vx = missile_speed * np.cos(missile_heading * np.pi / 180)
    missile_vy = missile_speed * np.sin(missile_heading * np.pi / 180)

    missile = Missile(
        x=missile_x, y=missile_y,
        vx=missile_vx, vy=missile_vy,
        max_speed=config['missile']['max_speed'],
        max_acceleration=config['missile']['max_acceleration'],
        pid_kp=kp, pid_ki=ki, pid_kd=kd
    )

    # Initialize target
    target_heading = np.random.uniform(0, 360)
    target = Target(
        x=target_x, y=target_y,
        speed=config['target']['speed'],
        maneuver_type=target_maneuver
    )
    target.heading = target_heading
    target.initial_heading = target_heading

    # Initialize renderer
    renderer = PygameRenderer(map_size=map_size, window_size=(1200, 1000), fps=60)

    # Simulation loop
    step = 0
    hit = False
    out_of_fuel = False
    out_of_bounds = False

    print("Simulation running...")

    while step < max_steps and renderer.is_running():
        # Calculate distance
        dx = target.x - missile.x
        dy = target.y - missile.y
        distance = np.sqrt(dx**2 + dy**2)

        # Update entities
        missile.update(target.x, target.y, dt)
        target.update(dt, missile_position=(missile.x, missile.y))

        # Render frame
        pid_gains = {
            'kp': missile.pid.kp,
            'ki': missile.pid.ki,
            'kd': missile.pid.kd
        }

        success = renderer.render_frame(
            missile_trajectory=missile.trajectory,
            target_trajectory=target.trajectory,
            missile_heading=missile.heading,
            target_heading=target.heading,
            hit_radius=hit_radius,
            step=step,
            distance=distance,
            pid_gains=pid_gains,
            fuel=missile.fuel_remaining,
            mode="Basic PID (Fixed)",
            title=f"Basic PID Demo - {target_maneuver.capitalize()} Target"
        )

        if not success:
            break

        # Check termination conditions
        if distance < hit_radius:
            hit = True
            print(f"\n✓ TARGET HIT at step {step}")
            print(f"   Distance: {distance:.2f}m")
            break

        if not missile.active:
            out_of_fuel = True
            print(f"\n✗ OUT OF FUEL at step {step}")
            break

        if (missile.x < 0 or missile.x > map_size or
            missile.y < 0 or missile.y > map_size):
            out_of_bounds = True
            print(f"\n✗ OUT OF BOUNDS at step {step}")
            break

        step += 1

    # Final stats
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Total Steps: {step}")
    print(f"Final Distance: {distance:.2f}m")
    print(f"Result: {'HIT' if hit else ('OUT OF FUEL' if out_of_fuel else ('OUT OF BOUNDS' if out_of_bounds else 'TIME OUT'))}")
    print(f"Fuel Remaining: {missile.fuel_remaining*100:.1f}%")
    print(f"PID Gains: Kp={missile.pid.kp:.3f}, Ki={missile.pid.ki:.3f}, Kd={missile.pid.kd:.3f}")
    print("=" * 60)

    # Keep window open until user closes
    print("\nClose window or press ESC to exit...")
    while renderer.is_running():
        renderer.render_frame(
            missile_trajectory=missile.trajectory,
            target_trajectory=target.trajectory,
            missile_heading=missile.heading,
            target_heading=target.heading,
            hit_radius=hit_radius,
            step=step,
            distance=distance,
            pid_gains=pid_gains,
            fuel=missile.fuel_remaining,
            mode="Basic PID (Fixed)",
            title=f"Final State - {('HIT!' if hit else 'MISS')}"
        )

    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic PID Missile Demo')
    parser.add_argument('--target', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--kp', type=float, default=None,
                       help='PID Kp parameter (default from config)')
    parser.add_argument('--ki', type=float, default=None,
                       help='PID Ki parameter (default from config)')
    parser.add_argument('--kd', type=float, default=None,
                       help='PID Kd parameter (default from config)')
    parser.add_argument('--use_optimal', action='store_true',
                       help='Use optimal PID parameters from config')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum simulation steps')

    args = parser.parse_args()

    # Build PID config
    config = load_config()
    if args.use_optimal:
        pid_config = config['optimal_pid']
    elif args.kp is not None or args.ki is not None or args.kd is not None:
        # Use custom parameters
        pid_config = {
            'kp': args.kp if args.kp is not None else config['default_pid']['kp'],
            'ki': args.ki if args.ki is not None else config['default_pid']['ki'],
            'kd': args.kd if args.kd is not None else config['default_pid']['kd'],
        }
    else:
        # Use default
        pid_config = config['default_pid']

    run_basic_pid_demo(
        target_maneuver=args.target,
        pid_config=pid_config,
        max_steps=args.max_steps
    )
