"""
Demo: Basic PID Missile vs Moving Target
3 Modes: fast (no render), realtime (smooth 60fps), replay
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
from warsim.simulator.simulation_engine import (
    SimulationEngine, RealtimeSimulation, ReplaySimulation
)
from warsim.visualization.pygame_renderer import PygameRenderer


def load_config(config_path: str = "config_pid.yaml") -> dict:
    """Load PID configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_basic_pid_demo(target_maneuver: str = "circular",
                       pid_config: dict = None,
                       max_steps: int = 50000,
                       mode: str = "realtime",
                       playback_speed: float = 1.0):
    """
    Run basic PID demonstration

    Args:
        target_maneuver: Target maneuver type
        pid_config: PID configuration dict
        max_steps: Maximum simulation steps
        mode: Simulation mode ('fast', 'realtime', 'replay')
        playback_speed: Playback speed for replay mode
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
    map_size = config['simulation']['map_size']
    hit_radius = config['simulation']['hit_radius']
    physics_hz = 100  # High frequency physics

    print("=" * 70)
    print("Basic PID Missile vs Moving Target - Hybrid Simulation System")
    print("=" * 70)
    print(f"Mode: {mode.upper()}")
    print(f"Target Maneuver: {target_maneuver}")
    print(f"PID Parameters: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    print(f"Physics Frequency: {physics_hz} Hz")
    if mode == "realtime":
        print(f"Render FPS: 60")
    elif mode == "replay":
        print(f"Playback Speed: {playback_speed}x")
    print("=" * 70)

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

    # Create simulation engine
    engine = SimulationEngine(
        missile=missile,
        target=target,
        map_size=map_size,
        hit_radius=hit_radius,
        physics_hz=physics_hz,
        max_steps=max_steps
    )

    # Run based on mode
    if mode == "fast":
        # Fast simulation without rendering
        print("\nRunning FAST mode (no rendering)...\n")
        history = engine.simulate_fast(record_interval=10)

        # Print results
        print("\n" + "=" * 70)
        print("Simulation Results")
        print("=" * 70)
        print(f"Result: {'HIT' if history.hit else 'MISS'}")
        print(f"Final Distance: {history.final_distance:.2f}m")
        print(f"Physics Steps: {history.total_steps:,}")
        print(f"Simulation Time: {history.total_time:.2f}s")
        print("=" * 70)

        # Ask if user wants to replay
        print("\nWould you like to replay this simulation? (y/n): ", end='')
        choice = input().strip().lower()
        if choice == 'y':
            renderer = PygameRenderer(map_size=map_size, window_size=(1200, 1000), fps=60)
            replay = ReplaySimulation(history, renderer)
            replay.replay(playback_speed=1.0, render_fps=60)
            renderer.close()

    elif mode == "realtime":
        # Real-time simulation with smooth rendering
        print("\nRunning REALTIME mode (100Hz physics, 60FPS render)...")
        print("Press ESC or Q to quit\n")

        renderer = PygameRenderer(map_size=map_size, window_size=(1200, 1000), fps=60)
        realtime = RealtimeSimulation(engine, renderer, render_fps=60)
        history = realtime.run(mode_info="Basic PID (Fixed)")

        # Final stats
        print("\n" + "=" * 70)
        print("Simulation Results")
        print("=" * 70)
        print(f"Result: {'HIT' if history.hit else 'MISS'}")
        print(f"Final Distance: {history.final_distance:.2f}m")
        print(f"Physics Steps: {history.total_steps:,}")
        print(f"Simulation Time: {history.total_time:.2f}s")
        print("=" * 70)

        # Keep final frame visible
        print("\nClose window to exit...")
        while renderer.is_running():
            state = engine.get_current_state()
            renderer.render_frame(
                missile_trajectory=history.missile_trajectory,
                target_trajectory=history.target_trajectory,
                missile_heading=state.missile_heading,
                target_heading=state.target_heading,
                hit_radius=hit_radius,
                step=state.step,
                distance=state.distance,
                pid_gains={'kp': state.pid_kp, 'ki': state.pid_ki, 'kd': state.pid_kd},
                fuel=state.missile_fuel,
                mode="Basic PID (Fixed)",
                title=f"Final State - {'HIT!' if history.hit else 'MISS'}"
            )

        renderer.close()

    elif mode == "replay":
        # This mode requires pre-recorded history
        print("\nREPLAY mode requires pre-recorded simulation.")
        print("Running fast simulation first...\n")

        history = engine.simulate_fast(record_interval=10)

        print(f"\nReplaying at {playback_speed}x speed...")
        print("Press ESC or Q to quit\n")

        renderer = PygameRenderer(map_size=map_size, window_size=(1200, 1000), fps=60)
        replay = ReplaySimulation(history, renderer)
        replay.replay(playback_speed=playback_speed, render_fps=60)

        print("\n" + "=" * 70)
        print("Simulation Results")
        print("=" * 70)
        print(f"Result: {'HIT' if history.hit else 'MISS'}")
        print(f"Final Distance: {history.final_distance:.2f}m")
        print(f"Physics Steps: {history.total_steps:,}")
        print(f"Simulation Time: {history.total_time:.2f}s")
        print("=" * 70)

        renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Basic PID Missile Demo - Hybrid Simulation')
    parser.add_argument('--target', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--mode', type=str, default='realtime',
                       choices=['fast', 'realtime', 'replay'],
                       help='Simulation mode: fast (no render), realtime (smooth 60fps), replay')
    parser.add_argument('--kp', type=float, default=None,
                       help='PID Kp parameter (default from config)')
    parser.add_argument('--ki', type=float, default=None,
                       help='PID Ki parameter (default from config)')
    parser.add_argument('--kd', type=float, default=None,
                       help='PID Kd parameter (default from config)')
    parser.add_argument('--use_optimal', action='store_true',
                       help='Use optimal PID parameters from config')
    parser.add_argument('--max_steps', type=int, default=50000,
                       help='Maximum simulation steps')
    parser.add_argument('--playback_speed', type=float, default=1.0,
                       help='Playback speed for replay mode (e.g., 0.5 = slow-mo, 2.0 = fast)')

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
        max_steps=args.max_steps,
        mode=args.mode,
        playback_speed=args.playback_speed
    )
