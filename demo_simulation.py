"""
Demo Simulation with Random PID Parameters
Visualize missile guidance with default/random PID values
"""
import sys
import os
import numpy as np
import time
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from warsim.simulator.missile import Missile
from warsim.simulator.target import Target
from warsim.visualization.neon_renderer import NeonRenderer


def run_demo(target_maneuver: str = "circular",
            pid_mode: str = "random",
            max_steps: int = 500,
            dt: float = 0.1,
            save_video: bool = False):
    """
    Run demo simulation with visualization

    Args:
        target_maneuver: Target maneuver type
        pid_mode: "random", "default", or "optimal"
        max_steps: Maximum simulation steps
        dt: Time step
        save_video: Save frames as images
    """
    print("=" * 60)
    print("Missile PID Control - Demo Simulation")
    print("=" * 60)
    print(f"Target Maneuver: {target_maneuver}")
    print(f"PID Mode: {pid_mode}")
    print("=" * 60)
    print("\nStarting simulation...\n")

    # Map size
    map_size = 10000.0

    # Initialize missile with PID parameters
    if pid_mode == "random":
        kp = np.random.uniform(0.5, 5.0)
        ki = np.random.uniform(0.0, 2.0)
        kd = np.random.uniform(0.0, 2.0)
        print(f"Random PID: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    elif pid_mode == "default":
        kp, ki, kd = 2.0, 0.1, 0.5
        print(f"Default PID: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    elif pid_mode == "optimal":
        # These are typical optimal values for circular target
        kp, ki, kd = 3.2, 0.15, 0.8
        print(f"Optimal PID: Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    else:
        kp, ki, kd = 2.0, 0.1, 0.5

    # Random initial positions
    missile_x = np.random.uniform(0, map_size * 0.3)
    missile_y = np.random.uniform(0, map_size * 0.3)
    target_x = np.random.uniform(map_size * 0.6, map_size)
    target_y = np.random.uniform(map_size * 0.6, map_size)

    # Initialize missile
    missile_heading = np.random.uniform(0, 360)
    missile_speed = 250.0
    missile_vx = missile_speed * np.cos(missile_heading * np.pi / 180)
    missile_vy = missile_speed * np.sin(missile_heading * np.pi / 180)

    missile = Missile(
        x=missile_x, y=missile_y,
        vx=missile_vx, vy=missile_vy,
        max_speed=300.0,
        max_acceleration=100.0,
        pid_kp=kp, pid_ki=ki, pid_kd=kd
    )

    # Initialize target
    target_heading = np.random.uniform(0, 360)
    target = Target(
        x=target_x, y=target_y,
        speed=150.0,
        maneuver_type=target_maneuver
    )
    target.heading = target_heading
    target.initial_heading = target_heading

    # Initialize renderer
    renderer = NeonRenderer(map_size=map_size, dpi=100)

    # Hit radius
    hit_radius = 50.0

    # Simulation loop
    step = 0
    hit = False

    if save_video:
        os.makedirs("demo_frames", exist_ok=True)

    print("Simulating... (Close window to stop)\n")

    try:
        while step < max_steps and missile.active:
            # Calculate distance
            dx = target.x - missile.x
            dy = target.y - missile.y
            distance = np.sqrt(dx**2 + dy**2)

            # Update entities
            missile.update(target.x, target.y, dt)
            target.update(dt, missile_position=(missile.x, missile.y))

            # Render frame
            if step % 2 == 0:  # Render every 2 steps for performance
                pid_gains = {
                    'kp': missile.pid.kp,
                    'ki': missile.pid.ki,
                    'kd': missile.pid.kd
                }

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
                    title=f"Missile PID Demo - {target_maneuver.capitalize()} Target"
                )

                if save_video:
                    renderer.save_frame(f"demo_frames/frame_{step:04d}.png")

                renderer.show()

            # Check hit
            if distance < hit_radius:
                hit = True
                print(f"\n✓ HIT! at step {step}, distance: {distance:.2f}m")
                break

            # Check out of bounds
            if (missile.x < 0 or missile.x > map_size or
                missile.y < 0 or missile.y > map_size):
                print(f"\n✗ OUT OF BOUNDS at step {step}")
                break

            step += 1
            time.sleep(0.01)  # Small delay for visualization

    except KeyboardInterrupt:
        print("\n\nSimulation interrupted by user")

    # Final stats
    print("\n" + "=" * 60)
    print("Simulation Complete")
    print("=" * 60)
    print(f"Total Steps: {step}")
    print(f"Final Distance: {distance:.2f}m")
    print(f"Hit: {'YES' if hit else 'NO'}")
    print(f"Fuel Remaining: {missile.fuel_remaining*100:.1f}%")
    print(f"Final PID Gains: Kp={missile.pid.kp:.3f}, Ki={missile.pid.ki:.3f}, Kd={missile.pid.kd:.3f}")
    print("=" * 60)

    if save_video:
        print(f"\nFrames saved to: demo_frames/")
        print("To create video:")
        print("  ffmpeg -r 30 -i demo_frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p demo_video.mp4")

    # Keep window open
    input("\nPress Enter to close...")
    renderer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Missile PID Demo Simulation')
    parser.add_argument('--target', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--pid_mode', type=str, default='random',
                       choices=['random', 'default', 'optimal'],
                       help='PID parameter mode')
    parser.add_argument('--max_steps', type=int, default=500,
                       help='Maximum simulation steps')
    parser.add_argument('--save_video', action='store_true',
                       help='Save frames for video creation')

    args = parser.parse_args()

    run_demo(
        target_maneuver=args.target,
        pid_mode=args.pid_mode,
        max_steps=args.max_steps,
        save_video=args.save_video
    )
