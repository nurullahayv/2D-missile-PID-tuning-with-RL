"""
Simple Demo without RL
"""
import argparse
import time
from src.missile import Missile
from src.target import Target
from src.renderer import SimpleRenderer


def demo(maneuver='circular', kp=2.0, ki=0.1, kd=0.5, max_steps=1000,
         missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """Run simple demo with fixed PID gains"""

    print(f"Running demo with fixed PID gains:")
    print(f"  Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
    print(f"  Target maneuver: {maneuver}")
    print(f"  Missile: {missile_speed} m/s, {missile_accel} m/s²")
    print(f"  Target: {target_speed} m/s")
    print("\nPress ESC or Q to quit")

    # Initialize
    missile = Missile(x=1000, y=5000, vx=0.8*missile_speed, vy=0,
                     max_speed=missile_speed, max_accel=missile_accel,
                     kp=kp, ki=ki, kd=kd)
    target = Target(x=8000, y=5000, speed=target_speed, maneuver=maneuver)
    renderer = SimpleRenderer(map_size=10000)

    dt = 0.01
    step = 0
    hit_radius = 50.0

    # Main loop
    while step < max_steps and missile.active:
        # Update simulation
        missile.update(target.position, dt)
        target.update(dt, missile.position)

        # Check hit
        distance = ((missile.x - target.x)**2 + (missile.y - target.y)**2)**0.5
        if distance < hit_radius:
            print(f"\n✓ HIT at step {step}! Distance: {distance:.1f}m")
            break

        # Render (every 10 steps for 100Hz physics, 60FPS render)
        if step % 2 == 0:
            cont = renderer.render(missile, target, hit_radius, step)
            if not cont:
                break

        step += 1

    if step >= max_steps:
        print(f"\n✗ Miss - Maximum steps reached")
    elif not missile.active:
        print(f"\n✗ Miss - Missile out of fuel/bounds")

    renderer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Demo missile simulation')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--kp', type=float, default=2.0,
                       help='Proportional gain')
    parser.add_argument('--ki', type=float, default=0.1,
                       help='Integral gain')
    parser.add_argument('--kd', type=float, default=0.5,
                       help='Derivative gain')
    parser.add_argument('--steps', type=int, default=2000,
                       help='Maximum steps')
    parser.add_argument('--missile_speed', type=float, default=1000.0,
                       help='Missile max speed (m/s)')
    parser.add_argument('--missile_accel', type=float, default=1000.0,
                       help='Missile max acceleration (m/s²)')
    parser.add_argument('--target_speed', type=float, default=1000.0,
                       help='Target speed (m/s)')

    args = parser.parse_args()

    demo(
        maneuver=args.maneuver,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        max_steps=args.steps,
        missile_speed=args.missile_speed,
        missile_accel=args.missile_accel,
        target_speed=args.target_speed
    )
