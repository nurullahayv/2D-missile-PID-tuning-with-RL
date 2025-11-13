"""
Simple Demo without RL
"""
import argparse
import time
from src.missile import Missile
from src.target import Target
from src.renderer import SimpleRenderer


def demo(maneuver='circular', kp=2.0, ki=0.1, kd=0.5, max_steps=1000):
    """Run simple demo with fixed PID gains"""

    print(f"Running demo with fixed PID gains:")
    print(f"  Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")
    print(f"  Target maneuver: {maneuver}")
    print("\nPress ESC or Q to quit")

    # Initialize
    missile = Missile(x=1000, y=5000, vx=250, vy=0,
                     kp=kp, ki=ki, kd=kd)
    target = Target(x=8000, y=5000, speed=150, maneuver=maneuver)
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

    args = parser.parse_args()

    demo(
        maneuver=args.maneuver,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        max_steps=args.steps
    )
