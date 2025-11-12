"""
Quick test script to verify the setup
"""
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    try:
        from envs.missile_pid_env import MissilePIDEnv
        from warsim.simulator.missile import Missile, PIDController
        from warsim.simulator.target import Target
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import error: {e}")
        return False


def test_missile():
    """Test missile dynamics"""
    print("\nTesting missile...")
    try:
        from warsim.simulator.missile import Missile

        missile = Missile(x=0, y=0, vx=100, vy=0,
                         pid_kp=2.0, pid_ki=0.1, pid_kd=0.5)

        # Simulate for a few steps
        for _ in range(10):
            missile.update(target_x=1000, target_y=1000, dt=0.1)

        assert missile.active, "Missile should be active"
        assert len(missile.trajectory) > 1, "Missile should have trajectory"
        print(f"  ✓ Missile simulation works (position: {missile.position})")
        return True
    except Exception as e:
        print(f"  ✗ Missile test error: {e}")
        return False


def test_target():
    """Test target dynamics"""
    print("\nTesting target...")
    try:
        from warsim.simulator.target import Target

        target = Target(x=1000, y=1000, speed=150, maneuver_type="circular")

        # Simulate for a few steps
        for _ in range(10):
            target.update(dt=0.1)

        assert len(target.trajectory) > 1, "Target should have trajectory"
        print(f"  ✓ Target simulation works (position: {target.position})")
        return True
    except Exception as e:
        print(f"  ✗ Target test error: {e}")
        return False


def test_environment():
    """Test Gymnasium environment"""
    print("\nTesting environment...")
    try:
        from envs.missile_pid_env import MissilePIDEnv

        env = MissilePIDEnv(max_steps=100, dt=0.1)

        # Reset
        obs, info = env.reset(seed=42)
        assert obs.shape == (14,), f"Observation shape should be (14,), got {obs.shape}"

        # Take a few random steps
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        print(f"  ✓ Environment works (steps taken, final reward: {reward:.2f})")
        return True
    except Exception as e:
        print(f"  ✗ Environment test error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")
    try:
        from config import get_default_config

        config = get_default_config("circular")
        assert config['target_maneuver'] == "circular"
        assert 'max_steps' in config
        print("  ✓ Configuration works")
        return True
    except Exception as e:
        print(f"  ✗ Config test error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Testing Missile PID RL Setup")
    print("=" * 60)

    results = []
    results.append(("Imports", test_imports()))
    results.append(("Missile", test_missile()))
    results.append(("Target", test_target()))
    results.append(("Environment", test_environment()))
    results.append(("Config", test_config()))

    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s}: {status}")

    all_passed = all(result for _, result in results)

    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
        print("\nYou can now:")
        print("1. Start training: python train.py")
        print("2. Or use Kaggle notebook: kaggle_training.ipynb")
    else:
        print("Some tests failed! ✗")
        print("Please check the errors above.")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
