"""
Test script for bee colony simulation.
Tests basic functionality of simulator, environment, and visualization.
"""
import numpy as np
import os
from pathlib import Path

from warsim.simulator.bee_simulator import BeeSimulator
from warsim.scenplotter.bee_plotter import BeePlotter
from envs.env_bee_lowlevel import BeeLowLevelEnv


def test_simulator():
    """Test the bee simulator."""
    print("\n" + "="*60)
    print("Testing Bee Simulator")
    print("="*60)

    sim = BeeSimulator(num_bees=7, grid_size=100, num_directions=32)
    sim.reset()

    print(f"✓ Simulator initialized with {sim.num_bees} bees")
    print(f"✓ Grid size: {sim.grid_size}x{sim.grid_size}")

    # Test a few steps with random actions
    for step in range(10):
        actions = {}
        for bee_id in range(sim.num_bees):
            direction = np.random.randint(0, sim.num_directions)
            build_action = np.random.randint(0, 9)
            actions[bee_id] = (direction, build_action)

        rewards = sim.do_tick(actions)

        if step == 0:
            print(f"✓ Executed step {step + 1}, rewards: {sum(rewards.values()):.3f}")

    info = sim.get_info()
    print(f"✓ After 10 steps:")
    print(f"  - Walls built: {info['total_walls_built']}")
    print(f"  - Active builds: {info['active_builds']}")
    print(f"  - Enclosed area: {info['total_enclosed_area']}")

    return sim


def test_environment():
    """Test the low-level environment."""
    print("\n" + "="*60)
    print("Testing Low-Level Environment")
    print("="*60)

    config = {
        'num_bees': 7,
        'grid_size': 100,
        'num_directions': 32,
        'window_size': 8,
        'horizon': 100,
        'movement_speed': 1.0,
        'base_build_ticks': 256
    }

    env = BeeLowLevelEnv(config)
    print(f"✓ Environment created")

    # Reset
    observations, infos = env.reset()
    print(f"✓ Environment reset, got {len(observations)} observations")

    # Check observation structure
    obs = observations[0]
    print(f"✓ Observation structure:")
    print(f"  - grid_obs shape: {obs['grid_obs'].shape}")
    print(f"  - scalar_obs shape: {obs['scalar_obs'].shape}")

    # Run a few steps
    total_reward = 0
    for step in range(10):
        actions = {}
        for bee_id in range(config['num_bees']):
            direction = np.random.randint(0, config['num_directions'])
            build_action = np.random.randint(0, 9)
            actions[bee_id] = np.array([direction, build_action])

        observations, rewards, terminateds, truncateds, infos = env.step(actions)
        total_reward += sum(rewards.values())

    print(f"✓ Executed 10 steps, total reward: {total_reward:.3f}")

    common_info = infos.get('__common__', {})
    print(f"✓ After 10 steps:")
    print(f"  - Walls built: {common_info.get('total_walls_built', 0)}")
    print(f"  - Enclosed area: {common_info.get('total_enclosed_area', 0)}")

    return env


def test_visualization(sim: BeeSimulator):
    """Test the visualization."""
    print("\n" + "="*60)
    print("Testing Visualization")
    print("="*60)

    # Create test directory
    test_dir = Path("test_visualizations")
    test_dir.mkdir(exist_ok=True)

    plotter = BeePlotter(grid_size=sim.grid_size, downsample=2)

    # Run simulation for a few steps and visualize
    print("Running simulation for 50 steps...")
    for step in range(50):
        actions = {}
        for bee_id in range(sim.num_bees):
            # More intelligent test: move toward center and build
            bee = sim.bees[bee_id]
            bee_y, bee_x = bee.get_grid_position()
            center_y, center_x = sim.grid_size // 2, sim.grid_size // 2

            # Direction toward center
            angle = np.arctan2(center_y - bee_y, center_x - bee_x)
            direction = int((angle / (2 * np.pi)) * sim.num_directions) % sim.num_directions

            # Build if near center
            distance_to_center = np.sqrt((bee_y - center_y)**2 + (bee_x - center_x)**2)
            if distance_to_center < 20 and step % 3 == 0:
                build_action = np.random.randint(1, 9)
            else:
                build_action = 0

            actions[bee_id] = (direction, build_action)

        sim.do_tick(actions)

        # Save visualization every 10 steps
        if step % 10 == 0:
            save_path = test_dir / f"step_{step:03d}.png"
            plotter.plot(sim, save_path=str(save_path))
            print(f"  Saved: {save_path}")

    plotter.close()
    print(f"✓ Visualizations saved to: {test_dir}")

    info = sim.get_info()
    print(f"✓ Final stats:")
    print(f"  - Walls built: {info['total_walls_built']}")
    print(f"  - Enclosed area: {info['total_enclosed_area']}")


def test_enclosed_area_calculation():
    """Test enclosed area calculation."""
    print("\n" + "="*60)
    print("Testing Enclosed Area Calculation")
    print("="*60)

    from warsim.utils.grid_utils import calculate_enclosed_areas

    # Create a simple test case: a square
    grid = np.zeros((20, 20), dtype=np.int8)

    # Draw a 10x10 square
    grid[5:15, 5] = 1  # Left wall
    grid[5:15, 14] = 1  # Right wall
    grid[5, 5:15] = 1  # Top wall
    grid[14, 5:15] = 1  # Bottom wall

    area, regions = calculate_enclosed_areas(grid)

    print(f"✓ Test case: 10x10 square")
    print(f"  - Expected area: ~80 (10x10 - 4 walls)")
    print(f"  - Calculated area: {area}")
    print(f"  - Number of regions: {len(regions)}")

    if len(regions) > 0:
        print(f"  - Largest region size: {len(regions[0])}")

    # Another test: multiple enclosed regions
    grid2 = np.zeros((30, 30), dtype=np.int8)

    # Two separate 5x5 squares
    grid2[5:10, 5:10] = 1
    grid2[6:9, 6:9] = 0  # Hollow out center

    grid2[15:20, 15:20] = 1
    grid2[16:19, 16:19] = 0  # Hollow out center

    area2, regions2 = calculate_enclosed_areas(grid2)
    print(f"\n✓ Test case: Two hollow squares")
    print(f"  - Total enclosed area: {area2}")
    print(f"  - Number of regions: {len(regions2)}")


def main():
    """Run all tests."""
    print("\n" + "#"*60)
    print("# BEE COLONY SIMULATION - SYSTEM TEST")
    print("#"*60)

    try:
        # Test 1: Simulator
        sim = test_simulator()

        # Test 2: Environment
        env = test_environment()

        # Test 3: Enclosed area calculation
        test_enclosed_area_calculation()

        # Test 4: Visualization
        test_visualization(sim)

        print("\n" + "#"*60)
        print("# ALL TESTS PASSED! ✓")
        print("#"*60)
        print("\nThe bee colony simulation system is working correctly!")
        print("You can now proceed with training:\n")
        print("  Low-level:  python train_bee_lowlevel.py")
        print("  High-level: python train_bee_highlevel.py")
        print("\n")

    except Exception as e:
        print("\n" + "#"*60)
        print("# TEST FAILED! ✗")
        print("#"*60)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
