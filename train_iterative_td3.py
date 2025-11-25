"""
Training script for Iterative PID Refinement with TD3

Multi-step refinement approach where agent learns to iteratively improve PID:
- Episode = Multiple refinement steps (max 15)
- Action = Delta PID (incremental changes)
- State = Current PID + Performance + History
- Agent learns optimal refinement strategy

Why TD3:
- Continuous control (delta PID in continuous space)
- Deterministic policy gradient (stable refinement)
- Twin Q-networks (reduced overestimation)
- Target policy smoothing (robust to local optima)

Usage:
    python train_iterative_td3.py --maneuver circular --timesteps 200000 --n_envs 8
    python train_iterative_td3.py --maneuver zigzag --timesteps 300000
"""
import argparse
import os
from datetime import datetime
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from src.iterative_pid_env import IterativePIDEnv


def make_env(rank, maneuver='circular', n_test_scenarios=5, history_size=3,
             max_iterations=15, max_steps=1300,
             missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """Create a monitored Iterative PID environment"""
    def _init():
        env = IterativePIDEnv(
            n_test_scenarios=n_test_scenarios,
            history_size=history_size,
            max_iterations_per_episode=max_iterations,
            max_steps=max_steps,
            target_maneuver=maneuver,
            missile_speed=missile_speed,
            missile_accel=missile_accel,
            target_speed=target_speed
        )
        env = Monitor(env)
        return env
    return _init


def train(maneuver='circular', n_envs=8,
          total_timesteps=200_000, save_freq=20_000,
          n_test_scenarios=5, history_size=3, max_iterations=15, max_steps=1300,
          missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """
    Train TD3 agent for iterative PID refinement

    Args:
        maneuver: Target maneuver type (straight, circular, zigzag, evasive)
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps (refinement steps)
        save_freq: Checkpoint save frequency
        n_test_scenarios: Number of scenarios to test each PID on (default: 5)
        history_size: History size for state (default: 3)
        max_iterations: Max refinement steps per episode (default: 15)
        max_steps: Maximum simulation steps per scenario (default: 1300)
        missile_speed: Missile max speed (m/s)
        missile_accel: Missile max acceleration (m/s²)
        target_speed: Target speed (m/s)
    """
    print(f"\n{'='*70}")
    print(f"ITERATIVE PID REFINEMENT WITH TD3")
    print(f"{'='*70}")
    print(f"Target Maneuver: {maneuver}")
    print(f"Missile: {missile_speed} m/s, {missile_accel} m/s²")
    print(f"Target: {target_speed} m/s")
    print(f"Test Scenarios per PID: {n_test_scenarios}")
    print(f"History Size: {history_size}")
    print(f"Max Iterations per Episode: {max_iterations}")
    print(f"Max Steps per Scenario: {max_steps}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"  → Approx Episodes: {total_timesteps // max_iterations:,}")
    print(f"  → Approx PID tests: {total_timesteps:,}")
    print(f"  → Approx simulations: {total_timesteps * n_test_scenarios:,}")
    print(f"{'='*70}\n")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/iterative_td3/{maneuver}_{timestamp}"
    log_dir = f"logs/iterative_td3/{maneuver}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Model directory: {model_dir}")
    print(f"Log directory: {log_dir}\n")

    # Create vectorized environments
    print("Creating environments...")
    print("Using SubprocVecEnv for parallel CPU execution (works with Numba JIT)")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, maneuver, n_test_scenarios, history_size, max_iterations, max_steps,
                    missile_speed, missile_accel, target_speed)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, maneuver, n_test_scenarios, history_size, max_iterations, max_steps,
                    missile_speed, missile_accel, target_speed)
        ])

    # Create eval environment
    eval_env = DummyVecEnv([
        make_env(0, maneuver, n_test_scenarios, history_size, max_iterations, max_steps,
                missile_speed, missile_accel, target_speed)
    ])

    # Initialize TD3 model
    print(f"Initializing TD3 model...")

    # Action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # TD3 configuration
    model = TD3(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        policy_delay=2,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        action_noise=action_noise,
        policy_kwargs={
            'net_arch': [256, 256, 256]  # 3-layer network
        },
        verbose=1,
        tensorboard_log=log_dir
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_dir,
        name_prefix="td3_iterative_pid"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=5_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Train
    print("\nStarting training...")
    print(f"Monitor training with: tensorboard --logdir {log_dir}\n")
    print(f"Note: Each timestep = 1 refinement step")
    print(f"      Each episode = up to {max_iterations} refinement steps\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_path = os.path.join(model_dir, "td3_iterative_pid_final.zip")
    model.save(final_path)
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*70}\n")

    # Test the learned model
    print("Testing learned Iterative TD3 agent...")
    test_learned_iterative_pid(model, maneuver, n_test_scenarios, history_size,
                                max_iterations, max_steps,
                                missile_speed, missile_accel, target_speed)

    return model


def test_learned_iterative_pid(model, maneuver='circular',
                                n_test_scenarios=5, history_size=3,
                                max_iterations=15, max_steps=1300,
                                missile_speed=1000.0, missile_accel=1000.0,
                                target_speed=1000.0, n_episodes=10):
    """
    Test the learned Iterative TD3 model

    Args:
        model: Trained TD3 model
        maneuver: Target maneuver type
        n_test_scenarios: Number of scenarios to test each PID on
        history_size: History size
        max_iterations: Max iterations per episode
        max_steps: Maximum simulation steps per scenario
        n_episodes: Number of episodes to test
    """
    print(f"\n{'='*70}")
    print(f"TESTING ITERATIVE TD3 AGENT")
    print(f"{'='*70}\n")

    env = IterativePIDEnv(
        n_test_scenarios=n_test_scenarios,
        history_size=history_size,
        max_iterations_per_episode=max_iterations,
        max_steps=max_steps,
        target_maneuver=maneuver,
        missile_speed=missile_speed,
        missile_accel=missile_accel,
        target_speed=target_speed
    )

    all_episode_results = []

    for episode in range(n_episodes):
        print(f"\n--- Episode {episode + 1} ---")

        obs, _ = env.reset()
        episode_reward = 0
        refinement_steps = []

        for step in range(max_iterations):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)

            episode_reward += reward
            refinement_steps.append({
                'step': step,
                'pid': (info['pid_kp'], info['pid_ki'], info['pid_kd']),
                'success_rate': info['success_rate'],
                'avg_time': info['avg_time']
            })

            print(f"  Step {step+1:2d}: "
                  f"PID=(Kp={info['pid_kp']:.0f}, Ki={info['pid_ki']:.1f}, Kd={info['pid_kd']:.1f}) | "
                  f"SR={info['success_rate']*100:.1f}% | "
                  f"Time={info['avg_time']:.1f}s | "
                  f"Reward={reward:7.1f}")

            if done:
                if info['converged']:
                    print(f"  → Converged at step {step+1}")
                else:
                    print(f"  → Max iterations reached")
                break

        final_info = refinement_steps[-1]
        all_episode_results.append({
            'episode_reward': episode_reward,
            'final_pid': final_info['pid'],
            'final_success_rate': final_info['success_rate'],
            'final_avg_time': final_info['avg_time'],
            'num_steps': len(refinement_steps),
            'best_success_rate': info['best_success_rate'],
            'best_avg_time': info['best_avg_time']
        })

        print(f"  Episode Reward: {episode_reward:.1f}")
        print(f"  Final SR: {final_info['success_rate']*100:.1f}% | "
              f"Final Time: {final_info['avg_time']:.1f}s")
        print(f"  Best SR: {info['best_success_rate']*100:.1f}% | "
              f"Best Time: {info['best_avg_time']:.1f}s")

    # Calculate statistics
    avg_final_sr = np.mean([r['final_success_rate'] for r in all_episode_results]) * 100
    avg_best_sr = np.mean([r['best_success_rate'] for r in all_episode_results]) * 100
    avg_final_time = np.mean([r['final_avg_time'] for r in all_episode_results])
    avg_best_time = np.mean([r['best_avg_time'] for r in all_episode_results])
    avg_num_steps = np.mean([r['num_steps'] for r in all_episode_results])

    # Final PID statistics
    final_pids = [r['final_pid'] for r in all_episode_results]
    avg_kp = np.mean([p[0] for p in final_pids])
    avg_ki = np.mean([p[1] for p in final_pids])
    avg_kd = np.mean([p[2] for p in final_pids])
    std_kp = np.std([p[0] for p in final_pids])
    std_ki = np.std([p[1] for p in final_pids])
    std_kd = np.std([p[2] for p in final_pids])

    print(f"\n{'='*70}")
    print(f"TEST RESULTS ({n_episodes} episodes)")
    print(f"{'='*70}")
    print(f"Average Final Success Rate: {avg_final_sr:.1f}%")
    print(f"Average Best Success Rate: {avg_best_sr:.1f}%")
    print(f"Average Final Time: {avg_final_time:.1f}s")
    print(f"Average Best Time: {avg_best_time:.1f}s")
    print(f"Average Refinement Steps: {avg_num_steps:.1f}/{max_iterations}")
    print(f"\nFinal PID Parameters (averaged):")
    print(f"  Kp = {avg_kp:.1f} ± {std_kp:.1f}")
    print(f"  Ki = {avg_ki:.2f} ± {std_ki:.2f}")
    print(f"  Kd = {avg_kd:.2f} ± {std_kd:.2f}")
    print(f"\nNote: Agent should converge to good PID with few refinement steps")
    print(f"{'='*70}\n")

    return avg_kp, avg_ki, avg_kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Iterative PID Refinement with TD3')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--n_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=200_000,
                       help='Total training timesteps (refinement steps)')
    parser.add_argument('--save_freq', type=int, default=20_000,
                       help='Checkpoint save frequency')
    parser.add_argument('--n_test_scenarios', type=int, default=5,
                       help='Number of scenarios to test each PID on')
    parser.add_argument('--history_size', type=int, default=3,
                       help='History size for state')
    parser.add_argument('--max_iterations', type=int, default=15,
                       help='Maximum refinement steps per episode')
    parser.add_argument('--max_steps', type=int, default=1300,
                       help='Maximum simulation steps per scenario')
    parser.add_argument('--missile_speed', type=float, default=1000.0,
                       help='Missile max speed (m/s)')
    parser.add_argument('--missile_accel', type=float, default=1000.0,
                       help='Missile max acceleration (m/s²)')
    parser.add_argument('--target_speed', type=float, default=1000.0,
                       help='Target speed (m/s)')

    args = parser.parse_args()

    train(
        maneuver=args.maneuver,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        n_test_scenarios=args.n_test_scenarios,
        history_size=args.history_size,
        max_iterations=args.max_iterations,
        max_steps=args.max_steps,
        missile_speed=args.missile_speed,
        missile_accel=args.missile_accel,
        target_speed=args.target_speed
    )
