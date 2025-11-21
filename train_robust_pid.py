"""
Training script for Robust Non-Adaptive PID tuning with RL

Key improvements over previous approach:
1. Standard episodic structure (no meta-episodes)
2. Each PID tested on multiple scenarios for robustness
3. History-based observation (no context confusion)
4. Continuous distance-based rewards (not just hit/miss)
5. Narrowed action space for faster convergence
6. Curriculum learning support (optional)

Usage:
    python train_robust_pid.py --algorithm PPO --maneuver circular --timesteps 100000
    python train_robust_pid.py --algorithm SAC --maneuver circular --timesteps 100000 --n_envs 8
"""
import argparse
import os
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.robust_pid_env import RobustPIDEnv


def make_env(rank, maneuver='circular', n_test_scenarios=5, window_size=10,
             max_steps=1300, missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """Create a monitored Robust PID environment"""
    def _init():
        env = RobustPIDEnv(
            n_test_scenarios=n_test_scenarios,
            window_size=window_size,
            max_steps=max_steps,
            target_maneuver=maneuver,
            missile_speed=missile_speed,
            missile_accel=missile_accel,
            target_speed=target_speed
        )
        env = Monitor(env)
        return env
    return _init


def train(algorithm='PPO', maneuver='circular', n_envs=8,
          total_timesteps=100_000, save_freq=10_000,
          n_test_scenarios=5, window_size=10, max_steps=1300,
          missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """
    Train RL agent for robust non-adaptive PID tuning

    Args:
        algorithm: RL algorithm (PPO, SAC, A2C)
        maneuver: Target maneuver type (straight, circular, zigzag, evasive)
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps (episodes)
        save_freq: Checkpoint save frequency
        n_test_scenarios: Number of scenarios to test each PID on (default: 5)
        window_size: History window size (default: 10)
        max_steps: Maximum simulation steps per scenario (default: 1300)
        missile_speed: Missile max speed (m/s)
        missile_accel: Missile max acceleration (m/s²)
        target_speed: Target speed (m/s)
    """
    print(f"\n{'='*70}")
    print(f"ROBUST NON-ADAPTIVE PID TUNING WITH {algorithm}")
    print(f"{'='*70}")
    print(f"Target Maneuver: {maneuver}")
    print(f"Missile: {missile_speed} m/s, {missile_accel} m/s²")
    print(f"Target: {target_speed} m/s")
    print(f"Test Scenarios per PID: {n_test_scenarios}")
    print(f"History Window Size: {window_size}")
    print(f"Max Steps per Scenario: {max_steps}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"  → Total PID trials: {total_timesteps:,}")
    print(f"  → Total simulations: {total_timesteps * n_test_scenarios:,}")
    print(f"  → Total simulation steps: {total_timesteps * n_test_scenarios * max_steps:,}")
    print(f"{'='*70}\n")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/robust_pid/{algorithm.lower()}_{maneuver}_{timestamp}"
    log_dir = f"logs/robust_pid/{algorithm.lower()}_{maneuver}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Model directory: {model_dir}")
    print(f"Log directory: {log_dir}\n")

    # Create vectorized environments
    print("Creating environments...")
    print("Using SubprocVecEnv for parallel CPU execution (works with Numba JIT)")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, maneuver, n_test_scenarios, window_size, max_steps,
                    missile_speed, missile_accel, target_speed)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, maneuver, n_test_scenarios, window_size, max_steps,
                    missile_speed, missile_accel, target_speed)
        ])

    # Create eval environment
    eval_env = DummyVecEnv([
        make_env(0, maneuver, n_test_scenarios, window_size, max_steps,
                missile_speed, missile_accel, target_speed)
    ])

    # Initialize model
    print(f"Initializing {algorithm} model...")

    if algorithm == 'PPO':
        # PPO with increased exploration
        policy_kwargs = dict(net_arch=[256, 256, 256])
        model = PPO(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048 // n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Exploration coefficient
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'SAC':
        # SAC with larger buffer (episodes are cheap now)
        policy_kwargs = dict(
            net_arch=dict(
                pi=[256, 256, 256],
                qf=[256, 256, 256]
            )
        )
        model = SAC(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=50_000,  # Larger buffer since episodes are simple
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',  # Auto-tune entropy
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'A2C':
        # A2C for faster training
        policy_kwargs = dict(net_arch=[256, 256, 256])
        model = A2C(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=7e-4,
            n_steps=5,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_dir,
        name_prefix=f"{algorithm.lower()}_robust_pid"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2_000 // n_envs,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    # Train
    print("\nStarting training...")
    print(f"Monitor training with: tensorboard --logdir {log_dir}\n")
    print(f"Note: Each timestep = 1 PID trial tested on {n_test_scenarios} scenarios\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_path = os.path.join(model_dir, f"{algorithm.lower()}_robust_pid_final.zip")
    model.save(final_path)
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*70}\n")

    # Test the learned model
    print("Testing learned Robust PID agent...")
    test_learned_robust_pid(model, algorithm, maneuver, n_test_scenarios, window_size, max_steps,
                             missile_speed, missile_accel, target_speed)

    return model


def test_learned_robust_pid(model, algorithm, maneuver='circular',
                             n_test_scenarios=5, window_size=10, max_steps=1300,
                             missile_speed=1000.0, missile_accel=1000.0,
                             target_speed=1000.0, n_trials=20):
    """
    Test the learned Robust PID model

    Args:
        model: Trained RL model
        algorithm: Algorithm name
        maneuver: Target maneuver type
        n_test_scenarios: Number of scenarios to test each PID on
        window_size: History window size
        max_steps: Maximum simulation steps per scenario
        n_trials: Number of trials to run
    """
    print(f"\n{'='*70}")
    print(f"TESTING ROBUST PID AGENT")
    print(f"{'='*70}\n")

    env = RobustPIDEnv(
        n_test_scenarios=n_test_scenarios,
        window_size=window_size,
        max_steps=max_steps,
        target_maneuver=maneuver,
        missile_speed=missile_speed,
        missile_accel=missile_accel,
        target_speed=target_speed
    )

    all_results = []

    for trial in range(n_trials):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        all_results.append(info)

        print(f"Trial {trial+1:2d}: "
              f"PID=(Kp={info['pid_kp']:.0f}, Ki={info['pid_ki']:.1f}, Kd={info['pid_kd']:.1f}) | "
              f"Hit Rate={info['avg_hit_rate']*100:.1f}% | "
              f"Avg Reward={info['avg_reward']:7.1f} | "
              f"Avg Distance={info['avg_final_distance']:7.1f}m")

    # Calculate statistics
    avg_hit_rate = np.mean([r['avg_hit_rate'] for r in all_results]) * 100
    avg_reward = np.mean([r['avg_reward'] for r in all_results])
    avg_distance = np.mean([r['avg_final_distance'] for r in all_results])

    # PID statistics
    kp_values = [r['pid_kp'] for r in all_results]
    ki_values = [r['pid_ki'] for r in all_results]
    kd_values = [r['pid_kd'] for r in all_results]

    avg_kp = np.mean(kp_values)
    avg_ki = np.mean(ki_values)
    avg_kd = np.mean(kd_values)
    std_kp = np.std(kp_values)
    std_ki = np.std(ki_values)
    std_kd = np.std(kd_values)

    print(f"\n{'='*70}")
    print(f"TEST RESULTS ({n_trials} trials × {n_test_scenarios} scenarios)")
    print(f"{'='*70}")
    print(f"Average Hit Rate: {avg_hit_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"Average Final Distance: {avg_distance:.1f}m")
    print(f"\nLearned PID Parameters:")
    print(f"  Kp = {avg_kp:.1f} ± {std_kp:.1f}")
    print(f"  Ki = {avg_ki:.2f} ± {std_ki:.2f}")
    print(f"  Kd = {avg_kd:.2f} ± {std_kd:.2f}")
    print(f"\nNote: Agent should converge to similar PID values (low std)")
    print(f"{'='*70}\n")

    return avg_kp, avg_ki, avg_kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Robust Non-Adaptive PID with RL')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'A2C'],
                       help='RL algorithm (PPO recommended for this task)')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--n_envs', type=int, default=8,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=100_000,
                       help='Total training timesteps (PID trials)')
    parser.add_argument('--save_freq', type=int, default=10_000,
                       help='Checkpoint save frequency')
    parser.add_argument('--n_test_scenarios', type=int, default=5,
                       help='Number of scenarios to test each PID on')
    parser.add_argument('--window_size', type=int, default=10,
                       help='History window size')
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
        algorithm=args.algorithm,
        maneuver=args.maneuver,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        save_freq=args.save_freq,
        n_test_scenarios=args.n_test_scenarios,
        window_size=args.window_size,
        max_steps=args.max_steps,
        missile_speed=args.missile_speed,
        missile_accel=args.missile_accel,
        target_speed=args.target_speed
    )
