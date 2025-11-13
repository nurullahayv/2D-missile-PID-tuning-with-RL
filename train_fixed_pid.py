"""
Training script for non-adaptive (fixed) PID parameter optimization with RL

This script trains an RL agent to find optimal FIXED PID parameters.
The agent selects PID values once at episode start, then they remain constant.

Usage:
    python train_fixed_pid.py --algorithm SAC --maneuver circular --timesteps 500000
"""
import argparse
import os
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.fixed_pid_env import FixedPIDEnv


def make_env(rank, maneuver='circular', missile_speed=1000.0,
             missile_accel=1000.0, target_speed=1000.0):
    """Create a monitored environment"""
    def _init():
        env = FixedPIDEnv(
            target_maneuver=maneuver,
            missile_speed=missile_speed,
            missile_accel=missile_accel,
            target_speed=target_speed
        )
        env = Monitor(env)
        return env
    return _init


def train(algorithm='SAC', maneuver='circular', n_envs=4,
          total_timesteps=500_000, save_freq=50_000,
          missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """
    Train RL agent to find optimal fixed PID parameters

    Args:
        algorithm: RL algorithm (PPO, SAC, TD3)
        maneuver: Target maneuver type (straight, circular, zigzag, evasive)
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        save_freq: Checkpoint save frequency
        missile_speed: Missile max speed (m/s)
        missile_accel: Missile max acceleration (m/s²)
        target_speed: Target speed (m/s)
    """
    print(f"\n{'='*60}")
    print(f"TRAINING FIXED PID PARAMETERS WITH {algorithm}")
    print(f"{'='*60}")
    print(f"Target Maneuver: {maneuver}")
    print(f"Missile: {missile_speed} m/s, {missile_accel} m/s²")
    print(f"Target: {target_speed} m/s")
    print(f"Environments: {n_envs}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"{'='*60}\n")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/fixed_pid/{algorithm.lower()}_{maneuver}_{timestamp}"
    log_dir = f"logs/fixed_pid/{algorithm.lower()}_{maneuver}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Model directory: {model_dir}")
    print(f"Log directory: {log_dir}\n")

    # Create vectorized environments
    print("Creating environments...")
    if n_envs > 1:
        env = DummyVecEnv([make_env(i, maneuver, missile_speed,
                                     missile_accel, target_speed)
                          for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, maneuver, missile_speed,
                                     missile_accel, target_speed)])

    # Create eval environment
    eval_env = DummyVecEnv([make_env(0, maneuver, missile_speed,
                                      missile_accel, target_speed)])

    # Initialize model
    print(f"Initializing {algorithm} model...")

    # Deep network configuration
    policy_kwargs = dict(
        net_arch=dict(
            pi=[256, 256, 256],  # Actor: 3 layers
            qf=[256, 256, 256]   # Critic: 3 layers (for SAC/TD3)
        )
    )

    if algorithm == 'PPO':
        policy_kwargs['net_arch'] = [256, 256, 256]  # PPO uses simpler format
        model = PPO(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=(1, "episode"),
            gradient_steps=-1,
            verbose=1,
            tensorboard_log=log_dir
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_dir,
        name_prefix=f"{algorithm.lower()}_fixed_pid"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=10_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Train
    print("\nStarting training...")
    print(f"Monitor training with: tensorboard --logdir {log_dir}\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_path = os.path.join(model_dir, f"{algorithm.lower()}_fixed_pid_final.zip")
    model.save(final_path)
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*60}\n")

    # Test the learned PID parameters
    print("Testing learned PID parameters...")
    test_learned_pid(model, maneuver, missile_speed, missile_accel, target_speed)

    return model


def test_learned_pid(model, maneuver='circular', missile_speed=1000.0,
                     missile_accel=1000.0, target_speed=1000.0, n_episodes=10):
    """
    Test the learned model and extract optimal PID parameters

    Args:
        model: Trained RL model
        maneuver: Target maneuver type
        n_episodes: Number of test episodes
    """
    print(f"\n{'='*60}")
    print(f"TESTING LEARNED PID PARAMETERS")
    print(f"{'='*60}\n")

    env = FixedPIDEnv(
        target_maneuver=maneuver,
        missile_speed=missile_speed,
        missile_accel=missile_accel,
        target_speed=target_speed
    )

    hits = 0
    total_reward = 0
    pid_values = []

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = env.step(action)
            episode_reward += reward

        total_reward += episode_reward
        if info['hit']:
            hits += 1

        # Store PID parameters (they're fixed for the episode)
        pid_values.append([info['pid_kp'], info['pid_ki'], info['pid_kd']])

        print(f"Episode {episode+1}: Reward={episode_reward:.2f}, Hit={info['hit']}, "
              f"Distance={info['distance']:.1f}m, "
              f"PID=(Kp={info['pid_kp']:.3f}, Ki={info['pid_ki']:.3f}, Kd={info['pid_kd']:.3f})")

    # Calculate statistics
    hit_rate = hits / n_episodes * 100
    avg_reward = total_reward / n_episodes

    # Average PID values
    pid_values = np.array(pid_values)
    avg_kp = np.mean(pid_values[:, 0])
    avg_ki = np.mean(pid_values[:, 1])
    avg_kd = np.mean(pid_values[:, 2])
    std_kp = np.std(pid_values[:, 0])
    std_ki = np.std(pid_values[:, 1])
    std_kd = np.std(pid_values[:, 2])

    print(f"\n{'='*60}")
    print(f"TEST RESULTS ({n_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Hit Rate: {hit_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"\nOptimal PID Parameters for '{maneuver}' target:")
    print(f"  Kp = {avg_kp:.3f} ± {std_kp:.3f}")
    print(f"  Ki = {avg_ki:.3f} ± {std_ki:.3f}")
    print(f"  Kd = {avg_kd:.3f} ± {std_kd:.3f}")
    print(f"{'='*60}\n")

    print(f"💡 Use these values in demo.py:")
    print(f"   python demo.py --maneuver {maneuver} "
          f"--kp {avg_kp:.3f} --ki {avg_ki:.3f} --kd {avg_kd:.3f}\n")

    return avg_kp, avg_ki, avg_kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train fixed PID parameters with RL')
    parser.add_argument('--algorithm', type=str, default='SAC',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='RL algorithm')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=500_000,
                       help='Total training timesteps')
    parser.add_argument('--save_freq', type=int, default=50_000,
                       help='Checkpoint save frequency')
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
        missile_speed=args.missile_speed,
        missile_accel=args.missile_accel,
        target_speed=args.target_speed
    )
