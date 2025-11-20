"""
Training script for Meta-Episodic PID optimization with RL

This script trains an RL agent using Meta-Episodic MDP formulation.
The agent learns to sequentially adapt PID parameters across multiple episodes.

Key Features:
- Each meta-episode = 10 consecutive episodes
- Each step = one full simulation with fixed PID
- Agent observes context + history, learns to adapt
- True MDP with sequential decision making

Usage:
    python train_meta_pid.py --algorithm RecurrentPPO --maneuver circular --timesteps 50000
    python train_meta_pid.py --algorithm PPO --maneuver circular --timesteps 50000 --n_envs 8
"""
import argparse
import os
from datetime import datetime
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from src.meta_episodic_pid_env import MetaEpisodicPIDEnv


def make_env(rank, maneuver='circular', episodes_per_meta=10, window_size=5,
             missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """Create a monitored Meta-Episodic environment"""
    def _init():
        env = MetaEpisodicPIDEnv(
            episodes_per_meta=episodes_per_meta,
            window_size=window_size,
            target_maneuver=maneuver,
            missile_speed=missile_speed,
            missile_accel=missile_accel,
            target_speed=target_speed
        )
        env = Monitor(env)
        return env
    return _init


def train(algorithm='RecurrentPPO', maneuver='circular', n_envs=4,
          total_timesteps=50_000, save_freq=5_000,
          episodes_per_meta=10, window_size=5,
          missile_speed=1000.0, missile_accel=1000.0, target_speed=1000.0):
    """
    Train RL agent with Meta-Episodic MDP formulation

    Args:
        algorithm: RL algorithm (RecurrentPPO, SAC, PPO)
        maneuver: Target maneuver type (straight, circular, zigzag, evasive)
        n_envs: Number of parallel environments
        total_timesteps: Total training timesteps
        save_freq: Checkpoint save frequency
        episodes_per_meta: Number of episodes per meta-episode (default: 10)
        window_size: History window size (default: 5)
        missile_speed: Missile max speed (m/s)
        missile_accel: Missile max acceleration (m/s²)
        target_speed: Target speed (m/s)
    """
    print(f"\n{'='*70}")
    print(f"META-EPISODIC PID TRAINING WITH {algorithm}")
    print(f"{'='*70}")
    print(f"Target Maneuver: {maneuver}")
    print(f"Missile: {missile_speed} m/s, {missile_accel} m/s²")
    print(f"Target: {target_speed} m/s")
    print(f"Episodes per Meta-Episode: {episodes_per_meta}")
    print(f"History Window Size: {window_size}")
    print(f"Parallel Environments: {n_envs}")
    print(f"Total Timesteps: {total_timesteps:,}")
    print(f"  → Meta-Episodes: {total_timesteps // episodes_per_meta:,}")
    print(f"  → Total Episodes: {total_timesteps:,}")
    print(f"{'='*70}\n")

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = f"models/meta_pid/{algorithm.lower()}_{maneuver}_{timestamp}"
    log_dir = f"logs/meta_pid/{algorithm.lower()}_{maneuver}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"Model directory: {model_dir}")
    print(f"Log directory: {log_dir}\n")

    # Create vectorized environments
    print("Creating environments...")
    print("Using SubprocVecEnv for parallel CPU execution (works with Numba JIT)")
    if n_envs > 1:
        env = SubprocVecEnv([
            make_env(i, maneuver, episodes_per_meta, window_size,
                    missile_speed, missile_accel, target_speed)
            for i in range(n_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, maneuver, episodes_per_meta, window_size,
                    missile_speed, missile_accel, target_speed)
        ])

    # Create eval environment
    eval_env = DummyVecEnv([
        make_env(0, maneuver, episodes_per_meta, window_size,
                missile_speed, missile_accel, target_speed)
    ])

    # Initialize model
    print(f"Initializing {algorithm} model...")

    if algorithm == 'RecurrentPPO':
        # LSTM policy for sequential decision making
        model = RecurrentPPO(
            'MlpLstmPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048 // n_envs,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05,  # Higher entropy for exploration
            policy_kwargs={
                'lstm_hidden_size': 256,
                'n_lstm_layers': 1,
                'enable_critic_lstm': True,
                'net_arch': [256, 256]  # Additional MLP layers
            },
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'PPO':
        # Standard MLP policy (no LSTM)
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
            ent_coef=0.05,
            verbose=1,
            tensorboard_log=log_dir
        )
    elif algorithm == 'SAC':
        # SAC policy
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
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=model_dir,
        name_prefix=f"{algorithm.lower()}_meta_pid"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=2_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Train
    print("\nStarting training...")
    print(f"Monitor training with: tensorboard --logdir {log_dir}\n")
    print(f"Note: 1 timestep = 1 episode (but {episodes_per_meta} episodes = 1 meta-episode)\n")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")

    # Save final model
    final_path = os.path.join(model_dir, f"{algorithm.lower()}_meta_pid_final.zip")
    model.save(final_path)
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"Final model saved to: {final_path}")
    print(f"{'='*70}\n")

    # Test the learned model
    print("Testing learned Meta-PID agent...")
    test_learned_meta_pid(model, algorithm, maneuver, episodes_per_meta, window_size,
                          missile_speed, missile_accel, target_speed)

    return model


def test_learned_meta_pid(model, algorithm, maneuver='circular',
                          episodes_per_meta=10, window_size=5,
                          missile_speed=1000.0, missile_accel=1000.0,
                          target_speed=1000.0, n_meta_episodes=5):
    """
    Test the learned Meta-Episodic model

    Args:
        model: Trained RL model
        algorithm: Algorithm name
        maneuver: Target maneuver type
        episodes_per_meta: Episodes per meta-episode
        window_size: History window size
        n_meta_episodes: Number of meta-episodes to test
    """
    print(f"\n{'='*70}")
    print(f"TESTING META-EPISODIC PID AGENT")
    print(f"{'='*70}\n")

    env = MetaEpisodicPIDEnv(
        episodes_per_meta=episodes_per_meta,
        window_size=window_size,
        target_maneuver=maneuver,
        missile_speed=missile_speed,
        missile_accel=missile_accel,
        target_speed=target_speed
    )

    total_hits = 0
    total_episodes = 0
    all_pid_values = []
    all_rewards = []

    for meta_ep in range(n_meta_episodes):
        print(f"\n--- Meta-Episode {meta_ep + 1} ---")

        obs, _ = env.reset()
        lstm_states = None
        meta_reward = 0
        meta_hits = 0

        for episode_in_meta in range(episodes_per_meta):
            # Predict action
            if algorithm == 'RecurrentPPO':
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
            else:
                action, _ = model.predict(obs, deterministic=True)

            # Step
            obs, reward, done, _, info = env.step(action)

            meta_reward += reward
            if info['hit']:
                meta_hits += 1
                total_hits += 1

            total_episodes += 1
            all_pid_values.append([info['pid_kp'], info['pid_ki'], info['pid_kd']])
            all_rewards.append(reward)

            print(f"  Ep {episode_in_meta+1:2d}: {'HIT' if info['hit'] else 'MISS':4s} | "
                  f"Time={info['hit_time']:3d} | "
                  f"Dist={info['final_distance']:7.1f}m | "
                  f"Reward={reward:7.1f} | "
                  f"PID=(Kp={info['pid_kp']:.0f}, Ki={info['pid_ki']:.1f}, Kd={info['pid_kd']:.1f})")

            if done:
                break

        print(f"  Meta-Episode Reward: {meta_reward:.1f}")
        print(f"  Meta-Episode Hit Rate: {meta_hits}/{episodes_per_meta} = {meta_hits/episodes_per_meta*100:.1f}%")

    # Calculate statistics
    hit_rate = total_hits / total_episodes * 100
    avg_reward = np.mean(all_rewards)

    # Average PID values
    pid_values = np.array(all_pid_values)
    avg_kp = np.mean(pid_values[:, 0])
    avg_ki = np.mean(pid_values[:, 1])
    avg_kd = np.mean(pid_values[:, 2])
    std_kp = np.std(pid_values[:, 0])
    std_ki = np.std(pid_values[:, 1])
    std_kd = np.std(pid_values[:, 2])

    print(f"\n{'='*70}")
    print(f"TEST RESULTS ({n_meta_episodes} meta-episodes, {total_episodes} total episodes)")
    print(f"{'='*70}")
    print(f"Overall Hit Rate: {hit_rate:.1f}%")
    print(f"Average Reward: {avg_reward:.1f}")
    print(f"\nAverage PID Parameters (across all episodes):")
    print(f"  Kp = {avg_kp:.1f} ± {std_kp:.1f}")
    print(f"  Ki = {avg_ki:.2f} ± {std_ki:.2f}")
    print(f"  Kd = {avg_kd:.2f} ± {std_kd:.2f}")
    print(f"\nNote: Agent adapts PID based on history, so values vary across episodes")
    print(f"{'='*70}\n")

    return avg_kp, avg_ki, avg_kd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Meta-Episodic PID with RL')
    parser.add_argument('--algorithm', type=str, default='RecurrentPPO',
                       choices=['RecurrentPPO', 'PPO', 'SAC'],
                       help='RL algorithm (RecurrentPPO recommended for sequential tasks)')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=50_000,
                       help='Total training timesteps (episodes)')
    parser.add_argument('--save_freq', type=int, default=5_000,
                       help='Checkpoint save frequency')
    parser.add_argument('--episodes_per_meta', type=int, default=10,
                       help='Number of episodes per meta-episode')
    parser.add_argument('--window_size', type=int, default=5,
                       help='History window size')
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
        episodes_per_meta=args.episodes_per_meta,
        window_size=args.window_size,
        missile_speed=args.missile_speed,
        missile_accel=args.missile_accel,
        target_speed=args.target_speed
    )
