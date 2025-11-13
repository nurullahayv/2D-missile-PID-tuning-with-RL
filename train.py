"""
Train RL Agent for PID Tuning
"""
import argparse
import os
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env

from src.environment import MissilePIDEnv


def make_env(rank, maneuver='circular'):
    """Create environment instance"""
    def _init():
        env = MissilePIDEnv(
            map_size=10000.0,
            hit_radius=50.0,
            max_steps=500,
            dt=0.01,
            target_maneuver=maneuver
        )
        return env
    return _init


def train(
    algorithm='PPO',
    maneuver='circular',
    n_envs=4,
    total_timesteps=1_000_000,
    save_freq=50_000,
    output_dir='models'
):
    """Train RL agent"""

    print(f"Starting training: {algorithm} on {maneuver} target")
    print(f"Environments: {n_envs}")
    print(f"Total timesteps: {total_timesteps:,}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{algorithm}_{maneuver}_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    # Create vectorized environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, maneuver) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, maneuver)])

    # Create eval environment
    eval_env = DummyVecEnv([make_env(0, maneuver)])

    # Initialize algorithm
    if algorithm == 'PPO':
        model = PPO(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log=os.path.join(run_dir, 'tensorboard')
        )
    elif algorithm == 'SAC':
        model = SAC(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef='auto',
            verbose=1,
            tensorboard_log=os.path.join(run_dir, 'tensorboard')
        )
    elif algorithm == 'TD3':
        model = TD3(
            'MlpPolicy',
            env,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            policy_delay=2,
            verbose=1,
            tensorboard_log=os.path.join(run_dir, 'tensorboard')
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,
        save_path=os.path.join(run_dir, 'checkpoints'),
        name_prefix='model'
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, 'best_model'),
        log_path=os.path.join(run_dir, 'eval_logs'),
        eval_freq=10_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    # Train
    print(f"\nTraining started... Output: {run_dir}")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    final_path = os.path.join(run_dir, 'final_model')
    model.save(final_path)
    print(f"\nTraining complete! Model saved to: {final_path}")

    # Cleanup
    env.close()
    eval_env.close()

    return final_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RL agent for PID tuning')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'SAC', 'TD3'],
                       help='RL algorithm')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--n_envs', type=int, default=4,
                       help='Number of parallel environments')
    parser.add_argument('--timesteps', type=int, default=1_000_000,
                       help='Total training timesteps')
    parser.add_argument('--output_dir', type=str, default='models',
                       help='Output directory')

    args = parser.parse_args()

    train(
        algorithm=args.algorithm,
        maneuver=args.maneuver,
        n_envs=args.n_envs,
        total_timesteps=args.timesteps,
        output_dir=args.output_dir
    )
