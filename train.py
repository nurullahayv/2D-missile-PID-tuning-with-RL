"""
Training script for Missile PID Tuning with RL
"""
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config
from envs.missile_pid_env import MissilePIDEnv


def train():
    """Main training function"""
    # Load configuration
    config = Config(mode="train")
    args = config.get_arguments

    print("=" * 60)
    print("Missile PID Tuning with Reinforcement Learning")
    print("=" * 60)
    print(f"Algorithm: {args.algorithm}")
    print(f"Target Maneuver: {args.target_maneuver}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Device: {args.device}")
    print(f"Experiment: {args.exp_name}")
    print("=" * 60)

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Create environment
    env_kwargs = config.get_env_config()

    # Training environment (vectorized for better performance)
    train_env = make_vec_env(
        lambda: MissilePIDEnv(**env_kwargs),
        n_envs=4,  # 4 parallel environments
        seed=args.seed
    )

    # Evaluation environment
    eval_env = MissilePIDEnv(**env_kwargs)
    eval_env = Monitor(eval_env)

    # Create model
    policy_kwargs = dict(
        net_arch=[args.hidden_size] * args.n_layers
    )

    if args.algorithm == "PPO":
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            gamma=args.gamma,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
            seed=args.seed
        )
    elif args.algorithm == "SAC":
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
            seed=args.seed
        )
    elif args.algorithm == "TD3":
        model = TD3(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            gamma=args.gamma,
            policy_kwargs=policy_kwargs,
            verbose=1,
            tensorboard_log=args.log_dir,
            device=args.device,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}")

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=args.save_dir,
        name_prefix="model",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.save_dir,
        log_path=args.log_dir,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False
    )

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )

    # Save final model
    final_model_path = os.path.join(args.save_dir, "final_model")
    model.save(final_model_path)
    print(f"\nFinal model saved to: {final_model_path}")

    print("\nTraining completed!")
    print(f"Logs saved to: {args.log_dir}")
    print(f"Models saved to: {args.save_dir}")
    print("\nTo view training progress, run:")
    print(f"  tensorboard --logdir {args.log_dir}")


if __name__ == "__main__":
    train()
