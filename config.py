"""
Configuration for Missile PID Tuning with RL
"""
import argparse
import os
from typing import Dict


class Config:
    """Configuration class for training and evaluation"""

    def __init__(self, mode: str = "train"):
        """
        Initialize configuration

        Args:
            mode: "train" or "eval"
        """
        self.mode = mode
        parser = argparse.ArgumentParser(description='Missile PID RL Training Config')

        # Environment parameters
        parser.add_argument('--max_steps', type=int, default=500,
                          help='Maximum steps per episode')
        parser.add_argument('--dt', type=float, default=0.1,
                          help='Time step in seconds')
        parser.add_argument('--map_size', type=float, default=10000.0,
                          help='Map size in meters')
        parser.add_argument('--hit_radius', type=float, default=50.0,
                          help='Hit radius in meters')
        parser.add_argument('--target_maneuver', type=str, default='straight',
                          choices=['straight', 'circular', 'zigzag', 'evasive'],
                          help='Target maneuver type')

        # Training parameters
        parser.add_argument('--algorithm', type=str, default='PPO',
                          choices=['PPO', 'SAC', 'TD3'],
                          help='RL algorithm to use')
        parser.add_argument('--total_timesteps', type=int, default=1_000_000,
                          help='Total training timesteps')
        parser.add_argument('--learning_rate', type=float, default=3e-4,
                          help='Learning rate')
        parser.add_argument('--batch_size', type=int, default=64,
                          help='Batch size for training')
        parser.add_argument('--n_steps', type=int, default=2048,
                          help='Number of steps per update (PPO)')
        parser.add_argument('--gamma', type=float, default=0.99,
                          help='Discount factor')

        # Model parameters
        parser.add_argument('--hidden_size', type=int, default=256,
                          help='Hidden layer size')
        parser.add_argument('--n_layers', type=int, default=2,
                          help='Number of hidden layers')

        # Logging and saving
        parser.add_argument('--log_dir', type=str, default='./logs',
                          help='Directory for logs')
        parser.add_argument('--save_dir', type=str, default='./models',
                          help='Directory for saving models')
        parser.add_argument('--save_freq', type=int, default=50000,
                          help='Save model every N steps')
        parser.add_argument('--eval_freq', type=int, default=10000,
                          help='Evaluate every N steps')
        parser.add_argument('--n_eval_episodes', type=int, default=10,
                          help='Number of episodes for evaluation')

        # Device
        parser.add_argument('--device', type=str, default='auto',
                          choices=['auto', 'cuda', 'cpu'],
                          help='Device to use for training')

        # Experiment
        parser.add_argument('--exp_name', type=str, default=None,
                          help='Experiment name')
        parser.add_argument('--seed', type=int, default=42,
                          help='Random seed')

        # Evaluation
        parser.add_argument('--model_path', type=str, default=None,
                          help='Path to trained model for evaluation')
        parser.add_argument('--render', action='store_true',
                          help='Render during evaluation')
        parser.add_argument('--save_video', action='store_true',
                          help='Save video during evaluation')

        self.args = parser.parse_args()
        self._process_args()

    def _process_args(self):
        """Process and validate arguments"""
        # Create directories
        os.makedirs(self.args.log_dir, exist_ok=True)
        os.makedirs(self.args.save_dir, exist_ok=True)

        # Set experiment name
        if self.args.exp_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.args.exp_name = f"missile_pid_{self.args.algorithm}_{self.args.target_maneuver}_{timestamp}"

        # Update paths with experiment name
        self.args.log_dir = os.path.join(self.args.log_dir, self.args.exp_name)
        self.args.save_dir = os.path.join(self.args.save_dir, self.args.exp_name)

        os.makedirs(self.args.log_dir, exist_ok=True)
        os.makedirs(self.args.save_dir, exist_ok=True)

    def get_env_config(self) -> Dict:
        """Get environment configuration"""
        return {
            'max_steps': self.args.max_steps,
            'dt': self.args.dt,
            'map_size': self.args.map_size,
            'hit_radius': self.args.hit_radius,
            'target_maneuver': self.args.target_maneuver,
        }

    @property
    def get_arguments(self):
        """Get all arguments"""
        return self.args


def get_default_config(target_maneuver: str = "straight") -> Dict:
    """Get default configuration without argparse"""
    return {
        # Environment
        'max_steps': 500,
        'dt': 0.1,
        'map_size': 10000.0,
        'hit_radius': 50.0,
        'target_maneuver': target_maneuver,

        # Training
        'algorithm': 'PPO',
        'total_timesteps': 1_000_000,
        'learning_rate': 3e-4,
        'batch_size': 64,
        'n_steps': 2048,
        'gamma': 0.99,

        # Model
        'hidden_size': 256,
        'n_layers': 2,

        # Other
        'seed': 42,
        'device': 'auto',
    }
