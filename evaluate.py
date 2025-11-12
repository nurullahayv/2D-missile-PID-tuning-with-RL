"""
Evaluation script for trained Missile PID RL models
"""
import os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import List, Tuple
import argparse

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from envs.missile_pid_env import MissilePIDEnv
from config import get_default_config
from warsim.visualization.pygame_renderer import PygameRenderer


def evaluate_model(model_path: str,
                   n_episodes: int = 10,
                   target_maneuver: str = "straight",
                   render: bool = False,
                   save_plots: bool = True,
                   output_dir: str = "./evaluation_results"):
    """
    Evaluate a trained model

    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        target_maneuver: Target maneuver type
        render: Whether to render during evaluation
        save_plots: Whether to save visualization plots
        output_dir: Directory to save results
    """
    print("=" * 60)
    print("Evaluating Missile PID RL Model")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Target Maneuver: {target_maneuver}")
    print("=" * 60)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    try:
        model = PPO.load(model_path)
        print("Loaded PPO model")
    except:
        try:
            model = SAC.load(model_path)
            print("Loaded SAC model")
        except:
            model = TD3.load(model_path)
            print("Loaded TD3 model")

    # Create environment
    config = get_default_config(target_maneuver)
    env = MissilePIDEnv(
        max_steps=config['max_steps'],
        dt=config['dt'],
        map_size=config['map_size'],
        hit_radius=config['hit_radius'],
        target_maneuver=target_maneuver
    )

    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    hit_success = []
    final_distances = []
    pid_trajectories = []

    print("\nRunning evaluation...")

    # Initialize renderer if rendering is enabled
    renderer = None
    if render:
        renderer = PygameRenderer(map_size=config['map_size'],
                                  window_size=(1200, 1000),
                                  fps=60)
        print("Live rendering enabled - Pygame real-time visualization")
        print("Press ESC or Q to skip to next episode\n")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Track PID gains over time
        pid_history = {'kp': [], 'ki': [], 'kd': []}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            # Record PID gains
            pid_history['kp'].append(info['pid_gains']['kp'])
            pid_history['ki'].append(info['pid_gains']['ki'])
            pid_history['kd'].append(info['pid_gains']['kd'])

            # Render with pygame real-time visualization
            if render and renderer:
                # Calculate distance
                missile_pos = info['missile_position']
                target_pos = info['target_position']
                distance = np.sqrt((target_pos[0] - missile_pos[0])**2 +
                                 (target_pos[1] - missile_pos[1])**2)

                success = renderer.render_frame(
                    missile_trajectory=env.missile.trajectory,
                    target_trajectory=env.target.trajectory,
                    missile_heading=env.missile.heading,
                    target_heading=env.target.heading,
                    hit_radius=config['hit_radius'],
                    step=episode_length,
                    distance=distance,
                    pid_gains=info['pid_gains'],
                    fuel=info['fuel'],
                    mode="RL Adaptive PID",
                    title=f"RL Evaluation - Episode {episode + 1}/{n_episodes} - {target_maneuver.capitalize()} Target"
                )

                # If user closed window or pressed ESC, skip rendering for rest
                if not success:
                    renderer = None

        # Record metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        hit_success.append(info.get('hit', False))
        final_distances.append(info.get('distance', float('inf')))
        pid_trajectories.append(pid_history)

        print(f"Episode {episode + 1}/{n_episodes}: "
              f"Reward={episode_reward:.2f}, "
              f"Length={episode_length}, "
              f"Hit={'Yes' if info.get('hit', False) else 'No'}, "
              f"Final Distance={info.get('distance', 0):.2f}m")

        # Save trajectory plot for first few episodes
        if save_plots and episode < 3:
            plot_trajectory(env, episode, output_dir)

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Hit Success Rate: {np.mean(hit_success) * 100:.1f}%")
    print(f"Average Final Distance: {np.mean(final_distances):.2f}m")
    print("=" * 60)

    # Save plots
    if save_plots:
        plot_evaluation_summary(episode_rewards, episode_lengths,
                               hit_success, final_distances,
                               pid_trajectories, output_dir)

    # Cleanup renderer
    if renderer:
        renderer.close()

    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'hit_success': hit_success,
        'final_distances': final_distances,
        'pid_trajectories': pid_trajectories
    }


def plot_trajectory(env: MissilePIDEnv, episode: int, output_dir: str):
    """Plot missile and target trajectories (static plot for summary)"""
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#0a0e27')
    ax.set_facecolor('#0a0e27')

    # Plot missile trajectory
    missile_traj = np.array(env.missile.trajectory)
    ax.plot(missile_traj[:, 0], missile_traj[:, 1],
            'b-', linewidth=2, label='Missile', alpha=0.7)
    ax.plot(missile_traj[0, 0], missile_traj[0, 1],
            'bo', markersize=10, label='Missile Start')
    ax.plot(missile_traj[-1, 0], missile_traj[-1, 1],
            'bs', markersize=10, label='Missile End')

    # Plot target trajectory
    target_traj = np.array(env.target.trajectory)
    ax.plot(target_traj[:, 0], target_traj[:, 1],
            'r-', linewidth=2, label='Target', alpha=0.7)
    ax.plot(target_traj[0, 0], target_traj[0, 1],
            'ro', markersize=10, label='Target Start')
    ax.plot(target_traj[-1, 0], target_traj[-1, 1],
            'rs', markersize=10, label='Target End')

    # Plot hit radius
    circle = plt.Circle((target_traj[-1, 0], target_traj[-1, 1]),
                       env.hit_radius, color='r', fill=False,
                       linestyle='--', label='Hit Radius')
    ax.add_patch(circle)

    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title(f'Episode {episode + 1} - Missile vs Target Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis('equal')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'trajectory_episode_{episode + 1}.png'), dpi=150)
    plt.close()


def plot_evaluation_summary(rewards: List[float],
                            lengths: List[int],
                            hit_success: List[bool],
                            final_distances: List[float],
                            pid_trajectories: List[dict],
                            output_dir: str):
    """Plot evaluation summary statistics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Rewards
    axes[0, 0].plot(rewards, 'o-')
    axes[0, 0].axhline(np.mean(rewards), color='r', linestyle='--', label='Mean')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Episode lengths
    axes[0, 1].plot(lengths, 'o-')
    axes[0, 1].axhline(np.mean(lengths), color='r', linestyle='--', label='Mean')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Steps')
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Hit success rate
    success_rate = np.cumsum(hit_success) / np.arange(1, len(hit_success) + 1)
    axes[0, 2].plot(success_rate, 'o-')
    axes[0, 2].axhline(np.mean(hit_success), color='r', linestyle='--', label='Final Rate')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Success Rate')
    axes[0, 2].set_title('Cumulative Hit Success Rate')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.1])

    # Final distances
    axes[1, 0].plot(final_distances, 'o-')
    axes[1, 0].axhline(np.mean(final_distances), color='r', linestyle='--', label='Mean')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Distance (m)')
    axes[1, 0].set_title('Final Distance to Target')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # PID gains evolution (average across episodes)
    # Handle different episode lengths by interpolating to common length
    max_len = max(len(traj['kp']) for traj in pid_trajectories)

    def interpolate_trajectory(traj_list, target_len):
        """Interpolate trajectory to target length"""
        if len(traj_list) == target_len:
            return np.array(traj_list)
        x_old = np.linspace(0, 1, len(traj_list))
        x_new = np.linspace(0, 1, target_len)
        return np.interp(x_new, x_old, traj_list)

    kp_interp = [interpolate_trajectory(traj['kp'], max_len) for traj in pid_trajectories]
    ki_interp = [interpolate_trajectory(traj['ki'], max_len) for traj in pid_trajectories]
    kd_interp = [interpolate_trajectory(traj['kd'], max_len) for traj in pid_trajectories]

    avg_kp = np.mean(kp_interp, axis=0)
    avg_ki = np.mean(ki_interp, axis=0)
    avg_kd = np.mean(kd_interp, axis=0)

    axes[1, 1].plot(avg_kp, label='Kp', linewidth=2)
    axes[1, 1].plot(avg_ki, label='Ki', linewidth=2)
    axes[1, 1].plot(avg_kd, label='Kd', linewidth=2)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Gain Value')
    axes[1, 1].set_title('Average PID Gains Evolution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Distribution of final PID gains
    final_kp = [traj['kp'][-1] for traj in pid_trajectories]
    final_ki = [traj['ki'][-1] for traj in pid_trajectories]
    final_kd = [traj['kd'][-1] for traj in pid_trajectories]

    x = np.arange(3)
    means = [np.mean(final_kp), np.mean(final_ki), np.mean(final_kd)]
    stds = [np.std(final_kp), np.std(final_ki), np.std(final_kd)]

    axes[1, 2].bar(x, means, yerr=stds, capsize=5, alpha=0.7)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(['Kp', 'Ki', 'Kd'])
    axes[1, 2].set_ylabel('Gain Value')
    axes[1, 2].set_title('Final PID Gains Distribution')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_summary.png'), dpi=150)
    plt.close()

    print(f"\nPlots saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate Missile PID RL Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of episodes to evaluate')
    parser.add_argument('--target_maneuver', type=str, default='straight',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')
    parser.add_argument('--render', action='store_true',
                       help='Render during evaluation (shows live visualization)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save results')

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model_path,
        n_episodes=args.n_episodes,
        target_maneuver=args.target_maneuver,
        render=args.render,
        save_plots=True,
        output_dir=args.output_dir
    )
