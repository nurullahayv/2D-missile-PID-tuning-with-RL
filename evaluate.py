"""
Evaluate Trained RL Agent
"""
import argparse
import numpy as np
from stable_baselines3 import PPO, SAC, TD3

from src.environment import MissilePIDEnv


def evaluate(model_path, n_episodes=10, render=False, maneuver='circular'):
    """Evaluate trained model"""

    # Load model
    if 'SAC' in model_path:
        model = SAC.load(model_path)
    elif 'TD3' in model_path:
        model = TD3.load(model_path)
    else:
        model = PPO.load(model_path)

    # Create environment
    env = MissilePIDEnv(
        map_size=10000.0,
        hit_radius=50.0,
        max_steps=500,
        dt=0.01,
        target_maneuver=maneuver,
        render_mode='human' if render else None
    )

    # Statistics
    hits = 0
    total_rewards = []
    distances = []
    steps_list = []

    print(f"\nEvaluating model: {model_path}")
    print(f"Target maneuver: {maneuver}")
    print(f"Episodes: {n_episodes}")
    print("-" * 60)

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            done = terminated or truncated

            if render:
                cont = env.render()
                if not cont:
                    break

        # Record stats
        total_rewards.append(episode_reward)
        distances.append(info['distance'])
        steps_list.append(step)

        if info['hit']:
            hits += 1
            status = "✓ HIT"
        else:
            status = "✗ MISS"

        print(f"Episode {episode+1:2d}: {status} | "
              f"Distance: {info['distance']:6.1f}m | "
              f"Reward: {episode_reward:7.1f} | "
              f"Steps: {step:3d} | "
              f"PID: Kp={info['pid_gains']['kp']:.2f} "
              f"Ki={info['pid_gains']['ki']:.2f} "
              f"Kd={info['pid_gains']['kd']:.2f}")

    # Summary
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Hit rate: {hits}/{n_episodes} ({100*hits/n_episodes:.1f}%)")
    print(f"  Avg reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"  Avg distance: {np.mean(distances):.1f}m ± {np.std(distances):.1f}m")
    print(f"  Avg steps: {np.mean(steps_list):.1f} ± {np.std(steps_list):.1f}")
    print(f"  Min distance: {np.min(distances):.1f}m")

    env.close()

    return {
        'hit_rate': hits / n_episodes,
        'avg_reward': np.mean(total_rewards),
        'avg_distance': np.mean(distances),
        'avg_steps': np.mean(steps_list)
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument('model_path', type=str,
                       help='Path to trained model')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--render', action='store_true',
                       help='Render episodes')
    parser.add_argument('--maneuver', type=str, default='circular',
                       choices=['straight', 'circular', 'zigzag', 'evasive'],
                       help='Target maneuver type')

    args = parser.parse_args()

    evaluate(
        args.model_path,
        n_episodes=args.n_episodes,
        render=args.render,
        maneuver=args.maneuver
    )
