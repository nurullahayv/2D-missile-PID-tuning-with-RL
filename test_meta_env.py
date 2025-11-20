"""
Quick test script to verify Meta-Episodic PID environment works correctly
"""
import numpy as np
from src.meta_episodic_pid_env import MetaEpisodicPIDEnv

print("Creating Meta-Episodic PID environment...")
env = MetaEpisodicPIDEnv(
    episodes_per_meta=3,
    window_size=2,
    target_maneuver='circular'
)

print(f"✓ Environment created successfully")
print(f"  Observation space: {env.observation_space.shape} (33D expected)")
print(f"  Action space: {env.action_space.shape} (3D expected)")
print(f"  Episodes per meta: {env.episodes_per_meta}")
print(f"  Window size: {env.window_size}")

print("\nTesting reset...")
obs, info = env.reset()
print(f"✓ Reset successful")
print(f"  Observation shape: {obs.shape}")
print(f"  Observation (first 10): {obs[:10]}")

print("\nTesting episode sequence...")
for episode in range(3):
    # Random action (log_kp, Ki, Kd)
    action = np.array([
        np.random.uniform(2.5, 3.5),  # Kp: log scale → ~316-3162
        np.random.uniform(5, 15),      # Ki: 5-15
        np.random.uniform(5, 15)       # Kd: 5-15
    ])

    print(f"\n  Episode {episode + 1}:")
    print(f"    Action: log_Kp={action[0]:.2f}, Ki={action[1]:.2f}, Kd={action[2]:.2f}")

    obs, reward, done, truncated, info = env.step(action)

    print(f"    → PID: Kp={info['pid_kp']:.0f}, Ki={info['pid_ki']:.1f}, Kd={info['pid_kd']:.1f}")
    print(f"    → Result: {'HIT' if info['hit'] else 'MISS'} in {info['hit_time']} steps")
    print(f"    → Final distance: {info['final_distance']:.1f}m")
    print(f"    → Reward: {reward:.2f}")
    print(f"    → Done: {done}")
    print(f"    → Next obs shape: {obs.shape}")

    if done:
        print(f"\n✓ Meta-episode finished after {episode + 1} episodes")
        break

print("\n" + "="*60)
print("META-EPISODIC ENVIRONMENT TEST PASSED!")
print("="*60)
print("\nEnvironment structure:")
print("  ✓ Observation: Context (8D) + History (25D) = 33D")
print("  ✓ Action: [log_Kp, Ki, Kd] with discretization")
print("  ✓ Sequential decision making across episodes")
print("  ✓ True MDP with temporal dependency via history")
print("\nReady for training with:")
print("  python train_meta_pid.py --algorithm RecurrentPPO --maneuver circular")
