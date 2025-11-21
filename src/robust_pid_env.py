"""
Robust PID Environment for Non-Adaptive RL Tuning

Key Design Principles:
1. SIMPLE EPISODIC: 1 episode = 1 PID selection → Test on N scenarios → Done
2. ROBUST LEARNING: Same PID tested on multiple diverse scenarios
3. HISTORY-BASED: Observation = Previous PID trials + performance (no context)
4. CONTINUOUS REWARD: Distance-based shaping for better gradients
5. NARROWED ACTION SPACE: Focused on practical PID ranges

Goal: Find a single robust PID that works well across all scenarios
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .episodic_fixed_pid_env import run_simulation_jit, MANEUVER_MAP


class RobustPIDEnv(gym.Env):
    """
    Robust PID tuning environment using standard episodic RL.

    Episode Structure:
    - Agent selects one PID (action)
    - Environment tests this PID on N different scenarios
    - Returns average reward across scenarios
    - Episode terminates (standard episodic, no meta-structure)

    Observation (history-based):
    - [Kp, Ki, Kd, avg_reward, avg_hit_rate] × window_size
    - No context (missile/target positions) since we want non-adaptive PID

    Action:
    - [Kp_log, Ki, Kd] with narrowed ranges for faster convergence

    Reward:
    - Continuous distance-based reward (not just hit/miss)
    - Averaged across all test scenarios
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self,
                 n_test_scenarios=5,
                 window_size=10,
                 map_size=10000.0,
                 hit_radius=50.0,
                 max_steps=1300,
                 dt=0.01,
                 target_maneuver='circular',
                 missile_speed=1000.0,
                 missile_accel=1000.0,
                 target_speed=1000.0):
        """
        Args:
            n_test_scenarios: Number of scenarios to test each PID on (default: 5)
            window_size: Number of previous trials to keep in history (default: 10)
            map_size: Map size (meters)
            hit_radius: Hit detection radius (meters)
            max_steps: Max steps per simulation (default: 1300)
            dt: Time step (seconds)
            target_maneuver: Target maneuver type
            missile_speed: Missile max speed (m/s)
            missile_accel: Missile max acceleration (m/s²)
            target_speed: Target speed (m/s)
        """
        super().__init__()

        # Environment parameters
        self.n_test_scenarios = n_test_scenarios
        self.window_size = window_size
        self.map_size = map_size
        self.hit_radius = hit_radius
        self.max_steps = max_steps
        self.dt = dt
        self.target_maneuver = target_maneuver
        self.missile_speed = missile_speed
        self.missile_accel = missile_accel
        self.target_speed = target_speed

        # Encode maneuver type for JIT
        self.maneuver_type = MANEUVER_MAP.get(target_maneuver, 0)

        # Action space: [Kp_log, Ki, Kd]
        # Narrowed ranges based on typical successful values:
        # Kp: [500, 5000] in log scale [10^2.7, 10^3.7]
        # Ki: [0, 20] linear
        # Kd: [0, 20] linear
        self.action_space = spaces.Box(
            low=np.array([2.7, 0.0, 0.0], dtype=np.float32),   # log10(500) ≈ 2.7
            high=np.array([3.7, 20.0, 20.0], dtype=np.float32), # log10(5000) ≈ 3.7
            dtype=np.float32
        )

        # Observation space: History only (no context)
        # [Kp_norm, Ki_norm, Kd_norm, avg_reward_norm, avg_hit_rate] × window_size
        # Total: 5 × window_size
        self.observation_space = spaces.Box(
            low=-5.0,  # Allow some negative values for normalized rewards
            high=5.0,
            shape=(5 * window_size,),
            dtype=np.float32
        )

        # State
        self.history = []  # List of trial results
        self.test_scenarios = []  # Fixed scenarios for current episode

    def reset(self, seed=None, options=None):
        """Reset environment and sample new test scenarios"""
        super().reset(seed=seed)

        # Sample N diverse test scenarios (fixed for this episode)
        self.test_scenarios = [
            self._sample_scenario() for _ in range(self.n_test_scenarios)
        ]

        # Keep history across episodes (for learning)
        # Don't reset history - agent learns from past trials

        # Initial observation
        obs = self._get_observation()

        return obs, {}

    def step(self, action):
        """
        Execute one trial: Test selected PID on all scenarios

        Args:
            action: [Kp_log, Ki, Kd]

        Returns:
            obs: Next observation (updated history)
            reward: Average reward across all test scenarios
            done: True (standard episodic)
            truncated: False
            info: Trial statistics
        """
        # 1. Extract and discretize PID
        log_kp = float(action[0])
        Kp = 10 ** log_kp
        Ki = float(action[1])
        Kd = float(action[2])

        # Discretize for interpretability
        Kp = round(Kp / 100) * 100  # Nearest 100
        Ki = round(Ki, 1)             # Nearest 0.1
        Kd = round(Kd, 1)

        # Clamp to valid ranges
        Kp = np.clip(Kp, 500, 5000)
        Ki = np.clip(Ki, 0, 20)
        Kd = np.clip(Kd, 0, 20)

        # 2. Test PID on all scenarios
        results = []
        for scenario in self.test_scenarios:
            trajectory_array, hit, hit_time, actual_steps = run_simulation_jit(
                scenario['missile_x'],
                scenario['missile_y'],
                scenario['missile_vx'],
                scenario['missile_vy'],
                scenario['target_x'],
                scenario['target_y'],
                scenario['target_heading'],
                self.target_speed,
                Kp, Ki, Kd,
                self.missile_speed, self.missile_accel,
                self.map_size, self.hit_radius,
                self.max_steps, self.dt,
                self.maneuver_type
            )

            # Trim trajectory
            trajectory_array = trajectory_array[:actual_steps]

            # Calculate reward for this scenario
            reward = self._calculate_reward(trajectory_array, hit, hit_time)

            results.append({
                'reward': reward,
                'hit': hit,
                'hit_time': hit_time,
                'final_distance': trajectory_array[-1, 8] if len(trajectory_array) > 0 else 10000.0
            })

        # 3. Calculate average metrics
        avg_reward = np.mean([r['reward'] for r in results])
        avg_hit_rate = np.mean([float(r['hit']) for r in results])
        avg_hit_time = np.mean([r['hit_time'] if r['hit'] else self.max_steps for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])

        # 4. Update history
        self.history.append({
            'pid': (Kp, Ki, Kd),
            'avg_reward': avg_reward,
            'avg_hit_rate': avg_hit_rate,
            'avg_hit_time': avg_hit_time,
            'avg_final_distance': avg_final_distance
        })

        # Keep only last window_size trials
        if len(self.history) > self.window_size * 10:  # Prune when too large
            self.history = self.history[-self.window_size * 2:]

        # 5. Next observation
        obs = self._get_observation()

        # 6. Done (standard episodic)
        done = True

        # 7. Info
        info = {
            'pid_kp': Kp,
            'pid_ki': Ki,
            'pid_kd': Kd,
            'avg_reward': avg_reward,
            'avg_hit_rate': avg_hit_rate,
            'avg_hit_time': avg_hit_time,
            'avg_final_distance': avg_final_distance,
            'n_scenarios_tested': len(results)
        }

        return obs, avg_reward, done, False, info

    def _sample_scenario(self):
        """
        Sample a random scenario (missile + target initial conditions)

        Returns:
            scenario: dict of parameters
        """
        # Random missile initialization
        missile_x = np.random.uniform(0, 0.2 * self.map_size)
        missile_y = np.random.uniform(0.2 * self.map_size, 0.8 * self.map_size)
        missile_vx = np.random.uniform(0.8 * self.missile_speed, 0.9 * self.missile_speed)
        missile_vy = 0.0

        # Random target initialization
        target_x = np.random.uniform(0.6 * self.map_size, 0.9 * self.map_size)
        target_y = np.random.uniform(0.3 * self.map_size, 0.7 * self.map_size)
        target_heading = np.random.uniform(0, 2 * np.pi)

        scenario = {
            'missile_x': missile_x,
            'missile_y': missile_y,
            'missile_vx': missile_vx,
            'missile_vy': missile_vy,
            'target_x': target_x,
            'target_y': target_y,
            'target_heading': target_heading,
        }

        return scenario

    def _get_observation(self):
        """
        Construct observation from history

        Observation (5 * window_size):
        - [Kp_norm, Ki_norm, Kd_norm, reward_norm, hit_rate] × window_size

        Returns:
            obs: (5 * window_size,) array
        """
        obs = np.zeros(5 * self.window_size, dtype=np.float32)

        # Get last window_size trials
        recent_history = self.history[-self.window_size:] if len(self.history) > 0 else []

        for i, trial in enumerate(recent_history):
            Kp, Ki, Kd = trial['pid']
            avg_reward = trial['avg_reward']
            avg_hit_rate = trial['avg_hit_rate']

            # Normalize values
            kp_norm = (Kp - 2750) / 2250  # Center around 2750, range ±2250
            ki_norm = (Ki - 10) / 10      # Center around 10, range ±10
            kd_norm = (Kd - 10) / 10      # Center around 10, range ±10
            reward_norm = avg_reward / 100.0  # Approximate normalization

            # Pack into observation
            obs[i*5 + 0] = kp_norm
            obs[i*5 + 1] = ki_norm
            obs[i*5 + 2] = kd_norm
            obs[i*5 + 3] = reward_norm
            obs[i*5 + 4] = avg_hit_rate

        return obs

    def _calculate_reward(self, trajectory, hit, hit_time):
        """
        Calculate reward for one scenario

        Design: Continuous distance-based reward (not just hit/miss)

        Components:
        1. Hit bonus: +100 (sparse)
        2. Distance reward: Continuous based on final/avg distance
        3. Time bonus: Faster is better (if hit)
        4. Approaching bonus: Reward for getting closer

        Returns:
            reward: float
        """
        reward = 0.0

        if len(trajectory) == 0:
            return -100.0

        final_distance = trajectory[-1, 8]

        # 1. Hit bonus (sparse)
        if hit:
            reward += 100.0
            # Time bonus: Faster intercept = better (max +50)
            time_bonus = (self.max_steps - hit_time) / self.max_steps * 50.0
            reward += time_bonus
        else:
            # Miss penalty based on how close we got
            miss_penalty = -50.0
            reward += miss_penalty

        # 2. Distance reward (continuous, always present)
        # Reward for getting close, even if miss
        # Map distance [0, 10000] → reward [50, 0]
        distance_reward = max(0, 50.0 * (1.0 - final_distance / 10000.0))
        reward += distance_reward

        # 3. Average distance penalty (trajectory quality)
        avg_distance = np.mean(trajectory[:, 8])
        # Penalize staying far on average
        avg_distance_penalty = -avg_distance / 1000.0
        reward += avg_distance_penalty

        # 4. Approaching bonus (reward for closing in)
        if len(trajectory) > 1:
            initial_distance = trajectory[0, 8]
            distance_reduced = initial_distance - final_distance
            # Reward for reducing distance
            approaching_bonus = distance_reduced / 1000.0
            reward += approaching_bonus

        return reward

    def render(self):
        # Can implement visualization if needed
        pass
