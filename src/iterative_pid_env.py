"""
Iterative Refinement PID Environment for TD3

Multi-step MDP where agent iteratively refines PID parameters:
- Episode = Multiple refinement steps
- Action = Delta PID (incremental changes)
- State = Current PID + Performance + History
- Done when converged or max iterations reached

Design Philosophy:
- Agent learns to make small, incremental improvements
- History provides context for next refinement
- Convergence-based termination
- Suitable for TD3 (continuous control, deterministic policy)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .episodic_fixed_pid_env import run_simulation_jit, MANEUVER_MAP


class IterativePIDEnv(gym.Env):
    """
    Iterative PID refinement environment using multi-step episodes.

    Episode Structure:
    - reset(): Initialize with random PID
    - step(delta_action): Apply incremental change → Test → Reward
    - Continue until convergence or max_iterations

    State (21D):
    - Current PID (3D): [Kp_norm, Ki_norm, Kd_norm]
    - Current performance (2D): [avg_time_norm, success_rate]
    - History (15D): [Kp, Ki, Kd, time, sr] × 3 recent trials
    - Progress (1D): iteration / max_iterations

    Action (3D):
    - Delta PID: [ΔKp, ΔKi, ΔKd] ∈ [-1, 1]
    - Scaled to actual bounds

    Reward:
    - Success rate (primary)
    - Time efficiency (secondary)
    - Stability penalty (discourage large changes)
    - Improvement bonus (vs previous trial)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self,
                 n_test_scenarios=5,
                 history_size=3,
                 max_iterations_per_episode=15,
                 convergence_threshold=0.05,
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
            history_size: Number of recent trials in history (default: 3)
            max_iterations_per_episode: Max refinement steps per episode (default: 15)
            convergence_threshold: Delta threshold for convergence (default: 0.05)
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
        self.history_size = history_size
        self.max_iterations_per_episode = max_iterations_per_episode
        self.convergence_threshold = convergence_threshold

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

        # PID bounds
        self.kp_bounds = [500.0, 5000.0]
        self.ki_bounds = [0.0, 20.0]
        self.kd_bounds = [0.0, 20.0]

        # Delta PID scaling (max change per step)
        self.delta_kp_scale = 500.0  # Max ±500 per step
        self.delta_ki_scale = 2.0    # Max ±2 per step
        self.delta_kd_scale = 2.0    # Max ±2 per step

        # Action space: Delta PID normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Observation space: [current_PID(3) + performance(2) + history(15) + progress(1)] = 21D
        self.observation_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(21,),
            dtype=np.float32
        )

        # Episode state
        self.current_pid = None
        self.current_performance = None
        self.history = []
        self.iteration = 0
        self.test_scenarios = []
        self.best_performance = None

    def reset(self, seed=None, options=None):
        """Reset episode with random initial PID"""
        super().reset(seed=seed)

        # Sample N diverse test scenarios (fixed for this episode)
        self.test_scenarios = [
            self._sample_scenario() for _ in range(self.n_test_scenarios)
        ]

        # Initialize with random PID (or guided initialization)
        self.current_pid = self._initialize_pid()

        # Test initial PID
        self.current_performance = self._test_pid(self.current_pid)

        # Reset episode state
        self.history = []
        self.iteration = 0
        self.best_performance = self.current_performance.copy()

        # Add initial state to history
        self._add_to_history(self.current_pid, self.current_performance)

        # Get observation
        obs = self._get_observation()

        return obs, {}

    def step(self, action):
        """
        Apply delta PID and evaluate

        Args:
            action: [ΔKp_norm, ΔKi_norm, ΔKd_norm] ∈ [-1, 1]

        Returns:
            obs, reward, done, truncated, info
        """
        # 1. Scale and apply delta
        delta_kp = float(action[0]) * self.delta_kp_scale
        delta_ki = float(action[1]) * self.delta_ki_scale
        delta_kd = float(action[2]) * self.delta_kd_scale

        # 2. Update PID
        new_kp = self.current_pid['Kp'] + delta_kp
        new_ki = self.current_pid['Ki'] + delta_ki
        new_kd = self.current_pid['Kd'] + delta_kd

        # Clip to bounds
        new_kp = np.clip(new_kp, self.kp_bounds[0], self.kp_bounds[1])
        new_ki = np.clip(new_ki, self.ki_bounds[0], self.ki_bounds[1])
        new_kd = np.clip(new_kd, self.kd_bounds[0], self.kd_bounds[1])

        # Round for discretization
        new_kp = round(new_kp / 100) * 100
        new_ki = round(new_ki, 1)
        new_kd = round(new_kd, 1)

        new_pid = {'Kp': new_kp, 'Ki': new_ki, 'Kd': new_kd}

        # 3. Test new PID
        new_performance = self._test_pid(new_pid)

        # 4. Calculate reward
        reward = self._calculate_reward(
            new_performance,
            self.current_performance,
            [delta_kp, delta_ki, delta_kd]
        )

        # 5. Update state
        self.current_pid = new_pid
        prev_performance = self.current_performance
        self.current_performance = new_performance
        self.iteration += 1

        # Update best performance
        if new_performance['success_rate'] > self.best_performance['success_rate']:
            self.best_performance = new_performance.copy()
        elif (new_performance['success_rate'] == self.best_performance['success_rate'] and
              new_performance['avg_time'] < self.best_performance['avg_time']):
            self.best_performance = new_performance.copy()

        # Add to history
        self._add_to_history(new_pid, new_performance)

        # 6. Check termination
        done = self._check_done(
            [delta_kp, delta_ki, delta_kd],
            new_performance,
            prev_performance
        )

        # 7. Get observation
        obs = self._get_observation()

        # 8. Info
        info = {
            'pid_kp': new_kp,
            'pid_ki': new_ki,
            'pid_kd': new_kd,
            'success_rate': new_performance['success_rate'],
            'avg_time': new_performance['avg_time'],
            'iteration': self.iteration,
            'converged': done and self.iteration < self.max_iterations_per_episode,
            'best_success_rate': self.best_performance['success_rate'],
            'best_avg_time': self.best_performance['avg_time'],
        }

        return obs, reward, done, False, info

    def _initialize_pid(self):
        """Initialize PID (random or guided)"""
        # Guided initialization around typical good values
        kp = np.random.uniform(1500, 3500)
        ki = np.random.uniform(0, 10)
        kd = np.random.uniform(0, 10)

        # Round
        kp = round(kp / 100) * 100
        ki = round(ki, 1)
        kd = round(kd, 1)

        return {'Kp': kp, 'Ki': ki, 'Kd': kd}

    def _test_pid(self, pid):
        """Test PID on all scenarios and return performance metrics"""
        Kp, Ki, Kd = pid['Kp'], pid['Ki'], pid['Kd']

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

            trajectory_array = trajectory_array[:actual_steps]

            results.append({
                'hit': hit,
                'hit_time': hit_time if hit else self.max_steps,
                'final_distance': trajectory_array[-1, 8] if len(trajectory_array) > 0 else self.map_size
            })

        # Calculate aggregate metrics
        success_rate = np.mean([r['hit'] for r in results])
        avg_time = np.mean([r['hit_time'] for r in results])
        avg_final_distance = np.mean([r['final_distance'] for r in results])

        return {
            'success_rate': success_rate,
            'avg_time': avg_time,
            'avg_final_distance': avg_final_distance
        }

    def _calculate_reward(self, new_perf, prev_perf, deltas):
        """
        Calculate reward for refinement step

        Components:
        1. Success rate (primary): 0-100
        2. Time efficiency (secondary): 0-20
        3. Improvement bonus: 0-30
        4. Delta penalty (stability): -5 to 0
        """
        reward = 0.0

        # 1. Success rate (primary objective)
        reward += new_perf['success_rate'] * 100.0

        # 2. Time efficiency (if successful)
        if new_perf['success_rate'] > 0:
            time_norm = new_perf['avg_time'] / self.max_steps
            time_reward = (1.0 - time_norm) * 20.0
            reward += time_reward

        # 3. Improvement bonus
        success_improvement = new_perf['success_rate'] - prev_perf['success_rate']
        if success_improvement > 0:
            reward += success_improvement * 50.0  # Big bonus for success improvement

        if new_perf['success_rate'] == prev_perf['success_rate'] and new_perf['success_rate'] > 0:
            # Same success rate, but faster?
            time_improvement = prev_perf['avg_time'] - new_perf['avg_time']
            if time_improvement > 0:
                reward += time_improvement / self.max_steps * 20.0

        # 4. Delta penalty (encourage stability, small changes)
        delta_magnitude = abs(deltas[0]) + abs(deltas[1]) + abs(deltas[2])
        delta_penalty = delta_magnitude / (self.delta_kp_scale + self.delta_ki_scale + self.delta_kd_scale) * 5.0
        reward -= delta_penalty

        # 5. Penalty for making things worse
        if success_improvement < 0:
            reward -= abs(success_improvement) * 30.0

        return reward

    def _check_done(self, deltas, new_perf, prev_perf):
        """Check if episode should terminate"""

        # Max iterations reached
        if self.iteration >= self.max_iterations_per_episode:
            return True

        # Convergence: small deltas + high success rate
        delta_magnitude = np.sqrt(
            (deltas[0] / self.delta_kp_scale) ** 2 +
            (deltas[1] / self.delta_ki_scale) ** 2 +
            (deltas[2] / self.delta_kd_scale) ** 2
        )

        if (delta_magnitude < self.convergence_threshold and
            new_perf['success_rate'] >= 0.8):
            return True

        # Performance plateaued (last 3 trials similar)
        if len(self.history) >= 3:
            recent_sr = [h['performance']['success_rate'] for h in self.history[-3:]]
            if np.std(recent_sr) < 0.02 and new_perf['success_rate'] >= 0.7:
                return True

        return False

    def _add_to_history(self, pid, performance):
        """Add trial to history"""
        self.history.append({
            'pid': pid.copy(),
            'performance': performance.copy()
        })

        # Keep only recent history
        if len(self.history) > self.history_size * 2:
            self.history = self.history[-self.history_size * 2:]

    def _get_observation(self):
        """
        Construct observation

        State (21D):
        - Current PID (3D): normalized
        - Current performance (2D): [time_norm, success_rate]
        - History (15D): [Kp_norm, Ki_norm, Kd_norm, time_norm, sr] × 3
        - Progress (1D): iteration / max_iterations
        """
        obs = np.zeros(21, dtype=np.float32)

        # Current PID (3D)
        obs[0] = (self.current_pid['Kp'] - 2750) / 2250  # Center at 2750
        obs[1] = (self.current_pid['Ki'] - 10) / 10
        obs[2] = (self.current_pid['Kd'] - 10) / 10

        # Current performance (2D)
        obs[3] = self.current_performance['avg_time'] / self.max_steps
        obs[4] = self.current_performance['success_rate']

        # History (15D = 3 trials × 5 features)
        recent_history = self.history[-self.history_size:] if len(self.history) > 0 else []
        for i, trial in enumerate(recent_history):
            idx = 5 + i * 5
            obs[idx + 0] = (trial['pid']['Kp'] - 2750) / 2250
            obs[idx + 1] = (trial['pid']['Ki'] - 10) / 10
            obs[idx + 2] = (trial['pid']['Kd'] - 10) / 10
            obs[idx + 3] = trial['performance']['avg_time'] / self.max_steps
            obs[idx + 4] = trial['performance']['success_rate']

        # Progress (1D)
        obs[20] = self.iteration / self.max_iterations_per_episode

        return obs

    def _sample_scenario(self):
        """Sample a random scenario"""
        missile_x = np.random.uniform(0, 0.2 * self.map_size)
        missile_y = np.random.uniform(0.2 * self.map_size, 0.8 * self.map_size)
        missile_vx = np.random.uniform(0.8 * self.missile_speed, 0.9 * self.missile_speed)
        missile_vy = 0.0

        target_x = np.random.uniform(0.6 * self.map_size, 0.9 * self.map_size)
        target_y = np.random.uniform(0.3 * self.map_size, 0.7 * self.map_size)
        target_heading = np.random.uniform(0, 2 * np.pi)

        return {
            'missile_x': missile_x,
            'missile_y': missile_y,
            'missile_vx': missile_vx,
            'missile_vy': missile_vy,
            'target_x': target_x,
            'target_y': target_y,
            'target_heading': target_heading,
        }

    def render(self):
        pass
