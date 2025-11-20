"""
Meta-Episodic PID Environment
RL learns optimal FIXED PID parameters through sequential decision making across multiple episodes

Key Features:
- One meta-episode = N consecutive episodes (default: 10)
- Each step = one full simulation with fixed PID
- Agent observes: Context (8D) + History (25D) = 33D
- Agent learns to adapt PID based on previous episode outcomes
- True MDP with temporal dependency via history buffer
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit
from .episodic_fixed_pid_env import run_simulation_jit, MANEUVER_MAP


class MetaEpisodicPIDEnv(gym.Env):
    """
    Meta-Episodic RL environment for sequential PID tuning.

    Unlike standard episodic approach (1 action → done), this creates a TRUE MDP:
    - Meta-episode contains N episodes (e.g., 10)
    - Each episode: Agent selects PID → Simulation runs → Observe outcome
    - Next episode: Agent adapts PID based on history
    - Done after N episodes

    Observation Space (33D):
    - Context (8D): Current scenario [missile_init, target_init, maneuver]
    - History (25D): Last 5 episodes' summary [PID, reward, hit] × 5

    Action Space (3D):
    - [Kp_log, Ki, Kd] where Kp is in log scale [2.0, 4.0] → [100, 10000]

    Benefits over standard episodic:
    1. Sequential decision making (not one-shot)
    2. Sample efficient (10 episodes → 10 samples per meta-episode)
    3. Agent learns meta-strategy: "If last PID worked, keep similar"
    4. Stationary initial observation (context, not zeros)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self,
                 episodes_per_meta=10,
                 window_size=5,
                 map_size=10000.0,
                 hit_radius=50.0,
                 max_steps=500,
                 dt=0.01,
                 target_maneuver='circular',
                 missile_speed=1000.0,
                 missile_accel=1000.0,
                 target_speed=1000.0):
        """
        Args:
            episodes_per_meta: Number of episodes per meta-episode (default: 10)
            window_size: Number of recent episodes to keep in history (default: 5)
            map_size: Map size (meters)
            hit_radius: Hit detection radius (meters)
            max_steps: Max steps per simulation
            dt: Time step (seconds)
            target_maneuver: Target maneuver type
            missile_speed: Missile max speed (m/s)
            missile_accel: Missile max acceleration (m/s²)
            target_speed: Target speed (m/s)
        """
        super().__init__()

        # Meta-episode parameters
        self.episodes_per_meta = episodes_per_meta
        self.window_size = window_size

        # Environment parameters
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
        # Kp: log scale [10^2, 10^4] = [100, 10000]
        # Ki, Kd: linear [0, 50]
        self.action_space = spaces.Box(
            low=np.array([2.0, 0.0, 0.0], dtype=np.float32),   # log10(100) = 2.0
            high=np.array([4.0, 50.0, 50.0], dtype=np.float32), # log10(10000) = 4.0
            dtype=np.float32
        )

        # Observation space: Context (8D) + History (25D) = 33D
        # Context: [m_x, m_y, m_vx, t_x, t_y, t_heading, maneuver_id, difficulty]
        # History: [Kp, Ki, Kd, reward, hit] × 5 episodes
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(33,),
            dtype=np.float32
        )

        # State
        self.episode_in_meta = 0  # Current episode number (0 to episodes_per_meta-1)
        self.history = []  # List of (context, PID, reward, hit) tuples
        self.current_context = None  # Current scenario parameters

    def reset(self, seed=None, options=None):
        """Reset meta-episode (start fresh with new scenarios)"""
        super().reset(seed=seed)

        # Reset meta-episode state
        self.episode_in_meta = 0
        self.history = []

        # Sample new scenario context
        self.current_context = self._sample_scenario()

        # Initial observation: context + empty history
        obs = self._get_observation()

        return obs, {}

    def step(self, action):
        """
        Execute one episode with given PID parameters

        Args:
            action: [Kp_log, Ki, Kd] PID parameters

        Returns:
            obs: Next observation (33D)
            reward: Episode reward
            done: True if meta-episode finished (after episodes_per_meta steps)
            truncated: False
            info: Episode statistics
        """
        # 1. Extract and transform PID parameters
        log_kp = float(action[0])
        Kp = 10 ** log_kp
        Ki = float(action[1])
        Kd = float(action[2])

        # 2. Discretize to interpretable values
        Kp = round(Kp / 100) * 100  # Nearest 100
        Ki = round(Ki / 5) * 5       # Nearest 5
        Kd = round(Kd / 5) * 5

        # 3. Clamp to valid ranges
        Kp = np.clip(Kp, 100, 10000)
        Ki = np.clip(Ki, 0, 50)
        Kd = np.clip(Kd, 0, 50)

        # 4. Run simulation with current context
        trajectory_array, hit, hit_time, actual_steps = run_simulation_jit(
            self.current_context['missile_x'],
            self.current_context['missile_y'],
            self.current_context['missile_vx'],
            self.current_context['missile_vy'],
            self.current_context['target_x'],
            self.current_context['target_y'],
            self.current_context['target_heading'],
            self.target_speed,
            Kp, Ki, Kd,
            self.missile_speed, self.missile_accel,
            self.map_size, self.hit_radius,
            self.max_steps, self.dt,
            self.maneuver_type
        )

        # Trim trajectory to actual steps
        trajectory_array = trajectory_array[:actual_steps]

        # 5. Calculate reward
        reward = self._calculate_reward(trajectory_array, hit, hit_time)

        # 6. Update history
        self.history.append({
            'context': self.current_context.copy(),
            'pid': (Kp, Ki, Kd),
            'reward': reward,
            'hit': hit,
            'hit_time': hit_time,
            'final_distance': trajectory_array[-1, 8] if len(trajectory_array) > 0 else 10000.0
        })

        # 7. Move to next episode in meta
        self.episode_in_meta += 1

        # 8. Check if meta-episode is done
        done = (self.episode_in_meta >= self.episodes_per_meta)

        # 9. Sample new scenario for next episode (if not done)
        if not done:
            self.current_context = self._sample_scenario()
            obs = self._get_observation()
        else:
            # Meta-episode finished, return zeros (will be reset anyway)
            obs = np.zeros(33, dtype=np.float32)

        # 10. Info
        info = {
            'hit': hit,
            'hit_time': hit_time,
            'final_distance': trajectory_array[-1, 8] if len(trajectory_array) > 0 else 10000.0,
            'pid_kp': Kp,
            'pid_ki': Ki,
            'pid_kd': Kd,
            'episode_in_meta': self.episode_in_meta - 1,  # 0-indexed
            'meta_episode_done': done
        }

        return obs, reward, done, False, info

    def _sample_scenario(self):
        """
        Sample a random scenario (missile + target initial conditions)

        Returns:
            context: dict of scenario parameters
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

        # Calculate difficulty (initial distance)
        dx = target_x - missile_x
        dy = target_y - missile_y
        initial_distance = np.sqrt(dx * dx + dy * dy)
        difficulty = initial_distance / self.map_size  # Normalized [0, 1]

        context = {
            'missile_x': missile_x,
            'missile_y': missile_y,
            'missile_vx': missile_vx,
            'missile_vy': missile_vy,
            'target_x': target_x,
            'target_y': target_y,
            'target_heading': target_heading,
            'maneuver_id': self.maneuver_type,
            'difficulty': difficulty
        }

        return context

    def _get_observation(self):
        """
        Construct observation from current context and history

        Observation (33D):
        - Context (8D): [m_x, m_y, m_vx, t_x, t_y, t_heading, maneuver_id, difficulty]
        - History (25D): [Kp, Ki, Kd, reward, hit] × 5 episodes

        Returns:
            obs: (33,) array
        """
        # Context (8D) - Normalized to [0, 1] ranges
        context_vec = np.array([
            self.current_context['missile_x'] / self.map_size,
            self.current_context['missile_y'] / self.map_size,
            self.current_context['missile_vx'] / self.missile_speed,
            self.current_context['target_x'] / self.map_size,
            self.current_context['target_y'] / self.map_size,
            self.current_context['target_heading'] / (2 * np.pi),
            self.current_context['maneuver_id'] / 3.0,  # [0, 3] → [0, 1]
            self.current_context['difficulty']
        ], dtype=np.float32)

        # History (25D) - Last 5 episodes
        history_vec = np.zeros(25, dtype=np.float32)

        # Get last window_size episodes
        recent_history = self.history[-self.window_size:] if len(self.history) > 0 else []

        for i, episode in enumerate(recent_history):
            # Normalize PID values
            Kp, Ki, Kd = episode['pid']
            reward = episode['reward']
            hit = float(episode['hit'])

            # Pack into 5D: [Kp/10000, Ki/50, Kd/50, reward/100, hit]
            history_vec[i*5 + 0] = Kp / 10000.0  # Normalized [0, 1]
            history_vec[i*5 + 1] = Ki / 50.0
            history_vec[i*5 + 2] = Kd / 50.0
            history_vec[i*5 + 3] = reward / 100.0  # Approximate normalization
            history_vec[i*5 + 4] = hit  # Binary [0, 1]

        # Concatenate
        obs = np.concatenate([context_vec, history_vec])

        return obs

    def _calculate_reward(self, trajectory, hit, hit_time):
        """
        Calculate episodic reward based on full trajectory

        Components:
        1. Hit bonus (+100)
        2. Time bonus (faster = better)
        3. Miss penalty (-50)
        4. Average distance penalty
        5. Trajectory smoothness bonus
        """
        reward = 0.0

        if len(trajectory) == 0:
            # Immediate failure
            return -100.0

        # 1. Hit/Miss
        if hit:
            reward += 100.0
            # Time bonus: Faster intercept is better
            time_bonus = (self.max_steps - hit_time) / 10.0
            reward += time_bonus
        else:
            # Miss penalty
            reward -= 50.0
            # Final distance penalty
            final_distance = trajectory[-1, 8]
            reward -= final_distance / 1000.0

        # 2. Average distance penalty (trajectory quality)
        avg_distance = np.mean(trajectory[:, 8])
        reward -= avg_distance / 1000.0

        # 3. Trajectory smoothness (less oscillation = better)
        # Calculate acceleration changes
        velocities = trajectory[:, 2:4]  # [vx, vy]
        if len(velocities) > 1:
            accelerations = np.diff(velocities, axis=0)
            jerk = np.diff(accelerations, axis=0)
            smoothness_penalty = np.mean(np.linalg.norm(jerk, axis=1))
            reward -= smoothness_penalty / 10000.0

        # 4. Closing velocity bonus (approaching target is good)
        avg_closing_vel = np.mean(trajectory[:, 10])
        if avg_closing_vel > 0:
            reward += avg_closing_vel / 1000.0

        return reward

    def render(self):
        # Can implement visualization if needed
        pass
