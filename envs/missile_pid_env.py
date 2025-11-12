"""
Gymnasium Environment for Missile PID Tuning with RL
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from warsim.simulator.missile import Missile
from warsim.simulator.target import Target


class MissilePIDEnv(gym.Env):
    """
    Gymnasium environment for training RL agent to tune PID parameters
    for missile guidance against maneuvering targets.

    Observation Space:
        - Missile position (x, y)
        - Missile velocity (vx, vy)
        - Missile heading
        - Target position (x, y)
        - Target velocity (vx, vy)
        - Target heading
        - Relative distance
        - Relative angle
        - Current PID gains (kp, ki, kd)
        - Fuel remaining

    Action Space:
        - Continuous: [delta_kp, delta_ki, delta_kd]
        - Each action is a small adjustment to current PID gains

    Reward:
        - Distance reduction to target
        - Penalty for large PID adjustments
        - Bonus for hitting target
        - Penalty for missing/running out of fuel
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self,
                 max_steps: int = 500,
                 dt: float = 0.1,
                 map_size: float = 10000.0,  # 10 km x 10 km
                 hit_radius: float = 50.0,  # 50 meters
                 target_maneuver: str = "straight",
                 render_mode: Optional[str] = None):

        super().__init__()

        self.max_steps = max_steps
        self.dt = dt
        self.map_size = map_size
        self.hit_radius = hit_radius
        self.target_maneuver = target_maneuver
        self.render_mode = render_mode

        # Define observation space (14 dimensions)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )

        # Define action space (3 dimensions: delta_kp, delta_ki, delta_kd)
        # Actions are normalized adjustments to PID gains
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32
        )

        # Environment state
        self.missile: Optional[Missile] = None
        self.target: Optional[Target] = None
        self.steps = 0
        self.prev_distance = 0.0

        # PID gain limits
        self.kp_range = (0.1, 10.0)
        self.ki_range = (0.0, 5.0)
        self.kd_range = (0.0, 5.0)

        # Reward weights
        self.reward_distance_weight = 1.0
        self.reward_hit_bonus = 1000.0
        self.reward_miss_penalty = -500.0
        self.reward_action_penalty_weight = 0.01

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        self.steps = 0

        # Random initial positions
        # Missile starts from bottom-left quadrant
        missile_x = np.random.uniform(0, self.map_size * 0.3)
        missile_y = np.random.uniform(0, self.map_size * 0.3)

        # Target starts from top-right quadrant
        target_x = np.random.uniform(self.map_size * 0.6, self.map_size)
        target_y = np.random.uniform(self.map_size * 0.6, self.map_size)

        # Initial velocities
        missile_heading = np.random.uniform(0, 360)
        missile_speed = 250.0
        missile_vx = missile_speed * np.cos(missile_heading * np.pi / 180)
        missile_vy = missile_speed * np.sin(missile_heading * np.pi / 180)

        # Initialize missile with default PID gains
        initial_kp = 2.0
        initial_ki = 0.1
        initial_kd = 0.5

        self.missile = Missile(
            x=missile_x,
            y=missile_y,
            vx=missile_vx,
            vy=missile_vy,
            max_speed=300.0,
            max_acceleration=100.0,
            pid_kp=initial_kp,
            pid_ki=initial_ki,
            pid_kd=initial_kd
        )

        # Initialize target
        target_heading = np.random.uniform(0, 360)
        self.target = Target(
            x=target_x,
            y=target_y,
            speed=150.0,
            maneuver_type=self.target_maneuver
        )
        self.target.heading = target_heading
        self.target.initial_heading = target_heading

        # Initialize previous distance
        self.prev_distance = self._get_distance()

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one step in the environment"""
        self.steps += 1

        # Apply action to adjust PID gains
        current_kp = self.missile.pid.kp
        current_ki = self.missile.pid.ki
        current_kd = self.missile.pid.kd

        # Scale actions to reasonable adjustments (e.g., ±10% of current value)
        delta_kp = action[0] * 0.1 * current_kp
        delta_ki = action[1] * 0.1 * current_ki if current_ki > 0 else action[1] * 0.05
        delta_kd = action[2] * 0.1 * current_kd if current_kd > 0 else action[2] * 0.05

        new_kp = np.clip(current_kp + delta_kp, self.kp_range[0], self.kp_range[1])
        new_ki = np.clip(current_ki + delta_ki, self.ki_range[0], self.ki_range[1])
        new_kd = np.clip(current_kd + delta_kd, self.kd_range[0], self.kd_range[1])

        self.missile.set_pid_gains(new_kp, new_ki, new_kd)

        # Update missile and target
        self.missile.update(self.target.x, self.target.y, self.dt)
        self.target.update(self.dt, missile_position=self.missile.position)

        # Calculate reward
        reward = self._calculate_reward(action)

        # Check termination conditions
        distance = self._get_distance()
        terminated = False
        truncated = False

        if distance < self.hit_radius:
            # Hit target
            reward += self.reward_hit_bonus
            terminated = True

        elif not self.missile.active:
            # Out of fuel or inactive
            reward += self.reward_miss_penalty
            terminated = True

        elif self._is_out_of_bounds():
            # Out of bounds
            reward += self.reward_miss_penalty
            terminated = True

        elif self.steps >= self.max_steps:
            # Max steps reached
            truncated = True

        obs = self._get_observation()
        info = self._get_info()
        info['distance'] = distance
        info['hit'] = distance < self.hit_radius

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        missile_state = self.missile.get_state()
        target_state = self.target.get_state()

        # Calculate relative measurements
        dx = target_state['x'] - missile_state['x']
        dy = target_state['y'] - missile_state['y']
        distance = np.sqrt(dx**2 + dy**2)
        relative_angle = np.arctan2(dy, dx) * 180 / np.pi

        # Normalize observations
        obs = np.array([
            missile_state['x'] / self.map_size,
            missile_state['y'] / self.map_size,
            missile_state['vx'] / 300.0,
            missile_state['vy'] / 300.0,
            missile_state['heading'] / 360.0,
            target_state['x'] / self.map_size,
            target_state['y'] / self.map_size,
            target_state['vx'] / 200.0,
            target_state['vy'] / 200.0,
            target_state['heading'] / 360.0,
            distance / (self.map_size * np.sqrt(2)),
            relative_angle / 360.0,
            self.missile.pid.kp / 10.0,
            self.missile.pid.ki / 5.0,
        ], dtype=np.float32)

        return obs

    def _get_distance(self) -> float:
        """Calculate distance between missile and target"""
        dx = self.target.x - self.missile.x
        dy = self.target.y - self.missile.y
        return np.sqrt(dx**2 + dy**2)

    def _is_out_of_bounds(self) -> bool:
        """Check if missile or target is out of bounds"""
        return (self.missile.x < 0 or self.missile.x > self.map_size or
                self.missile.y < 0 or self.missile.y > self.map_size)

    def _calculate_reward(self, action: np.ndarray) -> float:
        """Calculate reward for current step"""
        current_distance = self._get_distance()

        # Reward for reducing distance to target
        distance_reduction = self.prev_distance - current_distance
        reward = distance_reduction * self.reward_distance_weight

        # Penalty for large PID adjustments (encourage smooth tuning)
        action_penalty = np.sum(np.abs(action)) * self.reward_action_penalty_weight
        reward -= action_penalty

        # Update previous distance
        self.prev_distance = current_distance

        return reward

    def _get_info(self) -> dict:
        """Get additional info"""
        return {
            'steps': self.steps,
            'missile_position': self.missile.position,
            'target_position': self.target.position,
            'pid_gains': {
                'kp': self.missile.pid.kp,
                'ki': self.missile.pid.ki,
                'kd': self.missile.pid.kd
            },
            'fuel': self.missile.fuel_remaining
        }

    def render(self):
        """Render the environment (basic implementation)"""
        if self.render_mode == "human":
            print(f"Step: {self.steps}, Distance: {self._get_distance():.2f}m, "
                  f"PID: Kp={self.missile.pid.kp:.2f}, Ki={self.missile.pid.ki:.2f}, Kd={self.missile.pid.kd:.2f}")


# Register environment
gym.register(
    id='MissilePID-v0',
    entry_point='envs.missile_pid_env:MissilePIDEnv',
    max_episode_steps=500,
)
