"""
Gymnasium Environment for Missile PID Tuning
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.missile import Missile
from src.target import Target


class MissilePIDEnv(gym.Env):
    """
    Gym environment for training RL agent to tune PID parameters

    Observation: [missile_x, missile_y, missile_vx, missile_vy,
                  target_x, target_y, target_vx, target_vy,
                  distance, angle_error, kp, ki, kd, fuel]

    Action: [delta_kp, delta_ki, delta_kd] ∈ [-1, 1]^3

    Reward: -distance + hit_bonus - miss_penalty
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self,
                 map_size=10000.0,
                 hit_radius=50.0,
                 max_steps=500,
                 dt=0.01,
                 target_maneuver='circular',
                 render_mode=None):
        super().__init__()

        self.map_size = map_size
        self.hit_radius = hit_radius
        self.max_steps = max_steps
        self.dt = dt
        self.target_maneuver = target_maneuver
        self.render_mode = render_mode

        # Action space: [delta_kp, delta_ki, delta_kd]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Observation space: 14D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        # PID gain ranges
        self.kp_range = (0.1, 10.0)
        self.ki_range = (0.0, 5.0)
        self.kd_range = (0.0, 5.0)

        # Initialize entities
        self.missile = None
        self.target = None
        self.renderer = None

        # Episode state
        self.step_count = 0
        self.prev_distance = None

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        # Random initial positions
        missile_x = np.random.uniform(0, 2000)
        missile_y = np.random.uniform(2000, 8000)
        missile_vx = np.random.uniform(200, 250)
        missile_vy = 0.0

        target_x = np.random.uniform(7000, 9000)
        target_y = np.random.uniform(2000, 8000)
        target_heading = np.random.uniform(0, 2*np.pi)

        # Initialize missile with default PID
        self.missile = Missile(
            x=missile_x, y=missile_y,
            vx=missile_vx, vy=missile_vy,
            kp=2.0, ki=0.1, kd=0.5
        )

        # Initialize target
        self.target = Target(
            x=target_x, y=target_y,
            speed=150.0,
            maneuver=self.target_maneuver
        )
        self.target.heading = target_heading

        # Reset episode state
        self.step_count = 0
        self.prev_distance = np.linalg.norm(
            self.missile.position - self.target.position
        )

        obs = self._get_observation()
        info = {}

        return obs, info

    def step(self, action):
        """Execute one step"""
        self.step_count += 1

        # Apply PID gain adjustments
        delta_kp, delta_ki, delta_kd = action

        current_kp = self.missile.pid.kp
        current_ki = self.missile.pid.ki
        current_kd = self.missile.pid.kd

        # Update gains with clipping
        new_kp = np.clip(
            current_kp + delta_kp * 0.5,
            self.kp_range[0], self.kp_range[1]
        )
        new_ki = np.clip(
            current_ki + delta_ki * 0.2,
            self.ki_range[0], self.ki_range[1]
        )
        new_kd = np.clip(
            current_kd + delta_kd * 0.2,
            self.kd_range[0], self.kd_range[1]
        )

        self.missile.set_pid_gains(new_kp, new_ki, new_kd)

        # Update simulation
        self.missile.update(self.target.position, self.dt)
        self.target.update(self.dt, self.missile.position)

        # Calculate reward
        distance = np.linalg.norm(
            self.missile.position - self.target.position
        )

        reward = 0.0
        terminated = False
        truncated = False

        # Distance reward (normalized)
        reward -= distance / self.map_size

        # Progress reward
        if self.prev_distance is not None:
            progress = (self.prev_distance - distance) / self.map_size
            reward += progress * 10.0

        self.prev_distance = distance

        # Check hit
        if distance < self.hit_radius:
            reward += 100.0
            terminated = True

        # Check miss (out of bounds or fuel)
        if (not self.missile.active or
            self.missile.x < 0 or self.missile.x > self.map_size or
            self.missile.y < 0 or self.missile.y > self.map_size or
            self.target.x < 0 or self.target.x > self.map_size or
            self.target.y < 0 or self.target.y > self.map_size):
            reward -= 50.0
            terminated = True

        # Max steps
        if self.step_count >= self.max_steps:
            truncated = True

        obs = self._get_observation()
        info = {
            'distance': distance,
            'hit': distance < self.hit_radius,
            'step': self.step_count,
            'pid_gains': {
                'kp': self.missile.pid.kp,
                'ki': self.missile.pid.ki,
                'kd': self.missile.pid.kd
            }
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation"""
        missile_pos = self.missile.position
        missile_vel = self.missile.velocity
        target_pos = self.target.position
        target_vel = self.target.velocity

        # Distance and angle
        dx = target_pos[0] - missile_pos[0]
        dy = target_pos[1] - missile_pos[1]
        distance = np.sqrt(dx**2 + dy**2)

        desired_heading = np.arctan2(dy, dx)
        current_heading = self.missile.heading
        angle_error = desired_heading - current_heading
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

        # Normalize
        obs = np.array([
            missile_pos[0] / self.map_size,
            missile_pos[1] / self.map_size,
            missile_vel[0] / self.missile.max_speed,
            missile_vel[1] / self.missile.max_speed,
            target_pos[0] / self.map_size,
            target_pos[1] / self.map_size,
            target_vel[0] / self.target.speed,
            target_vel[1] / self.target.speed,
            distance / self.map_size,
            angle_error / np.pi,
            self.missile.pid.kp / 10.0,
            self.missile.pid.ki / 5.0,
            self.missile.pid.kd / 5.0,
            self.missile.fuel
        ], dtype=np.float32)

        return obs

    def render(self):
        """Render environment"""
        if self.render_mode is None:
            return

        if self.renderer is None:
            from src.renderer import SimpleRenderer
            self.renderer = SimpleRenderer(self.map_size)

        return self.renderer.render(
            self.missile,
            self.target,
            self.hit_radius,
            self.step_count
        )

    def close(self):
        """Close environment"""
        if self.renderer is not None:
            self.renderer.close()
