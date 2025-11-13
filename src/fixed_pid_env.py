"""
Non-Adaptive PID Environment
RL learns optimal FIXED PID parameters (set once at episode start)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .missile import Missile
from .target import Target


class FixedPIDEnv(gym.Env):
    """
    Non-adaptive RL environment for finding optimal fixed PID parameters.

    Difference from adaptive version:
    - Agent sets PID parameters ONCE at episode start
    - Parameters stay FIXED throughout the episode
    - Simpler action space: direct PID values [Kp, Ki, Kd]
    - Simpler observation: no need to track current PID values

    Goal: Find optimal static PID gains for each target maneuver type
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, map_size=10000.0, hit_radius=50.0, max_steps=500,
                 dt=0.01, target_maneuver='circular',
                 missile_speed=1000.0, missile_accel=1000.0,
                 target_speed=1000.0):
        super().__init__()

        # Action: Direct PID parameter values (set once per episode)
        # [Kp, Ki, Kd] with reasonable bounds
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0, 0.0], dtype=np.float32),   # Min values
            high=np.array([10.0, 2.0, 5.0], dtype=np.float32), # Max values
            dtype=np.float32
        )

        # Observation: missile state, target state, relative info
        # No need to include current PID values (they're fixed)
        # [m_x, m_y, m_vx, m_vy, t_x, t_y, t_vx, t_vy, distance, angle_error, fuel]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(11,),
            dtype=np.float32
        )

        # Environment parameters
        self.map_size = map_size
        self.hit_radius = hit_radius
        self.max_steps = max_steps
        self.dt = dt
        self.target_maneuver = target_maneuver
        self.missile_speed = missile_speed
        self.missile_accel = missile_accel
        self.target_speed = target_speed

        # State
        self.missile = None
        self.target = None
        self.steps = 0
        self.fixed_pid_set = False
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset missile (random initial velocity)
        initial_vx = np.random.uniform(0.8 * self.missile_speed, 0.9 * self.missile_speed)
        self.missile = Missile(
            x=0.0, y=0.0,
            vx=initial_vx, vy=0.0,
            max_speed=self.missile_speed,
            max_accel=self.missile_accel,
            kp=2.0, ki=0.1, kd=0.5  # Default values (will be overridden)
        )

        # Reset target
        target_x = np.random.uniform(0.6 * self.map_size, 0.9 * self.map_size)
        target_y = np.random.uniform(0.3 * self.map_size, 0.7 * self.map_size)
        self.target = Target(
            x=target_x, y=target_y,
            speed=self.target_speed,
            maneuver=self.target_maneuver
        )

        self.steps = 0
        self.fixed_pid_set = False
        self.trajectory = [(self.missile.x, self.missile.y)]

        return self._get_obs(), {}

    def step(self, action):
        # On FIRST step: set PID parameters from action
        if not self.fixed_pid_set:
            kp = float(action[0])
            ki = float(action[1])
            kd = float(action[2])

            # Set missile PID parameters (fixed for entire episode)
            self.missile.pid.kp = kp
            self.missile.pid.ki = ki
            self.missile.pid.kd = kd

            self.fixed_pid_set = True

        # On subsequent steps: action is IGNORED (PID stays fixed)

        # Update simulation
        self.missile.update(self.target.x, self.target.y, self.dt)
        self.target.update(self.dt)
        self.steps += 1

        # Track trajectory
        self.trajectory.append((self.missile.x, self.missile.y))

        # Calculate distance
        dx = self.target.x - self.missile.x
        dy = self.target.y - self.missile.y
        distance = np.sqrt(dx**2 + dy**2)

        # Check termination
        hit = distance < self.hit_radius
        out_of_bounds = (self.missile.x < -1000 or self.missile.x > self.map_size + 1000 or
                        self.missile.y < -1000 or self.missile.y > self.map_size + 1000)
        out_of_fuel = self.missile.fuel <= 0
        max_steps_reached = self.steps >= self.max_steps

        done = hit or out_of_bounds or out_of_fuel or max_steps_reached

        # Reward calculation
        reward = 0.0

        # Distance penalty (encourage getting closer)
        reward -= distance / 1000.0

        # Hit bonus
        if hit:
            reward += 100.0
            # Time bonus (faster intercept is better)
            time_bonus = (self.max_steps - self.steps) / 10.0
            reward += time_bonus

        # Miss penalty
        if (out_of_bounds or out_of_fuel or max_steps_reached) and not hit:
            reward -= 50.0

        # Fuel efficiency bonus
        if self.missile.fuel > 0:
            reward += self.missile.fuel / 10000.0

        info = {
            'hit': hit,
            'distance': distance,
            'steps': self.steps,
            'pid_kp': self.missile.pid.kp,
            'pid_ki': self.missile.pid.ki,
            'pid_kd': self.missile.pid.kd,
            'fuel': self.missile.fuel
        }

        return self._get_obs(), reward, done, False, info

    def _get_obs(self):
        """Get current observation (no PID values needed)"""
        # Missile state
        m_x = self.missile.x / self.map_size
        m_y = self.missile.y / self.map_size
        m_vx = self.missile.vx / self.missile.max_speed
        m_vy = self.missile.vy / self.missile.max_speed

        # Target state
        t_x = self.target.x / self.map_size
        t_y = self.target.y / self.map_size
        t_vx = self.target.vx / self.target.speed
        t_vy = self.target.vy / self.target.speed

        # Relative info
        dx = self.target.x - self.missile.x
        dy = self.target.y - self.missile.y
        distance = np.sqrt(dx**2 + dy**2) / self.map_size

        # Angle error
        desired_angle = np.arctan2(dy, dx)
        current_angle = np.arctan2(self.missile.vy, self.missile.vx)
        angle_error = desired_angle - current_angle
        angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))  # Normalize to [-pi, pi]

        # Fuel
        fuel = self.missile.fuel / 1000.0

        obs = np.array([
            m_x, m_y, m_vx, m_vy,
            t_x, t_y, t_vx, t_vy,
            distance, angle_error, fuel
        ], dtype=np.float32)

        return obs

    def render(self):
        # Can use the same renderer as adaptive version
        pass
