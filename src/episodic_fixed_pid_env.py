"""
Episodic Fixed PID Environment
RL learns optimal FIXED PID parameters by observing full simulation trajectory
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .missile import Missile
from .target import Target


class EpisodicFixedPIDEnv(gym.Env):
    """
    Episodic RL environment for finding optimal fixed PID parameters.

    Key differences from step-level:
    - Action sets PID parameters ONCE at episode start
    - Environment runs FULL simulation (500 steps)
    - Observation = downsampled trajectory (50 samples × 12 features = 600D)
    - Reward = episodic (based on full trajectory result)

    Workflow:
    1. RL agent selects [Kp, Ki, Kd]
    2. Environment runs full 500-step simulation
    3. Trajectory is downsampled (every 10 steps)
    4. Observation = flattened trajectory
    5. Reward = based on hit, time, trajectory quality
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, map_size=10000.0, hit_radius=50.0, max_steps=500,
                 dt=0.01, target_maneuver='circular',
                 missile_speed=1000.0, missile_accel=1000.0,
                 target_speed=1000.0, downsample_rate=10):
        super().__init__()

        # Action: Direct PID parameter values (WIDE range for large simulation)
        self.action_space = spaces.Box(
            low=np.array([0.1, 0.0, 0.0], dtype=np.float32),
            high=np.array([10000.0, 50.0, 50.0], dtype=np.float32),
            dtype=np.float32
        )

        # Observation: Downsampled trajectory
        # 500 steps / 10 = 50 samples
        # 12 features per sample: [m_x, m_y, m_vx, m_vy, t_x, t_y, t_vx, t_vy,
        #                           distance, angle_error, closing_velocity, heading_error]
        # Total: 50 × 12 = 600D
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(600,),
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
        self.downsample_rate = downsample_rate

        # State
        self.missile = None
        self.target = None
        self.trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset missile (random initial position and velocity)
        missile_x = np.random.uniform(0, 0.2 * self.map_size)
        missile_y = np.random.uniform(0.2 * self.map_size, 0.8 * self.map_size)
        initial_vx = np.random.uniform(0.8 * self.missile_speed, 0.9 * self.missile_speed)

        self.missile = Missile(
            x=missile_x, y=missile_y,
            vx=initial_vx, vy=0.0,
            max_speed=self.missile_speed,
            max_accel=self.missile_accel,
            kp=2.0, ki=0.1, kd=0.5  # Default (will be overridden)
        )

        # Reset target (random initial position)
        target_x = np.random.uniform(0.6 * self.map_size, 0.9 * self.map_size)
        target_y = np.random.uniform(0.3 * self.map_size, 0.7 * self.map_size)
        target_heading = np.random.uniform(0, 2 * np.pi)

        self.target = Target(
            x=target_x, y=target_y,
            speed=self.target_speed,
            maneuver=self.target_maneuver
        )
        self.target.heading = target_heading

        self.trajectory = []

        # Initial observation: zeros (no trajectory yet)
        return np.zeros(600, dtype=np.float32), {}

    def step(self, action):
        """
        Run FULL simulation with given PID parameters

        Args:
            action: [Kp, Ki, Kd] PID parameters

        Returns:
            obs: Downsampled trajectory (600D)
            reward: Episodic reward
            done: Always True (1 step = 1 episode)
            truncated: Always False
            info: Episode statistics
        """
        # 1. Set PID parameters from action
        Kp = float(action[0])
        Ki = float(action[1])
        Kd = float(action[2])

        self.missile.pid.kp = Kp
        self.missile.pid.ki = Ki
        self.missile.pid.kd = Kd

        # 2. Run FULL simulation (500 steps)
        self.trajectory = []
        hit = False
        hit_time = self.max_steps

        for step in range(self.max_steps):
            # Update simulation
            self.missile.update(self.target.position, self.dt)
            self.target.update(self.dt, missile_pos=self.missile.position)

            # Calculate metrics
            dx = self.target.x - self.missile.x
            dy = self.target.y - self.missile.y
            distance = np.sqrt(dx**2 + dy**2)

            # Angle error
            desired_heading = np.arctan2(dy, dx)
            current_heading = self.missile.heading
            angle_error = desired_heading - current_heading
            angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))

            # Closing velocity (rate of distance decrease)
            missile_vel = self.missile.velocity
            target_vel = self.target.velocity
            relative_vel = missile_vel - target_vel
            range_vec = np.array([dx, dy])
            if distance > 0:
                closing_velocity = -np.dot(relative_vel, range_vec) / distance
            else:
                closing_velocity = 0.0

            # Heading error (missile heading vs desired heading)
            heading_error = angle_error  # Same as angle_error

            # Record trajectory (12 features)
            self.trajectory.append([
                self.missile.x,
                self.missile.y,
                self.missile.vx,
                self.missile.vy,
                self.target.x,
                self.target.y,
                self.target.vx,
                self.target.vy,
                distance,
                angle_error,
                closing_velocity,
                heading_error
            ])

            # Check hit
            if distance < self.hit_radius:
                hit = True
                hit_time = step
                break

            # Check out of bounds
            if (self.missile.x < -1000 or self.missile.x > self.map_size + 1000 or
                self.missile.y < -1000 or self.missile.y > self.map_size + 1000):
                break

            # Check missile active
            if not self.missile.active:
                break

        # 3. Downsample trajectory (every 10 steps → 50 samples)
        trajectory_array = np.array(self.trajectory)

        if len(trajectory_array) > 0:
            # Downsample
            downsampled = trajectory_array[::self.downsample_rate]  # Every 10th step

            # Ensure exactly 50 samples (pad or trim)
            if len(downsampled) < 50:
                # Pad with last sample
                padding = np.tile(downsampled[-1], (50 - len(downsampled), 1))
                downsampled = np.vstack([downsampled, padding])
            elif len(downsampled) > 50:
                # Trim
                downsampled = downsampled[:50]

            # Flatten to 600D
            obs = downsampled.flatten().astype(np.float32)
        else:
            # No trajectory (immediate failure)
            obs = np.zeros(600, dtype=np.float32)

        # 4. Calculate episodic reward
        reward = self._calculate_reward(trajectory_array, hit, hit_time)

        # 5. Episode info
        final_distance = trajectory_array[-1, 8] if len(trajectory_array) > 0 else 10000.0

        info = {
            'hit': hit,
            'hit_time': hit_time,
            'final_distance': final_distance,
            'trajectory_length': len(trajectory_array),
            'pid_kp': Kp,
            'pid_ki': Ki,
            'pid_kd': Kd
        }

        # Episode always done after 1 step (full simulation)
        return obs, reward, True, False, info

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
        # Can use the same renderer as before if needed
        pass


# Alias for backward compatibility
FixedPIDEnv = EpisodicFixedPIDEnv
