"""
Episodic Fixed PID Environment
RL learns optimal FIXED PID parameters by observing full simulation trajectory
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit
from .missile import Missile
from .target import Target


# Maneuver type encoding for JIT
MANEUVER_STRAIGHT = 0
MANEUVER_CIRCULAR = 1
MANEUVER_ZIGZAG = 2
MANEUVER_EVASIVE = 3

MANEUVER_MAP = {
    'straight': MANEUVER_STRAIGHT,
    'circular': MANEUVER_CIRCULAR,
    'zigzag': MANEUVER_ZIGZAG,
    'evasive': MANEUVER_EVASIVE
}


@njit
def run_simulation_jit(
    missile_x, missile_y, missile_vx, missile_vy,
    target_x, target_y, target_heading, target_speed,
    kp, ki, kd,
    missile_max_speed, missile_max_accel,
    map_size, hit_radius, max_steps, dt,
    maneuver_type
):
    """
    JIT-compiled simulation loop for maximum performance

    Returns:
        trajectory: (max_steps, 12) array of trajectory data
        hit: bool
        hit_time: int
        actual_steps: int (actual number of steps before termination)
    """
    # Pre-allocate trajectory array
    trajectory = np.zeros((max_steps, 12), dtype=np.float32)

    # PID state
    pid_integral = 0.0
    pid_prev_error = 0.0

    # Target maneuver state
    maneuver_timer = 0.0

    # Simulation state
    hit = False
    hit_time = max_steps

    for step in range(max_steps):
        # ========== MISSILE UPDATE ==========
        # Calculate desired heading (proportional navigation)
        dx = target_x - missile_x
        dy = target_y - missile_y
        distance = np.sqrt(dx * dx + dy * dy)
        desired_heading = np.arctan2(dy, dx)

        # Current heading
        current_heading = np.arctan2(missile_vy, missile_vx)

        # Heading error (normalized to [-pi, pi])
        error = desired_heading - current_heading
        error = np.arctan2(np.sin(error), np.cos(error))

        # PID control
        pid_integral += error * dt
        derivative = (error - pid_prev_error) / dt if dt > 0 else 0.0
        control = kp * error + ki * pid_integral + kd * derivative
        pid_prev_error = error

        # Apply control to heading
        new_heading = current_heading + control * dt

        # Calculate desired velocity
        desired_vx = missile_max_speed * np.cos(new_heading)
        desired_vy = missile_max_speed * np.sin(new_heading)

        # Apply acceleration limits
        dvx = desired_vx - missile_vx
        dvy = desired_vy - missile_vy
        accel_magnitude = np.sqrt(dvx * dvx + dvy * dvy) / dt if dt > 0 else 0.0

        if accel_magnitude > missile_max_accel:
            scale = missile_max_accel / accel_magnitude
            dvx *= scale
            dvy *= scale

        # Update velocity
        missile_vx += dvx
        missile_vy += dvy

        # Limit speed
        speed = np.sqrt(missile_vx * missile_vx + missile_vy * missile_vy)
        if speed > missile_max_speed:
            missile_vx = (missile_vx / speed) * missile_max_speed
            missile_vy = (missile_vy / speed) * missile_max_speed

        # Update position
        missile_x += missile_vx * dt
        missile_y += missile_vy * dt

        # ========== TARGET UPDATE ==========
        maneuver_timer += dt

        # Apply maneuver
        if maneuver_type == MANEUVER_CIRCULAR:
            turn_rate = 0.5  # rad/s
            target_heading += turn_rate * dt

        elif maneuver_type == MANEUVER_ZIGZAG:
            if maneuver_timer > 2.0:
                target_heading += np.pi / 4
                maneuver_timer = 0.0

        elif maneuver_type == MANEUVER_EVASIVE:
            escape_dx = target_x - missile_x
            escape_dy = target_y - missile_y
            escape_distance = np.sqrt(escape_dx * escape_dx + escape_dy * escape_dy)

            if escape_distance < 2000.0:
                target_heading = np.arctan2(escape_dy, escape_dx)

        # Update target position
        target_vx = target_speed * np.cos(target_heading)
        target_vy = target_speed * np.sin(target_heading)
        target_x += target_vx * dt
        target_y += target_vy * dt

        # ========== CALCULATE METRICS ==========
        # Angle error (same as heading error)
        angle_error = error

        # Closing velocity
        relative_vx = missile_vx - target_vx
        relative_vy = missile_vy - target_vy
        if distance > 0:
            closing_velocity = -(relative_vx * dx + relative_vy * dy) / distance
        else:
            closing_velocity = 0.0

        heading_error = angle_error

        # Record trajectory (12 features)
        trajectory[step, 0] = missile_x
        trajectory[step, 1] = missile_y
        trajectory[step, 2] = missile_vx
        trajectory[step, 3] = missile_vy
        trajectory[step, 4] = target_x
        trajectory[step, 5] = target_y
        trajectory[step, 6] = target_vx
        trajectory[step, 7] = target_vy
        trajectory[step, 8] = distance
        trajectory[step, 9] = angle_error
        trajectory[step, 10] = closing_velocity
        trajectory[step, 11] = heading_error

        # ========== CHECK TERMINATION ==========
        # Hit check
        if distance < hit_radius:
            hit = True
            hit_time = step
            return trajectory, hit, hit_time, step + 1

        # Out of bounds check
        if (missile_x < -1000 or missile_x > map_size + 1000 or
            missile_y < -1000 or missile_y > map_size + 1000):
            return trajectory, hit, hit_time, step + 1

    return trajectory, hit, hit_time, max_steps


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

        # Encode maneuver type for JIT
        self.maneuver_type = MANEUVER_MAP.get(target_maneuver, MANEUVER_STRAIGHT)

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
        Run FULL simulation with given PID parameters (JIT-accelerated)

        Args:
            action: [Kp, Ki, Kd] PID parameters

        Returns:
            obs: Downsampled trajectory (600D)
            reward: Episodic reward
            done: Always True (1 step = 1 episode)
            truncated: Always False
            info: Episode statistics
        """
        # 1. Extract PID parameters from action
        Kp = float(action[0])
        Ki = float(action[1])
        Kd = float(action[2])

        # 2. Run JIT-compiled simulation (10-50x faster!)
        trajectory_array, hit, hit_time, actual_steps = run_simulation_jit(
            self.missile.x, self.missile.y,
            self.missile.vx, self.missile.vy,
            self.target.x, self.target.y,
            self.target.heading, self.target_speed,
            Kp, Ki, Kd,
            self.missile_speed, self.missile_accel,
            self.map_size, self.hit_radius,
            self.max_steps, self.dt,
            self.maneuver_type
        )

        # Trim trajectory to actual steps
        trajectory_array = trajectory_array[:actual_steps]

        # 3. Downsample trajectory (every 10 steps → 50 samples)

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
