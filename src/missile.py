"""
Simple 2D Missile with PID Controller
"""
import numpy as np


class PIDController:
    """Simple PID controller"""

    def __init__(self, kp=2.0, ki=0.1, kd=0.5):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        """Compute PID output"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        """Reset PID state"""
        self.integral = 0.0
        self.prev_error = 0.0

    def set_gains(self, kp, ki, kd):
        """Update gains"""
        self.kp = kp
        self.ki = ki
        self.kd = kd


class Missile:
    """2D Missile with PID guidance"""

    def __init__(self, x=0.0, y=0.0, vx=250.0, vy=0.0,
                 max_speed=300.0, max_accel=100.0,
                 kp=2.0, ki=0.1, kd=0.5):
        # State
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        # Constraints
        self.max_speed = max_speed
        self.max_accel = max_accel

        # PID controller
        self.pid = PIDController(kp, ki, kd)

        # History
        self.trajectory = [(x, y)]

        # Status
        self.active = True
        self.fuel = 1.0

    @property
    def position(self):
        return np.array([self.x, self.y])

    @property
    def velocity(self):
        return np.array([self.vx, self.vy])

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    @property
    def heading(self):
        """Heading in radians"""
        return np.arctan2(self.vy, self.vx)

    def update(self, target_pos, dt):
        """Update missile state"""
        if not self.active:
            return

        # Calculate desired heading
        dx = target_pos[0] - self.x
        dy = target_pos[1] - self.y
        desired_heading = np.arctan2(dy, dx)

        # Heading error
        error = desired_heading - self.heading
        # Normalize to [-pi, pi]
        error = np.arctan2(np.sin(error), np.cos(error))

        # PID control
        control = self.pid.compute(error, dt)
        control = np.clip(control, -self.max_accel, self.max_accel)

        # Apply lateral acceleration
        if self.speed > 0:
            # Perpendicular to velocity
            perp = np.array([-self.vy, self.vx]) / self.speed
            self.vx += perp[0] * control * dt
            self.vy += perp[1] * control * dt

        # Maintain cruise speed
        current_speed = self.speed
        if current_speed > 0:
            scale = self.max_speed / current_speed
            self.vx *= scale
            self.vy *= scale

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update fuel
        self.fuel -= 0.01 * dt
        if self.fuel <= 0:
            self.active = False

        # Record trajectory
        self.trajectory.append((self.x, self.y))

    def reset(self, x=0.0, y=0.0, vx=250.0, vy=0.0):
        """Reset missile state"""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.active = True
        self.fuel = 1.0
        self.trajectory = [(x, y)]
        self.pid.reset()

    def set_pid_gains(self, kp, ki, kd):
        """Update PID gains (RL action)"""
        self.pid.set_gains(kp, ki, kd)
