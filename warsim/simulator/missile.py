"""
2D Missile with PID Controller for Target Pursuit
"""
import numpy as np
from typing import Tuple


class PIDController:
    """PID Controller for missile guidance"""
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, error: float, dt: float) -> float:
        """Update PID controller with current error"""
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.prev_error = error

        return output

    def reset(self):
        """Reset PID controller state"""
        self.integral = 0.0
        self.prev_error = 0.0

    def set_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains"""
        self.kp = kp
        self.ki = ki
        self.kd = kd


class Missile:
    """2D Missile with PID guidance"""
    def __init__(self,
                 x: float,
                 y: float,
                 vx: float,
                 vy: float,
                 max_speed: float = 300.0,  # m/s
                 max_acceleration: float = 100.0,  # m/s^2
                 pid_kp: float = 2.0,
                 pid_ki: float = 0.1,
                 pid_kd: float = 0.5):

        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy

        self.max_speed = max_speed
        self.max_acceleration = max_acceleration

        # PID controller for heading control
        self.pid = PIDController(pid_kp, pid_ki, pid_kd)

        # History for visualization
        self.trajectory = [(x, y)]

        # Missile state
        self.active = True
        self.fuel_remaining = 1.0  # normalized fuel
        self.fuel_consumption_rate = 0.01  # per second

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position"""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity"""
        return (self.vx, self.vy)

    @property
    def speed(self) -> float:
        """Get current speed"""
        return np.sqrt(self.vx**2 + self.vy**2)

    @property
    def heading(self) -> float:
        """Get current heading in degrees (0-360)"""
        angle = np.arctan2(self.vy, self.vx) * 180 / np.pi
        return (angle + 360) % 360

    def update(self, target_x: float, target_y: float, dt: float):
        """
        Update missile state using PID control

        Args:
            target_x: Target x position
            target_y: Target y position
            dt: Time step
        """
        if not self.active:
            return

        # Calculate desired heading to target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = np.sqrt(dx**2 + dy**2)

        if distance < 1.0:  # Hit target
            self.active = False
            return

        desired_angle = np.arctan2(dy, dx)
        current_angle = np.arctan2(self.vy, self.vx)

        # Calculate heading error (normalized to [-pi, pi])
        error = desired_angle - current_angle
        error = np.arctan2(np.sin(error), np.cos(error))

        # PID control for heading correction
        control_signal = self.pid.update(error, dt)

        # Apply control signal as lateral acceleration
        # Clip to max acceleration
        lateral_acc = np.clip(control_signal, -self.max_acceleration, self.max_acceleration)

        # Update velocity with lateral acceleration
        # Perpendicular direction to current velocity
        if self.speed > 0:
            perp_vx = -self.vy / self.speed
            perp_vy = self.vx / self.speed
        else:
            perp_vx = 0
            perp_vy = 0

        self.vx += perp_vx * lateral_acc * dt
        self.vy += perp_vy * lateral_acc * dt

        # Maintain speed (simple cruise control)
        current_speed = self.speed
        if current_speed > 0:
            speed_factor = self.max_speed / current_speed
            self.vx *= speed_factor
            self.vy *= speed_factor

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Update fuel
        self.fuel_remaining -= self.fuel_consumption_rate * dt
        if self.fuel_remaining <= 0:
            self.fuel_remaining = 0
            self.active = False

        # Store trajectory
        self.trajectory.append((self.x, self.y))

    def set_pid_gains(self, kp: float, ki: float, kd: float):
        """Update PID gains (this is what RL will learn)"""
        self.pid.set_gains(kp, ki, kd)

    def reset(self, x: float, y: float, vx: float, vy: float):
        """Reset missile to new state"""
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.active = True
        self.fuel_remaining = 1.0
        self.trajectory = [(x, y)]
        self.pid.reset()

    def get_state(self) -> dict:
        """Get missile state for observation"""
        return {
            'x': self.x,
            'y': self.y,
            'vx': self.vx,
            'vy': self.vy,
            'speed': self.speed,
            'heading': self.heading,
            'fuel': self.fuel_remaining,
            'active': self.active
        }
