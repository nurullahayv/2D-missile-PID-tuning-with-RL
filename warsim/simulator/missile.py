"""
2D Missile Guidance Control System with PID Controller
======================================================

This module implements a closed-loop control system for missile guidance.
See CONTROL_SYSTEM_ARCHITECTURE.md for detailed system analysis.

Control System Components:
    - Controller: PID (this class)
    - Plant: Missile dynamics (2D kinematics)
    - Sensor: Position/velocity measurement (perfect sensing)
    - Reference: Target position (time-varying)
    - Actuator: Acceleration command with saturation
"""
import numpy as np
from typing import Tuple


class PIDController:
    """
    Proportional-Integral-Derivative (PID) Controller
    ==================================================

    Mathematical Formulation:
        u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de(t)/dt

    Where:
        - e(t): Error signal (desired - actual)
        - u(t): Control output (heading correction)
        - Kp: Proportional gain (reacts to current error)
        - Ki: Integral gain (eliminates steady-state error)
        - Kd: Derivative gain (predicts future error, provides damping)

    Discrete-Time Implementation:
        u[k] = Kp·e[k] + Ki·Σe[i]·Δt + Kd·(e[k]-e[k-1])/Δt

    Control Theory:
        - Type: Linear time-invariant (LTI) controller
        - Feedback: Negative feedback loop
        - Stability: Depends on gain tuning (Routh-Hurwitz criterion)
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0):
        """
        Initialize PID controller

        Args:
            kp: Proportional gain (recommended: 1.0-5.0)
            ki: Integral gain (recommended: 0.0-1.0)
            kd: Derivative gain (recommended: 0.0-2.0)
        """
        # PID Gains (tunable parameters)
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain

        # Controller state (memory)
        self.integral = 0.0      # Accumulated error (integral term)
        self.prev_error = 0.0    # Previous error (for derivative)

    def update(self, error: float, dt: float) -> float:
        """
        Compute control signal based on error

        This implements the PID control law in discrete time.

        Args:
            error: Error signal e(t) = r(t) - y(t)
            dt: Time step (sampling period)

        Returns:
            u(t): Control signal (heading correction in radians)

        Mathematical Implementation:
            P_term = Kp * e[k]
            I_term = Ki * Σe[i]·Δt  (accumulated)
            D_term = Kd * (e[k] - e[k-1]) / Δt
            u[k] = P_term + I_term + D_term
        """
        # Integral term (accumulate error over time)
        self.integral += error * dt

        # Derivative term (rate of change of error)
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        # PID control law
        proportional_term = self.kp * error
        integral_term = self.ki * self.integral
        derivative_term = self.kd * derivative

        output = proportional_term + integral_term + derivative_term

        # Update state for next iteration
        self.prev_error = error

        return output

    def reset(self):
        """Reset controller internal state (for new episode)"""
        self.integral = 0.0
        self.prev_error = 0.0

    def set_gains(self, kp: float, ki: float, kd: float):
        """
        Update PID gains (RL adaptation interface)

        Args:
            kp: New proportional gain
            ki: New integral gain
            kd: New derivative gain
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd


class Missile:
    """
    Plant: 2D Missile Dynamics with PID Guidance Control
    =====================================================

    This class represents the PLANT in the control system block diagram.

    State-Space Representation:
        State vector: x = [x_m, y_m, v_x, v_y]^T

        State equations (continuous-time):
            ẋ_m = v_x
            ẏ_m = v_y
            v̇_x = a_x
            v̇_y = a_y

        Matrix form: ẋ = Ax + Bu
        where:
            A = [0 0 1 0]    B = [0 0]
                [0 0 0 1]        [0 0]
                [0 0 0 0]        [1 0]
                [0 0 0 0]        [0 1]

            u = [a_x, a_y]^T (control input from actuator)

    Discrete-Time Update (Euler integration):
        x[k+1] = x[k] + v_x[k]·Δt
        y[k+1] = y[k] + v_y[k]·Δt
        v_x[k+1] = v_x[k] + a_x[k]·Δt
        v_y[k+1] = v_y[k] + a_y[k]·Δt

    Control Architecture:
        Reference r(t) → [Σ] → PID Controller → Actuator → Plant → Output y(t)
                          ↑                                              ↓
                          └──────────────── Feedback ───────────────────┘
    """

    def __init__(self,
                 x: float,
                 y: float,
                 vx: float,
                 vy: float,
                 max_speed: float = 300.0,        # m/s (actuator constraint)
                 max_acceleration: float = 100.0,  # m/s² (actuator constraint)
                 pid_kp: float = 2.0,
                 pid_ki: float = 0.1,
                 pid_kd: float = 0.5):
        """
        Initialize missile control system

        Args:
            x, y: Initial position (m)
            vx, vy: Initial velocity (m/s)
            max_speed: Maximum velocity constraint (m/s)
            max_acceleration: Maximum acceleration constraint (m/s²)
            pid_kp, pid_ki, pid_kd: Initial PID gains
        """

        # ==========================================
        # STATE VARIABLES (Plant state)
        # ==========================================
        self.x = x    # Position x (m)
        self.y = y    # Position y (m)
        self.vx = vx  # Velocity x (m/s)
        self.vy = vy  # Velocity y (m/s)

        # ==========================================
        # ACTUATOR CONSTRAINTS
        # ==========================================
        self.max_speed = max_speed              # v_max (m/s)
        self.max_acceleration = max_acceleration # a_max (m/s²)

        # ==========================================
        # CONTROLLER
        # ==========================================
        self.pid = PIDController(pid_kp, pid_ki, pid_kd)

        # ==========================================
        # SYSTEM OUTPUT (for visualization/logging)
        # ==========================================
        self.trajectory = [(x, y)]  # Output history: y(t) = [x_m(t), y_m(t)]

        # ==========================================
        # ADDITIONAL DYNAMICS
        # ==========================================
        self.active = True                      # System operational status
        self.fuel_remaining = 1.0               # Normalized fuel [0, 1]
        self.fuel_consumption_rate = 0.01       # Fuel consumption rate (1/s)

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
        Closed-Loop Control System Update (one time step)
        ==================================================

        This method implements the complete control loop:
        1. Measure output: y(t) = [x_m, y_m]
        2. Calculate error: e(t) = r(t) - y(t)
        3. Controller: u(t) = PID(e(t))
        4. Actuator: a(t) = saturate(u(t), a_max)
        5. Plant: Update state using dynamics

        Args:
            target_x, target_y: Reference input r(t) = [x_t, y_t]
            dt: Sampling period Δt (s)

        Control Flow:
            r(t) → [Σ] → PID → Actuator → Plant → y(t)
                    ↑─────────────────────────────┘
                         Feedback (negative)
        """
        if not self.active:
            return

        # ==========================================
        # 1. ERROR CALCULATION (Summing Junction)
        # ==========================================
        # Reference input: r(t) = [x_t, y_t]
        # Output: y(t) = [x_m, y_m]
        # Error: e(t) = r(t) - y(t)
        dx = target_x - self.x  # Error in x
        dy = target_y - self.y  # Error in y
        distance = np.sqrt(dx**2 + dy**2)

        # Terminal condition check
        if distance < 1.0:  # Within hit radius
            self.active = False
            return

        # Convert position error to heading error (angular error)
        desired_angle = np.arctan2(dy, dx)      # θ_desired
        current_angle = np.arctan2(self.vy, self.vx)  # θ_current

        # Heading error (wrapped to [-π, π])
        error = desired_angle - current_angle  # θ_error
        error = np.arctan2(np.sin(error), np.cos(error))  # Wrap angle

        # ==========================================
        # 2. CONTROLLER (PID)
        # ==========================================
        # Compute control signal u(t) using PID law
        # u(t) = Kp·e(t) + Ki·∫e(τ)dτ + Kd·de(t)/dt
        control_signal = self.pid.update(error, dt)  # u(t)

        # ==========================================
        # 3. ACTUATOR (with saturation constraints)
        # ==========================================
        # Convert control signal to acceleration command
        # Saturation: |a| ≤ a_max
        lateral_acc = np.clip(control_signal,
                             -self.max_acceleration,
                             self.max_acceleration)

        # ==========================================
        # 4. PLANT DYNAMICS (State update)
        # ==========================================
        # Apply lateral acceleration (perpendicular to velocity)
        if self.speed > 0:
            # Unit vector perpendicular to velocity
            perp_vx = -self.vy / self.speed
            perp_vy = self.vx / self.speed
        else:
            perp_vx = 0
            perp_vy = 0

        # Update velocity: v[k+1] = v[k] + a·Δt
        self.vx += perp_vx * lateral_acc * dt
        self.vy += perp_vy * lateral_acc * dt

        # Speed regulation (maintain cruise speed)
        # Constraint: ||v|| ≤ v_max
        current_speed = self.speed
        if current_speed > 0:
            speed_factor = self.max_speed / current_speed
            self.vx *= speed_factor
            self.vy *= speed_factor

        # Update position: x[k+1] = x[k] + v[k]·Δt
        self.x += self.vx * dt
        self.y += self.vy * dt

        # ==========================================
        # 5. ADDITIONAL DYNAMICS (Fuel consumption)
        # ==========================================
        self.fuel_remaining -= self.fuel_consumption_rate * dt
        if self.fuel_remaining <= 0:
            self.fuel_remaining = 0
            self.active = False

        # ==========================================
        # 6. OUTPUT LOGGING
        # ==========================================
        # Store output trajectory: y(t)
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
