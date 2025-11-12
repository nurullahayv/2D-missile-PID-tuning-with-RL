"""
2D Moving Target with Predefined Maneuvers
"""
import numpy as np
from typing import Tuple, List


class Target:
    """2D Moving Target with various maneuver patterns"""

    def __init__(self,
                 x: float,
                 y: float,
                 speed: float = 150.0,  # m/s
                 maneuver_type: str = "straight"):
        """
        Initialize target

        Args:
            x: Initial x position
            y: Initial y position
            speed: Target speed (m/s)
            maneuver_type: Type of maneuver pattern
                - "straight": Straight line motion
                - "circular": Circular motion
                - "zigzag": Zigzag pattern
                - "evasive": Evasive maneuvers (random turns)
        """
        self.x = x
        self.y = y
        self.speed = speed
        self.initial_heading = 45.0  # degrees
        self.heading = self.initial_heading

        self.maneuver_type = maneuver_type
        self.time = 0.0

        # Maneuver parameters
        self.turn_rate = 30.0  # degrees per second for circular
        self.zigzag_period = 3.0  # seconds
        self.zigzag_amplitude = 45.0  # degrees

        # Evasive maneuver state
        self.evasive_change_interval = 2.0  # seconds
        self.last_maneuver_change = 0.0

        # History
        self.trajectory = [(x, y)]

    @property
    def position(self) -> Tuple[float, float]:
        """Get current position"""
        return (self.x, self.y)

    @property
    def velocity(self) -> Tuple[float, float]:
        """Get current velocity"""
        vx = self.speed * np.cos(self.heading * np.pi / 180)
        vy = self.speed * np.sin(self.heading * np.pi / 180)
        return (vx, vy)

    def update(self, dt: float, missile_position: Tuple[float, float] = None):
        """
        Update target state based on maneuver type

        Args:
            dt: Time step
            missile_position: Optional missile position for evasive maneuvers
        """
        self.time += dt

        # Update heading based on maneuver type
        if self.maneuver_type == "straight":
            # No heading change
            pass

        elif self.maneuver_type == "circular":
            # Circular motion
            self.heading += self.turn_rate * dt
            self.heading = self.heading % 360

        elif self.maneuver_type == "zigzag":
            # Zigzag pattern
            zigzag_offset = self.zigzag_amplitude * np.sin(2 * np.pi * self.time / self.zigzag_period)
            self.heading = self.initial_heading + zigzag_offset

        elif self.maneuver_type == "evasive":
            # Evasive maneuvers (change direction periodically)
            if self.time - self.last_maneuver_change >= self.evasive_change_interval:
                # Random turn
                turn = np.random.uniform(-60, 60)
                self.heading += turn
                self.heading = self.heading % 360
                self.last_maneuver_change = self.time

                # If missile is close, turn more aggressively
                if missile_position is not None:
                    dx = missile_position[0] - self.x
                    dy = missile_position[1] - self.y
                    distance = np.sqrt(dx**2 + dy**2)

                    if distance < 1000:  # Within 1km
                        # Turn away from missile
                        missile_angle = np.arctan2(dy, dx) * 180 / np.pi
                        # Turn perpendicular to missile direction
                        self.heading = (missile_angle + 90) % 360

        # Update position based on velocity
        vx, vy = self.velocity
        self.x += vx * dt
        self.y += vy * dt

        # Store trajectory
        self.trajectory.append((self.x, self.y))

    def reset(self, x: float, y: float, heading: float = None, maneuver_type: str = None):
        """Reset target to new state"""
        self.x = x
        self.y = y
        if heading is not None:
            self.heading = heading
            self.initial_heading = heading
        else:
            self.heading = self.initial_heading

        if maneuver_type is not None:
            self.maneuver_type = maneuver_type

        self.time = 0.0
        self.last_maneuver_change = 0.0
        self.trajectory = [(x, y)]

    def get_state(self) -> dict:
        """Get target state for observation"""
        vx, vy = self.velocity
        return {
            'x': self.x,
            'y': self.y,
            'vx': vx,
            'vy': vy,
            'speed': self.speed,
            'heading': self.heading
        }
