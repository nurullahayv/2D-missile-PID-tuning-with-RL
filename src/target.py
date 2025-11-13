"""
Simple 2D Moving Target
"""
import numpy as np


class Target:
    """Moving target with different maneuver patterns"""

    def __init__(self, x=8000.0, y=5000.0, speed=150.0, maneuver='straight'):
        # State
        self.x = x
        self.y = y
        self.speed = speed
        self.heading = np.pi  # Start heading left

        # Maneuver type
        self.maneuver = maneuver
        self.time = 0.0

        # History
        self.trajectory = [(x, y)]

    @property
    def position(self):
        return np.array([self.x, self.y])

    @property
    def velocity(self):
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)
        return np.array([vx, vy])

    def update(self, dt, missile_pos=None):
        """Update target position"""
        self.time += dt

        # Update heading based on maneuver
        if self.maneuver == 'straight':
            pass  # Keep heading

        elif self.maneuver == 'circular':
            # Circular motion
            turn_rate = 0.3  # rad/s
            self.heading += turn_rate * dt

        elif self.maneuver == 'zigzag':
            # Zigzag pattern
            if int(self.time) % 4 < 2:
                turn_rate = 0.5
            else:
                turn_rate = -0.5
            self.heading += turn_rate * dt

        elif self.maneuver == 'evasive' and missile_pos is not None:
            # Evade missile
            dx = self.x - missile_pos[0]
            dy = self.y - missile_pos[1]
            distance = np.sqrt(dx**2 + dy**2)

            if distance < 3000:
                # Turn away from missile
                escape_heading = np.arctan2(dy, dx)
                heading_diff = escape_heading - self.heading
                heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
                turn_rate = 0.8 * np.sign(heading_diff)
                self.heading += turn_rate * dt

        # Update position
        vx = self.speed * np.cos(self.heading)
        vy = self.speed * np.sin(self.heading)
        self.x += vx * dt
        self.y += vy * dt

        # Record trajectory
        self.trajectory.append((self.x, self.y))

    def reset(self, x=8000.0, y=5000.0, heading=np.pi):
        """Reset target state"""
        self.x = x
        self.y = y
        self.heading = heading
        self.time = 0.0
        self.trajectory = [(x, y)]
