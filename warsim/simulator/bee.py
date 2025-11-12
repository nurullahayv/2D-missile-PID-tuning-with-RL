"""
Bee entity for honeycomb construction simulation.
"""
import numpy as np
from typing import Optional


class Bee:
    """
    A bee agent that can move and build honeycomb walls.
    """

    def __init__(self, bee_id: int, grid_size: int, num_directions: int = 32):
        """
        Initialize a bee.

        Args:
            bee_id: Unique identifier for this bee
            grid_size: Size of the grid world
            num_directions: Number of discrete movement directions (default 32)
        """
        self.bee_id = bee_id
        self.grid_size = grid_size
        self.num_directions = num_directions

        # Position (continuous for smooth movement, but snapped to grid)
        self.x = np.random.uniform(0, grid_size)
        self.y = np.random.uniform(0, grid_size)

        # Direction (index from 0 to num_directions-1)
        self.direction = np.random.randint(0, num_directions)

        # State
        self.alive = True
        self.current_task = 0  # High-level task assignment (0=explore, 1-3=build zones)
        self.building_at: Optional[tuple] = None  # (y, x) if currently building
        self.build_ticks_remaining = 0

        # Memory (visited locations, decays over time)
        self.visited_grid = np.zeros((grid_size, grid_size), dtype=np.float32)

    def get_grid_position(self) -> tuple:
        """Get current position snapped to grid."""
        return int(self.y), int(self.x)

    def set_position(self, y: float, x: float):
        """Set position (with bounds checking)."""
        self.x = np.clip(x, 0, self.grid_size - 1)
        self.y = np.clip(y, 0, self.grid_size - 1)

    def move(self, direction_idx: int, speed: float = 1.0):
        """
        Move the bee in the specified direction.

        Args:
            direction_idx: Direction index (0 to num_directions-1)
            speed: Movement speed in grid cells per tick
        """
        if not self.alive:
            return

        # Update direction
        self.direction = direction_idx % self.num_directions

        # Calculate movement delta
        angle_rad = (direction_idx * 2 * np.pi) / self.num_directions
        dx = speed * np.cos(angle_rad)
        dy = -speed * np.sin(angle_rad)  # Negative because y increases downward

        # Update position
        new_x = self.x + dx
        new_y = self.y + dy

        # Bounds checking (wrap or clamp)
        self.x = np.clip(new_x, 0, self.grid_size - 1)
        self.y = np.clip(new_y, 0, self.grid_size - 1)

        # Update visited grid
        grid_y, grid_x = self.get_grid_position()
        self.visited_grid[grid_y, grid_x] = 1.0

    def start_building(self, location: tuple, ticks_required: int):
        """
        Start building at a specific location.

        Args:
            location: (y, x) grid coordinates
            ticks_required: Number of ticks needed to complete
        """
        self.building_at = location
        self.build_ticks_remaining = ticks_required

    def update_building(self) -> bool:
        """
        Update building progress.

        Returns:
            True if building is complete, False otherwise
        """
        if self.building_at is None:
            return False

        self.build_ticks_remaining -= 1

        if self.build_ticks_remaining <= 0:
            # Building complete
            self.building_at = None
            self.build_ticks_remaining = 0
            return True

        return False

    def cancel_building(self):
        """Cancel current building activity."""
        self.building_at = None
        self.build_ticks_remaining = 0

    def is_building(self) -> bool:
        """Check if bee is currently building."""
        return self.building_at is not None

    def decay_visited_memory(self, decay_rate: float = 0.99):
        """
        Decay the visited grid memory over time.

        Args:
            decay_rate: Multiplier for decay (0.99 = 1% decay per tick)
        """
        self.visited_grid *= decay_rate

    def get_state_dict(self) -> dict:
        """Get bee state as dictionary for observation."""
        grid_y, grid_x = self.get_grid_position()
        return {
            'bee_id': self.bee_id,
            'x': self.x,
            'y': self.y,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'direction': self.direction,
            'alive': self.alive,
            'current_task': self.current_task,
            'is_building': self.is_building(),
            'building_at': self.building_at,
            'build_progress': 1.0 - (self.build_ticks_remaining / 256.0) if self.is_building() else 0.0
        }

    def __repr__(self):
        grid_y, grid_x = self.get_grid_position()
        status = "building" if self.is_building() else "idle"
        return f"Bee{self.bee_id}(pos=({grid_y},{grid_x}), dir={self.direction}, {status})"
