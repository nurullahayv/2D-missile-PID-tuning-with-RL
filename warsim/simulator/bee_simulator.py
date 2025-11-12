"""
Bee colony simulator with collaborative honeycomb construction.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

from warsim.simulator.bee import Bee
from warsim.utils.grid_utils import (
    calculate_enclosed_areas,
    is_adjacent_to_wall,
    get_build_location,
    direction_to_delta,
    get_observation_window
)


class BeeSimulator:
    """
    Simulator for bee colony honeycomb construction.
    """

    def __init__(self,
                 num_bees: int = 7,
                 grid_size: int = 500,
                 num_directions: int = 32,
                 base_build_ticks: int = 256,
                 movement_speed: float = 1.0,
                 memory_decay: float = 0.995):
        """
        Initialize the bee colony simulator.

        Args:
            num_bees: Number of bees in the colony
            grid_size: Size of the square grid world
            num_directions: Number of discrete movement directions
            base_build_ticks: Base ticks for single bee building (256)
            movement_speed: Movement speed in grid cells per tick
            memory_decay: Decay rate for visited memory per tick
        """
        self.num_bees = num_bees
        self.grid_size = grid_size
        self.num_directions = num_directions
        self.base_build_ticks = base_build_ticks
        self.movement_speed = movement_speed
        self.memory_decay = memory_decay

        # Initialize bees
        self.bees: List[Bee] = [
            Bee(bee_id=i, grid_size=grid_size, num_directions=num_directions)
            for i in range(num_bees)
        ]

        # Grid state
        self.wall_grid = np.zeros((grid_size, grid_size), dtype=np.int8)  # 0=empty, 1=wall

        # Build queue: location -> {bee_ids: set, ticks_remaining: int}
        self.build_queue: Dict[Tuple[int, int], Dict] = {}

        # Tracking
        self.total_enclosed_area = 0
        self.previous_enclosed_area = 0
        self.total_walls_built = 0
        self.tick_count = 0

        # Reward tracking per bee
        self.bee_rewards = defaultdict(float)

    def reset(self):
        """Reset the simulator to initial state."""
        # Reset bees
        self.bees = [
            Bee(bee_id=i, grid_size=self.grid_size, num_directions=self.num_directions)
            for i in range(self.num_bees)
        ]

        # Reset grids
        self.wall_grid.fill(0)

        # Reset build queue
        self.build_queue.clear()

        # Reset tracking
        self.total_enclosed_area = 0
        self.previous_enclosed_area = 0
        self.total_walls_built = 0
        self.tick_count = 0
        self.bee_rewards.clear()

    def get_state(self) -> dict:
        """Get current simulator state."""
        return {
            'bees': [bee.get_state_dict() for bee in self.bees],
            'wall_grid': self.wall_grid.copy(),
            'build_queue': dict(self.build_queue),
            'total_enclosed_area': self.total_enclosed_area,
            'total_walls_built': self.total_walls_built,
            'tick_count': self.tick_count
        }

    def do_tick(self, actions: Dict[int, Tuple[int, int]]):
        """
        Execute one simulation tick.

        Args:
            actions: Dict of bee_id -> (direction_idx, build_action)
                    direction_idx: 0 to num_directions-1
                    build_action: 0 = no build, 1-8 = build at neighbor
        """
        self.tick_count += 1
        self.bee_rewards.clear()

        # Phase 1: Process movement for non-building bees
        for bee in self.bees:
            if not bee.alive:
                continue

            # If bee is building, skip movement
            if bee.is_building():
                continue

            # Get action
            direction_idx, _ = actions.get(bee.bee_id, (0, 0))

            # Move bee
            bee.move(direction_idx, self.movement_speed)

            # Decay visited memory
            bee.decay_visited_memory(self.memory_decay)

        # Phase 2: Process build actions
        self._process_build_actions(actions)

        # Phase 3: Update build queue
        self._update_build_queue()

        # Phase 4: Calculate enclosed area and rewards
        self._calculate_rewards()

        return self.bee_rewards

    def _process_build_actions(self, actions: Dict[int, Tuple[int, int]]):
        """Process build actions from all bees."""
        build_requests: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        # Collect all build requests
        for bee in self.bees:
            if not bee.alive:
                continue

            _, build_action = actions.get(bee.bee_id, (0, 0))

            if build_action == 0:
                continue

            # Get build location
            bee_y, bee_x = bee.get_grid_position()
            build_y, build_x = get_build_location(bee_y, bee_x, build_action)

            # Validate build location
            if build_y is None or build_x is None:
                continue
            if not (0 <= build_y < self.grid_size and 0 <= build_x < self.grid_size):
                continue

            build_loc = (build_y, build_x)

            # Check if wall already exists
            if self.wall_grid[build_y, build_x] == 1:
                # Penalty for building on completed wall
                self.bee_rewards[bee.bee_id] -= 0.5
                continue

            # Add to build requests
            build_requests[build_loc].append(bee.bee_id)

        # Process build requests
        for build_loc, builder_ids in build_requests.items():
            self._add_to_build_queue(build_loc, builder_ids)

    def _add_to_build_queue(self, location: Tuple[int, int], builder_ids: List[int]):
        """
        Add a build task to the queue or join existing build.

        Args:
            location: (y, x) grid coordinates
            builder_ids: List of bee IDs wanting to build here
        """
        num_builders = len(builder_ids)

        # Calculate build time: 256 / (2^(n-1))
        # 1 bee: 256, 2 bees: 128, 3 bees: 64, ..., 8+ bees: 1
        ticks_required = max(1, self.base_build_ticks // (2 ** (num_builders - 1)))

        if location in self.build_queue:
            # Already being built, add new builders
            existing_builders = self.build_queue[location]['bee_ids']
            existing_builders.update(builder_ids)

            # Recalculate ticks with new number of builders
            total_builders = len(existing_builders)
            new_ticks = max(1, self.base_build_ticks // (2 ** (total_builders - 1)))

            # Update remaining ticks (collaborative speedup)
            self.build_queue[location]['ticks_remaining'] = min(
                self.build_queue[location]['ticks_remaining'],
                new_ticks
            )

            # Reward for joining collaborative build
            for bee_id in builder_ids:
                self.bee_rewards[bee_id] += 0.05

        else:
            # New build task
            self.build_queue[location] = {
                'bee_ids': set(builder_ids),
                'ticks_remaining': ticks_required,
                'initial_ticks': ticks_required
            }

            # Check if adjacent to existing wall (coordination bonus)
            if is_adjacent_to_wall(location[0], location[1], self.wall_grid, include_diagonal=True):
                for bee_id in builder_ids:
                    self.bee_rewards[bee_id] += 0.1

        # Update bee states
        for bee_id in builder_ids:
            self.bees[bee_id].start_building(location, ticks_required)

    def _update_build_queue(self):
        """Update all builds in progress."""
        completed_builds = []

        for location, build_info in self.build_queue.items():
            build_info['ticks_remaining'] -= 1

            # Update bee states
            for bee_id in build_info['bee_ids']:
                bee = self.bees[bee_id]
                is_complete = bee.update_building()

                if is_complete:
                    # Bee finished building, no longer needed at this location
                    pass

            # Check if build is complete
            if build_info['ticks_remaining'] <= 0:
                completed_builds.append(location)

        # Complete builds
        for location in completed_builds:
            self._complete_build(location)

    def _complete_build(self, location: Tuple[int, int]):
        """Complete a build at the specified location."""
        build_info = self.build_queue[location]
        builder_ids = build_info['bee_ids']

        # Place wall
        self.wall_grid[location[0], location[1]] = 1
        self.total_walls_built += 1

        # Reward builders
        base_reward = 0.2
        for bee_id in builder_ids:
            self.bee_rewards[bee_id] += base_reward

        # Remove from queue
        del self.build_queue[location]

        # Clear bee states
        for bee_id in builder_ids:
            self.bees[bee_id].cancel_building()

    def _calculate_rewards(self):
        """Calculate area-based rewards."""
        # Calculate current enclosed area
        current_area, _ = calculate_enclosed_areas(self.wall_grid)
        self.total_enclosed_area = current_area

        # Reward for area increase
        area_increase = current_area - self.previous_enclosed_area
        if area_increase > 0:
            # Distribute reward to all bees (cooperative)
            reward_per_bee = (area_increase * 0.5) / self.num_bees
            for bee in self.bees:
                if bee.alive:
                    self.bee_rewards[bee.bee_id] += reward_per_bee

        self.previous_enclosed_area = current_area

    def get_observation(self, bee_id: int, window_size: int = 8) -> dict:
        """
        Get observation for a specific bee.

        Returns observation dict with:
            - grid_obs: (window_size, window_size, 4) array
                Channel 0: Other bees (1.0 if present, 0.5 if building)
                Channel 1: Walls (1.0 if wall exists)
                Channel 2: Visited areas (decaying memory)
                Channel 3: Build progress (0 to 1)
            - scalar_obs: [x, y, direction, current_task]
        """
        bee = self.bees[bee_id]
        bee_y, bee_x = bee.get_grid_position()

        # Initialize observation channels
        bee_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        wall_channel = self.wall_grid.astype(np.float32)
        visited_channel = bee.visited_grid.copy()
        build_channel = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Fill bee channel
        for other_bee in self.bees:
            if other_bee.bee_id == bee_id or not other_bee.alive:
                continue

            oy, ox = other_bee.get_grid_position()
            if other_bee.is_building():
                bee_channel[oy, ox] = 0.5  # Building
            else:
                bee_channel[oy, ox] = 1.0  # Idle/moving

        # Fill build progress channel
        for location, build_info in self.build_queue.items():
            progress = 1.0 - (build_info['ticks_remaining'] / build_info['initial_ticks'])
            build_channel[location[0], location[1]] = progress

        # Stack channels
        full_grid = np.stack([bee_channel, wall_channel, visited_channel, build_channel], axis=-1)

        # Extract window
        grid_obs = get_observation_window(full_grid, bee_y, bee_x, window_size)

        # Scalar observations (normalized)
        scalar_obs = np.array([
            bee.x / self.grid_size,  # normalized x
            bee.y / self.grid_size,  # normalized y
            bee.direction / self.num_directions,  # normalized direction
            bee.current_task / 3.0  # normalized task (0-3 -> 0-1)
        ], dtype=np.float32)

        return {
            'grid_obs': grid_obs,
            'scalar_obs': scalar_obs
        }

    def get_global_observation(self, resolution: int = 16) -> np.ndarray:
        """
        Get downsampled global observation for high-level policy.

        Args:
            resolution: Size of downsampled grid

        Returns:
            Global observation of shape (resolution, resolution, 3)
                Channel 0: Bee density
                Channel 1: Wall density
                Channel 2: Build activity
        """
        scale = self.grid_size // resolution

        bee_density = np.zeros((resolution, resolution), dtype=np.float32)
        wall_density = np.zeros((resolution, resolution), dtype=np.float32)
        build_density = np.zeros((resolution, resolution), dtype=np.float32)

        # Downsample bee positions
        for bee in self.bees:
            if not bee.alive:
                continue
            y, x = bee.get_grid_position()
            by, bx = min(y // scale, resolution - 1), min(x // scale, resolution - 1)
            bee_density[by, bx] += 1.0

        # Downsample walls
        for i in range(resolution):
            for j in range(resolution):
                y_start, y_end = i * scale, (i + 1) * scale
                x_start, x_end = j * scale, (j + 1) * scale
                wall_density[i, j] = np.mean(self.wall_grid[y_start:y_end, x_start:x_end])

        # Downsample build activity
        for location in self.build_queue.keys():
            y, x = location
            by, bx = min(y // scale, resolution - 1), min(x // scale, resolution - 1)
            build_density[by, bx] += 1.0

        # Normalize
        bee_density /= max(1.0, self.num_bees)
        build_density /= max(1.0, len(self.build_queue))

        return np.stack([bee_density, wall_density, build_density], axis=-1)

    def get_info(self) -> dict:
        """Get info dict for logging."""
        return {
            'total_enclosed_area': self.total_enclosed_area,
            'total_walls_built': self.total_walls_built,
            'active_builds': len(self.build_queue),
            'tick_count': self.tick_count,
            'bees_alive': sum(1 for bee in self.bees if bee.alive),
        }
