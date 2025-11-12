"""
Grid utilities for bee colony simulation.
Includes flood fill algorithm for calculating enclosed areas.
"""
import numpy as np
from typing import Tuple, List, Set


def calculate_enclosed_areas(wall_grid: np.ndarray) -> Tuple[int, List[Set[Tuple[int, int]]]]:
    """
    Calculate the total area enclosed by walls using flood fill algorithm.

    Args:
        wall_grid: 2D binary array where 1 = wall, 0 = empty

    Returns:
        total_area: Total number of cells enclosed by walls
        enclosed_regions: List of sets, each containing grid coordinates of an enclosed region
    """
    height, width = wall_grid.shape
    visited = np.zeros_like(wall_grid, dtype=bool)
    enclosed_regions = []

    # Mark all cells reachable from borders as "exterior" (not enclosed)
    exterior = np.zeros_like(wall_grid, dtype=bool)

    def flood_fill_from_border(start_y: int, start_x: int):
        """Flood fill from border to mark all exterior cells."""
        stack = [(start_y, start_x)]

        while stack:
            y, x = stack.pop()

            if y < 0 or y >= height or x < 0 or x >= width:
                continue
            if exterior[y, x] or wall_grid[y, x] == 1:
                continue

            exterior[y, x] = True

            # 4-directional flood (not 8, to properly detect enclosed areas)
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

    # Start flood fill from all border cells
    for x in range(width):
        if wall_grid[0, x] == 0:
            flood_fill_from_border(0, x)
        if wall_grid[height-1, x] == 0:
            flood_fill_from_border(height-1, x)

    for y in range(height):
        if wall_grid[y, 0] == 0:
            flood_fill_from_border(y, 0)
        if wall_grid[y, width-1] == 0:
            flood_fill_from_border(y, width-1)

    # Now find all enclosed regions (cells that are neither walls nor exterior)
    def find_enclosed_region(start_y: int, start_x: int) -> Set[Tuple[int, int]]:
        """Find a single enclosed region using flood fill."""
        region = set()
        stack = [(start_y, start_x)]

        while stack:
            y, x = stack.pop()

            if y < 0 or y >= height or x < 0 or x >= width:
                continue
            if visited[y, x] or wall_grid[y, x] == 1 or exterior[y, x]:
                continue

            visited[y, x] = True
            region.add((y, x))

            # 4-directional for consistency
            stack.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

        return region

    # Find all enclosed regions
    for y in range(height):
        for x in range(width):
            if not visited[y, x] and wall_grid[y, x] == 0 and not exterior[y, x]:
                region = find_enclosed_region(y, x)
                if region:
                    enclosed_regions.append(region)

    total_area = sum(len(region) for region in enclosed_regions)
    return total_area, enclosed_regions


def get_neighbors_8(y: int, x: int, height: int, width: int) -> List[Tuple[int, int]]:
    """Get all 8 neighbors (including diagonal) of a cell."""
    neighbors = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                neighbors.append((ny, nx))
    return neighbors


def get_neighbors_4(y: int, x: int, height: int, width: int) -> List[Tuple[int, int]]:
    """Get 4 orthogonal neighbors of a cell."""
    neighbors = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            neighbors.append((ny, nx))
    return neighbors


def is_adjacent_to_wall(y: int, x: int, wall_grid: np.ndarray, include_diagonal: bool = True) -> bool:
    """
    Check if a position is adjacent to any existing wall.

    Args:
        y, x: Grid coordinates
        wall_grid: 2D binary array where 1 = wall
        include_diagonal: If True, diagonal neighbors count as adjacent

    Returns:
        True if adjacent to at least one wall
    """
    height, width = wall_grid.shape

    if include_diagonal:
        neighbors = get_neighbors_8(y, x, height, width)
    else:
        neighbors = get_neighbors_4(y, x, height, width)

    for ny, nx in neighbors:
        if wall_grid[ny, nx] == 1:
            return True

    return False


def direction_to_delta(direction_idx: int, num_directions: int = 32) -> Tuple[float, float]:
    """
    Convert direction index to movement delta (dy, dx).

    Args:
        direction_idx: Index from 0 to num_directions-1
        num_directions: Total number of discrete directions (default 32)

    Returns:
        (dy, dx): Normalized direction vector
    """
    angle_rad = (direction_idx * 2 * np.pi) / num_directions
    # In grid: y increases downward, x increases rightward
    # angle 0 = right (0°), increases counter-clockwise
    dx = np.cos(angle_rad)
    dy = -np.sin(angle_rad)  # Negative because y increases downward
    return dy, dx


def get_build_location(bee_y: int, bee_x: int, build_action: int) -> Tuple[int, int]:
    """
    Get the grid location for a build action.

    Args:
        bee_y, bee_x: Current bee position
        build_action: 0 = no build, 1-8 = 8 neighbors (top-left clockwise)

    Returns:
        (build_y, build_x): Grid coordinates to build at

    Neighbor mapping:
        1 2 3
        4 B 5
        6 7 8
    """
    if build_action == 0:
        return None, None

    # Map action 1-8 to relative positions (clockwise from top-left)
    neighbor_deltas = [
        (-1, -1),  # 1: top-left
        (-1,  0),  # 2: top
        (-1,  1),  # 3: top-right
        ( 0, -1),  # 4: left
        ( 0,  1),  # 5: right
        ( 1, -1),  # 6: bottom-left
        ( 1,  0),  # 7: bottom
        ( 1,  1),  # 8: bottom-right
    ]

    dy, dx = neighbor_deltas[build_action - 1]
    return bee_y + dy, bee_x + dx


def get_observation_window(grid: np.ndarray, center_y: int, center_x: int,
                           window_size: int = 8) -> np.ndarray:
    """
    Extract a window from the grid centered at (center_y, center_x).
    Pads with zeros if near boundaries.

    Args:
        grid: 2D or 3D array (if 3D, channels are preserved)
        center_y, center_x: Center coordinates
        window_size: Size of the square window

    Returns:
        window: Array of shape (window_size, window_size) or (window_size, window_size, channels)
    """
    half = window_size // 2
    height, width = grid.shape[:2]

    # Calculate window bounds
    y_start = center_y - half
    y_end = center_y + half
    x_start = center_x - half
    x_end = center_x + half

    # Create output window
    if grid.ndim == 2:
        window = np.zeros((window_size, window_size), dtype=grid.dtype)
    else:
        window = np.zeros((window_size, window_size, grid.shape[2]), dtype=grid.dtype)

    # Calculate valid ranges
    src_y_start = max(0, y_start)
    src_y_end = min(height, y_end)
    src_x_start = max(0, x_start)
    src_x_end = min(width, x_end)

    dst_y_start = src_y_start - y_start
    dst_y_end = dst_y_start + (src_y_end - src_y_start)
    dst_x_start = src_x_start - x_start
    dst_x_end = dst_x_start + (src_x_end - src_x_start)

    # Copy valid portion
    window[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        grid[src_y_start:src_y_end, src_x_start:src_x_end]

    return window
