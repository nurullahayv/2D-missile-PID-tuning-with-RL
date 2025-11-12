"""
Visualization for bee colony honeycomb construction.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from typing import Optional, List, Tuple


class BeePlotter:
    """
    Plotter for visualizing bee colony simulation.
    """

    def __init__(self, grid_size: int = 500, downsample: int = 5, dpi: int = 100):
        """
        Initialize bee plotter.

        Args:
            grid_size: Size of the grid world
            downsample: Downsample factor for visualization (plot every Nth cell)
            dpi: DPI for saved figures
        """
        self.grid_size = grid_size
        self.downsample = downsample
        self.dpi = dpi
        self.vis_size = grid_size // downsample

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(10, 10), dpi=dpi)

    def plot(self, simulator, save_path: Optional[str] = None, show: bool = False):
        """
        Plot the current state of the bee colony simulation.

        Args:
            simulator: BeeSimulator instance
            save_path: Path to save the figure (optional)
            show: Whether to display the figure
        """
        self.ax.clear()

        # Downsample grid for visualization
        vis_wall_grid = simulator.wall_grid[::self.downsample, ::self.downsample]

        # Create base image (walls)
        base_image = np.zeros((self.vis_size, self.vis_size, 3))

        # Draw walls (black)
        base_image[vis_wall_grid == 1] = [0, 0, 0]

        # Draw build queue (yellow/orange based on progress)
        for location, build_info in simulator.build_queue.items():
            y, x = location
            vis_y, vis_x = y // self.downsample, x // self.downsample
            if 0 <= vis_y < self.vis_size and 0 <= vis_x < self.vis_size:
                progress = 1.0 - (build_info['ticks_remaining'] / build_info['initial_ticks'])
                # Yellow to orange gradient
                base_image[vis_y, vis_x] = [1.0, 0.8 * (1 - progress), 0]

        # Calculate enclosed areas for highlighting
        from warsim.utils.grid_utils import calculate_enclosed_areas
        _, enclosed_regions = calculate_enclosed_areas(simulator.wall_grid)

        # Highlight enclosed areas (light green)
        enclosed_mask = np.zeros_like(simulator.wall_grid, dtype=bool)
        for region in enclosed_regions:
            for y, x in region:
                enclosed_mask[y, x] = True

        vis_enclosed_mask = enclosed_mask[::self.downsample, ::self.downsample]
        base_image[vis_enclosed_mask] = [0.7, 1.0, 0.7]  # Light green

        # Display base image
        self.ax.imshow(base_image, origin='upper', interpolation='nearest')

        # Draw bees
        for bee in simulator.bees:
            if not bee.alive:
                continue

            y, x = bee.get_grid_position()
            vis_y, vis_x = y / self.downsample, x / self.downsample

            # Bee color: blue if idle, red if building
            color = 'red' if bee.is_building() else 'blue'

            # Draw bee as a circle
            circle = plt.Circle((vis_x, vis_y), radius=2, color=color, alpha=0.8, zorder=10)
            self.ax.add_patch(circle)

            # Draw direction indicator (small arrow)
            angle = (bee.direction * 2 * np.pi) / simulator.num_directions
            dx_arrow = 3 * np.cos(angle)
            dy_arrow = -3 * np.sin(angle)  # Negative because y increases downward

            self.ax.arrow(vis_x, vis_y, dx_arrow, dy_arrow,
                         head_width=1.5, head_length=1, fc=color, ec=color, alpha=0.8, zorder=11)

            # Draw bee ID
            self.ax.text(vis_x, vis_y - 4, str(bee.bee_id),
                        color='white', fontsize=8, ha='center', va='center',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.7),
                        zorder=12)

        # Add info text
        info = simulator.get_info()
        info_text = (
            f"Step: {info['tick_count']}\n"
            f"Enclosed Area: {info['total_enclosed_area']}\n"
            f"Walls Built: {info['total_walls_built']}\n"
            f"Active Builds: {info['active_builds']}\n"
            f"Bees Alive: {info['bees_alive']}"
        )

        self.ax.text(0.02, 0.98, info_text,
                    transform=self.ax.transAxes,
                    fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='blue', label='Bee (Idle)'),
            Patch(facecolor='red', label='Bee (Building)'),
            Patch(facecolor='black', label='Wall (Complete)'),
            Patch(facecolor='orange', label='Wall (In Progress)'),
            Patch(facecolor='lightgreen', label='Enclosed Area')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=8)

        # Set axis properties
        self.ax.set_xlim(-5, self.vis_size + 5)
        self.ax.set_ylim(self.vis_size + 5, -5)  # Inverted Y axis
        self.ax.set_aspect('equal')
        self.ax.set_title('Bee Colony Honeycomb Construction', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.2)

        # Save or show
        if save_path:
            self.fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight')

        if show:
            plt.pause(0.01)

    def close(self):
        """Close the plotter."""
        plt.close(self.fig)


def visualize_episode(simulator, save_dir: str, interval_steps: int = 100):
    """
    Visualize an episode by saving snapshots at regular intervals.

    Args:
        simulator: BeeSimulator instance
        save_dir: Directory to save snapshots
        interval_steps: Steps between snapshots
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    plotter = BeePlotter(grid_size=simulator.grid_size)

    # Save initial state
    plotter.plot(simulator, save_path=os.path.join(save_dir, f"step_0000.png"))

    print(f"Visualization snapshots saved to: {save_dir}")
    plotter.close()
