"""
Advanced Neon Visualization for Missile PID Simulation
Dark digital blue background with gradient neon effects
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.collections import LineCollection
from typing import List, Tuple, Optional
import matplotlib.colors as mcolors


class NeonRenderer:
    """Renderer with dark digital blue background and gradient neon effects"""

    def __init__(self, map_size: float = 10000.0, dpi: int = 150):
        """
        Initialize neon renderer

        Args:
            map_size: Size of the map in meters
            dpi: DPI for rendering
        """
        self.map_size = map_size
        self.dpi = dpi

        # Color scheme - Dark digital blue with neon
        self.bg_color = '#0a0e27'  # Dark blue background
        self.grid_color = '#1a2f5c'  # Subtle blue grid
        self.missile_color = '#00ffff'  # Cyan neon
        self.target_color = '#ff00ff'  # Magenta neon
        self.hit_radius_color = '#ff0066'  # Hot pink

        # Setup figure
        self.fig = None
        self.ax = None
        self.setup_figure()

    def setup_figure(self):
        """Setup the figure with dark theme"""
        plt.style.use('dark_background')

        self.fig, self.ax = plt.subplots(figsize=(12, 12), dpi=self.dpi)
        self.fig.patch.set_facecolor(self.bg_color)
        self.ax.set_facecolor(self.bg_color)

        # Set limits
        self.ax.set_xlim(0, self.map_size)
        self.ax.set_ylim(0, self.map_size)

        # Add grid
        self.ax.grid(True, color=self.grid_color, alpha=0.3, linewidth=0.5,
                    linestyle='-', which='both')
        self.ax.set_axisbelow(True)

        # Add major grid lines
        major_ticks = np.arange(0, self.map_size + 1, self.map_size / 5)
        self.ax.set_xticks(major_ticks)
        self.ax.set_yticks(major_ticks)

        # Add minor grid lines
        minor_ticks = np.arange(0, self.map_size + 1, self.map_size / 20)
        self.ax.set_xticks(minor_ticks, minor=True)
        self.ax.set_yticks(minor_ticks, minor=True)
        self.ax.grid(which='minor', color=self.grid_color, alpha=0.1, linewidth=0.3)

        # Labels
        self.ax.set_xlabel('X Position (m)', fontsize=12, color='#00ffff')
        self.ax.set_ylabel('Y Position (m)', fontsize=12, color='#00ffff')

        # Equal aspect ratio
        self.ax.set_aspect('equal')

    def draw_gradient_trajectory(self,
                                 trajectory: List[Tuple[float, float]],
                                 color: str,
                                 label: str,
                                 linewidth: float = 3.0):
        """
        Draw trajectory with gradient neon effect

        Args:
            trajectory: List of (x, y) positions
            color: Base color for the trajectory
            label: Label for legend
            linewidth: Line width
        """
        if len(trajectory) < 2:
            return

        trajectory = np.array(trajectory)

        # Create segments
        points = trajectory.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create gradient (fade from transparent to full brightness)
        n_points = len(trajectory)
        alphas = np.linspace(0.2, 1.0, n_points - 1)

        # Create color array with gradient alpha
        colors = [mcolors.to_rgba(color, alpha=alpha) for alpha in alphas]

        # Create line collection with gradient
        lc = LineCollection(segments, colors=colors, linewidth=linewidth,
                           label=label, capstyle='round', joinstyle='round')
        self.ax.add_collection(lc)

        # Add glow effect (multiple layers with decreasing alpha)
        for glow_width in [linewidth * 2, linewidth * 3]:
            glow_alphas = alphas * 0.3
            glow_colors = [mcolors.to_rgba(color, alpha=alpha) for alpha in glow_alphas]
            glow_lc = LineCollection(segments, colors=glow_colors,
                                    linewidth=glow_width, capstyle='round',
                                    joinstyle='round', zorder=-1)
            self.ax.add_collection(glow_lc)

    def draw_entity(self,
                   x: float,
                   y: float,
                   heading: float,
                   color: str,
                   label: str,
                   size: float = 100.0):
        """
        Draw missile or target as a glowing point with direction arrow

        Args:
            x, y: Position
            heading: Heading in degrees
            color: Color
            label: Label
            size: Size of the marker
        """
        # Main marker with glow
        for s, alpha in [(size * 3, 0.2), (size * 2, 0.4), (size, 1.0)]:
            self.ax.scatter(x, y, s=s, c=color, alpha=alpha,
                          edgecolors='none', zorder=10)

        # Direction arrow
        arrow_length = self.map_size * 0.03
        dx = arrow_length * np.cos(heading * np.pi / 180)
        dy = arrow_length * np.sin(heading * np.pi / 180)

        arrow = FancyArrow(x, y, dx, dy,
                          width=arrow_length * 0.3,
                          head_width=arrow_length * 0.5,
                          head_length=arrow_length * 0.3,
                          color=color, alpha=0.8,
                          edgecolor='white', linewidth=1,
                          zorder=11)
        self.ax.add_patch(arrow)

        # Label
        self.ax.text(x, y + self.map_size * 0.02, label,
                    color=color, fontsize=10, ha='center',
                    fontweight='bold', zorder=12)

    def draw_hit_radius(self, x: float, y: float, radius: float):
        """
        Draw hit radius circle with pulsing effect

        Args:
            x, y: Center position
            radius: Radius
        """
        # Multiple circles for pulsing glow effect
        for r_mult, alpha, lw in [(1.0, 0.8, 2), (1.2, 0.4, 1.5), (1.4, 0.2, 1)]:
            circle = Circle((x, y), radius * r_mult,
                          fill=False, edgecolor=self.hit_radius_color,
                          linewidth=lw, alpha=alpha, linestyle='--',
                          zorder=5)
            self.ax.add_patch(circle)

    def draw_info_panel(self,
                       step: int,
                       distance: float,
                       pid_gains: dict,
                       fuel: float):
        """
        Draw info panel with current stats

        Args:
            step: Current step
            distance: Distance to target
            pid_gains: Dictionary with kp, ki, kd
            fuel: Remaining fuel (0-1)
        """
        info_text = (
            f"Step: {step}\n"
            f"Distance: {distance:.1f}m\n"
            f"PID Gains:\n"
            f"  Kp: {pid_gains['kp']:.3f}\n"
            f"  Ki: {pid_gains['ki']:.3f}\n"
            f"  Kd: {pid_gains['kd']:.3f}\n"
            f"Fuel: {fuel*100:.1f}%"
        )

        # Draw semi-transparent panel
        panel_x = self.map_size * 0.02
        panel_y = self.map_size * 0.98
        panel_width = self.map_size * 0.15
        panel_height = self.map_size * 0.20

        # Background
        from matplotlib.patches import Rectangle
        rect = Rectangle((panel_x, panel_y - panel_height),
                        panel_width, panel_height,
                        facecolor=self.bg_color, edgecolor=self.missile_color,
                        alpha=0.8, linewidth=2, zorder=20)
        self.ax.add_patch(rect)

        # Text
        self.ax.text(panel_x + panel_width * 0.05,
                    panel_y - panel_height * 0.05,
                    info_text,
                    fontsize=9, color='#00ffff',
                    verticalalignment='top',
                    fontfamily='monospace',
                    zorder=21)

    def render_frame(self,
                    missile_trajectory: List[Tuple[float, float]],
                    target_trajectory: List[Tuple[float, float]],
                    missile_heading: float,
                    target_heading: float,
                    hit_radius: float,
                    step: int,
                    distance: float,
                    pid_gains: dict,
                    fuel: float,
                    title: str = "Missile PID Control Simulation"):
        """
        Render complete frame

        Args:
            missile_trajectory: Missile trajectory
            target_trajectory: Target trajectory
            missile_heading: Missile heading in degrees
            target_heading: Target heading in degrees
            hit_radius: Hit radius
            step: Current step
            distance: Distance to target
            pid_gains: PID gains dict
            fuel: Fuel remaining
            title: Title
        """
        # Clear previous frame
        self.ax.clear()
        self.setup_figure()

        # Draw trajectories with gradient neon effect
        if len(missile_trajectory) > 1:
            self.draw_gradient_trajectory(missile_trajectory,
                                         self.missile_color,
                                         'Missile Trail',
                                         linewidth=3.0)

        if len(target_trajectory) > 1:
            self.draw_gradient_trajectory(target_trajectory,
                                         self.target_color,
                                         'Target Trail',
                                         linewidth=3.0)

        # Draw current positions
        if len(missile_trajectory) > 0:
            mx, my = missile_trajectory[-1]
            self.draw_entity(mx, my, missile_heading,
                           self.missile_color, 'MISSILE', size=150)

        if len(target_trajectory) > 0:
            tx, ty = target_trajectory[-1]
            self.draw_entity(tx, ty, target_heading,
                           self.target_color, 'TARGET', size=150)

            # Draw hit radius
            self.draw_hit_radius(tx, ty, hit_radius)

        # Draw info panel
        self.draw_info_panel(step, distance, pid_gains, fuel)

        # Title
        self.ax.set_title(title, fontsize=16, color='#00ffff',
                         fontweight='bold', pad=20)

        # Legend
        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(loc='upper right', fontsize=10,
                         framealpha=0.8, facecolor=self.bg_color,
                         edgecolor=self.missile_color)

        plt.tight_layout()

    def save_frame(self, filename: str):
        """Save current frame to file"""
        self.fig.savefig(filename, dpi=self.dpi, facecolor=self.bg_color,
                        edgecolor='none', bbox_inches='tight')

    def show(self):
        """Show the plot"""
        plt.pause(0.01)

    def close(self):
        """Close the figure"""
        plt.close(self.fig)
