"""
Real-time Pygame Renderer for Missile PID Simulation
Dark digital blue background with gradient neon effects
"""
import pygame
import numpy as np
from typing import List, Tuple, Optional
import math


class PygameRenderer:
    """Real-time renderer using Pygame"""

    def __init__(self,
                 map_size: float = 10000.0,
                 window_size: Tuple[int, int] = (1200, 1000),
                 fps: int = 60):
        """
        Initialize pygame renderer

        Args:
            map_size: Size of the simulation map in meters
            window_size: Window size in pixels
            fps: Target frames per second
        """
        pygame.init()

        self.map_size = map_size
        self.window_size = window_size
        self.fps = fps

        # Create window
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Missile PID Control - Real-time Simulation")

        # Clock for FPS control
        self.clock = pygame.time.Clock()

        # Colors - Dark digital blue theme with neon
        self.colors = {
            'bg': (10, 14, 39),           # #0a0e27 - Dark blue background
            'grid': (26, 47, 92),         # #1a2f5c - Subtle blue grid
            'missile': (0, 255, 255),     # #00ffff - Cyan neon
            'target': (255, 0, 255),      # #ff00ff - Magenta neon
            'hit_radius': (255, 0, 102),  # #ff0066 - Hot pink
            'text': (0, 255, 255),        # #00ffff - Cyan
            'panel_bg': (10, 14, 39),     # Panel background
            'panel_border': (0, 255, 255), # Panel border
        }

        # Fonts
        pygame.font.init()
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_small = pygame.font.Font(None, 20)

        # Trail history for gradient effect
        self.max_trail_length = 200

        self.running = True

    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates"""
        screen_x = int((x / self.map_size) * (self.window_size[0] - 100) + 50)
        screen_y = int(self.window_size[1] - 100 - (y / self.map_size) * (self.window_size[1] - 200))
        return screen_x, screen_y

    def draw_grid(self):
        """Draw background grid"""
        # Major grid lines (every 2000m)
        major_interval = self.map_size / 5
        for i in range(6):
            pos = i * major_interval
            x, y_bottom = self.world_to_screen(pos, 0)
            _, y_top = self.world_to_screen(pos, self.map_size)
            pygame.draw.line(self.screen, self.colors['grid'],
                           (x, y_bottom), (x, y_top), 1)

            x_left, y = self.world_to_screen(0, pos)
            x_right, _ = self.world_to_screen(self.map_size, pos)
            pygame.draw.line(self.screen, self.colors['grid'],
                           (x_left, y), (x_right, y), 1)

        # Minor grid lines (every 500m)
        minor_interval = self.map_size / 20
        grid_color_minor = tuple(int(c * 0.5) for c in self.colors['grid'])
        for i in range(21):
            pos = i * minor_interval
            x, y_bottom = self.world_to_screen(pos, 0)
            _, y_top = self.world_to_screen(pos, self.map_size)
            pygame.draw.line(self.screen, grid_color_minor,
                           (x, y_bottom), (x, y_top), 1)

            x_left, y = self.world_to_screen(0, pos)
            x_right, _ = self.world_to_screen(self.map_size, pos)
            pygame.draw.line(self.screen, grid_color_minor,
                           (x_left, y), (x_right, y), 1)

    def draw_gradient_trail(self,
                           trajectory: List[Tuple[float, float]],
                           color: Tuple[int, int, int],
                           max_points: int = 200):
        """Draw trail with gradient effect"""
        if len(trajectory) < 2:
            return

        # Limit trail length for performance
        start_idx = max(0, len(trajectory) - max_points)
        trail = trajectory[start_idx:]

        # Draw trail with fading effect
        n_points = len(trail)
        for i in range(n_points - 1):
            alpha = int(255 * (i + 1) / n_points)  # Fade from 0 to 255

            # Create color with alpha
            fade_color = (*color, alpha)

            x1, y1 = self.world_to_screen(trail[i][0], trail[i][1])
            x2, y2 = self.world_to_screen(trail[i + 1][0], trail[i + 1][1])

            # Draw line with varying thickness for glow effect
            thickness = max(1, int(3 * (i + 1) / n_points))

            # Create glow by drawing multiple lines
            for glow in range(3, 0, -1):
                glow_alpha = alpha // (4 - glow)
                glow_color = (*color, min(255, glow_alpha))

                # Create surface for alpha blending
                surf = pygame.Surface(self.window_size, pygame.SRCALPHA)
                pygame.draw.line(surf, glow_color, (x1, y1), (x2, y2),
                               thickness * glow)
                self.screen.blit(surf, (0, 0))

    def draw_entity(self,
                   x: float,
                   y: float,
                   heading: float,
                   color: Tuple[int, int, int],
                   label: str,
                   size: int = 15):
        """Draw missile or target with glow effect"""
        screen_x, screen_y = self.world_to_screen(x, y)

        # Draw glow layers
        for glow_size in [size * 3, size * 2, size]:
            alpha = 80 if glow_size == size * 3 else (160 if glow_size == size * 2 else 255)
            glow_surf = pygame.Surface((glow_size * 2, glow_size * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow_surf, (*color, alpha),
                             (glow_size, glow_size), glow_size)
            self.screen.blit(glow_surf,
                           (screen_x - glow_size, screen_y - glow_size))

        # Draw direction arrow
        arrow_length = 30
        end_x = screen_x + arrow_length * math.cos(math.radians(heading))
        end_y = screen_y - arrow_length * math.sin(math.radians(heading))

        pygame.draw.line(self.screen, color, (screen_x, screen_y),
                        (end_x, end_y), 3)

        # Arrow head
        arrow_head_length = 10
        arrow_angle = 25

        left_x = end_x - arrow_head_length * math.cos(math.radians(heading - arrow_angle))
        left_y = end_y + arrow_head_length * math.sin(math.radians(heading - arrow_angle))
        right_x = end_x - arrow_head_length * math.cos(math.radians(heading + arrow_angle))
        right_y = end_y + arrow_head_length * math.sin(math.radians(heading + arrow_angle))

        pygame.draw.polygon(self.screen, color,
                          [(end_x, end_y), (left_x, left_y), (right_x, right_y)])

        # Label
        text = self.font_small.render(label, True, color)
        self.screen.blit(text, (screen_x - text.get_width() // 2, screen_y - 30))

    def draw_hit_radius(self, x: float, y: float, radius: float):
        """Draw hit radius circle"""
        screen_x, screen_y = self.world_to_screen(x, y)
        screen_radius = int((radius / self.map_size) * (self.window_size[0] - 100))

        # Draw multiple circles for glow effect
        for i in range(3):
            alpha = 255 - i * 60
            circle_surf = pygame.Surface(self.window_size, pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, (*self.colors['hit_radius'], alpha),
                             (screen_x, screen_y), screen_radius + i * 2, 2)
            self.screen.blit(circle_surf, (0, 0))

    def draw_info_panel(self,
                       step: int,
                       distance: float,
                       pid_gains: dict,
                       fuel: float,
                       mode: str = "Basic PID"):
        """Draw info panel with stats"""
        panel_x = 20
        panel_y = 20
        panel_width = 250
        panel_height = 200

        # Semi-transparent background
        panel_surf = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        pygame.draw.rect(panel_surf, (*self.colors['panel_bg'], 200),
                        (0, 0, panel_width, panel_height))
        pygame.draw.rect(panel_surf, self.colors['panel_border'],
                        (0, 0, panel_width, panel_height), 2)
        self.screen.blit(panel_surf, (panel_x, panel_y))

        # Text
        y_offset = panel_y + 10

        # Mode
        text = self.font_medium.render(mode, True, self.colors['text'])
        self.screen.blit(text, (panel_x + 10, y_offset))
        y_offset += 30

        # Stats
        stats = [
            f"Step: {step}",
            f"Distance: {distance:.1f}m",
            f"PID Gains:",
            f"  Kp: {pid_gains['kp']:.3f}",
            f"  Ki: {pid_gains['ki']:.3f}",
            f"  Kd: {pid_gains['kd']:.3f}",
            f"Fuel: {fuel*100:.1f}%"
        ]

        for stat in stats:
            text = self.font_small.render(stat, True, self.colors['text'])
            self.screen.blit(text, (panel_x + 10, y_offset))
            y_offset += 20

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
                    mode: str = "Basic PID",
                    title: str = "Missile PID Control"):
        """Render complete frame"""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    self.running = False
                    return False

        # Clear screen
        self.screen.fill(self.colors['bg'])

        # Draw grid
        self.draw_grid()

        # Draw trails
        if len(missile_trajectory) > 1:
            self.draw_gradient_trail(missile_trajectory, self.colors['missile'])

        if len(target_trajectory) > 1:
            self.draw_gradient_trail(target_trajectory, self.colors['target'])

        # Draw current positions
        if len(missile_trajectory) > 0:
            mx, my = missile_trajectory[-1]
            self.draw_entity(mx, my, missile_heading,
                           self.colors['missile'], 'MISSILE')

        if len(target_trajectory) > 0:
            tx, ty = target_trajectory[-1]
            self.draw_entity(tx, ty, target_heading,
                           self.colors['target'], 'TARGET')
            # Draw hit radius
            self.draw_hit_radius(tx, ty, hit_radius)

        # Draw info panel
        self.draw_info_panel(step, distance, pid_gains, fuel, mode)

        # Title
        title_text = self.font_large.render(title, True, self.colors['text'])
        self.screen.blit(title_text,
                        (self.window_size[0] // 2 - title_text.get_width() // 2, 10))

        # Instructions
        instructions = self.font_small.render("Press ESC or Q to quit", True,
                                             self.colors['text'])
        self.screen.blit(instructions, (self.window_size[0] - 200,
                                       self.window_size[1] - 30))

        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def close(self):
        """Close pygame window"""
        pygame.quit()

    def is_running(self):
        """Check if window is still open"""
        return self.running
