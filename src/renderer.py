"""
Simple Pygame Renderer
"""
import pygame
import numpy as np


class SimpleRenderer:
    """Simple 2D renderer for missile simulation"""

    def __init__(self, map_size=10000.0, window_size=(1000, 800)):
        pygame.init()
        self.map_size = map_size
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Missile PID Tuning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)

        # Colors
        self.bg_color = (20, 20, 40)
        self.missile_color = (0, 255, 255)
        self.target_color = (255, 50, 50)
        self.trail_color = (100, 100, 150)
        self.text_color = (200, 200, 200)

    def world_to_screen(self, x, y):
        """Convert world coords to screen coords"""
        margin = 50
        screen_w = self.window_size[0] - 2 * margin
        screen_h = self.window_size[1] - 2 * margin

        sx = int((x / self.map_size) * screen_w + margin)
        sy = int(self.window_size[1] - margin - (y / self.map_size) * screen_h)
        return sx, sy

    def render(self, missile, target, hit_radius, step_count):
        """Render current frame"""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                    return False

        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw grid
        margin = 50
        for i in range(5):
            x = margin + i * (self.window_size[0] - 2*margin) // 4
            y = margin + i * (self.window_size[1] - 2*margin) // 4
            pygame.draw.line(self.screen, (40, 40, 60),
                           (x, margin), (x, self.window_size[1]-margin), 1)
            pygame.draw.line(self.screen, (40, 40, 60),
                           (margin, y), (self.window_size[0]-margin, y), 1)

        # Draw trajectories
        if len(missile.trajectory) > 1:
            points = [self.world_to_screen(x, y) for x, y in missile.trajectory[-100:]]
            if len(points) > 1:
                pygame.draw.lines(self.screen, self.trail_color, False, points, 2)

        if len(target.trajectory) > 1:
            points = [self.world_to_screen(x, y) for x, y in target.trajectory[-100:]]
            if len(points) > 1:
                pygame.draw.lines(self.screen, (150, 100, 100), False, points, 2)

        # Draw hit radius
        tx, ty = self.world_to_screen(target.x, target.y)
        radius_screen = int((hit_radius / self.map_size) * (self.window_size[0] - 100))
        pygame.draw.circle(self.screen, (100, 50, 50), (tx, ty), radius_screen, 2)

        # Draw missile
        mx, my = self.world_to_screen(missile.x, missile.y)
        pygame.draw.circle(self.screen, self.missile_color, (mx, my), 8)
        # Direction arrow
        heading = missile.heading
        arrow_len = 20
        ex = mx + arrow_len * np.cos(heading)
        ey = my - arrow_len * np.sin(heading)
        pygame.draw.line(self.screen, self.missile_color, (mx, my), (ex, ey), 3)

        # Draw target
        pygame.draw.circle(self.screen, self.target_color, (tx, ty), 8)
        # Direction arrow
        heading = target.heading
        ex = tx + arrow_len * np.cos(heading)
        ey = ty - arrow_len * np.sin(heading)
        pygame.draw.line(self.screen, self.target_color, (tx, ty), (ex, ey), 3)

        # Draw info panel
        distance = np.linalg.norm(missile.position - target.position)
        info_texts = [
            f"Step: {step_count}",
            f"Distance: {distance:.1f}m",
            f"PID: Kp={missile.pid.kp:.2f} Ki={missile.pid.ki:.2f} Kd={missile.pid.kd:.2f}",
            f"Speed: {missile.speed:.1f}m/s"
        ]

        y_offset = 10
        for text in info_texts:
            surface = self.font.render(text, True, self.text_color)
            self.screen.blit(surface, (10, y_offset))
            y_offset += 25

        # Update display
        pygame.display.flip()
        self.clock.tick(60)

        return True

    def close(self):
        """Close renderer"""
        pygame.quit()
