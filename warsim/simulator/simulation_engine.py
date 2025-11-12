"""
Simulation Engine with Decoupled Physics and Rendering
Supports 3 modes: fast, realtime, replay
"""
import numpy as np
from typing import List, Tuple, Dict, Optional
import time
from dataclasses import dataclass, field
from warsim.simulator.missile import Missile
from warsim.simulator.target import Target


@dataclass
class SimulationState:
    """Single simulation state snapshot"""
    step: int
    time: float
    missile_x: float
    missile_y: float
    missile_vx: float
    missile_vy: float
    missile_heading: float
    missile_fuel: float
    missile_active: bool
    target_x: float
    target_y: float
    target_vx: float
    target_vy: float
    target_heading: float
    distance: float
    pid_kp: float
    pid_ki: float
    pid_kd: float


@dataclass
class SimulationHistory:
    """Complete simulation history"""
    states: List[SimulationState] = field(default_factory=list)
    missile_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    target_trajectory: List[Tuple[float, float]] = field(default_factory=list)
    hit: bool = False
    final_distance: float = 0.0
    total_steps: int = 0
    total_time: float = 0.0


class SimulationEngine:
    """
    Decoupled simulation engine
    Physics runs at high frequency (100 Hz)
    Rendering can run at different frequency (60 FPS)
    """

    def __init__(self,
                 missile: Missile,
                 target: Target,
                 map_size: float,
                 hit_radius: float,
                 physics_hz: int = 100,
                 max_steps: int = 50000):
        """
        Initialize simulation engine

        Args:
            missile: Missile instance
            target: Target instance
            map_size: Map size in meters
            hit_radius: Hit detection radius
            physics_hz: Physics update frequency (Hz)
            max_steps: Maximum physics steps
        """
        self.missile = missile
        self.target = target
        self.map_size = map_size
        self.hit_radius = hit_radius
        self.physics_hz = physics_hz
        self.physics_dt = 1.0 / physics_hz
        self.max_steps = max_steps

        self.step = 0
        self.sim_time = 0.0
        self.running = True
        self.hit = False
        self.out_of_bounds = False
        self.out_of_fuel = False

    def reset(self, missile: Missile, target: Target):
        """Reset simulation with new entities"""
        self.missile = missile
        self.target = target
        self.step = 0
        self.sim_time = 0.0
        self.running = True
        self.hit = False
        self.out_of_bounds = False
        self.out_of_fuel = False

    def physics_step(self) -> bool:
        """
        Single physics update step
        Returns True if simulation should continue
        """
        if not self.running:
            return False

        # Update entities
        self.missile.update(self.target.x, self.target.y, self.physics_dt)
        self.target.update(self.physics_dt, missile_position=(self.missile.x, self.missile.y))

        # Update counters
        self.step += 1
        self.sim_time += self.physics_dt

        # Check termination conditions
        distance = self.get_distance()

        if distance < self.hit_radius:
            self.hit = True
            self.running = False
            return False

        if not self.missile.active:
            self.out_of_fuel = True
            self.running = False
            return False

        if self._is_out_of_bounds():
            self.out_of_bounds = True
            self.running = False
            return False

        if self.step >= self.max_steps:
            self.running = False
            return False

        return True

    def get_distance(self) -> float:
        """Calculate distance between missile and target"""
        dx = self.target.x - self.missile.x
        dy = self.target.y - self.missile.y
        return np.sqrt(dx**2 + dy**2)

    def _is_out_of_bounds(self) -> bool:
        """Check if missile is out of bounds"""
        return (self.missile.x < 0 or self.missile.x > self.map_size or
                self.missile.y < 0 or self.missile.y > self.map_size)

    def get_current_state(self) -> SimulationState:
        """Get current simulation state snapshot"""
        return SimulationState(
            step=self.step,
            time=self.sim_time,
            missile_x=self.missile.x,
            missile_y=self.missile.y,
            missile_vx=self.missile.vx,
            missile_vy=self.missile.vy,
            missile_heading=self.missile.heading,
            missile_fuel=self.missile.fuel_remaining,
            missile_active=self.missile.active,
            target_x=self.target.x,
            target_y=self.target.y,
            target_vx=self.target.velocity[0],
            target_vy=self.target.velocity[1],
            target_heading=self.target.heading,
            distance=self.get_distance(),
            pid_kp=self.missile.pid.kp,
            pid_ki=self.missile.pid.ki,
            pid_kd=self.missile.pid.kd
        )

    def simulate_fast(self, record_interval: int = 10) -> SimulationHistory:
        """
        Fast simulation without rendering
        Records state every N steps for replay

        Args:
            record_interval: Record state every N physics steps

        Returns:
            SimulationHistory with recorded states
        """
        history = SimulationHistory()

        print(f"Running fast simulation (Physics: {self.physics_hz} Hz)...")
        start_time = time.time()

        while self.physics_step():
            # Record state periodically
            if self.step % record_interval == 0:
                history.states.append(self.get_current_state())

        # Record final state
        history.states.append(self.get_current_state())

        # Store trajectories
        history.missile_trajectory = self.missile.trajectory.copy()
        history.target_trajectory = self.target.trajectory.copy()
        history.hit = self.hit
        history.final_distance = self.get_distance()
        history.total_steps = self.step
        history.total_time = self.sim_time

        elapsed = time.time() - start_time
        print(f"Fast simulation complete:")
        print(f"  Physics steps: {self.step:,}")
        print(f"  Sim time: {self.sim_time:.2f}s")
        print(f"  Real time: {elapsed:.2f}s")
        print(f"  Speed: {self.sim_time/elapsed:.1f}x real-time")
        print(f"  Result: {'HIT' if self.hit else 'MISS'}")

        return history


class RealtimeSimulation:
    """
    Real-time simulation with decoupled physics and rendering
    Physics: 100 Hz, Rendering: 60 FPS
    """

    def __init__(self,
                 engine: SimulationEngine,
                 renderer,
                 render_fps: int = 60):
        """
        Initialize real-time simulation

        Args:
            engine: SimulationEngine instance
            renderer: PygameRenderer instance
            render_fps: Target rendering FPS
        """
        self.engine = engine
        self.renderer = renderer
        self.render_fps = render_fps
        self.render_interval = 1.0 / render_fps

        # Calculate steps per frame
        self.physics_per_render = max(1, self.engine.physics_hz // render_fps)

    def run(self, mode_info: str = "Real-time Simulation"):
        """
        Run real-time simulation with rendering

        Args:
            mode_info: Info text to display

        Returns:
            SimulationHistory
        """
        history = SimulationHistory()
        last_render_time = time.time()

        print(f"Running real-time simulation:")
        print(f"  Physics: {self.engine.physics_hz} Hz")
        print(f"  Render: {self.render_fps} FPS")
        print(f"  Steps per frame: {self.physics_per_render}")

        frame_count = 0

        while self.engine.running:
            # Physics updates (multiple per frame)
            for _ in range(self.physics_per_render):
                if not self.engine.physics_step():
                    break

            # Render at target FPS
            current_time = time.time()
            if current_time - last_render_time >= self.render_interval:
                state = self.engine.get_current_state()

                success = self.renderer.render_frame(
                    missile_trajectory=self.engine.missile.trajectory,
                    target_trajectory=self.engine.target.trajectory,
                    missile_heading=state.missile_heading,
                    target_heading=state.target_heading,
                    hit_radius=self.engine.hit_radius,
                    step=state.step,
                    distance=state.distance,
                    pid_gains={'kp': state.pid_kp, 'ki': state.pid_ki, 'kd': state.pid_kd},
                    fuel=state.missile_fuel,
                    mode=mode_info,
                    title=f"Real-time Simulation - {self.render_fps} FPS"
                )

                if not success:
                    # User closed window
                    break

                last_render_time = current_time
                frame_count += 1

        # Build history
        history.missile_trajectory = self.engine.missile.trajectory.copy()
        history.target_trajectory = self.engine.target.trajectory.copy()
        history.hit = self.engine.hit
        history.final_distance = self.engine.get_distance()
        history.total_steps = self.engine.step
        history.total_time = self.engine.sim_time

        print(f"\nReal-time simulation complete:")
        print(f"  Physics steps: {self.engine.step:,}")
        print(f"  Rendered frames: {frame_count:,}")
        print(f"  Sim time: {self.engine.sim_time:.2f}s")
        print(f"  Result: {'HIT' if self.engine.hit else 'MISS'}")

        return history


class ReplaySimulation:
    """Replay recorded simulation history"""

    def __init__(self, history: SimulationHistory, renderer):
        """
        Initialize replay

        Args:
            history: SimulationHistory to replay
            renderer: PygameRenderer instance
        """
        self.history = history
        self.renderer = renderer

    def replay(self,
               playback_speed: float = 1.0,
               render_fps: int = 60):
        """
        Replay simulation at specified speed

        Args:
            playback_speed: Playback speed multiplier (1.0 = real-time)
            render_fps: Target rendering FPS
        """
        if len(self.history.states) == 0:
            print("No history to replay!")
            return

        print(f"Replaying simulation:")
        print(f"  Total states: {len(self.history.states)}")
        print(f"  Playback speed: {playback_speed}x")
        print(f"  Render FPS: {render_fps}")

        frame_interval = 1.0 / render_fps / playback_speed
        last_render_time = time.time()

        for i, state in enumerate(self.history.states):
            # Render at target FPS
            current_time = time.time()
            elapsed = current_time - last_render_time

            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

            # Reconstruct trajectories up to this point
            progress = (i + 1) / len(self.history.states)
            traj_len = int(len(self.history.missile_trajectory) * progress)

            missile_traj = self.history.missile_trajectory[:traj_len]
            target_traj = self.history.target_trajectory[:traj_len]

            success = self.renderer.render_frame(
                missile_trajectory=missile_traj,
                target_trajectory=target_traj,
                missile_heading=state.missile_heading,
                target_heading=state.target_heading,
                hit_radius=self.renderer.map_size * 0.005,  # Approximate
                step=state.step,
                distance=state.distance,
                pid_gains={'kp': state.pid_kp, 'ki': state.pid_ki, 'kd': state.pid_kd},
                fuel=state.missile_fuel,
                mode=f"Replay {playback_speed}x",
                title=f"Replay - {playback_speed}x Speed"
            )

            if not success:
                break

            last_render_time = time.time()

        print("Replay complete!")
