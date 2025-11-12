"""
High-level environment for bee colony honeycomb construction.
Each bee selects high-level tasks (explore, build zones) and executes them using low-level policies.
"""
import numpy as np
from typing import Dict, Tuple, Optional
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import Policy
import os

from warsim.simulator.bee_simulator import BeeSimulator


class BeeHighLevelEnv(MultiAgentEnv):
    """
    High-level multi-agent environment for bee colony.
    Uses temporal abstraction: high-level policy selects tasks every N steps,
    low-level policy executes actions for those tasks.

    Observation Space (per bee):
        - global_obs: (16, 16, 3) downsampled global view
            Channel 0: Bee density
            Channel 1: Wall density
            Channel 2: Build activity
        - scalar_obs: [x, y, direction, prev_task] (normalized)

    Action Space (per bee):
        - Discrete(4): Task selection
            0: Explore freely
            1: Build zone NW (northwest)
            2: Build zone NE (northeast)
            3: Build zone S (south)
    """

    def __init__(self, config: dict):
        """
        Initialize high-level bee environment.

        Args:
            config: Configuration dict with:
                - num_bees: Number of bees (default 7)
                - grid_size: Grid world size (default 500)
                - num_directions: Movement directions (default 32)
                - global_resolution: Global observation resolution (default 16)
                - horizon: Max episode steps (default 5000)
                - substeps_min: Min steps per high-level action (default 10)
                - substeps_max: Max steps per high-level action (default 20)
                - low_level_policy_path: Path to trained low-level policy checkpoint
        """
        super().__init__()
        self._skip_env_checking = True

        # Extract config
        self.num_bees = config.get('num_bees', 7)
        self.grid_size = config.get('grid_size', 500)
        self.num_directions = config.get('num_directions', 32)
        self.global_resolution = config.get('global_resolution', 16)
        self.horizon = config.get('horizon', 5000)
        self.substeps_min = config.get('substeps_min', 10)
        self.substeps_max = config.get('substeps_max', 20)
        self.window_size = config.get('window_size', 8)

        # Low-level policy path
        self.low_level_policy_path = config.get('low_level_policy_path', None)
        self.low_level_policy: Optional[Policy] = None

        # Initialize simulator
        self.sim = BeeSimulator(
            num_bees=self.num_bees,
            grid_size=self.grid_size,
            num_directions=self.num_directions,
            base_build_ticks=config.get('base_build_ticks', 256),
            movement_speed=config.get('movement_speed', 1.0)
        )

        # Agent IDs
        self._agent_ids = set(range(self.num_bees))

        # Define spaces
        self.observation_space = spaces.Dict({
            'global_obs': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.global_resolution, self.global_resolution, 3),
                dtype=np.float32
            ),
            'scalar_obs': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.Discrete(4)  # 4 tasks

        # Tracking
        self.steps = 0
        self.high_level_steps = 0
        self.episode_rewards = {i: 0.0 for i in range(self.num_bees)}
        self.current_tasks = {i: 0 for i in range(self.num_bees)}

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self.sim.reset()
        self.steps = 0
        self.high_level_steps = 0
        self.episode_rewards = {i: 0.0 for i in range(self.num_bees)}
        self.current_tasks = {i: 0 for i in range(self.num_bees)}

        # Get initial observations
        observations = {}
        for bee_id in self._agent_ids:
            observations[bee_id] = self._get_high_level_observation(bee_id)

        infos = {bee_id: {} for bee_id in self._agent_ids}
        infos['__common__'] = self.sim.get_info()

        return observations, infos

    def _get_high_level_observation(self, bee_id: int) -> Dict[str, np.ndarray]:
        """
        Get high-level observation for a bee.

        Returns:
            Dict with 'global_obs' and 'scalar_obs'
        """
        # Global observation
        global_obs = self.sim.get_global_observation(self.global_resolution)

        # Scalar observation
        bee = self.sim.bees[bee_id]
        scalar_obs = np.array([
            bee.x / self.grid_size,
            bee.y / self.grid_size,
            bee.direction / self.num_directions,
            self.current_tasks[bee_id] / 3.0  # Normalize task ID
        ], dtype=np.float32)

        return {
            'global_obs': global_obs,
            'scalar_obs': scalar_obs
        }

    def _execute_low_level_policy(self, task_dict: Dict[int, int], num_steps: int) -> float:
        """
        Execute low-level policy for the assigned tasks.

        Args:
            task_dict: Dict of bee_id -> task_id
            num_steps: Number of low-level steps to execute

        Returns:
            Total reward accumulated during execution
        """
        total_reward = 0.0

        for _ in range(num_steps):
            if self.steps >= self.horizon:
                break

            # Get low-level actions for each bee based on task
            low_level_actions = {}
            for bee_id, task_id in task_dict.items():
                # Update bee's current task
                self.sim.bees[bee_id].current_task = task_id

                # Get low-level observation
                obs = self.sim.get_observation(bee_id, self.window_size)

                # Get action from low-level policy (or use heuristic if policy not loaded)
                if self.low_level_policy is not None:
                    # Use trained low-level policy
                    action = self.low_level_policy.compute_single_action(obs)[0]
                else:
                    # Use simple heuristic policy
                    action = self._heuristic_low_level_action(bee_id, task_id, obs)

                low_level_actions[bee_id] = action

            # Execute simulation step
            step_rewards = self.sim.do_tick(low_level_actions)

            # Accumulate rewards
            for bee_id in task_dict.keys():
                reward = step_rewards.get(bee_id, 0.0)
                total_reward += reward
                self.episode_rewards[bee_id] += reward

            self.steps += 1

        return total_reward

    def _heuristic_low_level_action(self, bee_id: int, task_id: int,
                                    obs: Dict[str, np.ndarray]) -> Tuple[int, int]:
        """
        Simple heuristic policy for low-level actions (used during high-level training).

        Args:
            bee_id: Bee identifier
            task_id: Current task (0=explore, 1=build NW, 2=build NE, 3=build S)
            obs: Low-level observation

        Returns:
            (direction_idx, build_action)
        """
        bee = self.sim.bees[bee_id]
        bee_y, bee_x = bee.get_grid_position()

        # Determine target based on task
        if task_id == 0:
            # Explore: random walk with tendency to move away from visited areas
            direction_idx = np.random.randint(0, self.num_directions)
        elif task_id == 1:
            # Build zone NW (northwest)
            target_y, target_x = self.grid_size * 0.25, self.grid_size * 0.25
            angle = np.arctan2(target_y - bee_y, target_x - bee_x)
            direction_idx = int((angle / (2 * np.pi)) * self.num_directions) % self.num_directions
        elif task_id == 2:
            # Build zone NE (northeast)
            target_y, target_x = self.grid_size * 0.25, self.grid_size * 0.75
            angle = np.arctan2(target_y - bee_y, target_x - bee_x)
            direction_idx = int((angle / (2 * np.pi)) * self.num_directions) % self.num_directions
        else:  # task_id == 3
            # Build zone S (south)
            target_y, target_x = self.grid_size * 0.75, self.grid_size * 0.5
            angle = np.arctan2(target_y - bee_y, target_x - bee_x)
            direction_idx = int((angle / (2 * np.pi)) * self.num_directions) % self.num_directions

        # Build action: build if in target zone and near walls
        build_action = 0
        if task_id > 0:  # Build tasks
            wall_channel = obs['grid_obs'][:, :, 1]
            if np.sum(wall_channel) > 0:  # Walls nearby
                # Find best neighbor to build
                wall_neighbors = []
                for action in range(1, 9):
                    # Check if neighbor is adjacent to wall
                    from warsim.utils.grid_utils import get_build_location
                    build_y, build_x = get_build_location(bee_y, bee_x, action)
                    if build_y is not None and 0 <= build_y < self.grid_size and 0 <= build_x < self.grid_size:
                        if self.sim.wall_grid[build_y, build_x] == 0:  # Not already wall
                            # Check adjacency in local observation
                            local_y, local_x = self.window_size // 2, self.window_size // 2
                            dy, dx = build_y - bee_y, build_x - bee_x
                            check_y, check_x = local_y + dy, local_x + dx
                            if 0 <= check_y < self.window_size and 0 <= check_x < self.window_size:
                                # Check if adjacent to wall
                                for ny in range(max(0, check_y - 1), min(self.window_size, check_y + 2)):
                                    for nx in range(max(0, check_x - 1), min(self.window_size, check_x + 2)):
                                        if wall_channel[ny, nx] > 0:
                                            wall_neighbors.append(action)
                                            break

                if wall_neighbors:
                    build_action = np.random.choice(wall_neighbors)

        return (direction_idx, build_action)

    def step(self, action_dict: Dict[int, int]):
        """
        Execute one high-level step (multiple low-level steps).

        Args:
            action_dict: Dict of bee_id -> task_id (0-3)

        Returns:
            observations, rewards, terminateds, truncateds, infos
        """
        self.high_level_steps += 1

        # Update current tasks
        for bee_id, task_id in action_dict.items():
            self.current_tasks[bee_id] = task_id

        # Execute low-level policy for random number of substeps
        num_substeps = np.random.randint(self.substeps_min, self.substeps_max + 1)
        total_reward = self._execute_low_level_policy(action_dict, num_substeps)

        # Distribute reward among bees
        reward_per_bee = total_reward / self.num_bees

        # Collect observations
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for bee_id in self._agent_ids:
            observations[bee_id] = self._get_high_level_observation(bee_id)
            rewards[bee_id] = reward_per_bee  # Cooperative reward
            terminateds[bee_id] = False
            truncateds[bee_id] = False
            infos[bee_id] = {'substeps_executed': num_substeps}

        # Check episode termination
        done = self.steps >= self.horizon
        terminateds['__all__'] = done
        truncateds['__all__'] = done

        # Add common info
        sim_info = self.sim.get_info()
        sim_info['episode_rewards'] = self.episode_rewards.copy()
        sim_info['high_level_steps'] = self.high_level_steps
        infos['__common__'] = sim_info

        return observations, rewards, terminateds, truncateds, infos

    def render(self, mode='human'):
        """Render the environment."""
        if mode == 'human':
            info = self.sim.get_info()
            print(f"HL Step: {self.high_level_steps}, LL Step: {self.steps}, "
                  f"Enclosed Area: {info['total_enclosed_area']}, "
                  f"Walls: {info['total_walls_built']}")


class BeeHighLevelEnvRLlib(BeeHighLevelEnv):
    """
    RLlib-compatible wrapper for BeeHighLevelEnv.
    Adds centralized critic observations for CTDE.
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Extended observation space for centralized critic
        global_obs_size = self.global_resolution * self.global_resolution * 3
        scalar_obs_size = 4
        single_obs_size = global_obs_size + scalar_obs_size

        self.critic_observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_bees * single_obs_size,),
            dtype=np.float32
        )

    def get_centralized_critic_obs(self) -> np.ndarray:
        """Get centralized observation for critic."""
        all_obs = []
        for bee_id in range(self.num_bees):
            obs = self._get_high_level_observation(bee_id)
            global_flat = obs['global_obs'].flatten()
            scalar = obs['scalar_obs']
            all_obs.append(np.concatenate([global_flat, scalar]))

        return np.concatenate(all_obs)

    def step(self, action_dict: Dict[int, int]):
        """Execute step and add centralized critic observations."""
        observations, rewards, terminateds, truncateds, infos = super().step(action_dict)

        # Add centralized observation for critic
        centralized_obs = self.get_centralized_critic_obs()
        for bee_id in self._agent_ids:
            infos[bee_id]['critic_obs'] = centralized_obs

        return observations, rewards, terminateds, truncateds, infos

    def reset(self, *, seed=None, options=None):
        """Reset and add centralized critic observations."""
        observations, infos = super().reset(seed=seed, options=options)

        # Add centralized observation for critic
        centralized_obs = self.get_centralized_critic_obs()
        for bee_id in self._agent_ids:
            infos[bee_id]['critic_obs'] = centralized_obs

        return observations, infos
