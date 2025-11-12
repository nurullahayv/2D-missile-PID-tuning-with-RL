"""
Low-level environment for bee colony honeycomb construction.
Each bee executes movement and building actions based on current task.
"""
import numpy as np
from typing import Dict, Tuple
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from warsim.simulator.bee_simulator import BeeSimulator


class BeeLowLevelEnv(MultiAgentEnv):
    """
    Low-level multi-agent environment for bee colony.

    Observation Space (per bee):
        - grid_obs: (8, 8, 4) grid window around bee
            Channel 0: Other bees (1.0=idle, 0.5=building)
            Channel 1: Walls (1.0=wall exists)
            Channel 2: Visited areas (decaying memory)
            Channel 3: Build progress (0 to 1)
        - scalar_obs: [x, y, direction, current_task] (normalized)

    Action Space (per bee):
        - MultiDiscrete([32, 9])
            - direction: 0-31 (32 directions)
            - build: 0=no build, 1-8=8 neighbors
    """

    def __init__(self, config: dict):
        """
        Initialize low-level bee environment.

        Args:
            config: Configuration dict with:
                - num_bees: Number of bees (default 7)
                - grid_size: Grid world size (default 500)
                - num_directions: Movement directions (default 32)
                - window_size: Observation window size (default 8)
                - horizon: Max episode steps (default 5000)
                - movement_speed: Bee movement speed (default 1.0)
                - base_build_ticks: Base building time (default 256)
        """
        super().__init__()
        self._skip_env_checking = True

        # Extract config
        self.num_bees = config.get('num_bees', 7)
        self.grid_size = config.get('grid_size', 500)
        self.num_directions = config.get('num_directions', 32)
        self.window_size = config.get('window_size', 8)
        self.horizon = config.get('horizon', 5000)
        self.movement_speed = config.get('movement_speed', 1.0)
        self.base_build_ticks = config.get('base_build_ticks', 256)

        # Initialize simulator
        self.sim = BeeSimulator(
            num_bees=self.num_bees,
            grid_size=self.grid_size,
            num_directions=self.num_directions,
            base_build_ticks=self.base_build_ticks,
            movement_speed=self.movement_speed
        )

        # Agent IDs
        self._agent_ids = set(range(self.num_bees))

        # Define spaces
        self.observation_space = spaces.Dict({
            'grid_obs': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.window_size, self.window_size, 4),
                dtype=np.float32
            ),
            'scalar_obs': spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            )
        })

        self.action_space = spaces.MultiDiscrete([self.num_directions, 9])

        # Tracking
        self.steps = 0
        self.episode_rewards = {i: 0.0 for i in range(self.num_bees)}

    def reset(self, *, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            np.random.seed(seed)

        self.sim.reset()
        self.steps = 0
        self.episode_rewards = {i: 0.0 for i in range(self.num_bees)}

        # Get initial observations
        observations = {}
        for bee_id in self._agent_ids:
            observations[bee_id] = self.sim.get_observation(bee_id, self.window_size)

        infos = {bee_id: {} for bee_id in self._agent_ids}
        infos['__common__'] = self.sim.get_info()

        return observations, infos

    def step(self, action_dict: Dict[int, np.ndarray]):
        """
        Execute one step in the environment.

        Args:
            action_dict: Dict of bee_id -> action array [direction_idx, build_action]

        Returns:
            observations: Dict of bee_id -> observation
            rewards: Dict of bee_id -> reward
            terminateds: Dict of bee_id -> done flag
            truncateds: Dict of bee_id -> truncated flag
            infos: Dict of bee_id -> info dict
        """
        self.steps += 1

        # Convert actions to simulator format
        sim_actions = {}
        for bee_id, action in action_dict.items():
            if isinstance(action, np.ndarray):
                direction_idx = int(action[0])
                build_action = int(action[1])
            else:
                direction_idx, build_action = action
            sim_actions[bee_id] = (direction_idx, build_action)

        # Execute simulation step
        step_rewards = self.sim.do_tick(sim_actions)

        # Collect observations
        observations = {}
        rewards = {}
        terminateds = {}
        truncateds = {}
        infos = {}

        for bee_id in self._agent_ids:
            observations[bee_id] = self.sim.get_observation(bee_id, self.window_size)
            rewards[bee_id] = step_rewards.get(bee_id, 0.0)
            terminateds[bee_id] = False  # Bees don't die in this env
            truncateds[bee_id] = False
            infos[bee_id] = {}

            # Track episode rewards
            self.episode_rewards[bee_id] += rewards[bee_id]

        # Check episode termination
        done = self.steps >= self.horizon
        terminateds['__all__'] = done
        truncateds['__all__'] = done

        # Add common info
        sim_info = self.sim.get_info()
        sim_info['episode_rewards'] = self.episode_rewards.copy()
        infos['__common__'] = sim_info

        return observations, rewards, terminateds, truncateds, infos

    def render(self, mode='human'):
        """Render the environment (TODO: integrate with visualization)."""
        if mode == 'human':
            info = self.sim.get_info()
            print(f"Step: {self.steps}, "
                  f"Enclosed Area: {info['total_enclosed_area']}, "
                  f"Walls: {info['total_walls_built']}, "
                  f"Active Builds: {info['active_builds']}")


class BeeLowLevelEnvRLlib(BeeLowLevelEnv):
    """
    RLlib-compatible wrapper for BeeLowLevelEnv.
    Adds centralized critic observations for CTDE (Centralized Training, Decentralized Execution).
    """

    def __init__(self, config: dict):
        super().__init__(config)

        # Extended observation space for centralized critic
        # Includes all bees' observations concatenated
        grid_obs_size = self.window_size * self.window_size * 4
        scalar_obs_size = 4
        single_obs_size = grid_obs_size + scalar_obs_size

        self.critic_observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_bees * single_obs_size,),
            dtype=np.float32
        )

    def get_centralized_critic_obs(self) -> np.ndarray:
        """
        Get centralized observation for critic (sees all bees).

        Returns:
            Concatenated observations of all bees
        """
        all_obs = []
        for bee_id in range(self.num_bees):
            obs = self.sim.get_observation(bee_id, self.window_size)
            grid_flat = obs['grid_obs'].flatten()
            scalar = obs['scalar_obs']
            all_obs.append(np.concatenate([grid_flat, scalar]))

        return np.concatenate(all_obs)

    def step(self, action_dict: Dict[int, np.ndarray]):
        """
        Execute step and add centralized critic observations to info.
        """
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
