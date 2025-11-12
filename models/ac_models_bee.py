"""
Actor-Critic models for bee colony hierarchical MARL.
"""
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class BeeLowLevelModel(TorchModelV2, nn.Module):
    """
    Low-level policy model for bee agents.
    Uses CNN for spatial observation (8x8 grid) and outputs movement + build actions.
    Includes centralized critic for CTDE.
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Get config
        custom_config = model_config.get("custom_model_config", {})
        self.window_size = custom_config.get("window_size", 8)
        self.num_bees = custom_config.get("num_bees", 7)

        # Cache for value function
        self._critic_input = None

        # Actor network: processes local observation
        # CNN for grid observation (8x8x4)
        self.conv1 = nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4

        # Flattened CNN output + scalar obs
        cnn_output_size = 64 * 4 * 4  # 1024
        scalar_obs_size = 4
        actor_input_size = cnn_output_size + scalar_obs_size

        # Actor hidden layers
        self.actor_fc1 = SlimFC(
            actor_input_size,
            256,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.actor_fc2 = SlimFC(
            256,
            256,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        # Actor output (direction + build action logits)
        self.actor_out = SlimFC(
            256,
            num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.orthogonal_
        )

        # Critic network: processes centralized observation (all bees)
        # Input: flattened observations of all bees
        grid_obs_size = self.window_size * self.window_size * 4
        single_obs_size = grid_obs_size + scalar_obs_size
        critic_input_size = self.num_bees * single_obs_size

        self.critic_fc1 = SlimFC(
            critic_input_size,
            512,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.critic_fc2 = SlimFC(
            512,
            256,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.critic_out = SlimFC(
            256,
            1,
            activation_fn=None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for actor.

        Args:
            input_dict: Dict with 'obs' containing:
                - 'grid_obs': (batch, 8, 8, 4)
                - 'scalar_obs': (batch, 4)
                - 'critic_obs': (batch, centralized_obs_size) [optional, for critic]
        """
        obs = input_dict["obs"]

        # Process grid observation through CNN
        grid_obs = obs["grid_obs"]  # (batch, 8, 8, 4)
        # Transpose to (batch, 4, 8, 8) for PyTorch conv
        grid_obs = grid_obs.permute(0, 3, 1, 2)

        x = torch.relu(self.conv1(grid_obs))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten

        # Concatenate with scalar observation
        scalar_obs = obs["scalar_obs"]
        x = torch.cat([x, scalar_obs], dim=1)

        # Actor forward
        x = self.actor_fc1(x)
        x = self.actor_fc2(x)
        logits = self.actor_out(x)

        # Cache critic input if available
        if "critic_obs" in obs:
            self._critic_input = obs["critic_obs"]

        return logits, []

    @override(ModelV2)
    def value_function(self):
        """
        Compute value function using centralized critic.
        """
        assert self._critic_input is not None, "must call forward first!"

        x = self.critic_fc1(self._critic_input)
        x = self.critic_fc2(x)
        value = self.critic_out(x)

        return torch.reshape(value, [-1])


class BeeHighLevelModel(TorchModelV2, nn.Module):
    """
    High-level policy model for bee task selection.
    Uses CNN for global observation and outputs task selection (Discrete 4).
    Includes centralized critic for CTDE.
    """

    def __init__(self, observation_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Get config
        custom_config = model_config.get("custom_model_config", {})
        self.global_resolution = custom_config.get("global_resolution", 16)
        self.num_bees = custom_config.get("num_bees", 7)

        # Cache for value function
        self._critic_input = None

        # Actor network: processes global observation
        # CNN for global observation (16x16x3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # 8x8 -> 4x4

        # Flattened CNN output + scalar obs
        cnn_output_size = 64 * 4 * 4  # 1024
        scalar_obs_size = 4
        actor_input_size = cnn_output_size + scalar_obs_size

        # Actor hidden layers
        self.actor_fc1 = SlimFC(
            actor_input_size,
            256,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.actor_fc2 = SlimFC(
            256,
            128,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )

        # Actor output (task selection logits)
        self.actor_out = SlimFC(
            128,
            num_outputs,
            activation_fn=None,
            initializer=torch.nn.init.orthogonal_
        )

        # Critic network: processes centralized observation
        global_obs_size = self.global_resolution * self.global_resolution * 3
        single_obs_size = global_obs_size + scalar_obs_size
        critic_input_size = self.num_bees * single_obs_size

        self.critic_fc1 = SlimFC(
            critic_input_size,
            512,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.critic_fc2 = SlimFC(
            512,
            256,
            activation_fn=nn.Tanh,
            initializer=torch.nn.init.orthogonal_
        )
        self.critic_out = SlimFC(
            256,
            1,
            activation_fn=None,
            initializer=torch.nn.init.orthogonal_
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass for actor.

        Args:
            input_dict: Dict with 'obs' containing:
                - 'global_obs': (batch, 16, 16, 3)
                - 'scalar_obs': (batch, 4)
                - 'critic_obs': (batch, centralized_obs_size) [optional]
        """
        obs = input_dict["obs"]

        # Process global observation through CNN
        global_obs = obs["global_obs"]  # (batch, 16, 16, 3)
        # Transpose to (batch, 3, 16, 16) for PyTorch conv
        global_obs = global_obs.permute(0, 3, 1, 2)

        x = torch.relu(self.conv1(global_obs))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)  # Flatten

        # Concatenate with scalar observation
        scalar_obs = obs["scalar_obs"]
        x = torch.cat([x, scalar_obs], dim=1)

        # Actor forward
        x = self.actor_fc1(x)
        x = self.actor_fc2(x)
        logits = self.actor_out(x)

        # Cache critic input if available
        if "critic_obs" in obs:
            self._critic_input = obs["critic_obs"]

        return logits, []

    @override(ModelV2)
    def value_function(self):
        """
        Compute value function using centralized critic.
        """
        assert self._critic_input is not None, "must call forward first!"

        x = self.critic_fc1(self._critic_input)
        x = self.critic_fc2(x)
        value = self.critic_out(x)

        return torch.reshape(value, [-1])
