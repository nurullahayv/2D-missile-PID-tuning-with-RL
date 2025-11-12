import argparse
import os
import datetime

class Config(object):
    """
    Configurations for HHMARL Training. 
    Mode 0 = Low-level training
    Mode 1 = High-level training
    Mode 2 = Evaluation
    """
    def __init__(self, mode:int):
        self.mode = mode
        parser = argparse.ArgumentParser(description='HHMARL2D Training Config')

        # training mode
        parser.add_argument('--level', type=int, default=1, help='Training Level')
        parser.add_argument('--horizon', type=int, default=500, help='Length of horizon')
        parser.add_argument('--agent_mode', type=str, default="fight", help='Agent mode: Fight or Escape')
        parser.add_argument('--num_agents', type=int, default=2 if mode==0 else 3, help='Number of (trainable) agents')
        parser.add_argument('--num_opps', type=int, default=2 if mode==0 else 3, help='Number of opponents')
        parser.add_argument('--total_num', type=int, default=4 if mode==0 else 6, help='Total number of aircraft')
        parser.add_argument('--hier_opp_fight_ratio', type=int, default=75, help='Opponent fight policy selection probability [in %].')

        # env & training params
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation mode')
        parser.add_argument('--render', type=bool, default=False, help='Render the scene and show live behaviour')
        parser.add_argument('--restore', type=bool, default=False, help='Restore from model')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to stored model')
        parser.add_argument('--log_name', type=str, default=None, help='Experiment Name, defaults to Commander + date & time.')
        parser.add_argument('--log_path', type=str, default=None, help='Full Path to actual trained model')

        parser.add_argument('--gpu', type=float, default=0)
        parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel samplers')
        parser.add_argument('--epochs', type=int, default=10000, help='Number training epochs')
        parser.add_argument('--batch_size', type=int, default=2000 if mode==0 else 1000, help='PPO train batch size')
        parser.add_argument('--mini_batch_size', type=int, default=256, help='PPO train mini batch size')
        parser.add_argument('--map_size', type=float, default=0.3 if mode==0 else 0.5, help='Map size -> *100 = [km]')

        # rewards
        parser.add_argument('--glob_frac', type=float, default=0, help='Fraction of reward sharing')
        parser.add_argument('--rew_scale', type=int, default=1, help='Reward scale')
        parser.add_argument('--esc_dist_rew', type=bool, default=False, help='Activate per-time-step reward for Escape Training.')
        parser.add_argument('--hier_action_assess', type=bool, default=True, help='Give action rewards to guide hierarchical training.')
        parser.add_argument('--friendly_kill', type=bool, default=True, help='Consider friendly kill or not.')
        parser.add_argument('--friendly_punish', type=bool, default=False, help='If friendly kill occurred, if both agents to punish.')

        # eval
        parser.add_argument('--eval_info', type=bool, default=True if mode==2 else False, help='Provide eval statistic in step() function or not. Dont change for evaluation.')
        parser.add_argument('--eval_hl', type=bool, default=True, help='True=evaluation with Commander, False=evaluation of low-level policies.')
        parser.add_argument('--eval_level_ag', type=int, default=5, help='Agent low-level for evaluation.')
        parser.add_argument('--eval_level_opp', type=int, default=4, help='Opponent low-level for evaluation.')
        
        parser.add_argument('--env_config', type=dict, default=None, help='Environment values')
        
        self.args = parser.parse_args()
        self.set_metrics()

    def set_metrics(self):

        #self.args.log_name = f'Commander_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}'
        self.args.log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}' if self.mode == 0 else f'Commander_{self.args.num_agents}_vs_{self.args.num_opps}'
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if not self.args.restore and self.mode==0:
            if self.args.agent_mode == "fight" and os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level-1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')):
                self.args.restore = True
            elif self.args.agent_mode == "escape" and os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}')):
                self.args.restore = True

        if self.args.restore:
            if self.args.restore_path is None:
                if self.mode == 0:
                    try:
                        if self.args.agent_mode=="fight":
                            # take previous pi_fight
                            self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'L{self.args.level-1}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}', 'checkpoint')
                        else:
                            # escape-vs-pi_fight
                            self.args.restore_path = os.path.join(os.path.dirname(__file__), 'results', f'L3_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}', 'checkpoint')
                    except:
                        raise NameError(f'Could not restore previous {self.args.agent_mode} policy. Check restore_path.')
                else:
                    raise NameError('Specify full restore path to Commander Policy.')

        if self.args.agent_mode == "escape" and self.mode==0:
            if not os.path.exists(os.path.join(os.path.dirname(__file__), 'results', f'L3_escape_2-vs-2')):
                self.args.level = 3
            else:
                self.args.level = 5
            self.args.log_name = f'L{self.args.level}_{self.args.agent_mode}_{self.args.num_agents}-vs-{self.args.num_opps}'
            self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        if self.mode == 0:
            horizon_level = {1: 150, 2:200, 3:300, 4:350, 5:400}
            self.args.horizon = horizon_level[self.args.level]
        else:
            self.args.horizon = 500

        if self.mode == 2 and self.args.eval_hl:
            # when incorporating Commander, both teams are on same level.
            self.args.eval_level_ag = self.args.eval_level_opp = 5

        self.args.eval = True if self.args.render else self.args.eval

        self.args.total_num = self.args.num_agents + self.args.num_opps
        self.args.env_config = {"args": self.args}

    @property
    def get_arguments(self):
        return self.args


class BeeColonyConfig(object):
    """
    Configurations for Bee Colony Honeycomb Construction Training.
    Mode 0 = Low-level training (movement + building)
    Mode 1 = High-level training (task selection)
    Mode 2 = Evaluation
    """
    def __init__(self, mode: int):
        self.mode = mode
        parser = argparse.ArgumentParser(description='Bee Colony Training Config')

        # Training mode
        parser.add_argument('--level', type=int, default=1, help='Training Level (1-5 for curriculum)')
        parser.add_argument('--horizon', type=int, default=5000, help='Episode length in steps')
        parser.add_argument('--num_bees', type=int, default=7, help='Number of bees in colony')

        # Environment parameters
        parser.add_argument('--grid_size', type=int, default=500, help='Grid world size (500x500)')
        parser.add_argument('--num_directions', type=int, default=32, help='Number of movement directions')
        parser.add_argument('--window_size', type=int, default=8, help='Observation window size (8x8)')
        parser.add_argument('--global_resolution', type=int, default=16, help='Global observation resolution (16x16)')
        parser.add_argument('--movement_speed', type=float, default=1.0, help='Bee movement speed (grid cells per tick)')
        parser.add_argument('--base_build_ticks', type=int, default=256, help='Base building time for single bee')

        # High-level parameters
        parser.add_argument('--substeps_min', type=int, default=10, help='Min substeps per high-level action')
        parser.add_argument('--substeps_max', type=int, default=20, help='Max substeps per high-level action')
        parser.add_argument('--low_level_policy_path', type=str, default=None, help='Path to trained low-level policy')

        # Training parameters
        parser.add_argument('--gpu', type=float, default=0, help='GPU fraction to use')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of parallel workers')
        parser.add_argument('--epochs', type=int, default=5000, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=4000, help='PPO train batch size')
        parser.add_argument('--mini_batch_size', type=int, default=512, help='PPO mini batch size')
        parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
        parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
        parser.add_argument('--lambda_', type=float, default=0.95, help='GAE lambda')

        # Evaluation and logging
        parser.add_argument('--eval', type=bool, default=True, help='Enable evaluation')
        parser.add_argument('--render', type=bool, default=False, help='Render visualization')
        parser.add_argument('--restore', type=bool, default=False, help='Restore from checkpoint')
        parser.add_argument('--restore_path', type=str, default=None, help='Path to checkpoint')
        parser.add_argument('--log_name', type=str, default=None, help='Experiment name')
        parser.add_argument('--log_path', type=str, default=None, help='Path to logs')
        parser.add_argument('--save_freq', type=int, default=50, help='Checkpoint save frequency (epochs)')

        self.args = parser.parse_args()
        self.set_metrics()

    def set_metrics(self):
        """Set up logging paths and restore settings."""
        # Set log name
        if self.mode == 0:
            self.args.log_name = f'BeeColony_LowLevel_L{self.args.level}_{self.args.num_bees}bees'
        elif self.mode == 1:
            self.args.log_name = f'BeeColony_HighLevel_{self.args.num_bees}bees'
        else:
            self.args.log_name = f'BeeColony_Eval_{self.args.num_bees}bees'

        # Set log path
        self.args.log_path = os.path.join(os.path.dirname(__file__), 'results', self.args.log_name)

        # Auto-restore from previous level (curriculum learning)
        if not self.args.restore and self.mode == 0 and self.args.level > 1:
            prev_path = os.path.join(
                os.path.dirname(__file__),
                'results',
                f'BeeColony_LowLevel_L{self.args.level - 1}_{self.args.num_bees}bees'
            )
            if os.path.exists(prev_path):
                self.args.restore = True
                self.args.restore_path = os.path.join(prev_path, 'checkpoint')

        # For high-level training, need low-level policy
        if self.mode == 1 and self.args.low_level_policy_path is None:
            # Try to find latest low-level policy
            low_level_path = os.path.join(
                os.path.dirname(__file__),
                'results',
                f'BeeColony_LowLevel_L5_{self.args.num_bees}bees',
                'checkpoint'
            )
            if os.path.exists(low_level_path):
                self.args.low_level_policy_path = low_level_path
            else:
                print(f"Warning: Low-level policy not found at {low_level_path}")

        # Environment config
        self.args.env_config = {
            'num_bees': self.args.num_bees,
            'grid_size': self.args.grid_size,
            'num_directions': self.args.num_directions,
            'window_size': self.args.window_size,
            'global_resolution': self.args.global_resolution,
            'horizon': self.args.horizon,
            'movement_speed': self.args.movement_speed,
            'base_build_ticks': self.args.base_build_ticks,
            'substeps_min': self.args.substeps_min,
            'substeps_max': self.args.substeps_max,
            'low_level_policy_path': self.args.low_level_policy_path,
        }

    @property
    def get_arguments(self):
        return self.args

