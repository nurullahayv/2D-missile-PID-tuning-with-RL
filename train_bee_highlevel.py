"""
Training script for high-level bee colony policy.
High-level policy learns task selection (explore, build zones).
"""
import os
import time
import shutil
import torch
import numpy as np
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from config import BeeColonyConfig
from envs.env_bee_highlevel import BeeHighLevelEnvRLlib
from models.ac_models_bee import BeeHighLevelModel


def update_logs(args, log_dir, epoch):
    """Copy checkpoints from Ray log to experiment directory."""
    dirs = sorted(Path(log_dir).glob('*/'), key=os.path.getmtime)
    check = ''
    event = ''
    for item in dirs:
        if "checkpoint" in item.name:
            check = str(item)
        if "events" in item.name:
            event = str(item)

    result_dir = os.path.join(args.log_path, 'checkpoint')

    try:
        shutil.rmtree(result_dir)
    except:
        pass

    shutil.copytree(check, result_dir, symlinks=False, dirs_exist_ok=False)
    if event:
        shutil.copy(event, result_dir)


def evaluate(args, algo, env, epoch):
    """Evaluate the trained high-level policy."""
    state, _ = env.reset()
    total_reward = {i: 0.0 for i in range(args.num_bees)}
    done = False
    hl_step = 0

    while not done:
        actions = {}
        for bee_id in state.keys():
            action = algo.compute_single_action(
                observation=state[bee_id],
                policy_id="bee_highlevel_policy",
                explore=False
            )[0]
            actions[bee_id] = action

        state, rewards, terminateds, truncateds, infos = env.step(actions)
        done = terminateds.get("__all__", False) or truncateds.get("__all__", False)

        for bee_id, reward in rewards.items():
            total_reward[bee_id] += reward

        hl_step += 1

    # Get final metrics
    info = infos.get('__common__', {})
    total_rew = sum(total_reward.values())

    print(f"\n[Eval Epoch {epoch}]")
    print(f"  High-Level Steps: {hl_step}")
    print(f"  Low-Level Steps: {info.get('tick_count', 0)}")
    print(f"  Total Reward: {total_rew:.2f}")
    print(f"  Enclosed Area: {info.get('total_enclosed_area', 0)}")
    print(f"  Walls Built: {info.get('total_walls_built', 0)}")

    return total_rew, info


def make_checkpoint(args, algo, log_dir, epoch):
    """Save checkpoint."""
    algo.save()
    update_logs(args, log_dir, epoch)


class BeeHighLevelCallbacks(DefaultCallbacks):
    """Custom callbacks for logging high-level bee colony metrics."""

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        """Log episode metrics."""
        info = episode.last_info_for()
        if info and '__common__' in info:
            common_info = info['__common__']
            episode.custom_metrics["enclosed_area"] = common_info.get('total_enclosed_area', 0)
            episode.custom_metrics["walls_built"] = common_info.get('total_walls_built', 0)
            episode.custom_metrics["high_level_steps"] = common_info.get('high_level_steps', 0)


def main():
    # Load config
    config_obj = BeeColonyConfig(mode=1)  # Mode 1 = high-level training
    args = config_obj.get_arguments

    print(f"\n{'='*60}")
    print(f"Bee Colony High-Level Training")
    print(f"{'='*60}")
    print(f"Num Bees: {args.num_bees}")
    print(f"Grid Size: {args.grid_size}")
    print(f"Horizon: {args.horizon}")
    print(f"Epochs: {args.epochs}")
    print(f"Low-Level Policy: {args.low_level_policy_path}")
    print(f"Log Path: {args.log_path}")
    print(f"{'='*60}\n")

    # Register model
    ModelCatalog.register_custom_model("bee_highlevel_model", BeeHighLevelModel)

    # Define policy spec (shared among all bees)
    policy_spec = PolicySpec(
        policy_class=None,  # Use default
        observation_space=None,  # Auto-inferred
        action_space=None,  # Auto-inferred
        config={
            "model": {
                "custom_model": "bee_highlevel_model",
                "custom_model_config": {
                    "global_resolution": args.global_resolution,
                    "num_bees": args.num_bees,
                }
            }
        }
    )

    # Create policies dict (all bees share the same policy)
    policies = {
        "bee_highlevel_policy": policy_spec
    }

    # Policy mapping function
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "bee_highlevel_policy"

    # Configure PPO
    ppo_config = (
        PPOConfig()
        .environment(
            env=BeeHighLevelEnvRLlib,
            env_config=args.env_config
        )
        .framework("torch")
        .training(
            lr=args.lr,
            gamma=args.gamma,
            lambda_=args.lambda_,
            train_batch_size=args.batch_size,
            sgd_minibatch_size=args.mini_batch_size,
            num_sgd_iter=10,
            clip_param=0.2,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            use_gae=True,
            model={
                "custom_model": "bee_highlevel_model",
                "custom_model_config": {
                    "global_resolution": args.global_resolution,
                    "num_bees": args.num_bees,
                }
            }
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn
        )
        .resources(
            num_gpus=args.gpu,
            num_cpus_per_worker=1
        )
        .rollouts(
            num_rollout_workers=args.num_workers,
            rollout_fragment_length="auto"
        )
        .callbacks(BeeHighLevelCallbacks)
        .debugging(
            log_level="WARN"
        )
    )

    # Build algorithm
    algo = ppo_config.build()

    # Restore if needed
    if args.restore and args.restore_path:
        print(f"Restoring from: {args.restore_path}")
        algo.restore(args.restore_path)

    # Create log directory
    os.makedirs(args.log_path, exist_ok=True)

    # Create evaluation environment
    eval_env = BeeHighLevelEnvRLlib(args.env_config)

    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    log_dir = algo.logdir

    for epoch in range(args.epochs):
        # Train
        result = algo.train()

        # Log progress
        if epoch % 10 == 0:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print(f"  Episode Reward Mean: {result['episode_reward_mean']:.2f}")
            print(f"  Episode Length Mean: {result['episode_len_mean']:.1f}")
            if 'custom_metrics' in result:
                print(f"  Enclosed Area: {result['custom_metrics'].get('enclosed_area_mean', 0):.1f}")
                print(f"  Walls Built: {result['custom_metrics'].get('walls_built_mean', 0):.1f}")
                print(f"  HL Steps: {result['custom_metrics'].get('high_level_steps_mean', 0):.1f}")

        # Evaluate periodically
        if args.eval and epoch % 50 == 0 and epoch > 0:
            evaluate(args, algo, eval_env, epoch)

        # Save checkpoint
        if epoch % args.save_freq == 0 and epoch > 0:
            make_checkpoint(args, algo, log_dir, epoch)

    # Final checkpoint
    print("\nSaving final checkpoint...")
    make_checkpoint(args, algo, log_dir, args.epochs)

    # Final evaluation
    print("\nFinal evaluation...")
    evaluate(args, algo, eval_env, args.epochs)

    print(f"\nTraining complete! Results saved to: {args.log_path}")


if __name__ == "__main__":
    main()
