import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import UnifiedLogger
from ray.rllib.policy.policy import PolicySpec

from environment.gym_wrapper_rllib_decentralised import GymCoopEnvRLlibDecentralised

# Function: Get nested dictionary values
def get_nested(d, path, default=None):
    cur = d
    for p in path.split("/"):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

# Function: Map each environment agent to its own policy
def policy_mapping_fn(agent_id, *args, **kwargs):
    return "agent_1_policy" if agent_id == "agent_1" else "agent_2_policy"

def train_decentralised(level_name="level_3"):
    EXPERIMENT_NAME = f"ppo_decentralised_{level_name}"

    models_dir = os.path.abspath(f"models/{EXPERIMENT_NAME}")
    log_dir = "logs"

    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    ckpt_root = os.path.join(models_dir, "checkpoints")
    os.makedirs(ckpt_root, exist_ok=True)

    # Fixed starting seed for reproducibility
    TRAIN_SEED = 12345

    # Function: Create the RLlib environment
    def env_creator(env_config):
        return GymCoopEnvRLlibDecentralised(env_config)

    register_env("marl_coop_decentralised", env_creator)

    # Start Ray
    ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)

    # Build a dummy env to get the single-agent spaces for the policies
    dummy_env = GymCoopEnvRLlibDecentralised(
        {
            "level_name": level_name,
            "stack_n": 4,
            "render": False,
            "base_seed": TRAIN_SEED,
            "seed_envs_per_runner": 8,
        }
    )
    single_obs_space = dummy_env.single_observation_space
    single_action_space = dummy_env.single_action_space

    # PPO config
    cfg = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="marl_coop_decentralised",
            env_config={
                "level_name": level_name,
                "stack_n": 4,
                "render": False,
                "base_seed": TRAIN_SEED,
                "seed_envs_per_runner": 8,
            },
            disable_env_checking=True,
        )
        .env_runners(
            num_env_runners=0,
            num_envs_per_env_runner=8,
            rollout_fragment_length=4096,
            batch_mode="truncate_episodes",
        )
        .framework("torch")
        .resources(num_gpus=0)
        .debugging(seed=TRAIN_SEED)
        .multi_agent(
            policies={
                "agent_1_policy": PolicySpec(
                    observation_space=single_obs_space,
                    action_space=single_action_space,
                    config={},
                ),
                "agent_2_policy": PolicySpec(
                    observation_space=single_obs_space,
                    action_space=single_action_space,
                    config={},
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["agent_1_policy", "agent_2_policy"],
            count_steps_by="env_steps",
        )
    )

    # Keep PPO updates simple and stable on CPU
    cfg.simple_optimizer = True

    # Keep raw observations
    cfg._disable_preprocessor_api = True

    # Training settings
    core_training_kwargs = dict(
        lr=2e-4,
        gamma=0.995,
        lambda_=0.97,
        entropy_coeff=0.005,
        train_batch_size=4096 * 8,
        minibatch_size=512,
        clip_param=0.2,
        vf_loss_coeff=0.5,
        grad_clip=0.5,
        use_kl_loss=False,
        kl_coeff=0.0,
        vf_clip_param=1_000_000.0,
        model={
            "fcnet_hiddens": [512, 512, 256],
            "fcnet_activation": "tanh",
            "vf_share_layers": False,
        },
    )

    # Set training epochs
    try:
        cfg = cfg.training(
            num_epochs=12,
            shuffle_batch_per_epoch=True,
            **core_training_kwargs,
        )
    except TypeError:
        legacy_kwargs = dict(core_training_kwargs)
        legacy_kwargs["sgd_minibatch_size"] = legacy_kwargs.pop("minibatch_size")
        cfg = cfg.training(
            num_sgd_iter=12,
            **legacy_kwargs,
        )

    # Additional PPO settings
    cfg.num_epochs = 12
    cfg.num_sgd_iter = 12
    cfg.minibatch_size = 512
    cfg.sgd_minibatch_size = 512
    cfg.rollout_fragment_length = 4096
    cfg.batch_mode = "truncate_episodes"
    cfg.simple_optimizer = True
    cfg._disable_preprocessor_api = True
    cfg.model["vf_share_layers"] = False

    # Log directory
    run_log_dir = os.path.join(log_dir, EXPERIMENT_NAME)
    os.makedirs(run_log_dir, exist_ok=True)

    def logger_creator(config):
        return UnifiedLogger(config, run_log_dir, loggers=None)

    # Build the PPO algorithm
    algo = cfg.build(logger_creator=logger_creator)

    # Sanity check printout
    conf = algo.config
    epochs = getattr(conf, "num_sgd_iter", None)
    if epochs is None:
        epochs = getattr(conf, "num_epochs", None)

    # Number of training timesteps
    TIMESTEPS = 10_000_000

    # Save a checkpoint every 500k env steps
    save_every = 500_000
    next_save = save_every

    while True:
        result = algo.train()

        steps = int(result.get("num_env_steps_sampled_lifetime", 0))

        reward_mean = result.get("episode_reward_mean", None)
        len_mean = result.get("episode_len_mean", None)

        if reward_mean is None:
            reward_mean = get_nested(result, "env_runners/episode_return_mean", 0.0)
        if len_mean is None:
            len_mean = get_nested(result, "env_runners/episode_len_mean", 0.0)

        reward_mean = float(reward_mean) / 2.0 if reward_mean is not None else 0.0
        len_mean = float(len_mean) if len_mean is not None else 0.0

        print("-----------------------------------------------")
        print(f"t = {steps} steps\navg_reward = {reward_mean:.3f}\navg_len = {len_mean:.1f}")
        print("-----------------------------------------------")

        if steps >= next_save:
            while steps >= next_save:
                ckpt_dir = os.path.join(ckpt_root, f"checkpoint_{next_save}")
                os.makedirs(ckpt_dir, exist_ok=True)
                algo.save(checkpoint_dir=ckpt_dir)
                next_save += save_every

        if steps >= TIMESTEPS:
            break

    # Save the final model
    final_dir = os.path.join(ckpt_root, f"checkpoint_{TIMESTEPS}")
    os.makedirs(final_dir, exist_ok=True)
    algo.save(checkpoint_dir=final_dir)

    ray.shutdown()


if __name__ == "__main__":
    train_decentralised(level_name="level_3")