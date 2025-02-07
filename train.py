"""Train PPO policy using implementation from RSL_RL."""

import os, copy
import os.path as osp
from absl import app, flags
from ml_collections import config_flags

# env
import isaacgym
import torch
from src.envs.trot_env import TrotEnv
from src.envs.trot_env_e2e import TrotEnvE2E

from rsl_rl.runners import OnPolicyRunner
from src.envs import env_wrappers
from src.utilities.git_repo_manager import GitRepoManager

from loguru import logger

# repalce with argparse
training_config = config_flags.DEFINE_config_file(
    "training_config",
    osp.join(osp.dirname(__file__), "configs/training_config/trot.py"),
    "experiment configuration for training",
)
reward_config = config_flags.DEFINE_config_file(
    "reward_config",
    osp.join(osp.dirname(__file__), "configs/reward_config/trot.py"),
    "experiment configuration for reward",
)
flags.DEFINE_integer("num_envs", 4096, "Number of environments")
flags.DEFINE_bool("use_gpu", True, "Use GPU for training")
flags.DEFINE_bool("show_gui", True, "Show GUI")
flags.DEFINE_string("logdir", "logs", "Directory for logs")
flags.DEFINE_string("load_checkpoint", None, "Checkpoint to load")
flags.DEFINE_string("prefix", "test", "Prefix for log directory")
flags.DEFINE_integer("seed", 1, "Random seed")
flags.DEFINE_list("actor_hidden_dims", None, "Actor hidden dimensions")
flags.DEFINE_integer("max_iterations", 600, "Maximum iterations")
flags.DEFINE_string("friction_type", "pyramid", "Friction type")
flags.DEFINE_string("solver_type", "pdhg", "Solver type")
flags.DEFINE_float("env_dt", 0.02, "Environment time step")
args = flags.FLAGS


def get_git_repo_status():
    commit_number = os.popen('git log -1 --pretty=format:"%h"').read().strip("\n")
    branch_name = os.popen("git branch --show-current").read().strip("\n")
    git_diff_info = os.popen("git diff").read()

    return commit_number, branch_name, git_diff_info


def main(argv):
    device = "cuda" if args.use_gpu else "cpu"
    training_config = args.training_config
    reward_config = args.reward_config

    # use flags
    training_config.seed = args.seed
    # if args.actor_hidden_dims is not None:
    #     config.training.policy.actor_hidden_dims = [
    #         int(x) for x in args.actor_hidden_dims
    #     ]
    training_config.runner.max_iterations = args.max_iterations

    reward_config.friction_type = args.friction_type
    reward_config.solver_type = args.solver_type
    reward_config.env_dt = args.env_dt

    my_git_repo_manager = GitRepoManager(args.logdir, prefix=args.prefix)
    # save config.yaml
    os.makedirs(os.path.join(my_git_repo_manager.full_logdir, "config"), exist_ok=True)
    with open(
        os.path.join(my_git_repo_manager.full_logdir, "config/training_config.yaml"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(training_config.to_yaml())
    with open(
        os.path.join(my_git_repo_manager.full_logdir, "config/reward_config.yaml"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(reward_config.to_yaml())

    env = TrotEnv(
        num_envs=args.num_envs,
        device=device,
        config=reward_config,
        show_gui=args.show_gui,
    )
    std_env = env_wrappers.RangeNormalize(env)

    runner = OnPolicyRunner(
        std_env, training_config, my_git_repo_manager.full_logdir, device=device
    )

    if args.load_checkpoint:
        runner.load(args.load_checkpoint)
    runner.learn(
        num_learning_iterations=training_config.runner.max_iterations,
        init_at_random_ep_len=False,
    )


if __name__ == "__main__":
    app.run(main)
