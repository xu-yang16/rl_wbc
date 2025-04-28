"""Configuration for running PPO using the Pointmass Env"""

from ml_collections import ConfigDict


def get_training_config():
    """
    include:
    1. policy_config
    2. alg_config
    3. runner_config
    """
    config = ConfigDict()
    config.seed = 0

    policy_config = ConfigDict()
    policy_config.init_noise_std = 0.5
    policy_config.actor_hidden_dims = [512, 256, 128]
    policy_config.critic_hidden_dims = [512, 256, 128]
    policy_config.activation = "elu"
    config.policy = policy_config

    encoder_config = ConfigDict()
    encoder_config.num_output_dim = 3
    encoder_config.hidden_dims = [256, 128]
    encoder_config.activation = "elu"
    config.encoder = encoder_config

    alg_config = ConfigDict()
    alg_config.value_loss_coef = 1.0
    alg_config.use_clipped_value_loss = True
    alg_config.clip_param = 0.2
    alg_config.entropy_coef = 0.01
    alg_config.num_learning_epochs = 5
    alg_config.num_mini_batches = 4
    alg_config.learning_rate = 1e-3
    alg_config.schedule = "adaptive"
    alg_config.gamma = 0.99
    alg_config.lam = 0.95
    alg_config.desired_kl = 0.01
    alg_config.max_grad_norm = 1.0
    config.algorithm = alg_config

    runner_config = ConfigDict()
    runner_config.policy_class_name = "ActorCritic"
    runner_config.algorithm_class_name = "PPO"
    runner_config.num_steps_per_env = 24
    runner_config.save_interval = 100
    runner_config.experiment_name = "trot_pdhg"
    runner_config.max_iterations = 1000
    # runner_config.resume = False
    # runner_config.load_run = -1
    # runner_config.checkpoint = -1
    # runner_config.resume_path = None
    config.runner = runner_config
    return config


def get_config():
    return get_training_config()
