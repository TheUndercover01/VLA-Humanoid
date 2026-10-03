"""rsl_rl PPO settings and checkpoint loading for the RL experts (import after the app is launched)."""
from importlib import metadata

from rsl_rl.runners import OnPolicyRunner

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)


@configclass
class RawPPOCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1500
    save_interval = 100
    experiment_name = "raw"
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


class RslPolicy:
    """Eval-harness policy that runs a trained rsl_rl checkpoint on PandaTaskEnv."""

    def __init__(self, ckpt, env):
        self.wrapper = RslRlVecEnvWrapper(env)
        cfg = handle_deprecated_rsl_rl_cfg(RawPPOCfg(device=env.device), metadata.version("rsl-rl-lib"))
        runner = OnPolicyRunner(self.wrapper, cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(ckpt)
        self.pi = runner.get_inference_policy(device=env.device)

    def reset(self, env):
        pass

    def act(self, env):
        return self.pi(self.wrapper.get_observations()), None

    def update(self, env):
        pass
