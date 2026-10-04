"""PandaTaskEnv with the motion-vocabulary action: one env step = one motion primitive.

Action (n_skills + latent_dim): skill scores, then z. The skill is the argmax of the scores
(rsl_rl only has Gaussian actions, so the categorical choice is an argmax over Gaussian
scores). The vocabulary decoder turns (skill, z) into waypoints relative to the current
command; they are interpolated onto the 10 Hz control ticks and tracked by the same
controller as the raw action, with the same workspace and reach clamps.

Target mode (cfg.target_mode, the "skill + where" action): action = skill scores, an end
point in [-1, 1]^3 mapped onto the workspace box, then z. The decoded motion is warped so it
starts at the current command and ends exactly at that point (each waypoint shifted by its
time fraction of the end error); the clip still sets the path's shape and the gripper timing.
Without it, RL had to find the latent that lands a 2 s open-loop motion on the cube, which
it rarely did precisely enough (C2: 1/100).
"""
import torch

from isaaclab.utils import configclass

from motion.vae import load_vocab
from sim.envs.panda_env import CTRL_SUBSTEPS, PandaTaskEnv, PandaTaskEnvCfg

Z_LIMIT = 3.0          # latent clamp, in prior standard deviations
END_LOW = (-0.25, -0.32, 0.015)    # target-mode end points span the controller's workspace box
END_HIGH = (0.32, 0.32, 0.30)


@configclass
class PandaVocabEnvCfg(PandaTaskEnvCfg):
    vocab_path: str = "data/processed/vocab_standin.pt"
    target_mode: bool = False


class PandaVocabEnv(PandaTaskEnv):
    cfg: PandaVocabEnvCfg

    def __init__(self, cfg: PandaVocabEnvCfg, render_mode=None, **kwargs):
        dec, v = load_vocab(cfg.vocab_path)
        self.skills, self.latent_dim = v["skills"], v["latent_dim"]
        self.n_wp, self.wp_dt = v["n_waypoints"], v["dt"]
        # one primitive = the decoded motion, rounded to whole control ticks
        tick_dt = cfg.sim.dt * CTRL_SUBSTEPS
        self.n_ticks = max(1, round((self.n_wp - 1) * self.wp_dt / tick_dt))
        cfg.decimation = self.n_ticks * CTRL_SUBSTEPS
        cfg.sim.render_interval = CTRL_SUBSTEPS
        cfg.action_space = len(self.skills) + self.latent_dim + (3 if cfg.target_mode else 0)
        super().__init__(cfg, render_mode, **kwargs)
        self.decoder = dec.to(self.device)
        self.skill = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.plan = torch.zeros(self.num_envs, self.n_ticks, 5, device=self.device)
        self.end_low = torch.tensor(END_LOW, device=self.device)
        self.end_high = torch.tensor(END_HIGH, device=self.device)

    def _pre_physics_step(self, actions):
        self.extras["log"] = {}
        self._sub = 0
        self.succ_latch[:] = False
        self.rew_acc[:] = 0.0
        k = len(self.skills)
        self.skill = actions[:, :k].argmax(-1)
        rest = actions[:, k:]
        if self.cfg.target_mode:
            end = self.end_low + (rest[:, :3].clamp(-1, 1) + 1) / 2 * (self.end_high - self.end_low)
            rest = rest[:, 3:]
        z = rest.clamp(-Z_LIMIT, Z_LIMIT)
        with torch.no_grad():
            motion = self.decoder.decode(self.skill, z)                       # (n, W, 5) relative
        if self.cfg.target_mode:              # warp the path so it ends at the chosen point
            frac = torch.linspace(0, 1, self.n_wp, device=self.device)[None, :, None]
            err = (end - self.cmd) - motion[:, -1, :3]
            motion = motion.clone()
            motion[..., :3] = motion[..., :3] + frac * err[:, None]
        # waypoint times 0, dt, ..., (W-1) dt; control ticks at 0.1, 0.2, ... (W-1) dt
        t_wp = torch.arange(self.n_wp, device=self.device) * self.wp_dt
        t_tick = (torch.arange(1, self.n_ticks + 1, device=self.device) * (self.n_wp - 1) * self.wp_dt / self.n_ticks)
        j = torch.searchsorted(t_wp, t_tick).clamp(1, self.n_wp - 1)
        w = ((t_tick - t_wp[j - 1]) / self.wp_dt).clamp(0, 1)[None, :, None]
        plan = motion[:, j - 1] * (1 - w) + motion[:, j] * w                 # (n, T, 5)
        start = torch.cat([self.cmd, self.cmd_yaw[:, None]], dim=-1)
        plan[..., :4] = plan[..., :4] + start[:, None]
        self.plan = plan

    def _block_start(self, block):
        p = self.plan[:, block]
        self._command(p[:, :3].clone(), p[:, 3], 1.0 - p[:, 4])            # vocabulary grip: 1 = closed
