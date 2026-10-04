"""PandaTaskEnv driven by a chain of skill commands, rewarded only by what the clips show.

A command is (skill, cube, destination): skill in sim.clips.SKILLS, cube 0 = red / 1 = blue,
destination 0 = green target / 1 = on the other cube / 2 = none. The policy sees the current
command; once the command is done (below) for a few ticks the next one starts. One raw-action
policy learns all four skills.

Training (cfg.max_chain): every episode is a random chain of at most max_chain commands that
starts at home, from a stacked tower, or part-way through a clip replayed in sim (the state bank,
sim/record_state_bank.py). With max_chain = 2 every hand-off between two skills is practised but
no longer task ever is; with 1 each skill is trained alone. Evaluation (sim/eval_chains.py) sets
one fixed chain and scores it with tasks.success, which the reward never uses.

Reward. Every skill is handled by the same code, with numbers from motion/skill_ref.py:
  x = [fingertips - cube, cube - cube_end] in the skill frame (6-D), compared with the clips' mean
  path, which is warped at the command's start so that it starts where the robot is (the end is
  unchanged). cube_end is the commanded destination for skills that move the cube sideways
  (push, place-down), otherwise the cube's start plus the clips' mean displacement (the lift).
  Distance to a path point: z^2 = (|x - point| / sigma)^2 + ((closure - clip closure) / grip_sigma)^2,
  with sigma and grip_sigma the clips' spread at that phase (tolerant where the human varied).
  Progress = the furthest path point reached so far (only forward, only while z < 2).
  "clip":   progress * (1 + exp(-z^2)), in 0..2: nothing for standing still, more for being
            further along and closer to the path
  "sparse": none of the above (ablation: the clips only define when a skill is done)
  Both: + 2 per completed command and + 2 per tick while the last command is done, so finishing a
  command is always worth at least the most the path term can pay.
  Done: x within the clips' done radius of the path's end and the fingers on the same side
  (open / closed) as at the end of the clips.
"""
import numpy as np
import torch

from isaaclab.utils import configclass

from sim.clips import SKILLS
from sim.envs import tasks
from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg
from sim.envs.tasks import CUBE_HALF

REACH, PUSH, PICK, PLACE = (SKILLS.index(s) for s in ["reach", "push", "pick-lift", "place-down"])
TARGET, OTHER, NONE = 0, 1, 2
MAX_CMDS = 9
HOLD = 3                      # ticks a command must stay done before the next starts
STAGE = 2.0                   # = the most the path term can pay per tick
P_BANK, P_STACKED = 0.6, 0.1  # start-state mix for training; the rest start at home


@configclass
class PandaSkillEnvCfg(PandaTaskEnvCfg):
    ref_path: str = "data/processed/skill_ref_standin.npz"
    bank_path: str = "data/processed/state_bank_standin.npz"
    reward: str = "clip"                 # clip | sparse
    max_chain: int = 2
    final_check: str = ""                # eval: success also needs tasks.success(final_check)
    observation_space = 34
    episode_s = 15.0


def rot(v, yaw):
    """Express (n, 3) vectors in frames whose x axes are at angles yaw (n,)."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], -1)


class PandaSkillEnv(PandaTaskEnv):
    cfg: PandaSkillEnvCfg

    def __init__(self, cfg: PandaSkillEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        ref = np.load(cfg.ref_path)
        keys = [s.replace("-", "_") for s in SKILLS]
        t = lambda name: torch.tensor(np.stack([ref[f"{k}_{name}"] for k in keys]), device=dev).float()  # noqa: E731
        self.ref_path, self.ref_sigma, self.ref_grip = t("path"), t("sigma"), t("grip")     # (4, P, 6), (4, P), (4, P)
        self.ref_grip_sigma = t("grip_sigma")
        self.ref_delta, self.ref_radius = t("delta"), t("done_radius")
        self.ref_moves = t("moves").bool()
        self.n_phase = self.ref_path.shape[1]
        self.phase = torch.linspace(0, 1, self.n_phase, device=dev)
        b = dict(np.load(cfg.bank_path))
        good = (np.linalg.norm(b["cube_pos"] - b["clip_cube"], axis=1) < 0.03) & (b["phase"] < 0.8)
        self.bank = {k: v[good] for k, v in b.items() if len(v) == len(good)}
        self.bank_by_skill = [np.nonzero(self.bank["skill"] == k)[0] for k in range(len(SKILLS))]
        self.bank_end = b["clip_end_cube"]
        self.rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 0)
        self.eval_chain = None                 # list of (skill, cube, dest) for every env (eval)

        self.cmds = torch.zeros(n, MAX_CMDS, 3, dtype=torch.long, device=dev)
        self.n_cmd = torch.ones(n, dtype=torch.long, device=dev)
        self.cur = torch.zeros(n, dtype=torch.long, device=dev)
        self.completed = torch.zeros(n, dtype=torch.long, device=dev)
        self.completed_final = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_done = torch.zeros(n, dtype=torch.bool, device=dev)   # the last command has been done (held) at some point
        self.chold = torch.zeros(n, dtype=torch.long, device=dev)
        self.prog = torch.zeros(n, dtype=torch.long, device=dev)
        self.obj0 = torch.zeros(n, 3, device=dev)       # commanded cube at the command's start
        self.end0 = torch.zeros(n, 3, device=dev)       # cube_end for skills without a destination
        self.frame_yaw = torch.zeros(n, device=dev)
        self.warp = torch.zeros(n, 6, device=dev)       # start offset of the robot from the clips' path
        self.need_start = torch.ones(n, dtype=torch.bool, device=dev)
        self.last_episode["completed"] = torch.zeros(n, device=dev)
        self.last_episode["n_cmd"] = torch.zeros(n, device=dev)
        self.last_cmds = torch.zeros_like(self.cmds)
        self.skill_stats = np.zeros((len(SKILLS), 2))   # commanded, completed (training log)
        self.path_terms = torch.zeros(n, 2, device=dev)  # last tick's progress and closeness (tests)

    # ---------- command geometry ----------
    def command(self):
        c = self.cmds[torch.arange(self.num_envs, device=self.device), self.cur]
        return c[:, 0], c[:, 1], c[:, 2]

    def geometry(self, s):
        """skill, commanded cube, cube_end and x = [fingertips - cube, cube - cube_end] in the skill frame."""
        skill, cube, dest = self.command()
        blue = (cube == 1)[:, None]
        act, other = torch.where(blue, s["blue"], s["red"]), torch.where(blue, s["red"], s["blue"])
        tgt = s["target"].clone()
        tgt[:, 2] = CUBE_HALF
        dpos = torch.where((dest == OTHER)[:, None], other + torch.tensor([0, 0, 2 * CUBE_HALF], device=self.device), tgt)
        end = torch.where(self.ref_moves[skill][:, None], dpos, self.end0)
        x = torch.cat([rot(s["tcp"] - act, self.frame_yaw), rot(act - end, self.frame_yaw)], -1)
        return skill, act, end, x

    def start_commands(self, ids):
        """At a command's start: skill frame, cube_end for skills without a destination, path warp."""
        s = self.state()
        skill, act, end, _ = self.geometry(s)
        d = torch.where(self.ref_moves[skill][:, None], end - act, s["tcp"] - act)
        yaw = torch.atan2(d[:, 1], d[:, 0])
        c, s_ = torch.cos(yaw), torch.sin(yaw)
        dl = self.ref_delta[skill]                                  # back from the skill frame to the table
        delta = torch.stack([c * dl[:, 0] - s_ * dl[:, 1], s_ * dl[:, 0] + c * dl[:, 1], dl[:, 2]], -1)
        self.frame_yaw[ids], self.obj0[ids], self.end0[ids] = yaw[ids], act[ids], (act + delta)[ids]
        _, _, _, x = self.geometry(s)
        self.warp[ids] = (x - self.ref_path[skill, 0])[ids]
        self.prog[ids] = 0
        self.chold[ids] = 0
        self.need_start[ids] = False

    # ---------- per tick ----------
    def _tick(self):
        if self.need_start.any():
            self.start_commands(self.need_start.nonzero().squeeze(-1))
        s = self.state()
        self.ticks += 1
        self.was_lifted |= s["red"][:, 2] > tasks.LIFTED
        skill, act, end, x = self.geometry(s)
        # the clips' grip (1 = closed) is what the hand was told to do, not the finger gap: a closed
        # gripper holding a 5 cm cube still has a 4.6 cm gap. Compare with the commanded closure.
        closed = 1 - (self.grip_target / 0.04).clamp(0, 1)

        # progress along the clips' path (warped to start where the robot started)
        path = self.ref_path[skill] + (1 - self.phase)[None, :, None] * self.warp[:, None]   # (n, P, 6)
        z = (((path - x[:, None]).norm(dim=-1) / self.ref_sigma[skill]) ** 2
             + ((closed[:, None] - self.ref_grip[skill]) / self.ref_grip_sigma[skill]) ** 2).sqrt()
        idx = torch.arange(self.n_phase, device=self.device)
        z = torch.where(idx[None] < self.prog[:, None], torch.inf, z)
        zmin, imin = z.min(-1)
        self.prog = torch.where(zmin < 2, imin, self.prog)
        self.path_terms = torch.stack([self.prog / (self.n_phase - 1), torch.exp(-zmin.clamp(max=10) ** 2)], -1)

        # done: the end state of the clips
        end_err = (x - self.ref_path[skill, -1]).norm(dim=-1)
        ok = (end_err < self.ref_radius[skill]) & ((closed - self.ref_grip[skill, -1]).abs() < 0.5)
        self.end_err, self.x = end_err, x                     # for sim/check_skill_reward.py
        last = self.cur == self.n_cmd - 1
        if self.cfg.final_check:
            s["was_lifted"] = self.was_lifted
            ok &= ~last | tasks.success(self.cfg.final_check, s)
        self.chold = torch.where(ok, self.chold + 1, torch.zeros_like(self.chold))
        self.ok = ok & last
        self.done_ok = last & (self.chold >= tasks.SUCCESS_HOLD)
        self.succ_latch |= self.done_ok

        rew = STAGE * self.completed + STAGE * self.done_ok.float()
        if self.cfg.reward == "clip":
            rew = rew + self.path_terms[:, 0] * (1 + self.path_terms[:, 1])
        adv = ~last & (self.chold >= HOLD)
        self.completed += adv.long()
        self.cur += adv.long()
        self.need_start |= adv
        # counts a command once it was done, like an advanced command: without the latch the last command only
        # counted if it was still done when the episode timed out, so one-command (chain1) runs logged 0%
        self.last_done |= self.done_ok
        self.completed_final = self.completed + self.last_done.long()
        self.rew_acc += rew

        for cube in (s["red"], s["blue"]):
            self.knocked |= (cube[:, 2] < -0.05) | (cube[:, :2].abs() > 0.5).any(-1)
        self._metrics(s)
        if self.tick_callback is not None:
            self.tick_callback(self)

    def _get_observations(self):
        s = self.state()
        yaw = self.tcp_yaw()
        skill, cube, dest = self.command()
        _, act, end, _ = self.geometry(s)
        prog = self.prog.float()[:, None] / (self.n_phase - 1) if self.cfg.reward == "clip" \
            else torch.zeros(self.num_envs, 1, device=self.device)
        obs = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08,
                         s["red"], s["blue"], s["target"], act - s["tcp"], end - act,
                         torch.nn.functional.one_hot(skill, 4).float(), torch.nn.functional.one_hot(cube, 2).float(),
                         torch.nn.functional.one_hot(dest, 3).float(), end, prog], dim=-1)
        out = {"policy": obs}
        if self.cfg.cameras:
            out.update(self.images())
        return out

    # ---------- reset: sample a chain and a start state ----------
    def set_chain(self, chain):
        """Evaluation: every env runs this list of (skill name, cube, dest) from home."""
        self.eval_chain = [(SKILLS.index(k), c, d) for k, c, d in chain]

    def _next(self, k, cube, dest):
        if k == REACH:
            return (PICK, cube, NONE)
        if k == PICK:
            return (PLACE, cube, int(self.rng.integers(2)))
        if k == PLACE and dest == OTHER:
            return (REACH, cube, NONE)                    # the one on top: unstack it next
        return (REACH, int(self.rng.integers(2)), NONE)

    def _away(self, avoid, k=1):
        """k points in tasks.REGION at least 12 cm from every point in avoid and from each other."""
        lo, hi = np.array(tasks.REGION[0]), np.array(tasks.REGION[1])
        pts = []
        for _ in range(k):
            for _ in range(200):
                p = self.rng.uniform(lo, hi)
                if all(np.linalg.norm(p - a) > 0.12 for a in list(avoid) + pts):
                    break
            pts.append(p)
        return pts

    def _sample(self):
        """One training episode: (chain, layout (6,), bank frame or -1, stacked top cube or -1)."""
        r = self.rng.random()
        if r < P_BANK:
            k = int(self.rng.integers(len(SKILLS)))
            f = int(self.rng.choice(self.bank_by_skill[k]))
            cube = int(self.rng.integers(2))
            dest = TARGET if k == PUSH else (int(self.rng.integers(2)) if k == PLACE else NONE)
            act = self.bank["cube_pos"][f, :2]
            hand = self.bank["cmd"][f, :2]
            if k == PUSH:                                 # the target where this clip's push ended, give or take 3 cm
                tgt = self.bank_end[self.bank["clip"][f], :2] + self.rng.uniform(-0.03, 0.03, 2)
                tgt = np.clip(tgt, tasks.TARGET_BOX[0], tasks.TARGET_BOX[1])
                oth, = self._away([act, hand, tgt])
            else:
                oth, tgt = self._away([act, hand], 2)
            chain = [(k, cube, dest)]
            frame, top = f, -1
        elif r < P_BANK + P_STACKED:
            cube = int(self.rng.integers(2))
            oth, tgt = self._away([], 2)
            act, chain, frame, top = oth, [(REACH, cube, NONE)], -1, cube
        else:
            cube = int(self.rng.integers(2))
            push = self.rng.random() < 0.3
            lay = tasks.sample_layouts("push" if push else "c1", 1, torch.Generator().manual_seed(
                int(self.rng.integers(1 << 30))))[0].numpy()
            act, oth, tgt = lay[0:2], lay[2:4], lay[4:6]
            chain = [(PUSH, cube, TARGET) if push else (REACH, cube, NONE)]
            frame, top = -1, -1
        while len(chain) < self.cfg.max_chain:
            chain.append(self._next(*chain[-1]))
        red, blue = (act, oth) if cube == 0 else (oth, act)
        return chain, np.concatenate([red, blue, tgt]).astype(np.float32), frame, top

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        ran = self.episode_length_buf[env_ids] > 0
        done_ids = env_ids[ran]
        self.last_episode["completed"][done_ids] = self.completed_final[done_ids].float()
        self.last_episode["n_cmd"][done_ids] = self.n_cmd[done_ids].float()
        self.last_cmds[done_ids] = self.cmds[done_ids]
        if len(done_ids):
            comp = self.completed_final[done_ids].cpu().numpy()
            cm = self.cmds[done_ids, :, 0].cpu().numpy()
            for j in range(self.cfg.max_chain if self.eval_chain is None else len(self.eval_chain)):
                valid = self.n_cmd[done_ids].cpu().numpy() > j
                np.add.at(self.skill_stats[:, 0], cm[valid, j], 1)
                np.add.at(self.skill_stats[:, 1], cm[valid, j], (comp[valid] > j).astype(float))

        ids = env_ids.tolist()
        k = len(ids)
        lay = torch.zeros(k, 6)
        frames, tops = np.full(k, -1), np.full(k, -1)
        chains = []
        for j in range(k):
            if self.eval_chain is not None:
                chains.append(self.eval_chain)
            else:
                ch, lay_j, frames[j], tops[j] = self._sample()
                lay[j] = torch.from_numpy(lay_j)
                chains.append(ch)
        if self.eval_chain is None:
            if self.forced_layouts is None:
                self.forced_layouts = torch.zeros(self.num_envs, 6, device=self.device)
            self.forced_layouts[env_ids] = lay.to(self.device)
        super()._reset_idx(env_ids)
        if "log" in self.extras and len(done_ids):
            rate = self.skill_stats[:, 1] / np.maximum(self.skill_stats[:, 0], 1)
            for j, name in enumerate(SKILLS):
                self.extras["log"][f"Skill/{name}"] = float(rate[j])
            self.skill_stats[:] = 0

        cm = torch.zeros(k, MAX_CMDS, 3, dtype=torch.long)
        for j, ch in enumerate(chains):
            cm[j, :len(ch)] = torch.tensor(ch)
        self.cmds[env_ids] = cm.to(self.device)
        self.n_cmd[env_ids] = torch.tensor([len(ch) for ch in chains], device=self.device)
        self.cur[env_ids] = 0
        self.completed[env_ids] = 0
        self.last_done[env_ids] = False
        self.need_start[env_ids] = True

        dev = self.device
        origin = self.table_origin[env_ids]
        unit = torch.tensor([1.0, 0, 0, 0], device=dev)
        # stacked starts: the commanded cube sits on the other one
        st = np.nonzero(tops >= 0)[0]
        for j in st:
            cube = self.red if tops[j] == 0 else self.blue
            base = lay[j, 2:4] if tops[j] == 0 else lay[j, 0:2]
            pos = torch.cat([base.to(dev), torch.tensor([3 * CUBE_HALF], device=dev)]) + origin[j]
            e = env_ids[j:j + 1]
            cube.write_root_pose_to_sim(torch.cat([pos, unit])[None], env_ids=e)
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=e)
        # clip-state starts: arm and commanded cube from the state bank
        bs = np.nonzero(frames >= 0)[0]
        if len(bs):
            f = frames[bs]
            e = env_ids[torch.tensor(bs, device=dev)]
            q = torch.tensor(self.bank["joint_pos"][f], device=dev)
            self.robot.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=e)
            self.robot.set_joint_position_target(q, env_ids=e)
            self.cmd[e] = torch.tensor(self.bank["cmd"][f], device=dev)
            self.cmd_yaw[e] = torch.tensor(self.bank["cmd_yaw"][f], device=dev)
            self.grip_target[e] = 0.04 * torch.tensor(self.bank["opening"][f], device=dev)
            pose = torch.cat([torch.tensor(self.bank["cube_pos"][f], device=dev) + self.table_origin[e],
                              torch.tensor(self.bank["cube_quat"][f], device=dev)], -1)
            for cube_id, cube in [(0, self.red), (1, self.blue)]:
                m = torch.tensor([chains[j][0][1] == cube_id for j in bs], device=dev)
                if m.any():
                    cube.write_root_pose_to_sim(pose[m], env_ids=e[m])
                    cube.write_root_velocity_to_sim(torch.zeros(int(m.sum()), 6, device=dev), env_ids=e[m])
