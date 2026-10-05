"""PandaTaskEnv driven by a chain of object commands, rewarded by keypoint waypoints from the clips.

Only objects are scored, never the hand. A command is (skill, cube, destination): skill push,
pick-lift or place-down; cube 0 = red / 1 = blue; destination 0 = green target pad / 1 = on the other
cube / 2 = none (pick-lift). Each command has a goal pose for the cube: the destination (target pad,
or the top of the other cube), or for pick-lift the cube's start plus the clips' lift. The goal keeps
the cube's yaw at the start of the command: the clips carry, lift and push the cube without turning it.

Reward (motion/keypoint_ref.py makes the waypoints from the clips; same code for every skill):
  The cube's 8 corners are compared with the clips' waypoints: corner offsets from the goal pose, laid
  over this command's goal and travel direction, stretched per axis so its start distance matches this
  cube's (a short carry keeps the clips' shape at a smaller size), and any remaining start offset faded
  out towards the goal (the last waypoint is unchanged). Distance = mean over the cube's corners of the distance to the
  nearest waypoint corner, so a cube turned a quarter (or flipped) is the same cube, while one turned
  45 degrees is ~2 cm off. A waypoint counts as reached when that distance is
  below its tolerance, 2 x the clips' spread there (at least 1 cm); reaching a later one counts the
  earlier ones too.
  potential = waypoints reached + exp(-(d_next / tol_next)^2)
  per tick: change in potential - 0.02 (time), + 5 when a command is done, + 0.1 per tick while the
  last command is done. Done = last waypoint reached, the cube within DONE_TOL + the clips' end turn of
  the goal pose (push turns the cube in the clips, lift and place do not), and at rest, for 0.3 s.
Moving the potential's difference makes standing still worth nothing and going back cost what going
forward paid. Nothing here looks at the gripper; a cube resting at its goal is done however it got there.

Training starts (cfg.max_chain commands per episode): 60% from the clips' states replayed in sim
(sim/record_state_bank.py; reach frames start a pick-lift with the hand on its way), 10% from a
stacked tower, 30% at home. Evaluation (sim/eval_chains.py --keypoints) sets one fixed chain.
"""
import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat

from sim.clips import SKILLS
from sim.envs import tasks
from sim.envs.panda_env import PandaTaskEnv, PandaTaskEnvCfg
from sim.envs.tasks import CUBE_HALF

REACH, PUSH, PICK, PLACE = (SKILLS.index(s) for s in ["reach", "push", "pick-lift", "place-down"])
TARGET, OTHER, NONE = 0, 1, 2
MAX_CMDS = 7
HOLD = 3                     # ticks the done check must hold
WP_TOL_MIN = 0.01            # m, mean corner distance
DONE_TOL = 0.02              # m, mean corner distance from the goal pose: "the cube is there" (40% of its width)
REST = 0.03                  # m/s
REST_HOLD = 0.1              # m/s, with the hold fix: reachable for a held cube with exploration noise
STRETCH_MIN = 0.02           # m: the template is stretched along an axis only if the clips start this far from the goal on it
TIME_COST, DONE_BONUS, HOLD_BONUS = 0.02, 5.0, 0.1
HOLD_W = 0.1                 # hold fix: at most this per tick for staying at the goal (below the done bonus + next skill)
P_BANK, P_STACKED = 0.6, 0.1
LOCAL = CUBE_HALF * torch.tensor([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=torch.float32)


@configclass
class PandaKeypointEnvCfg(PandaTaskEnvCfg):
    ref_path: str = "data/processed/keypoint_ref_standin.npz"
    bank_path: str = "data/processed/state_bank_standin_v2.npz"
    max_chain: int = 2
    final_check: str = ""                 # eval: success also needs tasks.success(final_check)
    # hold fix: once every waypoint is reached, + HOLD_W x closeness to the goal per tick, and "at rest" is
    # REST_HOLD (1 cm per tick) instead of REST. Without it the policy lifted the cube along all waypoints but
    # never held it still in the air, so pick-lift was never done (PROGRESS.md, 4 Oct); with no speed check at
    # all, a cube slid through the goal while still held counted as placed
    hold_fix: bool = False
    # v3 (22:10, user): rest_speed overrides the at-rest speed (m/s, 0 = REST / REST_HOLD); release: done also
    # needs the gripper command to end as the clips' hand ends the skill (place-down open, pick-lift closed);
    # time_cost per tick; knock_penalty: a cube off the table ends the episode and costs the time of the ticks
    # left, so ending early never saves time cost
    rest_speed: float = 0.0
    release: bool = False
    time_cost: float = TIME_COST
    knock_penalty: bool = False
    # action_rate (5 Oct, user): - action_rate x sum over dx..dyaw of (a_t - a_{t-1})^2 per step. The RL teacher's
    # actions were bang-bang (~30% of steps flip sign), which a VLA cannot imitate; this makes the teacher smooth.
    action_rate: float = 0.0
    observation_space = 76
    episode_s = 15.0


def rot_z(v, yaw):
    """Rotate (n, ..., 3) vectors by yaw (n,) about z."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    while c.dim() < v.dim() - 1:
        c, s = c[..., None], s[..., None]
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], -1)


def yaw_of_quat(q):
    r = matrix_from_quat(q)
    return torch.atan2(r[:, 1, 0], r[:, 0, 0])


class PandaKeypointEnv(PandaTaskEnv):
    cfg: PandaKeypointEnvCfg

    def __init__(self, cfg: PandaKeypointEnvCfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        ref = np.load(cfg.ref_path)
        keys = {PUSH: "push", PICK: "pick_lift", PLACE: "place_down"}
        nwp = int(ref["n_wp"])
        self.n_wp = nwp
        self.wp = torch.zeros(len(SKILLS), nwp, 8, 3, device=dev)
        self.wp_tol = torch.ones(len(SKILLS), nwp, device=dev)
        self.delta = torch.zeros(len(SKILLS), 3, device=dev)
        self.moves = torch.zeros(len(SKILLS), dtype=torch.bool, device=dev)
        self.done_tol = torch.full((len(SKILLS),), DONE_TOL, device=dev)
        self.end_grip = torch.zeros(len(SKILLS), device=dev)
        for k, name in keys.items():
            self.wp[k] = torch.tensor(ref[f"{name}_waypoints"], device=dev)
            self.wp_tol[k] = torch.tensor(ref[f"{name}_spread"], device=dev).mul(2).clamp(min=WP_TOL_MIN)
            self.delta[k] = torch.tensor(ref[f"{name}_delta"], device=dev)
            self.moves[k] = bool(ref[f"{name}_moves"])
            self.end_grip[k] = float(ref[f"{name}_end_grip"]) if f"{name}_end_grip" in ref.files else 0.0
            self.done_tol[k] = DONE_TOL + float(ref[f"{name}_end_turn"])
            # the last waypoint is the goal pose: same tolerance as done (the clips' spread there is measured
            # against each clip's own final pose, so it cannot contain how much the clips turn the cube)
            self.wp_tol[k, -1] = torch.maximum(self.wp_tol[k, -1], self.done_tol[k])
        self.phase = torch.linspace(0, 1, nwp, device=dev)
        self.local = LOCAL.to(dev)
        b = dict(np.load(cfg.bank_path))
        first = {}
        for i, c in enumerate(b["clip"]):
            first.setdefault(int(c), i)
        self.clip_start = np.stack([b["cube_pos"][first[c]] for c in range(int(b["clip"].max()) + 1)])
        good = (np.linalg.norm(b["cube_pos"] - b["clip_cube"], axis=1) < 0.03) & (b["phase"] < 0.8)
        self.bank = {k: v[good] for k, v in b.items() if len(v) == len(good)}
        self.bank_by_skill = [np.nonzero(self.bank["skill"] == k)[0] for k in range(len(SKILLS))]
        self.bank_end = b["clip_end_cube"]
        self.rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 0)
        self.eval_chain = None

        self.cmds = torch.zeros(n, MAX_CMDS, 3, dtype=torch.long, device=dev)
        self.n_cmd = torch.ones(n, dtype=torch.long, device=dev)
        self.cur = torch.zeros(n, dtype=torch.long, device=dev)
        self.completed = torch.zeros(n, dtype=torch.long, device=dev)
        self.completed_final = torch.zeros(n, dtype=torch.long, device=dev)
        self.last_done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.chold = torch.zeros(n, dtype=torch.long, device=dev)
        self.wp_reached = torch.zeros(n, dtype=torch.long, device=dev)       # waypoints reached (index of the last)
        self.goal_corners = torch.zeros(n, 8, 3, device=dev)
        self.path = torch.zeros(n, nwp, 8, 3, device=dev)                 # this command's waypoints, table frame
        self.lift_start = torch.full((n, 3), torch.nan, device=dev)       # clip-start cube for pick-lift starts
        self.pot = torch.zeros(n, device=dev)
        self.need_start = torch.ones(n, dtype=torch.bool, device=dev)
        self.d_next = torch.zeros(n, device=dev)
        self.d_goal = torch.zeros(n, device=dev)
        self.prev_pos = torch.zeros(n, 3, device=dev)
        self.prev_action = torch.full((n, 5), torch.nan, device=dev)
        self.last_episode["completed"] = torch.zeros(n, device=dev)
        self.last_episode["n_cmd"] = torch.zeros(n, device=dev)
        self.skill_stats = np.zeros((len(SKILLS), 2))

    # ---------- keypoints ----------
    def corners(self, pos, quat):
        return pos[:, None] + torch.einsum("nij,kj->nki", matrix_from_quat(quat), self.local)

    def command(self):
        c = self.cmds[torch.arange(self.num_envs, device=self.device), self.cur]
        return c[:, 0], c[:, 1], c[:, 2]

    def objects(self):
        """Commanded and other cube: positions (table frame), quaternions, corners."""
        _, cube, _ = self.command()
        blue = (cube == 1)[:, None]
        rp, bp = self.to_table(self.red.data.root_pos_w), self.to_table(self.blue.data.root_pos_w)
        rq, bq = self.red.data.root_quat_w, self.blue.data.root_quat_w
        ap, aq = torch.where(blue, bp, rp), torch.where(blue, bq, rq)
        op, oq = torch.where(blue, rp, bp), torch.where(blue, rq, bq)
        return ap, aq, self.corners(ap, aq), op, oq, self.corners(op, oq)

    def start_commands(self, ids):
        """Goal pose and waypoint path for the current command of envs ids."""
        skill, _, dest = self.command()
        ap, aq, ac, op, oq, oc = self.objects()
        tgt = self.to_table(self.target.data.root_pos_w)
        tgt[:, 2] = CUBE_HALF
        on_other = op + torch.tensor([0, 0, 2 * CUBE_HALF], device=self.device)
        dest_pos = torch.where((dest == OTHER)[:, None], on_other, tgt)
        lift_from = torch.where(torch.isnan(self.lift_start), ap, self.lift_start)
        goal = torch.where(self.moves[skill][:, None], dest_pos, lift_from + self.delta[skill])
        frame = torch.where(self.moves[skill], torch.atan2(goal[:, 1] - ap[:, 1], goal[:, 0] - ap[:, 0]),
                            torch.zeros_like(ap[:, 0]))
        goal_yaw = yaw_of_quat(aq)
        gc = goal[:, None] + rot_z(self.local[None].expand(len(ap), 8, 3), goal_yaw)
        # fit the template to this start: in the travel frame, stretch each axis by (this start's distance from
        # the goal) / (the clips' start distance), so a short carry keeps the clips' shape at a smaller size;
        # what is left over (sideways offsets, the cube's turn) is added with a weight fading to 0 at the goal
        tmpl = self.wp[skill]                                                               # (n, P, 8, 3)
        start = rot_z(ac - gc, -frame)                                                      # (n, 8, 3)
        t0, s0 = tmpl[:, 0].mean(1), start.mean(1)                                          # (n, 3) centres
        scale = torch.where(t0.abs() > STRETCH_MIN, (s0 / t0.where(t0.abs() > STRETCH_MIN, torch.ones_like(t0))).clamp(0, 3),
                            torch.ones_like(t0))
        tmpl = tmpl * scale[:, None, None]
        tmpl = tmpl + (1 - self.phase)[None, :, None, None] * (start - tmpl[:, 0])[:, None]
        path = gc[:, None] + rot_z(tmpl, frame)                                             # (n, P, 8, 3)
        self.goal_corners[ids], self.path[ids] = gc[ids], path[ids]
        self.wp_reached[ids] = 0
        self.chold[ids] = 0
        self.lift_start[ids] = torch.nan
        self.need_start[ids] = False
        self.pot[ids] = self.potential(ac)[ids]
        self.prev_pos[ids] = ap[ids]

    @staticmethod
    def corner_dist(a, b):
        """Mean over a's corners of the distance to the nearest corner of b: a (n, 8, 3), b (n, ..., 8, 3)."""
        while a.dim() < b.dim():
            a = a.unsqueeze(1)
        a = a.expand(b.shape)
        return torch.cdist(a.reshape(-1, 8, 3), b.reshape(-1, 8, 3)).amin(-1).mean(-1).reshape(b.shape[:-2])

    def potential(self, ac):
        """Advance the reached waypoints, return the potential (and store distances for logs/tests)."""
        d = self.corner_dist(ac, self.path)                                                  # (n, P)
        skill = self.command()[0]
        tol = self.wp_tol[skill]
        idx = torch.arange(self.n_wp, device=self.device)
        hit = (d < tol) & (idx[None] >= self.wp_reached[:, None])
        furthest = torch.where(hit, idx[None], torch.full_like(d, -1, dtype=torch.long)).amax(-1)
        self.wp_reached = torch.maximum(self.wp_reached, furthest)
        nxt = (self.wp_reached + 1).clamp(max=self.n_wp - 1)
        rows = torch.arange(self.num_envs, device=self.device)
        self.d_next = d[rows, nxt]
        self.d_goal = self.corner_dist(ac, self.goal_corners)
        close = torch.exp(-(self.d_next / tol[rows, nxt]) ** 2)
        close = torch.where(self.wp_reached >= self.n_wp - 1, torch.ones_like(close), close)
        return self.wp_reached.float() + close

    def _pre_physics_step(self, actions):
        super()._pre_physics_step(actions)
        if self.cfg.action_rate > 0:            # after super(), which resets this step's reward accumulator
            a = actions.clamp(-1.0, 1.0)
            d = torch.nan_to_num(a[:, :4] - self.prev_action[:, :4], nan=0.0)   # no penalty on an episode's first step
            self.rew_acc -= self.cfg.action_rate * (d ** 2).sum(-1)
            self.prev_action = a.clone()

    # ---------- per tick ----------
    def _tick(self):
        if self.need_start.any():
            self.start_commands(self.need_start.nonzero().squeeze(-1))
        s = self.state()
        self.ticks += 1
        self.was_lifted |= s["red"][:, 2] > tasks.LIFTED
        ap, aq, ac, op, oq, oc = self.objects()
        # at rest from the position change per tick: PhysX reports ~0.1 m/s for a cube held still in the
        # fingers (contact jitter) while its position does not change by 1 mm
        speed = (ap - self.prev_pos).norm(dim=-1) / (self.cfg.sim.dt * self.cfg.decimation)
        self.prev_pos = ap.clone()
        pot = self.potential(ac)
        rew = pot - self.pot - self.cfg.time_cost
        self.pot = pot

        at_end = self.wp_reached >= self.n_wp - 1
        tol = self.done_tol[self.command()[0]]
        rest = self.cfg.rest_speed or (REST_HOLD if self.cfg.hold_fix else REST)
        ok = at_end & (self.d_goal < tol) & (speed < rest)
        if self.cfg.release:              # the gripper ends the skill as the clips' hand does
            closed = 1 - (self.grip_target / 0.04).clamp(0, 1)
            ok &= (closed > 0.5) == (self.end_grip[self.command()[0]] > 0.5)
        if self.cfg.hold_fix:
            rew = rew + HOLD_W * at_end.float() * torch.exp(-(self.d_goal / tol) ** 2)
        last = self.cur == self.n_cmd - 1
        if self.cfg.final_check:
            s["was_lifted"] = self.was_lifted
            ok &= ~last | tasks.success(self.cfg.final_check, s)
        self.chold = torch.where(ok, self.chold + 1, torch.zeros_like(self.chold))
        self.ok = ok & last
        self.done_ok = last & (self.chold >= HOLD)
        newly = self.done_ok & ~self.last_done
        self.succ_latch |= self.done_ok
        adv = ~last & (self.chold >= HOLD)
        rew = rew + DONE_BONUS * (adv | newly).float() + HOLD_BONUS * self.done_ok.float()
        self.last_done |= self.done_ok
        self.completed += adv.long()
        self.cur += adv.long()
        self.completed_final = self.completed + self.last_done.long()
        self.rew_acc += rew
        if adv.any():          # set the next command's goal now, so the policy never sees the finished one's
            self.start_commands(adv.nonzero().squeeze(-1))

        was_knocked = self.knocked.clone()
        for c in (s["red"], s["blue"]):
            self.knocked |= (c[:, 2] < -0.05) | (c[:, :2].abs() > 0.5).any(-1)
        if self.cfg.knock_penalty:
            left = (self.max_episode_length - self.episode_length_buf).float()
            self.rew_acc -= (self.knocked & ~was_knocked).float() * self.cfg.time_cost * left
        self._metrics(s)
        if self.tick_callback is not None:
            self.tick_callback(self)

    def _get_observations(self):
        s = self.state()
        yaw = self.tcp_yaw()
        skill, cube, dest = self.command()
        ap, aq, ac, *_ = self.objects()
        rows = torch.arange(self.num_envs, device=self.device)
        nxt = (self.wp_reached + 1).clamp(max=self.n_wp - 1)
        oh = torch.nn.functional.one_hot
        obs = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08,
                         s["red"], s["blue"], s["target"], ap - s["tcp"],
                         (ac - self.goal_corners).flatten(1), (self.path[rows, nxt] - ac).flatten(1),
                         oh(skill, 4).float(), oh(cube, 2).float(), oh(dest, 3).float(),
                         (self.wp_reached.float() / (self.n_wp - 1))[:, None]], dim=-1)
        out = {"policy": obs}
        if self.cfg.cameras:
            out.update(self.images())
        return out

    # ---------- reset ----------
    def set_chain(self, chain):
        """Evaluation: every env runs this list of (skill name, cube, dest) from home."""
        self.eval_chain = [(SKILLS.index(k), c, d) for k, c, d in chain]

    def _next(self, k, cube, dest):
        if k == PICK:
            return (PLACE, cube, int(self.rng.integers(2)))
        if k == PLACE and dest == OTHER:
            return (PICK, cube, NONE)                       # the one on top: unstack it next
        return (PICK, int(self.rng.integers(2)), NONE)

    def _away(self, avoid, k=1):
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
            k = int(self.rng.choice([REACH, PUSH, PICK, PLACE]))
            f = int(self.rng.choice(self.bank_by_skill[k]))
            cube = int(self.rng.integers(2))
            cmd = PICK if k == REACH else k                 # a reach clip state starts a pick-lift on its way
            dest = TARGET if cmd == PUSH else (int(self.rng.integers(2)) if cmd == PLACE else NONE)
            act, hand = self.bank["cube_pos"][f, :2], self.bank["cmd"][f, :2]
            if cmd == PUSH:
                tgt = self.bank_end[self.bank["clip"][f], :2] + self.rng.uniform(-0.03, 0.03, 2)
                tgt = np.clip(tgt, tasks.TARGET_BOX[0], tasks.TARGET_BOX[1])
                oth, = self._away([act, hand, tgt])
            else:
                oth, tgt = self._away([act, hand], 2)
            chain, frame, top = [(cmd, cube, dest)], f, -1
        elif r < P_BANK + P_STACKED:
            cube = int(self.rng.integers(2))
            oth, tgt = self._away([], 2)
            act, chain, frame, top = oth, [(PICK, cube, NONE)], -1, cube
        else:
            cube = int(self.rng.integers(2))
            push = self.rng.random() < 0.3
            lay = tasks.sample_layouts("push" if push else "c1", 1, torch.Generator().manual_seed(
                int(self.rng.integers(1 << 30))))[0].numpy()
            act, oth, tgt = lay[0:2], lay[2:4], lay[4:6]
            chain, frame, top = [(PUSH, cube, TARGET) if push else (PICK, cube, NONE)], -1, -1
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
        if len(done_ids):
            comp = self.completed_final[done_ids].cpu().numpy()
            cm = self.cmds[done_ids, :, 0].cpu().numpy()
            nc = self.n_cmd[done_ids].cpu().numpy()
            for j in range(int(nc.max())):
                valid = nc > j
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
                if j != REACH:
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
        self.lift_start[env_ids] = torch.nan
        self.prev_action[env_ids] = torch.nan

        dev = self.device
        origin = self.table_origin[env_ids]
        unit = torch.tensor([1.0, 0, 0, 0], device=dev)
        for j in np.nonzero(tops >= 0)[0]:
            cube = self.red if tops[j] == 0 else self.blue
            base = lay[j, 2:4] if tops[j] == 0 else lay[j, 0:2]
            pos = torch.cat([base.to(dev), torch.tensor([3 * CUBE_HALF], device=dev)]) + origin[j]
            e = env_ids[j:j + 1]
            cube.write_root_pose_to_sim(torch.cat([pos, unit])[None], env_ids=e)
            cube.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev), env_ids=e)
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
            # a pick-lift that starts part-way through a clip lifts to the clip's own goal, not 14 cm above here
            is_pick = torch.tensor([self.bank["skill"][fi] == PICK for fi in f], device=dev)
            start = torch.tensor(self.clip_start[self.bank["clip"][f]], device=dev, dtype=torch.float32)
            self.lift_start[e[is_pick]] = start[is_pick]
            for cube_id, cube in [(0, self.red), (1, self.blue)]:
                m = torch.tensor([chains[j][0][1] == cube_id for j in bs], device=dev)
                if m.any():
                    cube.write_root_pose_to_sim(pose[m], env_ids=e[m])
                    cube.write_root_velocity_to_sim(torch.zeros(int(m.sum()), 6, device=dev), env_ids=e[m])
