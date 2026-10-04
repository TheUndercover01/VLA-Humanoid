"""Does the clip reward (sim/envs/skill_env.py) score the right behaviour highest? Checked before any RL.

Runs one task's skill chain with a scripted policy that follows the env's current command, and
with cheating variants of it side by side (a quarter of the envs each), on training layouts.
Reports per behaviour: commands done, chain done (the reward's own "done"), task success by
tasks.success (the eval check, which the reward never sees), and the return under the clip and
sparse rewards.

    PYTHONPATH=. ./isaaclab.sh -p sim/check_skill_reward.py --headless --task c1
Behaviours: scripted (does each command, then waits for the next); idle (no motion); open_hand
(fingers never close: the hand can rise without the cube); no_release (fingers stay closed during
place-down: hovering at the destination with the cube); drop (opens over the destination at carry
height and lets the cube fall instead of placing it).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["push", "lift", "c1", "c2", "c3"])
parser.add_argument("--per_behaviour", type=int, default=32)
parser.add_argument("--debug", action="store_true", help="print env 0 every second")
parser.add_argument("--ref", default="data/processed/skill_ref_standin.npz")
parser.add_argument("--bank", default="data/processed/state_bank_standin.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.panda_env import MAX_DXYZ, MAX_DYAW  # noqa: E402
from sim.envs.skill_env import (NONE, OTHER, PICK, PLACE, PUSH, REACH, STAGE, TARGET,  # noqa: E402
                                PandaSkillEnv, PandaSkillEnvCfg)

CHAINS = {
    "push": [("push", 0, TARGET)],
    "c3": [("push", 1, TARGET), ("reach", 0, NONE), ("pick-lift", 0, NONE), ("place-down", 0, OTHER)],
    "lift": [("reach", 0, NONE), ("pick-lift", 0, NONE)],
    "c1": [("reach", 0, NONE), ("pick-lift", 0, NONE), ("place-down", 0, TARGET)],
    "c2": [("reach", 0, NONE), ("pick-lift", 0, NONE), ("place-down", 0, OTHER)],
}
BEHAVIOURS = ["scripted", "idle", "open_hand", "no_release", "drop"]
HOVER, CARRY = 0.12, 0.12      # m above the cube / the destination
PUSH_Z, PUSH_GRIP = 0.03, -0.25  # as sim/expert.py: fingertips 3 cm up, 3 cm apart


class Follower:
    """Scripted command follower (privileged state): does the current command, then waits for the
    next one. Each command goes through phases (reach: hover, descend; pick-lift: close, lift;
    place-down: carry, lower, open) that never step back, so it cannot dither at a boundary."""

    def __init__(self, env):
        self.cur = torch.full((env.num_envs,), -1, device=env.device)
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.t = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def act(self, env):
        s = env.state()
        skill, act, end, _ = env.geometry(s)
        tcp, cmd = s["tcp"], env.cmd
        n, dev = env.num_envs, env.device
        new = env.cur != self.cur
        self.cur = env.cur.clone()
        self.phase[new], self.t[new] = 0, 0
        self.t += 1
        up = lambda h: torch.tensor([0, 0, h], device=dev)  # noqa: E731
        near = lambda a, b, tol: (a - b).norm(dim=-1) < tol  # noqa: E731
        goal = cmd.clone()
        opening = torch.ones(n, device=dev)
        ph = self.phase
        # reach: hover over the cube, then descend around it, fingers open
        r = skill == REACH
        hover = act + up(HOVER)
        ph[r & (ph == 0) & near(cmd, hover, 0.005)] = 1
        goal[r] = torch.where((ph == 1)[r, None], act[r], hover[r])
        # pick-lift: close for 8 ticks, then lift the cube to where the command puts it
        p = skill == PICK
        ph[p & (ph == 0) & (self.t > 8)] = 1
        held_off = tcp - act                                       # hand relative to the cube in the fingers
        goal[p & (ph == 1)] = (end + held_off)[p & (ph == 1)]
        opening[p] = 0.0
        # place-down: carry above the destination, lower, open
        q = skill == PLACE
        ph[q & (ph == 0) & near(act[:, :2], end[:, :2], 0.008)] = 1
        ph[q & (ph == 1) & ((act[:, 2] - end[:, 2]).abs() < 0.004)] = 2
        lift_to = end + up(CARRY) + held_off
        goal[q & (ph == 0)] = lift_to[q & (ph == 0)]
        goal[q & (ph == 1)] = (end + held_off)[q & (ph == 1)]
        opening[q] = (ph == 2)[q].float()
        # push (as sim/expert.py): get behind the cube, come down, push it along its line to the target
        u_ = end[:, :2] - act[:, :2]
        dist = u_.norm(dim=-1, keepdim=True)
        u = u_ / dist.clamp(min=1e-6)
        w = skill == PUSH
        behind = torch.cat([act[:, :2] - 0.075 * u, torch.full((n, 1), PUSH_Z, device=dev)], -1)
        ph[w & (ph == 0) & near(tcp, behind + up(HOVER - PUSH_Z), 0.02)] = 1      # on the fingertips, not the
        ph[w & (ph == 1) & near(tcp, behind, 0.01)] = 2                          # command (the arm lags it)
        pushing = torch.cat([act[:, :2] + u * (dist.clamp(max=0.04) - 0.035), torch.full((n, 1), PUSH_Z, device=dev)], -1)
        goal[w & (ph == 0)] = (behind + up(HOVER - PUSH_Z))[w & (ph == 0)]
        goal[w & (ph == 1)] = behind[w & (ph == 1)]
        ph[w & (ph == 2) & (dist.squeeze(-1) < 0.01)] = 3                        # there: stop and wait
        goal[w & (ph == 2)] = pushing[w & (ph == 2)]
        line = torch.atan2(u[:, 1], u[:, 0])
        line = env.cmd_yaw + torch.remainder(line - env.cmd_yaw + torch.pi / 2, torch.pi) - torch.pi / 2
        line = torch.where(line > 2.0, line - torch.pi, torch.where(line < -2.0, line + torch.pi, line))
        a = torch.zeros(n, 5, device=dev)
        a[:, :3] = ((goal - cmd) / MAX_DXYZ).clamp(-1, 1)
        a[w & (ph == 2), :3] *= 0.5                                # push slowly so the cube does not slide on
        a[:, 3] = torch.where(w & (ph < 2), ((line - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1), torch.zeros_like(line))
        a[:, 4] = opening * 2 - 1
        a[w, 4] = PUSH_GRIP
        return a


def main():
    k = args.per_behaviour
    n = k * len(BEHAVIOURS)
    cfg = PandaSkillEnvCfg(task=args.task, terminate_on_success=False, ref_path=args.ref, bank_path=args.bank,
                           reward="clip", episode_s=20.0)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaSkillEnv(cfg)
    env.set_chain(CHAINS[args.task])
    env.set_layouts(tasks.sample_layouts(args.task, n, torch.Generator().manual_seed(555)))
    env.reset()
    dev = env.device
    group = torch.arange(n, device=dev) // k

    ret_clip = torch.zeros(n, device=dev)
    ret_sparse = torch.zeros(n, device=dev)
    task_hold = torch.zeros(n, dtype=torch.long, device=dev)
    task_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    chain_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    best = torch.zeros(n, dtype=torch.long, device=dev)
    n_cmd = len(CHAINS[args.task])
    per_cmd = torch.zeros(n_cmd, 2, device=dev)       # scripted: summed path reward, ticks

    def on_tick(e):
        sparse = STAGE * e.completed + STAGE * e.done_ok.float()
        path = e.path_terms[:, 0] * (1 + e.path_terms[:, 1])
        ret_sparse.add_(sparse)
        ret_clip.add_(sparse + path)
        s = e.state()
        s["was_lifted"] = e.was_lifted
        ok = tasks.success(args.task, s)
        task_hold.copy_(torch.where(ok, task_hold + 1, torch.zeros_like(task_hold)))
        task_ok.logical_or_(task_hold >= tasks.SUCCESS_HOLD)
        chain_ok.logical_or_(e.done_ok)
        if args.debug and e.ticks[0] % 10 == 0:
            i = 0
            print(f"t {e.ticks[i].item()} cmd {e.cur[i].item()} prog {e.prog[i].item()} end_err {e.end_err[i]:.3f} "
                  f"x {[round(v, 3) for v in e.x[i].tolist()]} grip {s['grip'][i]:.3f} tcp {[round(v, 3) for v in s['tcp'][i].tolist()]} "
                  f"red {[round(v, 3) for v in s['red'][i].tolist()]} cmdpos {[round(v, 3) for v in e.cmd[i].tolist()]}", flush=True)
        best.copy_(torch.maximum(best, e.completed_final))
        for j in range(n_cmd):
            sel = (group == 0) & (e.cur == j) & (e.completed_final == j)
            per_cmd[j, 0] += path[sel].sum()
            per_cmd[j, 1] += sel.sum()

    env.tick_callback = on_tick
    follower = Follower(env)
    for _ in range(int(env.max_episode_length) - 2):
        a = follower.act(env)
        a[group == 1] = 0.0
        a[group == 2, 4] = 1.0                                           # fingers stay open
        placing = (group == 3) & (env.command()[0] == PLACE)
        a[placing, 4] = -1.0                                             # fingers stay closed
        over = (group == 4) & (env.command()[0] == PLACE) & (follower.phase >= 1)
        a[over, :3] = 0.0                                                # stop at carry height ...
        a[over, 4] = 1.0                                                 # ... and let go
        env.step(a)

    chain = " > ".join(f"{s}" for s, _, _ in CHAINS[args.task])
    lines = [f"\n{args.task}: {chain} ({k} envs per behaviour, {env.max_episode_length * 0.1:.0f} s, {args.ref})",
             f"{'behaviour':12s} {'cmds done':>9s} {'chain done':>10s} {'task ok':>8s} {'agree':>6s} "
             f"{'return clip':>12s} {'return sparse':>14s}"]
    for g, name in enumerate(BEHAVIOURS):
        m = group == g
        agree = (chain_ok[m] == task_ok[m]).float().mean()
        lines.append(f"{name:12s} {best[m].float().mean():7.2f}/{n_cmd} {chain_ok[m].float().mean():10.0%} "
                     f"{task_ok[m].float().mean():8.0%} {agree:6.0%} {ret_clip[m].mean():12.0f} {ret_sparse[m].mean():14.0f}")
    m = group == 0
    lines.append(f"scripted, at the end: distance to the clips' end state (6-D) median {env.end_err[m].median() * 100:.1f} cm, "
                 f"done radius {env.ref_radius[env.command()[0][m]].mean() * 100:.1f} cm; "
                 f"x - clip end (cm) median {[round(v, 1) for v in ((env.x - env.ref_path[env.command()[0], -1])[m].median(0).values * 100).tolist()]}")
    prof = per_cmd[:, 0] / per_cmd[:, 1].clamp(min=1)
    lines.append("scripted, mean path reward per tick while doing each command: "
                 + ", ".join(f"{s} {p:.2f}" for (s, _, _), p in zip(CHAINS[args.task], prof.tolist())))
    print("\n".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
