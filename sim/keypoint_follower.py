"""Scripted follower of the keypoint env's current command (privileged state), for reward checks and videos."""
import torch

from sim.envs.keypoint_env import PICK, PLACE, PUSH
from sim.envs.panda_env import MAX_DXYZ, MAX_DYAW

HOVER, CARRY, GRASP_Z = 0.12, 0.12, 0.001
PUSH_Z, PUSH_GRIP = 0.03, -0.25


class Follower:
    """Scripted command follower (privileged state). Phases never step back: pick-lift hover, descend,
    close, lift to the goal; place-down carry over the goal, lower, open; push get behind, descend, push."""

    def __init__(self, env):
        n, dev = env.num_envs, env.device
        self.cur = torch.full((n,), -1, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)

    def act(self, env):
        s = env.state()
        skill, _, _ = env.command()
        ap, aq, ac, *_ = env.objects()
        goal = env.goal_corners.mean(1)
        tcp, cmd = s["tcp"], env.cmd
        n, dev = env.num_envs, env.device
        new = env.cur != self.cur
        self.cur = env.cur.clone()
        self.phase[new], self.t[new] = 0, 0
        self.t += 1
        up = lambda h: torch.tensor([0, 0, h], device=dev)  # noqa: E731
        near = lambda a, b, tol: (a - b).norm(dim=-1) < tol  # noqa: E731
        ph = self.phase
        target = cmd.clone()
        opening = torch.ones(n, device=dev)
        held_off = tcp - ap
        # pick-lift
        p = skill == PICK
        ph[p & (ph == 0) & near(cmd, ap + up(HOVER), 0.005)] = 1
        ph[p & (ph == 1) & near(tcp, ap + up(GRASP_Z), 0.01)] = 2
        ph[p & (ph == 2) & (self.t > 8)] = 3
        self.t[p & (ph == 2) & (self.t > 8)] = 0
        target[p & (ph == 0)] = (ap + up(HOVER))[p & (ph == 0)]
        target[p & (ph == 1)] = (ap + up(GRASP_Z))[p & (ph == 1)]
        target[p & (ph == 3)] = (goal + held_off)[p & (ph == 3)]
        opening[p & (ph >= 2)] = 0.0
        # place-down
        q = skill == PLACE
        ph[q & (ph == 0) & near(ap[:, :2], goal[:, :2], 0.008)] = 1
        ph[q & (ph == 1) & ((ap[:, 2] - goal[:, 2]).abs() < 0.004)] = 2
        target[q & (ph == 0)] = (goal + up(CARRY) + held_off)[q & (ph == 0)]
        target[q & (ph == 1)] = (goal + held_off)[q & (ph == 1)]
        opening[q] = (ph == 2)[q].float()
        # push (as sim/expert.py)
        w = skill == PUSH
        u_ = goal[:, :2] - ap[:, :2]
        dist = u_.norm(dim=-1, keepdim=True)
        u = u_ / dist.clamp(min=1e-6)
        behind = torch.cat([ap[:, :2] - 0.075 * u, torch.full((n, 1), PUSH_Z, device=dev)], -1)
        ph[w & (ph == 0) & near(tcp, behind + up(HOVER - PUSH_Z), 0.02)] = 1
        ph[w & (ph == 1) & near(tcp, behind, 0.01)] = 2
        ph[w & (ph == 2) & (dist.squeeze(-1) < 0.01)] = 3
        pushing = torch.cat([ap[:, :2] + u * (dist.clamp(max=0.04) - 0.035), torch.full((n, 1), PUSH_Z, device=dev)], -1)
        target[w & (ph == 0)] = (behind + up(HOVER - PUSH_Z))[w & (ph == 0)]
        target[w & (ph == 1)] = behind[w & (ph == 1)]
        target[w & (ph == 2)] = pushing[w & (ph == 2)]
        line = torch.atan2(u[:, 1], u[:, 0])
        line = env.cmd_yaw + torch.remainder(line - env.cmd_yaw + torch.pi / 2, torch.pi) - torch.pi / 2
        line = torch.where(line > 2.0, line - torch.pi, torch.where(line < -2.0, line + torch.pi, line))
        a = torch.zeros(n, 5, device=dev)
        a[:, :3] = ((target - cmd) / MAX_DXYZ).clamp(-1, 1)
        a[w & (ph == 2), :3] *= 0.5
        a[:, 3] = torch.where(w & (ph < 2), ((line - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1), torch.zeros_like(line))
        a[:, 4] = opening * 2 - 1
        a[w, 4] = PUSH_GRIP
        return a
