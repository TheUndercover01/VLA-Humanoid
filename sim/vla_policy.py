"""Eval-harness policy that asks a SmolVLA server (vla/server.py) for action chunks.

The env must have cameras on. Every `replan` steps the current images, 6-D state and prompt
of every env go to the server; the next `replan` actions of each returned chunk are executed.
With a plan (B1+LLM), each env follows its list of atomic prompts and moves to the next one
when a simple completion check on the sim state passes; the switch forces a replan.
"""
from multiprocessing.connection import Client

import numpy as np
import torch

from sim.envs.tasks import CUBE_HALF, RELEASED
from vla.prompts import ATOMIC_PROMPTS

AUTHKEY = b"vla-humanoid"


def skill_done(skill, s, start_red):
    """Per-env completion check for one atomic instruction (only used to switch prompts)."""
    red, tcp = s["red"], s["tcp"]
    if skill == "reach":
        return ((tcp[:, :2] - red[:, :2]).norm(dim=-1) < 0.02) & (tcp[:, 2] < CUBE_HALF + 0.02)
    if skill == "pick-lift":
        return red[:, 2] > CUBE_HALF + 0.06
    if skill == "place-down":
        return (red[:, 2] < CUBE_HALF + 0.01) & (s["grip"] > RELEASED)
    if skill == "push":
        return (red[:, :2] - start_red[:, :2]).norm(dim=-1) > 0.05
    raise ValueError(skill)


class VLAPolicy:
    def __init__(self, port, prompt, plan=None, replan=10):
        self.conn = Client(("localhost", port), authkey=AUTHKEY)
        self.prompt, self.replan = prompt, replan
        prompt_to_skill = {v: k for k, v in ATOMIC_PROMPTS.items()}
        self.plan = [prompt_to_skill[p] for p in plan] if plan else None

    def reset(self, env):
        n = env.num_envs
        self.queue = None
        self.k = 0
        self.stage = torch.zeros(n, dtype=torch.long, device=env.device)
        self.start_red = env.state()["red"].clone()

    def _prompts(self, n):
        if self.plan is None:
            return [self.prompt] * n
        return [ATOMIC_PROMPTS[self.plan[min(i, len(self.plan) - 1)]] for i in self.stage.tolist()]

    def act(self, env):
        n = env.num_envs
        if self.queue is None or self.k >= self.replan:
            img = env.images()
            s = env.state()
            yaw = env.tcp_yaw()
            state = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08], -1)
            front = img["front"][..., :3].to(torch.uint8).cpu().numpy()
            wrist = img["wrist"][..., :3].to(torch.uint8).cpu().numpy()
            self.conn.send({"front": front.tobytes(), "wrist": wrist.tobytes(), "hw": front.shape[1:3],
                            "state": state.float().cpu().numpy().tobytes(), "task": self._prompts(n)})
            r = self.conn.recv()
            self.queue = torch.from_numpy(np.frombuffer(r["actions"], np.float32).reshape(r["shape"]).copy()).to(env.device)
            self.k = 0
        a = self.queue[:, self.k].clamp(-1, 1)
        self.k += 1
        skill = None
        if self.plan is not None:
            skill = torch.tensor([["reach", "push", "pick-lift", "place-down"].index(self.plan[min(i, len(self.plan) - 1)])
                                  for i in self.stage.tolist()], device=env.device)
        return a, skill

    def update(self, env):
        if self.plan is None:
            return
        s = env.state()
        advanced = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        for j, skill in enumerate(self.plan[:-1]):
            on = self.stage == j
            if on.any():
                advanced |= on & skill_done(skill, s, self.start_red)
        if advanced.any():
            self.stage = self.stage + advanced.long()
            self.k = self.replan            # new instruction: replan for every env on the next step
