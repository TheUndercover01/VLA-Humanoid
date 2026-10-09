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
        a = self.queue[:, self.k]
        if a.shape[1] == 5:                 # raw action; vocabulary actions are [skill scores | z]
            a = a.clamp(-1, 1)
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


class CommandVLAPolicy:
    """SmolVLA on the keypoint env's command chain: each env's prompt is the instruction of its current command
    (vla/prompts.py: command_prompt); the env switches commands when one is done (the same privileged check
    the RL teacher is evaluated with), and a switch forces a replan."""

    def __init__(self, port, replan=1, whole=None, history=0, state_history=False, learned_done=None,
                 done_thresh=0.5, done_hold=3, memory=False, marks=False, k_norm=6.0, vote=1, recheck=None):
        """whole: one instruction for the entire episode (the whole task, no planner); the env still switches
        commands, but only to score progress, and the VLA is never told.
        learned_done: the chain's commands [(skill, cube, dest)]. The VLA (trained with build_dataset --done_frames) gets
        ONE command at a time and predicts "this command is done" itself (6th action channel); each env's pointer moves
        to the next command after the flag has been above done_thresh for done_hold ticks in a row. Nothing from the sim
        (env.cur) reaches the policy, the env's own switch only scores."""
        self.conn = Client(("localhost", port), authkey=AUTHKEY)
        self.chain, self.done_thresh, self.done_hold = learned_done, done_thresh, done_hold
        # memory (7 Oct, user: the WHOLE sentence at once, the done flag as the policy's memory): with learned_done=chain the VLA gets
        # the whole chain as one sentence every step, plus one extra state number = how many done flags it has raised so far / 6
        # (build_dataset --prompt_mode memory --done_frames 5); nothing from the sim reaches it.
        # marks (7 Oct evening, build_dataset --prompt_mode marks): the whole sentence, the commands the policy has flagged done carry
        # "(done)" in the text, no count in the state.
        # recheck (8 Oct, marks only): (delay, window, low, need) in ticks. From `delay` ticks after an advance, for `window` ticks, the same
        # VLA is also asked with the last mark removed ("is that command really done?"); if its done flag stays below `low` for `need`
        # ticks in a row, the mark is erased and the policy goes back to that command. Its own outputs only, nothing from the sim.
        # Put-downs and pushes only (a first test on every command erased correct pick-up marks: 1 s later the cube is being carried).
        self.recheck, self.since, self.low = recheck, None, None
        # 6th number of recheck = 1: ROLLBACK a pair (9 Oct, user: "the VLA should be able to recover"): when the last put-down turns out not
        # to hold any more, erase its mark AND the pick-up of the same object before it, so the sentence reads as a fresh "pick X, put X on Y",
        # the state the student was trained on (erasing the put-down alone leaves "X picked (done), put X on Y" with X on the table)
        self.recheck_pair = bool(recheck) and len(recheck) > 5 and bool(recheck[5])
        self.vote = vote          # 8 Oct: query the VLA this many times (it samples noise) and take the MIN of the done flags
        self.memory, self.marks, self.k_norm = memory or marks, marks, k_norm
        self.stage = self.streak = None
        self.replan, self.whole, self.history = replan, whole, history
        self.queue, self.k, self.cur = None, 0, None
        self.fronts = []                 # front images of the last `history` + 1 steps (eval: one episode per env)
        self.state_history, self.past, self.prev_a = state_history, [], None

    def chains(self, n):
        """One chain per env: the same one for all (eval_chains), or `chain` is already a list of chains (sim/eval_multi.py, one scene each)."""
        return self.chain if self.chain and isinstance(self.chain[0], list) else [self.chain] * n

    def window(self, n):
        """recheck: the envs whose last mark is being re-examined now: from `delay` ticks after the advance for `window` ticks, and for the
        final command (everything marked done) without end when the 5th number of --recheck is 1 (a finished task can be undone later)."""
        d0, win = self.recheck[0], self.recheck[1]
        forever = len(self.recheck) > 4 and self.recheck[4]
        lens = torch.tensor([len(ch) for ch in self.chains(n)])
        end = (self.stage >= lens) & bool(forever)
        return ((self.since >= d0) & ((self.since < d0 + win) | end)) & self.checkable(), end

    def checkable(self):
        """recheck: only a put-down or a push leaves a lasting result to look at (a pick-up's "lifted" is gone once the hand moves on)."""
        return torch.tensor([k > 0 and ch[k - 1][0] != "pick-lift" for ch, k in zip(self.chains(len(self.stage)), self.stage.tolist())])

    def act(self, env):
        from sim.clips import SKILLS
        from vla.prompts import command_prompt, marked_sentence
        switched = self.whole is None and self.cur is not None and (env.cur != self.cur).any()
        if self.state_history:           # every step, also the ones that replay a chunk
            s = env.state()
            yaw = env.tcp_yaw()
            cur = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08], -1)
            self.past = (self.past + [cur])[-11:]
            if self.prev_a is None:
                self.prev_a = torch.zeros(env.num_envs, 5, device=env.device)
        if self.history:
            self.fronts = (self.fronts + [env.images()["front"][..., :3].to(torch.uint8).cpu().numpy()])[-(self.history + 1):]
        if self.queue is None or self.k >= self.replan or switched:
            img = env.images()
            s = env.state()
            yaw = env.tcp_yaw()
            state = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08], -1)
            if self.state_history:       # same layout as vla/build_dataset.py --state_history
                state = torch.cat([state, self.prev_a] + [self.past[max(len(self.past) - 1 - k, 0)] - state for k in (5, 10)], -1)
            skill, cube, dest = (x.tolist() for x in env.command())
            if self.chain:
                if self.stage is None:
                    self.stage = torch.zeros(env.num_envs, dtype=torch.long)
                    self.streak = torch.zeros(env.num_envs, dtype=torch.long)
                chains = self.chains(env.num_envs)
                if self.marks:
                    prompts = [marked_sentence([command_prompt(*c) for c in ch], k) for ch, k in zip(chains, self.stage.tolist())]
                elif self.memory:
                    prompts = [", then ".join(command_prompt(*c) for c in ch) for ch in chains]
                    state = torch.cat([state, (self.stage.float() / self.k_norm)[:, None].to(state.device)], -1)
                else:
                    prompts = [command_prompt(*ch[i]) for ch, i in zip(chains, self.stage.tolist())]
            else:
                prompts = [self.whole] * env.num_envs if self.whole else \
                    [command_prompt(SKILLS[k], c, d) for k, c, d in zip(skill, cube, dest)]
            front = img["front"][..., :3].to(torch.uint8).cpu().numpy()
            wrist = img["wrist"][..., :3].to(torch.uint8).cpu().numpy()
            msg = {"front": front.tobytes(), "wrist": wrist.tobytes(), "hw": front.shape[1:3],
                   "state": state.float().cpu().numpy().tobytes(), "task": prompts}
            if self.history:           # the oldest kept frame (the first frame until `history` steps have passed)
                msg["front_past"] = self.fronts[0].tobytes()
            outs = []
            for _ in range(self.vote if self.chain else 1):
                self.conn.send(msg)
                r = self.conn.recv()
                outs.append(np.frombuffer(r["actions"], np.float32).reshape(r["shape"]).copy())
            self.chk = None
            if self.recheck and self.marks and self.since is not None:
                inwin, _ = self.window(env.num_envs)
                if inwin.any():           # one more query, the sentence with the last mark removed for the envs being checked
                    msg = dict(msg, task=[marked_sentence([command_prompt(*c) for c in ch], k - 1 if w else k)
                                          for ch, k, w in zip(self.chains(env.num_envs), self.stage.tolist(), inwin.tolist())])
                    self.conn.send(msg)
                    r = self.conn.recv()
                    self.chk = torch.from_numpy(np.frombuffer(r["actions"], np.float32).reshape(r["shape"])[:, 0, 5].copy())
            if len(outs) > 1:         # the actions of the first sample, the done flag every sample agrees on
                outs[0][..., 5] = np.min([o[..., 5] for o in outs], 0)
            self.queue = torch.from_numpy(outs[0]).to(env.device)
            self.k = 0
            self.cur = env.cur.clone()
        a = self.queue[:, self.k]
        self.k += 1
        if self.chain:
            self.done_p = a[:, 5].cpu()      # the latest done-flag value of every env (videos show it)
            self.streak = torch.where((a[:, 5] > self.done_thresh).cpu(), self.streak + 1, torch.zeros_like(self.streak))
            lens = torch.tensor([len(ch) for ch in self.chains(len(self.stage))])
            adv = (self.streak >= self.done_hold) & (self.stage < lens - (0 if self.memory else 1))
            if self.recheck and self.marks:
                if self.since is None:
                    self.since = torch.full_like(self.stage, -1)
                    self.low = torch.zeros_like(self.stage)
                    self.n_retract = 0
                d0, win, lo, need = self.recheck[:4]
                if self.chk is not None:
                    inwin, _ = self.window(env.num_envs)
                    if __import__("os").environ.get("VLA_DEBUG"):
                        i = int(__import__("os").environ["VLA_DEBUG"])
                        print(f"DBG stage {int(self.stage[i])} since {int(self.since[i])} inwin {bool(inwin[i])} chk {float(self.chk[i]):.2f} low {int(self.low[i])} done_p {float(self.done_p[i]) if hasattr(self, 'done_p') else -1:.2f}", flush=True)
                    self.low = torch.where(inwin & (self.chk < lo), self.low + 1, torch.zeros_like(self.low))
                    back = (self.low >= need) & (self.stage > 0)
                    if back.any():       # its own check says the last command is not done: erase that mark
                        dec = back.long()
                        if self.recheck_pair:
                            extra = torch.tensor([1 if (k >= 2 and ch[k - 2][0] == "pick-lift" and ch[k - 2][1] == ch[k - 1][1]) else 0
                                                  for ch, k in zip(self.chains(len(self.stage)), self.stage.tolist())])
                            dec = dec * (1 + extra)
                        self.stage = self.stage - dec
                        self.since = torch.where(back, torch.full_like(self.since, -1), self.since)
                        self.low = torch.where(back, torch.zeros_like(self.low), self.low)
                        self.streak = torch.where(back, torch.zeros_like(self.streak), self.streak)
                        self.n_retract += int(back.sum())
                        self.k = self.replan
                        adv = adv & ~back
                self.since = torch.where(self.since >= 0, self.since + 1, self.since)
                _, keep = self.window(env.num_envs)
                self.since = torch.where((self.since >= d0 + win) & ~keep, torch.full_like(self.since, -1), self.since)
            if adv.any():                # the VLA said the command is done: next command, replan right away
                self.stage = self.stage + adv.long()
                self.streak = torch.where(adv, torch.zeros_like(self.streak), self.streak)
                if self.recheck and self.marks:
                    self.since = torch.where(adv, torch.zeros_like(self.since), self.since)
                    self.low = torch.where(adv, torch.zeros_like(self.low), self.low)
                self.k = self.replan
            a = a[:, :5]
        a = a.clamp(-1, 1)
        if self.state_history:
            self.prev_a = a.clone()
        return a, None
