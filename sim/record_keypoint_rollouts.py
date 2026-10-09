"""Record a keypoint RL teacher with cameras: the distillation data for SmolVLA ("Ours").

The teacher runs in PandaKeypointEnv with its training distribution (chains of at most --max_chain
commands, random layouts, starts from the clips' states), so the data has single skills and hand-offs
between two of them, never a longer task. Each frame is labelled with the instruction of the command
active at that moment (vla/prompts.py: command_prompt), so the VLA only ever sees atomic instructions.
DART noise (--noise): the executed action is perturbed, the recorded label is the teacher's clean
action. With --action_filter (a teacher trained behind the low-pass filter) the label is the clean
executed command, 0.7 x the filter state + 0.3 x the teacher's action, and the noise is added to the
teacher's action before the filter (as its exploration noise in training): noise added after the filter
was beyond what the teacher can correct through the filter's 0.3 gain, so few episodes finished. The VLA
learns the smooth executed motion, not the teacher's raw output. Only episodes where
every command was done are kept.

    PYTHONPATH=. ./isaaclab.sh -p sim/record_keypoint_rollouts.py --headless --enable_cameras \
        --policy rsl:runs/rl/keypoints_hold_v3_chain2_s1/model_900.pt --hold_fix --release \
        --episodes 1000 --out data/processed/rollouts/ours_kp
Writes one npz per episode: front, wrist (T, H, W, 3) uint8, state (T, 6), action (T, 5), prompts (T,),
commands (the episode's command sentence).
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--policy", required=True, help="rsl:<checkpoint.pt> of a keypoint RL run")
parser.add_argument("--episodes", type=int, default=1000, help="successful episodes to keep")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--max_chain", type=int, default=2)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--noise", type=float, default=0.3, help="std of DART noise on executed dx..dyaw (0 = off)")
# the teacher's env settings (they change when a command counts as done, so they must match its training)
parser.add_argument("--hold_fix", action="store_true")
parser.add_argument("--release", action="store_true")
parser.add_argument("--rest_speed", type=float, default=0.0)
parser.add_argument("--action_filter", type=float, default=0.0)
parser.add_argument("--front_view", default="far", choices=["far", "near"])
parser.add_argument("--ref", default="data/processed/keypoint_ref_real_straight.npz",
                    help="template of the env; MUST be the one the teacher was trained with (8 Oct: this script used the env default, the stand-in template)")
parser.add_argument("--paired", action="store_true", help="fresh-layout episodes come twice: each cube commanded on the same layout")
parser.add_argument("--episode_s", type=float, default=0.0, help="0 = the env default (15 s)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.clips import SKILLS  # noqa: E402
from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.rl_cfg import RslPolicy  # noqa: E402
from vla.prompts import command_prompt  # noqa: E402


def main():
    cfg = PandaKeypointEnvCfg(task="c1", cameras=True, image_size=args.image_size, front_view=args.front_view, terminate_on_success=True,
                              max_chain=args.max_chain, seed=args.seed, hold_fix=args.hold_fix,
                              release=args.release, rest_speed=args.rest_speed, action_filter=args.action_filter, paired=args.paired)
    cfg.ref_path = args.ref
    if args.episode_s:
        cfg.episode_s = args.episode_s
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    env = PandaKeypointEnv(cfg)
    policy = RslPolicy(args.policy.removeprefix("rsl:"), env)
    obs, _ = env.reset()
    n, dev = env.num_envs, env.device
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["front", "wrist", "state", "action", "prompts", "cidx", "dnow"]
    recs = [{k: [] for k in keys} for _ in range(n)]
    cmds = [None] * n
    kept, tried = 0, 0
    while kept < args.episodes:
        s = env.state()
        yaw = env.tcp_yaw()
        state = torch.cat([s["tcp"], torch.sin(yaw)[:, None], torch.cos(yaw)[:, None], s["grip"][:, None] / 0.08], -1)
        action, _ = policy.act(env)
        skill, cube, dest = (x.tolist() for x in env.command())
        img = env.images()
        front, wrist = img["front"][..., :3].to(torch.uint8).cpu(), img["wrist"][..., :3].to(torch.uint8).cpu()
        clean = action.clamp(-1, 1)
        if args.action_filter > 0:
            clean[:, :4] = args.action_filter * env.a_filt + (1 - args.action_filter) * clean[:, :4]
        clean = clean.cpu()
        state = state.cpu()
        for i in range(n):
            r = recs[i]
            if not r["front"]:                          # a new episode in this env: remember its commands
                c = env.cmds[i, :env.n_cmd[i]].tolist()
                cmds[i] = " > ".join(command_prompt(SKILLS[k], cb, d) for k, cb, d in c)
            r["front"].append(front[i])
            r["wrist"].append(wrist[i])
            r["state"].append(state[i])
            r["action"].append(clean[i])
            r["prompts"].append(command_prompt(SKILLS[skill[i]], cube[i], dest[i]))
            r["cidx"].append(int(env.cur[i]))                 # which command of the instruction the env is on (goes back when work is undone)
        executed = action.clone()
        if args.noise > 0:
            executed[:, :4] += args.noise * torch.randn_like(executed[:, :4])
        env.episode_done[:] = False
        obs, *_ = env.step(executed)
        if hasattr(env, "done_now"):                           # 9 Oct: the env's own "this command is done now" check, per tick (the done flag's label)
            dn = env.done_now.cpu()
            for i in range(n):
                recs[i]["dnow"].append(bool(dn[i]))
        for i in env.episode_done.nonzero().squeeze(-1).tolist():
            tried += 1
            r = recs[i]
            # the work must be DONE at the end (a knocked-apart episode that never recovers is not kept, although its success latch was set once)
            if env.last_episode["success"][i] > 0 and env.last_episode["completed"][i] >= env.last_episode["n_cmd"][i] \
                    and kept < args.episodes and len(r["action"]) > 2:
                extra = {"cidx": np.array(r["cidx"]), "dnow": np.array(r["dnow"][:len(r["cidx"])])} if r["dnow"] else {}
                np.savez_compressed(out / f"ep_{kept:05d}.npz",
                                    **{k: torch.stack(r[k]).numpy() for k in ["front", "wrist", "state", "action"]},
                                    prompts=np.array(r["prompts"]), commands=cmds[i], **extra)
                kept += 1
                if kept % 100 == 0:
                    print(f"kept {kept}/{args.episodes} ({tried} tried)", flush=True)
            recs[i] = {k: [] for k in keys}
    print(f"kept {kept}/{args.episodes} successful episodes ({tried} tried) -> {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
