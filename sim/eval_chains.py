"""Run a skill policy on fixed skill chains, most of them longer than anything it was trained on.

The skill env switches to the next command once the current one is done (the clips' end state,
privileged sim state, like the B1+LLM completion check). Success is the chain's last command done
and, for c1/c2/c3/unstack, also the task's own tasks.success check. Chains (fixed before any result):

  lift     reach r > pick-lift r                                   (2: training length, sanity check)
  c1       reach r > pick-lift r > place-down r on target          (3)
  c2       reach r > pick-lift r > place-down r on blue            (3)
  c3       push b to target > reach r > pick-lift r > place-down r on blue    (4)
  swap     reach b > pick-lift b > place-down b on red             (3: blue on red, roles reversed)
  unstack  c2, then reach r > pick-lift r > place-down r on target (6)
  restack  unstack, then reach b > pick-lift b > place-down b on red   (9)

    PYTHONPATH=. ./isaaclab.sh -p sim/eval_chains.py --headless --chain c1 \
        --policy rsl:runs/rl/skills_clip_chain2_s1/model_2999.pt [--val]
Writes runs/eval/<name>_chain_<chain>.csv: one row per start state.

--keypoints: the keypoint env (sim/envs/keypoint_env.py, object-only reward), whose commands have no reach:
  push     push r to target                                        (1)
  lift     pick-lift r                                             (1)
  c1       pick-lift r > place-down r on target                    (2)
  c2       pick-lift r > place-down r on blue                      (2)
  c3       push b to target > pick-lift r > place-down r on blue   (3)
  swap     pick-lift b > place-down b on red                       (2)
  unstack  c2, then pick-lift r > place-down r on target           (4)
  restack  unstack, then pick-lift b > place-down b on red         (6)
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--chain", required=True)
parser.add_argument("--policy", required=True, help="rsl:<checkpoint.pt>, or vla (a vla/server.py on --vla_port; --keypoints)")
parser.add_argument("--vla_port", type=int, default=6061)
parser.add_argument("--replan", type=int, default=1, help="vla: steps executed from each chunk before asking again")
parser.add_argument("--whole_prompt", action="store_true",
                    help="vla: the whole chain as one instruction for the entire episode (no planner, no switching)")
parser.add_argument("--learned_done", action="store_true",
                    help="vla trained with --done_frames: one command at a time, the VLA's own 6th-channel done flag moves the "
                         "pointer to the next command (no sim signal reaches it)")
parser.add_argument("--memory_done", action="store_true", help="with --learned_done: the VLA gets the WHOLE chain sentence every step and its own done-flag count as an extra state number (build_dataset --prompt_mode memory)")
parser.add_argument("--marks", action="store_true", help="with --learned_done: the WHOLE chain sentence, the commands the policy flagged done marked \"(done)\" in the text (build_dataset --prompt_mode marks)")
parser.add_argument("--k_norm", type=float, default=6, help="--memory_done: the state number is k / k_norm (build_dataset --k_norm)")
parser.add_argument("--done_diag", action="store_true", help="audit the policy's done flag: at every pointer advance, was that command's outcome true? (writes <out>.diag.csv)")
parser.add_argument("--done_vote", type=int, default=1, help="query the VLA this many times per step, done flag = min over the samples")
parser.add_argument("--recheck", default=None, help="marks: delay,window,low,need (ticks, ticks, flag, ticks): the policy re-asks itself whether its last marked command is done and erases the mark if not")
parser.add_argument("--edit", default=None, help="vla whole sentence: change the instruction mid-run (EDITS below); scored against the new instruction")
parser.add_argument("--perturb", default="none", choices=["none", "knock", "drop", "move", "premark", "oracle"],
                    help="9 Oct perturbation study (whole-sentence VLA): knock = once the chain's end result has held --perturb_after ticks, throw "
                         "that object to a free spot; drop = once \"pick up\" is marked done and the object is >5 cm up, put it back on the table "
                         "at a free spot (it fell out of the hand); move = at tick --perturb_after move the first command's object; premark = the "
                         "first --unmark commands start marked done; oracle = the marks are written by the sim's in-order check, not the done flag")
parser.add_argument("--perturb_after", type=int, default=10)
parser.add_argument("--unmark", type=int, default=0, help="knock / drop / move: at the disturbance, erase the policy's last N done marks "
                                                          "(done -> not done at inference); premark: N commands marked done from the start")
parser.add_argument("--ordered_hold",type=int, default=5, help="ticks (0.1 s) a command's outcome must hold to count in the in-order score (default 5 = 0.5 s)")
parser.add_argument("--done_thresh", type=float, default=0.85)       # 7 Oct (user): 0.85 / 5 ticks; the first evals used 0.5 / 3
parser.add_argument("--done_hold", type=int, default=5, help="consecutive ticks the done flag must stay above the threshold")
parser.add_argument("--val", action="store_true", help="validation layouts (seed 4321) instead of the eval states")
parser.add_argument("--name", default=None)
parser.add_argument("--out", default=None)
parser.add_argument("--keypoints", action="store_true", help="keypoint env and chains (object-only reward)")
parser.add_argument("--hold_fix", action="store_true", help="keypoint env with the hold fix (as the policy was trained)")
parser.add_argument("--release", action="store_true")
parser.add_argument("--time_cost", type=float, default=0.02)
parser.add_argument("--knock_penalty", action="store_true")
parser.add_argument("--rest_speed", type=float, default=0.0)
parser.add_argument("--front_view", default="far", choices=["far", "near"])
parser.add_argument("--state_history", action="store_true", help="vla: state = [state, previous action, past hand motion] (build_dataset --state_history)")
parser.add_argument("--history", type=int, default=0, help="vla: send the front image this many steps back (build_dataset --history)")
parser.add_argument("--action_filter", type=float, default=0.0, help="RL teacher trained with --action_filter (not for a VLA, which outputs the executed action)")
parser.add_argument("--derived", action="store_true", help="9 Oct: per-skill rest speed and hold time from the clips (needs keypoint_ref_real_straight_v2.npz)")
parser.add_argument("--ref", default=None, help="default: the stand-in reference of the env")
parser.add_argument("--bank", default=None)
parser.add_argument("--video", default=None)
parser.add_argument("--video_envs", default="0,1")
parser.add_argument("--video_split", action="store_true", help="one video per env in --video_envs (<video>_env<id>.mp4) instead of one stacked video")
parser.add_argument("--num_envs", type=int, default=None, help="only the first N start states (videos: the policy loader "
                    "also stores camera frames, so 100 envs with cameras do not fit next to a training run)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video or args.policy == "vla":
    args.enable_cameras = True
app = AppLauncher(args).app

import csv  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from sim.clips import SKILLS  # noqa: E402
from sim.envs import tasks  # noqa: E402
from sim.envs.keypoint_env import PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.envs.skill_env import NONE, OTHER, TARGET, PandaSkillEnv, PandaSkillEnvCfg  # noqa: E402
from sim.rl_cfg import RslPolicy  # noqa: E402

R, B = 0, 1
STACK = [("reach", R, NONE), ("pick-lift", R, NONE), ("place-down", R, OTHER)]
TO_TARGET = [("reach", R, NONE), ("pick-lift", R, NONE), ("place-down", R, TARGET)]
BLUE_ON_RED = [("reach", B, NONE), ("pick-lift", B, NONE), ("place-down", B, OTHER)]
# name: (commands, layouts from, final tasks.success check or "")
CHAINS = {
    "lift": (TO_TARGET[:2], "c1", "lift"),
    "c1": (TO_TARGET, "c1", "c1"),
    "c2": (STACK, "c2", "c2"),
    "c3": ([("push", B, TARGET)] + STACK, "c3", "c3"),
    "swap": (BLUE_ON_RED, "c2", ""),
    "unstack": (STACK + TO_TARGET, "c2", "c1"),
    "restack": (STACK + TO_TARGET + BLUE_ON_RED, "c2", ""),
}
P_R, P_B = [("pick-lift", R, NONE)], [("pick-lift", B, NONE)]
KP_STACK = P_R + [("place-down", R, OTHER)]
KP_TO_TARGET = P_R + [("place-down", R, TARGET)]
KP_BLUE_ON_RED = P_B + [("place-down", B, OTHER)]


def outcome(chain, s, start):
    """Lenient check (5 Oct, user): does the world show the chain's end result right now? Only the last command's
    outcome for the commanded cube counts; the steps in between, release and rest are not checked."""
    k, cube, dest = chain[-1]
    me, other = (s["blue"], s["red"]) if cube == B else (s["red"], s["blue"])
    if k == "pick-lift":
        return me[:, 2] - start[:, 2] > 0.05
    if dest == OTHER:
        return ((me[:, :2] - other[:, :2]).norm(dim=-1) < 0.03) & ((me[:, 2] - other[:, 2] - (tasks.HALF_Z[0] + tasks.HALF_Z[1])).abs() < 0.015)
    return ((me[:, :2] - s["target"][:, :2]).norm(dim=-1) < 0.03) & (me[:, 2] < 0.045)


LENIENT_HOLD = 5          # ticks (0.5 s) the outcome must hold


def outcome_cmd(cmd, s, start):
    """One command's lenient end outcome (same rules as outcome(); start = the commanded cube's position when the
    command began, for the pick-lift rise)."""
    return outcome([cmd], s, start)


KP_CHAINS = {
    "push": ([("push", R, TARGET)], "push", "push"),
    "lift": (P_R, "c1", "lift"),
    "lift_cyl": (P_B, "c1", ""),                          # 6 Oct (user): with VLA_OBJECTS=cube_cylinder, pick up the cylinder
    "c1_cyl": (P_B + [("place-down", B, TARGET)], "c1", ""),
    "push_cyl": ([("push", B, TARGET)], "push", ""),      # "push" layouts with the two object slots swapped (below)
    "c1": (KP_TO_TARGET, "c1", "c1"),
    "c2": (KP_STACK, "c2", "c2"),
    "c3": ([("push", B, TARGET)] + KP_STACK, "c3", "c3"),
    "swap": (KP_BLUE_ON_RED, "c2", ""),
    "unstack": (KP_STACK + KP_TO_TARGET, "c2", "c1"),
    "restack": (KP_STACK + KP_TO_TARGET + KP_BLUE_ON_RED, "c2", ""),
}

# 7 Oct evening, "change of mind": (old chain, commands kept j, new commands, time in s). At that time the instruction becomes
# old[:j] + new; the policy keeps its own done count / marks (nothing from the sim). Scored against the new instruction; the csv
# also says whether the old instruction's end result was reached instead.
EDITS = {
    "c2_to_target": ("c2", 1, [("place-down", R, TARGET)], 3.5),            # carrying the cube to the cylinder: "on the target instead"
    "c1_to_cyl": ("c1", 0, P_B + [("place-down", B, TARGET)], 1.0),         # before the grasp: "the cylinder instead"
    "c2_extend": ("c2", 2, P_R + [("place-down", R, TARGET)], 8.0),         # after the stack: "then put the cube on the target" (= unstack)
}


def random_chain(n, seed):
    """7 Oct evening, the length curve and the 20-command run: a random valid chain of n commands for the keypoint env
    (chain name rand<n>_<seed>). Pick-up / put-down pairs of a free object (nothing on top of it) to the target (if free and it
    is not there already) or onto the other object (onto an object on the target only as the last move); an odd n starts with "push the cylinder to the target" (c3 layouts)."""
    import random
    rng = random.Random(seed)
    chain, pos = [], {R: "table", B: "table"}           # table / target / on (on top of the other object)
    if n % 2:
        chain.append(("push", B, TARGET))
        pos[B] = "target"
    while len(chain) < n:
        moves = [(o, d) for o in (R, B) if pos[1 - o] != "on"
                 for d in (TARGET, OTHER) if (d == TARGET and "target" not in pos.values()) or
                 (d == OTHER and pos[o] != "on" and (pos[1 - o] != "target" or len(chain) + 2 >= n))]   # a stack on the target is a dead end (no "put it on the table")
        o, d = rng.choice(moves)
        chain += [("pick-lift", o, NONE), ("place-down", o, d)]
        pos[o] = "target" if d == TARGET else "on"
    return chain, ("c3" if n % 2 else "c2"), ""


def place_object(env, i, obj, rng, speed=0.0):
    """Perturbation study: put object `obj` (R cube / B cylinder) of env i on a free table spot, at least 10 cm from the other
    object, the target and the hand; speed > 0 throws it (that horizontal speed in a random direction, dropped from 3 cm)."""
    s = env.state()
    keep = [s[k][i, :2].cpu().numpy() for k in ("red", "blue", "target", "tcp")]
    keep.pop(obj)                         # its own current spot is allowed
    lo, hi = np.array(tasks.REGION[0]), np.array(tasks.REGION[1])
    for _ in range(500):
        p = rng.uniform(lo, hi)
        if all(np.linalg.norm(p - q) > 0.10 for q in keep):
            break
    z = tasks.HALF_Z[obj] + (0.03 if speed else 0.002)
    pose = torch.tensor([[p[0], p[1], z, 1, 0, 0, 0]], dtype=torch.float32, device=env.device)
    pose[:, :3] += env.table_origin[i:i + 1]
    a = rng.uniform(0, 2 * np.pi)
    vel = torch.tensor([[speed * np.cos(a), speed * np.sin(a), 0, 0, 0, 0]], dtype=torch.float32, device=env.device)
    ids = torch.tensor([i], device=env.device)
    o = env.blue if obj == B else env.red
    o.write_root_pose_to_sim(pose, env_ids=ids)
    o.write_root_velocity_to_sim(vel, env_ids=ids)


SECONDS_PER_COMMAND = 6.0
STATES = Path(__file__).parent / "eval_states"


def main():
    if args.edit:
        assert args.policy == "vla" and args.learned_done, "--edit changes the sentence of a whole-sentence VLA (--learned_done)"
        old, keep, new, edit_t = EDITS[args.edit]
        old_chain, layout_task, _ = KP_CHAINS[old]
        chain, check = old_chain[:keep] + new, ""
        args.chain = args.edit
        print(f"edit {args.edit}: {old_chain} -> at {edit_t} s: {chain}", flush=True)
    elif args.chain.startswith("rand"):
        chain, layout_task, check = random_chain(*map(int, args.chain[4:].split("_")))
        print(f"random chain {args.chain}: {chain}", flush=True)
    else:
        chain, layout_task, check = (KP_CHAINS if args.keypoints else CHAINS)[args.chain]
    if args.val:
        layouts = tasks.sample_layouts(layout_task, 100, torch.Generator().manual_seed(4321))
    else:
        layouts = torch.load(STATES / f"{layout_task}.pt")["layouts"]
    layouts = layouts[:args.num_envs]
    if args.chain == "push_cyl":      # the push layouts put the pushed object in the first slot: put the cylinder (second slot) there
        layouts = torch.cat([layouts[:, 2:4], layouts[:, 0:2], layouts[:, 4:6]], -1)
    n = len(layouts)
    Cfg, Env = (PandaKeypointEnvCfg, PandaKeypointEnv) if args.keypoints else (PandaSkillEnvCfg, PandaSkillEnv)
    extra_s = {"knock": 15, "drop": 10, "move": 5}.get(args.perturb, 0)        # time to redo the work after a disturbance
    cfg = Cfg(task=layout_task, final_check=check, episode_s=SECONDS_PER_COMMAND * max(len(chain), len(old_chain) if args.edit else 0) + 5 + extra_s,
              cameras=args.video is not None or args.policy == "vla",
              image_size=256 if args.policy == "vla" else 384, front_view=args.front_view)       # the VLA sees the image size it was trained on
    cfg.ref_path = args.ref or cfg.ref_path
    if args.perturb != "none":
        cfg.terminate_on_success = False      # the episode runs on after the work is done (scored by the lenient / in-order checks below)
    if args.keypoints:
        cfg.hold_fix, cfg.release, cfg.time_cost = args.hold_fix, args.release, args.time_cost
        cfg.knock_penalty, cfg.rest_speed = args.knock_penalty, args.rest_speed
        cfg.action_filter = args.action_filter
        cfg.derived = args.derived
    cfg.bank_path = args.bank or cfg.bank_path
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = Env(cfg)
    env.set_chain(chain)
    env.set_layouts(layouts)
    if args.policy == "vla":
        from sim.vla_policy import CommandVLAPolicy
        whole = None
        if args.whole_prompt:
            from vla.prompts import command_prompt
            whole = ", then ".join(command_prompt(k, c, d) for k, c, d in chain)
            print(f"whole prompt: {whole}", flush=True)
        policy = CommandVLAPolicy(args.vla_port, replan=args.replan, whole=whole, history=args.history,
                                  state_history=args.state_history, learned_done=(old_chain if args.edit else chain) if args.learned_done else None, memory=args.memory_done, marks=args.marks, k_norm=args.k_norm, vote=args.done_vote, recheck=[float(x) if i == 2 else int(x) for i, x in enumerate(args.recheck.split(','))] if args.recheck else None,
                                  done_thresh=args.done_thresh, done_hold=args.done_hold)
    else:
        policy = RslPolicy(args.policy[4:], env)
    if args.perturb != "none":
        assert args.policy == "vla" and args.learned_done, "--perturb acts on a whole-sentence VLA's own memory"
    if args.perturb in ("premark", "oracle"):
        policy.stage = torch.full((n,), args.unmark if args.perturb == "premark" else 0, dtype=torch.long)
        policy.streak = torch.zeros(n, dtype=torch.long)
    if args.perturb == "oracle":
        policy.done_thresh = 2.0              # the flag can never fire; the marks come from the sim below
    env.reset()
    name = args.name or (Path(args.policy[4:]).parent.name if args.policy != "vla" else "vla")

    rec = None
    if args.video:
        from sim.video import Recorder
        ids = [int(i) for i in args.video_envs.split(",")]
        cap = f'VLA prompt: "{whole}"' if args.policy == "vla" and whole else ""
        rec = Recorder(args.video, env_ids=ids, caption=cap)
        recs = {i: Recorder(args.video.replace(".mp4", f"_env{i}.mp4"), env_ids=[i], caption=cap) for i in ids} if args.video_split else {}

        def frame(e):
            # per frame: the command, how far the robot is from the clips' end state of that skill against the
            # done radius, and where the fingertips are relative to the cube's centre (rows are the recorded envs)
            s = e.state()
            skill, cube, _ = e.command()
            act = torch.where((cube == 1)[:, None], s["blue"], s["red"])
            parts = []
            for i in ids:
                k = skill[i].item()
                if args.keypoints and args.learned_done and getattr(policy, "stage", None) is not None:
                    from vla.prompts import command_prompt
                    ptr = min(int(policy.stage[i]), len(chain) - 1)
                    p = float(policy.done_p[i]) if hasattr(policy, "done_p") else float("nan")
                    parts.append(f"VLA pointer {ptr + 1}/{len(chain)}: \"{command_prompt(*chain[ptr])}\" | done flag {p:.2f} | "
                                 f"commands done in order {int(stage[i])}/{len(chain)}" + perturb_tag(i))
                    continue
                if args.keypoints:
                    done = e.completed_final[i].item()
                    parts.append(f"command {min(e.cur[i].item() + 1, len(chain))}/{len(chain)}: {SKILLS[k]} {'rb'[cube[i].item()]} | "
                                 f"waypoints {e.wp_reached[i].item()}/19 | cube to goal {e.d_goal[i] * 100:.1f} cm "
                                 f"(done < {e.done_tol[k] * 100:.1f}) | commands done {done}")
                    continue
                parts.append(f"{SKILLS[k]}: to clip end {e.end_err[i] * 100:.1f} cm (done < {e.ref_radius[k] * 100:.1f}), "
                             f"hand {(s['tcp'][i, 2] - act[i, 2]) * 100:+.1f} cm vs cube centre, cube up {(act[i, 2] - 0.025) * 100:.1f}")
            if recs:
                img = e.images()
                for j, i in enumerate(ids):
                    recs[i].add(img, f"t {e.ticks[i].item() * 0.1:4.1f}s | " + parts[j])
            else:
                rec.add(e.images(), f"t {e.ticks[ids[0]].item() * 0.1:4.1f}s | " + " || ".join(parts))

        env.tick_callback = frame

    captured = torch.zeros(n, dtype=torch.bool, device=env.device)
    rows = [None] * n
    s0 = env.state()
    start = (s0["blue"] if chain[-1][1] == B else s0["red"]).clone()
    held = torch.zeros(n, dtype=torch.long, device=env.device)
    # diagnostics (6 Oct): how far the cube the first command names, and the other cube, rose above their start
    c0 = lambda sd: (sd["blue"], sd["red"]) if chain[0][1] == B else (sd["red"], sd["blue"])
    first0, other0 = (t.clone() for t in c0(s0))
    first_rise = torch.zeros(n, device=env.device)
    other_rise = torch.zeros(n, device=env.device)
    lenient = torch.zeros(n, dtype=torch.bool, device=env.device)
    if args.edit:                     # did it reach the OLD instruction's end result (it ignored the change)?
        old_start = (s0["blue"] if old_chain[-1][1] == B else s0["red"]).clone()
        old_held = torch.zeros(n, dtype=torch.long, device=env.device)
        old_ok = torch.zeros(n, dtype=torch.bool, device=env.device)
    # in-order (6 Oct, user): command j counts once its lenient outcome has held 0.5 s AFTER command j-1 counted;
    # stage = commands counted so far, snap = both cubes' positions when the current command began
    stage = torch.zeros(n, dtype=torch.long, device=env.device)
    ohold = torch.zeros(n, dtype=torch.long, device=env.device)
    snap = {"red": s0["red"].clone(), "blue": s0["blue"].clone()}
    # psnap[key][i, j] = that object's position when env i's pointer arrived at command j (kept when a mark is erased and re-made)
    diag, pstage = [], torch.zeros(n, dtype=torch.long, device=env.device)
    psnap = {key: s0[key][:, None].repeat(1, len(chain) + 1, 1).clone() for key in ("red", "blue")}
    # perturbation study (9 Oct): disturbed = when (tick), recovered = the end result held 0.5 s after the disturbance
    dist_t = torch.full((n,), -1, dtype=torch.long, device=env.device)
    recovered = torch.zeros(n, dtype=torch.bool, device=env.device)
    t_rec = torch.full((n,), float("nan"), device=env.device)
    stage_at = torch.full((n,), -1, dtype=torch.long)
    lifted = torch.zeros(n, dtype=torch.long, device=env.device)
    drng = np.random.default_rng(0)
    obj0 = chain[0][1]
    perturb_tag = lambda i: (f" | {args.perturb.upper()} at {dist_t[i].item() * 0.1:.1f}s" + (f", {args.unmark} marks erased" if args.unmark else "")
                             if dist_t[i] >= 0 else (" | marks from the sim" if args.perturb == "oracle" else ""))
    if args.perturb == "premark":
        stage += args.unmark
    for step in range(int(env.max_episode_length) + 1):
        if args.edit and step == round(edit_t * 10):
            policy.chain = chain      # the new sentence from now on; policy.stage (its own count) is kept
        action, _ = policy.act(env)
        if args.done_diag and getattr(policy, "stage", None) is not None:
            stg = policy.stage.to(env.device)
            moved = (stg > pstage) & ~captured
            if moved.any():       # the pointer moved this tick: was the command it left really done (the state before the step)?
                sdp = env.state()
                for i in moved.nonzero().squeeze(-1).tolist():
                    j = int(pstage[i]); cmd = chain[j]
                    si = {k_: v[i:i + 1] for k_, v in sdp.items() if torch.is_tensor(v)}
                    st = (psnap["blue"] if cmd[1] == B else psnap["red"])[i:i + 1, j]
                    diag.append({"env": i, "tick": step, "command": j, "skill": cmd[0], "ok": int(outcome_cmd(cmd, si, st)[0])})
                for i in moved.nonzero().squeeze(-1).tolist():
                    for key in ("red", "blue"):
                        psnap[key][i, int(stg[i])] = sdp[key][i]
            back = (stg < pstage) & ~captured
            for i in back.nonzero().squeeze(-1).tolist():   # the policy erased a mark: was that command really NOT done?
                j = int(stg[i]); cmd = chain[j]
                si = {k_: v[i:i + 1] for k_, v in env.state().items() if torch.is_tensor(v)}
                st = (psnap["blue"] if cmd[1] == B else psnap["red"])[i:i + 1, j]
                diag.append({"env": i, "tick": step, "command": j, "skill": cmd[0], "ok": -1 - int(outcome_cmd(cmd, si, st)[0])})
            pstage = stg.clone()
        env.step(action)
        sd = env.state()
        held = torch.where(outcome(chain, sd, start) & ~captured, held + 1, torch.zeros_like(held))
        f_now, o_now = c0(sd)
        first_rise = torch.where(~captured, torch.maximum(first_rise, f_now[:, 2] - first0[:, 2]), first_rise)
        other_rise = torch.where(~captured, torch.maximum(other_rise, o_now[:, 2] - other0[:, 2]), other_rise)
        lenient |= held >= LENIENT_HOLD
        if args.edit:
            old_held = torch.where(outcome(old_chain, sd, old_start) & ~captured, old_held + 1, torch.zeros_like(old_held))
            old_ok |= old_held >= LENIENT_HOLD
        ok_now = torch.zeros(n, dtype=torch.bool, device=env.device)
        for j, cmd in enumerate(chain):
            m = stage == j
            if m.any():
                st = snap["blue"] if cmd[1] == B else snap["red"]
                ok_now |= m & outcome_cmd(cmd, sd, st)
        ohold = torch.where(ok_now & ~captured, ohold + 1, torch.zeros_like(ohold))
        adv = ohold >= args.ordered_hold
        if adv.any():
            stage = torch.where(adv, stage + 1, stage)
            ohold = torch.where(adv, torch.zeros_like(ohold), ohold)
            for key in ("red", "blue"):
                snap[key] = torch.where(adv[:, None], sd[key], snap[key])
        if args.perturb == "oracle" and (stage.cpu() != policy.stage).any():
            policy.stage = stage.cpu().clone()
            policy.k = policy.replan
        if args.perturb in ("knock", "drop", "move"):
            ok0 = dist_t < 0
            if args.perturb == "knock":
                trig, obj = (held >= args.perturb_after) & ok0, chain[-1][1]
            elif args.perturb == "drop":
                up = (sd["blue"] if obj0 == B else sd["red"])[:, 2] - (s0["blue"] if obj0 == B else s0["red"])[:, 2] > 0.05
                lifted = torch.where(up, lifted + 1, torch.zeros_like(lifted))
                marked = (policy.stage >= 1).to(env.device)        # "pick up" already marked done, the object still in the air
                trig, obj = (lifted >= 3) & marked & ok0, obj0
            else:
                trig, obj = torch.full_like(ok0, step == args.perturb_after) & ok0, obj0
            for i in (trig & ~captured).nonzero().squeeze(-1).tolist():
                place_object(env, i, obj, drng, speed=0.2 if args.perturb == "knock" else 0.0)
                dist_t[i], held[i] = step, 0
                stage_at[i] = int(policy.stage[i])
                if args.unmark:              # done -> not done, at inference: the policy's own memory is edited, nothing else
                    policy.stage[i] = max(int(policy.stage[i]) - args.unmark, 0)
                    policy.streak[i] = 0
                    policy.k = policy.replan
            rec_now = (dist_t >= 0) & (held >= LENIENT_HOLD) & ~recovered & ~captured
            t_rec = torch.where(rec_now, (step - dist_t).float() * 0.1, t_rec)
            recovered |= rec_now
        new = env.episode_done & ~captured
        for i in new.nonzero().squeeze(-1).tolist():
            ok = bool(env.last_episode["success"][i])
            done = int(env.last_episode["completed"][i])
            rows[i] = {"model": name, "chain": args.chain, "state": i, "success": int(ok), "lenient": int(ok or lenient[i]),
                       **({"old_outcome": int(old_ok[i])} if args.edit else {}),
                       **({"disturbed": int(dist_t[i] >= 0), "t_disturb": round(dist_t[i].item() * 0.1, 1), "marks_at_disturb": int(stage_at[i]),
                           "recovered": int(recovered[i]), "t_recover": round(t_rec[i].item(), 1)} if args.perturb in ("knock", "drop", "move") else {}),
                       "ordered_done": len(chain) if ok else int(stage[i]), "ordered": int(ok or stage[i] >= len(chain)),   # a strict success ends the episode before 0.5 s
                       "first_rise_cm": round(100 * first_rise[i].item(), 1), "other_rise_cm": round(100 * other_rise[i].item(), 1),
                       "commands_done": done,
                       "n_commands": len(chain),
                       "failed_at": "" if ok else (f"{done + 1}:{chain[done][0]}" if done < len(chain) else "final check"),
                       "knocked": int(env.last_knocked[i]),
                       "time": round(env.last_episode["time"][i].item(), 2),
                       "jerk": round(env.last_episode["jerk"][i].item(), 3),
                       "peak_force": round(env.last_episode["peak_force"][i].item(), 2)}
        captured |= new
        if captured.all():
            break

    out = Path(args.out or f"runs/eval/{name}_chain_{args.chain}{'_val' if args.val else ''}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    succ = sum(r["success"] for r in rows)
    fails = {}
    for r in rows:
        if not r["success"]:
            fails[r["failed_at"]] = fails.get(r["failed_at"], 0) + 1
    mean_done = sum(r["commands_done"] for r in rows) / n
    print(f"{name} on {args.chain} ({len(chain)} commands): success {succ}/{n}, mean commands done {mean_done:.2f}, "
          f"failures at {dict(sorted(fails.items()))}, lenient {sum(r['lenient'] for r in rows)}/{n}, "
          f"in-order {sum(r['ordered'] for r in rows)}/{n} (mean commands in order {sum(r['ordered_done'] for r in rows) / n:.2f}/{len(chain)}) -> {out}", flush=True)
    if args.done_diag and diag:
        retr = [r for r in diag if r["ok"] < 0]
        diag = [r for r in diag if r["ok"] >= 0]
        if retr:
            print(f"recheck: {len(retr)} marks erased, {sum(r['ok'] == -1 for r in retr)} of them rightly (the command was not done)", flush=True)
        with open(str(out) + ".diag.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(diag[0].keys()))
            w.writeheader()
            w.writerows(diag)
        frac = lambda rows_: f"{sum(r['ok'] for r in rows_)}/{len(rows_)}"
        by_skill = {k_: frac([r for r in diag if r["skill"] == k_]) for k_ in sorted({r["skill"] for r in diag})}
        print(f"done flag audit: {len(diag)} advances, {sum(r['ok'] for r in diag)} of them really finished ({100 * sum(r['ok'] for r in diag) / len(diag):.0f}%); "
              f"by skill {by_skill}; first 2 commands {frac([r for r in diag if r['command'] < 2])}, later {frac([r for r in diag if r['command'] >= 2])}; "
              f"mean advances per episode {len(diag) / n:.2f} of {len(chain)}", flush=True)
    if args.perturb in ("knock", "drop", "move"):
        dn = [r for r in rows if r["disturbed"]]
        tr = sorted(r["t_recover"] for r in dn if r["recovered"])
        print(f"perturb {args.perturb} (unmark {args.unmark}): disturbed {len(dn)}/{n}, end result reached again {sum(r['recovered'] for r in dn)}/{max(len(dn), 1)}, "
              f"median time {tr[len(tr) // 2] if tr else float('nan')} s, marks at the disturbance {sorted({r['marks_at_disturb'] for r in dn})}", flush=True)
    if args.edit:
        print(f"edit {args.edit}: old instruction's end result reached {sum(r['old_outcome'] for r in rows)}/{n}", flush=True)
    if args.policy == "vla" and policy.chain is not None and policy.stage is not None:
        print(f"learned done: mean pointer at the end {policy.stage.float().mean() + (0 if args.memory_done or args.marks else 1):.2f}/{len(chain)} commands reached, "
              f"mean commands the sim counted {mean_done:.2f}", flush=True)
    if rec:
        for r_ in (recs.values() if recs else [rec]):
            r_.save()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
