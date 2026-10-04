"""Does the keypoint reward (sim/envs/keypoint_env.py) score the right behaviour highest? Checked before any RL.

Runs one chain with a scripted policy that follows the env's current command and with cheating
variants of it (a fifth of the envs each), on training layouts. Reports per behaviour: commands done,
chain done (the reward's own "done"), task success by tasks.success (the eval check, never used by the
reward), and the episode return.

    PYTHONPATH=. ./isaaclab.sh -p sim/check_keypoint_reward.py --headless --task c1
Behaviours: scripted; idle (no motion); open_hand (fingers never close, so nothing is lifted);
shove (drives the hand through the cube at cube height instead of grasping it: the cube is knocked and
spun, never lifted); drop (lets go over the destination at carry height); disturb (first knocks the other
cube, then does the task); spin (turns the wrist ~45 degrees while holding the cube); drag (place-down by
lowering the cube to the table and sliding it to the destination); hover (holds the cube 5 cm above
the destination and never sets it down). Also checks the corner distance's cube symmetry.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", required=True, choices=["push", "lift", "c1", "c2", "c3"])
parser.add_argument("--per_behaviour", type=int, default=32)
parser.add_argument("--ref", default="data/processed/keypoint_ref_standin.npz")
parser.add_argument("--debug", action="store_true", help="print env 0 every second")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.keypoint_env import NONE, OTHER, PICK, PLACE, TARGET, PandaKeypointEnv, PandaKeypointEnvCfg  # noqa: E402
from sim.envs.panda_env import MAX_DXYZ, MAX_DYAW  # noqa: E402
from sim.expert import ScriptedExpert  # noqa: E402
from sim.keypoint_follower import Follower  # noqa: E402

CHAINS = {
    "push": [("push", 0, TARGET)],
    "lift": [("pick-lift", 0, NONE)],
    "c1": [("pick-lift", 0, NONE), ("place-down", 0, TARGET)],
    "c2": [("pick-lift", 0, NONE), ("place-down", 0, OTHER)],
    "c3": [("push", 1, TARGET), ("pick-lift", 0, NONE), ("place-down", 0, OTHER)],
}
BEHAVIOURS = ["scripted", "idle", "open_hand", "shove", "drop", "disturb", "spin", "drag", "hover"]
def main():
    k = args.per_behaviour
    n = k * len(BEHAVIOURS)
    cfg = PandaKeypointEnvCfg(task=args.task, terminate_on_success=False, ref_path=args.ref, episode_s=20.0)
    cfg.scene.num_envs = n
    cfg.sim.device = args.device
    env = PandaKeypointEnv(cfg)
    env.set_chain(CHAINS[args.task])
    env.set_layouts(tasks.sample_layouts(args.task, n, torch.Generator().manual_seed(555)))
    env.reset()
    dev = env.device
    # cube symmetry of the corner distance: same pose turned a quarter / flipped = 0, turned 45 degrees > 0
    from isaaclab.utils.math import quat_from_euler_xyz
    pos = torch.tensor([[0.0, 0.0, 0.025]], device=dev)
    z = torch.zeros(1, device=dev)
    q = lambda r, p, y: quat_from_euler_xyz(r + z, p + z, y + z)  # noqa: E731
    ref = env.corners(pos, q(0.0, 0.0, 0.0))
    for name, rpy in [("quarter turn", (0, 0, torch.pi / 2)), ("upside down", (torch.pi, 0, 0)),
                      ("45 degrees", (0, 0, torch.pi / 4)), ("10 degrees", (0, 0, 0.175))]:
        d = env.corner_dist(env.corners(pos, q(*rpy)), ref).item()
        print(f"symmetry: cube {name:12s} -> corner distance {d * 100:.2f} cm", flush=True)
    group = torch.arange(n, device=dev) // k
    ret = torch.zeros(n, device=dev)
    task_hold = torch.zeros(n, dtype=torch.long, device=dev)
    task_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    chain_ok = torch.zeros(n, dtype=torch.bool, device=dev)
    best = torch.zeros(n, dtype=torch.long, device=dev)
    from sim.envs.keypoint_env import yaw_of_quat
    yaw0 = yaw_of_quat(env.red.data.root_quat_w).clone()
    turned = torch.zeros(n, device=dev)

    def on_tick(e):
        s = e.state()
        s["was_lifted"] = e.was_lifted
        ok = tasks.success(args.task, s) if args.task != "lift" else s["red"][:, 2] > 0.10
        task_hold.copy_(torch.where(ok, task_hold + 1, torch.zeros_like(task_hold)))
        task_ok.logical_or_(task_hold >= tasks.SUCCESS_HOLD)
        chain_ok.logical_or_(e.done_ok)
        best.copy_(torch.maximum(best, e.completed_final))
        dy = torch.remainder(yaw_of_quat(e.red.data.root_quat_w) - yaw0 + torch.pi / 4, torch.pi / 2) - torch.pi / 4
        turned.copy_(torch.maximum(turned, dy.abs()))
        if args.debug and e.ticks[0] % 5 == 0:
            ap = e.objects()[0]
            print(f"t {e.ticks[0].item():3d} cmd {e.cur[0].item()} phase {fol.phase[0].item()} wp {e.wp_reached[0].item():2d} "
                  f"d_next {e.d_next[0] * 100:5.1f} d_goal {e.d_goal[0] * 100:5.1f} cm | cube {[round(v * 100, 1) for v in ap[0].tolist()]} "
                  f"goal {[round(v * 100, 1) for v in e.goal_corners[0].mean(0).tolist()]} tcp {[round(v * 100, 1) for v in s['tcp'][0].tolist()]} "
                  f"gap {s['grip'][0] * 100:.1f} speed {s['red_speed'][0]:.3f} chold {e.chold[0].item()}", flush=True)

    fol = Follower(env)
    expert = None
    if args.task == "push":            # the project's scripted pusher (100% on push) instead of the follower's crude one
        expert = ScriptedExpert()
        expert.reset(env)
    spin0 = torch.full((n,), torch.nan, device=dev)
    env.tick_callback = on_tick
    for _ in range(int(env.max_episode_length) - 2):
        a = fol.act(env)
        if expert is not None:
            a = expert.act(env)[0].clone()
        a[group == 1] = 0.0
        a[group == 2, 4] = 1.0                                            # fingers never close
        shove = (group == 3) & (env.command()[0] == PICK) & (fol.phase >= 1)
        if shove.any():                                                   # drive through the cube at its height
            ap = env.objects()[0]
            through = ap + torch.tensor([0.15, 0.0, 0.0], device=dev)
            a[shove, :3] = ((through - env.cmd) / MAX_DXYZ).clamp(-1, 1)[shove]
            a[shove, 4] = 1.0
        drop = (group == 4) & (env.command()[0] == PLACE) & (fol.phase >= 1)
        a[drop, :3] = 0.0
        a[drop, 4] = 1.0
        skill = env.command()[0]
        ap, _, _, op, _, _ = env.objects()
        tcp = env.state()["tcp"]
        up = lambda h: torch.tensor([0, 0, h], device=dev)  # noqa: E731
        # disturb: the first 3 s of the episode go into the other cube at its height
        dis = (group == 5) & (env.ticks < 30)
        a[dis, :3] = ((op + up(0.0) - env.cmd) / MAX_DXYZ).clamp(-1, 1)[dis]
        # spin: once lifting, turn the wrist 45 degrees from where it was (the held cube turns with it)
        lifting = (group == 6) & (skill == PICK) & (fol.phase == 3)
        spin0.copy_(torch.where(lifting & torch.isnan(spin0), env.cmd_yaw, spin0))
        a[lifting, 3] = ((spin0 + torch.pi / 4 - env.cmd_yaw) / MAX_DYAW).clamp(-1, 1)[lifting]
        goal = env.goal_corners.mean(1)
        held_off = tcp - ap
        # drag: place-down along the table: down to table height where it is, slide over, then open
        drag = (group == 7) & (skill == PLACE)
        low = (ap[:, 2] - goal[:, 2]).abs() < 0.006
        there = (ap[:, :2] - goal[:, :2]).norm(dim=-1) < 0.008
        tgt = torch.where(low[:, None], torch.cat([goal[:, :2], ap[:, 2:]], -1), torch.cat([ap[:, :2], goal[:, 2:]], -1))
        a[drag, :3] = ((tgt + held_off - env.cmd) / MAX_DXYZ).clamp(-1, 1)[drag]
        a[drag, 4] = torch.where(there & low, 1.0, -1.0)[drag]
        # hover: hold the cube 5 cm over the destination, never lower or open
        hov = (group == 8) & (skill == PLACE)
        a[hov, :3] = ((goal + up(0.05) + held_off - env.cmd) / MAX_DXYZ).clamp(-1, 1)[hov]
        a[hov, 4] = -1.0
        env.step(a)
        if expert is not None:
            expert.update(env)
        ret += env.reward_buf if hasattr(env, "reward_buf") else 0
    chain = " > ".join(s for s, _, _ in CHAINS[args.task])
    nc = len(CHAINS[args.task])
    lines = [f"\n{args.task}: {chain} ({k} envs per behaviour, {env.max_episode_length * 0.1:.0f} s, {args.ref})",
             f"{'behaviour':10s} {'cmds done':>9s} {'chain done':>10s} {'task ok':>8s} {'agree':>6s} {'return':>8s} {'red turned':>10s}"]
    for g, name in enumerate(BEHAVIOURS):
        m = group == g
        agree = (chain_ok[m] == task_ok[m]).float().mean()
        lines.append(f"{name:10s} {best[m].float().mean():6.2f}/{nc} {chain_ok[m].float().mean():10.0%} "
                     f"{task_ok[m].float().mean():8.0%} {agree:6.0%} {ret[m].mean():8.1f} {turned[m].mean().rad2deg():8.0f} deg")
    print("\n".join(lines), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    app.close()
