"""Check the vocabulary action: z = 0 plays each skill's mean motion, random z gives varied, smooth motions.

    PYTHONPATH=. ./isaaclab.sh -p sim/test_vocab_action.py --headless --enable_cameras --video media/08_vocab_action.mp4

Env 0 runs reach -> pick-lift -> place-down -> reach with z = 0; envs 1-2 run the same skills
with random z; env 3 pushes with z = 0. Prints where each primitive ends relative to where it started.
"""
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--vocab", default="data/processed/vocab_standin.pt")
parser.add_argument("--video", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

from sim.envs import tasks  # noqa: E402
from sim.envs.vocab_env import PandaVocabEnv, PandaVocabEnvCfg  # noqa: E402
from sim.video import Recorder  # noqa: E402


def main():
    cfg = PandaVocabEnvCfg(task="c1", vocab_path=args.vocab, cameras=args.video is not None, image_size=384,
                           terminate_on_success=False)
    cfg.scene.num_envs = 4
    cfg.sim.device = args.device
    env = PandaVocabEnv(cfg)
    env.set_layouts(tasks.sample_layouts("c1", 4, torch.Generator().manual_seed(3)))
    env.reset()
    skills = env.skills
    seq = ["reach", "pick-lift", "place-down", "reach"]
    rec = None
    if args.video:
        rec = Recorder(args.video, env_ids=[0, 1], caption="vocabulary action: top z = 0, bottom random z")
        env.tick_callback = lambda e: rec.add(e.images(), f"skill: {skills[int(e.skill[0])]} / {skills[int(e.skill[1])]}")
    torch.manual_seed(0)
    for name in seq:
        a = torch.zeros(4, len(skills) + env.latent_dim, device=env.device)
        a[:3, skills.index(name)] = 1.0
        a[3, skills.index("push")] = 1.0
        a[1:3, len(skills):] = torch.randn(2, env.latent_dim, device=env.device)
        start = env.to_table(env.tcp_w()).clone()
        env.step(a)
        end = env.to_table(env.tcp_w())
        d = (end - start) * 100
        print(f"{name:10s} z=0: moved {[round(v, 1) for v in d[0].tolist()]} cm, gap {env.grip_gap()[0] * 1000:4.1f} mm | "
              f"random z: {[round(v, 1) for v in d[1].tolist()]}, {[round(v, 1) for v in d[2].tolist()]} | "
              f"push env moved {[round(v, 1) for v in d[3].tolist()]}", flush=True)
    s = env.state()
    print(f"env 0 red cube at {[round(v, 3) for v in s['red'][0].tolist()]}, target {[round(v, 3) for v in s['target'][0].tolist()]}")
    if rec:
        rec.save()
    env.close()


if __name__ == "__main__":
    main()
    app.close()
