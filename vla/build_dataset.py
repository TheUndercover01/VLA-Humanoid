"""Convert replay/rollout npz files (sim/replay_clips.py and later expert rollouts) into a local LeRobot dataset.

Run in the lerobot env:
    python -m vla.build_dataset --src data/processed/replays/standin --repo_id local/b1_standin \
        --root /media/storage/ayush/vla_data/lerobot/b1_standin
Each npz is one episode: front, wrist (T, H, W, 3) uint8, state (T, 6), action (T, A), and either prompt
(one instruction for the episode) or prompts (T,): one per frame, as recorded by sim/record_keypoint_rollouts.py
(the instruction of the command active at that frame).
A = 5 for the raw action (10 fps); vocabulary rollouts (one frame per ~2 s primitive) store
[skill one-hot | z] and use --fps 1 (LeRobot needs an integer rate; only the frame order matters).
--prompt_mode sequence (a VLA that gets the whole task, no planner): every frame of an episode gets the episode's
whole instruction sequence ("pick up the red cube, then put the red cube on the blue cube"), and each command is
also added as its own episode with its own instruction, so the training prompts are 1 or 2 steps long and the
VLA has to tell from the images which part of the sequence it is at.
--prompt_mode memory (7 Oct, user: "give the WHOLE prompt at once and use the done flag with the whole prompt, not a single prompt"):
every frame gets the whole instruction sentence of its episode, and the state carries one extra number, the policy's memory of
how many commands of the sentence are done (k / 6). k is the number of finished commands: in training the true count, at inference
the count of the policy's own done flags (the 6th action channel, see --done_frames), so nothing from the simulator reaches it.
The episodes have at most 2 commands, but the evaluation sentences have up to 6, so the sentence is padded with random commands
before and after the real ones (0-3 each, per episode): k then runs from 0 to 5 and the policy has to learn "do command number
k + 1 of the sentence, whatever the sentence holds", not "do the first one". No longer episodes are used, only a longer text.

--pad even (7 Oct evening): with the 0-3 padding above the real commands sit at k = 0..4 only, so "do command 6" (k = 5, the end of restack)
never occurs in training and k = 4 is rare (the memory student got restack 5/100, at most 4-5 of 6 in order). With --pad even the
number of padded commands before the real ones is drawn uniformly so that every k in 0..5 is equally common, and the sentence is
filled after them up to at most 8 commands (--max_cmds; 20 for the 20-command run, with --k_norm 20). --dup_push N adds every episode with a push N more times (another padding each).

--prompt_mode marks (7 Oct evening, "how should a VLA keep its place?"): the same whole padded sentence, but instead of the number k in the
state, the commands already done carry the word "(done)" in the text ("pick up the cube (done), then put the cube on the cylinder,
then ..."). At inference the marks are written by the policy's own done flag. The state stays the plain 6 numbers.
"""
import argparse
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from vla.prompts import marked_sentence

STATE_NAMES = ["tcp_x", "tcp_y", "tcp_z", "sin_yaw", "cos_yaw", "gripper_gap"]
# --state_history: the state also carries the previous action and how the hand moved over the last 0.5 s and 1 s
HIST_STEPS = (5, 10)
HIST_NAMES = STATE_NAMES + [f"prev_{n}" for n in ["dx", "dy", "dz", "dyaw", "grip"]] + \
    [f"{n}_{k}back" for k in HIST_STEPS for n in STATE_NAMES]
RAW_ACTION_NAMES = ["dx", "dy", "dz", "dyaw", "grip"]


def _vocab():
    from vla.prompts import MULTI, command_prompt, multi_vocab
    if MULTI:       # 8 Oct: made-up objects; the held-out object and pairs never appear in the padding either
        return multi_vocab(holdout=True)
    return [command_prompt(*c) for c in [("pick-lift", 0, 2), ("pick-lift", 1, 2), ("place-down", 0, 0), ("place-down", 1, 0),
                                         ("place-down", 0, 1), ("place-down", 1, 1), ("push", 0, 0), ("push", 1, 0)]]


VOCAB = _vocab()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+", help="one or more folders of npz episodes")
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max_episodes", type=int, default=None)
    ap.add_argument("--streaming", action="store_true", help="encode videos in background threads while adding frames")
    ap.add_argument("--state_history", action="store_true",
                    help="observation.state = [state, previous action, state 5 and 10 steps back minus state]: the label of a "
                         "filtered teacher is mostly its previous action, which the images do not show")
    ap.add_argument("--history", type=int, default=0,
                    help="add observation.images.front_past: the front image this many frames earlier (SmolVLA uses one frame)")
    ap.add_argument("--shard", default="", help="i/n: build only every n-th episode starting at i (parallel builds, then aggregate)")
    ap.add_argument("--prompt_mode", default="step", choices=["step", "sequence", "memory", "marks"])
    ap.add_argument("--pad", default="legacy", choices=["legacy", "even"], help="memory/marks padding, see the docstring")
    ap.add_argument("--max_cmds", type=int, default=8, help="--pad even: the longest padded sentence (commands); the real ones cover k = 0..max_cmds - 1")
    ap.add_argument("--k_norm", type=float, default=6, help="memory: the state number is k / k_norm")
    ap.add_argument("--dup_push", type=int, default=0, help="memory/marks: add episodes with a push this many more times")
    ap.add_argument("--done_frames", type=int, default=0,
                    help="6 Oct (user: the VLA predicts when a command is finished, nothing outside tells it): add a 6th action "
                         "channel, 1 on the last N frames of every command's segment and 0 elsewhere, so the VLA learns to say "
                         "\"this command is done\" itself. Needs --prompt_mode step (the prompt is the current command only).")
    args = ap.parse_args()

    files = [f for src in args.src for f in sorted(Path(src).glob("*.npz"))[:args.max_episodes]]   # cap per folder
    if args.shard:
        i, n = map(int, args.shard.split("/"))
        files = files[i::n]
    first = np.load(files[0])
    assert not args.done_frames or args.prompt_mode in ("step", "memory", "marks"), "--done_frames needs --prompt_mode step, memory or marks"
    a_dim = first["action"].shape[1]
    action_names = RAW_ACTION_NAMES if a_dim == 5 else [f"a{i}" for i in range(a_dim)]
    if args.done_frames:
        a_dim, action_names = a_dim + 1, action_names + ["done"]
    h, w = first["front"].shape[1:3]
    img = {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}
    mem = args.prompt_mode == "memory"
    features = {
        "observation.images.front": img,
        "observation.images.wrist": img,
        **({"observation.images.front_past": img} if args.history else {}),
        "observation.state": {"dtype": "float32", "shape": (len(HIST_NAMES),) if args.state_history else (7,) if mem else (6,),
                              "names": HIST_NAMES if args.state_history else STATE_NAMES + ["done_count"] if mem else STATE_NAMES},
        "action": {"dtype": "float32", "shape": (a_dim,), "names": action_names},
    }
    if Path(args.root).exists():
        shutil.rmtree(args.root)
    ds = LeRobotDataset.create(args.repo_id, args.fps, features, root=args.root, robot_type="franka_panda_sim",
                               use_videos=True, streaming_encoding=args.streaming, encoder_threads=8 if args.streaming else None)
    def state_of(ep, t):
        s = ep["state"].astype(np.float32)
        if not args.state_history:
            return s[t]
        prev = ep["action"][t - 1].astype(np.float32) if t > 0 else np.zeros(5, np.float32)
        return np.concatenate([s[t], prev] + [s[max(t - k, 0)] - s[t] for k in HIST_STEPS])

    def add(ep, frames, prompt_of, done=None, count=None):
        for t in frames:
            label = ep["action"][t].astype(np.float32)
            if done is not None:
                label = np.append(label, np.float32(done[t]))
            past = {"observation.images.front_past": ep["front"][max(t - args.history, 0)]} if args.history else {}
            ds.add_frame({"observation.images.front": ep["front"][t], "observation.images.wrist": ep["wrist"][t], **past,
                          "observation.state": state_of(ep, t) if count is None else np.append(state_of(ep, t), np.float32(count[t] / args.k_norm)),
                          "action": label, "task": prompt_of(t)})
        ds.save_episode()

    for f in files:
        ep = np.load(f, allow_pickle=True)
        T = len(ep["action"])
        prompts = [str(p) for p in ep["prompts"]] if "prompts" in ep.files else [str(ep["prompt"])] * T
        if args.prompt_mode == "step":
            done = None
            if args.done_frames:       # last N frames of each command's segment (the episode end closes the last one)
                ends = [t for t in range(1, T) if prompts[t] != prompts[t - 1]] + [T]
                done = np.zeros(T, np.float32)
                for e in ends:
                    done[max(e - args.done_frames, 0):e] = 1
            add(ep, range(T), lambda t: prompts[t], done)
            continue
        if args.prompt_mode in ("memory", "marks"):
            if "cidx" in ep.files:       # 9 Oct, episodes with work undone and redone: the env's own command index and done-check per tick
                cmds = [c.strip() for c in str(ep["commands"]).split(" > ")]
                dn = ep["dnow"].astype(bool)
                done = np.array([dn[t:t + args.done_frames].any() for t in range(T)], np.float32)   # 1 from done_frames before the check holds, while it holds
                base_count = ep["cidx"].astype(np.float32)
            else:
                ends = [t for t in range(1, T) if prompts[t] != prompts[t - 1]] + [T]
                done = np.zeros(T, np.float32)
                for e in ends:
                    done[max(e - args.done_frames, 0):e] = 1
                cmds = [prompts[0]] + [prompts[t] for t in range(1, T) if prompts[t] != prompts[t - 1]]
                base_count = np.zeros(T, np.float32)
                for j, e0 in enumerate([0] + ends[:-1]):
                    base_count[e0:ends[j]] = j
            copies = 1 + (args.dup_push if any(c.startswith("push") for c in cmds) else 0)
            for copy in range(copies):
                rng = np.random.default_rng(__import__("zlib").crc32(f"{f.name}{copy or ''}".encode()))
                if args.pad == "even":     # the real commands start at k0 = 0..max_cmds-n, so every k < max_cmds is covered
                    n_pre = int(rng.integers(0, args.max_cmds - len(cmds) + 1))
                    n_suf = int(rng.integers(0, min(3, args.max_cmds - n_pre - len(cmds)) + 1))
                else:
                    n_pre, n_suf = int(rng.integers(0, 4)), None
                pre = [VOCAB[i] for i in rng.integers(0, len(VOCAB), n_pre)]
                suf = [VOCAB[i] for i in rng.integers(0, len(VOCAB), int(rng.integers(0, 4)) if n_suf is None else n_suf)]
                count = len(pre) + base_count
                if args.prompt_mode == "memory":
                    text = ", then ".join(pre + cmds + suf)
                    add(ep, range(T), lambda t: text, done, count)
                else:                      # marks: the commands before number k + 1 carry "(done)", no count in the state
                    seq = pre + cmds + suf
                    add(ep, range(T), lambda t: marked_sentence(seq, int(count[t])), done)
            continue
        whole = ", then ".join(dict.fromkeys(prompts))         # the episode's commands in order
        add(ep, range(T), lambda t: whole)
        if whole != prompts[0]:                                 # and each command on its own
            cuts = [0] + [t for t in range(1, T) if prompts[t] != prompts[t - 1]] + [T]
            for a, b in zip(cuts[:-1], cuts[1:]):
                if b - a > 2:
                    add(ep, range(a, b), lambda t: prompts[t])
    ds.finalize()
    print(f"wrote {len(files)} episodes, {ds.num_frames} frames to {args.root}")


if __name__ == "__main__":
    main()
