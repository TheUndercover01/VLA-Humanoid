# I Never Showed It the Task

I filmed myself doing only **atomic skills** with a phone: reach, push, pick-lift and place-down. A Franka Panda in simulation (Isaac Lab) is then asked, in language, to do **combined tasks that were never demonstrated as a whole**:

| Task | Prompt | Needs |
|---|---|---|
| C1 | pick up the red cube and place it on the green target | reach → pick-lift → place-down |
| C2 | stack the red cube on the blue cube | reach → pick-lift → place-down, 2.5 cm precision |
| C3 | push the blue cube onto the green target, then stack the red cube on it | push → reach → pick-lift → place-down |

Given exactly the same phone clips, which approach composes the skills into these tasks better, and why?

- **Standard (B1).** Fine-tune SmolVLA on the clips, each labelled with its atomic instruction, then prompt it with the combined task.
- **Ours.** Turn the clips into a **motion vocabulary** (each skill plus a 3-D latent), train **RL** whose action is *skill + latent* (the combined task exists only in the reward), then **distil** the RL experts into SmolVLA.

> **Status: work in progress (draft of 3 Oct).** The infrastructure, baselines B0/B1 on stand-in data and the first RL runs exist; the phone clips are not processed yet. Numbers marked *stand-in* come from scripted stand-ins for the phone clips and will be replaced. Full log: [PROGRESS.md](PROGRESS.md).

## Pipeline

```
phone clips ─► hand tracking ─► retarget to the Panda TCP ─► clips (table frame, 10 Hz)
   (laptop: handtrack/)                                          │
                                       ┌─────────────────────────┴───────────────────────┐
                                       ▼                                                 ▼
                         B1: replay clips in sim with cameras             motion vocabulary (VAE, motion/)
                             → SmolVLA fine-tune on atomic prompts                 │
                                                                    RL: action = skill + latent,
                                                                    reward = combined task (sim/)
                                                                                   │
                                                                    distil: expert rollouts with cameras
                                                                    → SmolVLA (image + prompt → skill + latent)
```

## Models compared

| Model | Trained on | Sees at test time | Purpose |
|---|---|---|---|
| B0 | nothing (smolvla_base) | images, proprio, prompt | floor |
| B1 | phone clips replayed in sim, atomic prompts | images, proprio, prompt | does a VLA compose by itself? |
| B1+LLM | same as B1; a planner splits the prompt into atomic instructions | same | planning vs execution |
| B2 | rollouts of raw-action RL experts, combined prompts | same | isolates the vocabulary |
| **Ours** | rollouts of vocabulary RL experts, combined prompts | same | main method |
| RL experts | — (privileged cube/target positions) | 21-D sim state | upper reference |
| Oracle | scripted full-task demos, combined prompts | same as B0 | ceiling with full demos (context only) |

Every SmolVLA model sees only the front and wrist cameras (256×256), 6-D proprioception (TCP position, yaw, finger gap) and the prompt. Only the RL experts and the scripted expert use privileged state; they are teachers.

## Results

*To be filled in from `analysis/results_table.py`.* Protocol: 100 fixed start states per task (`sim/eval_states/`, seed 1234), success with 95% Wilson intervals, failure stage per episode (reach, grasp, lift, transport, place, knocked), episode time, end-effector jerk, peak contact force.

First numbers on *stand-in* clips (success out of 100):

| | push | lift | C1 | C2 | C3 |
|---|---|---|---|---|---|
| Scripted expert (privileged) | 100 | 100 | 100 | 100 | 90 |
| B0 smolvla_base | 0 | 0 | 0 | 0 | 0 |
| B1 (*stand-in*) | 16 | 0 | 0 | 0 | 0 |
| B1+LLM (*stand-in*) | 3 | 29 | 0 | 0 | 0 |

Even lift needs two clips chained (reach, then pick up). B1 alone reaches the cube in 24% of episodes and never grasps; given the plan it lifts in 29%. Nothing composes into C1–C3, and the failures sit at the hand-offs (grasp, transport).

## Design choices

- **One environment, five tasks.** `sim/envs/panda_env.py` is a single Isaac Lab DirectRLEnv; tasks differ only in start distribution, reward and success, all in `sim/envs/tasks.py`. Bookkeeping (success hold, stages, metrics, reward) runs on a 10 Hz control tick, so a raw action (one tick) and a vocabulary primitive (~20 ticks) are scored identically.
- **Action.** Raw: (dx, dy, dz, dyaw, grip) at 10 Hz, an integrated TCP command tracked by damped-least-squares IK at 100 Hz. Vocabulary: argmax over 4 skill scores plus a 3-D latent, decoded into 10 waypoints relative to the gripper and tracked by the same controller. rsl_rl only has Gaussian actions, so the skill is an argmax over Gaussian scores rather than a categorical.
- **Workspace from measurement, not guesses.** `sim/probe_reach.py` showed a vertical gripper reaches about 0.78 m from the base and no closer than about 0.3 m. Objects spawn inside that, and the TCP command is clamped to it (beyond it, the IK settled in a retracted, tilted pose).
- **Reward.** Staged and dense: reach, grasp by contact geometry, the cube's 3-D distance to its resting pose at the goal, an at-goal term that grows as the gripper opens, and a bonus for every step once success has held for 0.5 s. It took three iterations; each version is checked by profiling it along the scripted expert's trajectory, where it must rise at every phase (`tests/test_tasks.py` pins the failure cases).
- **Clips → sim.** `sim/clips.py` validates clips against the interface contract and infers the cube when it was not tracked (error on stand-ins: 0.2–2.4 cm). `sim/replay_clips.py` sets up each clip's starting situation with a scripted pre-roll, then commands the clip's poses with the raw action, which is what B1 learns.
- **Fairness.** B2, Ours and Oracle get the same number of distillation episodes (500 successful episodes per task) and the same fine-tuning budget. Raw and vocabulary RL use the identical reward, and learning curves are compared on simulated time, not iterations.

## What didn't work

- **C3 as first specified.** "The cube starts beyond grasp reach; push it closer, then pick it up" is impossible with a vertical gripper: to push a cube towards you, you must reach behind it, which is further away still. Tilting the hand gains only about 5 cm. C3 became "push blue onto the target, then stack red on it".
- **Three reward bugs, each found by watching the policy rather than the return curve.**
  1. "Held" only counted above 2 cm, so lowering the cube lost reward: PPO hovered over the goal forever.
  2. Grasp reward (gap < 65 mm) and success bonus (gap > 65 mm) paid side by side: the policy held the gripper exactly at 65 mm and flickered between the two. Success was true 55% of the time but never for 5 consecutive steps.
  3. Carry credit only while grasped: the vocabulary agent learned reach → pick-lift and then held the cube in the air, because any imprecise place-down lost reward. Meanwhile, a stay-near-the-cube term made the raw C2 agent keep jostling the stacked cube, so it never came to rest.
- **The stock high-PD Franka config** (damping 80 with the 12 Nm wrist torque limit) caps the wrist at 0.15 rad/s, so every Cartesian move crawled at about 6 cm/s. Damping 30 halved episode times.
- **Pushing with closed fingers** spun or tipped the cube (4% success). A two-fingertip contact, slow closed-loop pushing and re-approaching when the cube drifts off line fixed it (100% on push, 90% on C3 for the scripted expert).
- **Stock lift PPO is seed-sensitive.** Seed 42 never learned to lift in 1500 iterations; seed 1 matched NVIDIA's published checkpoint.

## Running it

Setup (two conda envs, because Isaac Sim 5.1 needs Python 3.11 and LeRobot 0.6.2 needs Python 3.12): [GPU_START.md](GPU_START.md) §5. Isaac scripts run as `PYTHONPATH=. <IsaacLab>/isaaclab.sh -p <script> --headless`, or detached with `scripts/run_isaac.sh <log> <script> [args]`.

```bash
python -m pytest tests -q                                             # task logic and clip handling
# scripted expert on the 100 eval states, with a video
sim/eval.py --task c1 --policy scripted --enable_cameras --video media/c1.mp4
# clips -> vocabulary -> RL
python -m sim.clips data/processed/clips                              # validate clips
python -m motion.vae --clips data/processed/clips --out data/processed/vocab.pt
sim/train_rl.py --task c1 --action vocab --vocab data/processed/vocab.pt --max_iterations 250
sim/train_rl.py --task c1 --action raw --max_iterations 2000
python -m analysis.learning_curves runs/rl/* --out results/learning_curves.png
# B1: replay clips with cameras -> LeRobot dataset -> SmolVLA fine-tune -> eval through the server
sim/replay_clips.py --enable_cameras --clips data/processed/clips --out data/processed/replays/b1
python -m vla.build_dataset --src data/processed/replays/b1 --repo_id local/b1 --root <dataset_root>   # lerobot env
scripts/finetune_smolvla.sh <dataset_root> <out_dir> 6000
python -m vla.server --policy <out_dir>/checkpoints/last/pretrained_model --port 6012               # lerobot env
scripts/eval_vla_suite.sh b1 6012
python -m analysis.results_table runs/eval/*.csv --out results/table.md
```

## Repo layout

| Folder | Contents |
|---|---|
| `handtrack/` | ChArUco board, camera calibration, hand tracking, retargeting (laptop side) |
| `motion/` | Motion vocabulary VAE (`vae.py`) |
| `sim/` | Isaac Lab env and tasks, scripted expert, eval harness, RL training, clip replay, rollout recording |
| `vla/` | LeRobot datasets, SmolVLA server, planner |
| `analysis/` | Results table with Wilson intervals, learning curves |
| `tests/` | Task logic and clip handling |
| `media/` | Progress videos (index in `media/README.md`; the mp4s are not committed) |

## Prior work

Learning from Play (Lynch et al. 2019); SPiRL / skill priors; Residual Skill Policies (arXiv 2211.02231); LAPA (2410.11758); CLAP (2601.04061); Motion-Focused Latent Action (2606.18955); VQ-VLA (ICCV 2025); Exp2VLA (2607.03146); VLA-OPD (2603.26666); SimpleVLA-RL (2509.09674); SmolVLA-RL (github.com/Mervin-James/SmolVLA-RL). The novelty here is not a new method but the question and the controlled comparison.
