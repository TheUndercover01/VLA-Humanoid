<p align="center">
  <img src="docs/logo.svg" width="64" height="64" alt="VLA-Humanoid logo">
</p>

<h1 align="center">I Never Showed It the Task</h1>

<p align="center">
  <b>Phone Clips of Single Skills&nbsp;&nbsp;|&nbsp;&nbsp;Object-Only Reward&nbsp;&nbsp;|&nbsp;&nbsp;One Sentence, Up to Six Commands&nbsp;&nbsp;|&nbsp;&nbsp;Its Own "(done)" Marks</b>
</p>

<p align="center">
  <img src="results/site_media/restack_6of6.gif" alt="A Franka arm carries out six commands from one sentence in order: stack, unstack, stack the other way. The caption shows the command the policy believes it is on and its done flag." width="70%">
</p>

<p align="center">
  <b>A 450M vision-language-action model gets one sentence with up to six commands and carries it out from camera images alone.<br>It never saw the task: every demonstration was one skill, filmed on a phone.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/sim-Isaac%20Lab%202.3.2%20%C2%B7%20Isaac%20Sim%205.1-2ea44f?style=flat-square&labelColor=16181d" alt="Isaac Lab 2.3.2, Isaac Sim 5.1">
  <img src="https://img.shields.io/badge/policy-SmolVLA%20450M%20(LeRobot%200.6.2)-2ea44f?style=flat-square&labelColor=16181d" alt="SmolVLA 450M, LeRobot 0.6.2">
  <img src="https://img.shields.io/badge/teacher-PPO%2C%2016%20384%20envs-2ea44f?style=flat-square&labelColor=16181d" alt="PPO teacher, 16384 envs">
  <img src="https://img.shields.io/badge/robot-Franka%20Panda-2ea44f?style=flat-square&labelColor=16181d" alt="Franka Panda">
</p>

<p align="center">
  <a href="https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/"><b>Project page (all videos)</b></a> · <a href="#headline-result">Results</a> · <a href="#how-it-works">How it works</a> · <a href="#what-we-learned-by-perturbing-the-same-checkpoint">Perturbation study</a> · <a href="#what-worked-and-what-did-not">What worked</a> · <a href="#run-it">Run it</a> · <a href="PROGRESS.md">Full log</a>
</p>

The policy gets a sentence like this one, once, at the start:

```
> pick up the cube, then put the cube on the cylinder, then pick up the cube,
  then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube
```

Nobody demonstrated this task, no planner splits the sentence, and nothing from the simulator tells the policy which command it is on. The policy keeps its own place: when it thinks a command is finished, it writes **"(done)"** after that command in the sentence it reads next.

## What it had to learn from

I filmed myself with a phone doing only **single skills** on a cube and a can. No clip shows two skills in a row.

| Data | Count | Used for |
| :--- | ---: | :--- |
| Phone clips: push | 12 | Object path → reward template |
| Phone clips: pick and lift | 10 | Object path → reward template |
| Phone clips: place down | 21 | Object path → reward template |
| Teacher episodes with **two or fewer** commands | 3000 paired + 1500 push | VLA training data |
| Episodes with **3, 4 or 6** commands | 0 | Evaluation only |

<p align="center">
  <img src="results/site_media/real_clip_to_template.gif" alt="A phone clip with the tracked ArUco markers, next to the object path and the straight template the reward uses" width="100%">
</p>

<p align="center"><sub>A phone clip with the tracked markers, the object's path in the table frame, and the straight template the reward uses. The hand is never scored.</sub></p>

## Headline result

The student was trained on episodes of **at most 2 commands**. Chains of 3, 4 and 6 commands are evaluation only. 100 validation layouts per chain. **Lenient** = the sentence's end result was reached and held 0.5 s. **In order** = every command's result held, one after the other.

| Chain | Commands | Never shown? | **Whole sentence + own "(done)" marks** (ours) | RL teacher (privileged state) |
| :--- | :---: | :---: | ---: | ---: |
| push the object to the target | 1 | | 44 | 95 |
| pick up the cube / the cylinder | 1 | | 51 / 46 | 100 / 98 |
| cube to the target (c1) | 2 | | 63 / 63 | 100 |
| cube on the cylinder (c2) | 2 | | 42 / 42 | 96 |
| cylinder on the cube (swap) | 2 | | 41 / 41 | 96 |
| push the cylinder to the target, then stack the cube on it (c3) | 3 | **yes** | 40 / 3 | 76 / 18 |
| stack, then unstack to the target (unstack) | 4 | **yes** | 60 / 29 (done flag 0.9/8: **54 / 45**) | 99 / 94 |
| stack, unstack, stack the other way (restack) | 6 | **yes** | 31 / **13** | 83 / 83 |

<sub>Cells: lenient / in order, out of 100. Teacher c2 and unstack measured with the real-clip template (see the [evaluation note](#evaluation-note)).</sub>

**Reading it.** On the never-shown 4-command chain the student finishes 60% of the time and does all four steps in order 29–45% of the time. On the 6-command chain it does all six in order 13% of the time. The teacher shows the task itself is not the limit: it does random 20-command chains at 90 lenient / 83 in order.

> [!NOTE]
> **Reference, not the method.** The same data with **one command given at a time** (the policy's own done flag moves a pointer held by the harness) reaches unstack 90 / 65 and restack 53 / 38. That is the upper reference for what a perfect "which command am I on" would give this student. Giving the whole sentence costs about 20–30 points.

## How it works

```
phone clips (atomic skills) ──► ArUco markers tracked ──► object path in the table frame ──► straight start→goal template (3 cm)
                                                                                                          │
                                                       object-keypoint reward (the hand is never scored) ◄┘
                                                                          │
                              PPO teacher, privileged state, episodes of ≤ 2 commands (Isaac Lab, 16 384 envs, 400 iterations)
                                                                          │
                      paired rollouts with cameras (same scene, each object commanded) + the teacher's command switches
                                                                          │
              SmolVLA dataset: the whole sentence (2 real commands padded with 0–18 random ones), commands already done marked "(done)",
                                    a 6th action channel = "this command is done" (last 5 frames of each command)
                                                                          │
                     inference: the sentence once; the policy's own done flag (> 0.85 for 5 ticks) writes the next "(done)" mark
```

| Step | What happens | Code |
| :--- | :--- | :--- |
| **1. Clips → reward** | 46 approved clips (43 trackable). ArUco markers on the cube (ids 1–5) and the can lid (0); the target (11) and a calibration marker (13) define the table plane. Only the object's start and end are kept: a straight line from the clips' mean start offset to the goal, 3 cm tolerance (the clips' median deviation from that line is 2.6–3.2 cm). Lift height (20.4 cm), drop height, push direction and end turn come from the clips. | `handtrack/`, `motion/keypoint_ref.py`, `analysis/clip_video.py` |
| **2. Teacher** | PPO on the object-keypoint reward (8 corners of the object follow the template), a hold-still bonus once the goal is reached, and an action low-pass of 0.95 so the actions are smooth enough to imitate. | `sim/envs/keypoint_env.py`, `sim/train_rl.py`, `scripts/train_teacher.sh` |
| **3. Distillation** | 3000 paired episodes (the same scene recorded once per object, so the policy must read the object's name) + 1500 extra push episodes. SmolVLA from `smolvla_base`, early-stopped at 34K steps. | `sim/record_keypoint_rollouts.py`, `vla/build_dataset.py --prompt_mode marks --pad even --max_cmds 20`, `scripts/distill_place.sh` |
| **4. Evaluation** | The sentence is given once. The only thing that moves the policy's place is its own done flag. | `sim/eval_chains.py --learned_done --marks --whole_prompt` |

## What we learned by perturbing the same checkpoint

All on the headline checkpoint, eval only, no retraining (`sim/eval_chains.py --perturb`, `scripts/perturb_study.sh`, 50 layouts per row, videos in `media/perturb/`). "Redone" = the sentence's end result was reached again (held 0.5 s) after the disturbance.

| Test | What is done to it | Result |
| :--- | :--- | :--- |
| **No marks** (ablation) | The done flag is ignored, the sentence never changes | unstack **0 / 50** (0.72 of 4 commands) |
| **Pre-written marks** | Unstack starts with the cube on the table and the first two commands already marked "(done)" | **42 / 50** finish (3.68 of 4 in order): it skips to command 3 |
| **Oracle marks** (ablation) | The simulator's check writes the marks instead of the policy's flag | unstack in order **31 / 50** (own flag: 29 / 100); restack 10 / 50 (own: 13 / 100) |
| Knock | The finished cube is thrown to a free spot, its marks untouched | 16 of 30 redone (median 11.5 s) |
| Knock + erase 2 marks | Same, and the last two "(done)" marks are erased at that moment | 14 of 29 redone (10.6 s) |
| Knock a stack + erase 2 marks (c2) | The cube is thrown off the cylinder | 5 of 29 redone |
| Drop | The cube is taken out of the gripper after "pick up" is marked | 10 of 36 redone |
| Drop + un-mark "pick up" | Same, "pick up" un-marked | 9 of 32 redone |
| Move | The cube is moved to a new spot while the arm reaches for it | 12 / 50 (undisturbed: 63 / 100) |

**What this says:**

- **The marks are the memory, and the policy reads them.** Without them the 4-command chain never finishes (0 / 50). With two marks written in advance it starts at the right command (42 / 50).
- **On 4 commands about half of the loss is the memory; on 6 it is the skill.** Perfect marks double unstack in order (29 → 62%) but barely move restack (13 → 20%): there the commands fail even with the right mark (stacking the cylinder on the cube is 41% on its own).
- **After the work is done, recovery comes from the image, not from the text.** A thrown-away cube is fetched back about half the time whether or not its marks are erased (16 / 30 vs 14 / 29), even though the sentence says everything is done. Training never showed a knock, so this is not learned recovery: the policy reacts to "cube not on the target" in the image.
- **Weak spots:** re-targeting an object that moves during the reach (12 / 50), rebuilding a stack (5 / 29), and the done flag firing early. In the drop test some drops happened with both commands already marked done while the cube was still in the air.

## What worked and what did not

| | Finding | Numbers |
| :---: | :--- | :--- |
| ✓ | **A reward built from object paths only.** The teacher learns every skill from it and composes. The length curve is flat, so any drop of the student is the student's. | Random chains of 4–20 commands: 44–48 of 50 in order |
| ✓ | **Paired counterfactual episodes** fixed which object the policy acts on. | Earlier red/blue students: lift 45 → 80 |
| ✓ | **Imitating the filtered, executed action**, not the raw RL action. | Raw bang-bang PPO actions: fit no better than the mean |
| ✓ | **The policy's own done flag as memory.** | Flag ignored: unstack 0 / 50. Flag writes marks: 60 lenient, 29 in order |
| ✓ | **Marks in the text beat a number in the state.** Same data, same budget. | Restack in order 13 vs 0, unstack 29 vs 4, push 44 vs 34 |
| ✓ | **More push data.** | Push 23 → 44 when push frames went 12% → 27% |
| ✗ | **The students lose their place after about two commands.** The done flag fires early, and one false mark puts every later step on the wrong command. | Random chains > 4: 0 / 50 in order for every student; teacher 44–46 / 50 |
| ✗ | **Fixing the flag by thresholds.** A stricter flag helps unstack but hurts the cylinder chains. Voting three samples is no better. | Unstack in order 31 → 45; never completes 8 commands |
| ✗ | **Self-check.** The VLA finds its wrong marks, but does not redo the command. | 93–100% of erased marks really not done; rand8 in order 2.5 → 1.0 |
| ✗ | **Padding the sentence to 20 commands** for the same training budget. | c2 75 → 42, lift 77 → 51 |

<details>
<summary>Earlier attempts that did not work</summary>

Kept in [PROGRESS.md](PROGRESS.md):

- A motion-vocabulary action space (skill + VAE latent).
- Fine-tuning SmolVLA directly on replayed clips (B1, on scripted stand-in clips): push 16, lift 0, combined tasks 0.
- A hand-shaped reward that lifted by pushing.
- An object-only reward with no starting poses, which learned nothing in 504 iterations.

</details>

## Run it

Environments: Isaac Lab 2.3.2 / Isaac Sim 5.1 (`isaaclab`), LeRobot 0.6.2 (`lerobot`), OpenCV with ArUco (tracking).

```bash
# clips -> template
python -m handtrack.cut_session ...; python -m handtrack.track_objects <clip>
python -m handtrack.make_clips <root> --calib ... --out data/processed/clips_real
python -m motion.keypoint_ref --clips data/processed/clips_real --straight --out data/processed/keypoint_ref_real_straight.npz

# teacher, rollouts, student
scripts/train_teacher.sh
scripts/distill_cyl_fixed.sh   # records the 3000 paired episodes (and the one-command reference student)
scripts/record_push.sh         # 1500 push episodes
NAME=ours_cyl_marks MODE=marks GPU=0 MAXC=20 KN=20 TOK=192 DUP=0 \
  ROLL="data/processed/rollouts/ours_cyl_fixed_done data/processed/rollouts/push_extra" scripts/distill_place.sh

# studies
scripts/length_curve.sh; scripts/done_study.sh; scripts/eval_edits.sh; scripts/perturb_study.sh; scripts/headline_videos.sh
```

Results: [results/comparative/](results/comparative/) (tables, per-episode CSVs, loss and teacher curves). The full log with the cause of every number: [PROGRESS.md](PROGRESS.md).

## Evaluation note

Some teacher rows in the comparison tables were evaluated with the stand-in template loaded in the observation (an evaluation default, fixed 8 Oct). With the real-clip template the teacher scores c2 96 (was 84) and unstack 99 / 94 (was 93 / 82). Student numbers do not depend on it.

## Prior work

Learning from Play (Lynch et al. 2019); SPiRL and residual skill policies; LAPA (2410.11758); expert-to-VLA distillation (Exp2VLA 2607.03146, VLA-OPD 2603.26666); RL for VLAs (SimpleVLA-RL 2509.09674); SmolVLA (LeRobot).
