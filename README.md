# I Never Showed It the Task

I filmed myself with a phone doing only **single skills** on a cube and a can: push, pick up, put down. No clip shows two skills in a row.
A Franka Panda in simulation (Isaac Lab) then gets **one sentence with up to six commands**, for example

> *"pick up the cube, then put the cube on the cylinder, then pick up the cube, then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube"*

and a vision-language-action model (SmolVLA, 450M) has to carry it out from camera images alone. Nobody demonstrated this task, no
planner splits the sentence, and nothing from the simulator tells the policy which command it is on. The policy keeps its own place:
when it thinks a command is finished it writes **"(done)"** after it in the sentence it reads next.

<!-- media/headline/ours_cyl_marks_restack_env0.mp4: 6 of 6 commands in order -->

## Headline result

The student was trained on episodes of **at most 2 commands**. Chains of 3, 4 and 6 commands are evaluation only.
100 validation layouts per chain; **lenient** = the sentence's end result was reached and held 0.5 s, **in order** = every command's result held, one after the other.

| chain | commands | never shown? | **whole sentence + own "(done)" marks** (ours) | RL teacher (privileged state) |
|---|---|---|---|---|
| push the object to the target | 1 | | 44 | 95 |
| pick up the cube / the cylinder | 1 | | 51 / 46 | 100 / 98 |
| cube to the target (c1) | 2 | | 63 / 63 | 100 |
| cube on the cylinder (c2) | 2 | | 42 / 42 | 96 |
| cylinder on the cube (swap) | 2 | | 41 / 41 | 96 |
| push the cylinder to the target, then stack the cube on it (c3) | 3 | **yes** | 40 / 3 | 76 / 18 |
| stack, then unstack to the target (unstack) | 4 | **yes** | 60 / 29 (done flag 0.9/8: **54 / 45**) | 99 / 94 |
| stack, unstack, stack the other way (restack) | 6 | **yes** | 31 / **13** | 83 / 83 |

Cells: lenient / in order. Teacher c2 and unstack measured with the real-clip template (see "evaluation note" below).

**Reading it.** On the never-shown 4-command chain the student finishes 60% of the time and does all four steps in order 29–45% of the time. On the 6-command chain it does all six in order 13% of the time. The teacher shows the task itself is not the limit: it does random 20-command chains at 90 lenient / 83 in order.

**Reference, not the method:** the same data with **one command given at a time** (the policy's own done flag moves a pointer held by the harness) reaches unstack 90 / 65 and restack 53 / 38. That is the upper reference for what a perfect "which command am I on" would give this student. Giving the whole sentence costs about 20–30 points.

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

1. **Clips → reward.** 46 approved clips (43 trackable): 12 push, 10 pick-lift, 21 place-down. ArUco markers on the cube (ids 1–5) and the can lid (0); the target (11) and a calibration marker (13) define the table plane. From each clip only the object's start and end are kept: the template is a straight line from the clips' mean start offset to the goal with a 3 cm tolerance (the clips' median deviation from that line is 2.6–3.2 cm). Lift height (20.4 cm), drop height, push direction and end turn come from the clips. `handtrack/`, `motion/keypoint_ref.py`, video: `media/site/real_clip_to_template.mp4`.
2. **Teacher.** PPO on the object-keypoint reward (8 corners of the object following the template), hold-still bonus once the goal is reached, action low-pass 0.95 so the actions are smooth enough to imitate. `sim/envs/keypoint_env.py`, `sim/train_rl.py`, `scripts/train_teacher.sh`.
3. **Distillation.** 3000 paired episodes (the same scene recorded once per object, so the policy must read the object's name) + 1500 extra push episodes. `sim/record_keypoint_rollouts.py`, `vla/build_dataset.py --prompt_mode marks --pad even --max_cmds 20`, `scripts/distill_place.sh`. SmolVLA from `smolvla_base`, early-stopped at 34K steps.
4. **Evaluation.** `sim/eval_chains.py --learned_done --marks --whole_prompt`: the sentence is given once; the only thing that moves the policy's place is its own done flag.

## What we learned by perturbing the same checkpoint

All on the headline checkpoint, eval only (`scripts/perturb_study.sh`, 50 layouts each). *Numbers being filled in.*

| test | what is done to it | result |
|---|---|---|
| knock | the finished object is thrown to a free spot | TBD |
| knock + erase 2 marks | same, and its last two "(done)" marks are erased (done → not done at inference) | TBD |
| drop | the object falls out of the gripper after "pick up" is marked | TBD |
| drop + erase 1 mark | same, "pick up" un-marked | TBD |
| move | the object is moved while the arm reaches for it | TBD |
| pre-marked | unstack, the first two commands already marked "(done)" at the start | TBD |
| no marks (ablation) | the done flag is ignored, the sentence never changes | TBD |
| oracle marks (ablation) | the marks are written by the simulator's check, not the policy | TBD |

## What worked and what did not

**Worked**
* **A reward built from object paths only.** The teacher learns every skill from it and composes: random chains of 4 to 20 commands at 44–48 of 50 in order (`results/comparative/eval_csvs_8oct/len_teacher_*`). The length curve is flat, so the task does not get harder with length; any drop of the student is the student's.
* **Paired counterfactual episodes** fixed which object the policy acts on (earlier red/blue students: lift 45 → 80 when every scene was recorded once per cube).
* **Imitating the filtered, executed action**, not the raw RL action. Raw bang-bang PPO actions could not be fitted at all (the fit was no better than the mean).
* **A learned done flag as memory.** Without a progress signal the whole-sentence student completed unstack in order 1 time in 100; with its own done count, 42.
* **Marks in the text beat a number in the state.** Same data, same budget: restack in order 13 vs 0, unstack 29 vs 4, push 44 vs 34. The marks student's flag is right 59–72% of the time, the number student's 28–31%.
* **More push data.** Push went 23 → 44 when push episodes went from 12% to 27% of the frames.

**Did not work**
* **The students lose their place after about two commands.** On random chains longer than 4 commands, in order is 0/50 for every student, while the teacher stays at 44–46/50. At the end of an episode the pointer is far ahead of the real progress (marks 5.6/8 vs 1.0–2.0 done): **the done flag fires early**, and one false mark puts every later step on the wrong command.
* **Fixing the flag by thresholds.** A stricter flag (0.9 / 8 ticks) raises unstack in order 31 → 45 but hurts the cylinder chains and never completes 8 commands. Voting three samples is no better.
* **Self-check.** Asking the same VLA, with its last mark removed, whether the command is really done finds wrong marks (93–100% of the erased marks were really not done), but the student does not redo the command: rand8 in order 2.5 → 1.0 commands. With a per-command success around 0.6, long chains are out of reach whatever the memory does.
* **Padding the sentence to 20 commands** (so every position occurs in training) made every short task worse (c2 75 → 42, lift 77 → 51) for the same training budget.
* **Earlier attempts** (kept in `PROGRESS.md`): a motion-vocabulary action space (skill + VAE latent) and fine-tuning SmolVLA directly on replayed clips (B1, on scripted stand-in clips: push 16, lift 0, combined tasks 0); a hand-shaped reward that lifted by pushing; an object-only reward with no starting poses, which learned nothing in 504 iterations.

## Evaluation note
Some teacher rows in the comparison tables were evaluated with the stand-in template loaded in the observation (an evaluation default, fixed 8 Oct); with the real-clip template the teacher scores c2 96 (was 84) and unstack 99 / 94 (was 93 / 82). Student numbers do not depend on it.

## Run it
Environments: Isaac Lab 2.3.2 / Isaac Sim 5.1 (`isaaclab`), LeRobot 0.6.2 (`lerobot`), OpenCV with ArUco (tracking).
```
# clips -> template
python -m handtrack.cut_session ...; python -m handtrack.track_objects <clip>; python -m handtrack.make_clips <root> --calib ... --out data/processed/clips_real
python -m motion.keypoint_ref --clips data/processed/clips_real --straight --out data/processed/keypoint_ref_real_straight.npz
# teacher, rollouts, student
scripts/train_teacher.sh
scripts/distill_cyl_fixed.sh            # records the 3000 paired episodes (and the one-command reference student)
scripts/record_push.sh                  # 1500 push episodes
NAME=ours_cyl_marks MODE=marks GPU=0 MAXC=20 KN=20 TOK=192 DUP=0 ROLL="data/processed/rollouts/ours_cyl_fixed_done data/processed/rollouts/push_extra" scripts/distill_place.sh
# studies
scripts/length_curve.sh; scripts/done_study.sh; scripts/eval_edits.sh; scripts/perturb_study.sh; scripts/headline_videos.sh
```
Results: `results/comparative/` (tables, per-episode csvs, loss and teacher curves). Full log with the cause of every number: `PROGRESS.md`.

## Prior work
Learning from Play (Lynch et al. 2019); SPiRL and residual skill policies; LAPA (2410.11758); expert-to-VLA distillation (Exp2VLA 2607.03146, VLA-OPD 2603.26666); RL for VLAs (SimpleVLA-RL 2509.09674); SmolVLA (LeRobot).
