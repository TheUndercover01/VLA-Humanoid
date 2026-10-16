<h1 align="center">I Never Showed It the Task</h1>
<p align="center"><b>Composing multi-step manipulation from single-skill videos</b></p>
<p align="center"><i>Can a robot that only knows single skills do long tasks it was never shown?</i></p>

<p align="center">
  <img src="results/site_media/restack_6of6.gif" alt="A Franka arm carries out six commands from one sentence, in order" width="70%">
</p>
<p align="center"><sub><b>Six commands from one sentence, in order (3×).</b> The example sentence: <i>"Pick up the cube, then put the cube on the cylinder, then pick up the cube, then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube."</i> Left: front camera. Right: wrist camera.</sub></p>

**Abstract.** People don't practise a sequence before doing it. Someone who knows how to pick an object up and how to put one down can already pick something up and put it somewhere else; switching from one skill to the next comes for free once both are known. Can a small vision-language-action model (VLA) learn the same way, from single skills and the switch between them, and then carry out long tasks it was never trained on?

I filmed myself on my phone doing single skills (pushing, picking up, putting down) and never recorded two consecutive skills. From the tracked object paths I built a reward, trained an RL teacher in simulation on episodes of at most two commands, and distilled it into SmolVLA (450M), which sees only two cameras and one sentence listing every command. The hard part is memory: from the camera alone the model can't tell which step it is on, so it has to track its own progress through the sentence. It completes a 4-step task in order 81 times in 100 and a 6-step task 40 times. It was never trained on either task and never saw one done: **both are composed at inference**.

For example, the robot gets this sentence once, at the start:

```
pick up the cube, then put the cube on the cylinder, then pick up the cube,
then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube
```

Nothing splits it up, and the simulator never tells the policy which command it's on.

<p align="center">
  <a href="https://theundercover01.github.io/ayushdeshmukh/projects/play/"><img src="https://img.shields.io/badge/%E2%96%B6%20%20Play%20with%20the%20robot%20live-d9480f?style=for-the-badge" alt="Play with the robot live" height="64"></a>
</p>
<p align="center">
  <a href="https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/"><img src="https://img.shields.io/badge/Project%20page%20with%20videos%20%E2%80%94%20my%20submission-1e4fa3?style=for-the-badge" alt="Project page with videos (my submission)" height="64"></a>
</p>
<p align="center"><sub>The project page with videos on my website is my submission. Its videos are also in this README.</sub></p>

## Results

Success rate (%) over 100 held-out table layouts per task. A run succeeds only if the simulator registers every command of the sentence as completed, in order (§3).

| Commands in the sentence | # | Student (2 RGB cameras + sentence) | Teacher (privileged state, told the current command) |
| :--- | :---: | ---: | ---: |
| pick up cube | 1 | 94 | 100 |
| pick up cylinder | 1 | 59 | 100 |
| push cube to target | 1 | 59 | 100 |
| pick up cube → place on target | 2 | 97 | 100 |
| pick up cylinder → place on target | 2 | 77 | 100 |
| pick up cube → place on cylinder | 2 | 96 | 100 |
| pick up cylinder → place on cube | 2 | 81 | 100 |
| push cylinder to target → pick up cube → place on cylinder | 3 | 62 | 94 |
| **pick up cube → place on cylinder → pick up cube → place on target** | **4** | **81** | 100 |
| **the 4 above → pick up cylinder → place on cube** (the example sentence) | **6** | **40** | 94 |
| the 4 above, twice | 8 | 0 | 100 |

Training episodes contain at most 2 commands; every row with 3 or more is out of distribution in length. Final model: `ours_cyl_marks_nolift`.

<p align="center">
  <img src="results/site_media/unstack_4of4.gif" alt="Four steps done in order" width="49%">
  <img src="results/site_media/restack8_marks_race.gif" alt="Eight steps: the marks run ahead after four" width="49%">
</p>
<p align="center"><sub>Left: <b>Stack, then unstack</b> (4 commands, 2×). Right: the 8-step task. Its "(done)" marks run ahead after four steps and it stops.</sub></p>

<p align="center">
  <img src="results/site_media/c1.gif" alt="Cube to the target" width="49%">
  <img src="results/site_media/c2.gif" alt="Cube on the can" width="49%">
</p>
<p align="center"><sub>Left: <b>Cube to the target</b> (2 commands, 2×). Right: <b>Cube on the can</b> (2 commands, 2×).</sub></p>

**Play with it.** The robot runs live on the [play page](https://theundercover01.github.io/ayushdeshmukh/projects/play/) when my GPU is on. *Sabotage the robot*: it gets a 2-, 4- or 6-step task and you get one sabotage per round (knock the object away or write a fake "(done)"); you win if the task isn't really done when it stops, and winning takes understanding how its memory works. *Tic-tac-toe*: you're O, the robot has to physically place each X. *Build a sentence*: up to ten commands. The tic-tac-toe robot is a fine-tune of an earlier student on the nine board squares and then on 2,000 recordings with flat X / O pieces on the board, which took placement with pieces from 38 to 97 of 100.

## 1 · Hypotheses

The question: if a robot is only ever shown single skills, can it carry out a multi-step instruction nobody demonstrated?

- **H1. Single-skill clips are enough to learn the skills.** A reward built only from how the object moves in my phone clips can train an RL policy on every skill and on the handoff from one skill to the next.
- **H2. Skills plus handoffs compose.** A VLA distilled from a teacher that only ever ran one or two commands per episode can do longer sentences at inference, because a long task is just more handoffs of kinds it has seen.
- **H3. The missing piece is memory, and it can live in the sentence.** From the camera alone a policy can't tell which step it's on. If the model writes its own progress into its input as "(done)" marks, it can track progress without a planner or an outside counter.

## 2 · Method

<p align="center">
  <img src="docs/pipeline.svg" alt="Film skills, infer the reward, train an RL teacher, distill into a VLA, compose tasks" width="100%">
</p>

### 2.1 Clips

<p align="center">
  <img src="results/site_media/phone_clips.gif" alt="Phone clips: push, pick up, put down, put on the other object" width="100%">
</p>
<p align="center"><sub>Push · Pick up · Put down · Put on the other object</sub></p>

43 usable phone clips of me with a real cube and a crisps can: 12 pushes, 10 pick-ups, 21 put-downs, never two skills in a row. ArUco markers on each object (cube 1–5, can 0), the target (11) and a calibration marker (13) give each object's position and orientation on the table in every frame. I track the object and not my hand, because a hand and a two-finger gripper grab things differently; what carries over is where the object goes.

### 2.2 Reward from the clips

<p align="center">
  <img src="results/site_media/real_clip_to_template.gif" alt="A phone clip with tracked markers next to the template the reward uses" width="100%">
</p>
<p align="center"><sub>The tracked path (orange) and the template the reward uses (green).</sub></p>

Dense, object-centric keypoint reward. The 8 corner keypoints of the commanded object are tracked against a per-skill reference trajectory (start → goal), and the reward is the keypoints' progress along it inside a 3 cm tube (3 cm ≈ the median deviation of my clips from their mean path). Taken from the clips: lift height (≈20 cm), place height, push direction, allowed yaw change. Set by hand: the completion test (keypoint error < ≈2 cm, object speed < 0.1 m/s, held 0.3 s), a per-step time penalty and a per-command completion bonus. There is no grasp term.

Two changes on top of that:

- **Exploration.** With an object-only reward the signal is zero until first contact; a run from the home pose learned nothing in 504 iterations. I prepend a gripper → object leg (straight-line reference from the TCP to the object centre, same 3 cm tube) and randomize the initial arm configuration every episode.
- **Reward hacking.** The teacher then solved *push* by grasp-and-carry (85 of 100 push episodes lifted the object > 3 cm). A lift penalty that is active only during push commands removed it (0 of 100). This is the "no-lift" in the model name.

Per-task hand-written rewards were tried first and all were exploited (§4.5). The keypoint reward is embodiment-agnostic and shared across every combination of skills.

### 2.3 RL teacher

A Franka Panda in Isaac Lab, PPO with 16,384 simulated robots in parallel, 1000 iterations (a couple of hours). The teacher sees the exact object positions, the next point on the path and the current command. Each episode has one or two commands, so it learns every skill and the handoff between two of them: finish, notice it's finished, start the next one from exactly that state.

RL is cheap here because the teacher sees everything. Running RL directly on a 450M-parameter model that only sees images would be far slower, so the teacher does the trial and error, and distillation turns it into a data source: I can record as many labelled episodes as I want, from any layout.

One problem showed up at distillation time. PPO learned "bang-bang" control: 30 to 55% of its actions sat at the limits and flipped direction from one step to the next. That works for the teacher but looks like noise to a model trying to copy it, and the VLA fit those actions no better than predicting the average. The fix was to make the arm move smoothly: it executes a running average of the teacher's commands (95% previous, 5% new), and the student learns the motion the arm actually made.

### 2.4 Distillation

SmolVLA, starting from the public `smolvla_base` weights, trained for 32K steps on 4,500 teacher episodes. It sees a front camera, a wrist camera and the sentence: no positions and no step counter.

The 4,500 are split 3,000 + 1,500 on purpose. The 3,000 are *paired*: every scene is recorded twice, once with "the cube" in the sentence and once with "the cylinder". The images are the same, so the only way to know which object to move is to read the sentence (in an earlier version this took picking the right object from 45 to 80 of 100). The extra 1,500 are pushes, the weakest skill; in an earlier version more push data took push from 23 to 44.

### 2.5 Memory: "(done)" marks in the sentence

The model has one extra output, "is the current step done?". In training it is 1 for the half-second before the teacher moved on to its next command. At run time, when it stays above 0.85 for half a second, "(done)" is written after the current command, and the next time the model reads:

> ~~pick up the cube~~ **(done)**, then ~~put the cube on the cylinder~~ **(done)**, then **pick up the cube** ← now, then put the cube on the target

The first command without a mark is the one to do. Training episodes have only one or two real commands, so sentences are padded with up to 10 commands, some already marked, and every position shows up.

<p align="center">
  <img src="results/site_media/nomarks_stuck.gif" alt="Marks off: stuck on command one" width="49%">
  <img src="results/site_media/restack_6of6.gif" alt="Its own marks: all six in order" width="49%">
</p>
<p align="center"><sub>Left: <b>Marks off</b> (3×): stuck on command one. Right: <b>Its own marks</b> (3×): all six in order.</sub></p>

## 3 · Experimental setup

100 validation layouts per task (seed 4321, never used in training), the arm starts at home, the whole sentence is given once, and each run has 6 s per command plus 5 s. Two scores: **in order** (the headline: every command done, one after the other, as counted by the simulator; the robot's own marks don't count) and **lenient** (only the final scene). For 6 steps: 40 in order, 49 lenient.

| Command | Counts as done when |
| :--- | :--- |
| pick up | raised at least 4 cm |
| on / pushed to the target | within 5 cm of its centre |
| on the other object | within 5 cm, and within ±3 cm of resting height on top |
| every result | held for 0.2 s |

This is a loose check. With my first, strict check (1.5 cm, 0.5 s) the same model scores 12 on the 6-step task, 24 with the teacher's own check, 40 with this one: its put-downs are slightly off, not failed. A model that really stalls doesn't benefit (one stays at 6 under every check). Only the judge is looser; the robot's own done signal is unchanged, and the teacher is scored the same way.

## 4 · Analysis

### 4.1 H1: the skills are learnable from clips (supported)

The teacher, trained only on the clip-derived reward, does every single skill 100 of 100 and chains them: 94 of 100 on the 6-step task and 100 of 100 on the 8-step one. Single-skill clips are enough to learn the skills and the handoffs.

### 4.2 H2: skills compose, up to a point (partly supported)

The student composes tasks it never saw: 62 on 3 steps, 81 on 4, 40 on 6. But it drops to 0 on 8, while the teacher stays at 100. The skills transfer; something else limits length.

### 4.3 H3: memory in the sentence works, and it's the bottleneck (supported, with a limit)

From the camera alone you can't tell where you are. In the 6-step task the arm reaches for the cube in step 1 and again in step 3, and the scenes look almost the same; after step 4 the board looks exactly like a finished 4-step task. Where the memory lives makes the difference (rows above the last are earlier versions of the same pipeline):

| Where the memory lives | Result |
| :--- | :--- |
| Nowhere: just the whole sentence | 4 steps: 14 of 100, in order once |
| Older camera frames, past actions | no help: an old frame doesn't say which step that was either |
| A script that feeds one command at a time | 4 steps: 65 in order, but the memory is outside the model |
| A number in the input ("k steps done") | 4 steps: 75 lenient; 6 steps: 5 (it never saw position 6) |
| **"(done)" written into the sentence** | **6 steps: 40 in order** |

The model really uses its marks (tested on an earlier model with the same recipe). With marking switched off the 4-step task never finishes (0 of 50); with the first two marks written in advance it skips straight to step 3 and finishes 42 of 50.

<p align="center">
  <img src="results/site_media/knock_refetch.gif" alt="Knocked away after it was done" width="49%">
  <img src="results/site_media/drop_repick.gif" alt="Dropped mid-carry" width="49%">
</p>
<p align="center"><sub>Left: <b>Knocked away after it was done</b> (2×). Right: <b>Dropped mid-carry</b> (2×).</sub></p>

The limit is the done signal itself. It fires when an object is lifted and held still, whichever command is current, so it reads the image more than the sentence. On 8 steps the marks reach 8 of 8 after four steps and the robot stops; in 60 of 100 runs exactly four steps were done in order.

### 4.4 Where the 6-step runs fail

<p align="center">
  <img src="results/site_media/restack_fail.gif" alt="Four steps done, then the cylinder is knocked over" width="49%">
  <img src="results/site_media/restack_fail_early.gif" alt="A mark written too early skips a step" width="49%">
</p>
<p align="center"><sub>Left: <b>Knocks the cylinder over</b> at command five (3×). Right: <b>Marked done too early</b>, so a step is skipped (3×).</sub></p>

Of 100 runs, the failures stop at step 5, picking up the cylinder (25), step 4 (15) and step 2 (13); 21 runs knock something over. Two causes: early marks, and the cylinder, the weakest object (pick up 59, to the target 77, onto the cube 81, against 94–97 for the cube).

### 4.5 What didn't work

<p align="center">
  <img src="results/site_media/fail_never_release.gif" alt="Reward for carrying: never lets go" width="49%">
  <img src="results/site_media/fail_hold_up.gif" alt="Motion primitives: holds the cube up forever" width="49%">
</p>
<p align="center"><sub>Left: <b>Reward for carrying:</b> never lets go. Right: <b>Motion primitives:</b> holds the cube up forever.</sub></p>
<p align="center">
  <img src="results/site_media/fail_top_edge.gif" alt="Loose arrived check: grabs the top edge, the cube slips" width="49%">
  <img src="results/site_media/fail_push_not_lift.gif" alt="Distance-to-goal reward: pushes instead of lifting" width="49%">
</p>
<p align="center"><sub>Left: <b>Loose "arrived" check:</b> grabs the top edge, the cube slips. Right: <b>Distance-to-goal reward:</b> pushes instead of lifting.</sub></p>

| Attempt | Result |
| :--- | :--- |
| Fine-tune the VLA directly on single skills | push 16, pick-up 0, every combined task 0 |
| Clean scripted demos | 6 on 2-step tasks: it never saw a mistake, so it couldn't recover |
| Reward for grasping and carrying | toggled the gripper on the release threshold to collect both rewards, 0 |
| Reward for distance to the goal | pushed instead of lifting, 19 |
| Motion-primitive vocabulary | ~75% of successes were pushes |
| Imitating raw RL actions | 0 on everything but push (14) |
| Action-change penalty / smoothing labels afterwards | loss 0.93 → 0.77, same poor fit |
| Rest speed and hold time from the clips | student 4 steps 17/50, 6 steps 7/50 |
| Stricter put-down from the clips (1.34 cm) | better teacher; student stalls at step 3–4 of 6 |
| Camera history, unfrozen vision | worse than the plain student |
| Padding to 20 commands | short tasks drop (cube on cylinder 75 → 42) |
| Stricter done signal (10–15 steps) | many runs never mark |
| Voting over 3 samples, re-arming, self-check | no gain |
| "(in progress)" tag on the current command | no change |
| Random pauses + the same image under two sentences | the signal stops firing; 4 steps 51 → 11 |

## 5 · Limitations

- **"Done" has no time dimension, so the robot never learns to let go.** A step is complete when the object reaches its goal pose and comes to rest. The test never looks at the gripper (on purpose: object-level, any embodiment), and it never asks the object to *stay* there once the next step starts. Releasing earns nothing. Example: *pick up the cube, put the cube on the target, pick up the cylinder, put the cylinder on the cube.* The cube reaches the target and gets marked done while still in the gripper. The robot then drags it toward the cylinder, opens whenever it happens to, grasps the cylinder and puts it on the cube wherever the cube now is. Object-centric goals need a spatio-temporal part: where the object has to be, and for how long (until the task ends or a later step moves it). The scoring has the same blind spot. A step counts once its result holds for 0.2 s, so this run can still score as a success.
- **8 steps: 0 of 100.** The done signal follows the image, not the sentence. On the 8-step task, 60 of 100 runs stop at exactly 4 steps in order while the marks have already reached 8 of 8. The scene after step 4 looks like a finished 4-step task, so the model marks the rest done (4.3).
- **No undo.** A wrong "(done)" is never taken back, so the skipped step is never redone. A self-check that erases wrong marks finds them (93–100% correct), but the robot doesn't complete the step on the second try.
- **The headline uses a loose check.** Positions within 5 cm and results held 0.2 s. With 1.5 cm / 0.5 s the 6-step score is 12 in order, not 40 (§3).
- **No separate test set.** The final model and the check were both chosen on the same 100 validation layouts the numbers are reported on.
- **Cylinder and pushing are weak** (59 each, the teacher gets 100). Grasped too high, the cylinder tips, and once it's on its side it can't be picked up (never in the data). Most sentences that start with the cylinder fail.
- **The student is 3–54 points behind its teacher** on every task. The teacher sees the true object state; the student has two cameras.
- **Not every part of the reward comes from the clips.** The completion thresholds, time penalty, completion bonus and the gripper→object leg are set by hand.
- **One scene, in simulation.** One cube, one cylinder, a fixed target, two fixed cameras. Anything new on the table hurts until it's in the data.
- **The tic-tac-toe robot** in the games is a separate fine-tune, not the final model.

## 6 · Future work

1. **A done judge.** A small separate model, image + sentence → "is the current step finished?", trained with frames from the *other* steps as negatives, so it has to read the text.
2. **RL on the student itself**, starting from the distilled model, on the long tasks; it is 3–54 points behind the teacher per task.
3. **Undo.** A self-check already erases mostly wrong marks (93–100% of them really weren't done); it needs to redo the step too.
4. **More cylinder data**, and a teacher that grasps it lower so it doesn't tip.
5. **Varied scenes** in the recordings (distractors, textures, lighting); one fine-tune on new pieces was enough to fix them.
6. **A real robot.** The clips are real; the robot isn't yet.

## Reproduce

Isaac Lab 2.3.2 / Isaac Sim 5.1, LeRobot 0.6.2, OpenCV with ArUco.

```bash
# clips -> reward template
python -m handtrack.track_objects <clip>
python -m handtrack.make_clips <root> --calib ... --out data/processed/clips_real
python -m motion.keypoint_ref --clips data/processed/clips_real --straight --out data/processed/keypoint_ref_real_straight.npz

# teacher -> recordings -> student -> eval
scripts/run_nolift_pipeline.sh

# eval with the loose check (student, teacher)
NAME=ours_cyl_marks_nolift G=0 scripts/eval_cliptol_loose.sh
G=0 scripts/eval_teacher_loose.sh

# the live games
play/run_sabotage.sh start; play/run_builder.sh start; play/run_tictactoe.sh start ours_cyl_marks_ttt_tiles
PUSH=1 play/run_tunnels.sh start; play/watchdog.sh
```

| Step | Code |
| :--- | :--- |
| Clip tracking | `handtrack/`, `motion/keypoint_ref.py` |
| Teacher | `sim/envs/keypoint_env.py`, `sim/train_rl.py` |
| Recording | `sim/record_keypoint_rollouts.py` |
| Student | `vla/build_dataset.py --prompt_mode marks`, `scripts/distill_place.sh` |
| Eval | `sim/eval_chains.py --learned_done --marks --whole_prompt` |
| Games | `play/` |

Paths in the scripts point to my machine (`/media/storage/...`); change them to yours.

## Related work

Learning from Play (Lynch et al. 2019), SPiRL, LAPA, Exp2VLA, VLA-OPD, SimpleVLA-RL, SmolVLA (LeRobot).
