<h1 align="center">I Never Showed It the Task</h1>

<p align="center">
  <img src="results/site_media/restack_6of6.gif" alt="A Franka arm carries out six commands from one sentence, in order" width="70%">
</p>

I filmed single skills on my phone (push, pick up, put down), turned the clips into an RL reward, trained a teacher in Isaac Lab and distilled it into SmolVLA (450M). The VLA gets one sentence with several commands and works through them from two cameras, marking each finished command with "(done)". It never trained on more than two commands in a row.

```
pick up the cube, then put the cube on the cylinder, then pick up the cube,
then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube
```

No planner splits the sentence, and the simulator doesn't tell the policy which command it's on.

[Project page with videos](https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/)

## Results

All commands done in order, out of 100 unseen layouts. Final model: `ours_cyl_marks_nolift`.

| Task | Commands | Student (cameras + sentence) | Teacher (exact positions) |
| :--- | :---: | ---: | ---: |
| pick up the cube / the cylinder | 1 | 94 / 59 | 100 |
| push the cube to the target | 1 | 59 | 100 |
| cube to the target / cylinder to the target | 2 | 97 / 76 | 100 |
| cube on the cylinder / cylinder on the cube | 2 | 96 / 81 | 100 |
| push the cylinder to the target, stack the cube on it | 3 | 62 | 94 |
| **stack, then unstack** | **4** | **81** | 100 |
| **stack, unstack, restack** | **6** | **40** | 94 |
| stack, unstack, twice | 8 | 0 | 100 |

Everything with 3+ commands was never seen in training. The teacher is told which command it's on; the student has to keep track itself, which is where the gap comes from.

<p align="center">
  <img src="results/site_media/unstack_4of4.gif" alt="Stack, then unstack: four commands in order" width="49%">
  <img src="results/site_media/restack8_marks_race.gif" alt="Eight commands: the marks run ahead after four" width="49%">
</p>
<p align="center"><sub>Left: 4 commands in order. Right: 8 commands, the marks run ahead after four and it stops.</sub></p>

## How it works

<p align="center">
  <img src="docs/pipeline.svg" alt="Film skills, infer the reward, train an RL teacher, distill into a VLA, compose tasks" width="100%">
</p>

1. **Film single skills.** 43 phone clips, one skill each.
2. **Infer the reward.** The tracked object paths become a template the simulated object follows.
3. **Train an RL teacher.** PPO, 16,384 parallel envs, episodes of one or two commands.
4. **Distill into SmolVLA.** Two cameras + the sentence, plus a "command done" output.
5. **Compose.** The whole sentence at once; the model's own "(done)" marks keep its place.

### Why this way

- **Fine-tune a VLA on the clips?** It doesn't combine skills: trained on single skills it got 0 on every combined task.
- **Copy my hand?** A hand and a two-finger gripper grasp differently. The object's motion is what transfers.
- **Record the long tasks?** Every combination would need its own demos. Filming each skill once is the point.
- **Hand-written rewards?** Every one got gamed. A reward on the object's path doesn't care how it's held.
- **RL on the VLA directly?** Too slow on a 450M image model. A small teacher with exact positions learns in hours.
- **A planner feeding one command at a time?** Works and scores higher, but then the memory lives outside the model.

## 1 · Clips

<p align="center">
  <img src="results/site_media/phone_clips.gif" alt="Phone clips: push, pick up, put down, put on the other object" width="100%">
</p>

43 usable clips: 12 pushes, 10 pick-ups, 21 put-downs, with a real cube and a crisps can. Never two skills in a row. ArUco markers on each object (cube 1–5, can 0), the target (11) and a calibration marker (13).

## 2 · Reward

<p align="center">
  <img src="results/site_media/real_clip_to_template.gif" alt="A phone clip with tracked markers next to the template the reward uses" width="100%">
</p>

Eight points on the object follow the skill's template from start to goal; the robot is paid for progress along it.

- **From the clips:** lift height (20 cm), drop height, push direction, how much the object may turn, and the 3 cm tolerance (the spread of my own clips).
- **Set by hand:** "finished" = within ~2 cm of the goal, slower than 0.1 m/s for 0.3 s; step cost, per-command bonus, small hold-still bonus.
- **Hand leg:** an object-only reward gives nothing until the gripper touches the object (nothing learned in 504 iterations). The reward is one chain, gripper → object → target, and every episode starts from a random arm pose.
- **No lifting during a push:** with the hand leg, the teacher pushed by carrying (85/100). A small penalty fixed it (0/100), hence "no-lift".

## 3 · Teacher

Franka Panda in Isaac Lab, PPO, 16,384 envs, 1000 iterations. It sees exact object positions, the next template point and the current command. One or two commands per episode is enough to learn the skills and the handoff between them; a 6-step task is five handoffs in a row.

PPO first learned bang-bang control that a VLA can't imitate (no better than predicting the mean). The arm now executes a low-pass filtered action (95% previous, 5% new) and the student learns the executed motion. Smoother teacher, better student.

## 4 · Student

SmolVLA from `smolvla_base`, trained on 3000 paired + 1500 push teacher episodes (32K steps). It sees a front camera, a wrist camera and the sentence; no positions, no counter. Every scene is recorded twice, once saying "cube" and once "cylinder", so it has to read the sentence to pick the object (earlier version: 45 → 80).

## The memory problem

The image doesn't say which step you're on: in the 6-step task the arm reaches for the cube in step 1 and again in step 3, and the scenes look almost the same.

| Memory | Result |
| :--- | :--- |
| None, just the whole sentence | 4 commands: 14, in order once |
| Past frames / past actions | no help |
| Outside pointer, one command at a time | 4 commands: 90 lenient / 65 in order (memory outside the model) |
| Counter in the input ("k done") | 4 commands: 75, 6 commands: 5 |
| **"(done)" marks in the sentence** | **6 commands: 40 in order** |

> ~~pick up the cube~~ **(done)**, then ~~put the cube on the cylinder~~ **(done)**, then **pick up the cube** ← now, then put the cube on the target

- **Label:** the done output is 1 for the last 0.5 s before the teacher switched commands.
- **Mark:** above 0.85 for 5 steps → "(done)" after the current command. The first unmarked command is the one to do.
- **Padding:** training sentences are padded to 10 commands, some pre-marked, so every position shows up.
- **It uses the marks:** marks off, 4 commands succeed 0/50; two marks pre-written, it skips to command 3 and finishes 42/50.

## Evaluation

100 validation layouts per task (seed 4321), home pose, whole sentence once, 6 s per command + 5 s.

- **In order (headline):** the simulator saw every command done, one after the other. The robot's marks don't count.
- **Lenient:** only the final scene. 6 commands: 49 lenient vs 40 in order.

| Command | Counts when (loose check, used everywhere) |
| :--- | :--- |
| pick up | raised 4 cm |
| on / pushed to the target | within 5 cm of its centre |
| on the other object | within 5 cm, height within ±3 cm of resting on top |
| every result | held 0.2 s |

The strict check (3 cm, ±1.5 cm, 0.5 s) failed runs where the cube sat on the cylinder a few mm off. Same model, 6 commands in order: strict 12, teacher's check 24, loose 40. A student that really stalls stays at 6 under every check. Only the judge is looser; the robot's done flag is unchanged.

## Failures

Where the 6-command runs stop (of 100): command 5, pick up the cylinder **25**; command 4, 15; command 2, 13; others 7. 21 runs knock something over.

<p align="center">
  <img src="results/site_media/restack_fail.gif" alt="Four commands done, then the cylinder is knocked over" width="49%">
  <img src="results/site_media/restack_fail_early.gif" alt="A mark written too early skips a command" width="49%">
</p>
<p align="center"><sub>Left: knocks the cylinder over at command 5. Right: a mark written too early skips a step.</sub></p>

On 8 commands the done flag fires whenever the cube is lifted and held still, so the marks reach 8/8 after four commands and the robot stops. The teacher does the same task 100/100, so the limit is memory, not skills.

## What didn't work

| Attempt | Result |
| :--- | :--- |
| Fine-tune the VLA on single skills | push 16, pick-up 0, combined tasks 0 |
| Clean scripted demos | 6 on 2-step tasks (never saw a mistake) |
| Reward for grasping and carrying | toggled the gripper on the release threshold, 0 |
| Reward for distance to goal | pushed instead of lifting, 19 |
| Motion-primitive vocabulary | ~75% of successes were pushes |
| Imitating raw RL actions | 0 on everything but push (14) |
| Action-change penalty / smoothing labels after | loss 0.93 → 0.77, same poor fit |
| Rest speed and hold time from the clips | student unstack 17/50, restack 7/50 |
| Stricter put-down from the clips (1.34 cm) | better teacher, student stalls at command 3–4 of 6 |
| Camera history, unfrozen vision | worse than the plain student |
| Padding to 20 commands | short tasks drop (c2 75 → 42) |
| Stricter done flag (10–15 steps) | many runs never mark |
| Voting over 3 samples, re-arm, self-check | no gain |
| "(in progress)" tag | no change |
| Random pauses + same image under two sentences | flag stops firing; 4 commands 51 → 11 |

## Run it

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
```

| Step | Code |
| :--- | :--- |
| Clip tracking | `handtrack/`, `motion/keypoint_ref.py` |
| Teacher | `sim/envs/keypoint_env.py`, `sim/train_rl.py` |
| Recording | `sim/record_keypoint_rollouts.py` |
| Student | `vla/build_dataset.py --prompt_mode marks`, `scripts/distill_place.sh` |
| Eval | `sim/eval_chains.py --learned_done --marks --whole_prompt` |

## Related work

Learning from Play (Lynch et al. 2019), SPiRL, LAPA, Exp2VLA, VLA-OPD, SimpleVLA-RL, SmolVLA (LeRobot).
