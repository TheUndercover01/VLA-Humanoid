<h1 align="center">I Never Showed It the Task</h1>

<p align="center">
  <img src="results/site_media/restack_6of6.gif" alt="A Franka arm carries out six commands from one sentence, in order" width="70%">
</p>

I filmed myself doing single skills on my phone (pushing, picking up, putting down), turned those clips into a reward, trained a robot in simulation with RL, and then taught a vision-language-action model (SmolVLA, 450M) to copy it. The final model gets one sentence with several commands and works through them using only two cameras. It never trained on more than two commands in a row.

For example, it gets this sentence once, at the start:

```
pick up the cube, then put the cube on the cylinder, then pick up the cube,
then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube
```

Nothing splits the sentence up for it, and the simulator never tells it which command it's on. When it thinks a command is finished, it writes "(done)" after that command in the sentence it reads next. That's its only memory.

[Project page with videos](https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/)

## Play with it

The robot runs live on the [play page](https://theundercover01.github.io/ayushdeshmukh/projects/play/) whenever my GPU is on.

- **Sabotage the robot.** It gets a task of 2, 4 or 6 steps, and you get one sabotage per round: knock the object away, or write a fake "(done)" into its sentence. You win if the task isn't really done when it stops. To win you need to understand how it works: its only memory is its own "(done)" marks, and it never goes back to a step it has marked.
- **Tic-tac-toe.** You're O, the robot is X. On its turn the game picks a square and moves the green target there, and the robot has to actually place a cube on it. If it misses, it loses the turn. The pieces are drawn flat on the table.
- **Build a sentence.** Put together up to ten commands and watch it work through them.

The tic-tac-toe robot is a fine-tune of an earlier student with the same recipe: first on recordings with the target on the nine squares, then on 2,000 more with random X and O pieces on the board. With pieces on the board it places its cube 97 times out of 100, up from 38 before that second step.

## Results

Each row is one sentence, tried on 100 table layouts the model never saw. A run only counts if every command was done, one after the other. One step is one command: "put the cube on the target" is two steps, because the robot has to pick it up first.

| The sentence asks the robot to... | Steps | Robot (two cameras + the sentence) | Teacher (knows exact positions and the current step) |
| :--- | :---: | ---: | ---: |
| pick up the cube | 1 | 94 | 100 |
| pick up the cylinder | 1 | 59 | 100 |
| push the cube to the target | 1 | 59 | 100 |
| put the cube on the target | 2 | 97 | 100 |
| put the cylinder on the target | 2 | 77 | 100 |
| put the cube on the cylinder | 2 | 96 | 100 |
| put the cylinder on the cube | 2 | 81 | 100 |
| push the cylinder to the target, then stack the cube on it | 3 | 62 | 94 |
| **stack the cube on the cylinder, then move the cube to the target** | **4** | **81** | 100 |
| **the same, then stack the cylinder on the cube (the example above)** | **6** | **40** | 94 |
| stack the cube on the cylinder and move it to the target, twice | 8 | 0 | 100 |

Every task with three or more steps is longer than anything in training. The teacher is told which step it's on; the robot has to keep track by itself, and that's where most of the gap comes from. Final model: `ours_cyl_marks_nolift`.

<p align="center">
  <img src="results/site_media/unstack_4of4.gif" alt="Four steps done in order" width="49%">
  <img src="results/site_media/restack8_marks_race.gif" alt="Eight steps: the marks run ahead after four" width="49%">
</p>
<p align="center"><sub>Left: the 4-step task, done in order. Right: the 8-step task. Its "(done)" marks run ahead after four steps and it stops.</sub></p>

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

**Why not just fine-tune a VLA on the clips?** Because it doesn't learn to combine skills. When I trained SmolVLA directly on single-skill demonstrations, it learned a bit of pushing (16 of 100), no picking up, and scored 0 on every task with two or more steps. Seeing each skill on its own isn't enough to know how to chain them.

**Why not copy my hand?** My hand and a two-finger gripper grab things in completely different ways, so copying the hand would teach the wrong motion. What does carry over is what happens to the object: where it goes, how high, and in which direction. So I track the object and let the robot work out how to make that happen with its own gripper.

**Why not record robot demos of the long tasks?** Even with two objects and three skills there are a lot of possible 4-, 6- and 8-step tasks, and every new combination would need its own demos. The point was to film each skill once and get the combinations for free.

**Why not hand-write a reward for each task?** I tried, and the robot found a way around every one: it sat on the gripper's release threshold to collect two rewards at once, and later pushed things instead of lifting them. A reward that only follows the object's path doesn't care how the object is held, and it works for any combination of skills.

**Why train an RL teacher first and then distill?** RL is easy in simulation when the policy can see everything: 16,384 robots run in parallel, each knows the exact object positions, and a teacher learns every skill in a couple of hours. Running RL directly on a 450M-parameter model that only sees camera images would be far slower. So the teacher does the trial and error, and distillation turns it into a data machine: I can record as many labelled episodes as I want, from any layout, and the VLA only has to copy them.

**Why give the whole sentence instead of feeding one command at a time?** Feeding one command at a time works, and it scores higher. But then the hard part, knowing when a step is really finished and what comes next, lives in a script outside the model. I wanted the model itself to read the instruction and keep its place.

## 1 · Clips

<p align="center">
  <img src="results/site_media/phone_clips.gif" alt="Phone clips: push, pick up, put down, put on the other object" width="100%">
</p>

43 usable clips: 12 pushes, 10 pick-ups, 21 put-downs, with a real cube and a crisps can. Never two skills in a row. ArUco markers on each object (cube 1–5, can 0), the target (11) and a calibration marker (13).

## 2 · Reward

<p align="center">
  <img src="results/site_media/real_clip_to_template.gif" alt="A phone clip with tracked markers next to the template the reward uses" width="100%">
</p>

The reward only looks at the object. I track eight points on it, and they have to follow the skill's path, from where the object starts to where the command wants it. The robot gets paid for progress along that path, and the command counts as finished once the object reaches the goal and stays there.

The path itself comes from my clips: how high a pick-up goes (about 20 cm), where a put-down ends, which way a push goes and how much the object turns on the way. The 3 cm tolerance around the path is how much my own clips wander. A few things a clip can't tell you I set by hand: "finished" means within about 2 cm of the goal and almost still (under 0.1 m/s) for 0.3 s, plus a small cost per step and a bonus for each finished command.

Two fixes were needed on top. First, a reward on the object alone gives the arm nothing to go on until it happens to touch the object, and one run learned nothing in 504 iterations. So the reward also includes a straight path from the gripper to the object, and every episode starts from a random arm pose. Second, with that path in place the teacher learned to "push" by picking the cube up and carrying it (85 of 100 pushes). A small penalty for lifting during a push stopped it (0 of 100), which is where the final model's name, "no-lift", comes from.

## 3 · Teacher

Franka Panda in Isaac Lab, PPO, 16,384 envs, 1000 iterations. It sees exact object positions, the next template point and the current command. One or two commands per episode is enough to learn the skills and the handoff between them; a 6-step task is five handoffs in a row.

One problem showed up at distillation time. PPO first learned "bang-bang" control: 30 to 55% of its actions sat right at the limits and flipped direction from one step to the next, which works for the teacher but looks like noise to a model trying to copy it. The VLA fit those actions no better than just predicting the average action. The fix was to make the arm move smoothly: it executes a running average of the teacher's commands (95% the previous one, 5% the new one), and the student learns the motion the arm actually made. Each time I made the teacher smoother, the student got better.

## 4 · Student

SmolVLA, starting from the public `smolvla_base` weights, trained for 32K steps on 4,500 recorded teacher episodes. It sees a front camera, a wrist camera and the sentence, and nothing else: no positions and no step counter.

The 4,500 episodes are split 3,000 + 1,500 on purpose. The 3,000 are *paired*: every scene is recorded twice, once with "the cube" in the sentence and once with "the cylinder". The images are the same, so the only way to know which object to move is to read the sentence (in an earlier version this took picking the right object from 45 to 80 of 100). The extra 1,500 are pushes, because pushing was the weakest skill; in an earlier version, more push data took push from 23 to 44 of 100.

## The memory problem

This is the core of the project. Doing a long task takes three things: the skills, the handoffs between them, and knowing where you are in the list. The teacher learns the first two, and the student copies them. The third is the hard part, and the teacher never has to solve it, because the simulator tells it which command is current. The student has only two cameras and the sentence.

And the camera doesn't tell you where you are. In the 6-step example, the arm reaches for the cube in step 1 and again in step 3, and the two scenes look almost the same. Later the cube sits on the target after step 4, which looks exactly like a finished 4-step task. A policy that only looks at the current picture can't know whether it has done a step yet, so it repeats steps or skips them. It needs memory, and the question is where to put it.

| Where the memory lives | Result |
| :--- | :--- |
| Nowhere: just the whole sentence | 4 steps: 14 of 100, in order once |
| Older camera frames, past actions | no help: an old frame doesn't say which step that was either |
| A script that feeds one command at a time | 4 steps: 65 in order, but the memory is outside the model |
| A number in the input ("k steps done") | 4 steps: 75 lenient, 6 steps: 5. It had never seen position 6 |
| **"(done)" written into the sentence** | **6 steps: 40 in order** |

The answer that worked is to let the model write its memory in the one place it already reads: the sentence. It has one extra output, "is this step done?". When that stays high for half a second, "(done)" is added after the current command, and the next time it reads:

> ~~pick up the cube~~ **(done)**, then ~~put the cube on the cylinder~~ **(done)**, then **pick up the cube** ← now, then put the cube on the target

The first command without a mark is the one to do. No extra network and no planner: a language model turns out to be much better at noticing which words are marked than at using a number. In training, that output is 1 for the half-second before the teacher moved on to its next command, and the sentences are padded with up to 10 commands, some already marked, so every position shows up even though episodes have only one or two real commands.

It really does use the marks. With marking switched off, the 4-step task never finishes (0 of 50). With the first two marks written in advance, it skips straight to step 3 and finishes 42 of 50.

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

## Where it still fails

| Weakness | Now |
| :--- | :--- |
| **8 commands** | 0 of 100 in order. The done flag fires whenever an object is lifted and held still, whatever the command, so the marks run ahead and it stops after 4. The teacher does it 100/100. |
| **6 commands** | 40 in order, 49 lenient. 25 runs fail at "pick up the cylinder"; 21 of 100 knock something over. |
| **The cylinder** | pick up 59, to the target 77, onto the cube 81 (cube: 94–97). It tips easily. Sentences that start with the cylinder mostly fail. |
| **Push** | 59 (teacher 100). |
| **No undo** | A wrong "(done)" is never taken back; the skipped step is never redone. |
| **Strict judge** | With a 1.5 cm / 0.5 s check the 6-command score is 12, not 40: put-downs are a little loose. |
| **One scene** | One cube, one cylinder, one target, two fixed cameras, in simulation. Anything new on the table hurts until it's in the data: flat tic-tac-toe pieces dropped placement to 38/100, back to 97 after a fine-tune on recordings that show them. |
| **Two models** | The games' tic-tac-toe robot is a separate fine-tune, not the final model. |

## What I'd do next

1. **A done judge.** A small separate model, image + sentence → "is the current command finished?", trained with frames of the *other* commands as negatives. The action model's flag reads the image, not the text.
2. **RL on the student itself**, starting from the distilled model, on the long tasks. The teacher shows the skills are learnable; the student is 3–54 points behind it per task.
3. **Let it undo a mark.** A self-check already works one way: 93–100% of the marks it erases really weren't done. It needs to also redo the step.
4. **More cylinder data**, and a teacher that grasps it lower so it doesn't tip.
5. **Varied scenes in the recordings** (distractors, textures, lighting), since one fine-tune on new pieces was enough to fix them.
6. **One model for all**: fold the tic-tac-toe data into the final model.
7. **A real robot.** The clips are real; the robot isn't yet.

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
