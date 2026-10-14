# How it works

[← back to the README](../README.md) · [Evaluation](EVALUATION.md) · [Results](RESULTS.md) · [Failures](FAILURES.md) · [What didn't work](WHAT_DIDNT_WORK.md) · [Reproduce](REPRODUCE.md)

<p align="center">
  <img src="pipeline.svg" alt="Five steps: film atomic skills, infer the reward, train an RL teacher, distill into a VLA, compose new tasks" width="100%">
</p>

1. **Film atomic skills.** Phone clips of me doing one skill at a time: push, pick up, put down.
2. **Infer the reward from the clips.** The tracked object paths become a template that the simulated object has to follow.
3. **Train an RL teacher.** PPO in Isaac Lab, 16,384 scenes in parallel, episodes of one or two commands.
4. **Distill into a VLA.** SmolVLA (450M) copies the teacher from two cameras and the sentence only.
5. **Compose at inference.** One sentence with up to six commands (or more). The model keeps its own place by writing "(done)" into the sentence.

## Why this way?

**"Why not just fine-tune a VLA on the clips?"** You can't show it every task, and a VLA doesn't learn to combine skills just because each skill is somewhere in its data. When I trained SmolVLA directly on single-skill demonstrations, it learned a bit of pushing (16 of 100), no picking up, and got **0 on every combined task**. If it has seen what a lift looks like and what a put-down looks like, it should be able to work out how to do one after the other, in whatever order the sentence asks for.

**"And why not copy my hand from the clips?"** My hand and a two-finger gripper grab things in completely different ways. The clips don't contain any gripper commands either. What they do show is what happens to the *object*: where it goes, how high, in which direction. The robot figures out how to make that happen with its own gripper, which is what RL is good at.

**"Then why not record robot demonstrations of the long tasks?"** That's the cost I wanted to avoid. Even with two objects, a target and three skills, there are a lot of possible 4-, 6- and 8-step tasks, and each would need its own teleoperated demos. I wanted to film each skill once and get the combinations for free.

**"Why not hand-write a reward for each task?"** I tried that first, and the policy found a way around every one: it sat on the gripper's release threshold, and later pushed things instead of lifting them. A reward built from the object's path doesn't care how the robot holds the object, and the same reward works for any combination of skills.

**"Why an RL teacher first, instead of RL on the VLA directly?"** RL directly on a 450M-parameter model that looks at images would be very slow. A small teacher that gets the exact object positions learns every skill in a couple of hours, running 16,384 simulated scenes in parallel. Then I distill it into a model that only uses cameras and the sentence.

**"Why give the whole sentence, instead of a planner feeding one command at a time?"** That works, and it scores higher (see [Results](RESULTS.md)). But then the hardest part, knowing when a step is really finished and what comes next, lives in code outside the model. I wanted the model itself to read the whole instruction and keep track of where it is.

## 1 · Film atomic skills

The only demonstrations are short phone clips of me doing one skill at a time with a real cube and a crisps can. **43 clips** were usable: 12 pushes, 10 pick-ups and 21 put-downs (on the target, on the table, or on the other object). I never filmed two skills in a row, and none of the clips shows any of the tasks in the results.

<p align="center">
  <img src="../results/site_media/phone_clips.gif" alt="Four phone clips side by side: push, pick up, put down, put on the other object" width="100%">
</p>

| Data | Count | Used for |
| :--- | ---: | :--- |
| Phone clips: push / pick up / put down | 12 / 10 / 21 | Object path → reward |
| Teacher episodes with **two or fewer** commands | 3000 paired + 1500 push | VLA training data |
| Episodes with **3 or more** commands | **0** | Evaluation only |

**Why track the object and not my hand.** ArUco markers on each object and on the target (cube ids 1–5, can lid 0, target 11) give each object's position and orientation on the table in every frame. One more marker (13) is only there for calibration (table plane and scale).

## 2 · Infer the reward from the clips

<p align="center">
  <img src="../results/site_media/real_clip_to_template.gif" alt="A phone clip with the tracked ArUco markers, next to the object path and the straight template the reward uses" width="100%">
</p>

Eight points on the object have to follow the skill's template from where the object starts to where the command wants it. The robot is rewarded for progress along the template, and a command counts as finished once the object reaches its goal and stays there. From the clips:

- **Where the object goes:** lift height (20 cm), drop height and push direction, averaged over each skill's clips.
- **How precisely:** the template's tolerance is 3 cm, which is how much my own clips wander around the average path.
- **How far a skill may turn the object:** push turns it a little, lift and put-down don't.

Set by hand, because a clip can't tell you: "finished" = within about 2 cm of the goal, slower than 0.1 m/s, for 0.3 s; a small cost per step; a bonus per finished command; a small bonus for holding still at the goal.

**The hand's leg.** An object-only reward gives a fresh arm nothing to go on until the gripper touches the object by chance (nothing learned in 504 iterations). Earlier teachers got around that by starting some episodes in the middle of a skill. The final teacher doesn't need that: the reward is one chain, **gripper → object → target**. The gripper → object leg is a straight line from where the hand is to the object (hand-chosen, not from the clips), with the same 3 cm tolerance. Every episode starts from a **random arm pose**, and none from a scripted mid-skill state.

**No lifting during a push.** With the hand leg, the teacher learned to "push" by picking the cube up and carrying it (85 of 100 push episodes lifted it). Its student then couldn't tell push from pick up. A small penalty for raising the object while a push command runs fixed it (0 of 100). That's why the final model is called **no-lift**.

> [!WARNING]
> **Every reward that involved the gripper directly got gamed.** My first rewards paid for grasping and carrying. The policy sat on the release threshold and opened and closed the fingers every other step, collecting both terms without ever finishing (0 of 100). Paying for "the object is near the goal" made it push the cube onto the target instead of lifting it. That's why the reward follows the object's path: pushing can't get the cube 20 cm up in the air.

## 3 · Train an RL teacher

A Franka Panda in Isaac Lab, PPO on **16,384 parallel scenes**, 1000 iterations (`keypoints_cyl_nolift_s1`). It gets privileged information: the exact position of both objects, the next point on the template, and which command it's on. Every training episode has **one or two commands**. That's enough to learn each skill and the **handoff** from one skill to the next: finish, notice it's finished, start the next one from exactly that state. A 6-step task is then five handoffs in a row, each of a kind it has seen.

Because it's told which command it's on, the teacher can chain as many commands as you give it: 94 of 100 six-command restacks in order with the strict check (see [Results](RESULTS.md) for the loose numbers). Length on its own isn't the problem. The hard part is doing a long task without anyone telling you where you are, and that's the student's job.

> [!TIP]
> **A low-pass filter on the teacher's arm.** PPO first learned "bang-bang" control: actions at the limits, flipping sign every step. A VLA trained on those fit them no better than predicting the average action, and scored 0. Now the arm executes a running average of the teacher's commands (95% previous, 5% new), and the student learns from the motion the arm actually made. Each time the teacher got smoother, the student got better.

## 4 · Distill into a vision-language-action model

I record the teacher with cameras on (3000 paired episodes + 1500 push episodes, all from random start poses) and train **SmolVLA** (450M parameters, from `smolvla_base`) to copy it. It sees a front camera, a wrist camera and the sentence. No object positions, no command counter. Final student: `ours_cyl_marks_nolift`, 32K steps (early-stopped), loss 0.248.

**Every scene is recorded twice:** once with "the cube" in the sentence and once with "the cylinder". The images are the same, so the model can only tell which object to move by reading the sentence.

## The memory problem

A long task needs three things: the **skills**, the **handoffs** between them, and knowing **where you are** in the list. The teacher never had to deal with the third, because the simulator tells it which command is current. The student only has two cameras and the sentence, and the images don't tell you where you are. In the 6-step task (*pick up the cube, put it on the cylinder, pick it up again, put it on the target, pick up the cylinder, put it on the cube*), the arm reaches for the cube in step 1 and again in step 3, and the two scenes look almost the same.

| # | Memory | Result | Why |
| :---: | :--- | :--- | :--- |
| 1 ✗ | **Nothing: just the whole sentence** | 4-step: 14 of 100, in order once | The picture doesn't say which step it's on. |
| 2 ✗ | **Show it the past** (an older frame, past actions) | no help | A past frame doesn't say which command that was either. |
| 3 ~ | **An outside pointer, one command at a time** | 4-step: 90 lenient / 65 in order | Works, but the memory is in a script, not the model. |
| 4 ~ | **A counter in the input** ("k commands done") | 4-step: 75 · 6-step: 5 | The 6th position never came up in training. |
| 5 ✓ | **"(done)" marks in the sentence itself** | first version 6-step: 31 (counter: 20) · final model: **40 in order** ([loose check](EVALUATION.md)) | A language model notices marked words far better than a number. |

The model writes the marks itself. Halfway through the 4-step task, the sentence it reads is:

> ~~pick up the cube~~ **(done)**, then ~~put the cube on the cylinder~~ **(done)**, then **`pick up the cube`** ← *now*, then put the cube on the target

1. **The label.** A 6th action channel, "this command is done", is 1 for the last 5 frames (half a second) before the teacher switched to its next command, and 0 otherwise.
2. **Writing a mark.** When it stays above 0.85 for five steps in a row, "(done)" goes after the current command.
3. **Using it.** The first command without a mark is the one to do.
4. **Padding.** Training episodes have one or two real commands, but the sentence is padded to up to 10 with extra commands, some already marked done, so the current command shows up at every position.

**Does the model actually use its marks?** (tested on an earlier model, same recipe)

- **Marks switched off** (the sentence never changes): the 4-step task finishes **0 times in 50**.
- **Two marks written in advance:** it skips straight to command three and finishes **42 times in 50**.

<p align="center">
  <img src="../results/site_media/nomarks_stuck.gif" alt="With the marks switched off the sentence never changes and the arm stays on command one" width="60%">
</p>

The weak spot is the done signal itself: if it fires before a step is really finished, every later step is on the wrong command. See [Failures](FAILURES.md) and [What didn't work](WHAT_DIDNT_WORK.md).
