<p align="center">
  <img src="docs/logo.svg" width="64" height="64" alt="VLA-Humanoid logo">
</p>

<h1 align="center">I Never Showed It the Task</h1>

<p align="center">
  <b>Phone clips of single skills&nbsp;&nbsp;|&nbsp;&nbsp;RL teacher&nbsp;&nbsp;|&nbsp;&nbsp;One sentence, six commands in a row&nbsp;&nbsp;|&nbsp;&nbsp;Its own "(done)" marks</b>
</p>

<p align="center">
  <img src="results/site_media/restack_6of6.gif" alt="A Franka arm carries out six commands from one sentence in order: stack, unstack to the target, stack the cylinder on the cube. The caption shows the command the policy believes it is on and its done flag." width="70%">
</p>

<p align="center">
  <b>A 450M vision-language-action model gets one sentence with six commands and does them in order, using only two cameras.<br>It never saw a task longer than two commands. The only demonstrations were phone clips of me doing single skills.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/sim-Isaac%20Lab%202.3.2%20%C2%B7%20Isaac%20Sim%205.1-d9480f?style=flat-square&labelColor=211d19" alt="Isaac Lab 2.3.2, Isaac Sim 5.1">
  <img src="https://img.shields.io/badge/policy-SmolVLA%20450M%20(LeRobot%200.6.2)-d9480f?style=flat-square&labelColor=211d19" alt="SmolVLA 450M, LeRobot 0.6.2">
  <img src="https://img.shields.io/badge/teacher-PPO%2C%2016%20384%20envs-e8bd2a?style=flat-square&labelColor=211d19" alt="PPO teacher, 16384 envs">
  <img src="https://img.shields.io/badge/robot-Franka%20Panda-1e4fa3?style=flat-square&labelColor=211d19" alt="Franka Panda">
</p>

<p align="center">
  <a href="https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/"><b>Project page (all videos)</b></a> · <a href="docs/HOW_IT_WORKS.md">How it works</a> · <a href="docs/EVALUATION.md">Evaluation</a> · <a href="docs/RESULTS.md">All results</a> · <a href="docs/FAILURES.md">Failures</a> · <a href="docs/WHAT_DIDNT_WORK.md">What didn't work</a> · <a href="docs/REPRODUCE.md">Run it</a>
</p>

The robot gets this sentence once, at the start:

```
pick up the cube, then put the cube on the cylinder, then pick up the cube,
then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube
```

Nobody demonstrated this task, there's no planner splitting it up, and the simulator doesn't tell the policy which command it's on. When the policy thinks a command is done, it writes **"(done)"** after it in the sentence it reads next. That's its only memory.

## Results

| | | |
| :---: | :---: | :---: |
| **40 / 100** | **81 / 100** | **0 / 100** |
| **6 commands in a row**, all in order (49 end in the right final scene) | **4 commands in a row**, all in order | **8 commands** in order: the next bar |

Every task is longer than anything in training (at most 2 commands per episode). 100 table layouts the model never trained on; a run counts **in order** only if the simulator saw every command done, one after the other. [How the scoring works →](docs/EVALUATION.md)

| Task | Commands | Student (cameras + sentence) | RL teacher (exact positions, told the command) |
| :--- | :---: | ---: | ---: |
| pick up the cube | 1 | 94 | 100 |
| push the cube to the target | 1 | 59 | 100 |
| cube to the target · cube on the cylinder · cylinder on the cube | 2 | 97 · 96 · 81 | 100 · 100 · 100 |
| stack, then unstack to the target | **4** | **81** | 100 |
| stack, unstack, stack the cylinder on the cube | **6** | **40** | 94 |
| stack, unstack, stack, unstack | 8 | 0 | 100 |

<sub>In order, out of 100 validation layouts, loose check (5 cm, ±3 cm, held 0.2 s). Final model `ours_cyl_marks_nolift`. Lenient and strict numbers: [All results](docs/RESULTS.md).</sub>

<p align="center">
  <img src="results/site_media/unstack_4of4.gif" alt="Stack, then unstack to the target: four commands done in order" width="49%">
  <img src="results/site_media/restack_fail.gif" alt="A failed six-command run: four commands done, then the cylinder is knocked over" width="49%">
</p>
<p align="center"><sub>Left: <b>4 commands</b>, all in order (2×). Right: <b>a 6-command failure</b> (3×): four done, then it knocks the cylinder over. More in <a href="docs/FAILURES.md">Failures</a>.</sub></p>

## How it works

<p align="center">
  <img src="docs/pipeline.svg" alt="Five steps: film atomic skills, infer the reward, train an RL teacher, distill into a VLA, compose new tasks" width="100%">
</p>

1. **Film single skills** on a phone: push, pick up, put down. 43 clips, never two skills in a row.
2. **Turn the clips into a reward**: the tracked object paths become a template the simulated object has to follow.
3. **Train an RL teacher** (PPO, 16,384 scenes in parallel) on episodes of one or two commands.
4. **Distill it into SmolVLA**, which sees two cameras and the sentence, and learns a "this command is done" output.
5. **Compose at inference**: the whole sentence at once; the model's own "(done)" marks keep its place.

The details, design decisions and numbers behind each step: [How it works](docs/HOW_IT_WORKS.md).

## Read more

- **[How it works](docs/HOW_IT_WORKS.md)**: the clips, the reward, the teacher, distillation, and the memory problem.
- **[Evaluation](docs/EVALUATION.md)**: what counts as done, lenient vs in order, the loose check and why.
- **[All results](docs/RESULTS.md)**: every task, student vs teacher, loose and strict, and the other students compared.
- **[Failures](docs/FAILURES.md)**: where the 6- and 8-command runs break, with videos.
- **[What didn't work](docs/WHAT_DIDNT_WORK.md)**: everything that was tried, with numbers.
- **[Reproduce it](docs/REPRODUCE.md)**: the code and commands. Full log with the cause of every number: [PROGRESS.md](PROGRESS.md).

<p align="center"><sub>Ayush Deshmukh · <a href="https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/">project page</a> · <a href="https://github.com/TheUndercover01/ayushdeshmukh">more side quests</a></sub></p>
