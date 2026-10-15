<h1 align="center">I Never Showed It the Task</h1>

<p align="center">
  <img src="results/site_media/restack_6of6.gif" alt="A Franka arm carries out six commands from one sentence, in order" width="70%">
</p>

I filmed single skills on my phone (push, pick up, put down), turned the clips into an RL reward, trained a teacher in Isaac Lab and distilled it into SmolVLA. The VLA gets one sentence with several commands and works through them from two cameras, marking each finished command with "(done)". It never trained on more than two commands.

```
pick up the cube, then put the cube on the cylinder, then pick up the cube,
then put the cube on the target, then pick up the cylinder, then put the cylinder on the cube
```

[Project page with videos](https://theundercover01.github.io/ayushdeshmukh/projects/never-shown/) · [Details](docs/DETAILS.md) · [Run it](docs/REPRODUCE.md)

## Results

All commands done in order, out of 100 unseen layouts.

| Task | Commands | Student (VLA) | Teacher (RL) |
| :--- | :---: | ---: | ---: |
| pick up the cube | 1 | 94 | 100 |
| cube to the target / on the cylinder | 2 | 97 / 96 | 100 |
| stack, then unstack | 4 | **81** | 100 |
| stack, unstack, restack | 6 | **40** | 94 |
| stack, unstack, twice | 8 | 0 | 100 |

## Method

<p align="center">
  <img src="docs/pipeline.svg" alt="Film skills, infer the reward, train an RL teacher, distill into a VLA, compose tasks" width="100%">
</p>

1. Film single skills: 43 phone clips, ArUco markers on the objects.
2. The tracked object paths become the reward.
3. PPO teacher with exact object positions, episodes of one or two commands.
4. Distill into SmolVLA (450M): two cameras and the sentence.
5. At test time the model writes its own "(done)" marks to keep its place.
