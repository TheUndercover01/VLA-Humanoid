# Failures

[← back to the README](../README.md) · [How it works](HOW_IT_WORKS.md) · [Evaluation](EVALUATION.md) · [Results](RESULTS.md) · [What didn't work](WHAT_DIDNT_WORK.md) · [Reproduce](REPRODUCE.md)

Final model `ours_cyl_marks_nolift`, 100 validation layouts per task, [loose check](EVALUATION.md). The videos are from a separate run of 16 layouts (restack: 5 of 16 in order, unstack: 14 of 16, 8 commands: 0 of 16). The caption on top of each video is what the model itself believes: its pointer in the sentence, its done flag, and, for comparison, the commands the simulator counted as done in order.

## Where the 6-command task stops

*pick up the cube → put the cube on the cylinder → pick up the cube → put the cube on the target → pick up the cylinder → put the cylinder on the cube*

| Stuck at command | | Runs (of 100) |
| :---: | :--- | ---: |
| 1 | pick up the cube | 1 |
| 2 | put the cube on the cylinder | 13 |
| 3 | pick up the cube | 1 |
| 4 | put the cube on the target | 15 |
| **5** | **pick up the cylinder** | **25** |
| 6 | put the cylinder on the cube | 5 |
| — | all six in order | **40** |

21 of the 100 runs knock an object over, and 9 reach the right final scene without doing the commands in order.

**The commonest failure is picking up the cylinder (25)**, the only time the sentence asks for the cylinder after four commands about the cube. In the video below the arm knocks the cylinder over going for it.

<p align="center">
  <img src="../results/site_media/restack_fail.gif" alt="A failed six-command run: four commands done in order, then the cylinder is knocked over" width="70%">
</p>
<p align="center"><sub><b>Stuck at "pick up the cylinder" (3×).</b> The cube is held in the air for a long time before it's put on the target. Then the arm goes for the cylinder and knocks it over. Four of six in order.</sub></p>

**The second kind is a mark written too early.** The done flag fires before a put-down is really finished, the pointer moves on, and that command is never redone. The run can still end in the right final scene, but not in order.

<p align="center">
  <img src="../results/site_media/restack_fail_early.gif" alt="The policy marks 'put the cube on the cylinder' done before the cube is on it, skips the command and still reaches the final scene" width="70%">
</p>
<p align="center"><sub><b>Marked done too early (3×).</b> After about 6 s the pointer is already on command 3 while the cube never sat on the cylinder. The rest goes fine and the final scene is right, so it counts as lenient but not in order.</sub></p>

## The 4-command task

81 of 100 in order. The 19 failures are spread out: 3 at each of the first three commands and 10 at the last put-down on the target.

## 8 commands: the next bar

*stack, unstack to the target, stack again, unstack to the target again*: **0 of 100 in order**. 82 end in the right final scene, but that scene (the cube on the target) is also what the board looks like after command 4, so lenient means little here. In 60 runs the simulator counts exactly 4 commands done in order. After command 4 the cube is on the target, which looks exactly like the finished 4-command task. Here the marks are the only memory, and they have raced ahead: the pointer reaches 8 of 8 while the robot has done the first four, so it stops.

<p align="center">
  <img src="../results/site_media/restack8_marks_race.gif" alt="Eight commands: the marks race to the end of the sentence while only the first four commands are done" width="70%">
</p>
<p align="center"><sub><b>8 commands (3×).</b> By about 15 s the pointer reads 8 of 8 while only the first four commands are done. The robot then holds still: as far as its sentence says, it's finished.</sub></p>

The teacher, which is told which command it's on, does the same 8-command task 100 of 100 in order, so the skills aren't the limit. The limit is the memory. In the videos the done flag fires when the cube is lifted and held still, whichever command is current, so the marks run ahead of the robot. What was tried against that is in [What didn't work](WHAT_DIDNT_WORK.md).
