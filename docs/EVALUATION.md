# How it's evaluated

[← back to the README](../README.md) · [How it works](HOW_IT_WORKS.md) · [Results](RESULTS.md) · [Failures](FAILURES.md) · [What didn't work](WHAT_DIDNT_WORK.md) · [Reproduce](REPRODUCE.md)

## One run

1. A **validation layout** is drawn: where the cube, the cylinder and the target are (seed 4321, never used in training). Each task is tried on **100 layouts**.
2. The robot starts at its home pose and gets the **whole sentence once**, e.g. *"pick up the cube, then put the cube on the cylinder, then …"*.
3. It sees only the front camera, the wrist camera and the sentence. **Nothing outside the model tells it which command it's on.** Its own done flag (a 6th action output) writes "(done)" after the current command when it stays above 0.85 for 5 steps (0.5 s) in a row.
4. The run has **6 s per command + 5 s** (41 s for 6 commands).

## Two scores

- **In order** (the headline): the simulator keeps its own count. Command 1's result has to be reached and held, then command 2's, and so on. A run counts only if **every command was done, one after the other**. The robot's marks don't count for anything here; only what happened on the table does.
- **Lenient**: only the final scene the sentence asks for, held at the end, however the robot got there. A run that skips a step but still ends right counts here, not in order.

## What "done" means for each kind of command

| Command | Counts when | Strict (old default) | **Loose (used for all results)** |
| :--- | :--- | :---: | :---: |
| pick up X | X rose at least … above where it started | 5 cm | **4 cm** |
| put X on the target | X is within … of the target's centre (xy) and below … | 3 cm, 4.5 cm | **5 cm, 6 cm** |
| put X on the other object | X is within … of the other object's centre (xy), and its height is within … of resting on top | 3 cm, ±1.5 cm | **5 cm, ±3 cm** |
| push X to the target | same as "put X on the target" | 3 cm, 4.5 cm | **5 cm, 6 cm** |
| every result | must hold for … | 0.5 s | **0.2 s** |

The **robot is not loosened**: its done flag still needs 0.85 for 5 steps, and the teacher it learned from was trained with its own unchanged check (about 2 cm). Only the judge got less picky.

## Why loose

In the videos, many runs the strict check called failures had done the task: the cube sat on the cylinder, a few millimetres outside the ±1.5 cm band. The same student on the same 6-command task:

| check | lenient | in order |
| :--- | ---: | ---: |
| ±1.5 cm, 0.5 s (strict) | 37 | 12 |
| ±2.14 cm (the teacher's own done check) | 45 | 24 |
| **loose** | **49** | **40** |

Loosening is not a free lunch for every model. The clip-tolerance student (`cliptol`) stays at **6–7 in order** on the same task under every check, because it really stops at command 3–4. And the teacher is scored with the same loose check (94 in order on 6 commands, 100 on 8).

At 5 cm and ±3 cm the check is about as wide as it can be while still meaning "on it": the cube is about 5 cm wide, so at the limit it hangs half off the cylinder. The remaining failures are real: wrong object, knocked over, or never put down (see [Failures](FAILURES.md)).

## Strict numbers, for the record

`ours_cyl_marks_nolift`, 100 layouts, strict check (±1.5 cm, 0.5 s), lenient / in order: push 23, lift 96, lift the cylinder 62, cube to target 96, cylinder to target 77 / 76, cube on cylinder 93, cylinder on cube 66, **4 commands 78 / 32**, **6 commands 37 / 12**. At ±2.14 cm: cube on cylinder 89, cylinder on cube 82, **4 commands 81 / 52**, **6 commands 45 / 24**.

## Rerun it

```bash
# student (any NAME), loose check, 100 validation layouts per task
NAME=ours_cyl_marks_nolift G=0 CHAINS="restack unstack restack8" scripts/eval_cliptol_loose.sh
# teacher, same check
G=0 CHAINS="restack unstack restack8" scripts/eval_teacher_loose.sh
# one task by hand
scripts/run_isaac.sh out.log sim/eval_chains.py --headless --keypoints --hold_fix --chain restack --val \
  --policy vla --vla_port 6292 --replan 1 --learned_done --marks --whole_prompt \
  --xy_tol 0.05 --stack_ztol 0.03 --target_z 0.06 --lift_rise 0.04 --lenient_hold 2 --ordered_hold 2
```

Per-episode CSVs: `runs/eval/nolift_loose_<task>.csv` (student), `runs/eval/teacher_loose_nolift_<task>.csv` (teacher). Columns `lenient`, `ordered`, `ordered_done` (commands done in order), `knocked`.
