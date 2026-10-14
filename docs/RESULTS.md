# All results

[← back to the README](../README.md) · [How it works](HOW_IT_WORKS.md) · [Evaluation](EVALUATION.md) · [Failures](FAILURES.md) · [What didn't work](WHAT_DIDNT_WORK.md) · [Reproduce](REPRODUCE.md)

100 validation layouts per task. Cells are **lenient / in order** (one number when they're equal). [How a run is scored →](EVALUATION.md)

## Final model: `ours_cyl_marks_nolift` vs its teacher (loose check)

| Task | Commands | Never shown? | Student: cameras + sentence + own "(done)" marks | Mean commands in order | RL teacher: exact positions, told the command |
| :--- | :---: | :---: | ---: | ---: | ---: |
| pick up the cube | 1 | | 94 | | 100 |
| pick up the cylinder | 1 | | 59 | | 100 |
| push the cube to the target | 1 | | 59 | | 100 |
| cube to the target | 2 | | 97 | 1.94 / 2 | 100 |
| cylinder to the target | 2 | | 77 | 1.59 / 2 | 100 |
| cube on the cylinder | 2 | | 96 | 1.92 / 2 | 100 |
| cylinder on the cube | 2 | | 81 | 1.64 / 2 | 100 |
| push the cylinder to the target, then stack the cube on it | 3 | **yes** | 64 / 62 | 2.16 / 3 | 94 |
| **stack, then unstack to the target** | **4** | **yes** | **84 / 81** | 3.63 / 4 | 100 |
| **stack, unstack, stack the cylinder on the cube** | **6** | **yes** | **49 / 40** | 4.25 / 6 | 94 |
| stack, unstack, stack, unstack | 8 | **yes** | 82 / 0 | 2.75 / 8 | 100 |
| random 8-command chain, seed 1 | 8 | **yes** | 85 / 0 | 2.96 / 8 | 100 |
| random 8-command chain, seed 2 | 8 | **yes** | 54 / 0 | 2.08 / 8 | 100 |

- **Up to 6 commands work; 8 don't yet.** On 4 commands 81 of 100 runs do every step in order, on 6 commands 40. On 8, no run does all eight in order; on average it gets 2–3 done before its marks run ahead of it (see [Failures](FAILURES.md)).
- **The teacher is close to perfect** on everything, 8 commands included (100 of 100), because it's told which command it's on. The gap between the columns is the price of the student keeping its own place from two cameras.
- **The weakest single skills involve the cylinder** (lift 59, to the target 77, on the cube 81) and pushing (59). Over six commands those misses add up.
- The 3-command task pushes the cylinder, a command that was later removed from the vocabulary because the cylinder topples; it's kept here as recorded.

## Other students on the long tasks

Same robot, same tasks, different training. In order, out of 100.

| Student | What's different | 4 commands | 6 commands | Check |
| :--- | :--- | ---: | ---: | :--- |
| **`nolift` (final)** | teacher: hand leg, random start, no lifting during a push | **81** | **40** | loose |
| `cliptol` | teacher trained with a stricter 1.34 cm put-down check from the clips | 85 | 6 | loose |
| `nolift` | same model, strict check | 32 | 12 | ±1.5 cm, 0.5 s |
| `nolift` | same model | 52 | 24 | ±2.14 cm |
| `progress` | "(in progress)" written on the current command | 52 | 7 | ±2.14 cm |
| `handleg_rand` | teacher: hand leg + random start, pushes by lifting | 54 | 6 | ±2.14 cm |
| `realref` (the earlier headline) | scripted mid-skill starts, sentence padded to 20 | 35 | 15 | ±1.5 cm |
| `pause` | random pauses + text-contrast frames in training | 11 | 0 | 1.34 cm |

`cliptol` matches `nolift` on 4 commands but stalls at command 3–4 of 6 under any check; `nolift` is the only student that gets far on 6. Reasons for each attempt: [What didn't work](WHAT_DIDNT_WORK.md).

## For comparison: one command at a time

When an outside script feeds the student one command at a time (the student's own done flag still decides when to move on, but it never has to find its place in a sentence), an earlier student did 4 commands 65 in order (90 lenient) and 6 commands 38 in order (53 lenient), strict check. That's the ceiling of the memory part, and the reason the whole-sentence student is the interesting one: it gets close without any outside help.

## Videos

16 layouts per task, same loose check (separate run): 6 commands 5 of 16 in order, 4 commands 14 of 16, 8 commands 0 of 16, cube to target and cube on cylinder 8 of 8. The picked success and failure videos and the contact sheets are in `media/headline/final/` (`REPORT.md` lists every run).
