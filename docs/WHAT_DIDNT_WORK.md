# What didn't work

[← back to the README](../README.md) · [How it works](HOW_IT_WORKS.md) · [Evaluation](EVALUATION.md) · [Results](RESULTS.md) · [Failures](FAILURES.md) · [Reproduce](REPRODUCE.md)

A lot of things didn't work before this did. Every number has its cause in the full log, [PROGRESS.md](../PROGRESS.md). Numbers are out of 100 validation layouts unless noted; "in order" = every command's result held, one after the other.

## Rewards and the teacher

| | Attempt | What happened | Numbers |
| :---: | :--- | :--- | :--- |
| ✗ | **Fine-tune the VLA on single skills** | It didn't combine them. | push 16 · pick-up 0 · every combined task 0 |
| ✗ | **Clean scripted demonstrations** | It fit the perfect demos but had never seen a mistake, so it couldn't recover from its own. | 6 on the 2-step task |
| ✗ | **A reward for grasping and carrying** | The policy sat on the release threshold and opened and closed the fingers to collect both terms. | 0, while "placing" 99.6% of the time in training |
| ✗ | **A reward for distance to the goal** | Pushing the cube onto the target was the cheap way to get closer, so it stopped lifting. | 19 when a real lift was required |
| ✗ | **A vocabulary of motion primitives** | RL over learned 2-second skills preferred pushing even more. | ~75% of successes were pushes |
| ✗ | **Starting every episode at home, object-only reward** | Nothing changes until the gripper touches the object by chance. | nothing learned in 504 iterations |
| ✗ | **A loose "hand has arrived" check** | The fingers closed on the cube's top edge and it slipped. | best lift 1.6 cm (the clips lift 14 cm) |
| ✗ | **Imitating raw RL actions** | Bang-bang actions flip sign every step; the VLA fit them no better than the average. | 0 on every task but push (14) |
| ✗ | **Penalising action changes / smoothing labels afterwards** | Barely smoothed the teacher; lower loss, same poor fit. | loss 0.93 → 0.77 |
| ✗ | **The hand leg without a push penalty** | The teacher "pushed" by picking the cube up and carrying it, so its student couldn't tell push from pick up. | push episodes lifted 85 of 100; student push 10 |
| ✗ | **Rest speed and hold time measured from the clips** | The teacher trained well but its student scored below every other student. | student unstack 17 / 50, restack 7 / 50 |
| ✗ | **A stricter put-down check taken from the clips (1.34 cm, "cliptol")** | The teacher got better at long chains (restack 83, 8 commands 98 in order), but its student stalls around command 3–4 of 6, at any scoring check. | student restack in order 7 (strict) / 6 (loose); unstack 85 loose |

## Memory and the done flag

| | Attempt | What happened | Numbers |
| :---: | :--- | :--- | :--- |
| ✗ | **A camera history and a trained vision encoder** | More inputs (an older frame, past actions, unfreezing vision) didn't help. | worse than the plain student |
| ✗ | **A counter in the input instead of marks** | The 6th position never came up in training. | restack 0 in order (marks: 13) |
| ✗ | **Padding the sentence to 20 commands** | Every position comes up, but the short tasks pay for it at the same budget. | c2 75 → 42, lift 77 → 51 |
| ✗ | **Stricter done flag (hold 10 / 15 ticks, or 0.9 / 8)** | Marks get more accurate, but many runs never mark at all. | hold 10: unstack 54 lenient / 45 in order, restack 13 / 5 |
| ✗ | **Voting over 3 samples** | No better than one. | unstack 32 in order (single: 31) |
| ✗ | **Re-arm (the flag must drop before the next mark)** | The flag isn't stuck after a mark; it fires too early. Re-arming changes nothing. | restack 15 / 1 |
| ✗ | **Self-check (re-ask whether the last mark is still true)** | It finds its wrong marks (93–100% of erased marks really weren't done) but doesn't redo the step. | random 8-chains in order 2.5 → 1.0 commands |
| ✗ | **"(in progress)" on the current command** | The tag didn't change the long-chain scores. | unstack 31 in order (without: 30–32), restack 2 |
| ✗ | **Random pauses in the recordings + "text contrast" frames at every handoff** (14 Oct) | Meant to teach the flag to read the sentence: the same image labelled done under one sentence and not done under the next. The model averaged the two labels instead, so the flag stays well below 0.85 and almost never writes a mark. The pauses also made the arm hesitate. | lift 97 → 56, unstack in order 51 → 11, restack 0; lift runs with a mark: 30 → **0** |

## What did work

| | Finding | Numbers |
| :---: | :--- | :--- |
| ✓ | **A reward built from object paths**, plus the hand's leg. The teacher learns every skill from a random start pose, no scripted states. | teacher: restack 94 / 100 in order |
| ✓ | **Paired episodes** (same scene, "cube" vs "cylinder") fixed which object the policy acts on. | earlier students: lift 45 → 80 |
| ✓ | **Imitating the filtered, executed action**, not the raw RL action. | raw actions: no better than the mean |
| ✓ | **The policy's own "(done)" marks as memory.** | marks off: unstack 0 / 50; two marks pre-written: 42 / 50 |
| ✓ | **Marks in the text beat a number in the state.** | restack in order 13 vs 0, unstack 29 vs 4 |
| ✓ | **No lifting during a push** (the final teacher). | the best whole-sentence student on 6 commands: 40 in order |
