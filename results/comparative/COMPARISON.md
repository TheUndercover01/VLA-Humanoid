# Comparative study: every student, every chain

Success rates out of 100 held-out layouts per chain, lenient / in order (in order only where it was measured; the lenient check only looks at the last command, in order needs every command done in turn). Cells are the latest evaluation of that student and setting; `-` = not evaluated. Chains: push, lift (cube), lift_cyl (cylinder), c1 (to target), c1_cyl, c2 (cube on cylinder; in the red/blue runs red on blue), c3 (push then stack), swap (cylinder on cube; in the red/blue runs blue on red), unstack (4 commands), restack (6 commands). **Groups A and B use red/blue cubes with a random target, group C uses a cube and a cylinder with a fixed target, so numbers across groups are not like for like.** Student metadata: `students.csv`; every evaluation line: `evals_long.csv`; all raw evaluation csvs: `eval_csvs/`.

## A. Earlier students, red/blue cubes, random target, 1000 unpaired episodes, 20k steps, frozen SmolVLA (strict success is not shown; lenient only)

| student / mode | push | lift | lift_cyl | c1 | c1_cyl | c2 | c3 | swap | unstack | restack |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.85 teacher, whole sentence | 5/5 | 64/64 | - | 54/51 | - | 41/41 | 14/0 | 41/41 | 16/0 | 23/0 |
| 0.85 teacher, one command, env done (oracle) | 3 | 57 | - | 57 | - | 45 | 9 | 37 | 24 | 17 |
| 0.95 teacher, whole sentence | 48 | 44 | - | 36 | - | 35 | 0 | 34 | 7 | 7 |
| 0.95 teacher, one command, env done (oracle) | 41 | 45 | - | 39 | - | 42 | 3 | 29 | 41 | 15 |
| 0.85 + state history, whole sentence | 14 | 27 | - | 9 | - | 15 | 1 | 21 | 0 | 1 |
| v2 near view + VLM trained + history frame, whole | 9 | 45 | - | 37 | - | 43 | 7 | 40 | 2 | 30 |
| X-VLA on 0.85 data, whole sentence | 6 | 70 | - | 31 | - | 25 | 1 | 28 | 9 | 2 |

## B. Paired counterfactual episodes (same layout, each object commanded), red/blue cubes, random target, 3000 episodes

| student / mode | push | lift | lift_cyl | c1 | c1_cyl | c2 | c3 | swap | unstack | restack |
|---|---|---|---|---|---|---|---|---|---|---|
| paired 49K, whole sentence, no done | 40/40 | 80/80 | - | 80/80 | - | 75/75 | 4/0 | 66/66 | 14/1 | 12/0 |
| paired 49K, one command, env done (oracle) | 41/41 | 78/78 | - | 81/81 | - | 75/75 | 16/5 | 82/82 | 76/62 | 53/53 |
| paired 32K snapshot, whole sentence | 32/32 | 83/83 | - | 73/72 | - | 72/72 | 2/0 | 75/75 | 22/3 | 9/0 |
| paired 32K snapshot, one command, env done (oracle) | 42/42 | 81/81 | - | 76/76 | - | 83/83 | 13/3 | 80/80 | 76/60 | 53/52 |

## C. Cube + cylinder, fixed target, real-clip straight template, 3000 paired episodes, 33K steps (early stopped)

| student / mode | push | lift | lift_cyl | c1 | c1_cyl | c2 | c3 | swap | unstack | restack |
|---|---|---|---|---|---|---|---|---|---|---|
| one command + learned done flag 0.5 / 3 ticks | 61/61 | 93/93 | 75/75 | 91/91 | 75/74 | 87/87 | 40/12 | 72/72 | 88/21 | 57/11 |
| one command + learned done flag 0.7 / 5 ticks | - | - | - | - | - | - | 43/14 | 76/76 | 86/51 | 62/41 |
| one command + learned done flag 0.85 / 8 ticks | - | - | - | - | - | - | 31/18 | 49/49 | 69/64 | 36/35 |
| one command + learned done flag 0.85 / 5 ticks | 57/57 | 94/94 | 72/72 | 91/91 | 68/68 | 90/90 | 38/14 | 68/68 | 90/65 | 53/38 |
| WHOLE sentence + done count (memory), 14K steps | - | - | - | - | - | 48/48 | 34/10 | - | 56/29 | 7/0 |
| WHOLE sentence + done count (memory), 33K steps | 23/23 | 77/77 | 55/55 | 83/83 | 53/52 | 75/75 | 43/12 | 49/49 | 75/42 | 5/0 |
| teacher (RL policy, privileged state), model 399 | 95/95 | 100/100 | 98/98 | 100/100 | 88/88 | 84/84 | 76/18 | 96/96 | 93/82 | 83/83 |
