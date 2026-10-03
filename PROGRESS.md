# Progress

## Fri 2 – Sat 3 Oct (laptop)
Done:
- Repo scaffold, ChArUco board generator (`handtrack/make_board.py`). One board covers camera calibration and the table frame.
- `CLOUD_PROMPT.md`: the full brief (challenge, plan, baselines, protocol, rules).
- **MuJoCo prototype of the task suite** in `sim/`, written but **not yet run or tested**:
  - `scene.py`: Panda + table + red/blue cubes + green target, damped-least-squares Cartesian controller.
  - `env.py`: gymnasium env with tasks `push`, `lift`, `c1`, `c2`, `c3`, success checks, failure-stage tracking, reward, jerk and peak-force metrics.
  - `scripted.py`: privileged-state waypoint expert for every task.
  - `fetch_assets.py`: downloads the Panda model (not committed).

Not done / open:
- The laptop venv (Python 3.11) exists, but the libraries (torch, mujoco, gymnasium, ...) are not installed: the first install failed on a DNS error. Retry `uv pip install`.
- **C3 definition changed.** A "cube in a slot" is physically unsound: any wall that blocks a top-down grasp also blocks the pusher. C3 is now "the cube starts beyond comfortable grasp reach (x in 0.46–0.52 m), push it closer, then pick and place". `C3_FAR_X` in `sim/env.py` is a guess. Measure the real reach limit in sim and set it.
- Eval harness (`sim/eval.py`, saved start states), synthetic vocabulary, VAE, `VocabAction`, RL training, VLA pipeline: not started.

## GPU machine

### Sat 3 Oct: Phase 0 done (gate passed)
Machine: 2× RTX 4090 24 GB (the brief assumed one), driver 535.183.06 (CUDA 12.2), Ubuntu 22.04.4, kernel 6.8, 48 cores, 125 GB RAM. Disk: `/` has only 39 GB free, so envs, caches, Isaac Lab and LeRobot live on `/media/storage` (2.2 TB free).

Envs (recipe in `GPU_START.md` §5): `isaaclab` (Py 3.11, Isaac Sim 5.1.0, Isaac Lab 2.3.2, torch 2.7.0+cu128, rsl-rl-lib 5.0.1) and `lerobot` (Py 3.12, LeRobot 0.6.2, torch 2.11.0+cu128). Two envs because the Python and numpy requirements conflict.

Lift smoke test (`Isaac-Lift-Cube-Franka-v0`, rsl_rl PPO, 4096 envs, headless), scored by `sim/isaac_check_lift.py` (1024 envs, one 5 s episode):

| Checkpoint | Iterations | Time lifted | Final dist to goal < 5 cm | Median final dist |
|---|---|---|---|---|
| NVIDIA published | — | 0.988 | 1.000 | 3 mm |
| seed 42 (default) | 1500 | 0.032 | 0.000 | 457 mm |
| seed 1 | 3000 | 0.985 | 0.996 | 3 mm |
| seed 2 | 3000 | 0.976 | 0.295 | 65 mm |

1500 iterations take about 13 min on one GPU. Seed 1 matches the published checkpoint, so the install is good. Takeaway for our own RL: the stock lift reward is **seed-sensitive**. Seed 42 never found the lift in 1500 iterations; seed 2 lifts but tracks the goal poorly even at 3000. Plan for several seeds and longer runs than the defaults.

SmolVLA (`vla/smoke_smolvla.py`): `lerobot/smolvla_base` loads (450M params), one 50×6 action chunk takes 198 ms median on a 4090 (first call 619 ms), peak VRAM 0.86 GiB.

### Sat 3 Oct: Phase 1.1–1.3 (task envs, eval harness, scripted expert)
Isaac Lab port of the MuJoCo prototype. The MuJoCo files (`sim/env.py`, `scene.py`, `scripted.py`, `fetch_assets.py`) are superseded but kept for reference.
- `sim/envs/panda_env.py`: one DirectRLEnv for all tasks. Panda (high-PD config), table, 5 cm red/blue cubes, visual green target, optional front + wrist cameras. Same 5-D action as the prototype (dx, dy, dz, dyaw, grip), 10 Hz, integrated TCP command tracked by DLS IK at 100 Hz. 21-D state observation in the table frame.
- `sim/envs/tasks.py`: everything task-specific in one module: layout sampling, success, failure-stage tracking, shaped reward.
- `sim/expert.py`: batched scripted expert. `sim/eval.py`: runs any policy on the 100 saved start states per task (`sim/eval_states/*.pt`, seed 1234), writes a CSV per episode (success, failure stage, time, jerk, peak force, skill sentence), optional video. `analysis/results_table.py`: markdown tables with Wilson CIs.

Scripted expert on the eval states (gate "≈100% on C1" passed):

| push | lift | c1 | c2 | c3 |
|---|---|---|---|---|
| 100% | 100% | 100% | 100% | 68% (parked) |

Mean episode time 3.3 s (lift, push), 5.6 s (c1), 5.1 s (c2). Videos: `media/` (see `media/README.md`).

Findings and changes (all feed the README's "what didn't work"):
- **Reach.** Measured with `sim/probe_reach.py`: a vertical gripper reaches ~0.78 m from the base (x = 0.275 at y = 0 in the table frame, 0.25 at y = 0.2) and no closer than ~0.3 m. Tilting the hand gains only ~5 cm. The prototype's ranges (cubes up to x = 0.30, C3 at 0.46–0.52) were out of reach. Objects now spawn in x ∈ [-0.12, 0.18], |y| ≤ 0.2; the TCP command is clamped to 0.32–0.77 m from the base (beyond that the DLS IK settled in a retracted, tilted pose). **Laptop side: retargeted clips should land inside this box when replayed in sim.**
- **C3 redefined again (user decision).** "Push it closer, then pick" is impossible with a vertical gripper: getting behind a cube that is out of grasp reach is even further out. C3 is now "push the blue cube onto the green target, then stack the red cube on it". Its start layouts require the pushing position behind blue to be reachable (0.36–0.74 m from the base); this was added after seeing expert failures on the C3 eval states, before any model was evaluated. **C3 is parked:** the expert gets 68%, all failures are timeouts (long pushes plus pick and place do not fit in 15 s, and pushes from the edge of reach). Likely fixes: a tighter pushability margin and a longer C3 time limit.
- **Pushing.** A single closed-finger contact spun or tipped the cube (4% success). Fixed: hand yawed along the push, fingers at a 3 cm gap so both fingertips touch the face, push at 0.2 m/s, closed loop on the cube's current line to the target, push height 3 cm (2 cm drags the fingertips on the table). Tuned on separate layouts (seed 999), not the eval states. Cube friction 1.5 → 1.0.
- **Slow arm.** The stock high-PD Franka config (damping 80) with the 12 Nm wrist torque limit caps the wrist at 0.15 rad/s, so every Cartesian move crawled at ~6 cm/s. Damping 80 → 30 halved episode times and fixed C1's timeouts.
- **Stock lift reward is seed-sensitive** (Phase 0): plan several seeds for RL.
- Known issue: with cameras on, physics is not bit-identical to headless (push 98% vs 100%). VLA evals run with cameras, so compare models under the same setting.

### Sat 3 Oct: Phase 1.4 started (raw-action PPO, the B2 teachers)
- `sim/train_rl.py` (rsl_rl PPO, 4096 envs, 32 steps/iter, MLP 256-128-64, obs normalisation; config in `sim/rl_cfg.py`). 2.3 s/iteration, i.e. ~13,000 simulated seconds per iteration. Logs success and per-stage reach rates. `analysis/learning_curves.py`: success vs simulated hours, per task and action space.
- `sim/eval.py --policy rsl:<checkpoint>` evaluates a trained policy on the eval states.
- **What didn't work: the prototype's reward makes PPO hover over the goal.** First C1/C2 runs (seed 1, ~600-700 iterations): reach, grasp, lift and transport in ~100% of episodes, place ~0%, success 0%. "Held" only counted while the cube was ≥ 2 cm up, so lowering it onto the target lost reward before any place reward appeared (≈5.5/step hovering vs ≈3 while lowering). Also, success counted "gripper open" as a gap > 4 cm, which is true while holding a 5 cm cube. Fixed: grasp detected by contact geometry, carry rewarded by 3-D distance to the resting pose, an at-goal term that pays whether or not the cube is held, and release = gap > 6.5 cm. Scripted expert still 100% on C1/C2 under the stricter check. Training also switched from success termination with a one-off +20 to a +5 bonus for every step in the success state (eval still terminates once success has held for 0.5 s). Runs restarted.
- Training can use either GPU; **camera rendering only works on cuda:0** in Isaac Sim 5.1.
- Videos: `media/06_scripted_c3.mp4` added (scripted C3, two successful episodes).

### Sat 3 Oct: C3 un-parked and fixed (user request)
Scripted expert on the eval states: push 100, lift 100, c1 100, c2 100, **c3 89** (was 68).
- C3 time limit 15 s → 25 s (`tasks.EPISODE_S`): it chains push + pick + place; successful episodes take 9.7 s on average, up to ~20 s.
- C3 layouts: the pushing spot behind blue must be 0.36–0.68 m from the base (was 0.74; pushes from the edge of reach failed). Eval states for c3 regenerated (no model evaluated yet).
- Expert: the grasp follows red's live pose and yaw (the push nudged or spun red, and the old plan grasped at its start position); grasp yaw kept in ±45° and push yaw in ±2 rad so joint 7 stays in range (yaw ±135° made the hand tilt and stall); for C3's long pushes, if blue drifts off the line the gripper lifts, goes round behind it and pushes again (this hurt the short push task, 100 → 97, so it is only used for C3).
- Expert changes were debugged on training layouts (seeds 555, 777, 999: 92–97%); eval states only for the final numbers. Remaining C3 failures: blue slides off sideways at the table edge, a grasp landing on top of the cube, stacking that shifts blue > 3 cm off the target.

### Sat 3 Oct (evening): reward fixed again, stand-in clips, stand-in vocabulary
- **What didn't work (2): the policy sat on the release threshold.** With the first reward fix, C1 reached "place" in 99.6% of training episodes but success-at-end was 0.0000 and eval 0/100. Trace: the cube rests on the target and the policy opens the gripper to ~65 mm, exactly the release threshold, so it collects the grasp/carry reward (gap < 65 mm) and the success bonus (gap > 65 mm) on alternate steps; success never holds 5 steps. Fixed: grasp/carry only pay away from the goal pose; at the goal only the at-goal term pays, growing with the gripper opening; the success bonus needs success held for 0.5 s (as in eval). Checked by profiling the reward along the scripted expert (rises at every phase: lower ≈ 3.6–4.0, open ≈ 8.5, done ≈ 11.2). C1/C2 retrained from scratch (2000 iterations) so the B2 learning curves use a single reward.
- Expert bug found via the clips: the C3 fix made the lift follow red's live position, so the hand chased the cube it was holding (up to 20 cm drift while lifting; tasks still succeeded). Close and lift now move vertically. Expert: lift/c1/c2 100, c3 90.
- **Stand-in clips** (`sim/make_standin_clips.py`, one source task per process): 50 each of reach, pick-lift, place-down, push in the clip interface format, cut from scripted-expert episodes on training layouts (never a whole combined task). In `data/processed/clips_standin/` (not committed).
- **Motion vocabulary** (`motion/vae.py`): skill-conditioned VAE, 10 waypoints of (dx, dy, dz, dyaw, grip) relative to the start pose, latent_dim 3, implements the `vocab.pt` contract (`SkillDecoder.decode`). Stand-in vocabulary `data/processed/vocab_standin.pt`: waypoint reconstruction error reach 8 mm, push 9 mm, pick-lift 0.6 mm, place-down 16 mm; z = 0 decodes to the mean motion of each skill. **Laptop side: `motion/` did not exist yet, so this is the reference implementation; train the real vocabulary with `python -m motion.vae --clips data/processed/clips --out data/processed/vocab.pt` rather than writing a second one.**

### Sat 3 Oct (night): vocabulary action, vocabulary RL, B1 pipeline running end to end
- Env refactor: bookkeeping (success hold, stages, metrics, reward) runs on a 10 Hz control tick inside the physics loop, so one env step can span many ticks. Raw action = 1 tick per step; checked identical per-episode results for the scripted expert (push, c1, c2: 100/100 episodes same time and outcome).
- `sim/envs/vocab_env.py` (Phase 2.1): action = 4 skill scores + 3 latent dims; skill = argmax (rsl_rl only has Gaussian actions), decoded by `SkillDecoder` into 10 waypoints relative to the current command, interpolated onto ~20 control ticks (≈ 2 s per primitive), same controller and clamps as the raw action. `sim/test_vocab_action.py`: z = 0 plays each skill's mean motion (reach ≈ 27 cm down, pick-lift 14 cm straight up and closes, place-down lowers 13 cm and opens), random z gives varied motions. Video `media/08_vocab_action.mp4`.
- `sim/train_rl.py --action vocab`: γ = 0.99²⁰ per primitive (same horizon in simulated seconds as raw). Vocabulary RL on C1 (stand-in vocabulary) is running; one iteration covers 10× more simulated time than a raw one, so curves are compared on simulated hours.
- B1 pipeline: `sim/replay_clips.py` replays every clip in sim with cameras (scripted pre-roll sets up the start situation, then the clip's poses are commanded with the raw action, which is what is recorded). Stand-in replays track the clips to 0.4–4 mm; pick-lift lifts the cube, place-down sets it down, push moves it 12.6 cm. `vla/build_dataset.py` → LeRobot dataset (200 episodes, 4330 frames, 10 fps, front + wrist video, 6-D state, 5-D raw action, atomic prompts only). `scripts/finetune_smolvla.sh` fine-tunes `lerobot/smolvla_base` (front/wrist → camera1/camera2): 0.31 s/step at batch 32, 7.7 GB. B1 dry run: 6000 steps.
- `vla/llm_planner.py`: no LLM API key, so the B1+LLM planner is a hand-written lookup (prompt → atomic instructions). Stated as such.
- Infrastructure notes: run long jobs detached (`scripts/run_isaac.sh`, `setsid nohup`), background shells are killed after 2 h; the lerobot env needs `lerobot[dataset,training]` with torch pinned to cu128; accelerate needs `NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1` on RTX 4090s; torchcodec does not load (LeRobot falls back to PyAV).
- Raw RL after the reward fix (training metrics, not eval): C1 at iteration 767 success-at-end 14%, place 55% and rising; C2 at 600 place 17%.

### Sat 3 Oct (late): first baseline numbers; reward fix 3; all RL restarted
- VLA bridge: `vla/server.py` (lerobot env) serves SmolVLA action chunks over a local socket (raw bytes, so numpy 1/2 does not matter; sub-batches of 25 envs to bound GPU memory); `sim/vla_policy.py` replans every 10 steps; `--plan` follows the planner's atomic prompts. `scripts/eval_vla_suite.sh` runs B and B+LLM on all tasks.
- `vla/check_fit.py`: the B1 stand-in model reproduces its own training actions with MAE 0.03–0.085 vs 0.17–0.39 for a mean predictor, so the VLA pipeline is correct.
- **Baseline results on the eval states (stand-in clips; the real B1 comes from the phone clips):**

| Model | push | lift | c1 | c2 | c3 |
|---|---|---|---|---|---|
| B0 smolvla_base | 0 | 0 | 0 | 0 | 0 |
| B1 (stand-in clips) | 16 | 0 | 0 | 0 | 0 |
| B1+LLM (lookup planner) | 3 | 29 | 0 | 0 | 0 |

  Lift needs reach + pick-lift chained: B1 alone never grasps (reaches in 24%), with the plan it lifts in 29%. The plan hurts push (switching reach → push mid-motion). Nothing composes into c1–c3.
- **Privileged information:** RL experts and the scripted expert observe exact cube/target positions (21-D state): they are the teachers and the upper reference. Every SmolVLA model sees only the front and wrist images, 6-D proprioception (TCP pose, yaw, finger gap) and the prompt. Caveat for the README: B1+LLM switches instructions on a completion check computed from sim state, which is privileged help (allowed by the brief, makes B1+LLM stronger than on a real robot).
- **What didn't work (3):** with the reward of fix 2, the vocabulary agent learned reach → pick-lift (99.7%) and then held the cube in the air (0/100 on the eval states, all failures at transport): carry credit only counted while grasped, and place-down always ends by opening, so any imprecise place-down lost reward. Raw C2 put red on blue in 90% of training episodes but never held success: the stay-near-the-cube term kept the open fingers jostling the stacked cube (speed ~0.09 m/s > 0.03 rest threshold). Fix: the cube's distance to its goal pose pays whether or not it is held; at the goal pose the grasp and stay-near terms switch off. Reward profile along the expert rises at every phase for c1, c2, c3. **All three RL runs restarted from scratch** (raw c1, raw c2, vocab c1) so raw and vocabulary use the identical reward.
- Raw C2 had crashed once (Isaac `carb` mutex assertion while the GPU was overloaded); `--resume` added to `sim/train_rl.py`.
- Videos: `media/09_b0_c1.mp4` (B0 wandering), `media/10_vocab_rl_c1_wip.mp4` (vocabulary agent holding the cube up, before fix 3).

### Sat 3 Oct (night): C1 RL solved, but by pushing; C1 now requires a lift (user decision)
- With reward fix 3, C1 RL works: raw-action PPO reached 98% training success (iteration ~1750); vocabulary PPO (stand-in vocabulary, 250 iterations) scored **81/100 on the eval states**.
- **What didn't work (4): both agents mostly pushed the cube onto the target instead of picking it up.** Skill sentences of the vocabulary agent were mostly `reach > push > reach`; re-scored with "the cube must have been lifted ≥ 5 cm", it drops to **19/100** (58 episodes never grasp), and the raw agent's last checkpoint gets **47/100**. Fix 3 made the cube's distance to the goal pay whether or not it is held, which made pushing the cheap solution. Video: `media/11_vocab_rl_c1_push_vs_lift.mp4` (top: pushed success, bottom: picked and placed). A finding for the README: with coarse 2 s primitives, RL preferred pushing even more often (~75% of successes) than with raw actions (~50%).
- **C1 redefined (user decision):** success also needs the cube to have been lifted ≥ 5 cm at some point (`tasks.LIFTED`, latched per episode in the env as `was_lifted`), matching the prompt "pick up ... and place". C1 reward: goal and at-goal credit only after the lift, plus a lift bonus while grasped before it. Scripted expert still 100/100 on C1; reward profile rises at every phase; 22 tests pass (new: pushing to the goal earns less than a lifted cube far away). C1 RL (raw and vocabulary) restarted. C2 and C3 cannot be solved by pushing and are unchanged. B0/B1 C1 numbers were 0 and stay 0.
- Oracle distillation data recorded: 500 successful scripted episodes each for c1, c2, c3 (c3 needed 600 tries), with front/wrist images; LeRobot dataset being built.
- RL runs now: raw c1 + raw c2 (GPU 0), vocab c1 + vocab c2 (GPU 1).

- Cheating check (user request). Old raw C1 (iteration 1700) re-scored with the eval code from before the change: **100/100 under the old rule, 47/100 with the lift rule**, so 53 of its successes were pushes (vocabulary: 62 of 81). Raw C2 at iteration 800: **76/100, red lifted in all 100 episodes**, red sits on blue in the videos, so no shortcut; but it stacks hard (peak contact force ~60–75 N and jerk ~12–14 vs ~17 N and ~6.5 for the scripted expert). Eval CSVs now carry a `lifted` column (red cube ≥ 5 cm up at some point). Video `media/12_raw_rl_c2_it800.mp4`.

Next: evaluate the RL runs when they finish; then stand-in atomic clips (cut from the scripted expert, one skill per clip) for the B1 pipeline and the synthetic vocabulary.
