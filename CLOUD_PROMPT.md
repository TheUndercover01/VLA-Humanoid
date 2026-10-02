# Brief for the GPU-machine agent: "I Never Showed It the Task"

You are working on the GPU machine (an RTX 4090 with 24 GB) for a research project that is also a job application. Read this whole brief before doing anything. It is the single source of truth for the project's goal, the plan, and how your work fits with work happening elsewhere.

---

## 1. Why this project exists

The user is applying for a **Robot Learning Research internship at Humanoid**, a London company that builds the HMND-01 humanoid robot. The team areas are RL, world models, VLA pre- and post-training, and inference optimisation. The application requires a challenge, submitted as a **public GitHub repo**.

**Deadline: Friday 9 October 2026, 23:59 BST.** Work started on Fri 2 Oct, so there are 7 days in total. Plan everything around that.

### The challenge (paraphrased from the posting)
> Use real data **you personally collected** (e.g. a phone video of your own hand doing manipulation) to drive a robot manipulator in a simple simulation environment, creatively showcasing knowledge of **VLA and/or world models**. Example given: egocentric hand video → SmolVLA policy driving a Panda in LIBERO. Suggested directions: post-train a policy on your data, creative retargeting, bootstrap a policy and improve it with RL, world modelling, or optimise a policy to run faster. *"We're not looking for standard solutions, we're looking for how you think."*

### What they score (verbatim criteria)
1. **Creativity in approach**, while the collected data plays a real role.
2. **Performance of the policy in simulation** and/or quality of world-model predictions.
3. **Implementation simplicity and clear presentation of results, without AI slop.**
4. Submission: a README covering run instructions, example outputs, design choices, **what worked and what didn't**.

### What this means for how you work
- The reviewers are researchers. They value **honest, rigorous, well-explained results** over flashy claims.
- A negative result that is clearly explained is acceptable. An unexplained number is not.
- Keep the code **simple and readable**: small scripts, no framework-building, no speculative abstractions.
- Never pad the README or the code comments with filler.

---

## 2. The research idea

**Title:** *I Never Showed It the Task: composing human motion skills into unseen tasks with RL.*

**Setup.**
- The user films only **atomic skills** with a phone: `reach`, `push`, `pick-lift`, `place-down`.
- Each clip is one skill. **No clip ever shows a combined task.**
- The robot (Franka Panda, Isaac Lab) is then asked, by language prompt, to do **combined tasks never demonstrated as a whole**:

| ID | Prompt | Skill chain needed | Success criterion |
|---|---|---|---|
| C1 | "pick up the red cube and place it on the green target" | pick-lift → place-down | cube within 3 cm of target, resting |
| C2 | "stack the red cube on the blue cube" | pick-lift → place-down (precise) | red resting on blue for 1 s |
| C3 | "push the red cube out of the corner, then place it on the green target" | push → pick-lift → place-down | C1 criterion, starting from a corner where the cube can't be grasped |

Atomic tasks (push, lift, put down) are also evaluated, as a sanity check.

**Research question.** Given **exactly the same phone clips**, which approach composes skills into unseen multi-step tasks better, and *why*?
- **Standard:** fine-tune SmolVLA directly on the retargeted clips (labelled "push the cube", "pick up the cube", ...), then prompt it with the combined task.
- **Ours:**
  1. Turn the clips into a **motion vocabulary**: named skills, each with a small VAE latent for direction, height and distance.
  2. Train **RL** (PPO) whose action is *pick a skill + its latent*. The combined task exists **only in the reward**.
  3. **Distil** the RL experts into SmolVLA (image + prompt → skill + latent), so one VLA follows combined prompts.

**Hypotheses** (write the README results section against these):
- **H1, composition.** A VLA fine-tuned on atomic clips does not compose by itself. B1 succeeds on atomic tasks, but its success on C1–C3 is near zero.
- **H2, where it fails.** Given an LLM planner, B1 improves, but it fails at **skill transitions** (hand-offs such as the grasp pose before a lift). Planning isn't the bottleneck; execution between skills is.
- **H3, RL fixes transitions.** RL in the vocabulary learns the hand-offs from reward, so Ours beats B1+LLM on C1–C3, most clearly on C3, the longest chain.
- **H4, the vocabulary matters.** With the same RL budget, the vocabulary action space learns faster and gives smoother motion than raw end-effector actions (B2). It may *cap* precision on C2, and that is an expected, reportable limitation.

Every hypothesis may turn out false. The job is to find out and explain why.

**Prior work** (cite in the README; don't claim a new method; the novelty is the combination and the question):
- Learning from Play (Lynch et al. 2019)
- SPiRL / skill priors; Residual Skill Policies (arXiv 2211.02231)
- LAPA (2410.11758); CLAP (2601.04061); Motion-Focused Latent Action (2606.18955); VQ-VLA (ICCV 2025)
- Expert-to-VLA distillation: Exp2VLA (2607.03146); VLA-OPD (2603.26666)
- RL for VLAs: SimpleVLA-RL (2509.09674); SmolVLA-RL (github.com/Mervin-James/SmolVLA-RL)

---

## 3. Models compared (the core results table)

| Model | Training data | Combined prompts handled by | Purpose |
|---|---|---|---|
| **B0** SmolVLA zero-shot | none | directly | Floor. Report it, but it isn't the comparison. |
| **B1** SmolVLA on atomic clips | retargeted phone clips, replayed in sim for images | the whole prompt directly | **Main baseline:** does a VLA compose by itself? |
| **B1+LLM** | same as B1 | an LLM splits the prompt into atomic instructions, run in sequence (switch on a simple per-skill completion check) | Separates planning failure from execution failure |
| **B2** SmolVLA ← raw-action RL experts | rollouts of RL experts using raw IK-delta actions | trained on combined prompts | Isolates the vocabulary's contribution |
| **Ours** SmolVLA ← vocabulary RL experts | phone clips → vocabulary → RL rollouts | trained on combined prompts (no human demo of them) | Main method |
| **RL experts** (state-based) | — | — | Upper reference: how much distillation loses |
| **Oracle** SmolVLA ← scripted expert demos of C1–C3 | scripted sim demos | trained on combined prompts | Ceiling showing what full task demos would give. Context only. |

### Evaluation protocol (non-negotiable)
- **100 fixed randomised start states per task**, generated once with a fixed seed, saved to `sim/eval_states/*.pt`, and reused for **every** model.
- **3 training seeds** per model, wherever compute allows. If fewer are used, say so in the README.
- Report the **success rate with 95% CIs** (Wilson interval).
- **Failure breakdown:** the stage where each failure happened (reach, grasp, transport, place, or knocked over), detected from sim state.
- Secondary metrics: episode time, end-effector jerk, peak contact force.
- Log everything to CSV under `runs/` (gitignored). Only final tables and figures are committed, under `results/`.

---

## 4. Division of work

| Where | Owns |
|---|---|
| **Laptop** (Windows, no strong GPU) | `handtrack/` (ChArUco board, calibration, MediaPipe hand tracking, retargeting) and `motion/` (skill segmentation, vocabulary VAE). The phone clips are filmed and processed here. |
| **GPU machine (you)** | `sim/` (Isaac Lab envs, `VocabAction`, RL, rollouts, eval), `vla/` (LeRobot datasets, SmolVLA fine-tunes, LLM planner), `analysis/` |

The real phone data doesn't exist yet. Filming is planned for Sat 3 Oct. **Build everything so it runs on synthetic and scripted stand-ins first**, so that real data is a drop-in swap.

### Interface contracts (both sides must follow these)

**1. Retargeted skill clip:** `data/processed/clips/<skill>_<idx>.npz`

| Key | Shape | Meaning |
|---|---|---|
| `t` | (T,) | seconds, 10 Hz |
| `ee_pos` | (T, 3) | metres, **table frame**: origin at the table centre in front of the robot, x forward away from the robot base, y left, z up |
| `ee_yaw` | (T,) | radians, gripper yaw about z |
| `grip` | (T,) | 0 = open, 1 = closed |
| `obj_pos` | (T, 3) | cube position in the table frame, if tracked; NaN otherwise |
| `skill` | str | one of `reach`, `push`, `pick-lift`, `place-down` |

**2. Motion vocabulary checkpoint:** `data/processed/vocab.pt` (a torch dict)
- `skills`: list of skill names, which fixes the index order
- `latent_dim`: int
- `n_waypoints`: int
- `dt`: seconds between waypoints
- `decoder_state_dict` plus `decoder_config`, for the `motion.vae.SkillDecoder` class
- **Decoder API:** `decode(skill_idx: LongTensor[B], z: Tensor[B, latent_dim]) -> Tensor[B, n_waypoints, 5]`. The 5 values per waypoint are `(dx, dy, dz, dyaw, grip)`, deltas relative to the end-effector pose at the start of the motion, with grip absolute.

Until the real `vocab.pt` exists, create a **synthetic vocabulary** with the same API from scripted skill motions (`motion/synthetic.py`), and use it to build and test everything.

---

## 5. Your tasks, in priority order

Commit small, working steps often. After each milestone, update `PROGRESS.md` at the repo root: what's done, the numbers, open problems, and the next step. The laptop side reads it.

### Phase 0: environment (today)
1. Check: `nvidia-smi`, the OS (Ubuntu 22.04 or 24.04 expected), disk space (Isaac needs about 50 GB or more), and the driver version.
2. Install **Isaac Sim + Isaac Lab** following the **current official Isaac Lab pip installation docs**. That's Isaac Sim 5.x with Python 3.11 in a conda or uv env. Check the docs for exact versions; don't assume them from memory.
3. Smoke test: train stock PPO on `Isaac-Lift-Cube-Franka-v0` (rsl_rl, headless) and confirm it reaches the documented success or reward level. **Gate: don't move on until this works.**
4. Install **LeRobot** with SmolVLA support, in a separate env if the dependencies clash with Isaac. Check that `lerobot/smolvla_base` loads and runs inference on the 4090.

### Phase 1: baselines and infrastructure (no phone data needed)
Get **all baselines and the eval harness standing first.** They are the reference everything else is judged against.
1. **Task envs:** `sim/envs/` with C1, C2, C3 and the atomic tasks.
   - Start from the stock Franka lift and stack configs, with a shared scene: a table, red and blue cubes, a green target marker, and front and wrist cameras that can be switched off for state-based RL.
   - Tasks differ only in reset distribution, reward and success check.
   - Write the success and failure-stage detectors once, in a single module.
2. **Eval harness:** `sim/eval.py` plus `sim/eval_states/`. It runs any policy (state RL, scripted, or VLA) on the saved start states and writes a CSV with per-episode success, failure stage and metrics. `analysis/results_table.py` turns the CSVs into the markdown table with CIs.
3. **Scripted expert** for C1, C2, C3 and the atomic tasks: waypoint-based, using privileged state. It is a test oracle for the harness and the source of the Oracle data.
4. **Raw-action RL experts (B2 teachers):**
   - PPO with an IK-delta action on C1, C2, C3, with dense shaped rewards (reach → grasp → lift → transport → place stages).
   - Report learning curves against **simulated seconds**, and record the final success rate on the eval states.
   - This is also the "RL experts" upper-reference row.
5. **B0:** run zero-shot SmolVLA on all tasks through the harness. Expect about 0%; it's a floor.
6. **B1 pipeline dry run:**
   - Use scripted atomic-skill demos as a stand-in for the phone clips.
   - Build the full path: replay clips in sim with cameras → LeRobot dataset with atomic prompts → SmolVLA fine-tune → evaluate on atomic tasks and C1–C3.
   - When the real clips arrive, rerun it unchanged. That rerun is the real B1.
7. **`vla/llm_planner.py`** for B1+LLM: prompt → ordered list of atomic instructions. Keep it a fixed, simple prompt to an LLM API. If no API key is available, use a hand-written lookup and say so honestly in the README.

### Phase 2: our method
1. **`sim/actions/vocab_action.py`, an Isaac Lab ActionTerm.**
   - The action is `(skill logits | z)`.
   - `process_actions` decodes the motion with the vocabulary decoder.
   - `apply_actions` tracks the waypoints with the differential-IK controller over the physics substeps.
   - One env step equals one motion primitive.
   - Test it: z = 0 should give the mean motion per skill, and random z should give a smooth, varied motion. Record a short video.
2. **PPO with a hybrid action** (a categorical skill plus a Gaussian latent) via rsl_rl, or a minimal custom PPO if rsl_rl can't do hybrid actions cleanly. Train on C1, C2, C3 with the **synthetic vocabulary** first, then the real one.
3. Log the **skill sentences** per episode, e.g. `reach → pick-lift → place-down`. They are a key interpretability figure.
4. **Distillation:**
   - Roll out the experts with cameras on and convert the rollouts to a LeRobot dataset with combined prompts.
   - Fine-tune SmolVLA with the action vector `[skill one-hot | z]` to produce **Ours**, and with raw actions to produce **B2**.
   - Keep the data amount and fine-tuning budget identical across B2, Ours and Oracle.

### Phase 3: analysis (Wed–Thu)
- The results table: all models × all tasks, success rate with CIs.
- A failure-stage breakdown chart per model. This is what tests H2 and H3.
- RL learning curves: vocabulary vs. raw.
- Skill-sentence examples, and per-task frequency of each skill.
- Side-by-side videos: phone clip → robot doing the unseen combined task.
- For each hypothesis, write whether it held or not, plus the *why*, backed by the breakdowns.

### Parked (only if the core is done)
- Human vocabulary vs. a random-smooth vocabulary (is the benefit from human motion or from temporal abstraction alone?).
- Removing one skill to show the ceiling.
- Clips-per-skill scaling.
- Primitive length.

---

## 6. Schedule and gates

| Day | GPU machine (you) | Laptop | Gate |
|---|---|---|---|
| Fri 2 | Phase 0 | board, tracking code | Stock lift PPO works |
| Sat 3 | Phase 1.1–1.3 (envs, harness, scripted expert) | user films the clips | Harness scores the scripted expert ≈ 100% on C1 |
| Sun 4 | Phase 1.4–1.6 (raw RL experts, B0, B1 dry run); VocabAction on synthetic vocab | real tracking → vocab.pt | Raw RL expert solves C1 |
| Mon 5 | Vocabulary RL on C1–C3 (real vocab) | — | **Vocab RL solves C1. If not, drop distillation and submit the RL study.** |
| Tue 6 | Real B1, B1+LLM, rollout datasets | — | — |
| Wed 7 | Fine-tune B2, Ours, Oracle; full eval | — | Results table complete |
| Thu 8 | Figures, videos, README; repo made **public** | — | — |
| Fri 9 | Buffer; submit by 23:59 BST | — | — |

**If a phase slips, cut scope in this order:**
1. C3
2. Oracle
3. 3 seeds down to 1, stated honestly
4. B2's SmolVLA fine-tune, keeping the B2 RL expert
5. the whole distillation stage

Never cut the eval protocol or the honesty of the README.

---

## 7. Rules
- **Data:** the only human data is the user's own phone clips. Never add external human or robot datasets to the training of B1, B2 or Ours. Pretrained SmolVLA weights are fine, since they are standard.
- **"Never shown":** combined tasks must never appear in any human-derived training data. RL reward and scripted Oracle data are allowed, and are labelled as such.
- **Fairness:** B1 and Ours start from the same clips. B2, Ours and Oracle have the same distillation data size and fine-tuning budget. Every model uses the same eval start states.
- **Honesty:** never tune on the eval states. Report failed experiments in `PROGRESS.md`, because they feed the README's "what didn't work" section.
- **Simplicity:** plain scripts with argparse, and configs as small dataclasses or YAML. Comments match the surrounding code. No AI-slop prose.
- **Compute:** one 4090. Prefer headless, state-based RL with many parallel envs, and only turn cameras on for rollout recording and VLA evaluation. Check VRAM before running Isaac and a SmolVLA fine-tune at the same time.
- **Git:** small commits with clear messages. Never commit raw video, checkpoints or `runs/`. If you find the repo isn't on GitHub yet, ask the user before creating a **private** repo. It is made public only on Thu 8 / Fri 9.
- **Stop and ask the user** before anything destructive, anything that costs money, or any change to the research question, tasks or baselines.
