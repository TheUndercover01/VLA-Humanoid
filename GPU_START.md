# Starting on the GPU machine

## 1. Get the code
```bash
git clone https://github.com/TheUndercover01/VLA-Humanoid.git
cd VLA-Humanoid
```
The repo is private for now, so git will ask for a login. Use a GitHub personal access token as the password, or run `gh auth login`.

## 2. Start Claude Code in the repo and paste this

```text
Read CLOUD_PROMPT.md fully, then PROGRESS.md, then sim/env.py, sim/scene.py and sim/scripted.py.

Context that is newer than the brief:
- A MuJoCo prototype of the task suite already exists in sim/ (written, never run). It defines the
  tasks push / lift / c1 / c2 / c3, their success checks, failure-stage tracking, rewards and a
  scripted expert. Treat it as the specification of the tasks.
- The brief prefers Isaac Sim + Isaac Lab. Do Phase 0 for Isaac Lab first. If it is not working
  after about 3 hours, tell me and fall back to MuJoCo (pip install mujoco; python -m sim.fetch_assets)
  and run the existing prototype instead of porting it.
- C3 was redefined (see PROGRESS.md): the cube starts beyond comfortable grasp reach and must be
  pushed closer before it is picked and placed. Measure the real reach limit and set C3_FAR_X.
- The real phone data does not exist yet. Build and test everything on scripted / synthetic stand-ins.

Start with Phase 0 and report back with: GPU and driver, OS, disk space, and whether the stock
Franka lift PPO smoke test passes. Then continue with Phase 1 in the order the brief gives.
Update PROGRESS.md and commit after every milestone. Ask me before anything destructive or
anything that costs money.
```

## 3. What you provide as input
| When | What |
|---|---|
| Now | Nothing except the paste above. |
| If asked | Your GitHub token (to push) and an LLM API key, only if you want the B1+LLM planner to call a real model. Otherwise it uses a hand-written planner and says so. |
| After you film (Sat 3) | The raw phone videos, copied to `data/raw/<skill>/` on the laptop. You do **not** need to send them to the GPU machine: the laptop turns them into `data/processed/clips/*.npz` and `vocab.pt`, which you commit or copy over. |

## 4. Pushing results back
The agent commits on a branch (`git checkout -b gpu`) and pushes it. Merge into `main` from the laptop, so that both machines never edit the same files at once.
