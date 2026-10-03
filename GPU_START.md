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

## 5. Environments on the GPU machine (as installed, 3 Oct)
Isaac Sim 5.1 needs Python 3.11 and numpy 1.26; LeRobot 0.6.2 needs Python ≥ 3.12 and numpy 2, so there are two conda envs. They exchange files only (datasets, checkpoints).

Driver is 535.183 (CUDA 12.2). Isaac Sim 5.1 was tested on 580, but it runs on 535. Torch must be a **cu128** build in both envs: cu130 wheels need driver ≥ 580.

```bash
# Isaac Lab (official pip install, Isaac Sim 5.1.0 + Isaac Lab main @ b0542fe, v2.3.2)
conda create -n isaaclab python=3.11
pip install "isaacsim[all,extscache]==5.1.0" --extra-index-url https://pypi.nvidia.com
pip install -U torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/isaac-sim/IsaacLab.git && cd IsaacLab && OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh --install

# LeRobot + SmolVLA (lerobot main @ ff71cae, v0.6.2)
conda create -n lerobot -c conda-forge python=3.12 "ffmpeg>=6,<8"
git clone https://github.com/huggingface/lerobot.git && cd lerobot && pip install -e ".[smolvla]"
pip install --force-reinstall torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install "numpy>=2.0,<2.3"
```
On this machine the shell profile puts another env first on `PATH`, so call `<env>/bin/python` explicitly. Isaac scripts run through `IsaacLab/isaaclab.sh -p <script>` with the `isaaclab` env active.
