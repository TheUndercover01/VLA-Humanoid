# I Never Showed It the Task

I film myself doing only **atomic skills** with a phone: reach, push, pick-lift and place-down. A Franka Panda in simulation is then asked, with a language prompt, to do **combined tasks I never demonstrated**:

| Task | Prompt |
|---|---|
| C1 | pick up the red cube and place it on the green target |
| C2 | stack the red cube on the blue cube |
| C3 | push the red cube closer, then place it on the green target |

Given the *same phone clips*, this project compares two ways of getting there:

| Approach | Pipeline |
|---|---|
| **Standard** | Fine-tune SmolVLA directly on the clips |
| **Ours** | Turn the clips into a **motion vocabulary**, let **RL** discover how to chain the skills from the task reward alone, then distil the RL experts into SmolVLA |

The question is which one composes skills into unseen tasks better, and why. The full research plan, baselines and evaluation protocol are in [CLOUD_PROMPT.md](CLOUD_PROMPT.md).

> **Status: work in progress.** There are no results yet. See [PROGRESS.md](PROGRESS.md) for what is done and what isn't.

## Pipeline

```
phone clips (atomic skills) -> hand tracking -> retarget to Panda -> motion vocabulary
                                                                   -> RL (combined-task reward, actions = skills)
                                                                   -> SmolVLA distillation (image + prompt -> skill)
```

## Repo layout

| Folder | Contents |
|---|---|
| `handtrack/` | ChArUco board, camera calibration, hand tracking (MediaPipe), retargeting to the Panda end effector |
| `motion/` | Skill segmentation and the motion-vocabulary VAE |
| `sim/` | Panda scene, task environments, scripted expert, RL training and evaluation |
| `vla/` | SmolVLA datasets, fine-tuning, and the LLM planner baseline |
| `analysis/` | Tables, plots and videos |

## Quick start (simulation prototype)

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
uv pip install mujoco numpy scipy gymnasium stable-baselines3 matplotlib pandas
python -m sim.fetch_assets          # downloads the Panda model from mujoco_menagerie
```

Setup on the GPU machine: [GPU_START.md](GPU_START.md).
