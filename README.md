# I Never Showed It the Task

I filmed myself doing only **atomic skills** with a phone: reach, push, pick-lift and place-down. A Franka Panda in Isaac Sim is then asked, with a language prompt, to do **combined tasks I never demonstrated**:

- pick and place,
- stack,
- push, then place.

Given the *same phone clips*, this project compares two approaches:

| Approach | Pipeline |
|---|---|
| **Standard** | Fine-tune SmolVLA directly on the clips |
| **Ours** | Turn the clips into a **motion vocabulary**, let **RL** discover how to chain the skills, then distil into SmolVLA |

The question is which one composes skills into unseen tasks better, and why.

> Status: work in progress. Results, videos and run instructions will be added here.

## Pipeline

```
phone clips (atomic skills) -> hand tracking -> retarget to Panda -> motion vocabulary
                                                                   -> RL in Isaac Lab (combined-task reward)
                                                                   -> SmolVLA distillation (image + prompt -> skill)
```

## Repo layout

| Folder | Contents |
|---|---|
| `handtrack/` | Camera calibration, hand tracking (MediaPipe + ArUco), retargeting to the Panda end effector |
| `motion/` | Skill segmentation and the motion-vocabulary VAE |
| `sim/` | Isaac Lab environments, the vocabulary action term, RL training and evaluation |
| `vla/` | SmolVLA datasets, fine-tuning, and the LLM planner baseline |
| `analysis/` | Tables, plots and videos |
