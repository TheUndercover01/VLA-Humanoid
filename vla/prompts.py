"""Language prompts. Atomic prompts label the skill clips (B1 training data); task prompts are the eval prompts."""
from sim.envs.tasks import PROMPTS as TASK_PROMPTS  # noqa: F401

ATOMIC_PROMPTS = {
    "reach": "reach the red cube",
    "push": "push the red cube to the green target",
    "pick-lift": "pick up the red cube",
    "place-down": "put the red cube down",
}

# Keypoint commands (sim/envs/keypoint_env.py): one instruction per (skill, cube, destination). These are the only
# prompts in the distillation data (each frame carries the command active at that moment); combined tasks never
# appear, they are made by switching between these.
CUBES = ["red", "blue"]
DESTS = ["the green target", "the {other} cube", ""]


def command_prompt(skill, cube, dest):
    """'pick up the red cube', 'put the red cube on the blue cube', 'push the blue cube to the green target'."""
    c, other = CUBES[cube], CUBES[1 - cube]
    if skill == "pick-lift":
        return f"pick up the {c} cube"
    where = DESTS[dest].format(other=other)
    if skill == "place-down":
        return f"put the {c} cube on {where}"
    if skill == "push":
        return f"push the {c} cube to {where}"
    raise ValueError(skill)
