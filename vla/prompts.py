"""Language prompts. Atomic prompts label the skill clips (B1 training data); task prompts are the eval prompts."""
from sim.envs.tasks import PROMPTS as TASK_PROMPTS  # noqa: F401

ATOMIC_PROMPTS = {
    "reach": "reach the red cube",
    "push": "push the red cube to the green target",
    "pick-lift": "pick up the red cube",
    "place-down": "put the red cube down",
}
