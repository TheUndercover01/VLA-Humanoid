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
import os  # noqa: E402
if os.environ.get("VLA_OBJECTS", "red_blue") == "cube_cylinder":       # 6 Oct: object 0 a cube, object 1 a cylinder
    CUBES = ["cube", "cylinder"]
    DESTS = ["the target", "the {other}", ""]
    NOUN = ["", ""]
else:
    CUBES = ["red", "blue"]
    DESTS = ["the green target", "the {other} cube", ""]
    NOUN = [" cube", " cube"]


MULTI = os.environ.get("VLA_OBJECTS", "red_blue") == "multi"      # 8 Oct: six named object types (sim/envs/tasks.py ROSTER)
if MULTI:
    from sim.envs.tasks import HOLDOUT_PAIRS, NO_PUSH, ROSTER, STACK_BASES, TRAIN_TYPES
    NAMES = [r[0] for r in ROSTER]


def multi_vocab(holdout=True):
    """Every command a training episode can contain, for the padding of build_dataset (the held-out object and pairs left out)."""
    objs = list(TRAIN_TYPES) if holdout else list(range(len(NAMES)))
    out = [command_prompt("pick-lift", t, 7) for t in objs] + [command_prompt("place-down", t, 6) for t in objs]
    out += [command_prompt("push", t, 6) for t in objs if t not in NO_PUSH]
    out += [command_prompt("place-down", t, b) for t in objs for b in STACK_BASES if b in objs and b != t and not (holdout and (t, b) in HOLDOUT_PAIRS)]
    return out


def command_prompt(skill, cube, dest):
    """'pick up the red cube', 'put the red cube on the blue cube', 'push the blue cube to the green target'."""
    if MULTI:       # dest: 0-5 = on top of that object type, 6 = the target, 7 = none
        where = "the target" if dest == 6 else f"the {NAMES[dest]}" if dest < 6 else ""
        return {"pick-lift": f"pick up the {NAMES[cube]}", "place-down": f"put the {NAMES[cube]} on {where}",
                "push": f"push the {NAMES[cube]} to {where}"}[skill]
    c, other = CUBES[cube], CUBES[1 - cube]
    n = NOUN[cube]
    if skill == "pick-lift":
        return f"pick up the {c}{n}"
    where = DESTS[dest].format(other=other)
    if skill == "place-down":
        return f"put the {c}{n} on {where}"
    if skill == "push":
        return f"push the {c}{n} to {where}"
    raise ValueError(skill)


def marked_sentence(cmds, k):
    """The whole instruction with the first k commands marked as done (build_dataset --prompt_mode marks; at inference
    sim/vla_policy.py writes the marks from the policy's own done flag)."""
    return ", then ".join(c + (" (done)" if i < k else "") for i, c in enumerate(cmds))
