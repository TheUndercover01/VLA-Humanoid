"""B1+LLM planner: task prompt -> ordered list of atomic instructions (the B1 training prompts).

No LLM API key was available, so this is a hand-written lookup that plays the role of the
LLM planner. It is the plan an LLM would be expected to produce; the B1+LLM baseline tests
whether B1 can *execute* such a plan, which is the question it exists to answer (H2).

    python -m vla.llm_planner "stack the red cube on the blue cube"
"""
import sys

from vla.prompts import ATOMIC_PROMPTS, TASK_PROMPTS

R, P, PL, PD = (ATOMIC_PROMPTS[k] for k in ["reach", "push", "pick-lift", "place-down"])
PLANS = {
    TASK_PROMPTS["push"]: [R, P],
    TASK_PROMPTS["lift"]: [R, PL],
    TASK_PROMPTS["c1"]: [R, PL, PD],
    TASK_PROMPTS["c2"]: [R, PL, PD],
    TASK_PROMPTS["c3"]: [P, R, PL, PD],
}
SOURCE = "hand-written lookup (no LLM API key available)"


def plan(prompt):
    if prompt not in PLANS:
        raise KeyError(f"no plan for prompt: {prompt!r}")
    return list(PLANS[prompt])


if __name__ == "__main__":
    for step in plan(" ".join(sys.argv[1:])):
        print(step)
