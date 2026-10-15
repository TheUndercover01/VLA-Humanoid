# Details

## Reward from clips

ArUco markers give each object's path in 43 single-skill clips (12 pushes, 10 pick-ups, 21 put-downs). Eight points on the simulated object have to follow the skill's path within 3 cm (the spread of my own clips). The reward never looks at the gripper; every gripper-based reward I tried got gamed.

## Teacher

PPO in Isaac Lab, 16,384 parallel envs, random start poses, one or two commands per episode. The arm executes a low-pass filtered action (0.95), otherwise PPO learns bang-bang control the VLA can't imitate. A small penalty for lifting during a push stops it from pushing by carrying.

## Student

SmolVLA from `smolvla_base`, trained on 4500 teacher episodes. Each scene is recorded twice, once with "cube" and once with "cylinder", so it has to read the sentence to pick the object. An extra output predicts "this command is done"; when it stays above 0.85 for 0.5 s, "(done)" is written into the sentence. Training sentences are padded to 10 commands so every position shows up.

Without the marks the 4-command task succeeds 0/50 times. A counter in the input instead of marks: 5/100 on 6 commands.

## Scoring

A run counts only if the simulator saw every command done in order. Pick up = raised 4 cm; on the target / other object = within 5 cm, held 0.2 s. With a stricter 1.5 cm check the 6-command score is 12.

## Limits

The done flag fires too early. On 8 commands the marks run ahead after four and the robot stops. The cylinder skills are the weakest (59 pick-up).
