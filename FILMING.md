# Filming the skill clips (Mon 5 Oct)

The reward is built from **where the cubes go** in these clips (their 8 corners over time) and **when the
hand closes and opens**. So what matters most: the cube's full pose in every frame (ArUco), a fixed calibrated
camera, and one clean skill per clip.

## Setup (once)
- **Camera**: phone on a tripod, fixed for the whole session, about 45 degrees down onto the table, the whole
  workspace and the ChArUco board in view. 1080p, 30 or 60 fps. If the phone moves, film the board alone again.
- **Board**: ChArUco board flat on the table at a fixed spot; it defines the table frame. Film 5 s of the
  board alone at the start of the session. Mark on the table which side is "the robot" (x points away from it).
- **Workspace**: keep cubes within about 30 cm x 40 cm in front of the board's origin, the area the sim robot
  reaches (sim: x -12..18 cm, y -20..20 cm from the table centre).
- **Cubes**: two cubes, red and blue, ideally **5 cm** (the sim size; tell me the exact size if not). An
  **ArUco marker on the top face** of each (different IDs; note the marker size), more on the sides if easy.
- **Target**: a **6 cm square** green pad (paper) with its own ArUco marker next to it or on it.
- **Hand**: thumb and index finger only, like a two-finger gripper. Grasp from the sides so the top marker
  stays visible; keep the fingertips in view.
- Even light, no glare on the markers, plain table.

## Every clip
- **One skill per clip.** Never film a whole task (e.g. pick up then place) in one take: the method's claim is
  that no clip shows a combined task.
- Hold still ~1 s at the start and ~1 s at the end.
- Natural speed, nothing staged.

## What to film (about 60-80 clips, 1-2 hours)
| skill | clips | how |
|---|---|---|
| **push** | 15+ | fingertips behind the cube, slide it 10-20 cm along the table onto the target pad; vary the direction (forward, sideways, diagonal) and start spots; finish with the cube resting on the pad, hand off it |
| **pick-lift** | 15+ | hand around the cube (start with fingers open next to it), close, lift straight up ~10-15 cm, **hold still ~1 s** at the top |
| **place-down** | 15+: 8 onto the target pad, 8 **onto the other cube** (stacking) | **start recording already holding the cube in the air** (don't film the pick-up), carry 15-30 cm, lower, **open the fingers, move the hand away**, hold still |
| **reach** (optional) | 10 | hand from various start points to around a cube, fingers open, stop without touching it (used only as start states) |
| **failures** | 3-5 per skill | a push that stops short, a grasp that slips, a place that misses or knocks the cube: name them with `_fail` |

## Files
- Name: `<skill>_<NN>.mp4` (e.g. `place-down_07.mp4`), failures `<skill>_<NN>_fail.mp4`; for place-down add
  `_stack` or `_target` (e.g. `place-down_03_stack.mp4`).
- A short `notes.txt`: cube size, marker IDs and marker size, board squares/size, which board edge faces the
  robot, phone model and fps.

## What happens with them
Board -> camera pose -> table frame; ArUco -> each cube's pose per frame -> 8 corners; MediaPipe -> finger
closure (open/closed). Each clip becomes `t, ee_pos, ee_yaw, grip, obj_pos, obj_quat, skill` at 10 Hz, then
`motion/keypoint_ref.py` turns them into waypoints, spread, lift height, end grip and turn allowance, exactly as
with the stand-ins. Failures (`_fail`) tighten the tolerances.
