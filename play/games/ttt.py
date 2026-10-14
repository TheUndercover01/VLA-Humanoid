"""Tic-tac-toe rules and the robot's choice of square (10 Oct). The robot does not learn this: the game decides WHERE a piece goes, the one
student only ever hears "pick up the cube, then put the cube on the target" with the target pad moved to that square.

Squares are numbered like sim/envs/multi_env.SQUARES: index = row * 3 + col with row along the table's x (0 = far from the camera) and
col along y (0 = left on screen). 'X' = the robot (cubes), 'O' = the human (cylinders)."""
import random
from functools import lru_cache

# 10 Oct (user: "robot should lose sometimes"): the chance that the robot plays its perfect move (otherwise a random free square).
# Measured over 600 games each with the human moving first: normal = a human wins about a quarter to a third of the games, hard never loses.
DIFFICULTY = {"easy": 0.5, "normal": 0.8, "hard": 1.0}
DEFAULT_DIFFICULTY = "normal"
LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8), (0, 4, 8), (2, 4, 6)]


def winner(board):
    """'X' / 'O' if that player has three in a row, 'draw' if the board is full, None while the game goes on."""
    for a, b, c in LINES:
        if board[a] and board[a] == board[b] == board[c]:
            return board[a]
    return "draw" if all(board) else None


def choose(board, me="X", skill=1.0, rng=random):
    """The robot's square. skill = chance of playing the perfect move, otherwise a random free square (1.0 = never loses)."""
    free = [i for i, v in enumerate(board) if not v]
    if not free:
        return None
    if rng.random() >= skill:
        return rng.choice(free)
    if len(free) == 9:
        return rng.choice([0, 2, 4, 6, 8])                  # any corner or the centre is perfect; vary the opening
    best, moves = -2, []
    other = "O" if me == "X" else "X"
    for q in free:
        board[q] = me
        s = _value(board, other, me)
        board[q] = None
        if s > best:
            best, moves = s, [q]
        elif s == best:
            moves.append(q)
    return rng.choice(moves)


@lru_cache(maxsize=None)
def _value_t(board, to_move, me):
    w = winner(list(board))
    if w == "draw":
        return 0
    if w in ("X", "O"):
        return 1 if w == me else -1
    other = "O" if to_move == "X" else "X"
    vals = [_value_t(board[:q] + (to_move,) + board[q + 1:], other, me) for q in range(9) if not board[q]]
    return max(vals) if to_move == me else min(vals)


def _value(board, to_move, me):
    """Game value for `me` (1 win, 0 draw, -1 loss) with `to_move` to play and both sides perfect."""
    return _value_t(tuple(board), to_move, me)
