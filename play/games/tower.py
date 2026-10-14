"""Tower duel rules (10 Oct). Two players alternately add a flat piece to one tower: the human adds a TILE (5.5 cm square) with an offset
of their choice, the robot adds a DISC (r 3.5 cm) centred on the piece below it, which is what the one student does for "pick up the disc,
then put the disc on the tile". Physics decides the rest: whoever's piece makes the tower fall (or any piece slide off) loses. A tower of
MAX_PIECES that stands is a draw. The robot never decides anything here: it cannot place badly on purpose, the human can only try to
make the robot's placement hard by leaning the tower."""
from dataclasses import dataclass, field

MAX_PIECES = 6
MAX_OFFSET = 0.045                  # m: the human's piece may be put this far off the centre of the piece below (x and y)
TILE, DISC = 4, 3                   # tasks.ROSTER indices (slot types)
FALL_XY = 0.02                      # m: a piece that moved this far sideways since it was settled counts as fallen
FALL_Z = 0.012                      # m: or dropped this far below where its height in the tower should be


@dataclass
class Tower:
    human_first: bool = True
    pieces: list = field(default_factory=list)          # [{'owner': 'human'|'robot', 'type': TILE|DISC}] bottom to top
    over: str = ""                                      # '' | 'human_wins' | 'robot_wins' | 'draw'
    skips: int = 0                                      # turns lost (the robot fumbled its disc): the order of play shifts by one each

    def next_player(self):
        if self.over:
            return None
        n = len(self.pieces) + self.skips
        return ("human" if self.human_first else "robot") if n % 2 == 0 else ("robot" if self.human_first else "human")

    def piece_type(self, player):
        return TILE if player == "human" else DISC

    def place(self, player):
        assert player == self.next_player(), (player, self.next_player())
        self.pieces.append({"owner": player, "type": self.piece_type(player)})
        if len(self.pieces) >= MAX_PIECES:
            self.over = "draw"

    def skip(self):
        """The player to move loses the turn (the robot dropped its disc next to the tower)."""
        self.skips += 1

    def fell(self, player):
        """The tower fell right after `player` placed a piece: the other one wins."""
        self.over = "robot_wins" if player == "human" else "human_wins"

    def clamp_offset(self, dx, dy):
        c = lambda v: max(-MAX_OFFSET, min(MAX_OFFSET, float(v)))  # noqa: E731
        return c(dx), c(dy)


def has_fallen(positions, rest):
    """positions: {piece index: (x, y, z)} now; rest: the same when the tower last settled. True if any piece moved sideways or dropped."""
    for i, (x, y, z) in rest.items():
        px, py, pz = positions[i]
        if abs(px - x) > FALL_XY or abs(py - y) > FALL_XY or pz < z - FALL_Z:
            return True
    return False
