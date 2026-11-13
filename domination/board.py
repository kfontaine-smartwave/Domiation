"""Board model shared by the live agent and the local simulator.

`Topology` is the static map (cells, adjacency, regions); `State` is one position seen from the
player about to act. Players are small integers (0..P-1), neutral territories have owner -1.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

NEUTRAL = -1


class Topology:
    def __init__(self, ids, neighbours, regions=()):
        """ids: cell names; neighbours: list of index lists; regions: [(bonus, [cell indices])]."""
        self.ids = list(ids)
        self.index = {name: i for i, name in enumerate(self.ids)}
        self.n = len(self.ids)
        self.nbrs = [sorted(set(v)) for v in neighbours]
        deg = max((len(v) for v in self.nbrs), default=1) or 1
        self.nbr_pad = np.full((self.n, deg), self.n, dtype=np.int64)   # index n = dummy cell
        for i, v in enumerate(self.nbrs):
            self.nbr_pad[i, :len(v)] = v
        self.regions = [(float(b), sorted(c)) for b, c in regions if c]
        self.n_regions = len(self.regions)
        self.region_bonus = np.array([b for b, _ in self.regions], dtype=np.float32)
        self.region_size = np.array([len(c) for _, c in self.regions], dtype=np.float32)
        self.region_member = np.zeros((self.n_regions, self.n), dtype=np.float32)
        self.region_of = np.full(self.n, -1, dtype=np.int64)
        for r, (_, cells) in enumerate(self.regions):
            self.region_member[r, cells] = 1.0
            self.region_of[cells] = r
        self.total_bonus = float(self.region_bonus.sum())


@dataclass
class State:
    topo: Topology
    owner: np.ndarray            # int8  [n]
    armies: np.ndarray           # int32 [n]
    n_players: int
    me: int
    turn: int = 1
    turn_max: int = 150
    income_div: int = 3          # one reinforcement per `income_div` territories (minimum 3)
    pending: float = 0.0         # bonus armies already earned for my next turn (quest reward)
    mission: tuple | None = None  # (cell index, bonus) of my active conquest quest
    revolt: tuple = (True, 0.5, 0.1)   # (active, share-of-map threshold, chance per weak cell)
    objective: tuple | None = None     # (regions needed, territories needed) in "objectifs" mode
    names: list = field(default_factory=list)   # server keys of the players ("j1", ...)
    order: list = field(default_factory=list)   # players in turn order (indices); empty = 0..P-1

    def copy(self) -> "State":
        return replace(self, owner=self.owner.copy(), armies=self.armies.copy())

    @property
    def turn_frac(self) -> float:
        return min(1.0, self.turn / max(1, self.turn_max))

    def cells_of(self, player: int) -> np.ndarray:
        return np.flatnonzero(self.owner == player)

    def income(self, player: int) -> int:
        cells = self.owner == player
        t = int(cells.sum())
        if t == 0:
            return 0
        bonus = 0.0
        topo = self.topo
        if topo.n_regions:
            held = topo.region_member @ cells.astype(np.float32) >= topo.region_size - 0.5
            bonus = float(topo.region_bonus[held].sum())
        return max(3, t // self.income_div) + int(round(bonus))

    def alive(self) -> list[int]:
        return [q for q in range(self.n_players) if (self.owner == q).any()]


def state_from_view(view: dict, topo: Topology | None = None) -> State:
    """Build a State from the JSON returned by the agent API (`/api/agent/attendre`, `/action`...)."""
    game = view["partie"]
    board = game["plateau"]
    cells = board["territoires"]
    if topo is None or topo.ids != [c["id"] for c in cells]:
        index = {c["id"]: i for i, c in enumerate(cells)}
        neighbours = [[index[a] for a in c["adjacents"] if a in index] for c in cells]
        regions = [(r.get("bonus", 0), [index[t] for t in r["territoires"] if t in index])
                   for r in (board.get("regions") or [])]
        topo = Topology([c["id"] for c in cells], neighbours, regions)
    names = sorted(game["joueurs"])
    number = {name: i for i, name in enumerate(names)}
    owner = np.array([number.get(c["proprietaire"], NEUTRAL) for c in cells], dtype=np.int8)
    # A hidden army count (fog during secret deployment) is treated as a small garrison.
    armies = np.array([c["armees"] if c.get("armees") is not None else 2 for c in cells], dtype=np.int32)
    me = number[view["vous"]]
    state = State(topo, owner, armies, len(names), me, turn=int(game.get("tour") or 1),
                  turn_max=int(game.get("tour_max") or 150), names=names)
    state.order = [number[k] for k in (game.get("ordre") or names) if k in number]
    state.income_div = _infer_income_divisor(game, names, owner)
    mission = game.get("mission")
    if mission and mission.get("territoire") in topo.index:
        if mission.get("accomplie"):
            state.pending = float(mission.get("bonus") or 0)
        else:
            state.mission = (topo.index[mission["territoire"]], float(mission.get("bonus") or 0))
    rev = game.get("revoltes") or {}
    state.revolt = (bool(rev.get("actives")), float(rev.get("seuil_pct", 50)) / 100.0,
                    float(rev.get("chance_pct", 10)) / 100.0)
    obj = game.get("objectifs")
    if game.get("mode_victoire") == "objectifs" and obj:
        state.objective = (int(obj.get("regions_cible", 0)), int(obj.get("territoires_cible", 0)))
    return state


def _infer_income_divisor(game: dict, names: list, owner: np.ndarray) -> int:
    """The "long game" economy hands out fewer reinforcements: recover the divisor from the
    reinforcement forecast the server publishes for each player."""
    votes: dict[int, int] = {}
    for i, name in enumerate(names):
        info = game["joueurs"][name]
        t = int((owner == i).sum())
        if t == 0 or info.get("renforts_prevus") is None:
            continue
        base = info["renforts_prevus"] - (info.get("bonus_regions") or 0)
        for div in (3, 2, 4, 5, 6, 8):
            if max(3, t // div) == base:
                votes[div] = votes.get(div, 0) + 1
    if not votes:
        return 4 if game.get("economie") == "longue" else 3
    best = max(votes.values())
    for div in (3, 4, 5, 6, 2, 8):
        if votes.get(div) == best:
            return div
    return 3
