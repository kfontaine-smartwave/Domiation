"""Local simulator of the game, used for self-play training and offline benchmarks.

It reproduces what was observed on the real server: grid maps (oval or rectangle) with mirrored
regions, players starting on opposite sides with 3 armies per territory and neutral armies in
between, reinforcements = territories // 3 (minimum 3) + bonuses of fully owned regions,
classic dice, one fortification per turn, conquest quests (+3 / +6 armies), revolts (a player
holding half of the map may see conquered territories left with one army turn neutral with two
armies), and the turn limit decided on territories then armies. What the server keeps private
(when exactly a quest is handed out, which territories count as "conquered") is approximated.
"""
from __future__ import annotations

import numpy as np

from .battle import expected_blitz, simulate_blitz
from .board import NEUTRAL, State, Topology
from .features import extract

# ---------------------------------------------------------------------------- maps


def grid_cells(width: int, height: int, oval: bool = True):
    cells = []
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    rx, ry = width / 2.0 + 0.6, height / 2.0 + 0.4
    for y in range(height):
        for x in range(width):
            if not oval or ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0:
                cells.append((x, y))
    return cells


def _mirrored_regions(cells, width, rng):
    """Random contiguous regions on the left half, mirrored to the right (as on the server)."""
    pos = {c: i for i, c in enumerate(cells)}
    half = [c for c in cells if c[0] < width / 2.0]
    free = set(half)
    groups = []
    while free:
        seed = min(free, key=lambda c: (c[0] + rng.random() * 2, c[1] + rng.random() * 2))
        size = int(rng.integers(4, 13))
        group, frontier = [seed], [seed]
        free.discard(seed)
        while frontier and len(group) < size:
            x, y = frontier[int(rng.integers(len(frontier)))]
            options = [(x + dx, y + dy) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (x + dx, y + dy) in free]
            if not options:
                frontier.remove((x, y))
                continue
            nxt = options[int(rng.integers(len(options)))]
            free.discard(nxt)
            group.append(nxt)
            frontier.append(nxt)
        groups.append(group)
    small = [g for g in groups if len(g) < 3]          # merge crumbs into a neighbouring region
    groups = [g for g in groups if len(g) >= 3] or groups
    for g in small:
        if g not in groups:
            target = min(groups, key=lambda h: min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for a in g for b in h))
            target.extend(g)
    regions = []
    for g in groups:
        bonus = int(np.clip(round(len(g) / 4.0 + rng.uniform(-0.7, 0.3)), 1, 6))
        regions.append((bonus, [pos[c] for c in g]))
        mirror = [(width - 1 - x, y) for x, y in g]
        regions.append((bonus, [pos[c] for c in mirror if c in pos]))
    return regions


def random_setup(rng, n_players: int | None = None, size: tuple | None = None):
    """Returns (topology, owner, armies, n_players) for a fresh random game."""
    if size is None:
        size = [(10, 7), (10, 7), (10, 7), (10, 7), (10, 7), (8, 6), (12, 8), (14, 10)][int(rng.integers(8))]
    if n_players is None:
        n_players = int(rng.choice([2, 3, 4], p=[0.4, 0.35, 0.25]))
    width, height = size
    cells = grid_cells(width, height, oval=rng.random() < 0.8)
    pos = {c: i for i, c in enumerate(cells)}
    neighbours = [[pos[(x + dx, y + dy)] for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)) if (x + dx, y + dy) in pos]
                  for x, y in cells]
    scattered = rng.random() < 0.15                    # "planetary conquest"-like start
    if scattered:                                       # sea routes: a few long-distance links
        for _ in range(max(2, len(cells) // 12)):
            a, b = (int(v) for v in rng.integers(len(cells), size=2))
            if a != b and b not in neighbours[a]:
                neighbours[a].append(b)
                neighbours[b].append(a)
    regions = _mirrored_regions(cells, width, rng) if rng.random() < 0.9 else []
    ids = [f"{chr(65 + x)}{y + 1}" for x, y in cells]
    topo = Topology(ids, neighbours, regions)
    n = len(cells)
    owner = np.full(n, NEUTRAL, dtype=np.int8)
    armies = rng.integers(2, 6, size=n).astype(np.int32)
    if scattered:
        order = rng.permutation(n)
        for k, c in enumerate(order):
            owner[c] = k % n_players
        armies[:] = 1
        total = {2: 40, 3: 35, 4: 30}[n_players]
        for q in range(n_players):
            mine = np.flatnonzero(owner == q)
            extra = max(0, total - len(mine))
            np.add.at(armies, rng.choice(mine, size=extra), 1)
    elif n_players == 2:
        band = max(1, round(0.3 * width))
        for i, (x, _) in enumerate(cells):
            if x < band:
                owner[i] = 0
            elif x >= width - band:
                owner[i] = 1
        armies[owner >= 0] = 3
    else:
        corners = [(0, 0), (width - 1, height - 1), (width - 1, 0), (0, height - 1)][:n_players]
        share = max(3, int(n * 0.14))
        taken: set[int] = set()
        for q, (cx, cy) in enumerate(corners):
            ranked = sorted((i for i in range(n) if i not in taken),
                            key=lambda i: abs(cells[i][0] - cx) * 1.0 + abs(cells[i][1] - cy) * 1.3)
            for i in ranked[:share]:
                owner[i] = q
                taken.add(i)
        armies[owner >= 0] = 3
    return topo, owner, armies, n_players


# ---------------------------------------------------------------------------- game loop


class Game:
    expected = False        # True: battles resolve to their expected outcome instead of dice

    def __init__(self, topo, owner, armies, n_players, rng, turn_max: int = 150, income_div: int = 3,
                 revolt=(True, 0.5, 0.1), quests: bool = True):
        self.topo = topo
        self.owner = owner.copy()
        self.armies = armies.copy()
        self.n_players = n_players
        self.rng = rng
        self.turn_max = turn_max
        self.income_div = income_div
        self.revolt = revolt
        self.quests = quests
        self.turn = 1
        self.pending = [0.0] * n_players
        self.mission: list = [None] * n_players
        self.conquered = np.zeros(topo.n, dtype=bool)
        self.order = [int(q) for q in rng.permutation(n_players)]      # the server draws the order too
        self.samples: list[tuple[int, np.ndarray]] = []                # (player, features after its turn)
        self.stats = {"turns": 0, "attacks": 0}

    def state(self, player: int) -> State:
        return State(self.topo, self.owner, self.armies, self.n_players, player, self.turn, self.turn_max,
                     self.income_div, self.pending[player], self.mission[player], self.revolt, order=self.order)

    @classmethod
    def from_state(cls, s: State, rng) -> "Game":
        """A simulated continuation of a real position (used by the look-ahead simulations)."""
        g = cls(s.topo, s.owner, s.armies, s.n_players, rng, s.turn_max, s.income_div, s.revolt, quests=False)
        g.turn = s.turn
        g.order = list(s.order) or list(range(s.n_players))
        g.pending[s.me] = s.pending
        g.mission[s.me] = s.mission
        return g

    def alive(self) -> list[int]:
        return [q for q in range(self.n_players) if (self.owner == q).any()]

    def _maybe_quest(self, q: int):
        if not self.quests or self.mission[q] is not None or self.rng.random() > 0.3:
            return
        targets = sorted({t for c in np.flatnonzero(self.owner == q) for t in self.topo.nbrs[int(c)]
                          if self.owner[t] != q})
        if targets:
            t = int(targets[int(self.rng.integers(len(targets)))])
            self.mission[q] = (t, 6.0 if (self.owner[t] >= 0 or self.armies[t] >= 5) else 3.0)

    def play_turn(self, q: int, agent, record: bool = True):
        self.begin_turn(q, agent)
        self.finish_turn(q, agent, "attack", record)

    def begin_turn(self, q: int, agent):
        """Income, quest hand-out and reinforcement phase."""
        s = self.state(q)
        income = s.income(q) + int(self.pending[q])
        self.pending[q] = 0.0
        s.pending = 0.0
        self._maybe_quest(q)
        s.mission = self.mission[q]
        placed = 0
        for cell, nb in agent.place(s, income) or []:
            nb = min(int(nb), income - placed)
            if nb > 0 and self.owner[cell] == q:
                self.armies[cell] += nb
                placed += nb
        if placed < income:                                 # the server requires every army to be placed
            mine = np.flatnonzero(self.owner == q)
            self.armies[int(mine[np.argmax(self.armies[mine])])] += income - placed

    def attack(self, q: int, src: int, dst: int, occupation: str) -> bool:
        """Resolves one blitz. Returns True on conquest. With `expected` set, the dice are replaced
        by the most likely result with average losses (noise-free look-ahead)."""
        if self.expected:
            p, exp_att, exp_def = expected_blitz(int(self.armies[src]), int(self.armies[dst]))
            if p < 0.5:
                self.armies[src], self.armies[dst] = 1, max(1, int(round(exp_def)))
                return False
            left = max(2, int(round(exp_att)))
            self.conquer(q, src, dst, left, left - 1 if occupation == "max" else min(3, left - 1))
            return True
        left, remaining, dice = simulate_blitz(int(self.armies[src]), int(self.armies[dst]), self.rng)
        if remaining:
            self.armies[src], self.armies[dst] = left, remaining
            return False
        self.conquer(q, src, dst, left, left - 1 if occupation == "max" else min(dice, left - 1))
        return True

    def conquer(self, q: int, src: int, dst: int, left: int, moved: int):
        self.owner[dst] = q
        self.armies[dst] = moved
        self.armies[src] = left - moved
        self.conquered[dst] = True
        if self.mission[q] is not None and self.mission[q][0] == dst:
            self.pending[q] += self.mission[q][1]
            self.mission[q] = None

    def finish_turn(self, q: int, agent, phase: str = "attack", record: bool = False):
        """Plays the rest of a turn from `phase` ("attack" or "fortify")."""
        if phase == "attack":
            for _ in range(getattr(getattr(agent, "cfg", None), "max_attacks", 60)):
                choice = agent.choose_attack(self.state(q))
                if choice is None:
                    break
                src, dst, occupation = choice
                if self.owner[src] != q or self.owner[dst] == q or self.armies[src] < 2 or dst not in self.topo.nbrs[src]:
                    break
                self.stats["attacks"] += 1
                if self.attack(q, src, dst, occupation) and len(self.alive()) == 1:
                    break
        if len(self.alive()) > 1:
            move = agent.choose_fortify(self.state(q))
            if move is not None:
                src, dst, nb = move
                nb = min(int(nb), int(self.armies[src]) - 1)
                if nb > 0 and self.owner[src] == q and self.owner[dst] == q and src != dst:
                    self.armies[src] -= nb
                    self.armies[dst] += nb
        self.stats["turns"] += 1
        if record:
            feats, _ = extract(self.topo, self.owner[None], self.armies[None], q, self.n_players,
                               min(1.0, self.turn / self.turn_max), self.pending[q], self.income_div, self.revolt)
            self.samples.append((q, feats[0]))

    def _revolts(self):
        active, threshold, chance = self.revolt
        if not active:
            return
        for q in self.alive():
            mine = self.owner == q
            if mine.sum() / self.topo.n >= threshold:
                weak = np.flatnonzero(mine & self.conquered & (self.armies <= 1))
                for c in weak[self.rng.random(len(weak)) < chance]:
                    self.owner[c] = NEUTRAL
                    self.armies[c] = 2              # observed: "redevient neutre avec 2 armées"
                    self.conquered[c] = False

    def play(self, agents, record: bool = True) -> int:
        """Runs the game to its end. Returns the winner's index, or -1 for a draw."""
        while True:
            for q in self.order:
                if not (self.owner == q).any():
                    continue
                self.play_turn(q, agents[q], record)
                alive = self.alive()
                if len(alive) == 1:
                    return alive[0]
            self._revolts()
            alive = self.alive()
            if len(alive) == 1:
                return alive[0]
            self.turn += 1
            if self.turn > self.turn_max:
                rank = sorted(alive, key=lambda q: (int((self.owner == q).sum()),
                                                    int(self.armies[self.owner == q].sum())), reverse=True)
                a, b = rank[0], rank[1]
                key = lambda q: (int((self.owner == q).sum()), int(self.armies[self.owner == q].sum()))
                return -1 if key(a) == key(b) else a


def new_game(rng, n_players: int | None = None, size: tuple | None = None) -> Game:
    topo, owner, armies, n_players = random_setup(rng, n_players, size)
    turn_max = int(rng.choice([150, 100, 60, 40]))
    return Game(topo, owner, armies, n_players, rng, turn_max=turn_max)
