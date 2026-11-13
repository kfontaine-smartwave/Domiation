"""Decision making: a shallow expectimax search whose leaves are scored by the value network.

For every decision the agent builds the positions that could result from each option (using the
exact battle odds for the win / fail branches, and following short chains of conquests), sends
all of them to the evaluator in one batch (that batch is what runs on the NPU) and keeps the
option with the best expected win probability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .battle import P_WIN, MAX_ARMIES, expected_blitz
from .board import State
from .evaluators import logit, sigmoid
from .features import extract


@dataclass
class SearchConfig:
    depth: int = 2            # extra conquests explored after a successful attack (chains)
    beam: int = 3             # attacks explored per chain step
    min_prob: float = 0.25    # attacks less likely than this are not considered
    max_place: int = 18       # reinforcement spots explored
    margin: float = 0.02      # minimum gain, in log-odds, to attack instead of stopping
    temperature: float = 0.0  # > 0: sample among good options, in log-odds (self-play exploration)
    max_attacks: int = 60     # safety cap per turn


LIGHT = SearchConfig(depth=1, beam=2, max_place=10)


class _Tree:
    """Collects positions to evaluate; values are resolved after one batched evaluation."""

    def __init__(self, state: State):
        self.s = state
        self.owners: list[np.ndarray] = []
        self.armies: list[np.ndarray] = []
        self.pending: list[float] = []

    def leaf(self, owner, armies, pending):
        self.owners.append(owner)
        self.armies.append(armies)
        self.pending.append(pending)
        return ("leaf", len(self.owners) - 1)

    def evaluate(self, evaluator) -> np.ndarray:
        s = self.s
        owner = np.stack(self.owners)
        armies = np.stack(self.armies)
        pending = np.asarray(self.pending, dtype=np.float32)
        feats, terminal = extract(s.topo, owner, armies, s.me, s.n_players, s.turn_frac, pending,
                                  s.income_div, s.revolt)
        values = sigmoid(evaluator.logits(feats))          # float64 win probabilities
        values = np.where(np.isnan(terminal), values, terminal.astype(np.float64))
        if s.objective is not None:
            values = _objective_shaping(s, owner, values)
        return values


def _objective_shaping(s: State, owner: np.ndarray, values: np.ndarray) -> np.ndarray:
    """"Objectifs" mode: holding enough whole regions and territories until my next turn wins.
    The network is trained for domination, so positions that satisfy (or let an opponent satisfy)
    the objective are nudged strongly."""
    need_regions, need_cells = s.objective
    topo = s.topo
    out = values.copy()
    for q in range(s.n_players):
        mask = (owner == q).astype(np.float32)
        cells = mask.sum(1)
        regions = (mask @ topo.region_member.T >= topo.region_size - 0.5).sum(1) if topo.n_regions else 0
        met = (cells >= need_cells) & (regions >= need_regions)
        progress = np.minimum(cells / max(need_cells, 1), 1.0) * 0.5 + \
            np.minimum(regions / max(need_regions, 1), 1.0) * 0.5
        if q == s.me:
            out = np.where(met, 0.6 + 0.4 * out, out + 0.05 * progress)
        else:
            out = np.where(met, 0.15 * out, out - 0.05 * progress)
    return np.clip(out, 0.0, 1.0)


def _resolve(expr, values):
    kind = expr[0]
    if kind == "leaf":
        return float(values[expr[1]])
    if kind == "max":
        return max(_resolve(e, values) for e in expr[1])
    _, p, win, lose = expr
    return p * _resolve(win, values) + (1.0 - p) * _resolve(lose, values)


class SearchAgent:
    def __init__(self, evaluator, config: SearchConfig | None = None, rng=None):
        self.evaluator = evaluator
        self.cfg = config or SearchConfig()
        self.rng = rng or np.random.default_rng()

    # ------------------------------------------------------------------ tree construction
    def _targets(self, s: State, owner, armies, src: int, limit: int):
        """Non-owned neighbours of `src` worth attacking, most likely conquests first."""
        a = int(armies[src])
        if a < 2:
            return []
        out = []
        for t in s.topo.nbrs[src]:
            if owner[t] != s.me:
                p = float(P_WIN[min(a, MAX_ARMIES), min(int(armies[t]), MAX_ARMIES)])
                if p >= self.cfg.min_prob:
                    out.append((p, t))
        out.sort(reverse=True)
        return [t for _, t in out[:limit]]

    def _attack(self, tree: _Tree, owner, armies, pending, src: int, dst: int, depth: int):
        """Chance node for a blitz src -> dst. Returns (probability, win_max, win_min, lose) expressions."""
        s = tree.s
        a, d = int(armies[src]), int(armies[dst])
        p, exp_att, exp_def = expected_blitz(a, d)
        left = max(2, int(round(exp_att)))
        gain = s.mission[1] if (s.mission is not None and s.mission[0] == dst) else 0.0
        o_win = owner.copy()
        o_win[dst] = s.me
        a_max = armies.copy()
        a_max[src], a_max[dst] = 1, left - 1
        win_max = self._decision(tree, o_win, a_max, pending + gain, dst, depth)
        moved = min(3, left - 1)
        if moved < left - 1:                       # leaving the bulk behind is a different position
            a_min = armies.copy()
            a_min[src], a_min[dst] = left - moved, moved
            win_min = tree.leaf(o_win, a_min, pending + gain)
        else:
            win_min = None
        a_lose = armies.copy()
        a_lose[src], a_lose[dst] = 1, max(1, int(round(exp_def)))
        lose = tree.leaf(owner, a_lose, pending)
        return p, win_max, win_min, lose

    def _decision(self, tree: _Tree, owner, armies, pending, head: int, depth: int):
        """Stop here, or push on from the stack at `head`."""
        options = [tree.leaf(owner, armies, pending)]
        if depth > 0:
            for t in self._targets(tree.s, owner, armies, head, self.cfg.beam):
                p, win_max, win_min, lose = self._attack(tree, owner, armies, pending, head, t, depth - 1)
                win = win_max if win_min is None else ("max", [win_max, win_min])
                options.append(("mix", p, win, lose))
        return options[0] if len(options) == 1 else ("max", options)

    def _pick(self, scores: list[float]) -> int:
        if self.cfg.temperature > 0 and len(scores) > 1:
            z = logit(scores) / self.cfg.temperature
            z -= z.max()
            w = np.exp(z)
            return int(self.rng.choice(len(scores), p=w / w.sum()))
        return int(np.argmax(scores))

    # ------------------------------------------------------------------ reinforcements
    def place_options(self, s: State, n: int) -> list[tuple[list, float]]:
        """Reinforcement plans [(cell, armies), ...] with their estimated value, best first."""
        topo, me = s.topo, s.me
        mine = s.cells_of(me)
        if n <= 0 or not len(mine):
            return []
        border = [int(c) for c in mine if any(s.owner[t] != me for t in topo.nbrs[c])]
        if not border:
            return [([(int(mine[np.argmax(s.armies[mine])]), n)], 0.5)]
        if len(border) > self.cfg.max_place:
            border = self._shortlist(s, border, n)
        tree = _Tree(s)
        exprs = []
        for c in border:
            armies = s.armies.copy()
            armies[c] += n
            exprs.append(self._decision(tree, s.owner, armies, s.pending, c, self.cfg.depth + 1))
        values = tree.evaluate(self.evaluator)
        scores = [_resolve(e, values) for e in exprs]
        options = [([(c, n)], q) for c, q in zip(border, scores)]
        best = int(np.argmax(scores))
        if n >= 4 and len(border) > 1:              # try sharing with the runner-up spots
            order = [i for i in np.argsort(scores)[::-1] if i != best][:4]
            tree = _Tree(s)
            splits = []
            for i in order:
                for k in sorted({n // 2, n // 3} - {0}):
                    armies = s.armies.copy()
                    armies[border[best]] += n - k
                    armies[border[i]] += k
                    splits.append(([(border[best], n - k), (border[i], k)],
                                   self._decision(tree, s.owner, armies, s.pending, border[best], self.cfg.depth + 1)))
            if splits:
                values = tree.evaluate(self.evaluator)
                options += [(plan, _resolve(e, values) - 1e-4) for plan, e in splits]
        options.sort(key=lambda o: -o[1])
        return options

    def place(self, s: State, n: int) -> list[tuple[int, int]]:
        options = self.place_options(s, n)
        if not options:
            return []
        return options[self._pick([q for _, q in options])][0]

    def _shortlist(self, s: State, border: list[int], n: int) -> list[int]:
        """Cheap ranking of reinforcement spots when the front is too long to search fully."""
        scored = []
        for c in border:
            a = min(int(s.armies[c]) + n, MAX_ARMIES)
            best = threat = 0.0
            for t in s.topo.nbrs[c]:
                if s.owner[t] == s.me:
                    continue
                d = min(int(s.armies[t]), MAX_ARMIES)
                enemy = s.owner[t] >= 0
                best = max(best, float(P_WIN[a, d]) * (1.5 if enemy else 1.0))
                if enemy:
                    threat = max(threat, float(P_WIN[d, min(int(s.armies[c]), MAX_ARMIES)]))
            scored.append((best + threat, c))
        scored.sort(reverse=True)
        return [c for _, c in scored[:self.cfg.max_place]]

    # ------------------------------------------------------------------ attacks
    def attack_options(self, s: State):
        """(value of stopping, [(source, target, occupation, value, win probability)] best first)."""
        me = s.me
        tree = _Tree(s)
        stop = tree.leaf(s.owner, s.armies, s.pending)
        options = []
        for src in s.cells_of(me):
            src = int(src)
            if s.armies[src] < 2:
                continue
            for dst in self._targets(s, s.owner, s.armies, src, 99):
                options.append((src, dst, self._attack(tree, s.owner, s.armies, s.pending, src, dst, self.cfg.depth)))
        values = tree.evaluate(self.evaluator)
        ranked = []
        for src, dst, (p, win_max, win_min, lose) in options:
            v_max = _resolve(win_max, values)
            v_min = _resolve(win_min, values) if win_min is not None else -1.0
            ranked.append((src, dst, "max" if v_max >= v_min else "min",
                           p * max(v_max, v_min) + (1.0 - p) * _resolve(lose, values), p))
        ranked.sort(key=lambda o: -o[3])
        return _resolve(stop, values), ranked

    def choose_attack(self, s: State):
        """Returns (source, target, "min"|"max") or None to end the attack phase."""
        stop_value, options = self.attack_options(s)
        # Compared in log-odds: when far ahead (or behind) all probabilities are nearly equal.
        threshold = float(logit(stop_value)) + self.cfg.margin
        good = [o for o in options if float(logit(o[3])) > threshold]
        if not good:
            return None
        return good[self._pick([o[3] for o in good])][:3]

    # ------------------------------------------------------------------ fortification
    def fortify_options(self, s: State):
        """(value of not moving, [(source, destination, armies, value)] best first)."""
        topo, me = s.topo, s.me
        mine = [int(c) for c in s.cells_of(me)]
        movable = [c for c in mine if s.armies[c] >= 2]
        tree = _Tree(s)
        stay = tree.leaf(s.owner, s.armies, s.pending)
        moves = []
        if movable and len(mine) >= 2:
            comp = _components(s, mine)
            enemy_near = {c: max((int(s.armies[t]) for t in topo.nbrs[c] if s.owner[t] >= 0 and s.owner[t] != me),
                                 default=0) for c in mine}
            is_border = {c: any(s.owner[t] != me for t in topo.nbrs[c]) for c in mine}
            sources = sorted(movable, key=lambda c: (is_border[c], -int(s.armies[c])))[:6]
            borders = [c for c in mine if is_border[c]]
            for src in sources:
                dests = sorted((c for c in borders if c != src and comp[c] == comp[src]),
                               key=lambda c: (-enemy_near[c], -int(s.armies[c])))[:6]
                amounts = {int(s.armies[src]) - 1}
                if is_border[src] and s.armies[src] >= 4:
                    amounts.add((int(s.armies[src]) - 1) // 2)
                for dst in dests:
                    for nb in amounts:
                        armies = s.armies.copy()
                        armies[src] -= nb
                        armies[dst] += nb
                        moves.append((src, dst, nb, tree.leaf(s.owner, armies, s.pending)))
        values = tree.evaluate(self.evaluator)
        ranked = sorted(((src, dst, nb, _resolve(e, values)) for src, dst, nb, e in moves), key=lambda m: -m[3])
        return _resolve(stay, values), ranked

    def choose_fortify(self, s: State):
        """Returns (source, destination, armies) or None. One move, between connected own cells."""
        stay_value, moves = self.fortify_options(s)
        if not moves:
            return None
        move = moves[self._pick([m[3] for m in moves])]
        if float(logit(move[3])) <= float(logit(stay_value)) + 0.005:
            return None
        return move[:3]


def _components(s: State, mine: list[int]) -> dict[int, int]:
    """Connected groups of my territories (fortification needs a path through my own cells)."""
    comp: dict[int, int] = {}
    for start in mine:
        if start in comp:
            continue
        comp[start] = start
        stack = [start]
        while stack:
            for t in s.topo.nbrs[stack.pop()]:
                if s.owner[t] == s.me and t not in comp:
                    comp[t] = start
                    stack.append(t)
    return comp


class ScriptedBot:
    """Rule-of-thumb player resembling the game's built-in AI: spreads reinforcements over its
    best fronts, attacks when the odds are good, brings rear armies forward. Used as a sparring
    partner in self-play and as a benchmark."""

    def __init__(self, rng=None, min_prob: float = 0.55, spread: int = 3):
        self.rng = rng or np.random.default_rng()
        self.min_prob = min_prob
        self.spread = spread

    def place(self, s: State, n: int):
        me = s.me
        scored = []
        for c in s.cells_of(me):
            c = int(c)
            targets = [int(s.armies[t]) for t in s.topo.nbrs[c] if s.owner[t] != me]
            if targets:
                scored.append((int(s.armies[c]) - min(targets) + self.rng.random(), c))
        if not scored:
            mine = s.cells_of(me)
            return [(int(mine[0]), n)] if len(mine) else []
        scored.sort(reverse=True)
        spots = [c for _, c in scored[:self.spread]]
        counts = {c: 0 for c in spots}
        for i in range(n):
            counts[spots[i % len(spots)]] += 1
        return [(c, k) for c, k in counts.items() if k]

    def choose_attack(self, s: State):
        me = s.me
        best = None
        for src in s.cells_of(me):
            src = int(src)
            a = int(s.armies[src])
            if a < 2:
                continue
            for t in s.topo.nbrs[src]:
                if s.owner[t] != me:
                    p = float(P_WIN[min(a, MAX_ARMIES), min(int(s.armies[t]), MAX_ARMIES)])
                    if p >= self.min_prob and (best is None or p > best[0]):
                        best = (p, src, t)
        return None if best is None else (best[1], best[2], "max")

    def choose_fortify(self, s: State):
        me = s.me
        mine = [int(c) for c in s.cells_of(me)]
        rear = [c for c in mine if s.armies[c] >= 2 and all(s.owner[t] == me for t in s.topo.nbrs[c])]
        if not rear:
            return None
        src = max(rear, key=lambda c: int(s.armies[c]))
        comp = _components(s, mine)
        fronts = [c for c in mine if comp[c] == comp[src] and any(s.owner[t] != me for t in s.topo.nbrs[c])]
        if not fronts:
            return None
        dst = max(fronts, key=lambda c: sum(int(s.armies[t]) for t in s.topo.nbrs[c] if s.owner[t] != me))
        return src, dst, int(s.armies[src]) - 1
