"""Look-ahead by simulation: spends the spare thinking time (and NPU throughput) on each decision.

The quick search (`SearchAgent`) rates an option with a single network evaluation of the position
it leads to. `RolloutAgent` keeps the few best options of that search and, for each of them,
plays the future out many times in the local simulator: the rest of my turn, every opponent's
reply, then my following turn, all with real dice and with the quick search choosing the moves of
every player. The network scores the position at the end of each simulation and the option with
the best average wins. This is what catches moves that look good now but leave a region or a
thin line of single armies for the opponent to take back.

Every simulated decision is itself a batch of network evaluations, so a turn uses tens of
thousands of positions instead of a few hundred.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .battle import expected_blitz
from .board import State
from .evaluators import logit, sigmoid
from .features import extract
from .policy import LIGHT, SearchAgent, SearchConfig
from .simulator import Game


@dataclass
class RolloutConfig:
    sims: int = 0             # 0: one noise-free simulation per option (battles take their expected
                              # outcome); > 0: that many dice simulations shared between the options
    candidates: int = 3       # options of the quick search that get simulated
    clear: float = 2.0        # log-odds lead at which the quick search is trusted without simulating
    own_turn: bool = True     # simulate through my next turn (else stop after the opponents' replies)
    share: float = 0.25       # largest share of the remaining thinking time one decision may use


class RolloutAgent(SearchAgent):
    def __init__(self, evaluator, config: SearchConfig | None = None, rng=None,
                 rollout: RolloutConfig | None = None, base_config: SearchConfig | None = None):
        super().__init__(evaluator, config, rng)
        self.ro = rollout or RolloutConfig()
        self.base = SearchAgent(evaluator, base_config or LIGHT, self.rng)   # plays inside simulations
        self.deadline = None        # wall-clock limit for the current turn (live games only)
        self.sim_seconds = 0.05     # running average cost of one simulation
        self.sims_run = 0
        self.decisions = [0, 0]     # [decided by simulation, simulation overruled the quick search]

    # ------------------------------------------------------------------ time control
    def begin_turn(self, seconds: float | None):
        self.deadline = None if seconds is None else time.time() + max(0.0, seconds)

    def _budget(self, options: int) -> int:
        wanted = self.ro.sims if self.ro.sims > 0 else 2 * options
        if self.deadline is None:
            return wanted
        left = self.deadline - time.time()
        return int(min(wanted, left * self.ro.share / max(self.sim_seconds, 1e-3)))

    # ------------------------------------------------------------------ simulation
    def _simulate(self, s: State, prepare, phase: str) -> float:
        """One simulated future of `s` after `prepare(game)`; returns my win estimate at its end."""
        t0 = time.perf_counter()
        me = s.me
        g = Game.from_state(s, self.rng)
        g.expected = self.ro.sims <= 0
        prepare(g)
        if phase != "done" and len(g.alive()) > 1:
            g.finish_turn(me, self.base, phase)
        order = g.order
        k = order.index(me)
        for step in range(1, len(order) + 1):
            if len(g.alive()) < 2 or not (g.owner == me).any():
                break
            pos = (k + step) % len(order)
            if pos == 0:                              # a full round has been played
                g._revolts()
                g.turn += 1
            q = order[pos]
            if q == me and not self.ro.own_turn:
                break
            if (g.owner == q).any():
                g.play_turn(q, self.base, record=False)
        feats, terminal = extract(g.topo, g.owner[None], g.armies[None], me, g.n_players,
                                  min(1.0, g.turn / g.turn_max), g.pending[me], g.income_div, g.revolt)
        value = float(sigmoid(self.evaluator.logits(feats)[0])) if np.isnan(terminal[0]) else float(terminal[0])
        self.sims_run += 1
        self.sim_seconds += 0.1 * (time.perf_counter() - t0 - self.sim_seconds)
        return value

    def _compare(self, s: State, candidates: list, budget: int) -> int:
        """candidates: [[(probability, prepare, phase), ...] per option]. Returns the best index.
        Two rounds: every option gets a first batch, the two leaders share the rest."""
        sums = [[0.0] * len(c) for c in candidates]
        counts = [[0] * len(c) for c in candidates]

        def run(i: int, n: int):
            strata = candidates[i]
            for j, (prob, prepare, phase) in enumerate(strata):
                for _ in range(max(1, int(round(n * prob))) if prob >= 0.04 or len(strata) == 1 else 0):
                    sums[i][j] += self._simulate(s, prepare, phase)
                    counts[i][j] += 1

        def mean(i: int) -> float:
            total = weight = 0.0
            for j, (prob, _, _) in enumerate(candidates[i]):
                if counts[i][j]:
                    total += prob * sums[i][j] / counts[i][j]
                    weight += prob
            return total / weight if weight else 0.0

        if self.ro.sims <= 0:                         # noise-free: one pass per branch is exact
            for i in range(len(candidates)):
                run(i, 1)
            return max(range(len(candidates)), key=mean)
        first = max(2, budget // (2 * len(candidates)))
        for i in range(len(candidates)):
            run(i, first)
        leaders = sorted(range(len(candidates)), key=mean, reverse=True)[:2]
        rest = max(0, budget - first * len(candidates)) // len(leaders)
        if rest >= 2:
            for i in leaders:
                run(i, rest)
        return max(leaders, key=mean)

    def _decide(self, s: State, candidates: list, quick_choice: int) -> int:
        budget = self._budget(len(candidates))
        if len(candidates) < 2 or budget < 2 * len(candidates):
            return quick_choice
        best = self._compare(s, candidates, budget)
        self.decisions[0] += 1
        self.decisions[1] += best != quick_choice
        return best

    # ------------------------------------------------------------------ decisions
    def place(self, s: State, n: int):
        options = self.place_options(s, n)
        if not options:
            return []
        top = options[:self.ro.candidates]

        def placing(plan):
            def prepare(g):
                for cell, nb in plan:
                    g.armies[cell] += nb
            return prepare

        best = self._decide(s, [[(1.0, placing(plan), "attack")] for plan, _ in top], 0)
        return top[best][0]

    def choose_attack(self, s: State):
        stop_value, options = self.attack_options(s)
        if not options:
            return None
        stop_logit = float(logit(stop_value))
        lead = float(logit(options[0][3])) - stop_logit
        if lead > self.ro.clear:                       # obviously worth it
            return options[0][:3]
        if lead < -self.ro.clear:                      # nothing comes close to stopping
            return None
        near = [o for o in options if float(logit(o[3])) > stop_logit - self.ro.clear][:self.ro.candidates - 1]
        candidates = [[(1.0, lambda g: None, "fortify")]]          # option 0: stop attacking
        me = s.me
        for src, dst, occupation, _, _ in near:
            p, exp_att, exp_def = expected_blitz(int(s.armies[src]), int(s.armies[dst]))
            left = max(2, int(round(exp_att)))
            moved = left - 1 if occupation == "max" else min(3, left - 1)
            remaining = max(1, int(round(exp_def)))

            def won(g, src=src, dst=dst, left=left, moved=moved):
                g.conquer(me, src, dst, left, moved)

            def failed(g, src=src, dst=dst, remaining=remaining):
                g.armies[src], g.armies[dst] = 1, remaining

            candidates.append([(p, won, "attack"), (1.0 - p, failed, "attack")])
        quick = 1 if lead > self.cfg.margin else 0
        best = self._decide(s, candidates, quick)
        return None if best == 0 else near[best - 1][:3]

    def choose_fortify(self, s: State):
        stay_value, moves = self.fortify_options(s)
        if not moves:
            return None
        lead = float(logit(moves[0][3])) - float(logit(stay_value))
        if lead < -self.ro.clear:
            return None
        top = moves[:self.ro.candidates - 1]

        def moving(src, dst, nb):
            def prepare(g):
                g.armies[src] -= nb
                g.armies[dst] += nb
            return prepare

        candidates = [[(1.0, lambda g: None, "done")]] + [[(1.0, moving(src, dst, nb), "done")] for src, dst, nb, _ in top]
        best = self._decide(s, candidates, 1 if lead > 0.005 else 0)
        return None if best == 0 else top[best - 1][:3]
