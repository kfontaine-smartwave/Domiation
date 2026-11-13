"""Head-to-head benchmark in the local simulator.

    python -m domination.bench --a rollout --b search --players 3 --games 60

One player of kind A sits against players of kind B in every other seat (the seat rotates), on
the default 10 x 7 map. Kinds: `rollout` (search + noise-free look-ahead), `rollout:<n>` (search
+ n dice simulations per decision), `search` or `search:<depth>` (quick search only), `formula` (quick search with the hand-written formula), `scripted` (rule-of-thumb bot).
A's fair share of wins is 1 / players.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from .evaluators import DEFAULT_WEIGHTS, LinearEvaluator, MLPEvaluator
from .parallel import worker_pool
from .policy import ScriptedBot, SearchAgent, SearchConfig
from .rollout import RolloutAgent, RolloutConfig
from .simulator import new_game


def build(kind: str, weights, rng):
    name, _, arg = kind.partition(":")
    if name == "scripted":
        return ScriptedBot(rng, min_prob=float(rng.uniform(0.45, 0.75)), spread=int(rng.integers(1, 5)))
    if name == "formula":
        return SearchAgent(LinearEvaluator(), SearchConfig(), rng)
    evaluator = MLPEvaluator(weights)
    if name == "rollout":
        return RolloutAgent(evaluator, SearchConfig(), rng, RolloutConfig(sims=int(arg or 0)))
    if name == "search":
        return SearchAgent(evaluator, SearchConfig(depth=int(arg or 2)), rng)
    raise SystemExit(f"unknown player kind: {kind}")


def _game(args):
    seed, index, kind_a, kind_b, weights_a, weights_b, players, size = args
    rng = np.random.default_rng(seed)
    game = new_game(rng, n_players=players, size=size)
    game.turn_max = 80
    seat = index % players
    agents = [build(kind_a, weights_a, rng) if q == seat else build(kind_b, weights_b, rng) for q in range(players)]
    winner = game.play(agents, record=False)
    return (2 if winner < 0 else 0 if winner == seat else 1), game.turn


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="rollout")
    ap.add_argument("--b", default="search")
    ap.add_argument("--weights-a", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--weights-b", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 6))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", default="10x7")
    a = ap.parse_args(argv)
    size = tuple(int(v) for v in a.size.split("x"))
    jobs = [(a.seed * 1009 + i, i, a.a, a.b, a.weights_a, a.weights_b, a.players, size) for i in range(a.games)]
    tally, rounds, t0 = [0, 0, 0], [], time.time()
    with worker_pool(a.workers) as pool:
        for outcome, length in pool.imap_unordered(_game, jobs):
            tally[outcome] += 1
            rounds.append(length)
            if sum(tally) % max(1, a.games // 6) == 0:
                print(f"  {sum(tally)}/{a.games}: {tally[0]} wins, {tally[1]} losses, {tally[2]} draws", flush=True)
    n = sum(tally)
    rate = tally[0] / n
    err = 1.96 * (rate * (1 - rate) / n) ** 0.5
    print(f"{a.a} vs {a.b}, {a.players} players: {tally[0]} wins / {tally[1]} losses / {tally[2]} draws "
          f"= {rate * 100:.0f} % +/- {err * 100:.0f} (fair share {100 / a.players:.0f} %), "
          f"mean {np.mean(rounds):.0f} rounds, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
