"""Self-play training of the value network.

    python -m domination.train --generations 2 --games 8000 --resume

Every position reached at the end of a player's turn in a simulated game is stored with the final
result of that game (1 = that player won); an MLP is fitted on those samples to predict the win
probability. Each generation plays with the newest network against a league (earlier networks,
a hand-written formula, a scripted bot), in games with 2, 3 or 4 players. Positions from real
games recorded in traces/ are added to the training set with a higher weight. Every generation
is benchmarked against the first network (2 and 3 players) and a scripted bot, and the best one
(the starting network included) is kept: models/value.npz, exported as an OpenVINO IR
(models/value.xml) that the live agent compiles for the NPU.
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from .evaluators import DEFAULT_WEIGHTS, MODEL_DIR, LinearEvaluator, MLPEvaluator, export_openvino
from .features import N_FEATURES
from .history import TRACE_DIR, load_samples
from .parallel import worker_pool
from .policy import LIGHT, ScriptedBot, SearchAgent, SearchConfig
from .simulator import new_game

BASELINE = MODEL_DIR / "value_v1.npz"       # the first network, kept as a fixed yardstick
EXPLORE = SearchConfig(**{**LIGHT.__dict__, "temperature": 0.05})


def make_agent(spec, rng):
    """spec: ("net", weights path, explore) | ("formula", None, explore) | ("scripted", None, _)."""
    kind, weights, explore = spec
    if kind == "net":
        return SearchAgent(MLPEvaluator(weights), EXPLORE if explore else LIGHT, rng)
    if kind == "formula":
        return SearchAgent(LinearEvaluator(noise=0.25 if explore else 0.0, rng=rng), EXPLORE if explore else LIGHT, rng)
    return ScriptedBot(rng, min_prob=float(rng.uniform(0.45, 0.75)), spread=int(rng.integers(1, 5)))


def _selfplay_worker(args):
    seed, n_games, learner, league = args
    rng = np.random.default_rng(seed)
    xs, ys, lengths = [], [], []
    for _ in range(n_games):
        game = new_game(rng)
        specs = [learner]
        for _ in range(game.n_players - 1):
            specs.append(learner if rng.random() < 0.5 else league[int(rng.integers(len(league)))])
        rng.shuffle(specs)
        winner = game.play([make_agent(spec, rng) for spec in specs])
        for player, feats in game.samples:
            xs.append(feats)
            ys.append(0.5 if winner < 0 else float(player == winner))
        lengths.append(game.turn)
    return np.asarray(xs, np.float32).reshape(-1, N_FEATURES), np.asarray(ys, np.float32), lengths


def selfplay(n_games: int, learner, league, workers: int, seed: int):
    per = max(1, n_games // (workers * 6))
    jobs = [(seed * 100003 + i, per, learner, league) for i in range((n_games + per - 1) // per)]
    parts = []
    with worker_pool(workers) as pool:
        for part in pool.imap_unordered(_selfplay_worker, jobs):
            parts.append(part)
            if len(parts) % max(1, len(jobs) // 4) == 0:
                print(f"    self-play {len(parts) * 100 // len(jobs)} %", flush=True)
    return (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]),
            [n for p in parts for n in p[2]])


def _duel_worker(args):
    seed, n_games, spec_a, spec_b, n_players, size = args
    rng = np.random.default_rng(seed)
    wins = [0, 0, 0]
    for g in range(n_games):
        game = new_game(rng, n_players=n_players, size=size)
        seat = g % n_players
        agents = [make_agent(spec_a if q == seat else spec_b, rng) for q in range(n_players)]
        winner = game.play(agents, record=False)
        wins[2 if winner < 0 else (0 if winner == seat else 1)] += 1
    return wins


def duel(spec_a, spec_b, n_games=200, workers=8, seed=7, n_players=2, size=(10, 7)):
    """One `spec_a` player against `spec_b` in every other seat, on the default map.
    Returns (wins of A, wins of the others, draws); A's fair share is 1 / n_players."""
    per = max(1, n_games // workers)
    jobs = [(seed * 7919 + i, per, spec_a, spec_b, n_players, size) for i in range(workers)]
    with worker_pool(workers) as pool:
        res = np.sum(pool.map(_duel_worker, jobs), axis=0)
    return int(res[0]), int(res[1]), int(res[2])


def fit(x: np.ndarray, y: np.ndarray, hidden=(64, 64), epochs: int = 16, lr: float = 2e-3,
        batch: int = 2048, seed: int = 0, weight=None, log=print) -> dict:
    """MLP trained with Adam on binary cross-entropy (plain numpy)."""
    rng = np.random.default_rng(seed)
    w = np.ones(len(x), np.float32) if weight is None else weight.astype(np.float32)
    n_val = max(1, len(x) // 12)
    perm = rng.permutation(len(x))
    val, tr = perm[:n_val], perm[n_val:]
    mean = x[tr].mean(0)
    std = x[tr].std(0) + 1e-3
    xn = ((x - mean) / std).astype(np.float32)
    sizes = [x.shape[1], *hidden, 1]
    params = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        params += [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32), np.zeros(b, np.float32)]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    step = 0

    def forward(inp):
        acts = [inp]
        h = inp
        for i in range(0, len(params), 2):
            h = h @ params[i] + params[i + 1]
            if i < len(params) - 2:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return acts

    def loss_of(idx):
        total = 0.0
        for start in range(0, len(idx), 65536):
            part = idx[start:start + 65536]
            p = 1.0 / (1.0 + np.exp(-forward(xn[part])[-1][:, 0]))
            t = y[part]
            total += float(-(t * np.log(p + 1e-7) + (1 - t) * np.log(1 - p + 1e-7)).sum())
        return total / len(idx)

    for epoch in range(epochs):
        rng.shuffle(tr)
        rate = lr * (0.5 ** (epoch // 5))
        for start in range(0, len(tr), batch):
            idx = tr[start:start + batch]
            acts = forward(xn[idx])
            p = 1.0 / (1.0 + np.exp(-acts[-1][:, 0]))
            grad = ((p - y[idx]) * w[idx] / w[idx].sum())[:, None]
            grads = [None] * len(params)
            for i in range(len(params) - 2, -1, -2):
                layer_in = acts[i // 2]
                grads[i] = layer_in.T @ grad + 1e-5 * params[i]
                grads[i + 1] = grad.sum(0)
                if i > 0:
                    grad = (grad @ params[i].T) * (layer_in > 0)
            step += 1
            for k, g in enumerate(grads):
                m[k] = 0.9 * m[k] + 0.1 * g
                v[k] = 0.999 * v[k] + 0.001 * g * g
                params[k] -= rate * (m[k] / (1 - 0.9 ** step)) / (np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
        if epoch % 5 == 4 or epoch == epochs - 1:
            log(f"    epoch {epoch + 1:2d}  train loss {loss_of(tr[:200000]):.4f}  validation loss {loss_of(val):.4f}")
    out = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    for i in range(0, len(params), 2):
        out[f"W{i // 2}"] = params[i]
        out[f"b{i // 2}"] = params[i + 1]
    return out


def benchmark(candidate: str, n_games: int, workers: int, log=print) -> float:
    """Head-to-head results of a network on the default map; returns a score to rank generations."""
    cand = ("net", candidate, False)
    w, l, d = duel(cand, ("scripted", None, False), n_games, workers, seed=11)
    log(f"  2 players vs scripted bot  : {w} wins / {l} losses / {d} draws")
    score = w / n_games
    rival = ("net", str(BASELINE), False) if BASELINE.exists() else ("formula", None, False)
    label = "first network" if BASELINE.exists() else "hand-written formula"
    w, l, d = duel(cand, rival, n_games, workers, seed=12)
    log(f"  2 players vs {label:14s}: {w} wins / {l} losses / {d} draws")
    score += w / n_games
    w, l, d = duel(cand, rival, n_games, workers, seed=13, n_players=3)
    log(f"  3 players vs two x {label}: {w} wins / {l} losses / {d} draws (fair share: {n_games // 3})")
    return score + 1.5 * w / n_games


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=4)
    ap.add_argument("--games", type=int, default=4000, help="self-play games per generation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 6))
    ap.add_argument("--duel-games", type=int, default=240, help="benchmark games per match-up")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="first generation plays with models/value.npz")
    ap.add_argument("--start", help="first generation plays with these weights instead")
    ap.add_argument("--hidden", default="64,64",
                    help="hidden layer sizes (measured: larger networks overfit and play worse)")
    ap.add_argument("--extra-data", help=".npz file with earlier self-play positions (arrays x, y) to train on as well")
    ap.add_argument("--real", default=str(TRACE_DIR), help="directory of recorded real games")
    ap.add_argument("--real-weight", type=float, default=20.0, help="weight of a real position vs a simulated one")
    a = ap.parse_args(argv)
    MODEL_DIR.mkdir(exist_ok=True)
    hidden = tuple(int(v) for v in a.hidden.split(","))
    real_x, real_y = load_samples(a.real) if Path(a.real).is_dir() else (np.zeros((0, N_FEATURES), np.float32), np.zeros(0, np.float32))
    league = [("formula", None, True), ("scripted", None, False), ("scripted", None, False)]
    if BASELINE.exists():
        league.append(("net", str(BASELINE), True))
    start = MODEL_DIR / "value_start.npz"
    weights = None
    source = Path(a.start) if a.start else DEFAULT_WEIGHTS if a.resume else None
    if source is not None and source.exists():
        np.savez(start, **dict(np.load(source)))     # value.npz itself is rewritten at the end
        weights = str(start)
    data_x, data_y = [], []
    if a.extra_data:
        extra = np.load(a.extra_data)
        data_x.append(extra["x"][:, :N_FEATURES].astype(np.float32))
        data_y.append(extra["y"].astype(np.float32))
        print(f"earlier self-play data: {len(data_y[0])} positions")
    best = (-1.0, None)                              # self-play is not monotonic: keep the best generation
    if weights:                                      # the starting network competes too: never regress
        print("starting network:")
        best = (benchmark(weights, a.duel_games, a.workers), weights)
        print(f"  score {best[0]:.3f}", flush=True)
    for gen in range(a.generations):
        t0 = time.time()
        learner = ("net", weights, True) if weights else ("formula", None, True)
        x, y, lengths = selfplay(a.games, learner, league, a.workers, a.seed + gen)
        data_x.append(x)
        data_y.append(y)
        print(f"generation {gen + 1}: {a.games} games in {time.time() - t0:.0f} s -> {len(x)} positions, "
              f"mean length {np.mean(lengths):.0f} rounds", flush=True)
        xs = np.concatenate(data_x[-4:] + [real_x])
        ys = np.concatenate(data_y[-4:] + [real_y])
        wts = np.concatenate([np.ones(len(xs) - len(real_x), np.float32), np.full(len(real_x), a.real_weight, np.float32)])
        model = fit(xs, ys, hidden, seed=a.seed + gen, weight=wts)
        candidate = MODEL_DIR / f"value_gen{gen + 1}.npz"
        np.savez(candidate, **model)
        score = benchmark(str(candidate), a.duel_games, a.workers)
        print(f"  score {score:.3f}", flush=True)
        best = max(best, (score, str(candidate)))
        weights = str(candidate)
        league.append(("net", str(candidate), True))
    np.savez(DEFAULT_WEIGHTS, **dict(np.load(best[1])))
    xml = export_openvino(DEFAULT_WEIGHTS)
    print(f"kept {Path(best[1]).name} (best benchmark); saved {DEFAULT_WEIGHTS} and {xml}")


if __name__ == "__main__":
    main()
