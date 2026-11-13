"""Self-play training of the value network.

    python -m domination.train --generations 3 --games 3000

Generation 0 plays with a hand-written evaluation formula. Every position reached at the end of a
player's turn is stored with the final result of that game (1 = that player won); a small MLP is
fitted on those samples to predict the win probability. The next generation plays with that
network, and so on. The final weights are saved to models/value.npz and exported as an OpenVINO
IR (models/value.xml) that the live agent compiles for the NPU.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

from .evaluators import DEFAULT_WEIGHTS, MODEL_DIR, LinearEvaluator, MLPEvaluator, export_openvino
from .features import N_FEATURES
from .policy import LIGHT, ScriptedBot, SearchAgent, SearchConfig
from .simulator import new_game


def _make_agent(kind: str, weights, rng):
    if kind == "net":
        return SearchAgent(MLPEvaluator(weights), SearchConfig(**{**LIGHT.__dict__, "temperature": 0.05}), rng)
    if kind == "net_greedy":
        return SearchAgent(MLPEvaluator(weights), LIGHT, rng)
    if kind == "formula":
        return SearchAgent(LinearEvaluator(noise=0.25, rng=rng),
                           SearchConfig(**{**LIGHT.__dict__, "temperature": 0.05}), rng)
    if kind == "formula_greedy":
        return SearchAgent(LinearEvaluator(), LIGHT, rng)
    return ScriptedBot(rng, min_prob=float(rng.uniform(0.45, 0.75)), spread=int(rng.integers(1, 5)))


def _selfplay_worker(args):
    seed, n_games, weights = args
    rng = np.random.default_rng(seed)
    learner = "net" if weights else "formula"
    pool = [learner, learner, learner, "formula", "scripted", "scripted"]
    xs, ys, lengths = [], [], []
    for _ in range(n_games):
        game = new_game(rng)
        kinds = [learner] + [pool[int(rng.integers(len(pool)))] for _ in range(game.n_players - 1)]
        rng.shuffle(kinds)
        agents = [_make_agent(k, weights, rng) for k in kinds]
        winner = game.play(agents)
        for player, feats in game.samples:
            xs.append(feats)
            ys.append(0.5 if winner < 0 else float(player == winner))
        lengths.append(game.turn)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32), lengths


def selfplay(n_games: int, weights, workers: int, seed: int):
    per = max(1, n_games // (workers * 6))
    jobs = [(seed * 100003 + i, per, weights) for i in range((n_games + per - 1) // per)]
    parts = []
    with mp.Pool(workers) as pool:
        for part in pool.imap_unordered(_selfplay_worker, jobs):
            parts.append(part)
            if len(parts) % max(1, len(jobs) // 5) == 0:
                print(f"    self-play {len(parts) * 100 // len(jobs)} %", flush=True)
    x = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    lengths = [n for p in parts for n in p[2]]
    return x, y, lengths


def _duel_worker(args):
    seed, n_games, kind_a, kind_b, weights_a, weights_b, size = args
    rng = np.random.default_rng(seed)
    wins = [0, 0, 0]
    for g in range(n_games):
        game = new_game(rng, n_players=2, size=size)
        a, b = _make_agent(kind_a, weights_a, rng), _make_agent(kind_b, weights_b, rng)
        seat_a = g % 2
        winner = game.play([a, b] if seat_a == 0 else [b, a], record=False)
        wins[2 if winner < 0 else (0 if winner == seat_a else 1)] += 1
    return wins


def duel(kind_a, kind_b, weights_a=None, weights_b=None, n_games=200, workers=8, seed=7, size=(10, 7)):
    """Head-to-head on the default 10 x 7 two-player map. Returns (wins_a, wins_b, draws)."""
    per = max(1, n_games // workers)
    jobs = [(seed * 7919 + i, per, kind_a, kind_b, weights_a, weights_b, size) for i in range(workers)]
    with mp.Pool(workers) as pool:
        res = np.sum(pool.map(_duel_worker, jobs), axis=0)
    return int(res[0]), int(res[1]), int(res[2])


def fit(x: np.ndarray, y: np.ndarray, hidden=(96, 96), epochs: int = 14, lr: float = 2e-3,
        batch: int = 2048, seed: int = 0, log=print) -> dict:
    """Small MLP trained with Adam on binary cross-entropy (plain numpy)."""
    rng = np.random.default_rng(seed)
    n_val = max(1, len(x) // 10)
    perm = rng.permutation(len(x))
    val, tr = perm[:n_val], perm[n_val:]
    mean = x[tr].mean(0)
    std = x[tr].std(0) + 1e-3
    xn = ((x - mean) / std).astype(np.float32)
    sizes = [N_FEATURES, *hidden, 1]
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
        z = forward(xn[idx])[-1][:, 0]
        p = 1.0 / (1.0 + np.exp(-z))
        t = y[idx]
        return float(-(t * np.log(p + 1e-7) + (1 - t) * np.log(1 - p + 1e-7)).mean())

    for epoch in range(epochs):
        rng.shuffle(tr)
        rate = lr * (0.5 ** (epoch // 5))
        for start in range(0, len(tr), batch):
            idx = tr[start:start + batch]
            acts = forward(xn[idx])
            p = 1.0 / (1.0 + np.exp(-acts[-1][:, 0]))
            grad = ((p - y[idx]) / len(idx))[:, None]
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
        if epoch % 3 == 2 or epoch == epochs - 1:
            log(f"    epoch {epoch + 1:2d}  train loss {loss_of(tr[:200000]):.4f}  validation loss {loss_of(val):.4f}")
    out = {"mean": mean.astype(np.float32), "std": std.astype(np.float32)}
    for i in range(0, len(params), 2):
        out[f"W{i // 2}"] = params[i]
        out[f"b{i // 2}"] = params[i + 1]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--games", type=int, default=3000, help="self-play games per generation")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 4))
    ap.add_argument("--duel-games", type=int, default=200, help="benchmark games after each generation")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="start from the existing models/value.npz")
    a = ap.parse_args(argv)
    MODEL_DIR.mkdir(exist_ok=True)
    weights = str(DEFAULT_WEIGHTS) if (a.resume and DEFAULT_WEIGHTS.exists()) else None
    data_x, data_y = [], []
    best = (-1, None)                               # self-play is not monotonic: keep the best generation
    for gen in range(a.generations):
        t0 = time.time()
        x, y, lengths = selfplay(a.games, weights, a.workers, a.seed + gen)
        data_x.append(x)
        data_y.append(y)
        print(f"generation {gen}: {a.games} games by {'the network' if weights else 'the hand-written formula'}"
              f" in {time.time() - t0:.0f} s -> {len(x)} positions, mean length {np.mean(lengths):.0f} rounds,"
              f" {np.mean(np.array(lengths) > 150) * 100:.0f} % reached turn 150")
        keep = slice(-2, None)                      # train on the two most recent generations
        model = fit(np.concatenate(data_x[keep]), np.concatenate(data_y[keep]), seed=a.seed + gen)
        candidate = MODEL_DIR / f"value_gen{gen + 1}.npz"
        np.savez(candidate, **model)
        w1, w2, d = duel("net_greedy", "scripted", str(candidate), None, a.duel_games, a.workers)
        print(f"  network gen {gen + 1} vs scripted bot      : {w1} wins / {w2} losses / {d} draws")
        score = w1
        w1, w2, d = duel("net_greedy", "formula_greedy", str(candidate), None, a.duel_games, a.workers)
        print(f"  network gen {gen + 1} vs hand-written formula: {w1} wins / {w2} losses / {d} draws")
        best = max(best, (score + w1, str(candidate)))
        if weights:
            w1, w2, d = duel("net_greedy", "net_greedy", str(candidate), weights, a.duel_games, a.workers)
            print(f"  network gen {gen + 1} vs previous network   : {w1} wins / {w2} losses / {d} draws")
        weights = str(candidate)
    np.savez(DEFAULT_WEIGHTS, **dict(np.load(best[1])))
    xml = export_openvino(DEFAULT_WEIGHTS)
    print(f"kept {Path(best[1]).name} (best benchmark); saved {DEFAULT_WEIGHTS} and {xml}")


if __name__ == "__main__":
    main()
