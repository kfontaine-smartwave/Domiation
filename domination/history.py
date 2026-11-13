"""Recorded real games -> training samples.

`domination.play` records every game it plays in `traces/` (one JSON line per server reply).
The game server itself keeps no move history (only the last lines of the journal), so these
recordings are the only source of real-game positions.

    python -m domination.history            # summary of the recorded games

A recording yields one sample per turn observed: the position at the end of each of my turns
(from my point of view) and the position I am handed at the start of each of my turns (from the
point of view of the opponent who has just played), each labelled with the final result.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .board import state_from_view
from .features import N_FEATURES, extract

TRACE_DIR = Path(__file__).resolve().parent.parent / "traces"


def read_trace(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass                        # a line cut short by an interrupted run
    return rows


def game_summary(rows: list[dict]) -> dict | None:
    views = [r["view"] for r in rows if isinstance(r.get("view"), dict) and "partie" in r["view"]]
    if not views:
        return None
    last = views[-1]["partie"]
    me = views[-1]["vous"]
    return {"finished": bool(last.get("termine")), "winner": last.get("vainqueur"), "me": me,
            "won": last.get("vainqueur") == me, "turn": last.get("tour"), "reason": last.get("raison"),
            "players": {k: (v.get("type"), v.get("nom")) for k, v in last["joueurs"].items()},
            "mode": last.get("mode_victoire"), "cells": len(last["plateau"]["territoires"])}


def samples_from_trace(rows: list[dict]):
    """Returns (features [M, F], results [M]) or None if the game was not recorded to its end."""
    info = game_summary(rows)
    if not info or not info["finished"]:
        return None
    xs, ys = [], []
    topo = None
    previous_my_turn = False
    previous_phase = None
    for r in rows:
        view = r.get("view")
        if not isinstance(view, dict) or "partie" not in view:
            continue
        game = view["partie"]
        if game.get("deploiement"):
            continue
        try:
            s = state_from_view(view, topo)
        except (KeyError, ValueError):
            continue
        topo = s.topo
        my_turn = bool(view.get("mon_tour")) and not game.get("termine")
        action = (r.get("order") or {}).get("action")
        ended = action == "fortification" or (action == "passer" and previous_phase == "fortification") \
            or (previous_my_turn and not my_turn and action is not None)
        previous_phase = game.get("phase")
        perspective = None
        if ended:
            perspective = s.me                   # my turn has just ended
        elif my_turn and action is None and len(s.order) > 1:
            k = s.order.index(s.me)              # an opponent has just handed me the board
            for step in range(1, len(s.order)):
                q = s.order[(k - step) % len(s.order)]
                if (s.owner == q).any():
                    perspective = q
                    break
        previous_my_turn = my_turn
        if perspective is None:
            continue
        feats, _ = extract(s.topo, s.owner[None], s.armies[None], perspective, s.n_players, s.turn_frac,
                           s.pending if perspective == s.me else 0.0, s.income_div, s.revolt)
        xs.append(feats[0])
        winner = info["winner"]
        ys.append(0.5 if not winner else float(s.names[perspective] == winner))
    if not xs:
        return None
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def load_samples(directory=TRACE_DIR, log=print):
    """All samples of the finished games recorded in `directory`."""
    xs, ys, games = [np.zeros((0, N_FEATURES), np.float32)], [np.zeros(0, np.float32)], 0
    for path in sorted(Path(directory).glob("*.jsonl")):
        got = samples_from_trace(read_trace(path))
        if got is not None:
            xs.append(got[0])
            ys.append(got[1])
            games += 1
    x, y = np.concatenate(xs), np.concatenate(ys)
    if log:
        log(f"recorded games: {games} finished, {len(x)} positions")
    return x, y


def main():
    paths = sorted(TRACE_DIR.glob("*.jsonl"))
    if not paths:
        print(f"no recorded game in {TRACE_DIR} yet: every game played with domination.play is recorded there")
        return
    won = finished = 0
    for path in paths:
        info = game_summary(read_trace(path))
        if not info:
            continue
        finished += info["finished"]
        won += info["finished"] and info["won"]
        rivals = ", ".join(f"{name} ({kind})" for k, (kind, name) in info["players"].items() if k != info["me"])
        result = ("WON" if info["won"] else "lost" if info["winner"] else "draw") if info["finished"] else "unfinished"
        print(f"{path.name}: {result} at turn {info['turn']} | {info['cells']} territories, {info['mode']} | vs {rivals}")
    print(f"{finished} finished games, {won} won")


if __name__ == "__main__":
    main()
