"""Exact combat odds for the game's dice rules (classic Risk dice).

Observed on the server: the attacker rolls min(3, armies - 1) dice, the defender min(2, armies);
dice are compared highest to highest, ties go to the defender. A "blitz" repeats rounds until the
target is conquered or the attacker is down to one army.

Tables (indexed [attacker_armies, defender_armies], both clamped to MAX_ARMIES):
  P_WIN    probability that a blitz conquers the target
  EXP_ATT  expected attacker armies left on the source *given* a conquest (before moving in)
  EXP_DEF  expected defender armies left *given* a failed blitz
"""
from __future__ import annotations

import itertools

import numpy as np

MAX_ARMIES = 160


def _round_outcomes(na: int, nd: int) -> list[tuple[int, int, float]]:
    """[(attacker_losses, defender_losses, probability)] for one round with na vs nd dice."""
    counts: dict[tuple[int, int], int] = {}
    for roll in itertools.product(range(1, 7), repeat=na + nd):
        att = sorted(roll[:na], reverse=True)
        dfn = sorted(roll[na:], reverse=True)
        la = ld = 0
        for a, d in zip(att, dfn):
            if a > d:
                ld += 1
            else:
                la += 1
        counts[(la, ld)] = counts.get((la, ld), 0) + 1
    total = 6 ** (na + nd)
    return [(la, ld, c / total) for (la, ld), c in sorted(counts.items())]


ROUND = {(na, nd): _round_outcomes(na, nd) for na in (1, 2, 3) for nd in (1, 2)}


def _build_tables(n: int):
    p = np.zeros((n + 1, n + 1), dtype=np.float64)
    ea = np.zeros_like(p)   # E[attacker armies left * 1{win}]
    ed = np.zeros_like(p)   # E[defender armies left * 1{fail}]
    p[:, 0] = 1.0
    ea[:, 0] = np.arange(n + 1)
    ed[1, 1:] = np.arange(1, n + 1)
    ed[0, 1:] = np.arange(1, n + 1)
    for a in range(2, n + 1):
        na = min(3, a - 1)
        for d in range(1, n + 1):
            sp = sa = sd = 0.0
            for la, ld, pr in ROUND[(na, min(2, d))]:
                sp += pr * p[a - la, d - ld]
                sa += pr * ea[a - la, d - ld]
                sd += pr * ed[a - la, d - ld]
            p[a, d], ea[a, d], ed[a, d] = sp, sa, sd
    with np.errstate(divide="ignore", invalid="ignore"):
        exp_att = np.where(p > 1e-12, ea / np.maximum(p, 1e-12), 0.0)
        exp_def = np.where(p < 1 - 1e-12, ed / np.maximum(1 - p, 1e-12), 0.0)
    return p.astype(np.float32), exp_att.astype(np.float32), exp_def.astype(np.float32)


P_WIN, EXP_ATT, EXP_DEF = _build_tables(MAX_ARMIES)


def _clamp(a: int, d: int) -> tuple[int, int]:
    """Scale an oversized match-up down so that it fits the tables (keeps the ratio)."""
    m = max(a, d)
    if m <= MAX_ARMIES:
        return max(a, 0), max(d, 0)
    k = MAX_ARMIES / m
    return max(2, int(a * k)), max(1, int(d * k))


def win_probability(attacker: int, defender: int) -> float:
    if attacker < 2:
        return 0.0
    a, d = _clamp(attacker, defender)
    return float(P_WIN[a, d])


def expected_blitz(attacker: int, defender: int) -> tuple[float, float, float]:
    """(p_win, attacker armies left if it wins, defender armies left if it fails)."""
    if attacker < 2:
        return 0.0, float(attacker), float(defender)
    a, d = _clamp(attacker, defender)
    scale = attacker / a if a else 1.0
    return float(P_WIN[a, d]), float(EXP_ATT[a, d]) * scale, float(EXP_DEF[a, d]) * (defender / d if d else 1.0)


_CUM = {k: (np.cumsum([o[2] for o in v]), [(o[0], o[1]) for o in v]) for k, v in ROUND.items()}


def simulate_blitz(attacker: int, defender: int, rng) -> tuple[int, int, int]:
    """Play a blitz with real randomness. Returns (attacker_left, defender_left, dice_of_last_roll)."""
    last = 0
    while attacker >= 2 and defender >= 1:
        last = min(3, attacker - 1)
        cum, outs = _CUM[(last, min(2, defender))]
        la, ld = outs[int(np.searchsorted(cum, rng.random(), side="right").clip(0, len(outs) - 1))]
        attacker -= la
        defender -= ld
    return attacker, defender, last
