"""Turns a batch of positions into the fixed-size feature vectors fed to the value network.

Features are global, relative quantities (shares, ratios, risk estimates), so the same network
works on any map size, on non-grid maps and with 2 to 4 players. Everything is computed from the
point of view of player `me`, for a position at the end of one of `me`'s own actions.
"""
from __future__ import annotations

import numpy as np

from .battle import MAX_ARMIES, P_WIN
from .board import Topology

FEATURE_NAMES = [
    "terr_me", "terr_rival", "terr_others", "terr_neutral",
    "army_share_me", "army_share_rival", "army_share_neutral",
    "income_share_me", "income_share_rival", "army_logratio", "income_logratio",
    "opponents_alive", "two_players",
    "region_potential_me", "region_potential_rival", "bonus_held_me", "bonus_held_rival",
    "bonus_at_risk_me", "bonus_at_risk_rival",
    "threat_cells", "threat_share_of_mine", "opportunity_enemy", "opportunity_neutral",
    "front_army_frac", "idle_army_frac", "biggest_stack_frac", "enemy_contact_frac",
    "weak_border_frac", "revolt_exposure",
    "turn_frac", "endgame_lead", "pending_bonus", "army_per_cell_me", "army_per_cell_rival",
]
N_FEATURES = len(FEATURE_NAMES)


def extract(topo: Topology, owner: np.ndarray, armies: np.ndarray, me: int, n_players: int,
            turn_frac: float = 0.0, pending=0.0, income_div: int = 3,
            revolt: tuple = (True, 0.5, 0.1)):
    """owner, armies: [B, n]. Returns (features [B, N_FEATURES] float32, terminal [B] float32)
    where terminal is 1.0 if `me` is the only player left, 0.0 if `me` is eliminated, NaN otherwise."""
    B, n = owner.shape
    X = armies.astype(np.float32)
    rows = np.arange(B)
    masks = np.stack([owner == q for q in range(n_players)], 0).astype(np.float32)   # [P, B, n]
    neutral = (owner < 0).astype(np.float32)
    T = masks.sum(2)                                   # [P, B] territories
    A = (masks * X).sum(2)                             # [P, B] armies
    if topo.n_regions:
        cnt = masks @ topo.region_member.T             # [P, B, R]
        full = cnt >= topo.region_size - 0.5
        held = (full * topo.region_bonus).sum(2)       # [P, B]
        potential = ((cnt / topo.region_size) ** 2 * topo.region_bonus).sum(2) / max(topo.total_bonus, 1e-6)
    else:
        full = np.zeros((n_players, B, 0), dtype=bool)
        held = np.zeros_like(T)
        potential = np.zeros_like(T)
    income = np.where(T > 0, np.maximum(3.0, np.floor(T / income_div)) + held, 0.0)
    pending = np.broadcast_to(np.asarray(pending, dtype=np.float32), (B,))
    income[me] = income[me] + np.where(T[me] > 0, pending, 0.0)

    strength = A + 2.0 * income
    strength[me] = -1.0
    rival = strength.argmax(0)                         # [B] strongest opponent
    T_me, A_me, inc_me = T[me], A[me], income[me]
    T_rv, A_rv, inc_rv = T[rival, rows], A[rival, rows], income[rival, rows]
    T_all = T.sum(0)
    A_all = A.sum(0)
    A_neu = (neutral * X).sum(1)
    inc_all = income.sum(0)
    alive_opp = (T > 0).sum(0) - (T_me > 0)

    # Neighbourhood: strongest adjacent stack that is hostile to each cell's owner, and my
    # strongest stack next to each cell. Index n is a dummy cell with no army and no owner.
    Xp = np.concatenate([X, np.zeros((B, 1), np.float32)], 1)
    Op = np.concatenate([owner, np.full((B, 1), -2, owner.dtype)], 1)
    nX = Xp[:, topo.nbr_pad]                           # [B, n, deg]
    nO = Op[:, topo.nbr_pad]
    mine = masks[me]
    enemy = (owner >= 0) & (owner != me)
    hostile_to_me = (nO >= 0) & (nO != me)
    max_enemy = (nX * hostile_to_me).max(2)            # strongest enemy stack touching the cell
    max_mine = (nX * (nO == me)).max(2)                # my strongest stack touching the cell
    foreign = ((nO != me) & (nO != -2)).any(2)         # cell touches something that is not mine

    d_idx = np.clip(armies, 0, MAX_ARMIES).astype(np.int64)
    # The opponent reinforces before attacking, and so do I before my next attacks.
    a_enemy = np.clip(max_enemy + inc_rv[:, None], 0, MAX_ARMIES).astype(np.int64)
    a_mine = np.clip(max_mine + inc_me[:, None], 0, MAX_ARMIES).astype(np.int64)
    p_lose = P_WIN[a_enemy, d_idx] * (max_enemy > 0) * mine           # my cells: chance to be taken
    p_take = P_WIN[a_mine, d_idx] * (max_mine > 0)                    # other cells: chance I take them
    threat = p_lose.sum(1)
    opp_enemy = (p_take * enemy).sum(1)
    opp_neutral = (p_take * neutral).sum(1)

    risk_me = np.zeros(B, np.float32)
    risk_rv = np.zeros(B, np.float32)
    if topo.n_regions:
        full_rv = full[rival, rows]                    # [B, R]
        for r, (bonus, cells) in enumerate(topo.regions):
            fm = full[me][:, r]
            if fm.any():
                risk_me += bonus * fm * p_lose[:, cells].max(1)
            fr = full_rv[:, r]
            if fr.any():
                risk_rv += bonus * fr * p_take[:, cells].max(1)
        risk_me /= max(topo.total_bonus, 1e-6)
        risk_rv /= max(topo.total_bonus, 1e-6)

    contact = mine * (max_enemy > 0)
    front_army = (X * contact).sum(1)
    interior = mine * (~foreign)
    idle = (np.maximum(X - 1.0, 0.0) * interior).sum(1)
    biggest = (X * mine).max(1)
    border = mine * foreign
    weak_border = (contact * (X <= 1.0)).sum(1)
    active, threshold, chance = revolt
    exposed = (mine * (X <= 1.0)).sum(1) / n * (T_me / n >= threshold) * (10.0 * chance if active else 0.0)

    eps = 1e-6
    safe_T = np.maximum(T_me, 1.0)
    safe_A = np.maximum(A_me, 1.0)
    f = np.stack([
        T_me / n, T_rv / n, (T_all - T_me) / n, neutral.sum(1) / n,
        A_me / (A_all + 1.0), A_rv / (A_all + 1.0), A_neu / (A_all + A_neu + 1.0),
        inc_me / (inc_all + eps), inc_rv / (inc_all + eps),
        np.clip(np.log((A_me + 1.0) / (A_rv + 1.0)), -3, 3) / 3.0,
        np.clip(np.log((inc_me + 1.0) / (inc_rv + 1.0)), -2, 2) / 2.0,
        alive_opp / 3.0, (alive_opp == 1).astype(np.float32),
        potential[me], potential[rival, rows],
        held[me] / max(topo.total_bonus, 1.0), held[rival, rows] / max(topo.total_bonus, 1.0),
        risk_me, risk_rv,
        threat / n, threat / safe_T, opp_enemy / n, opp_neutral / n,
        front_army / safe_A, idle / safe_A, biggest / safe_A, contact.sum(1) / np.maximum(border.sum(1), 1.0),
        weak_border / safe_T, exposed,
        np.full(B, turn_frac, np.float32), (T_me - T_rv) / n * turn_frac ** 3,
        pending / 10.0, np.clip(A_me / safe_T, 0, 10) / 10.0,
        np.clip(A_rv / np.maximum(T_rv, 1.0), 0, 10) / 10.0,
    ], 1).astype(np.float32)

    terminal = np.full(B, np.nan, np.float32)
    terminal[(T_me > 0) & (alive_opp == 0)] = 1.0
    terminal[T_me == 0] = 0.0
    return f, terminal
