"""Play a DOM-IA-TION game with the NPU-backed agent.

    python -m domination.play --jeton <seat token>

The token comes from the game lobby (seat set to "Mon script Python" / "Script externe"); it can
also be given through the IA_JETON environment variable so that it stays out of the shell history.
The agent waits for its turn, plays it order by order (it re-reads the board after every battle)
and stops by itself when the game is over or when the token is revoked.

Every game is recorded in traces/ (positions and final result); `python -m domination.train`
adds those real positions to its training set and `python -m domination.history` lists them.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error

import numpy as np

from .board import state_from_view
from .client import DEFAULT_URL, ApiError, Client
from .evaluators import load_best_evaluator, sigmoid
from .features import extract
from .history import TRACE_DIR
from .policy import SearchAgent, SearchConfig
from .rollout import RolloutAgent, RolloutConfig

SAFETY_SECONDS = 15        # stop attacking when the turn clock gets this low


class Player:
    def __init__(self, client: Client, agent: SearchAgent, log=print, record_dir=None, think: float = 45.0):
        self.client, self.agent, self.log = client, agent, log
        self.record_dir, self.think = record_dir, think
        self.trace = None
        self.topo = None
        self.turns = 0
        self.deployed = False

    def _record(self, view: dict, order=None):
        """Appends a server reply to the recording of this game (the journal is left out, except
        its last lines at the end of the game: positions and the result are what training needs)."""
        if self.record_dir is None:
            return
        try:
            game = view["partie"]
            if self.trace is None:
                os.makedirs(self.record_dir, exist_ok=True)
                name = f"{game.get('id') or 'game'}_{view.get('vous', 'x')}.jsonl".replace("/", "-").replace(":", "-")
                self.trace = open(os.path.join(self.record_dir, name), "a", encoding="utf-8")
            keep = 30 if game.get("termine") else 1
            slim = dict(view, partie=dict(game, journal=(game.get("journal") or [])[-keep:]))
            self.trace.write(json.dumps({"t": time.time(), "order": order, "view": slim}, ensure_ascii=False) + "\n")
            self.trace.flush()
        except (OSError, KeyError, TypeError) as exc:
            self.log(f"  recording stopped ({exc})")
            self.record_dir = None

    # ------------------------------------------------------------------ helpers
    def _state(self, view):
        s = state_from_view(view, self.topo)
        self.topo = s.topo
        return s

    def _act(self, action: str, **fields) -> dict:
        view = self.client.act(action, **fields)
        self._record(view, {"action": action, **fields})
        return view

    def _value(self, s) -> float:
        feats, terminal = extract(s.topo, s.owner[None], s.armies[None], s.me, s.n_players,
                                  s.turn_frac, s.pending, s.income_div, s.revolt)
        return float(sigmoid(self.agent.evaluator.logits(feats)[0])) if np.isnan(terminal[0]) else float(terminal[0])

    # ------------------------------------------------------------------ one turn
    def play_turn(self, view: dict) -> dict:
        """Plays from whatever phase the turn is in (so an interrupted turn can be resumed)."""
        ids = None
        started = time.time()
        budget = float(view.get("delai_restant") or 120)
        notes = {"placed": [], "attacks": 0, "conquests": 0, "fortify": None}
        if hasattr(self.agent, "begin_turn"):          # thinking time for the simulations
            self.agent.begin_turn(min(self.think, budget - SAFETY_SECONDS - 15))
        me = view["vous"]
        before = view["partie"]["joueurs"][me]["stats"]["territoires"]

        def mine(v):
            return v["partie"]["joueurs"][me]["stats"]["territoires"]

        def running(v):
            g = v["partie"]
            return v.get("mon_tour") and not g["termine"] and g["courant"] == me

        if running(view) and view["partie"]["phase"] == "renfort":
            s = self._state(view)
            ids = s.topo.ids
            remaining = int(view["partie"]["renforts_restants"])
            for cell, nb in self.agent.place(s, remaining):
                nb = min(int(nb), remaining)
                if nb > 0:
                    view = self._act("renfort", territoire=ids[cell], nb=nb)
                    notes["placed"].append(f"+{nb} {ids[cell]}")
                    remaining = int(view["partie"]["renforts_restants"])
            if remaining > 0:                          # every reinforcement must be placed before moving on
                s = self._state(view)
                cells = s.cells_of(s.me)
                cell = int(cells[np.argmax(s.armies[cells])])
                view = self._act("renfort", territoire=ids[cell], nb=remaining)
                notes["placed"].append(f"+{remaining} {ids[cell]}")
            view = self._act("passer")

        if running(view) and view["partie"]["phase"] == "attaque":
            for _ in range(self.agent.cfg.max_attacks):
                if budget - (time.time() - started) < SAFETY_SECONDS:
                    self.log("  clock running low: ending the attack phase")
                    break
                s = self._state(view)
                ids = s.topo.ids
                choice = self.agent.choose_attack(s)
                if choice is None:
                    break
                src, dst, occupation = choice
                held = mine(view)
                try:
                    view = self._act("attaque", source=ids[src], cible=ids[dst], blitz=True, occupation=occupation)
                except ApiError as exc:
                    if exc.fatal or exc.code != 400:
                        raise
                    self.log(f"  order refused ({exc.message}): ending the attack phase")
                    view = self.client.state()
                    break
                notes["attacks"] += 1
                notes["conquests"] += mine(view) > held
                if not running(view):
                    break
            if running(view) and view["partie"]["phase"] == "attaque":
                view = self._act("passer")

        if running(view) and view["partie"]["phase"] == "fortification":
            s = self._state(view)
            ids = s.topo.ids
            move = self.agent.choose_fortify(s)
            done = False
            if move is not None:
                src, dst, nb = move
                try:
                    view = self._act("fortification", source=ids[src], cible=ids[dst], nb=int(nb))
                    notes["fortify"] = f"{ids[src]}->{ids[dst]} ({nb})"
                    done = True
                except ApiError as exc:
                    if exc.fatal or exc.code != 400:
                        raise
                    self.log(f"  fortification refused ({exc.message})")
            if not done:
                view = self._act("passer")

        self.turns += 1
        s = self._state(view)
        notes.update(view=view, before=before, after=mine(view), value=self._value(s),
                     seconds=time.time() - started)
        return notes

    def _deploy(self, view: dict) -> None:
        """Secret initial placement of the "Conquête planétaire" mode: every player spreads a
        reserve over its territories before the first turn. The reserve is placed in three
        waves chosen by the search; whatever is left is spread by the server itself."""
        self.deployed = True
        info = view["partie"].get("deploiement") or {}
        reserve = int(info.get("reserve") or 0)
        if info.get("valide") or reserve <= 0 or info.get("siege") not in (None, view["vous"]):
            return
        try:
            placed = []
            for wave in range(3):
                s = self._state(view)
                nb = reserve if wave == 2 else max(1, reserve // (3 - wave))
                for cell, k in self.agent.place(s, nb):
                    view = self._act("deployer", territoire=s.topo.ids[cell], nb=int(k))
                    placed.append(f"+{k} {s.topo.ids[cell]}")
                    reserve -= int(k)
                if reserve <= 0:
                    break
            self._act("valider_deploiement")
            self.log(f"initial placement: {', '.join(placed)}")
        except ApiError as exc:
            if exc.fatal:
                raise
            self.log(f"initial placement left to the server ({exc.message or exc.code})")

    # ------------------------------------------------------------------ main loop
    def run(self, max_turns: int | None = None) -> str | None:
        failures = 0
        while True:
            try:
                view = self.client.wait()
                failures = 0
                game = view["partie"]
                if game["termine"]:
                    self._record(view)
                    winner = game.get("vainqueur")
                    name = game["joueurs"][winner]["nom"] if winner else "nobody (draw)"
                    result = "WON" if winner == view["vous"] else ("draw" if not winner else "lost")
                    self.log(f"Game over: {result}. Winner: {name}. {game.get('raison') or ''}".rstrip())
                    return result
                if game.get("deploiement") and not self.deployed:
                    self._deploy(view)
                    continue
                if not view.get("mon_tour"):
                    continue
                turn = game["tour"]
                self._record(view)
                notes = self.play_turn(view)
            except ApiError as exc:
                if exc.fatal:
                    self.log(f"Stopping: {exc.message or exc}")
                    return None
                if exc.code == 400:                    # e.g. the turn timed out under our feet
                    self.log(f"  order refused ({exc.message}); re-reading the game")
                    time.sleep(1)
                    continue
                failures += 1
                self._pause(failures, f"server unavailable ({exc})")
                continue
            except (urllib.error.URLError, OSError, ValueError) as exc:
                failures += 1
                self._pause(failures, f"network unavailable ({type(exc).__name__})")
                continue
            ev = self.agent.evaluator
            usage = f" | {ev.device}: {ev.positions} positions, {ev.seconds:.2f} s" if hasattr(ev, "positions") else ""
            if hasattr(ev, "positions"):
                ev.positions, ev.seconds = 0, 0.0
            if hasattr(self.agent, "sims_run"):
                usage += f" | {self.agent.sims_run} simulations"
                self.agent.sims_run = 0
            self.log(f"turn {turn}: {', '.join(notes['placed']) or 'no reinforcement'}; "
                     f"{notes['attacks']} attacks, {notes['conquests']} conquests; "
                     f"fortify {notes['fortify'] or '-'} | territories {notes['before']} -> {notes['after']} | "
                     f"win estimate {notes['value'] * 100:.0f} % | {notes['seconds']:.1f} s{usage}")
            if max_turns and self.turns >= max_turns:
                return None

    def _pause(self, failures: int, reason: str):
        delay = min(60, 5 * 2 ** min(failures - 1, 4))
        self.log(f"{reason}; retrying in {delay} s")
        time.sleep(delay)


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jeton", "--token", dest="token", default=os.environ.get("IA_JETON"),
                    help="seat token (or environment variable IA_JETON)")
    ap.add_argument("--url", default=os.environ.get("IA_URL", DEFAULT_URL))
    ap.add_argument("--device", default="NPU", help="OpenVINO device for the value network (NPU, GPU, CPU)")
    ap.add_argument("--tours", "--turns", dest="turns", type=int, default=None, help="stop after N turns")
    ap.add_argument("--depth", type=int, default=2, help="conquest chains explored after each attack")
    ap.add_argument("--lookahead", action="store_true",
                    help="experimental: also simulate the opponents' replies before deciding "
                         "(measured: no gain with the current network, so it is off by default)")
    ap.add_argument("--sims", type=int, default=0,
                    help="with --lookahead: dice simulations per decision (0 = one noise-free simulation per option)")
    ap.add_argument("--think", type=float, default=45.0, help="with --lookahead: seconds of simulations per turn")
    ap.add_argument("--record-dir", default=str(TRACE_DIR), help="where games are recorded")
    ap.add_argument("--no-record", action="store_true", help="do not record the game")
    a = ap.parse_args(argv)
    if not a.token:
        ap.error("missing token: --jeton <token> or environment variable IA_JETON")
    evaluator = load_best_evaluator(a.device)
    print(f"Evaluator: {evaluator.name}")
    if a.lookahead:
        agent = RolloutAgent(evaluator, SearchConfig(depth=a.depth), rollout=RolloutConfig(sims=a.sims))
        print(f"Search: with simulated look-ahead, up to {a.think:.0f} s per turn")
    else:
        agent = SearchAgent(evaluator, SearchConfig(depth=a.depth))
    player = Player(Client(a.url, a.token), agent, record_dir=None if a.no_record else a.record_dir, think=a.think)
    print("Connected. Waiting for my turn...")
    try:
        player.run(a.turns)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if player.trace:
            player.trace.close()
            print(f"Game recorded in {player.trace.name}")


if __name__ == "__main__":
    main()
