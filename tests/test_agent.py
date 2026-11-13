"""Offline tests: `python -m unittest discover -s tests` (no network, no NPU needed)."""
import unittest

import numpy as np

from domination import battle
from domination.board import NEUTRAL, state_from_view
from domination.client import ApiError
from domination.evaluators import LinearEvaluator
from domination.features import N_FEATURES, extract
from domination.play import Player
from domination.policy import LIGHT, ScriptedBot, SearchAgent
from domination.simulator import Game, new_game, random_setup


class FakeServer:
    """Speaks the agent API on top of the local simulator: seat j1 is the agent under test,
    seat j2 a scripted bot that plays as soon as the agent's turn ends."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        topo, owner, armies, _ = random_setup(rng, n_players=2, size=(8, 6))
        self.game = Game(topo, owner, armies, 2, rng, turn_max=25)
        self.bot = ScriptedBot(rng)
        self.phase, self.left, self.finished, self.winner = "renfort", 0, False, None
        self.orders = []
        self._begin()

    def _begin(self):
        self.phase = "renfort"
        self.left = self.game.state(0).income(0)

    def view(self):
        g, t = self.game, self.game.topo
        owner = lambda c: "neutre" if g.owner[c] == NEUTRAL else f"j{g.owner[c] + 1}"
        players = {}
        for q in range(2):
            mask = g.owner == q
            players[f"j{q + 1}"] = {"nom": f"P{q + 1}", "type": "distant" if q == 0 else "heuristique",
                                    "stats": {"territoires": int(mask.sum()), "armees": int(g.armies[mask].sum())},
                                    "renforts_prevus": g.state(q).income(q), "bonus_regions": 0}
            if t.n_regions and mask.any():
                held = t.region_member @ mask.astype(np.float32) >= t.region_size - 0.5
                players[f"j{q + 1}"]["bonus_regions"] = int(t.region_bonus[held].sum())
        return {"vous": "j1", "mon_tour": not self.finished, "delai_restant": 120, "partie": {
            "tour": g.turn, "tour_max": g.turn_max, "mode_victoire": "domination", "economie": "classique",
            "objectifs": None, "courant": "j1", "phase": self.phase, "renforts_restants": self.left,
            "termine": self.finished, "vainqueur": self.winner, "raison": "", "mission": None,
            "revoltes": {"actives": True, "seuil_pct": 50, "chance_pct": 10}, "joueurs": players,
            "plateau": {"territoires": [{"id": t.ids[c], "proprietaire": owner(c), "armees": int(g.armies[c]),
                                         "adjacents": [t.ids[v] for v in t.nbrs[c]],
                                         "region": f"R{t.region_of[c]}" if t.region_of[c] >= 0 else None}
                                        for c in range(t.n)],
                        "regions": [{"id": f"R{r}", "bonus": b, "territoires": [t.ids[c] for c in cells]}
                                    for r, (b, cells) in enumerate(t.regions)]}}}

    def wait(self, timeout=25):
        return self.view()

    state = wait

    def _end_turn(self):
        g = self.game
        if len(g.alive()) > 1:
            g.play_turn(1, self.bot, record=False)
            g.turn += 1
        alive = g.alive()
        if len(alive) == 1 or g.turn > g.turn_max:
            self.finished = True
            self.winner = f"j{alive[0] + 1}" if len(alive) == 1 else None
        else:
            self._begin()

    def act(self, action, **f):
        g, idx = self.game, self.game.topo.index
        self.orders.append(action)
        if self.finished:
            raise ApiError(400, "game over")
        if action == "renfort":
            c = idx[f["territoire"]]
            if self.phase != "renfort" or g.owner[c] != 0 or not 0 < f["nb"] <= self.left:
                raise ApiError(400, "bad reinforcement")
            g.armies[c] += f["nb"]
            self.left -= f["nb"]
        elif action == "attaque":
            s, d = idx[f["source"]], idx[f["cible"]]
            if self.phase != "attaque" or g.owner[s] != 0 or g.owner[d] == 0 or g.armies[s] < 2 or d not in g.topo.nbrs[s]:
                raise ApiError(400, "bad attack")
            left, rest, dice = battle.simulate_blitz(int(g.armies[s]), int(g.armies[d]), g.rng)
            if rest == 0:
                moved = left - 1 if f.get("occupation") == "max" else min(dice, left - 1)
                g.owner[d], g.armies[d], g.armies[s] = 0, moved, left - moved
                if len(g.alive()) == 1:
                    self.finished, self.winner = True, "j1"
            else:
                g.armies[s], g.armies[d] = left, rest
        elif action == "fortification":
            s, d = idx[f["source"]], idx[f["cible"]]
            if self.phase != "fortification" or g.owner[s] != 0 or g.owner[d] != 0 or not 0 < f["nb"] < g.armies[s]:
                raise ApiError(400, "bad fortification")
            g.armies[s] -= f["nb"]
            g.armies[d] += f["nb"]
            self._end_turn()
        elif action == "passer":
            if self.phase == "renfort":
                if self.left:
                    raise ApiError(400, "place every reinforcement first")
                self.phase = "attaque"
            elif self.phase == "attaque":
                self.phase = "fortification"
            else:
                self._end_turn()
        else:
            raise ApiError(400, "unknown action")
        return self.view()


class BattleTests(unittest.TestCase):
    def test_known_odds(self):
        self.assertAlmostEqual(battle.win_probability(2, 1), 15 / 36, places=5)
        self.assertAlmostEqual(battle.win_probability(3, 2), 0.3627, places=3)
        self.assertEqual(battle.win_probability(1, 1), 0.0)
        self.assertGreater(battle.win_probability(10, 3), 0.95)

    def test_simulation_matches_table(self):
        rng = np.random.default_rng(0)
        wins = sum(battle.simulate_blitz(6, 4, rng)[1] == 0 for _ in range(4000))
        self.assertAlmostEqual(wins / 4000, battle.win_probability(6, 4), delta=0.03)


class SimulatorTests(unittest.TestCase):
    def test_games_finish_and_features_are_finite(self):
        rng = np.random.default_rng(3)
        for players in (2, 3, 4):
            game = new_game(rng, n_players=players, size=(8, 6))
            game.turn_max = 20
            agents = [SearchAgent(LinearEvaluator(), LIGHT, rng)] + [ScriptedBot(rng) for _ in range(players - 1)]
            winner = game.play(agents)
            self.assertIn(winner, range(-1, players))
            feats = np.array([f for _, f in game.samples])
            self.assertEqual(feats.shape[1], N_FEATURES)
            self.assertTrue(np.isfinite(feats).all())

    def test_terminal_values(self):
        rng = np.random.default_rng(1)
        topo, owner, armies, _ = random_setup(rng, n_players=2, size=(8, 6))
        owner[:] = 0
        _, terminal = extract(topo, owner[None], armies[None], 0, 2)
        self.assertEqual(terminal[0], 1.0)
        _, terminal = extract(topo, owner[None], armies[None], 1, 2)
        self.assertEqual(terminal[0], 0.0)


class LiveLoopTests(unittest.TestCase):
    def test_full_game_through_the_api(self):
        server = FakeServer(seed=5)
        player = Player(server, SearchAgent(LinearEvaluator(), LIGHT), log=lambda *_: None)
        result = player.run()
        self.assertIn(result, ("WON", "lost", "draw"))
        self.assertGreater(player.turns, 0)
        self.assertIn("attaque", server.orders)

    def test_view_parsing(self):
        server = FakeServer(seed=2)
        s = state_from_view(server.view())
        self.assertEqual(s.me, 0)
        self.assertEqual(s.income_div, 3)
        self.assertEqual(s.income(0), server.left)
        self.assertEqual(int((s.owner == 0).sum()), server.view()["partie"]["joueurs"]["j1"]["stats"]["territoires"])


if __name__ == "__main__":
    unittest.main()
