# DOM-IA-TION agent (runs on the local NPU)

An autonomous player for [DOM-IA-TION](https://ia-domination.yogui.ovh/), a Risk-like conquest
game. The agent connects to the game's public agent API with a seat token, waits for its turn and
plays it by itself. Its decisions come from a small search whose positions are scored by a neural
network that runs on the laptop's NPU (Intel AI Boost) through OpenVINO.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 1. On the website: create (or join) a game, open your seat and choose "Mon script Python".
#    Copy the private token that appears (it is shown only once).
# 2. Start the agent with that token:
$env:IA_JETON = "<token>"
.\.venv\Scripts\python -m domination.play
```

The first contact of the agent with the server marks the seat as ready; the game starts as soon
as every seat is ready. You can watch the game in the browser while the agent plays. The agent
stops by itself at the end of the game.

Options of `domination.play`:

| Option | Meaning |
| --- | --- |
| `--jeton <token>` | seat token (otherwise read from the `IA_JETON` environment variable) |
| `--device NPU` | OpenVINO device for the network: `NPU` (default), `GPU` or `CPU` |
| `--depth 2` | how many chained conquests the search looks ahead after an attack |
| `--tours N` | stop after N turns |
| `--no-record` | do not record the game (by default every game is saved in `traces/`) |
| `--lookahead` | experimental: also simulate the opponents' replies before deciding (see Results) |
| `--url` | another server (default `https://ia-domination.yogui.ovh`, or `IA_URL`) |

`python -m domination.check` verifies that the network runs on the NPU and prints its speed.

## The game, as the agent sees it

Facts below were read from the site's rules and observed on a real game:

- A turn has three phases: place reinforcements, attack, then one optional fortification (move
  armies between two of your territories connected through your own land), which ends the turn.
- Reinforcements = territories ÷ 3 (minimum 3) + the bonus of every region you own entirely.
- Battles use classic dice: the attacker rolls up to 3 dice (armies − 1), the defender up to 2,
  ties go to the defender. A "blitz" repeats rounds until conquest or until the attacker has one
  army left. After a conquest you move in either the minimum (the dice of the last roll) or the
  maximum (all but one).
- Default map: 10 × 7 oval grid, each player starts on one side with 3 armies per territory,
  neutral armies (2 to 5) fill the middle.
- Extras: conquest quests (take a designated territory for +3 or +6 armies next turn), revolts
  (when a player holds half of the map, conquered territories left with 1 army may turn neutral),
  a turn limit (then most territories wins, then most armies), an "Objectifs" victory mode and a
  "Conquête planétaire" map with a secret initial placement.
- A remote agent has 120 s per turn; three timeouts in a row hand the seat to the built-in AI.

## How the agent decides

1. **Exact battle odds** ([domination/battle.py](domination/battle.py)). The probability that a
   blitz succeeds, and the armies expected to survive, are computed exactly for every match-up.
2. **Search** ([domination/policy.py](domination/policy.py)). For each decision (where to
   reinforce, which attack to launch or whether to stop, which fortification to make) the agent
   builds the positions each option leads to: the "conquest" and "failure" branches of every
   battle weighted by their odds, following short chains of conquests. It plays order by order
   and re-reads the real board after every battle.
3. **Value network on the NPU** ([domination/evaluators.py](domination/evaluators.py)). All those
   candidate positions (hundreds to a few thousand per decision) are described by
   34 map-independent features ([domination/features.py](domination/features.py)) and scored in
   batches by a small neural network that estimates the chance of winning the game. The network
   is compiled by OpenVINO for the NPU; if no NPU is available it falls back to GPU, then CPU, and
   says so at start-up.
4. **Self-play training** ([domination/train.py](domination/train.py)). The network is trained
   on this machine from games played in a local simulator of the rules
   ([domination/simulator.py](domination/simulator.py)), with 2, 3 and 4 players: each position
   reached at the end of a turn is labelled with the final result of its game. Each generation
   plays against a league (earlier networks, a hand-written formula, a scripted bot), is
   benchmarked, and the best one is kept.
5. **Real games** ([domination/history.py](domination/history.py)). Every game the agent plays is
   recorded in `traces/`; the trainer adds those real positions to its training set with a higher
   weight than simulated ones.

## Results

What was measured on 13 Nov 2026 (simulated games on the default 10 x 7 map unless stated):

- **Second network vs first network.** Trained on 829,000 positions from 8,000 self-play games
  (60 % of them with 3 or 4 players) instead of 6,000 mostly two-player games: 62.5 % of wins
  head-to-head in two-player games (200 of 320) and 55 % of wins as one player against two copies
  of the first network (198 of 360, where an equal player gets 33 %). Against the scripted bot it
  stays at 89 % (286 of 320).
- **One more generation did not help.** A further 8,000 self-play games played by the second
  network (1.9 million positions in total) gave a network that scored 60 % and 47 % in the same
  two tests, below the second network's 62.5 % and 56.5 %, so the trainer kept the second one.
  This approach has reached a plateau; real recorded games are the missing ingredient.
- **A much larger network is worse.** Same data, hidden layers 192-192-96 instead of 64-64: 49 %
  against the first network in two-player games and 30 % in three-player games; with 16 extra
  input features it drops to 41 % and 29 %. The larger networks fit the training games better but
  play worse (they hoard armies), so the network stays small.
- **More computation per move does not help.** With the same network, searching one conquest
  deeper scores 51 % (two players) and 35 % (three players): no difference. Simulating the
  opponents' replies before each decision (`--lookahead`) scores 42 % and 28 %, and with dice in
  the simulations 27 % (40 games): slightly worse, the simulations add more noise than insight. Both stay
  available but are off.
- **Live games** against the site's built-in "IA locale equilibre" (two players): the first
  network won on 12 Nov (10 turns), the second on 13 Nov (20 turns), both by elimination, about
  1 to 2 s per turn, every decision scored on the NPU (300 to 1,600 positions per turn).
- **Still unknown:** how it does against a human. The one recorded loss is a three-player game
  won by a human; games against people have to be played (and are now recorded) to find out.
- **NPU** (Intel AI Boost, Core Ultra 7 155H): the network compiles in about 0.2 s and scores a
  batch of 256 positions in 0.5 to 0.7 ms. The network is small, so the NPU is not faster than
  the CPU here; it keeps the CPU free. `--device CPU` gives the same play.

## Recorded games

The game server keeps only the result, the final position and the last lines of the journal of a
finished game, so past games cannot be replayed from it. From now on the agent records every
game it plays in `traces/` (one file per game, positions and result):

```powershell
.\.venv\Scripts\python -m domination.history     # list the recorded games and their results
```

## Retraining and benchmarking

```powershell
.\.venv\Scripts\python -m domination.train --generations 2 --games 8000 --resume
.\.venv\Scripts\python -m domination.bench --a search --b scripted --players 3 --games 240
```

`train` starts from the current network, plays self-play games (about 18 minutes per 8,000 games
on 16 cores), adds the recorded real games, benchmarks every generation against the first network
(`models/value_v1.npz`) and the scripted bot, and writes the best one (the starting network
included) to `models/value.npz` and `models/value.xml` / `.bin` for the NPU. Training runs on the
CPU; only playing uses the NPU. `bench` plays any two kinds of player against each other.

## Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
```

The tests need neither the network nor the NPU; one of them plays a complete game through an
in-memory imitation of the server API.

## Limits

- Verified on the real server: two complete two-player "Domination" games on the default map
  against the built-in AI. The "Objectifs" victory mode, larger maps and the "Conquête planétaire" map are handled by the code and exercised
  in the simulator where it supports them, but they have not been played live. In particular
  the secret initial placement of the planetary mode uses the order names of the web client
  (`deployer`, `valider_deploiement`); if the agent API refuses them, the agent logs it and lets
  the server place the armies.
- "Objectifs" mode is handled by a simple adjustment on top of the network (which is trained for
  elimination games), not by dedicated training.
- The simulator approximates what the server does not publish (when quests are handed out,
  which territories can revolt, how maps and regions are drawn), so the network's win estimates
  are indicative. Strength is measured against the agent's own earlier versions and a scripted
  bot, which says little about how a good human plays.
- Self-play is not monotonic: a new generation can be weaker than the previous one, so the
  trainer benchmarks every generation and keeps the best one.
- The seat token is a secret: pass it through `IA_JETON`, never commit it.
