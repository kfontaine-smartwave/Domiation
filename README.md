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
| `--trace file.jsonl` | record every order sent and every server reply |
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
4. **Self-play training** ([domination/train.py](domination/train.py)). The network was trained
   on this machine from games played in a local simulator of the rules
   ([domination/simulator.py](domination/simulator.py)): each position reached at the end of a
   turn is labelled with the final result of its game, and successive generations of the network
   play against themselves, an earlier hand-written formula and a scripted bot.

## Results

- **Live game** (12 Nov 2026, private game on the default 10 × 7 map against the site's built-in
  "IA locale equilibre"): the agent won by eliminating its opponent in 10 turns. 87 orders sent,
  none rejected, no timeout, 0.7 to 2.4 s per turn; every decision was scored on the NPU
  (300 to 1,600 positions per turn).
- **Local benchmark** (320 simulated two-player games on the 10 × 7 map against a scripted bot
  that imitates the built-in AI): 292 wins, 28 losses (91 %). Against the hand-written
  evaluation formula it was bootstrapped from, the network is level (316 wins, 324 losses over
  640 games).
- **NPU** (Intel AI Boost, Core Ultra 7 155H): the network compiles in 0.14 s and scores a batch
  of 256 positions in about 0.7 ms. The network is deliberately small, so the NPU is not faster
  than the CPU here (0.1 ms for the same batch); it is used because that was the goal, and it
  keeps the CPU free. `--device CPU` gives the same play.

## Retraining

```powershell
.\.venv\Scripts\python -m domination.train --generations 3 --games 3000
```

writes `models/value.npz` (weights) and `models/value.xml` / `.bin` (OpenVINO model for the NPU).
This is the run that produced the shipped model: about 20 minutes on 16 CPU cores, benchmarks
included. Training runs on the CPU; only playing uses the NPU.

## Tests

```powershell
.\.venv\Scripts\python -m unittest discover -s tests
```

The tests need neither the network nor the NPU; one of them plays a complete game through an
in-memory imitation of the server API.

## Limits

- Verified on the real server: one complete two-player "Domination" game on the default map
  (plus one manual session used to learn the API). The "Objectifs" victory mode, games with 3 or
  4 players, larger maps and the "Conquête planétaire" map are handled by the code and exercised
  in the simulator where it supports them, but they have not been played live. In particular
  the secret initial placement of the planetary mode uses the order names of the web client
  (`deployer`, `valider_deploiement`); if the agent API refuses them, the agent logs it and lets
  the server place the armies.
- "Objectifs" mode is handled by a simple adjustment on top of the network (which is trained for
  elimination games), not by dedicated training.
- The simulator approximates what the server does not publish (when quests are handed out,
  which territories can revolt, how maps and regions are drawn), so the network's win estimates
  are indicative; early in the live game it rated its own position at about 35 % and still won.
- Self-play is not monotonic: the third generation was weaker than the second, so the trainer
  benchmarks every generation and keeps the best one.
- The seat token is a secret: pass it through `IA_JETON`, never commit it.
