# Skyjo strategy simulator

Phase 1: rules engine, baseline bots, simulation runner. Phase 2: CMA-ES-tuned heuristics.

## Layout
```
skyjo/engine.py    rules (2–8 players), step-based state machine, 26-action space + legal masks
skyjo/agents.py    RandomAgent, greedy, parameterised HeuristicAgent
skyjo/simulate.py  play_game, match (seat rotation), report (win rate & round score with 95% CI)
tests/             rules tests (pytest)
run_baseline.py    first tournaments
```

## Rules as implemented
- PDF rules: 12 cards in 4 columns × 3 rows, reveal 2, the highest sum starts (the previous closer starts later rounds), take the discard or draw, a column of three equal revealed cards is removed (also at the final reveal) and the cards go on top of the exchanged card, every other player gets one last turn, and the closer's score is doubled if positive and another player has the same or fewer points. The game ends at ≥100, lowest total wins.
- Defaults the PDF doesn't cover: 150-card deck (5×−2, 10×−1, 15×0, 10× each of 1–12), the discard pile except its top card is reshuffled when the draw pile runs out, and starting-player ties are broken at random.
- `SkyjoGame(max_rounds=1)` gives the single-round (round-score) objective.

## Action space (shared by scripted agents and RL)
`0` take discard · `1` draw deck · `2–13` replace position p · `14–25` flip position p
(position p = column*3 + row). `game.legal_mask()` gives the mask.

## Run locally
Requires Python 3.9+.
```
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install cma pytest numba numpy   # or: conda install numba numpy cma pytest

python -m pytest -q                # rules tests
python run_baseline.py             # quick sanity tournament (single core)
python tune.py win --minutes 30    # CMA-ES, win objective (resumes from results/cma_win.pkl)
python tune.py round --minutes 30  # CMA-ES, round-score objective
python evaluate.py                 # round robin, writes results/tournament.json + strategies.json
```
All parallel scripts use **every CPU core by default**; limit with `--processes N`
(e.g. to keep the machine responsive). Speed scales roughly linearly with cores.
`tune.py` and `evaluate.py` use the numba engine by default (`--engine python` for the reference).
The first run compiles the numba code (~10–20 s); it's cached in `__pycache__` afterwards.

Notes:
- `tune.py` resumes from its checkpoint. Delete `results/cma_<objective><tag>.pkl` and
  `.jsonl` to start fresh, or use `--tag` to start a separate run.
- Self-play iteration example:
  `python tune.py win --tag _selfplay --opponent results/tuned_win_v1.json --start results/tuned_win_v1.json --sigma0 0.3`
- With more cores you can afford less noise: raise `--games` (games per candidate,
  default 500) and `--popsize` (default 10).
- `evaluate.py` overwrites `results/tournament.json` and `results/strategies.json`.

## Fast engine (numba)
`skyjo/fast.py` is a compiled copy of the engine *and* the heuristic agent. State is a tuple of
NumPy arrays (boards as `int8[8,12]`, piles as `int8[150]` + lengths, scalars in `int64[24]`),
every function is `@njit`, and `play_many` runs games in parallel with `prange` (threads, not
processes). The heuristic reuses scratch arrays and caches the denial check per decision.

`tests/test_fast.py` keeps it honest. It drives the Python engine and the numba engine with
the same moves, copies the numba deal into the Python game after every random event (deal,
reshuffle, starting-player tie), and compares the full state after every move, for 2, 3, 5 and 8
players. It also checks that the compiled heuristic picks the same move as the Python one for
every saved strategy (random tie-breaks excepted).

Speed per core, 2 players, full games:

| | Python | numba |
|---|---|---|
| phase-1 heuristic vs itself | ~135 games/s | ~1,650 games/s |
| tuned_win vs phase1 | ~90 games/s | ~960 games/s |
| random agents (engine only) | – | ~14,000 games/s |

The heuristic dominates the cost. Tuned strategies are slower because their cautious closing
makes rounds longer and the denial check roughly doubles the work per decision.
The full `evaluate.py` round robin went from 405 s to 38 s on 2 cores.

For RL: `FastGame` exposes `step`, `legal_actions` and the raw arrays; a batched
environment can call the compiled `step` / `legal_mask` directly.

## Phase 2: tuned heuristics (2 players)

`tune.py` runs CMA-ES over `HeuristicParams` (10 continuous parameters, common random
numbers, 500 games per candidate, population 10), resumable via `results/cma_*.pkl`.
`evaluate.py` runs the round robin (6,000 games per pairing). Final parameters are in
`results/strategies.json`; learning curves are in `results/cma_*.jsonl`.

| A vs B | A game win rate | A single-round win rate |
|---|---|---|
| tuned_win vs phase1 | 0.690 | 0.547 |
| tuned_win vs tuned_round | 0.661 | 0.581 |
| tuned_round vs phase1 | 0.597 | 0.455 |
| tuned_win vs greedy | 0.961 | 0.762 |

Self-play check: CMA-ES started at tuned_win and played against tuned_win found
nothing better than ~50% (8 generations). So it is a local equilibrium in this strategy family.

Agent features added in phase 2: thresholds interpolated by round progress, a pair bonus,
a denial penalty (don't discard cards the next player wants), closing margin scaled by
lead in totals, an end-of-game lookahead, and an exact expected-value final turn.

## Phase 4: RL agent (PPO)

```
skyjo/rl/env.py   batched environment on the numba engine (obs encoding, masks, opponents, rewards)
skyjo/rl/ppo.py   actor-critic network, PPO update, GAE, evaluation
train_rl.py       behaviour cloning + PPO with prioritised self-play, schedules, checkpoints, resume
eval_rl.py        evaluate a checkpoint against all saved heuristics
plot_runs.py      learning curves of one or more runs -> runs/learning_curves.png
tests/test_env.py environment tests (masks, symmetry, rewards, heuristic win rate through the env)
```

- **Environment:** B games in parallel. In each game one random seat is the learner; the other seats are heuristics
  (default tuned_win 70%, phase1 20%, greedy 10%) or frozen earlier copies of the agent (30% of seats).
  Observation = 463 numbers for 2 players (see the env.py docstring), plus the legal-move mask.
- **Symmetry:** with canonical ordering (default) every board is shown in a sorted order: cards inside each column
  sorted, then columns sorted. Equivalent positions look identical to the network, and actions are mapped back
  to real positions.
- **Rewards:** win objective: +1/−1 at the end of the game (optional `--shaping` adds a decaying round-score term).
  Round objective: single rounds, reward −score/20.
- **Training:** behaviour cloning from tuned_win first (`--bc-iters 40`, ~5M moves), then PPO: 512×3 MLP,
  masked logits, GAE (γ 0.997, λ 0.95), 1024 envs × 128 steps per iteration. Learning rate decays to 10% and the
  entropy bonus from 0.01 to 0.002 over the run. Self-play snapshots every 20 iterations (pool of 12), picked
  30% newest, otherwise weighted towards snapshots the agent beats least.
- **CPU check:** behaviour cloning alone (1.3M moves, 256-wide net) reached 79% move agreement with the teacher
  and 39.5% wins vs tuned_win, about what the first PPO run (`win2p`, no warm start) reached in 30 minutes.

```
python -m pytest -q tests/test_env.py
python train_rl.py --name v2 --minutes 240        # resumes if runs/v2/latest.pt exists
python eval_rl.py runs/v2/best.pt --games 10000
python plot_runs.py win2p v2
```
Every 50 iterations the agent is evaluated against tuned_win (4,000 games, best move each time); the best
checkpoint is saved as `runs/<name>/best.pt`. Older runs without canonical ordering resume with `--no-canonical`.

## Next phases
3. **RL:** Gymnasium wrapper (opponents inside the env), PPO with action masking, self-play against a pool of earlier checkpoints; port the engine to numba or vectorise it for speed.
4. **Analysis:** RL vs the tuned heuristic, policy distillation into readable rules, and how the strategy shifts with player count.
