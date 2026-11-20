"""The numba engine and heuristic must match the Python reference step for step.

Randomness differs between the engines (deals, reshuffles, ties), so after
every random event the numba state is copied into the Python game; between
those events every move is compared in full.
"""
import json
import os
import random

import numpy as np
import pytest

from skyjo import engine as E
from skyjo import fast as F
from skyjo.agents import HeuristicAgent, HeuristicParams

PHASES = {F.SETUP: E.SETUP, F.CHOOSE: E.CHOOSE_SOURCE, F.PLACE: E.PLACE_TAKEN,
          F.DECIDE: E.DECIDE_DRAWN, F.GAME_OVER: E.GAME_OVER}


def sync(py, fg):
    values, st, deck, disc, m, totals, acc = fg.g
    n = fg.n_players
    for i, b in enumerate(py.boards):
        b.values = [int(x) for x in values[i]]
        b.revealed = [bool(x != F.HIDDEN) for x in st[i]]
        b.removed = [bool(x == F.REMOVED) for x in st[i]]
    py.deck = [int(x) for x in deck[:m[F.M_DLEN]]]
    py.discard = [int(x) for x in disc[:m[F.M_CLEN]]]
    py.hand = None if m[F.M_HAND] == F.NOCARD else int(m[F.M_HAND])
    py.phase = PHASES[int(m[F.M_PHASE])]
    py.current = int(m[F.M_CUR])
    py.closer = None if m[F.M_CLOSER] < 0 else int(m[F.M_CLOSER])
    py.final_turns_left = int(m[F.M_FINAL])
    py._setup_reveals_left = [2] * n
    py._setup_reveals_left[py.current % n] = int(m[F.M_SETUP])
    py.totals = [int(x) for x in totals[:n]]
    py.next_starter = None if m[F.M_NEXT] < 0 else int(m[F.M_NEXT])


def assert_same(py, fg):
    values, st, deck, disc, m, totals, acc = fg.g
    for i, b in enumerate(py.boards):
        assert b.values == [int(x) for x in values[i]]
        assert [r or q for r, q in zip(b.revealed, b.removed)] == [bool(x != F.HIDDEN) for x in st[i]]
        assert b.removed == [bool(x == F.REMOVED) for x in st[i]]
    assert py.deck == [int(x) for x in deck[:m[F.M_DLEN]]]
    assert py.discard == [int(x) for x in disc[:m[F.M_CLEN]]]
    assert py.hand == (None if m[F.M_HAND] == F.NOCARD else int(m[F.M_HAND]))
    assert py.phase == PHASES[int(m[F.M_PHASE])]
    assert py.current == fg.current
    assert py.closer == (None if m[F.M_CLOSER] < 0 else int(m[F.M_CLOSER]))
    assert py.totals == [int(x) for x in totals[:fg.n_players]]
    assert sorted(py.legal_actions()) == fg.legal_actions()


def run_pair(n, seed, choose, max_rounds=None):
    """choose(py, fg) -> action. Returns number of steps compared."""
    fg = F.FastGame(n, seed=seed, max_rounds=max_rounds)
    py = E.SkyjoGame(n, seed=seed, max_rounds=max_rounds)
    sync(py, fg)
    steps = 0
    while not fg.done:
        assert_same(py, fg)
        a = choose(py, fg)
        r0, resh0, ph0 = fg.m[F.M_NROUNDS], fg.m[F.M_NRESH], fg.phase
        py.step(a)
        fg.step(a)
        steps += 1
        if fg.m[F.M_NROUNDS] != r0:
            rr = py.round_results[-1]
            acc = fg.g[6]
            assert rr.raw_scores == [int(x) for x in acc[4, :n]]
            assert rr.scores == [int(x) for x in acc[3, :n]]
            assert rr.closer == fg.m[F.M_LAST_CLOSER]
            assert rr.doubled == bool(fg.m[F.M_LAST_DOUBLED])
            assert py.totals == [int(x) for x in fg.g[5][:n]]
            assert py.done == fg.done
            sync(py, fg)
        elif fg.m[F.M_NRESH] != resh0:
            assert py.n_reshuffles >= 1
            values, st, deck, disc, m, totals, acc = fg.g
            assert py.discard == [int(x) for x in disc[:m[F.M_CLEN]]]
            # each engine drew a different card from its own shuffle
            assert sorted(py.deck + [py.hand]) == sorted(
                [int(x) for x in deck[:m[F.M_DLEN]]] + [int(m[F.M_HAND])])
            sync(py, fg)
        elif ph0 == F.SETUP and fg.phase == F.CHOOSE:
            sums = [b.visible_sum() for b in py.boards]
            if py.next_starter is None:
                assert sums[fg.current] == max(sums)
            else:
                assert fg.current == py.next_starter
            sync(py, fg)
    assert py.done
    return steps


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_engine_matches_reference_random_play(n):
    total = 0
    for s in range(40 if n <= 3 else 10):
        rng = random.Random(s)
        total += run_pair(n, 1000 * n + s, lambda py, fg: rng.choice(fg.legal_actions()))
    assert total > 1000


def strategy_sets():
    sets = {"default": HeuristicParams().as_dict(),
            "greedy": {**HeuristicParams().as_dict(), "take_gain_early": 0, "take_gain_late": 0,
                       "keep_gain_early": 0, "keep_gain_late": 0, "pair_bonus": 0,
                       "close_margin": -1e9, "end_lookahead": 0, "setup_same_col": 0},
            "denial": {**HeuristicParams().as_dict(), "deny_weight": 0.5, "close_lead_coef": 1.0,
                       "hidden_bonus": 0.7, "final_turn_greedy": 0.0}}
    path = os.path.join(os.path.dirname(__file__), "..", "results", "strategies.json")
    if os.path.exists(path):
        sets.update(json.load(open(path)))
    return sets


def flip_key(st_row, q):
    b = 3 * (q // 3)
    return sum(st_row[r] == F.HIDDEN for r in range(b, b + 3))


@pytest.mark.parametrize("name", list(strategy_sets()))
@pytest.mark.parametrize("n", [2, 3])
def test_heuristic_matches_reference(name, n):
    # seat 0 plays the strategy, the others the default heuristic (two
    # round-score-tuned bots facing each other would never end a round)
    seat_params = [strategy_sets()[name]] + [HeuristicParams().as_dict()] * (n - 1)
    vecs = [F.params_vector(p) for p in seat_params]
    stats = {"compared": 0, "random_ties": 0}

    for s in range(25):
        agents = [HeuristicAgent(HeuristicParams(**p), seed=s * 10 + i)
                  for i, p in enumerate(seat_params)]

        def choose(py, fg):
            params = seat_params[py.current]
            a_py = agents[py.current].act(py.observation())
            a_nb = fg.heuristic_action(vecs[py.current])
            st_row = fg.g[1][fg.current]
            if a_py != a_nb:
                if fg.phase == F.SETUP:
                    forced = params["setup_same_col"] > 0.5 and any(
                        sum(st_row[q] == F.HIDDEN for q in range(3 * c, 3 * c + 3)) == 2
                        for c in range(4))
                    assert not forced, (a_py, a_nb)
                    assert a_nb in fg.legal_actions()
                else:
                    assert a_py >= F.FLIP0 and a_nb >= F.FLIP0, (name, a_py, a_nb, fg.phase)
                    assert flip_key(st_row, a_py - F.FLIP0) == flip_key(st_row, a_nb - F.FLIP0)
                stats["random_ties"] += 1
            else:
                stats["compared"] += 1
            return a_py

        run_pair(n, 77 + s, choose)
    assert stats["compared"] > 500


def test_card_conservation_and_termination():
    for s in range(200):
        fg = F.FastGame(3, seed=s)
        while not fg.done:
            values, st, deck, disc, m, totals, acc = fg.g
            cnt = np.zeros(15, int)
            for i in range(3):
                for p in range(12):
                    if st[i, p] != F.REMOVED:
                        cnt[values[i, p] + 2] += 1
            for v in deck[:m[F.M_DLEN]]:
                cnt[v + 2] += 1
            for v in disc[:m[F.M_CLEN]]:
                cnt[v + 2] += 1
            if m[F.M_HAND] != F.NOCARD:
                cnt[m[F.M_HAND] + 2] += 1
            assert (cnt == F.FULL_COUNTS).all()
            fg.step(int(F.random_act(fg.g)), check=False)
        assert fg.g[5][:3].max() >= 100


def test_play_many_deterministic():
    p = F.params_vector(HeuristicParams().as_dict())
    seeds = np.arange(200, dtype=np.int64)
    a = F.play_many(p, p[None, :], seeds, 0, 100)
    b = F.play_many(p, p[None, :], seeds, 0, 100)
    assert (a == b).all()
    assert abs(a[:, 0].mean() - 0.5) < 0.1  # mirror match
