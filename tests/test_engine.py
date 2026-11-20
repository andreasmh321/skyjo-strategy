import random

import pytest

from skyjo.engine import (
    A_DRAW_DECK, A_TAKE_DISCARD, CHOOSE_SOURCE, DECIDE_DRAWN, DECK_COUNTS, DECK_SIZE,
    GAME_OVER, N_POS, PLACE_TAKEN, SETUP, SkyjoGame, flip_action, replace_action,
)
from skyjo.agents import RandomAgent, HeuristicAgent, greedy_agent
from skyjo.simulate import play_game


def finish_setup(g, reveals=((0, 1),) * 8):
    while g.phase == SETUP:
        p = g.current
        a, b = reveals[p]
        g.step(flip_action(a))
        g.step(flip_action(b))


def all_cards(g):
    cards = list(g.deck) + list(g.discard)
    for b in g.boards:
        cards += [b.values[p] for p in range(N_POS) if not b.removed[p]]
    if g.hand is not None:
        cards.append(g.hand)
    return cards


def test_deck_composition():
    assert DECK_SIZE == 150
    g = SkyjoGame(2, seed=1)
    assert len(all_cards(g)) == 150
    assert len(g.discard) == 1 and len(g.deck) == 150 - 24 - 1


def test_starting_player_highest_sum():
    g = SkyjoGame(2, seed=3)
    g.boards[0].values[:2] = [-2, 0]
    g.boards[1].values[:2] = [12, 12]
    finish_setup(g)
    assert g.current == 1 and g.phase == CHOOSE_SOURCE


def test_triple_removed_and_goes_on_discard():
    g = SkyjoGame(2, seed=5)
    finish_setup(g)
    me = g.current
    b = g.boards[me]
    b.values[0:3] = [7, 7, 3]
    b.revealed[0] = b.revealed[1] = True
    b.revealed[2] = True
    g.discard.append(7)
    g.step(A_TAKE_DISCARD)
    g.step(replace_action(2))
    assert b.removed[0] and b.removed[1] and b.removed[2]
    # the exchanged 3 lies under the three 7s
    assert g.discard[-4:] == [3, 7, 7, 7]


def test_triple_at_final_reveal():
    g = SkyjoGame(2, seed=7, max_rounds=1)
    finish_setup(g)
    closer = g.current
    other = 1 - closer
    # other player: column 1 holds three 5s, only one revealed
    ob = g.boards[other]
    ob.values[3:6] = [5, 5, 5]
    ob.revealed[3:6] = [True, False, False]
    # make the closer close this turn: reveal all but one, then flip it
    cb = g.boards[closer]
    for p in range(N_POS):
        cb.revealed[p] = True
    cb.revealed[11] = False
    g.step(A_DRAW_DECK)
    g.step(flip_action(11))
    assert g.closer == closer
    # other's final turn: draw and flip something outside column 1
    g.step(A_DRAW_DECK)
    hid = [p for p in g.boards[other].hidden_positions() if p not in (3, 4, 5)]
    g.step(flip_action(hid[0]))
    res = g.round_results[-1]
    assert all(ob.removed[p] for p in (3, 4, 5))
    assert res.raw_scores[other] == ob.score()


def _round_with_scores(closer_score, other_score, seed=11):
    g = SkyjoGame(2, seed=seed, max_rounds=1)
    finish_setup(g)
    c = g.current
    o = 1 - c
    # all distinct-ish columns so no triples: build boards summing to targets
    def fill(target):
        vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0]
        diff = target - sum(vals)
        vals[-1] += diff
        return vals
    g.boards[c].values = fill(closer_score)
    g.boards[o].values = fill(other_score)
    for p in range(N_POS):
        g.boards[c].revealed[p] = True
    g.boards[c].revealed[11] = False
    g.step(A_DRAW_DECK)
    drawn = g.hand
    g.step(flip_action(11))
    g.step(A_DRAW_DECK)
    g.step(flip_action(g.boards[o].hidden_positions()[0]))
    return g.round_results[-1]


def test_closer_doubled_on_tie():
    r = _round_with_scores(70, 70)
    assert r.doubled and r.scores[r.closer] == 140


def test_closer_not_doubled_when_strictly_lowest():
    r = _round_with_scores(60, 70)
    assert not r.doubled


def test_closer_not_doubled_when_nonpositive():
    r = _round_with_scores(0, -5)
    assert not r.doubled


def test_final_turns_each_other_player_once():
    g = SkyjoGame(4, seed=13, max_rounds=1)
    finish_setup(g)
    c = g.current
    for p in range(N_POS):
        g.boards[c].revealed[p] = True
    g.boards[c].revealed[11] = False
    g.step(A_DRAW_DECK)
    g.step(flip_action(11))
    turns = 0
    while g.phase != GAME_OVER:
        assert g.current != c
        g.step(A_DRAW_DECK)
        g.step(flip_action(g.boards[g.current].hidden_positions()[0]))
        turns += 1
    assert turns == 3


def test_reshuffle_keeps_top_and_conserves_cards():
    g = SkyjoGame(2, seed=17)
    finish_setup(g)
    g.discard.extend(g.deck)
    g.deck = []
    top = g.discard[-1]
    g.step(A_DRAW_DECK)
    assert g.discard == [top]
    assert g.n_reshuffles == 1
    assert len(all_cards(g)) == 150


@pytest.mark.parametrize("n", [2, 3, 5, 8])
def test_random_games_terminate_and_conserve(n):
    rng = random.Random(n)
    for _ in range(30):
        agents = [RandomAgent(rng.random()) for _ in range(n)]
        g = play_game(agents, seed=rng.getrandbits(32))
        assert g.done and max(g.totals) >= 100
        assert sum(sum(r.scores) for r in g.round_results) == sum(g.totals)


def test_heuristic_games_terminate():
    for s in range(30):
        g = play_game([HeuristicAgent(seed=s), greedy_agent(seed=s + 1)], seed=s)
        assert g.done


def test_unseen_counts_sum():
    g = SkyjoGame(3, seed=19)
    finish_setup(g)
    obs = g.observation()
    hidden = sum(len(b.hidden_positions()) for b in g.boards)
    assert sum(obs.unseen.values()) == len(g.deck) + hidden
    assert all(v >= 0 for v in obs.unseen.values())
