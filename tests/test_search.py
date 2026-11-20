import json, os
import numpy as np
from skyjo import fast as F
from skyjo.agents import HeuristicParams
from skyjo.rl.env import VecEnv
from skyjo.rl.search import Batch, _determinize

DEFAULT = HeuristicParams().as_dict()


def test_determinize_keeps_everything_visible_and_card_counts():
    env = VecEnv(16, n_players=3, opponents=[DEFAULT], seed=1)
    for _ in range(40):
        env.step(env.heuristic_actions(DEFAULT))
    b = Batch(env, np.arange(16))
    before = Batch(env, np.arange(16))
    _determinize(*b.g())
    for i in range(16):
        n = env.M[i, F.M_NP]
        vis = before.S[i, :n] != F.HIDDEN
        assert (b.V[i, :n][vis] == before.V[i, :n][vis]).all()          # visible cards unchanged
        assert (b.S[i] == before.S[i]).all() and (b.C[i] == before.C[i]).all()
        assert (b.M[i] == before.M[i]).all()                               # hand, phase etc unchanged
        dl = env.M[i, F.M_DLEN]
        pool_a = np.sort(np.concatenate([before.V[i, :n][~vis], before.D[i, :dl]]))
        pool_b = np.sort(np.concatenate([b.V[i, :n][~vis], b.D[i, :dl]]))
        assert (pool_a == pool_b).all()                                    # same unseen cards
    changed = sum((b.V[i] != before.V[i]).any() or (b.D[i] != before.D[i]).any() for i in range(16))
    assert changed >= 14
