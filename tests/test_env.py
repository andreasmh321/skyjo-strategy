import json
import os

import numpy as np
import pytest

from skyjo import fast as F
from skyjo.agents import HeuristicParams
from skyjo.rl.env import VecEnv, encode, obs_dim

ROOT = os.path.join(os.path.dirname(__file__), "..")
DEFAULT = HeuristicParams().as_dict()


def strategies():
    p = os.path.join(ROOT, "results", "strategies.json")
    return json.load(open(p)) if os.path.exists(p) else {"tuned_win": DEFAULT, "phase1": DEFAULT}


def random_actions(masks, rng):
    out = np.empty(len(masks), np.int64)
    for i, m in enumerate(masks):
        out[i] = rng.choice(np.flatnonzero(m))
    return out


@pytest.mark.parametrize("canonical", [True, False])
@pytest.mark.parametrize("n", [2, 3])
def test_obs_and_masks(n, canonical):
    env = VecEnv(16, n_players=n, opponents=[DEFAULT], seed=1, canonical=canonical)
    rng = np.random.default_rng(0)
    obs, masks = env.obs, env.masks
    assert obs.shape == (16, obs_dim(n))
    for _ in range(300):
        # every board position has exactly one active feature
        boards = obs[:, :n * 12 * 17].reshape(16, n * 12, 17)
        assert np.all(boards.sum(-1) == 1)
        # phase one-hot; learner is always to move
        o = n * 12 * 17 + 30
        assert np.all(obs[:, o:o + 4].sum(-1) == 1)
        assert np.all(env.kinds[np.arange(16), env.M[:, F.M_CUR]] == -1)
        # masks equal the engine's legal moves, mapped through the permutation
        for b in range(16):
            g = (env.V[b], env.S[b], env.D[b], env.C[b], env.M[b], env.T[b], env.A[b])
            ref = np.empty(26, np.bool_)
            F.legal_mask(g, ref)
            perm = env.perms[b]
            assert sorted(perm) == list(range(12))
            if not canonical:
                assert (perm == np.arange(12)).all()
            mapped = np.concatenate([ref[:2], ref[2:14][perm], ref[14:][perm]])
            assert (mapped == masks[b]).all()
        obs, masks, r, d = env.step(random_actions(masks, rng))


def test_illegal_action_rejected():
    env = VecEnv(4, opponents=[DEFAULT], seed=2)
    bad = np.array([np.flatnonzero(~m)[0] for m in env.masks])
    with pytest.raises(ValueError):
        env.step(bad)


def test_heuristic_through_env_matches_play_many():
    """Learner seat driven by the tuned heuristic vs phase1 opponents must
    reproduce the tournament win rate (~0.69)."""
    S = strategies()
    env = VecEnv(64, opponents=[S["phase1"]], seed=3)
    eps = []
    while sum(len(e) for e in eps) < 1500:
        env.step(env.heuristic_actions(S["tuned_win"]))
        eps.append(env.pop_episodes())
    eps = np.concatenate(eps)
    ref = F.play_many(F.params_vector(S["tuned_win"]), F.params_vector(S["phase1"])[None],
                      np.arange(4000, dtype=np.int64), 0, 100)[:, 0].mean()
    wr = eps[:, 0].mean()
    assert abs(wr - ref) < 4 * np.sqrt(0.25 / len(eps)), (wr, ref)


@pytest.mark.parametrize("shaping", [0.0, 0.5])
def test_reward_accounting_win(shaping):
    env = VecEnv(32, opponents=[DEFAULT], shaping=shaping, seed=4)
    rng = np.random.default_rng(1)
    ret = np.zeros(32)
    rounds_rew, n_done = [], 0
    returns = []
    while n_done < 200:
        obs, masks, r, d = env.step(random_actions(env.masks, rng))
        ret += r
        for b in np.flatnonzero(d):
            returns.append(ret[b])
            ret[b] = 0
            n_done += 1
    eps = env.pop_episodes()
    returns = np.array(returns)
    if shaping == 0.0:
        assert set(np.round(returns, 6)) <= {-1.0, 0.0, 1.0}
        assert np.allclose(np.sort(returns), np.sort(2 * eps[:len(returns), 0] - 1))
    else:
        # shaping adds non-integer parts, terminal part still +-1
        assert np.any(np.abs(returns - np.round(returns)) > 1e-6)


def test_round_objective():
    env = VecEnv(32, opponents=[DEFAULT], objective="round", seed=5)
    rng = np.random.default_rng(2)
    ret = np.zeros(32)
    rets = []
    while len(rets) < 200:
        obs, masks, r, d = env.step(random_actions(env.masks, rng))
        ret += r
        for b in np.flatnonzero(d):
            rets.append(ret[b]); ret[b] = 0
    eps = env.pop_episodes()
    assert np.all(eps[:, 1] == 1)  # one round per episode
    assert np.allclose(sorted(-20 * np.array(rets)), sorted(eps[:len(rets), 2]))


def test_net_opponents_selfplay():
    rng = np.random.default_rng(3)
    calls = {"n": 0}

    def net(obs, masks):
        calls["n"] += len(obs)
        return random_actions(masks, rng)

    env = VecEnv(32, n_players=3, opponents=[DEFAULT], probs=[0.0, 1.0], seed=6, net_policy=net)
    for _ in range(3000):
        env.step(random_actions(env.masks, rng))
    eps = env.pop_episodes()
    assert len(eps) > 10 and calls["n"] > 1000
    assert np.all(eps[:, 3] == -2)


def test_canonical_encoding_is_symmetry_invariant():
    """Shuffling columns and cards within columns of any board must not change
    the canonical observation."""
    rng = np.random.default_rng(7)
    env = VecEnv(8, n_players=3, opponents=[DEFAULT], seed=8)
    for _ in range(150):
        env.step(env.heuristic_actions(DEFAULT))
        for b in range(8):
            g = (env.V[b].copy(), env.S[b].copy(), env.D[b], env.C[b], env.M[b], env.T[b], env.A[b])
            o1 = np.zeros(env.obs_dim, np.float32); p1 = np.zeros(12, np.int64)
            encode(g, o1, p1, True)
            for pl in range(3):
                cols = rng.permutation(4)
                newpos = np.concatenate([3 * c + rng.permutation(3) for c in cols])
                g[0][pl] = g[0][pl][newpos]
                g[1][pl] = g[1][pl][newpos]
            o2 = np.zeros(env.obs_dim, np.float32); p2 = np.zeros(12, np.int64)
            encode(g, o2, p2, True)
            assert (o1 == o2).all()


def test_endless_round_is_aborted():
    """A player that always swaps the discard for an equal card can stall forever;
    the env must stop such a game and mark it aborted."""
    env = VecEnv(4, opponents=[DEFAULT], seed=9, max_round_turns=60)
    for _ in range(2000):
        # always take the discard and put it back on the same position it came from
        acts = np.where(env.masks[:, 0], 0, np.argmax(env.masks, 1))
        env.step(acts)
    eps = env.pop_episodes()
    assert len(eps) > 0 and eps.shape[1] == 5
