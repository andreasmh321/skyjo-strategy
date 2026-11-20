"""Search at play time: determinized rollouts (perfect-information Monte Carlo).

For a decision of player `me`:
  1. candidates = the network's most likely moves (prior >= min_prior, at most
     max_cands); no search if the top move has prior >= skip_above,
  2. K determinizations: all cards `me` cannot see (every hidden board card,
     including its own, plus the draw pile) are reshuffled into those places,
  3. every candidate is applied in each of the same K worlds and played forward
     with the network for all seats, until `me` has started `depth` more turns,
     a new round starts, or the game ends,
  4. the result is the value head's estimate (for `me`) at that point, or the
     real game result; the candidate with the best mean wins.

Many root decisions (from many games) are searched together in one batch so
the network runs on large batches.
"""
from __future__ import annotations

import numpy as np
import torch
from numba import njit, prange

from .. import fast as F
from .env import _encode_all, _view, to_real

N_POS = F.N_POS


@njit(cache=True, parallel=True)
def _determinize(V, S, D, C, M, T, A):
    """Reshuffle every hidden board card and the draw pile, in place."""
    N = V.shape[0]
    for i in prange(N):
        n = M[i, F.M_NP]
        pool = np.empty(F.DECK_SIZE, np.int8)
        k = 0
        for pl in range(n):
            for p in range(N_POS):
                if S[i, pl, p] == F.HIDDEN:
                    pool[k] = V[i, pl, p]
                    k += 1
        for j in range(M[i, F.M_DLEN]):
            pool[k] = D[i, j]
            k += 1
        np.random.shuffle(pool[:k])
        k = 0
        for pl in range(n):
            for p in range(N_POS):
                if S[i, pl, p] == F.HIDDEN:
                    V[i, pl, p] = pool[k]
                    k += 1
        for j in range(M[i, F.M_DLEN]):
            D[i, j] = pool[k]
            k += 1


@njit(cache=True, parallel=True)
def _apply(V, S, D, C, M, T, A, sel, actions, perms):
    """Apply canonical actions (mapped through perms) to the selected games."""
    for i in prange(V.shape[0]):
        if sel[i]:
            g = _view(V, S, D, C, M, T, A, i)
            F.step(g, to_real(actions[i], perms[i]))


class Batch:
    """N game copies as stacked arrays (same layout as VecEnv)."""

    def __init__(self, env, idx):
        self.V, self.S = env.V[idx].copy(), env.S[idx].copy()
        self.D, self.C = env.D[idx].copy(), env.C[idx].copy()
        self.M, self.T, self.A = env.M[idx].copy(), env.T[idx].copy(), env.A[idx].copy()

    def g(self):
        return (self.V, self.S, self.D, self.C, self.M, self.T, self.A)

    def take(self, idx):
        b = Batch.__new__(Batch)
        b.V, b.S, b.D, b.C = self.V[idx], self.S[idx], self.D[idx], self.C[idx]
        b.M, b.T, b.A = self.M[idx], self.T[idx], self.A[idx]
        return b


@torch.no_grad()
def rollout_values(batch, me, model, device, depth=4, canonical=True, max_plies=300,
                   chunk=16384, greedy=True):
    """Play every game in `batch` forward with the network for all seats (its top
    move if greedy, else sampled) and return the value for player me[i]."""
    N = batch.V.shape[0]
    g = batch.g()
    r0 = batch.M[:, F.M_NROUNDS].copy()
    turns = np.zeros(N, np.int64)
    value = np.zeros(N, np.float32)
    active = np.ones(N, np.bool_)
    obs = np.zeros((N, model.config["obs_dim"]), np.float32)
    masks = np.zeros((N, F.N_ACTIONS), np.bool_)
    perms = np.zeros((N, N_POS), np.int64)
    acts = np.full(N, -1, np.int64)
    n = int(batch.M[0, F.M_NP])
    for ply in range(max_plies + 1):
        M = batch.M
        over = M[:, F.M_PHASE] == F.GAME_OVER
        if (over & active).any():
            idx = np.flatnonzero(over & active)
            tot = batch.T[idx, :n]
            best = tot.min(1)
            mine = tot[np.arange(len(idx)), me[idx]]
            share = np.where(mine == best, 1.0 / (tot == best[:, None]).sum(1), 0.0)
            value[idx] = 2 * share - 1
            active[idx] = False
        if not active.any():
            break
        cur = M[:, F.M_CUR]
        my_turn = active & (cur == me)
        new_turn = my_turn & (M[:, F.M_PHASE] == F.CHOOSE)
        turns += new_turn
        stop = my_turn & ((M[:, F.M_NROUNDS] > r0) | (turns > depth))
        if ply == max_plies:
            stop = active.copy()
        _encode_all(*g, active, obs, masks, perms, canonical)
        idx = np.flatnonzero(active)
        for s in range(0, len(idx), chunk):
            j = idx[s:s + chunk]
            o = torch.from_numpy(obs[j]).to(device)
            mk = torch.from_numpy(masks[j]).to(device)
            logits, v = model(o, mk)
            a = logits.argmax(-1) if greedy else torch.distributions.Categorical(logits=logits).sample()
            acts[j] = a.cpu().numpy()
            vj = v.float().cpu().numpy()
            sj = stop[j]
            # value is from the point of view of the player to move
            sign = np.where(cur[j] == me[j], 1.0, -1.0)
            value[j[sj]] = (vj * sign)[sj]
        active &= ~stop
        if not active.any():
            break
        _apply(*g, active, acts, perms)
    return value


@torch.no_grad()
def search_actions(env, model, device, K=32, depth=4, min_prior=0.02, max_cands=6,
                   skip_above=0.95, canonical=True, z=2.0, greedy=True):
    """Choose an action for the player to move in every game of `env`.
    The network's top move is replaced only if another candidate beats it by more
    than z standard errors of the paired (same-world) difference.
    Returns (actions, info) with actions in the env's canonical slots."""
    B = env.B
    o = torch.from_numpy(env.obs).to(device)
    mk = torch.from_numpy(env.masks).to(device)
    logits, _ = model(o, mk)
    probs = torch.softmax(logits, -1).cpu().numpy()
    actions = probs.argmax(1)
    games, cands = [], []
    for b in range(B):
        order = np.argsort(-probs[b])
        c = [int(a) for a in order[:max_cands] if probs[b, a] >= min_prior and env.masks[b, a]]
        if len(c) >= 2 and probs[b, order[0]] < skip_above:
            games.append(b)
            cands.append(c)
    info = {"searched": len(games), "rollouts": 0, "changed": 0}
    if not games:
        return actions, info
    # K determinized copies per searched game (shared by all its candidates)
    base_idx = np.repeat(np.array(games), K)
    base = Batch(env, base_idx)
    _determinize(*base.g())
    # expand: one copy per (game, candidate, k)
    slot_base, slot_act, slot_game, slot_me, slot_perm = [], [], [], [], []
    for gi, (b, c) in enumerate(zip(games, cands)):
        for a in c:
            for k in range(K):
                slot_base.append(gi * K + k)
                slot_act.append(a)
                slot_game.append(gi)
        slot_me += [env.M[b, F.M_CUR]] * (len(c) * K)
        slot_perm += [env.perms[b]] * (len(c) * K)
    batch = base.take(np.array(slot_base))
    acts = np.array(slot_act, np.int64)
    sel = np.ones(len(acts), np.bool_)
    _apply(*batch.g(), sel, acts, np.array(slot_perm, np.int64))
    vals = rollout_values(batch, np.array(slot_me, np.int64), model, device, depth, canonical,
                          greedy=greedy)
    info["rollouts"] = len(acts)
    # paired comparison against the network's own top move (candidate 0)
    pos = 0
    for b, c in zip(games, cands):
        v = vals[pos: pos + len(c) * K].reshape(len(c), K)
        pos += len(c) * K
        diff = v[1:] - v[0]                       # same worlds -> paired differences
        mean = diff.mean(1)
        se = diff.std(1, ddof=1) / np.sqrt(K) + 1e-9
        best_i = int(np.argmax(mean))
        if mean[best_i] > 0 and mean[best_i] > z * se[best_i]:
            actions[b] = c[best_i + 1]
            info["changed"] += 1
    return actions, info
