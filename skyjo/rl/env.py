"""Batched Skyjo environment for RL, built on the numba engine (skyjo/fast.py).

B games run side by side. In every game one seat is the *learner*; the other
seats are opponents. An opponent is either a heuristic (one of K parameter
vectors) or a frozen copy of the network ("net" opponent, for self-play).

Flow per call of `VecEnv.step(actions)`:
  1. the learner's action is applied in every game,
  2. heuristic opponents are played inside compiled code until it is the
     learner's (or a net opponent's) turn,
  3. games waiting for a net opponent are batched through the frozen policy
     in Python until every game is back at the learner's turn,
  4. finished games are recorded and reset automatically.

Observation (from the point of view of the player to move; boards ordered
me, next player, ...):
  boards   n x 12 x 17  one-hot per position: hidden, removed, value -2..12
  discard  15           top of discard pile (one-hot, zeros if empty)
  hand     15           card in hand (one-hot, zeros if none)
  phase    4            setup / choose source / place taken / decide drawn
  unseen   15           counts of cards the player cannot see, / 10
  misc     1 + n + n + 1  deck size/150, totals/100, closer one-hot, E/12

Canonical ordering (canonical=True): the rules only care which cards share a
column, so cards inside a column and the four columns themselves can be put in
any order. Each board is encoded in a sorted order (cards within a column by
state/value, then columns by their sorted contents). The network acts in that
sorted order; masks and actions are mapped to and from real positions, so all
equivalent positions look identical to the network.

Rewards (to the learner only):
  win objective   : +1 for a win, -1 for a loss (ties split: 2*share-1) at the
                    end of the game, plus `shaping` * (mean opponent round
                    score - own round score) / 20 at the end of every round
  round objective : games are single rounds, reward = -own score / 20
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .. import fast as F

N_POS = F.N_POS
POS_FEATS = 17
LEARNER = -1
NET = -2


def obs_dim(n_players: int) -> int:
    return n_players * N_POS * POS_FEATS + 15 + 15 + 4 + 15 + 1 + 2 * n_players + 1


# ----------------------------------------------------------------- numba core
@njit(cache=True)
def _view(V, S, D, C, M, T, A, b):
    return (V[b], S[b], D[b], C[b], M[b], T[b], A[b])


@njit(cache=True)
def _reset_game(g, n, max_rounds, end_score):
    values, st, deck, disc, m, totals, acc = g
    m[:] = 0
    totals[:] = 0
    acc[:, :] = 0
    m[F.M_NP] = n
    m[F.M_MAXR] = max_rounds
    m[F.M_END] = end_score
    m[F.M_ROUND] = -1
    m[F.M_NEXT] = -1
    m[F.M_NREVEAL] = 2
    F.start_round(g)


@njit(cache=True)
def _assign_seats(kind_row, n, probs):
    """Random learner seat; other seats sampled from probs (last entry = net)."""
    seat = np.random.randint(0, n)
    for s in range(n):
        if s == seat:
            kind_row[s] = LEARNER
        else:
            u = np.random.random()
            c = 0.0
            k = probs.shape[0] - 1
            for j in range(probs.shape[0]):
                c += probs[j]
                if u < c:
                    k = j
                    break
            kind_row[s] = NET if k == probs.shape[0] - 1 else k
    return seat


@njit(cache=True)
def _code(st, values, pl, p):
    s = st[pl, p]
    if s == F.HIDDEN:
        return 0
    if s == F.REMOVED:
        return 1
    return 2 + values[pl, p] + 2


@njit(cache=True)
def canon_perm(st, values, pl, perm):
    """perm[k] = real position shown at canonical slot k."""
    keys = np.empty(4, np.int64)
    cols = np.empty((4, 3), np.int64)
    for c in range(4):
        ps = [3 * c, 3 * c + 1, 3 * c + 2]
        cs = [_code(st, values, pl, ps[0]), _code(st, values, pl, ps[1]),
              _code(st, values, pl, ps[2])]
        # sort the three cards of the column by code (stable)
        for i in range(1, 3):
            j = i
            while j > 0 and cs[j - 1] > cs[j]:
                cs[j - 1], cs[j] = cs[j], cs[j - 1]
                ps[j - 1], ps[j] = ps[j], ps[j - 1]
                j -= 1
        for i in range(3):
            cols[c, i] = ps[i]
        keys[c] = cs[0] * 289 + cs[1] * 17 + cs[2]
    order = np.argsort(keys, kind="mergesort")
    for k in range(4):
        for i in range(3):
            perm[3 * k + i] = cols[order[k], i]


@njit(cache=True)
def encode(g, out, perm, canonical):
    """Write the observation of the player to move into float32 `out`; perm
    receives the canonical->real position map of that player's board."""
    values, st, deck, disc, m, totals, acc = g
    n = m[F.M_NP]
    me = m[F.M_CUR]
    out[:] = 0.0
    o = 0
    bp = np.empty(N_POS, np.int64)
    for r in range(n):
        pl = (me + r) % n
        if canonical:
            canon_perm(st, values, pl, bp)
        else:
            for p in range(N_POS):
                bp[p] = p
        if r == 0:
            perm[:] = bp
        for k in range(N_POS):
            p = bp[k]
            s = st[pl, p]
            if s == F.HIDDEN:
                out[o] = 1.0
            elif s == F.REMOVED:
                out[o + 1] = 1.0
            else:
                out[o + 2 + values[pl, p] + 2] = 1.0
            o += POS_FEATS
    if m[F.M_CLEN] > 0:
        out[o + disc[m[F.M_CLEN] - 1] + 2] = 1.0
    o += 15
    if m[F.M_HAND] != F.NOCARD:
        out[o + m[F.M_HAND] + 2] = 1.0
    o += 15
    out[o + m[F.M_PHASE]] = 1.0  # SETUP 0, CHOOSE 1, PLACE 2, DECIDE 3
    o += 4
    counts = np.empty(15, np.int64)
    E = F._expected_hidden(g, counts)
    for v in range(15):
        out[o + v] = counts[v] / 10.0
    o += 15
    out[o] = m[F.M_DLEN] / 150.0
    o += 1
    for r in range(n):
        out[o + r] = totals[(me + r) % n] / 100.0
    o += n
    if m[F.M_CLOSER] >= 0:
        out[o + (m[F.M_CLOSER] - me) % n] = 1.0
    o += n
    out[o] = E / 12.0


@njit(cache=True)
def _on_step_done(g, seat, rew, b, nrounds_before, objective, shaping, stats):
    """Accumulate rewards after any step. Returns True if the game is over."""
    values, st, deck, disc, m, totals, acc = g
    n = m[F.M_NP]
    if m[F.M_NROUNDS] != nrounds_before:
        mine = acc[3, seat]
        if objective == 1:
            rew[b] += -mine / 20.0
        elif shaping != 0.0:
            s = 0.0
            for i in range(n):
                if i != seat:
                    s += acc[3, i]
            rew[b] += shaping * (s / (n - 1) - mine) / 20.0
        stats[b, 1] += mine
        stats[b, 2] += 1
    if m[F.M_PHASE] == F.GAME_OVER:
        best = totals[:n].min()
        k = 0
        for i in range(n):
            if totals[i] == best:
                k += 1
        share = 1.0 / k if totals[seat] == best else 0.0
        stats[b, 0] = share
        if objective == 0:
            rew[b] += 2.0 * share - 1.0
        return True
    return False


@njit(cache=True, parallel=True)
def _advance(V, S, D, C, M, T, A, kinds, seats, hparams, apply_mask, actions, perms,
             rew, done, ep_out, ep_n, stats, probs, n, max_rounds, end_score,
             objective, shaping, waiting, errors, max_round_turns):
    """For every game with apply_mask[b]: apply actions[b] for the player to
    move, then play heuristic opponents until a learner/net seat must act.
    Finished games are logged to ep_out and reset. waiting[b] = 0 learner to
    move, 1 net opponent to move."""
    B = V.shape[0]
    for b in prange(B):
        if not apply_mask[b]:
            continue
        g = _view(V, S, D, C, M, T, A, b)
        m = M[b]
        ws = F.new_workspace()
        a = actions[b]
        if a >= 0:  # external action: canonical slot -> real position, check legality
            a = to_real(a, perms[b])
            lm = np.empty(F.N_ACTIONS, np.bool_)
            F.legal_mask(g, lm)
            if not lm[a]:
                errors[b] = 1
                continue
        while True:
            if a >= 0:
                nr = m[F.M_NROUNDS]
                F.step(g, a)
                a = -1
                over = _on_step_done(g, seats[b], rew, b, nr, objective, shaping, stats)
                aborted = (not over) and m[F.M_NTURNS] > max_round_turns
                if over or aborted:
                    # aborted: a round that never ends (e.g. two deterministic players
                    # swapping equal cards back and forth) -- no reward, logged as aborted
                    done[b] = True
                    # log episode: win share, rounds, own round-score sum, opponent kind
                    j = ep_n[b]
                    if j < ep_out.shape[1]:
                        ep_out[b, j, 0] = stats[b, 0]
                        ep_out[b, j, 1] = stats[b, 2]
                        ep_out[b, j, 2] = stats[b, 1]
                        opp = -3
                        for s in range(n):
                            if kinds[b, s] != LEARNER:
                                opp = kinds[b, s]
                                break
                        ep_out[b, j, 3] = opp
                        ep_out[b, j, 4] = 1.0 if aborted else 0.0
                        ep_n[b] = j + 1
                    stats[b, :] = 0.0
                    _reset_game(g, n, max_rounds, end_score)
                    seats[b] = _assign_seats(kinds[b], n, probs)
            cur = m[F.M_CUR]
            k = kinds[b, cur]
            if k == LEARNER:
                waiting[b] = 0
                break
            if k == NET:
                waiting[b] = 1
                break
            a = F.heuristic_act(g, hparams[k], ws)


@njit(cache=True)
def to_real(a, perm):
    if a >= F.FLIP0:
        return F.FLIP0 + perm[a - F.FLIP0]
    if a >= F.REP0:
        return F.REP0 + perm[a - F.REP0]
    return a


@njit(cache=True)
def to_canon(a, perm):
    if a < F.REP0:
        return a
    base = F.FLIP0 if a >= F.FLIP0 else F.REP0
    p = a - base
    for k in range(N_POS):
        if perm[k] == p:
            return base + k
    return -1


@njit(cache=True, parallel=True)
def _encode_all(V, S, D, C, M, T, A, sel, obs, masks, perms, canonical):
    B = V.shape[0]
    for b in prange(B):
        if sel[b]:
            g = _view(V, S, D, C, M, T, A, b)
            encode(g, obs[b], perms[b], canonical)
            lm = np.empty(F.N_ACTIONS, np.bool_)
            F.legal_mask(g, lm)
            mk = masks[b]
            mk[F.TAKE] = lm[F.TAKE]
            mk[F.DRAW] = lm[F.DRAW]
            for k in range(N_POS):
                mk[F.REP0 + k] = lm[F.REP0 + perms[b, k]]
                mk[F.FLIP0 + k] = lm[F.FLIP0 + perms[b, k]]


@njit(cache=True, parallel=True)
def _heuristic_all(V, S, D, C, M, T, A, P, perms, out):
    """Heuristic action (params P) for the player to move, in canonical slots."""
    B = V.shape[0]
    for b in prange(B):
        g = _view(V, S, D, C, M, T, A, b)
        ws = F.new_workspace()
        out[b] = to_canon(F.heuristic_act(g, P, ws), perms[b])


@njit(cache=True)
def _seed_all(seed):
    np.random.seed(seed)


@njit(cache=True, parallel=True)
def _reset_all(V, S, D, C, M, T, A, kinds, seats, probs, n, max_rounds, end_score, seeds):
    B = V.shape[0]
    for b in prange(B):
        np.random.seed(seeds[b])
        g = _view(V, S, D, C, M, T, A, b)
        _reset_game(g, n, max_rounds, end_score)
        seats[b] = _assign_seats(kinds[b], n, probs)


# ------------------------------------------------------------------ Python API
class VecEnv:
    """B parallel games. `opponents` is a list of heuristic param dicts; `probs`
    gives the chance of each one per opponent seat, plus a final entry for a
    net (self-play) opponent. Set `net_policy` (obs, mask -> actions) before
    stepping if that probability is > 0."""

    def __init__(self, n_envs, n_players=2, opponents=(), probs=None, objective="win",
                 shaping=0.0, seed=0, end_score=100, net_policy=None, canonical=True,
                 max_round_turns=500):
        self.B, self.n = n_envs, n_players
        self.objective = 0 if objective == "win" else 1
        self.max_rounds = 0 if objective == "win" else 1
        self.end_score = end_score
        self.shaping = float(shaping)
        self.canonical = bool(canonical)
        self.max_round_turns = int(max_round_turns)
        self.obs_dim = obs_dim(n_players)
        K = len(opponents)
        self.hparams = (np.stack([F.params_vector(o) for o in opponents]) if K
                        else np.zeros((1, 13)))
        if probs is None:
            probs = [1.0 / K] * K + [0.0]
        self.set_probs(probs)
        B = n_envs
        self.V = np.zeros((B, F.MAX_P, N_POS), np.int8)
        self.S = np.zeros((B, F.MAX_P, N_POS), np.int8)
        self.D = np.zeros((B, F.DECK_SIZE), np.int8)
        self.C = np.zeros((B, F.DECK_SIZE), np.int8)
        self.M = np.zeros((B, F.M_SIZE), np.int64)
        self.T = np.zeros((B, F.MAX_P), np.int64)
        self.A = np.zeros((B, 5, F.MAX_P), np.int64)
        self.kinds = np.zeros((B, F.MAX_P), np.int64)
        self.seats = np.zeros(B, np.int64)
        self.stats = np.zeros((B, 3), np.float64)
        self.rew = np.zeros(B, np.float32)
        self.done = np.zeros(B, np.bool_)
        self.waiting = np.zeros(B, np.int64)
        self.obs = np.zeros((B, self.obs_dim), np.float32)
        self.masks = np.zeros((B, F.N_ACTIONS), np.bool_)
        self.perms = np.tile(np.arange(N_POS, dtype=np.int64), (B, 1))
        self.errors = np.zeros(B, np.int64)
        self.ep_out = np.zeros((B, 64, 5), np.float64)
        self.ep_n = np.zeros(B, np.int64)
        self.net_policy = net_policy
        self.rng = np.random.default_rng(seed)
        _seed_all(seed % 2 ** 32)
        self.reset()

    def set_probs(self, probs):
        p = np.asarray(probs, np.float64)
        assert p.shape[0] == self.hparams.shape[0] + 1 and abs(p.sum() - 1) < 1e-9, p
        self.probs = p

    def _g(self):
        return (self.V, self.S, self.D, self.C, self.M, self.T, self.A)

    def reset(self):
        seeds = self.rng.integers(0, 2 ** 31, self.B)
        _reset_all(*self._g(), self.kinds, self.seats, self.probs, self.n, self.max_rounds,
                   self.end_score, seeds)
        self.stats[:] = 0
        everyone = np.ones(self.B, np.bool_)
        self._advance(everyone, np.full(self.B, -1, np.int64))
        self._resolve_net()
        _encode_all(*self._g(), everyone, self.obs, self.masks, self.perms, self.canonical)
        return self.obs.copy(), self.masks.copy()

    def _advance(self, sel, actions):
        _advance(*self._g(), self.kinds, self.seats, self.hparams, sel, actions, self.perms,
                 self.rew, self.done, self.ep_out, self.ep_n, self.stats, self.probs,
                 self.n, self.max_rounds, self.end_score, self.objective, self.shaping,
                 self.waiting, self.errors, self.max_round_turns)
        if self.errors.any():
            bad = np.flatnonzero(self.errors)
            self.errors[:] = 0
            raise ValueError(f"illegal action in games {bad[:10]}")

    def heuristic_actions(self, params):
        """Actions of a heuristic for the player to move, in the env's (canonical) slots."""
        from .. import fast as _F
        out = np.empty(self.B, np.int64)
        _heuristic_all(*self._g(), _F.params_vector(params), self.perms, out)
        return out

    def _resolve_net(self):
        guard = 0
        while True:
            sel = self.waiting == 1
            if not sel.any():
                return
            _encode_all(*self._g(), sel, self.obs, self.masks, self.perms, self.canonical)
            idx = np.flatnonzero(sel)
            acts = np.full(self.B, -1, np.int64)
            acts[idx] = self.net_policy(self.obs[idx], self.masks[idx])
            self._advance(sel, acts)
            guard += 1
            if guard > 100_000:
                raise RuntimeError("net opponents never yielded")

    def step(self, actions):
        """actions: int array [B] for the learner. Returns obs, masks, rewards, dones."""
        actions = np.asarray(actions, np.int64)
        if not self.masks[np.arange(self.B), actions].all():
            bad = np.flatnonzero(~self.masks[np.arange(self.B), actions])
            raise ValueError(f"illegal learner actions in games {bad[:10]}")
        self.rew[:] = 0.0
        self.done[:] = False
        self._advance(np.ones(self.B, np.bool_), actions)
        self._resolve_net()
        _encode_all(*self._g(), np.ones(self.B, np.bool_), self.obs, self.masks, self.perms,
                    self.canonical)
        return self.obs.copy(), self.masks.copy(), self.rew.copy(), self.done.copy()

    def pop_episodes(self):
        """Finished episodes since last call: array [k, 5] = win share, rounds,
        own round-score sum, opponent kind (heuristic index or -2 = net),
        aborted (1 = stopped after max_round_turns in one round)."""
        out = [self.ep_out[b, :self.ep_n[b]] for b in range(self.B) if self.ep_n[b]]
        self.ep_n[:] = 0
        return np.concatenate(out) if out else np.zeros((0, 5))
