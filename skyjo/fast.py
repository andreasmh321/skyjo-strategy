"""Numba-compiled Skyjo engine + heuristic agent.

Same rules, action space and heuristic as `engine.py` / `agents.py`, but the
whole game (engine and agents) runs as compiled machine code, and
`play_many` spreads games over all cores with `prange`.

State of one game is a tuple of NumPy arrays (see `new_state`):

    values  int8 [8, 12]   card values on each board
    st      int8 [8, 12]   HIDDEN / REVEALED / REMOVED
    deck    int8 [150]     draw pile, top = deck[m[M_DLEN] - 1]
    disc    int8 [150]     discard pile, top = disc[m[M_CLEN] - 1]
    m       int64[24]      scalar fields (phase, current player, ... see M_*)
    totals  int64[8]       game totals
    acc     int64[5, 8]    0 sum of round scores, 1 closes, 2 times doubled,
                           3 last round scores, 4 last round raw scores

The Python engine stays the reference; tests/test_fast.py checks that both
engines (and both heuristic implementations) agree step for step.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from .agents import HeuristicParams

N_POS = 12
MAX_P = 8
DECK_SIZE = 150
HIDDEN, REVEALED, REMOVED = 0, 1, 2
SETUP, CHOOSE, PLACE, DECIDE, GAME_OVER = 0, 1, 2, 3, 5
TAKE, DRAW, REP0, FLIP0, N_ACTIONS = 0, 1, 2, 14, 26
NOCARD = 99

(M_NP, M_PHASE, M_CUR, M_HAND, M_DLEN, M_CLEN, M_CLOSER, M_FINAL, M_NTURNS, M_ROUND,
 M_NEXT, M_SETUP, M_MAXR, M_END, M_NRESH, M_NROUNDS, M_LAST_CLOSER, M_LAST_DOUBLED,
 M_NREVEAL) = range(19)
M_SIZE = 24

FULL_COUNTS = np.array([5, 10, 15] + [10] * 12, dtype=np.int64)  # index = value + 2

PARAM_NAMES = HeuristicParams.names()
(P_TAKE_E, P_TAKE_L, P_KEEP_E, P_KEEP_L, P_HIDDEN, P_PAIR, P_CLOSE, P_LEADCOEF,
 P_DENY, P_LOOKAHEAD, P_END_MARGIN, P_SETUP_COL, P_FINAL_GREEDY) = range(13)
assert PARAM_NAMES == [
    "take_gain_early", "take_gain_late", "keep_gain_early", "keep_gain_late",
    "hidden_bonus", "pair_bonus", "close_margin", "close_lead_coef", "deny_weight",
    "end_lookahead", "end_margin", "setup_same_col", "final_turn_greedy"], PARAM_NAMES


def params_vector(d: dict) -> np.ndarray:
    return np.array([float(d[k]) for k in PARAM_NAMES], dtype=np.float64)


# =============================================================== engine
@njit(cache=True)
def nb_seed(s):
    np.random.seed(s)


@njit(cache=True)
def new_state(n_players, max_rounds, end_score):
    values = np.zeros((MAX_P, N_POS), np.int8)
    st = np.zeros((MAX_P, N_POS), np.int8)
    deck = np.zeros(DECK_SIZE, np.int8)
    disc = np.zeros(DECK_SIZE, np.int8)
    m = np.zeros(M_SIZE, np.int64)
    totals = np.zeros(MAX_P, np.int64)
    acc = np.zeros((5, MAX_P), np.int64)
    m[M_NP] = n_players
    m[M_MAXR] = max_rounds  # 0 = unlimited
    m[M_END] = end_score
    m[M_ROUND] = -1
    m[M_NEXT] = -1
    m[M_NREVEAL] = 2
    g = (values, st, deck, disc, m, totals, acc)
    start_round(g)
    return g


@njit(cache=True)
def start_round(g):
    values, st, deck, disc, m, totals, acc = g
    m[M_ROUND] += 1
    k = 0
    for v in range(15):
        for _ in range(FULL_COUNTS[v]):
            deck[k] = v - 2
            k += 1
    np.random.shuffle(deck)
    dlen = DECK_SIZE
    for i in range(m[M_NP]):
        for p in range(N_POS):
            dlen -= 1
            values[i, p] = deck[dlen]
            st[i, p] = HIDDEN
    dlen -= 1
    disc[0] = deck[dlen]
    m[M_CLEN] = 1
    m[M_DLEN] = dlen
    m[M_HAND] = NOCARD
    m[M_CLOSER] = -1
    m[M_FINAL] = 0
    m[M_NTURNS] = 0
    m[M_NRESH] = 0
    m[M_PHASE] = SETUP
    m[M_CUR] = 0
    m[M_SETUP] = m[M_NREVEAL]


@njit(cache=True)
def _draw(g):
    values, st, deck, disc, m, totals, acc = g
    if m[M_DLEN] == 0:
        clen = m[M_CLEN]
        top = disc[clen - 1]
        for j in range(clen - 1):
            deck[j] = disc[j]
        m[M_DLEN] = clen - 1
        np.random.shuffle(deck[:clen - 1])
        disc[0] = top
        m[M_CLEN] = 1
        m[M_NRESH] += 1
    m[M_DLEN] -= 1
    return deck[m[M_DLEN]]


@njit(cache=True)
def _push_discard(g, v):
    disc, m = g[3], g[4]
    disc[m[M_CLEN]] = v
    m[M_CLEN] += 1


@njit(cache=True)
def _resolve_triples(g, i):
    values, st = g[0], g[1]
    for c in range(4):
        b = 3 * c
        if st[i, b] == REMOVED:
            continue
        if (st[i, b] == REVEALED and st[i, b + 1] == REVEALED and st[i, b + 2] == REVEALED
                and values[i, b] == values[i, b + 1] and values[i, b] == values[i, b + 2]):
            for q in range(b, b + 3):
                st[i, q] = REMOVED
                _push_discard(g, values[i, q])


@njit(cache=True)
def legal_mask(g, out):
    """Fill bool[26] `out` with the legal actions of the current player."""
    values, st, deck, disc, m, totals, acc = g
    out[:] = False
    ph, cur = m[M_PHASE], m[M_CUR]
    if ph == SETUP:
        for p in range(N_POS):
            out[FLIP0 + p] = st[cur, p] == HIDDEN
    elif ph == CHOOSE:
        out[TAKE] = True
        out[DRAW] = True
    elif ph == PLACE or ph == DECIDE:
        for p in range(N_POS):
            out[REP0 + p] = st[cur, p] != REMOVED
            if ph == DECIDE:
                out[FLIP0 + p] = st[cur, p] == HIDDEN


@njit(cache=True)
def _finish_setup(g):
    values, st, deck, disc, m, totals, acc = g
    n = m[M_NP]
    if m[M_NEXT] >= 0:
        start = m[M_NEXT]
    else:
        sums = np.zeros(n, np.int64)
        for i in range(n):
            for p in range(N_POS):
                if st[i, p] == REVEALED:
                    sums[i] += values[i, p]
        best = sums.max()
        n_best = 0
        for i in range(n):
            if sums[i] == best:
                n_best += 1
        k = np.random.randint(0, n_best)
        start = -1
        for i in range(n):
            if sums[i] == best:
                if k == 0:
                    start = i
                    break
                k -= 1
    m[M_CUR] = start
    m[M_PHASE] = CHOOSE


@njit(cache=True)
def _end_round(g):
    values, st, deck, disc, m, totals, acc = g
    n = m[M_NP]
    for i in range(n):
        for p in range(N_POS):
            if st[i, p] == HIDDEN:
                st[i, p] = REVEALED
        _resolve_triples(g, i)
    c = m[M_CLOSER]
    others_min = 1 << 30
    for i in range(n):
        raw = 0
        for p in range(N_POS):
            if st[i, p] != REMOVED:
                raw += values[i, p]
        acc[4, i] = raw
        if i != c and raw < others_min:
            others_min = raw
    doubled = acc[4, c] > 0 and others_min <= acc[4, c]
    for i in range(n):
        s = acc[4, i]
        if doubled and i == c:
            s *= 2
        acc[3, i] = s
        acc[0, i] += s
        totals[i] += s
    acc[1, c] += 1
    if doubled:
        acc[2, c] += 1
    m[M_LAST_CLOSER] = c
    m[M_LAST_DOUBLED] = 1 if doubled else 0
    m[M_NROUNDS] += 1
    m[M_NEXT] = c
    mx = totals[:n].max()
    if mx >= m[M_END] or (m[M_MAXR] > 0 and m[M_NROUNDS] >= m[M_MAXR]):
        m[M_PHASE] = GAME_OVER
    else:
        start_round(g)


@njit(cache=True)
def _end_turn(g):
    values, st, deck, disc, m, totals, acc = g
    m[M_NTURNS] += 1
    pl = m[M_CUR]
    if m[M_CLOSER] < 0:
        all_rev = True
        for p in range(N_POS):
            if st[pl, p] == HIDDEN:
                all_rev = False
                break
        if all_rev:
            m[M_CLOSER] = pl
            m[M_FINAL] = m[M_NP] - 1
    else:
        m[M_FINAL] -= 1
    if m[M_CLOSER] >= 0 and m[M_FINAL] == 0:
        _end_round(g)
        return
    m[M_CUR] = (pl + 1) % m[M_NP]
    m[M_PHASE] = CHOOSE


@njit(cache=True)
def step(g, a):
    """Apply action `a` (assumed legal; use legal_mask to check)."""
    values, st, deck, disc, m, totals, acc = g
    ph, cur = m[M_PHASE], m[M_CUR]
    if ph == SETUP:
        st[cur, a - FLIP0] = REVEALED
        m[M_SETUP] -= 1
        if m[M_SETUP] == 0:
            m[M_CUR] += 1
            m[M_SETUP] = m[M_NREVEAL]
            if m[M_CUR] == m[M_NP]:
                _finish_setup(g)
        return
    if ph == CHOOSE:
        if a == TAKE:
            m[M_CLEN] -= 1
            m[M_HAND] = disc[m[M_CLEN]]
            m[M_PHASE] = PLACE
        else:
            m[M_HAND] = _draw(g)
            m[M_PHASE] = DECIDE
        return
    if a < FLIP0:  # replace
        p = a - REP0
        old = values[cur, p]
        values[cur, p] = m[M_HAND]
        st[cur, p] = REVEALED
        _push_discard(g, old)
    else:  # flip
        p = a - FLIP0
        _push_discard(g, m[M_HAND])
        st[cur, p] = REVEALED
    m[M_HAND] = NOCARD
    _resolve_triples(g, cur)
    _end_turn(g)


# ============================================================ heuristic
@njit(cache=True)
def new_workspace():
    """Scratch arrays reused by the heuristic (avoids allocations per call)."""
    return (np.empty(N_POS, np.float64), np.empty(N_POS, np.int64),
            np.empty(N_POS, np.bool_), np.empty(N_POS, np.bool_),
            np.empty(N_POS, np.float64), np.empty(N_POS, np.int64),
            np.empty(15, np.int64), np.empty(15, np.float64))
    # 0 gs, 1 qs, 2 pair, 3 hid, 4 gs2, 5 qs2, 6 counts, 7 opp-gain cache


@njit(cache=True)
def _expected_hidden(g, counts):
    """Fill counts[15] with unseen cards (current player's view); return mean."""
    values, st, deck, disc, m, totals, acc = g
    for v in range(15):
        counts[v] = FULL_COUNTS[v]
    for i in range(m[M_NP]):
        for p in range(N_POS):
            if st[i, p] == REVEALED:
                counts[values[i, p] + 2] -= 1
    for j in range(m[M_CLEN]):
        counts[disc[j] + 2] -= 1
    if m[M_HAND] != NOCARD:
        counts[m[M_HAND] + 2] -= 1
    num = 0
    n = 0
    for v in range(15):
        num += (v - 2) * counts[v]
        n += counts[v]
    return num / n if n > 0 else 0.0


@njit(cache=True)
def _col_gains(g, pl, E, card, og, oq, opair, ohid):
    """Port of agents.column_value_gains. Returns number of entries."""
    values, st = g[0], g[1]
    k = 0
    for c in range(4):
        b = 3 * c
        if st[pl, b] == REMOVED:
            continue
        v0 = E if st[pl, b] == HIDDEN else float(values[pl, b])
        v1 = E if st[pl, b + 1] == HIDDEN else float(values[pl, b + 1])
        v2 = E if st[pl, b + 2] == HIDDEN else float(values[pl, b + 2])
        col_sum = v0 + v1 + v2
        for i in range(3):
            q = b + i
            vi = v0 if i == 0 else (v1 if i == 1 else v2)
            o1 = b + (i + 1) % 3
            o2 = b + (i + 2) % 3
            r1 = st[pl, o1] == REVEALED
            r2 = st[pl, o2] == REVEALED
            e1 = r1 and values[pl, o1] == card
            e2 = r2 and values[pl, o2] == card
            if e1 and e2:
                og[k] = col_sum
                opair[k] = False
            else:
                og[k] = vi - card
                same_here = st[pl, q] == REVEALED and values[pl, q] == card
                opair[k] = (e1 != e2) and not same_here
            oq[k] = q
            ohid[k] = st[pl, q] == HIDDEN
            k += 1
    return k


@njit(cache=True)
def _sort_desc(gs, qs, k):
    """Sort (g, q) pairs descending like Python's sort(reverse=True)."""
    for i in range(1, k):
        gv, qv = gs[i], qs[i]
        j = i - 1
        while j >= 0 and (gs[j] < gv or (gs[j] == gv and qs[j] < qv)):
            gs[j + 1] = gs[j]
            qs[j + 1] = qs[j]
            j -= 1
        gs[j + 1] = gv
        qs[j + 1] = qv


@njit(cache=True)
def _scored(g, me, P, E, card, final_greedy, gs, qs, ws):
    opair, ohid = ws[2], ws[3]
    k = _col_gains(g, me, E, card, gs, qs, opair, ohid)
    if not final_greedy:
        pb = P[P_PAIR] * max(0.0, 1.0 - max(card, 0) / 12.0)
        for j in range(k):
            if ohid[j]:
                gs[j] += P[P_HIDDEN]
            if opair[j]:
                gs[j] += pb
    _sort_desc(gs, qs, k)
    return k


@njit(cache=True)
def _max_gain(g, pl, E, card, ws):
    gs, qs, opair, ohid = ws[4], ws[5], ws[2], ws[3]
    k = _col_gains(g, pl, E, card, gs, qs, opair, ohid)
    mx = -np.inf
    for j in range(k):
        if gs[j] > mx:
            mx = gs[j]
    return mx, k


@njit(cache=True)
def _opp_gain(g, me, E, card, ws):
    m = g[4]
    nxt = (me + 1) % m[M_NP]
    if m[M_CLOSER] >= 0 and nxt == m[M_CLOSER]:
        return 0.0
    cache = ws[7]
    if not np.isnan(cache[card + 2]):
        return cache[card + 2]
    mx, k = _max_gain(g, nxt, E, card, ws)
    if k == 0:
        mx = 0.0
    r = max(0.0, mx)
    cache[card + 2] = r
    return r


@njit(cache=True)
def _progress(g):
    st, m = g[1], g[4]
    best = 0.0
    for i in range(m[M_NP]):
        act = 0
        hid = 0
        for p in range(N_POS):
            if st[i, p] != REMOVED:
                act += 1
            if st[i, p] == HIDDEN:
                hid += 1
        if act:
            best = max(best, 1.0 - hid / act)
    return best


@njit(cache=True)
def _final_est(g, me, E, card, pos):
    """Own estimated round score if `card` replaces `pos` (NOCARD = flip pos)."""
    values, st = g[0], g[1]
    total = 0.0
    for c in range(4):
        b = 3 * c
        if st[me, b] == REMOVED:
            continue
        exact = True
        s = 0.0
        first = 0.0
        same = True
        for i in range(3):
            q = b + i
            if q == pos and card != NOCARD:
                v = float(card)
            elif st[me, q] == HIDDEN:
                v = E
                exact = False
            else:
                v = float(values[me, q])
            if i == 0:
                first = v
            elif v != first:
                same = False
            s += v
        if exact and same:
            continue
        total += s
    return total


@njit(cache=True)
def _est_score(g, i, E):
    values, st = g[0], g[1]
    vs = 0
    nh = 0
    for p in range(N_POS):
        if st[i, p] == REVEALED:
            vs += values[i, p]
        elif st[i, p] == HIDDEN:
            nh += 1
    return vs + nh * E


@njit(cache=True)
def _closing_ok(g, me, P, E, my_est):
    m, totals = g[4], g[5]
    n = m[M_NP]
    min_opp = np.inf
    for i in range(n):
        if i != me:
            min_opp = min(min_opp, _est_score(g, i, E))
    if P[P_LOOKAHEAD] > 0.5:
        doubled = my_est > 0 and min_opp <= my_est
        my_tot = totals[me] + my_est * (2 if doubled else 1)
        mx = my_tot
        min_tot = np.inf
        for i in range(n):
            if i != me:
                ot = totals[i] + _est_score(g, i, E)
                mx = max(mx, ot)
                min_tot = min(min_tot, ot)
        if mx >= m[M_END]:
            return my_tot < min_tot - P[P_END_MARGIN]
    if my_est <= 0:
        return True
    min_tot_i = 1 << 60
    for i in range(n):
        if i != me and totals[i] < min_tot_i:
            min_tot_i = totals[i]
    lead = totals[me] - min_tot_i
    margin = P[P_CLOSE] + P[P_LEADCOEF] * lead / 10.0
    return my_est < min_opp - margin


@njit(cache=True)
def _closes(g, me, pos):
    st, m = g[1], g[4]
    if m[M_CLOSER] >= 0 or st[me, pos] != HIDDEN:
        return False
    nh = 0
    for p in range(N_POS):
        if st[me, p] == HIDDEN:
            nh += 1
    return nh == 1


@njit(cache=True)
def _best_safe(g, me, P, E, card, deny, final_greedy, ws):
    """Returns (found, gain, pos)."""
    values, st = g[0], g[1]
    gs, qs = ws[0], ws[1]
    k = _scored(g, me, P, E, card, final_greedy, gs, qs, ws)
    dw = P[P_DENY] if deny else 0.0
    found = False
    bg = 0.0
    bq = -1
    for j in range(k):
        gv = gs[j]
        q = qs[j]
        if dw != 0.0:
            if st[me, q] == REVEALED:
                gv -= dw * _opp_gain(g, me, E, values[me, q], ws)
        if _closes(g, me, q) and not _closing_ok(g, me, P, E, _final_est(g, me, E, card, q)):
            continue
        if not found or gv > bg:
            found = True
            bg = gv
            bq = q
    return found, bg, bq


@njit(cache=True)
def _choose_flip(g, me):
    st = g[1]
    best = -1
    for q in range(N_POS):
        if st[me, q] == HIDDEN:
            b = 3 * (q // 3)
            key = (st[me, b] == HIDDEN) + (st[me, b + 1] == HIDDEN) + (st[me, b + 2] == HIDDEN)
            if key > best:
                best = key
    n = 0
    for q in range(N_POS):
        if st[me, q] == HIDDEN:
            b = 3 * (q // 3)
            if (st[me, b] == HIDDEN) + (st[me, b + 1] == HIDDEN) + (st[me, b + 2] == HIDDEN) == best:
                n += 1
    k = np.random.randint(0, n)
    for q in range(N_POS):
        if st[me, q] == HIDDEN:
            b = 3 * (q // 3)
            if (st[me, b] == HIDDEN) + (st[me, b + 1] == HIDDEN) + (st[me, b + 2] == HIDDEN) == best:
                if k == 0:
                    return q
                k -= 1
    return -1


@njit(cache=True)
def _random_hidden(g, me):
    st = g[1]
    n = 0
    for q in range(N_POS):
        if st[me, q] == HIDDEN:
            n += 1
    k = np.random.randint(0, n)
    for q in range(N_POS):
        if st[me, q] == HIDDEN:
            if k == 0:
                return q
            k -= 1
    return -1


@njit(cache=True)
def heuristic_act(g, P, ws):
    """Port of HeuristicAgent.act for the current player. `ws` = new_workspace()."""
    values, st, deck, disc, m, totals, acc = g
    me = m[M_CUR]
    ph = m[M_PHASE]
    if ph == SETUP:
        if P[P_SETUP_COL] > 0.5:
            n_hid = 0
            for c in range(4):
                b = 3 * c
                nh = (st[me, b] == HIDDEN) + (st[me, b + 1] == HIDDEN) + (st[me, b + 2] == HIDDEN)
                n_hid += nh
                if nh == 2:
                    for q in range(b, b + 3):
                        if st[me, q] == HIDDEN:
                            return FLIP0 + q
            if n_hid == N_POS:
                return FLIP0 + 3 * np.random.randint(0, 4)
        return FLIP0 + _random_hidden(g, me)

    counts = ws[6]
    ws[7][:] = np.nan
    E = _expected_hidden(g, counts)
    is_final = m[M_CLOSER] >= 0
    greedy = is_final and P[P_FINAL_GREEDY] > 0.5
    t = _progress(g)
    take_min = 0.0 if greedy else P[P_TAKE_E] + t * (P[P_TAKE_L] - P[P_TAKE_E])
    keep_min = 0.0 if greedy else P[P_KEEP_E] + t * (P[P_KEEP_L] - P[P_KEEP_E])
    deny = not greedy
    hand = m[M_HAND]

    if ph == CHOOSE:
        top = disc[m[M_CLEN] - 1]
        if greedy:
            mx, k = _max_gain(g, me, E, top, ws)
            take = max(0.0, mx)
            n = 0
            ev = 0.0
            for v in range(15):
                n += counts[v]
                if counts[v]:
                    mx, k = _max_gain(g, me, E, v - 2, ws)
                    ev += counts[v] * max(0.0, mx)
            ev /= max(n, 1)
            return TAKE if take > ev else DRAW
        found, bg, bq = _best_safe(g, me, P, E, top, deny, greedy, ws)
        if found and bg >= take_min:
            return TAKE
        return DRAW

    if ph == PLACE:
        found, bg, bq = _best_safe(g, me, P, E, hand, deny, greedy, ws)
        if not found:
            gs, qs = ws[0], ws[1]
            _scored(g, me, P, E, hand, greedy, gs, qs, ws)
            bq = qs[0]
        return REP0 + bq

    # DECIDE
    found, bg, bq = _best_safe(g, me, P, E, hand, deny, greedy, ws)
    pen = P[P_DENY] * _opp_gain(g, me, E, hand, ws) if deny else 0.0
    if found and bg >= keep_min - pen:
        return REP0 + bq
    q = _choose_flip(g, me)
    if _closes(g, me, q) and not _closing_ok(g, me, P, E, _final_est(g, me, E, NOCARD, q)):
        gs, qs = ws[0], ws[1]
        k = _scored(g, me, P, E, hand, greedy, gs, qs, ws)
        for j in range(k):
            if st[me, qs[j]] != HIDDEN:
                return REP0 + qs[j]
    return FLIP0 + q


@njit(cache=True)
def random_act(g):
    mask = np.empty(N_ACTIONS, np.bool_)
    legal_mask(g, mask)
    n = 0
    for a in range(N_ACTIONS):
        n += mask[a]
    k = np.random.randint(0, n)
    for a in range(N_ACTIONS):
        if mask[a]:
            if k == 0:
                return a
            k -= 1
    return -1


# ============================================================ simulation
@njit(cache=True)
def play_one(seat_params, n_players, seed, max_rounds, end_score):
    """Play one full game; seat_params[i] = parameter vector for seat i.
    Returns the final state tuple."""
    np.random.seed(seed)
    g = new_state(n_players, max_rounds, end_score)
    ws = new_workspace()
    m = g[4]
    guard = 0
    while m[M_PHASE] != GAME_OVER:
        step(g, heuristic_act(g, seat_params[m[M_CUR]], ws))
        guard += 1
        if guard > 1_000_000:
            raise RuntimeError("game did not terminate")
    return g


@njit(cache=True, parallel=True)
def play_many(cand, opps, seeds, max_rounds, end_score):
    """Candidate `cand` (params vector) against opponents `opps` (k x 13).
    The candidate's seat rotates with the game index. Returns per game:
    out[:, 0] win share, 1 sum of round scores, 2 rounds, 3 closes, 4 doubled."""
    n = opps.shape[0] + 1
    G = seeds.shape[0]
    out = np.zeros((G, 5), np.float64)
    for gi in prange(G):
        seat = gi % n
        seat_params = np.empty((n, cand.shape[0]), np.float64)
        j = 0
        for s in range(n):
            if s == seat:
                seat_params[s] = cand
            else:
                seat_params[s] = opps[j]
                j += 1
        g = play_one(seat_params, n, seeds[gi], max_rounds, end_score)
        totals, acc, m = g[5], g[6], g[4]
        best = totals[:n].min()
        n_best = 0
        for s in range(n):
            if totals[s] == best:
                n_best += 1
        if totals[seat] == best:
            out[gi, 0] = 1.0 / n_best
        out[gi, 1] = acc[0, seat]
        out[gi, 2] = m[M_NROUNDS]
        out[gi, 3] = acc[1, seat]
        out[gi, 4] = acc[2, seat]
    return out


# ============================================================ Python API
class FastGame:
    """Thin Python wrapper, mainly for tests, debugging and RL environments."""

    def __init__(self, n_players=2, seed=None, max_rounds=None, end_score=100):
        if seed is not None:
            nb_seed(seed % (2 ** 32))
        self.n_players = n_players
        self.g = new_state(n_players, max_rounds or 0, end_score)
        self._mask = np.empty(N_ACTIONS, np.bool_)
        self._ws = new_workspace()

    @property
    def m(self):
        return self.g[4]

    @property
    def phase(self):
        return int(self.m[M_PHASE])

    @property
    def current(self):
        return int(self.m[M_CUR])

    @property
    def done(self):
        return self.phase == GAME_OVER

    def legal_actions(self):
        legal_mask(self.g, self._mask)
        return [int(a) for a in np.flatnonzero(self._mask)]

    def step(self, a, check=True):
        if check and a not in self.legal_actions():
            raise ValueError(f"illegal action {a} in phase {self.phase}")
        step(self.g, a)

    def heuristic_action(self, params):
        return int(heuristic_act(self.g, np.asarray(params, np.float64), self._ws))


class FastEvaluator:
    """Drop-in replacement for parallel.Evaluator using compiled games."""

    def __init__(self, processes=None):
        import numba
        if processes:
            numba.set_num_threads(min(processes, numba.config.NUMBA_NUM_THREADS))
        self.processes = numba.get_num_threads()

    def close(self):
        pass

    def evaluate(self, candidates, opponents, seeds, max_rounds=None, chunk=None,
                 end_score=100):
        opps = np.stack([params_vector(o) for o in opponents])
        sd = np.array([s % (2 ** 32) for s in seeds], dtype=np.int64)
        res = []
        for c in candidates:
            out = play_many(params_vector(c), opps, sd, max_rounds or 0, end_score)
            res.append({"win_rate": float(out[:, 0].mean()),
                        "mean_round": float(out[:, 1].sum() / out[:, 2].sum()),
                        "games": len(seeds),
                        "closes": int(out[:, 3].sum()), "doubled": int(out[:, 4].sum())})
        return res
