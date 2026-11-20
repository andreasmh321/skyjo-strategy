"""Scripted Skyjo agents.

Every agent implements `act(obs) -> action` where `obs` is an
`engine.Observation` and the action is an int from `obs.legal`.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, asdict, fields
from typing import Optional

from .engine import (
    A_DRAW_DECK, A_TAKE_DISCARD, CHOOSE_SOURCE, DECIDE_DRAWN, N_COLS, N_POS,
    PLACE_TAKEN, SETUP, Observation, flip_action, replace_action,
)

COLS = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]


class Agent:
    name = "agent"

    def act(self, obs: Observation) -> int:  # pragma: no cover - interface
        raise NotImplementedError


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> int:
        return self.rng.choice(obs.legal)


@dataclass
class HeuristicParams:
    """Tunable knobs. Gains are in expected points saved this round.

    Thresholds are interpolated between `*_early` and `*_late` by round
    progress (fraction of cards revealed by the most advanced player).
    """
    take_gain_early: float = 1.5   # take the discard if best placement saves >= this
    take_gain_late: float = 1.5
    keep_gain_early: float = 1.0   # keep a drawn card if best placement saves >= this
    keep_gain_late: float = 1.0
    hidden_bonus: float = 0.0      # extra value for replacing a hidden card (tempo/info)
    pair_bonus: float = 1.5        # bonus for making a revealed pair in a column
    close_margin: float = 2.0      # close only if my est < every opponent est - margin
    close_lead_coef: float = 0.0   # margin += coef * (my_total - best_opp_total) / 10
    deny_weight: float = 0.0       # penalty * next player's gain from a card we discard
    end_lookahead: float = 1.0     # >0.5: if closing would end the game, close iff we'd win
    end_margin: float = 0.0        # required lead in final totals when lookahead closes
    setup_same_col: float = 1.0    # >0.5: reveal both initial cards in one column
    final_turn_greedy: float = 1.0  # >0.5: last turn = exact expected-score minimisation

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @classmethod
    def from_vector(cls, x, names=None) -> "HeuristicParams":
        names = names or cls.names()
        return cls(**dict(zip(names, map(float, x))))


def column_value_gains(vis, hid, rem, E, card):
    """Expected-score reduction for placing `card` at each active position.

    Returns list of (gain, pos, made_pair, was_hidden). Triples that would be
    completed count the whole column as removed.
    """
    out = []
    for ps in COLS:
        if rem[ps[0]]:
            continue
        vals = [E if hid[q] else vis[q] for q in ps]
        col_sum = vals[0] + vals[1] + vals[2]
        for i, q in enumerate(ps):
            o1, o2 = ps[(i + 1) % 3], ps[(i + 2) % 3]
            v1, v2 = vis[o1], vis[o2]  # None if hidden
            if v1 is not None and v2 is not None and v1 == card and v2 == card:
                gain = col_sum
                pair = False
            else:
                gain = vals[i] - card
                pair = ((v1 == card) != (v2 == card)) and vis[q] != card
            out.append((gain, q, pair, hid[q]))
    return out


class HeuristicAgent(Agent):
    """Expected-value placement heuristic with closing / score-situation logic."""

    def __init__(self, params: Optional[HeuristicParams] = None, name: str = "heuristic",
                 seed: Optional[int] = None):
        self.p = params or HeuristicParams()
        self.name = name
        self.rng = random.Random(seed)

    # ------------------------------------------------------------ helpers
    def _scored(self, obs: Observation, card: int):
        me, p = obs.me, self.p
        res = []
        if obs.is_final_turn and p.final_turn_greedy > 0.5:
            res = [(g, q) for g, q, *_ in column_value_gains(
                obs.visible[me], obs.hidden[me], obs.removed[me], obs.expected_hidden, card)]
            res.sort(reverse=True)
            return res
        pb = p.pair_bonus * max(0.0, 1.0 - max(card, 0) / 12.0)
        for gain, q, pair, was_hidden in column_value_gains(
                obs.visible[me], obs.hidden[me], obs.removed[me], obs.expected_hidden, card):
            g = gain
            if was_hidden:
                g += p.hidden_bonus
            if pair:
                g += pb
            res.append((g, q))
        res.sort(reverse=True)
        return res

    def _opp_gain(self, obs: Observation, card: int) -> float:
        """Best raw gain the next player could get from `card` if we discard it."""
        nxt = (obs.me + 1) % obs.n_players
        if obs.is_final_turn and nxt == obs.closer:
            return 0.0
        key = (nxt, card)
        if key not in self._opp_cache:
            gs = column_value_gains(obs.visible[nxt], obs.hidden[nxt], obs.removed[nxt],
                                    obs.expected_hidden, card)
            self._opp_cache[key] = max((g for g, *_ in gs), default=0.0)
        return max(0.0, self._opp_cache[key])

    def _progress(self, obs: Observation) -> float:
        best = 0.0
        for i in range(obs.n_players):
            act = N_POS - sum(obs.removed[i])
            if act:
                best = max(best, 1.0 - sum(obs.hidden[i]) / act)
        return best

    def _final_est(self, obs: Observation, card: Optional[int], pos: int) -> float:
        """My estimated round score if `card` replaces `pos` (card None = flip pos)."""
        me, E = obs.me, obs.expected_hidden
        vis, hid, rem = obs.visible[me], obs.hidden[me], obs.removed[me]
        total = 0.0
        for ps in COLS:
            if rem[ps[0]]:
                continue
            vals = []
            exact = True
            for q in ps:
                if q == pos and card is not None:
                    vals.append(card)
                elif hid[q]:
                    vals.append(E)
                    exact = False
                else:
                    vals.append(vis[q])
            if exact and vals[0] == vals[1] == vals[2]:
                continue
            total += sum(vals)
        return total

    def _closing_ok(self, obs: Observation, my_est: float) -> bool:
        p = self.p
        others = [i for i in range(obs.n_players) if i != obs.me]
        opp_est = {i: obs.estimated_score(i) for i in others}
        min_opp = min(opp_est.values())
        if p.end_lookahead > 0.5:
            doubled = my_est > 0 and min_opp <= my_est
            my_tot = obs.totals[obs.me] + my_est * (2 if doubled else 1)
            opp_tot = [obs.totals[i] + opp_est[i] for i in others]
            if max([my_tot] + opp_tot) >= obs.end_score:
                return my_tot < min(opp_tot) - p.end_margin
        if my_est <= 0:
            return True
        lead = obs.totals[obs.me] - min(obs.totals[i] for i in others)
        margin = p.close_margin + p.close_lead_coef * lead / 10.0
        return my_est < min_opp - margin

    def _closes(self, obs: Observation, pos: int) -> bool:
        me = obs.me
        return (not obs.is_final_turn) and obs.hidden[me][pos] and obs.n_hidden(me) == 1

    def _best_safe(self, obs: Observation, card: int, deny: bool):
        """Best placement (value incl. denial penalty for the card we discard)."""
        me = obs.me
        dw = self.p.deny_weight if deny else 0.0
        best = None
        for g, q in self._scored(obs, card):
            if dw:
                old = obs.visible[me][q]
                # a hidden card we discard is unknown to us -- approximate with E
                g -= dw * (self._opp_gain(obs, old) if old is not None else 0.0)
            if self._closes(obs, q) and not self._closing_ok(obs, self._final_est(obs, card, q)):
                continue
            if best is None or g > best[0]:
                best = (g, q)
        return best

    def _choose_flip(self, obs: Observation) -> int:
        me = obs.me
        hid = obs.hidden[me]
        hidden = [q for q in range(N_POS) if hid[q]]
        def key(q):
            c = COLS[q // 3]
            return hid[c[0]] + hid[c[1]] + hid[c[2]]
        best = max(key(q) for q in hidden)
        return self.rng.choice([q for q in hidden if key(q) == best])

    # ----------------------------------------------------------------- policy
    def act(self, obs: Observation) -> int:
        p = self.p
        me = obs.me
        self._opp_cache = {}
        if obs.phase == SETUP:
            if p.setup_same_col > 0.5:
                hid = obs.hidden[me]
                for ps in COLS:
                    n = sum(hid[q] for q in ps)
                    if n == 2:  # finish a column we started
                        return flip_action(next(q for q in ps if hid[q]))
                return flip_action(self.rng.choice(COLS)[0]) if sum(hid) == N_POS \
                    else self.rng.choice(obs.legal)
            return self.rng.choice(obs.legal)

        greedy = obs.is_final_turn and p.final_turn_greedy > 0.5
        t = self._progress(obs)
        take_min = 0.0 if greedy else p.take_gain_early + t * (p.take_gain_late - p.take_gain_early)
        keep_min = 0.0 if greedy else p.keep_gain_early + t * (p.keep_gain_late - p.keep_gain_early)
        deny = not greedy

        if obs.phase == CHOOSE_SOURCE:
            if greedy:  # last turn: exact one-step expected-value comparison
                take = max(0.0, max(g for g, *_ in column_value_gains(
                    obs.visible[me], obs.hidden[me], obs.removed[me],
                    obs.expected_hidden, obs.discard_top)))
                n = sum(obs.unseen_list)
                ev = 0.0
                for i, cnt in enumerate(obs.unseen_list):
                    if cnt:
                        ev += cnt * max(0.0, max(g for g, *_ in column_value_gains(
                            obs.visible[me], obs.hidden[me], obs.removed[me],
                            obs.expected_hidden, i - 2)))
                ev /= max(n, 1)
                return A_TAKE_DISCARD if take > ev else A_DRAW_DECK
            best = self._best_safe(obs, obs.discard_top, deny)
            if best is not None and best[0] >= take_min:
                return A_TAKE_DISCARD
            return A_DRAW_DECK

        if obs.phase == PLACE_TAKEN:
            best = self._best_safe(obs, obs.hand, deny)
            if best is None:
                best = self._scored(obs, obs.hand)[0]
            return replace_action(best[1])

        if obs.phase == DECIDE_DRAWN:
            best = self._best_safe(obs, obs.hand, deny)
            discard_pen = p.deny_weight * self._opp_gain(obs, obs.hand) if deny else 0.0
            if best is not None and best[0] >= keep_min - discard_pen:
                return replace_action(best[1])
            q = self._choose_flip(obs)
            if self._closes(obs, q) and not self._closing_ok(obs, self._final_est(obs, None, q)):
                revealed = [(g, r) for g, r in self._scored(obs, obs.hand)
                            if not obs.hidden[me][r]]
                if revealed:
                    return replace_action(revealed[0][1])
            return flip_action(q)

        raise RuntimeError(f"unexpected phase {obs.phase}")


def greedy_agent(seed=None) -> HeuristicAgent:
    """No thresholds, no pair bonus, closes whenever it can."""
    return HeuristicAgent(HeuristicParams(take_gain_early=0, take_gain_late=0,
                                          keep_gain_early=0, keep_gain_late=0,
                                          pair_bonus=0.0, close_margin=-1e9),
                          name="greedy", seed=seed)
