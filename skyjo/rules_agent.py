"""The simplified 'table rules' distilled from the RL agent, as a bot.

1. Take the discard if it's <=1, or 2 late in the round (<=2 hidden cards), or if it
   completes a column.
2. Keep a drawn card if it's <=3 (or completes a column): put it on your highest card
   if that's 7+, otherwise on a hidden spot. Throw away anything higher.
3. Reveal your first two cards in different columns.
4. End the round when about 7+ ahead with no card above 5 showing, or 15+ ahead
   regardless; when 20+ behind in the game, 3+ ahead is enough.
5. With one hidden card left and the round not worth ending, don't reveal it: put the
   card on a revealed spot instead (stall), even if it costs a few points.
On the last turn (after someone else ended the round) just minimise your own score.
"""
import random

from .agents import Agent
from .engine import (A_DRAW_DECK, A_TAKE_DISCARD, CHOOSE_SOURCE, DECIDE_DRAWN, N_POS,
                     PLACE_TAKEN, SETUP, flip_action, replace_action)


class RuleAgent(Agent):
    name = "table_rules"

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    # ---------------------------------------------------------------- helpers
    def _triple_pos(self, obs, card):
        me = obs.me
        for c in range(4):
            ps = (3 * c, 3 * c + 1, 3 * c + 2)
            if obs.removed[me][ps[0]]:
                continue
            for p in ps:
                others = [q for q in ps if q != p]
                if all(obs.visible[me][q] == card for q in others) and obs.visible[me][p] != card:
                    return p
        return None

    def _highest_revealed(self, obs):
        me = obs.me
        best = None
        for p in range(N_POS):
            v = obs.visible[me][p]
            if v is not None and (best is None or v > obs.visible[me][best]):
                best = p
        return best

    def _hidden(self, obs):
        return [p for p in range(N_POS) if obs.hidden[obs.me][p]]

    def _ok_to_end(self, obs, my_est):
        opp = min(obs.estimated_score(i) for i in range(obs.n_players) if i != obs.me)
        lead = opp - my_est
        me = obs.me
        max_vis = max((v for v in obs.visible[me] if v is not None), default=0)
        game_diff = obs.totals[me] - min(obs.totals[i] for i in range(obs.n_players) if i != me)
        if game_diff >= 20 and lead >= 3:
            return True
        return (lead >= 7 and max_vis <= 5) or lead >= 15

    def _gain(self, obs, card, p):
        v = obs.visible[obs.me][p]
        return (obs.expected_hidden if v is None else v) - card

    # ------------------------------------------------------------- last turn
    def _final(self, obs):
        me = obs.me
        spots = [p for p in range(N_POS) if not obs.removed[me][p]]
        def best(card):
            t = self._triple_pos(obs, card)
            if t is not None:
                return t, 99.0
            p = max(spots, key=lambda q: self._gain(obs, card, q))
            return p, self._gain(obs, card, p)
        if obs.phase == CHOOSE_SOURCE:
            return A_TAKE_DISCARD if best(obs.discard_top)[1] >= 3 else A_DRAW_DECK
        p, g = best(obs.hand)
        if obs.phase == PLACE_TAKEN or g > 0 or not self._hidden(obs):
            return replace_action(p)
        return flip_action(self.rng.choice(self._hidden(obs)))

    # ----------------------------------------------------------------- policy
    def act(self, obs):
        me = obs.me
        if obs.phase == SETUP:
            hid = self._hidden(obs)
            done_cols = {p // 3 for p in range(N_POS) if not obs.hidden[me][p]}
            choices = [p for p in hid if p // 3 not in done_cols] or hid
            return flip_action(self.rng.choice(choices))
        if obs.is_final_turn:
            return self._final(obs)

        n_hid = len(self._hidden(obs))
        if obs.phase == CHOOSE_SOURCE:
            c = obs.discard_top
            if c <= 1 or (c == 2 and n_hid <= 2) or self._triple_pos(obs, c) is not None:
                return A_TAKE_DISCARD
            return A_DRAW_DECK

        card = obs.hand
        keep = obs.phase == PLACE_TAKEN or card <= 3 or self._triple_pos(obs, card) is not None
        if keep:
            t = self._triple_pos(obs, card)
            hi = self._highest_revealed(obs)
            if t is not None:
                target = t
            elif hi is not None and obs.visible[me][hi] >= 7 and obs.visible[me][hi] > card:
                target = hi
            elif n_hid:
                target = self.rng.choice(self._hidden(obs))
            else:
                target = hi
            if obs.hidden[me][target] and n_hid == 1:  # this would end the round
                if not self._ok_to_end(obs, obs.estimated_score(me) - obs.expected_hidden + card):
                    target = hi if hi is not None else target
            return replace_action(target)

        # throw the drawn card away and reveal a hidden card -- unless that ends the round
        if n_hid == 1:
            p = self._hidden(obs)[0]
            v_after = obs.estimated_score(me)  # hidden card stays an unknown average
            if not self._ok_to_end(obs, v_after):
                hi = self._highest_revealed(obs)
                if hi is not None:
                    return replace_action(hi)  # stall: cheapest place for the card
            return flip_action(p)
        return flip_action(self.rng.choice(self._hidden(obs)))
