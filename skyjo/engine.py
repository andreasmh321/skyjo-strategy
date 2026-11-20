"""Skyjo rules engine.

Implements the rules from the Danish rulebook (Magilano) plus the official
defaults for things the rulebook does not spell out:

* Deck: 150 cards -- 5x(-2), 10x(-1), 15x(0), 10x each of 1..12.
* When the draw pile is empty, the discard pile except its top card is
  shuffled into a new draw pile.
* Ties for the starting player (highest sum of the two revealed cards) are
  broken at random.

Board layout: 4 columns x 3 rows. Position index p = col * 3 + row, so the
column of p is p // 3 and column c holds positions 3c, 3c+1, 3c+2.

The engine is a step-based state machine with a flat action space so it can be
driven both by hand-written agents and by RL (with action masks):

    0            take the top card of the discard pile
    1            draw the top card of the draw pile
    2  .. 13     REPLACE position p (= action - 2) with the card in hand
    14 .. 25     FLIP position p (= action - 14): during setup this reveals a
                 card; after drawing from the deck it means "discard the drawn
                 card and reveal hidden card p"
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

N_ROWS, N_COLS = 3, 4
N_POS = N_ROWS * N_COLS
CARD_VALUES = tuple(range(-2, 13))  # -2 .. 12
DECK_COUNTS = {-2: 5, -1: 10, 0: 15, **{v: 10 for v in range(1, 13)}}
DECK_SIZE = sum(DECK_COUNTS.values())  # 150

# Actions
A_TAKE_DISCARD = 0
A_DRAW_DECK = 1
A_REPLACE0 = 2
A_FLIP0 = 14
N_ACTIONS = 26


def replace_action(p: int) -> int:
    return A_REPLACE0 + p


def flip_action(p: int) -> int:
    return A_FLIP0 + p


def decode(action: int) -> tuple[str, Optional[int]]:
    if action == A_TAKE_DISCARD:
        return "take_discard", None
    if action == A_DRAW_DECK:
        return "draw_deck", None
    if A_REPLACE0 <= action < A_FLIP0:
        return "replace", action - A_REPLACE0
    if A_FLIP0 <= action < N_ACTIONS:
        return "flip", action - A_FLIP0
    raise ValueError(f"bad action {action}")


def column_positions(c: int) -> tuple[int, int, int]:
    return (3 * c, 3 * c + 1, 3 * c + 2)


# Phases
SETUP = "setup"                # current player reveals initial cards
CHOOSE_SOURCE = "choose_source"  # take discard or draw from deck
PLACE_TAKEN = "place_taken"    # holding the discard card: must replace a position
DECIDE_DRAWN = "decide_drawn"  # holding a deck card: replace a position, or discard + flip
ROUND_OVER = "round_over"
GAME_OVER = "game_over"


@dataclass
class Board:
    values: list[int]
    revealed: list[bool] = field(default_factory=lambda: [False] * N_POS)
    removed: list[bool] = field(default_factory=lambda: [False] * N_POS)

    def hidden_positions(self) -> list[int]:
        return [p for p in range(N_POS) if not self.revealed[p] and not self.removed[p]]

    def active_positions(self) -> list[int]:
        return [p for p in range(N_POS) if not self.removed[p]]

    def all_revealed(self) -> bool:
        return all(self.revealed[p] or self.removed[p] for p in range(N_POS))

    def visible(self, p: int) -> Optional[int]:
        """Value if revealed and still on the board, else None."""
        if self.removed[p] or not self.revealed[p]:
            return None
        return self.values[p]

    def visible_sum(self) -> int:
        return sum(self.values[p] for p in range(N_POS) if self.revealed[p] and not self.removed[p])

    def score(self) -> int:
        """Sum of all cards still on the board (hidden ones included)."""
        return sum(self.values[p] for p in range(N_POS) if not self.removed[p])

    def find_triples(self) -> list[int]:
        """Columns whose three cards are all revealed and equal."""
        out = []
        for c in range(N_COLS):
            ps = column_positions(c)
            if any(self.removed[p] for p in ps):
                continue
            if all(self.revealed[p] for p in ps) and len({self.values[p] for p in ps}) == 1:
                out.append(c)
        return out


@dataclass
class RoundResult:
    raw_scores: list[int]
    scores: list[int]          # after closer doubling
    closer: int
    doubled: bool
    n_turns: int


class SkyjoGame:
    """Full game (rounds until someone reaches `end_score`).

    Set `max_rounds=1` to simulate single rounds (round-score objective).
    """

    def __init__(self, n_players: int = 2, seed: Optional[int] = None,
                 end_score: int = 100, max_rounds: Optional[int] = None,
                 n_initial_reveals: int = 2):
        if not 2 <= n_players <= 8:
            raise ValueError("Skyjo is for 2-8 players")
        self.n_players = n_players
        self.rng = random.Random(seed)
        self.end_score = end_score
        self.max_rounds = max_rounds
        self.n_initial_reveals = n_initial_reveals
        self.totals = [0] * n_players
        self.round_results: list[RoundResult] = []
        self.round_index = -1
        self.next_starter: Optional[int] = None  # closer of previous round
        self._start_round()

    # ------------------------------------------------------------------ setup
    def _start_round(self) -> None:
        self.round_index += 1
        deck = [v for v, n in DECK_COUNTS.items() for _ in range(n)]
        self.rng.shuffle(deck)
        self.deck = deck
        self.boards = [Board(values=[self.deck.pop() for _ in range(N_POS)])
                       for _ in range(self.n_players)]
        self.discard = [self.deck.pop()]
        self.hand: Optional[int] = None
        self.closer: Optional[int] = None
        self.final_turns_left = 0
        self.n_turns = 0
        self.n_reshuffles = 0
        # setup: players reveal in seat order
        self.phase = SETUP
        self.current = 0
        self._setup_reveals_left = [self.n_initial_reveals] * self.n_players

    def _finish_setup(self) -> None:
        if self.next_starter is not None:
            start = self.next_starter
        else:
            sums = [b.visible_sum() for b in self.boards]
            best = max(sums)
            start = self.rng.choice([i for i, s in enumerate(sums) if s == best])
        self.current = start
        self.phase = CHOOSE_SOURCE

    # ------------------------------------------------------------- utilities
    @property
    def discard_top(self) -> int:
        return self.discard[-1]

    def _draw(self) -> int:
        if not self.deck:
            top = self.discard.pop()
            self.deck = self.discard
            self.rng.shuffle(self.deck)
            self.discard = [top]
            self.n_reshuffles += 1
        return self.deck.pop()

    def _resolve_triples(self, player: int) -> list[int]:
        b = self.boards[player]
        cols = b.find_triples()
        for c in cols:
            for p in column_positions(c):
                b.removed[p] = True
                self.discard.append(b.values[p])  # triples go on top of the discard
        return cols

    # ------------------------------------------------------------- legality
    def legal_actions(self) -> list[int]:
        b = self.boards[self.current]
        if self.phase == SETUP:
            return [flip_action(p) for p in b.hidden_positions()]
        if self.phase == CHOOSE_SOURCE:
            return [A_TAKE_DISCARD, A_DRAW_DECK]
        if self.phase == PLACE_TAKEN:
            return [replace_action(p) for p in b.active_positions()]
        if self.phase == DECIDE_DRAWN:
            return ([replace_action(p) for p in b.active_positions()]
                    + [flip_action(p) for p in b.hidden_positions()])
        return []

    def legal_mask(self) -> list[bool]:
        m = [False] * N_ACTIONS
        for a in self.legal_actions():
            m[a] = True
        return m

    # ------------------------------------------------------------------ step
    def step(self, action: int) -> None:
        if action not in self.legal_actions():
            raise ValueError(f"illegal action {action} ({decode(action)}) in phase {self.phase}")
        kind, p = decode(action)
        b = self.boards[self.current]

        if self.phase == SETUP:
            b.revealed[p] = True
            self._setup_reveals_left[self.current] -= 1
            if self._setup_reveals_left[self.current] == 0:
                self.current += 1
                if self.current == self.n_players:
                    self._finish_setup()
            return

        if self.phase == CHOOSE_SOURCE:
            if kind == "take_discard":
                self.hand = self.discard.pop()
                self.phase = PLACE_TAKEN
            else:
                self.hand = self._draw()
                self.phase = DECIDE_DRAWN
            return

        # PLACE_TAKEN / DECIDE_DRAWN
        if kind == "replace":
            old = b.values[p]
            b.values[p] = self.hand
            b.revealed[p] = True
            self.discard.append(old)
        else:  # flip (only from DECIDE_DRAWN)
            self.discard.append(self.hand)
            b.revealed[p] = True
        self.hand = None
        self._resolve_triples(self.current)
        self._end_turn()

    def _end_turn(self) -> None:
        self.n_turns += 1
        player = self.current
        if self.closer is None:
            if self.boards[player].all_revealed():
                self.closer = player
                self.final_turns_left = self.n_players - 1
        else:
            self.final_turns_left -= 1
        if self.closer is not None and self.final_turns_left == 0:
            self._end_round()
            return
        self.current = (player + 1) % self.n_players
        self.phase = CHOOSE_SOURCE

    def _end_round(self) -> None:
        for i, b in enumerate(self.boards):
            for p in b.hidden_positions():
                b.revealed[p] = True
            self._resolve_triples(i)  # rule also applies at the final reveal
        raw = [b.score() for b in self.boards]
        scores = list(raw)
        c = self.closer
        others_min = min(raw[i] for i in range(self.n_players) if i != c)
        doubled = raw[c] > 0 and others_min <= raw[c]
        if doubled:
            scores[c] *= 2
        self.round_results.append(RoundResult(raw, scores, c, doubled, self.n_turns))
        for i in range(self.n_players):
            self.totals[i] += scores[i]
        self.next_starter = c
        if max(self.totals) >= self.end_score or (
                self.max_rounds is not None and len(self.round_results) >= self.max_rounds):
            self.phase = GAME_OVER
        else:
            self._start_round()

    # --------------------------------------------------------------- results
    @property
    def done(self) -> bool:
        return self.phase == GAME_OVER

    def winners(self) -> list[int]:
        best = min(self.totals)
        return [i for i, t in enumerate(self.totals) if t == best]

    # ----------------------------------------------------------- observation
    def unseen_counts(self, viewer: int) -> dict[int, int]:
        """Composition of all cards the viewer cannot currently see
        (draw pile + every hidden card on every board). This is public
        information anyone at the table could work out."""
        return _unseen_counts(self, viewer)

    def observation(self, viewer: Optional[int] = None) -> "Observation":
        viewer = self.current if viewer is None else viewer
        return Observation(self, viewer)


class Observation:
    """What player `me` can see. Rich, convenient view for scripted agents."""

    def __init__(self, g: SkyjoGame, me: int):
        self._g = g
        self.me = me
        self.n_players = g.n_players
        self.phase = g.phase
        self.hand = g.hand if g.current == me else None
        self.discard_top = g.discard[-1] if g.discard else None
        self.deck_size = len(g.deck)
        self.totals = g.totals
        self.round_index = g.round_index
        self.closer = g.closer
        self.is_final_turn = g.closer is not None
        self.end_score = g.end_score
        counts = [0] * 15  # index v + 2
        for v, n in DECK_COUNTS.items():
            counts[v + 2] = n
        vis_all, hid_all, rem_all = [], [], []
        for b in g.boards:
            vals, rev, rem = b.values, b.revealed, b.removed
            vis = [None] * N_POS
            hid = [False] * N_POS
            for p in range(N_POS):
                if rem[p]:
                    continue
                if rev[p]:
                    v = vals[p]
                    vis[p] = v
                    counts[v + 2] -= 1
                else:
                    hid[p] = True
            vis_all.append(vis)
            hid_all.append(hid)
            rem_all.append(rem)
        for v in g.discard:
            counts[v + 2] -= 1
        if self.hand is not None:
            counts[self.hand + 2] -= 1
        self.visible, self.hidden, self.removed = vis_all, hid_all, rem_all
        self.unseen_list = counts
        n_unseen = sum(counts)
        self.expected_hidden = (sum((i - 2) * n for i, n in enumerate(counts)) / n_unseen
                                if n_unseen else 0.0)

    @property
    def unseen(self) -> dict[int, int]:
        return {i - 2: n for i, n in enumerate(self.unseen_list)}

    @property
    def legal(self) -> list[int]:
        return self._g.legal_actions()

    # helpers ---------------------------------------------------------------
    def n_hidden(self, player: int) -> int:
        return sum(self.hidden[player])

    def visible_sum(self, player: int) -> int:
        return sum(v for v in self.visible[player] if v is not None)

    def estimated_score(self, player: int) -> float:
        return self.visible_sum(player) + self.n_hidden(player) * self.expected_hidden


def _unseen_counts(g: SkyjoGame, viewer: int) -> dict[int, int]:
    return Observation(g, viewer).unseen
