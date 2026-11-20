"""Run games between agents and summarise the results."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .engine import SkyjoGame


def play_game(agents: Sequence, seed: Optional[int] = None, max_rounds: Optional[int] = None,
              end_score: int = 100) -> SkyjoGame:
    g = SkyjoGame(n_players=len(agents), seed=seed, max_rounds=max_rounds, end_score=end_score)
    guard = 0
    while not g.done:
        a = agents[g.current].act(g.observation())
        g.step(a)
        guard += 1
        if guard > 200_000:
            raise RuntimeError("game did not terminate")
    return g


@dataclass
class SeatStats:
    name: str
    wins: float = 0.0
    games: int = 0
    round_scores: list = field(default_factory=list)
    closes: int = 0
    doubled: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else float("nan")

    def win_ci(self, z: float = 1.96) -> float:
        p, n = self.win_rate, self.games
        return z * math.sqrt(p * (1 - p) / n) if n else float("nan")

    @property
    def mean_round(self) -> float:
        return sum(self.round_scores) / len(self.round_scores)

    def round_ci(self, z: float = 1.96) -> float:
        xs = self.round_scores
        m = self.mean_round
        var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
        return z * math.sqrt(var / len(xs))


def match(make_agents: Callable[[], Sequence], n_games: int, seed: int = 0,
          max_rounds: Optional[int] = None, rotate_seats: bool = True) -> list[SeatStats]:
    """Play n_games. Agents keep their identity; seats are rotated each game so
    no one benefits from seat order. Returns stats per agent (in the order
    `make_agents` returns them)."""
    rng = random.Random(seed)
    proto = make_agents()
    stats = [SeatStats(a.name) for a in proto]
    n = len(proto)
    for gi in range(n_games):
        agents = make_agents()
        order = list(range(n))
        if rotate_seats:
            k = gi % n
            order = order[k:] + order[:k]
        seated = [agents[i] for i in order]
        g = play_game(seated, seed=rng.getrandbits(64), max_rounds=max_rounds)
        winners = g.winners()
        for seat, ai in enumerate(order):
            s = stats[ai]
            s.games += 1
            if seat in winners:
                s.wins += 1 / len(winners)
            for r in g.round_results:
                s.round_scores.append(r.scores[seat])
                if r.closer == seat:
                    s.closes += 1
                    s.doubled += r.doubled
    return stats


def report(stats: list[SeatStats], title: str = "") -> str:
    lines = [title] if title else []
    lines.append(f"{'agent':<22}{'win rate':>16}{'mean round score':>22}{'closes':>8}{'doubled':>9}")
    for s in stats:
        lines.append(f"{s.name:<22}{s.win_rate:>9.3f} ±{s.win_ci():.3f}"
                     f"{s.mean_round:>14.2f} ±{s.round_ci():.2f}{s.closes:>8}{s.doubled:>9}")
    return "\n".join(lines)
