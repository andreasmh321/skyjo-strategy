"""Parallel evaluation of HeuristicParams against fixed opponents.

Common random numbers: every candidate plays the same game seeds, with seats
rotated, so differences between candidates are not drowned in deal luck.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import random
from typing import Optional, Sequence

from .agents import HeuristicAgent, HeuristicParams
from .simulate import play_game


def _play_chunk(args):
    cand, opps, seeds, max_rounds = args
    n = len(opps) + 1
    wins = 0.0
    round_sum = 0.0
    n_rounds = 0
    for gi, seed in enumerate(seeds):
        rng = random.Random(seed)
        agents = [HeuristicAgent(HeuristicParams(**cand), seed=rng.getrandbits(32))] + \
                 [HeuristicAgent(HeuristicParams(**o), seed=rng.getrandbits(32)) for o in opps]
        seat = gi % n  # rotate the candidate's seat
        order = list(range(1, n))
        order.insert(seat, 0)
        seated = [agents[i] for i in order]
        g = play_game(seated, seed=seed, max_rounds=max_rounds)
        w = g.winners()
        if seat in w:
            wins += 1 / len(w)
        for r in g.round_results:
            round_sum += r.scores[seat]
            n_rounds += 1
    return wins, len(seeds), round_sum, n_rounds


class Evaluator:
    def __init__(self, processes: Optional[int] = None):
        processes = processes or os.cpu_count() or 1
        self.pool = mp.Pool(processes)
        self.processes = processes

    def close(self):
        self.pool.close()
        self.pool.join()

    def evaluate(self, candidates: Sequence[dict], opponents: Sequence[dict], seeds: Sequence[int],
                 max_rounds=None, chunk: int = 100) -> list[dict]:
        """Returns per candidate: win_rate, mean_round."""
        tasks, owner = [], []
        for ci, c in enumerate(candidates):
            for k in range(0, len(seeds), chunk):
                tasks.append((c, list(opponents), list(seeds[k:k + chunk]), max_rounds))
                owner.append(ci)
        res = self.pool.map(_play_chunk, tasks)
        agg = [[0.0, 0, 0.0, 0] for _ in candidates]
        for ci, (w, n, rs, nr) in zip(owner, res):
            a = agg[ci]
            a[0] += w; a[1] += n; a[2] += rs; a[3] += nr
        return [{"win_rate": a[0] / a[1], "mean_round": a[2] / a[3], "games": a[1]} for a in agg]


def make_evaluator(engine: str = "fast", processes: Optional[int] = None):
    """engine='fast' (numba, threads) or 'python' (reference engine, processes)."""
    if engine == "fast":
        from .fast import FastEvaluator
        return FastEvaluator(processes)
    return Evaluator(processes)
