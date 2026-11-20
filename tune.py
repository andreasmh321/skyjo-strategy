"""CMA-ES tuning of HeuristicParams.

Objectives
  win   : maximise win rate in full 2-player games (to 100)
  round : minimise own mean score in single rounds

Usage: python tune.py win --minutes 8      (resumes from checkpoint if present)
"""
import argparse
import json
import os
import pickle
import random
import time

import cma

from skyjo.agents import HeuristicParams
from skyjo.parallel import make_evaluator

SPACES = {
    # name: scale (search happens in x / scale units)
    "win": {"take_gain_early": 1.5, "take_gain_late": 1.5, "keep_gain_early": 1.5,
            "keep_gain_late": 1.5, "hidden_bonus": 1.0, "pair_bonus": 1.5,
            "close_margin": 3.0, "close_lead_coef": 2.0, "deny_weight": 0.3,
            "end_margin": 3.0},
    "round": {"take_gain_early": 1.5, "take_gain_late": 1.5, "keep_gain_early": 1.5,
              "keep_gain_late": 1.5, "hidden_bonus": 1.0, "pair_bonus": 1.5,
              "close_margin": 3.0, "deny_weight": 0.3},
}
PHASE1_BASELINE = {**HeuristicParams().as_dict(), "end_lookahead": 0.0, "setup_same_col": 0.0}


def to_params(names, scales, z, base):
    d = dict(base)
    for n, s, v in zip(names, scales, z):
        d[n] = float(v) * s
    d["deny_weight"] = max(0.0, d["deny_weight"])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("objective", choices=["win", "round"])
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--processes", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--games", type=int, default=None,
                    help="games per candidate (default 2000 with --engine fast, 500 with python)")
    ap.add_argument("--engine", choices=["fast", "python"], default="fast")
    ap.add_argument("--tag", default="")
    ap.add_argument("--opponent", default=None, help="json file with opponent params")
    ap.add_argument("--start", default=None, help="json file with start params")
    ap.add_argument("--sigma0", type=float, default=0.5)
    args = ap.parse_args()

    ckpt = f"results/cma_{args.objective}{args.tag}.pkl"
    log = f"results/cma_{args.objective}{args.tag}.jsonl"
    os.makedirs("results", exist_ok=True)
    space = SPACES[args.objective]
    names, scales = list(space), list(space.values())
    base = HeuristicParams().as_dict()
    opponents = [json.load(open(args.opponent))] if args.opponent else [PHASE1_BASELINE]
    if args.start:
        base = json.load(open(args.start))
    max_rounds = 1 if args.objective == "round" else None
    if args.games is None:
        args.games = 2000 if args.engine == "fast" else 500
    n_games = args.games * (3 if args.objective == "round" else 1)

    if os.path.exists(ckpt):
        es, gen = pickle.load(open(ckpt, "rb"))
    else:
        x0 = [base[n] / s for n, s in zip(names, scales)]
        es = cma.CMAEvolutionStrategy(x0, args.sigma0, {"popsize": args.popsize, "seed": 1,
                                                "verbose": -9})
        gen = 0

    ev = make_evaluator(args.engine, args.processes)
    t0 = time.time()
    while time.time() - t0 < args.minutes * 60:
        rng = random.Random(1000 + gen)
        seeds = [rng.getrandbits(48) for _ in range(n_games)]
        zs = es.ask()
        cands = [to_params(names, scales, z, base) for z in zs]
        cands.append(to_params(names, scales, es.mean, base))  # track the mean too
        res = ev.evaluate(cands, opponents, seeds, max_rounds=max_rounds)
        if args.objective == "win":
            losses = [-r["win_rate"] for r in res[:-1]]
        else:
            losses = [r["mean_round"] for r in res[:-1]]
        es.tell(zs, losses)
        rec = {"gen": gen, "mean_params": cands[-1], "mean_eval": res[-1],
               "best_loss": min(losses), "sigma": es.sigma, "t": time.time() - t0}
        open(log, "a").write(json.dumps(rec) + "\n")
        print(f"gen {gen:3d}  mean: win {res[-1]['win_rate']:.3f} round {res[-1]['mean_round']:.2f}"
              f"  best {min(losses):.3f}  sigma {es.sigma:.3f}", flush=True)
        gen += 1
        pickle.dump((es, gen), open(ckpt, "wb"))
    ev.close()


if __name__ == "__main__":
    main()
