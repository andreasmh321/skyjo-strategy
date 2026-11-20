"""Evaluate the agent WITH search against a heuristic or against the plain network.

  python search_eval.py runs/v2/best.pt --opponent tuned_win --games 1000
  python search_eval.py runs/v2/best.pt --opponent self --games 1000     # search vs no search
  python search_eval.py runs/v2/best.pt --opponent tuned_win --games 1000 --no-search   # baseline

Knobs: --K worlds per candidate, --depth own turns to look ahead, --max-cands,
--skip-above (no search when the network is this sure), --envs games in parallel.
"""
import argparse
import json
import math
import time

import numpy as np
import torch

from skyjo.rl.env import VecEnv
from skyjo.rl.ppo import load_model, make_policy_fn
from skyjo.rl.search import search_actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--opponent", default="tuned_win", help="heuristic name, or 'self' = plain network")
    ap.add_argument("--games", type=int, default=1000)
    ap.add_argument("--envs", type=int, default=100)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--max-cands", type=int, default=6)
    ap.add_argument("--min-prior", type=float, default=0.02)
    ap.add_argument("--skip-above", type=float, default=0.95)
    ap.add_argument("--z", type=float, default=2.0, help="evidence needed to overrule the network")
    ap.add_argument("--sample-rollouts", action="store_true", help="sample moves in rollouts")
    ap.add_argument("--no-search", action="store_true")
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.checkpoint, device)
    canonical = ck.get("args", {}).get("canonical", False)
    S = json.load(open("results/strategies.json"))
    if args.opponent == "self":
        env = VecEnv(args.envs, opponents=[S["tuned_win"]], probs=[0.0, 1.0], seed=args.seed,
                     canonical=canonical, net_policy=make_policy_fn(model, device, deterministic=True))
    else:
        env = VecEnv(args.envs, opponents=[S[args.opponent]], probs=[1.0, 0.0], seed=args.seed,
                     canonical=canonical)
    quota = math.ceil(args.games / args.envs)
    taken = np.zeros(args.envs, np.int64)
    rows = []
    t0 = time.time()
    steps = searched = changed = rollouts = 0
    while (taken < quota).any():
        if args.no_search:
            with torch.no_grad():
                a, _, _ = model.act(torch.from_numpy(env.obs).to(device),
                                    torch.from_numpy(env.masks).to(device), True)
            acts = a.cpu().numpy()
        else:
            acts, info = search_actions(env, model, device, K=args.K, depth=args.depth,
                                        min_prior=args.min_prior, max_cands=args.max_cands,
                                        skip_above=args.skip_above, canonical=canonical,
                                        z=args.z, greedy=not args.sample_rollouts)
            searched += info["searched"]; changed += info["changed"]; rollouts += info["rollouts"]
        steps += env.B
        env.step(acts)
        for b in np.flatnonzero(env.ep_n):
            for j in range(env.ep_n[b]):
                if taken[b] < quota:
                    rows.append(env.ep_out[b, j].copy())
                    taken[b] += 1
        env.ep_n[:] = 0
        if steps % (env.B * 50) == 0:
            done = len(rows)
            print(f"  {done} games done | {time.time() - t0:.0f}s | searched {searched / max(steps, 1):.0%} of decisions,"
                  f" search changed the move in {changed / max(searched, 1):.0%} of those", flush=True)
    rows = np.array(rows)
    ab = int(rows[:, 4].sum())
    rows = rows[rows[:, 4] == 0]
    p = rows[:, 0].mean()
    ci = 1.96 * math.sqrt(p * (1 - p) / len(rows))
    dt = time.time() - t0
    label = "no search" if args.no_search else f"search K={args.K} depth={args.depth}"
    print(f"\n{label} vs {args.opponent}: win rate {p:.3f} ± {ci:.3f}, own mean round score "
          f"{rows[:, 2].sum() / rows[:, 1].sum():.2f} ({len(rows)} games, {ab} aborted, {dt:.0f}s)")
    if not args.no_search:
        print(f"searched {searched / steps:.0%} of decisions; search changed the network's move in "
              f"{changed / max(searched, 1):.0%} of searched decisions; {rollouts / max(searched, 1):.0f} rollouts each")


if __name__ == "__main__":
    main()
