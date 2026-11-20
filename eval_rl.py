"""Evaluate a trained agent against the heuristics.

  python eval_rl.py runs/win2p/best.pt --games 10000
"""
import argparse
import json

import torch

from skyjo.rl.ppo import evaluate, load_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--games", type=int, default=10000)
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--objective", choices=["win", "round"], default="win")
    ap.add_argument("--sample", action="store_true", help="sample moves instead of best move")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(args.checkpoint, device)
    canonical = ck.get("args", {}).get("canonical", False)
    print(f"{args.checkpoint}: iteration {ck.get('iteration')}, device {device}, canonical {canonical}")
    strat = json.load(open("results/strategies.json"))
    for name in ["tuned_win", "tuned_round", "phase1", "greedy"]:
        if name not in strat:
            continue
        r = evaluate(model, strat[name], args.games, args.players, device,
                     deterministic=not args.sample, objective=args.objective, canonical=canonical)
        print(f"vs {name:<12} win rate {r['win_rate']:.3f} ± {r['ci95']:.3f}   "
              f"own mean round score {r['mean_round']:.2f}   ({r['games']} games"
              + (f", {r['aborted']} aborted: endless round" if r['aborted'] else "") + ")", flush=True)


if __name__ == "__main__":
    main()
