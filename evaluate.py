import argparse, json, random, itertools, time
from skyjo.agents import HeuristicParams
from skyjo.parallel import make_evaluator
from tune import PHASE1_BASELINE


def main():

    def last_mean(path):
        return [json.loads(l) for l in open(path)][-1]["mean_params"]

    greedy = {**HeuristicParams().as_dict(), "take_gain_early":0,"take_gain_late":0,"keep_gain_early":0,
              "keep_gain_late":0,"pair_bonus":0,"close_margin":-1e9,"end_lookahead":0,"setup_same_col":0}
    agents = {"greedy": greedy, "phase1": PHASE1_BASELINE,
              "tuned_round": last_mean("results/cma_round.jsonl"),
              "tuned_win": last_mean("results/cma_win.jsonl")}
    json.dump(agents, open("results/strategies.json","w"), indent=1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--processes", type=int, default=None, help="worker processes (default: all cores)")
    ap.add_argument("--games", type=int, default=6000)
    ap.add_argument("--engine", choices=["fast", "python"], default="fast")
    args = ap.parse_args()
    ev = make_evaluator(args.engine, args.processes); rng = random.Random(2026)
    seeds = [rng.getrandbits(48) for _ in range(args.games)]
    rows = []
    t=time.time()
    for a, b in itertools.combinations(agents, 2):
        if a == b: continue
        full = ev.evaluate([agents[a]], [agents[b]], seeds)[0]
        one = ev.evaluate([agents[a]], [agents[b]], seeds, max_rounds=1)[0] if not (a=="tuned_round" and b=="tuned_round") else None
        rows.append((a, b, full, one))
        print(f"{a:>12} vs {b:<12} game win {full['win_rate']:.3f}  round win {one['win_rate']:.3f}  own mean round (full games) {full['mean_round']:.2f}  (single rounds) {one['mean_round']:.2f}", flush=True)
    json.dump([(a,b,f,o) for a,b,f,o in rows], open("results/tournament.json","w"), indent=1)
    print(f"{time.time()-t:.0f}s"); ev.close()


if __name__ == "__main__":
    main()
