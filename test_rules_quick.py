import json, time
from multiprocessing import Pool
from skyjo.agents import HeuristicAgent, HeuristicParams
from skyjo.rules_agent import RuleAgent
from skyjo.simulate import match
S = json.load(open('results/strategies.json'))
def run(args):
    opp, seed, n = args
    st = match(lambda: [RuleAgent(seed=seed), HeuristicAgent(HeuristicParams(**S[opp]), name=opp, seed=seed+1)], n, seed=seed)
    a = st[0]
    return opp, a.wins, a.games, sum(a.round_scores), len(a.round_scores), a.closes, a.doubled
if __name__ == "__main__":
    t = time.time()
    jobs = [(o, s, 1000) for o in ['tuned_win', 'phase1', 'greedy'] for s in (1, 2)]
    with Pool(2) as p:
        res = p.map(run, jobs)
    agg = {}
    for o, w, g, rs, nr, c, d in res:
        a = agg.setdefault(o, [0, 0, 0, 0, 0, 0]); a[0]+=w; a[1]+=g; a[2]+=rs; a[3]+=nr; a[4]+=c; a[5]+=d
    for o, (w, g, rs, nr, c, d) in agg.items():
        p = w / g
        print(f"table rules vs {o:<10} win {p:.3f} ± {1.96*(p*(1-p)/g)**.5:.3f}  round {rs/nr:.2f}  ended {c} rounds, doubled {d}  ({g} games)")
    print(f"{time.time()-t:.0f}s")
