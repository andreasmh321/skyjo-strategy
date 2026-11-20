import time
from skyjo import HeuristicAgent, RandomAgent, greedy_agent, match, report

t=time.time()
s = match(lambda: [HeuristicAgent(), RandomAgent()], 400, seed=1)
print(report(s, "heuristic vs random (2p, full games)")); print(f"{time.time()-t:.1f}s\n")
t=time.time()
s = match(lambda: [HeuristicAgent(), greedy_agent()], 2000, seed=2)
print(report(s, "heuristic vs greedy (2p, full games)")); print(f"{time.time()-t:.1f}s\n")
s = match(lambda: [greedy_agent(), RandomAgent()], 400, seed=3)
print(report(s, "greedy vs random"))
