import time, random
from skyjo.agents import HeuristicParams as HP
from skyjo.parallel import Evaluator


def main():
    base = HP().as_dict()
    ev = Evaluator()
    seeds = [random.Random(42).getrandbits(48) for _ in range(1)]
    rng = random.Random(42); seeds=[rng.getrandbits(48) for _ in range(4000)]
    variants = {
     "default": {},
     "end_lookahead": {"end_lookahead": 1.0},
     "setup_same_col": {"setup_same_col": 1.0},
     "final_turn_greedy OFF": {"final_turn_greedy": 0.0},
     "deny_weight=0.5": {"deny_weight": 0.5},
    }
    t=time.time()
    cands=[{**base, **v} for v in variants.values()]
    full = ev.evaluate(cands, [base], seeds)
    single = ev.evaluate(cands, [base], seeds[:3000], max_rounds=1)
    for name, f, s in zip(variants, full, single):
        print(f"{name:<24} win {f['win_rate']:.3f}   1-round mean {s['mean_round']:.2f}")
    print(f"{time.time()-t:.0f}s")
    ev.close()


if __name__ == "__main__":
    main()
