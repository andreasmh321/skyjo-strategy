"""Log every decision of a policy (trained agent or heuristic) against an opponent.

  python analyze_agent.py runs/v2/best.pt --games 3000 --out results/decisions_agent.parquet
  python analyze_agent.py heuristic:tuned_win --games 3000 --out results/decisions_heur.parquet

One row per decision of the logged player (2-player games), with the situation
(card, round progress, estimated scores, totals) and the decoded action. The
tuned_win heuristic's choice in the same position is stored as well.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch

from skyjo import fast as F
from skyjo.rl.env import VecEnv
from skyjo.rl.ppo import load_model

PHASE_NAMES = {F.SETUP: "setup", F.CHOOSE: "choose", F.PLACE: "place_taken", F.DECIDE: "decide_drawn"}


def decode(a_canon, perm):
    """canonical action -> (kind, real position)"""
    if a_canon == F.TAKE:
        return "take", -1
    if a_canon == F.DRAW:
        return "draw", -1
    if a_canon >= F.FLIP0:
        return "flip", int(perm[a_canon - F.FLIP0])
    return "replace", int(perm[a_canon - F.REP0])


def situation(env, b, counts):
    V, S, M, T, C = env.V[b], env.S[b], env.M[b], env.T[b], env.C[b]
    me = int(M[F.M_CUR]); op = 1 - me
    g = (env.V[b], env.S[b], env.D[b], env.C[b], env.M[b], env.T[b], env.A[b])
    E = F._expected_hidden(g, counts)
    phase = int(M[F.M_PHASE])
    top = int(C[M[F.M_CLEN] - 1]) if M[F.M_CLEN] > 0 else None
    hand = None if M[F.M_HAND] == F.NOCARD else int(M[F.M_HAND])
    card = top if phase == F.CHOOSE else hand

    def board(pl):
        vis = [int(V[pl, p]) if S[pl, p] == F.REVEALED else None for p in range(12)]
        hid = [S[pl, p] == F.HIDDEN for p in range(12)]
        rem = [S[pl, p] == F.REMOVED for p in range(12)]
        return vis, hid, rem
    mv, mh, mr = board(me)
    ov, oh, orr = board(op)
    my_vis = sum(v for v in mv if v is not None); op_vis = sum(v for v in ov if v is not None)
    row = dict(phase=PHASE_NAMES.get(phase, phase), card=card, E=E,
               my_hidden=sum(mh), opp_hidden=sum(oh), my_removed_cols=sum(mr) // 3,
               my_vis_sum=my_vis, opp_vis_sum=op_vis,
               my_est=my_vis + sum(mh) * E, opp_est=op_vis + sum(oh) * E,
               my_total=int(T[me]), opp_total=int(T[op]), final_turn=bool(M[F.M_CLOSER] >= 0),
               my_max_visible=max([v for v in mv if v is not None], default=None))
    if card is not None and phase != F.SETUP:
        # best saving on a revealed card, and whether the card completes / starts a column
        best_rev, triple, pair = None, False, False
        for p in range(12):
            if mr[p]:
                continue
            c = p // 3
            others = [q for q in (3 * c, 3 * c + 1, 3 * c + 2) if q != p]
            ov2 = [mv[q] for q in others]
            if all(v == card for v in ov2):
                triple = True
            elif sum(v == card for v in ov2) == 1 and mv[p] != card:
                pair = True
            if mv[p] is not None:
                best_rev = mv[p] - card if best_rev is None else max(best_rev, mv[p] - card)
        row.update(best_rev_gain=best_rev, hidden_gain=E - card if sum(mh) else None,
                   can_triple=triple, can_pair=pair)
    return row, (mv, mh, mr), me


def action_info(kind, pos, boards, card, my_hidden, final):
    mv, mh, mr = boards
    d = dict(action=kind, pos=pos)
    if kind in ("replace", "flip"):
        d["target_hidden"] = bool(mh[pos])
        d["closes"] = bool(mh[pos] and my_hidden == 1 and not final)
    if kind == "replace":
        d["replaced_value"] = mv[pos]
        d["equal_swap"] = mv[pos] is not None and mv[pos] == card
        c = pos // 3
        others = [mv[q] for q in (3 * c, 3 * c + 1, 3 * c + 2) if q != pos]
        d["made_triple"] = all(v == card for v in others)
        d["made_pair"] = (not d["made_triple"]) and sum(v == card for v in others) == 1
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="checkpoint path, or heuristic:<name>")
    ap.add_argument("--opponent", default="tuned_win")
    ap.add_argument("--games", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=300)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()

    S = json.load(open("results/strategies.json"))
    canonical = True
    model = None
    if args.policy.startswith("heuristic:"):
        heur = S[args.policy.split(":", 1)[1]]
    else:
        model, ck = load_model(args.policy)
        canonical = ck.get("args", {}).get("canonical", False)
    env = VecEnv(args.envs, opponents=[S[args.opponent]], probs=[1.0, 0.0], seed=args.seed,
                 canonical=canonical)
    quota = -(-args.games // args.envs)
    gid = np.zeros(args.envs, np.int64)
    rows, results = [], []
    counts = np.empty(15, np.int64)
    while (gid < quota).any():
        if model is not None:
            with torch.no_grad():
                a, _, _ = model.act(torch.from_numpy(env.obs), torch.from_numpy(env.masks), True)
            acts = a.numpy()
        else:
            acts = env.heuristic_actions(heur)
        ref = env.heuristic_actions(S["tuned_win"])
        for b in range(args.envs):
            if gid[b] >= quota:
                continue
            row, boards, me = situation(env, b, counts)
            kind, pos = decode(int(acts[b]), env.perms[b])
            rkind, rpos = decode(int(ref[b]), env.perms[b])
            row.update(action_info(kind, pos, boards, row["card"], row["my_hidden"], row["final_turn"]))
            row.update(env=b, game=int(gid[b]), round=int(env.M[b, F.M_NROUNDS]),
                       heur_action=rkind, heur_target_hidden=(boards[1][rpos] if rpos >= 0 else None),
                       same_as_heur=bool(acts[b] == ref[b]))
            rows.append(row)
        env.step(acts)
        for b in np.flatnonzero(env.ep_n):
            for j in range(env.ep_n[b]):
                e = env.ep_out[b, j]
                results.append(dict(env=b, game=int(gid[b]), win=e[0], rounds=e[1],
                                    round_sum=e[2], aborted=e[4]))
                gid[b] += 1
        env.ep_n[:] = 0
    df = pd.DataFrame(rows)
    res = pd.DataFrame(results)
    df = df.merge(res[["env", "game", "win", "aborted"]], on=["env", "game"], how="inner")
    df.to_parquet(args.out)
    res.to_parquet(args.out.replace(".parquet", "_games.parquet"))
    print(f"{len(df):,} decisions from {len(res)} games, win rate {res.win[res.aborted == 0].mean():.3f}")


if __name__ == "__main__":
    main()
