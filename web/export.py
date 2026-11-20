"""Export v2 weights + reference cases for the browser version."""
import base64, json, sys
import numpy as np, torch
from skyjo import fast as F
from skyjo.rl.env import VecEnv
from skyjo.rl.ppo import load_model
sys.setrecursionlimit(10000)
model, ck = load_model("runs/v2/best.pt")
sd = model.state_dict()
order = ["trunk.0", "trunk.1", "trunk.3", "trunk.4", "trunk.6", "trunk.7", "pi", "v"]
parts, meta, off = [], [], 0
for name in order:
    for kind in ("weight", "bias"):
        t = sd[f"{name}.{kind}"].float().numpy().astype("<f4").ravel()
        meta.append({"name": f"{name}.{kind}", "shape": list(sd[f"{name}.{kind}"].shape), "offset": off, "size": t.size})
        parts.append(t); off += t.size
blob = np.concatenate(parts).tobytes()
json.dump({"meta": meta, "b64": base64.b64encode(blob).decode()}, open("web/weights.json", "w"))
print("weights:", off, "floats,", len(blob) / 1e6, "MB")

S = json.load(open("results/strategies.json"))
def snap(env, b):
    m = env.M[b]; n = int(m[F.M_NP])
    return {"values": env.V[b, :n].tolist(), "st": env.S[b, :n].tolist(),
            "deck": env.D[b, :m[F.M_DLEN]].tolist(), "disc": env.C[b, :m[F.M_CLEN]].tolist(),
            "phase": int(m[F.M_PHASE]), "cur": int(m[F.M_CUR]),
            "hand": None if m[F.M_HAND] == F.NOCARD else int(m[F.M_HAND]),
            "closer": int(m[F.M_CLOSER]), "finalLeft": int(m[F.M_FINAL]), "nTurns": int(m[F.M_NTURNS]),
            "nextStarter": int(m[F.M_NEXT]), "setupLeft": int(m[F.M_SETUP]),
            "totals": env.T[b, :n].tolist(), "nRounds": int(m[F.M_NROUNDS]),
            "lastScores": env.A[b, 3, :n].tolist(), "lastDoubled": int(m[F.M_LAST_DOUBLED]),
            "lastCloser": int(m[F.M_LAST_CLOSER]), "nResh": int(m[F.M_NRESH])}
env = VecEnv(40, opponents=[S["tuned_win"]], probs=[1.0, 0.0], seed=5, canonical=True)
cases, trans = [], []
rng = np.random.default_rng(0)
for step in range(240):
    with torch.no_grad():
        logits, v = model(torch.from_numpy(env.obs), torch.from_numpy(env.masks))
    if step % 4 in (0, 1):
        for b in range(0, 40, 4):
            cases.append({"state": snap(env, b), "obs": env.obs[b].tolist(), "mask": env.masks[b].tolist(),
                          "perm": env.perms[b].tolist(), "logits": logits[b].tolist(), "value": float(v[b])})
    acts = logits.argmax(1).numpy()
    if rng.random() < 0.3:  # some random legal moves for coverage
        acts = np.array([rng.choice(np.flatnonzero(mk)) for mk in env.masks])
    env.step(acts)
json.dump(cases, open("web/cases.json", "w"))
print("cases:", len(cases), " phases:", sorted(set(c["state"]["phase"] for c in cases)))

# engine transitions: single game stepped directly with random legal moves
g = F.FastGame(2, seed=11)
class E: pass
for k in range(6000):
    vals, st, deck, disc, m, tot, acc = g.g
    before = {"values": vals[:2].tolist(), "st": st[:2].tolist(), "deck": deck[:m[F.M_DLEN]].tolist(),
              "disc": disc[:m[F.M_CLEN]].tolist(), "phase": int(m[F.M_PHASE]), "cur": int(m[F.M_CUR]),
              "hand": None if m[F.M_HAND] == F.NOCARD else int(m[F.M_HAND]), "closer": int(m[F.M_CLOSER]),
              "finalLeft": int(m[F.M_FINAL]), "nTurns": int(m[F.M_NTURNS]), "nextStarter": int(m[F.M_NEXT]),
              "setupLeft": int(m[F.M_SETUP]), "totals": tot[:2].tolist(), "nRounds": int(m[F.M_NROUNDS])}
    legal = g.legal_actions()
    a = int(rng.choice(legal))
    r0, rs0 = int(m[F.M_NROUNDS]), int(m[F.M_NRESH])
    ph0 = int(m[F.M_PHASE])
    g.step(a)
    vals, st, deck, disc, m, tot, acc = g.g
    after = {"values": vals[:2].tolist(), "st": st[:2].tolist(), "deck": deck[:m[F.M_DLEN]].tolist(),
             "disc": disc[:m[F.M_CLEN]].tolist(), "phase": int(m[F.M_PHASE]), "cur": int(m[F.M_CUR]),
             "hand": None if m[F.M_HAND] == F.NOCARD else int(m[F.M_HAND]), "closer": int(m[F.M_CLOSER]),
             "finalLeft": int(m[F.M_FINAL]), "totals": tot[:2].tolist(), "lastScores": acc[3, :2].tolist(),
             "lastRaw": acc[4, :2].tolist(), "lastDoubled": int(m[F.M_LAST_DOUBLED])}
    kind = "round_end" if int(m[F.M_NROUNDS]) != r0 else ("reshuffle" if int(m[F.M_NRESH]) != rs0 else
            ("setup_done" if ph0 == F.SETUP and int(m[F.M_PHASE]) != F.SETUP else "plain"))
    trans.append({"before": before, "legal": legal, "action": a, "after": after, "kind": kind})
    if g.done:
        g = F.FastGame(2, seed=1000 + k)
json.dump(trans, open("web/transitions.json", "w"))
from collections import Counter
print("transitions:", len(trans), Counter(t["kind"] for t in trans))
