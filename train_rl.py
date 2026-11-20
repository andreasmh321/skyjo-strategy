"""Train a Skyjo agent with PPO.

Phases of a fresh run:
  1. behaviour cloning (--bc-iters): the network copies the tuned_win heuristic's
     moves and learns the value of its positions,
  2. PPO: opponents are the tuned heuristics plus, once snapshots exist, earlier
     copies of the agent picked by prioritised self-play (snapshots it beats
     less often are picked more often).
The learning rate and entropy bonus decay linearly over the PPO part of the run.

Examples
  python train_rl.py --name v2 --minutes 240              # overnight run
  python train_rl.py --name v2 --minutes 60               # resume (schedules restart)
  python train_rl.py --name quick --minutes 5 --n-envs 256 --bc-iters 5

Output in runs/<name>/: latest.pt, best.pt (best eval vs tuned_win), log.jsonl
"""
import argparse
import copy
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as Fnn

from skyjo.rl.env import VecEnv, obs_dim
from skyjo.rl.ppo import ActorCritic, compute_gae, evaluate, make_policy_fn, ppo_update


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v2")
    ap.add_argument("--objective", choices=["win", "round"], default="win")
    ap.add_argument("--players", type=int, default=2)
    ap.add_argument("--minutes", type=float, default=60, help="total wall time incl. BC")
    ap.add_argument("--n-envs", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-final", type=float, default=0.1, help="final lr as a fraction of --lr")
    ap.add_argument("--gamma", type=float, default=0.997)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-final", type=float, default=0.002)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--max-grad-norm", type=float, default=0.5)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--no-canonical", dest="canonical", action="store_false",
                    help="disable sorted board order (symmetry)")
    ap.add_argument("--bc-iters", type=int, default=40, help="behaviour-cloning rollouts (0 = off)")
    ap.add_argument("--bc-teacher", default="tuned_win")
    ap.add_argument("--bc-lr", type=float, default=1e-3)
    ap.add_argument("--shaping", type=float, default=0.0,
                    help="round-score shaping weight at start (win objective); decays to 0")
    ap.add_argument("--shaping-decay-iters", type=int, default=300)
    ap.add_argument("--opp-weights", default="tuned_win=0.7,phase1=0.2,greedy=0.1")
    ap.add_argument("--selfplay", type=float, default=0.3, help="share of opponent seats from the pool")
    ap.add_argument("--snapshot-every", type=int, default=20)
    ap.add_argument("--pool-size", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--eval-games", type=int, default=4000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def save(path, model, opt, it, pool, best, args, extra=None):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "iteration": it,
                "pool": pool, "best": best, "model_config": model.config, "args": vars(args),
                **(extra or {})}, path)


def choose_snapshot(pool):
    """30%: newest; otherwise prioritised by how rarely we beat it."""
    if random.random() < 0.3:
        return len(pool) - 1
    w = np.array([(1.0 - e["wr"]) ** 2 + 0.02 for e in pool])
    return int(np.random.choice(len(pool), p=w / w.sum()))


def rollout(env, model, bufs, T, device, teacher=None):
    """Fill the buffers with T steps. With `teacher` (heuristic params) the
    teacher picks the moves (behaviour cloning) and its actions are stored."""
    obs_b, mask_b, act_b, logp_b, val_b, rew_b, done_b = bufs
    model.eval()
    for t in range(T):
        o = torch.from_numpy(env.obs).to(device)
        m = torch.from_numpy(env.masks).to(device)
        a, logp, v = model.act(o, m)
        if teacher is not None:
            a = torch.from_numpy(env.heuristic_actions(teacher)).to(device)
        obs_b[t], mask_b[t], act_b[t], logp_b[t] = o, m, a, logp
        val_b[t] = v.cpu().numpy()
        _, _, r, d = env.step(a.cpu().numpy())
        rew_b[t], done_b[t] = r, d
    with torch.no_grad():
        _, last_v = model(torch.from_numpy(env.obs).to(device), torch.from_numpy(env.masks).to(device))
    return last_v.cpu().numpy()


def main():
    args = parse()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    out = os.path.join("runs", args.name)
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    t0 = time.time()
    deadline = t0 + args.minutes * 60

    strat = json.load(open("results/strategies.json"))
    weights = dict(kv.split("=") for kv in args.opp_weights.split(","))
    opp_names = [k for k in weights if k in strat and float(weights[k]) > 0]
    opponents = [strat[k] for k in opp_names]
    w = np.array([float(weights[k]) for k in opp_names]); w = w / w.sum()

    D = obs_dim(args.players)
    model = ActorCritic(D, args.hidden, args.layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)
    it, pool, best = 0, [], -1.0
    latest = os.path.join(out, "latest.pt")
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        if ck["args"].get("canonical", False) != args.canonical:
            raise SystemExit("checkpoint was trained with a different --no-canonical setting")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        it, pool, best = ck["iteration"], ck["pool"], ck["best"]
        pool = [p if isinstance(p, dict) and "state" in p else {"state": p, "wr": 0.5} for p in pool]
        print(f"resumed {latest} at iteration {it}")
    json.dump(vars(args), open(os.path.join(out, "config.json"), "w"), indent=1)

    frozen = ActorCritic(D, args.hidden, args.layers).to(device).eval()
    net_fn = make_policy_fn(frozen, device)

    def probs():
        sp = args.selfplay if pool else 0.0
        return list(w * (1 - sp)) + [sp]

    shaping0 = args.shaping if args.objective == "win" else 0.0
    env = VecEnv(args.n_envs, n_players=args.players, opponents=opponents, probs=probs(),
                 objective=args.objective, shaping=shaping0, seed=args.seed + it,
                 net_policy=net_fn, canonical=args.canonical)
    if pool:
        frozen.load_state_dict(pool[-1]["state"])

    T, B = args.n_steps, args.n_envs
    bufs = (torch.zeros((T, B, D), dtype=torch.float32, device=device),
            torch.zeros((T, B, 26), dtype=torch.bool, device=device),
            torch.zeros((T, B), dtype=torch.long, device=device),
            torch.zeros((T, B), device=device),
            np.zeros((T, B), np.float32), np.zeros((T, B), np.float32), np.zeros((T, B), np.float32))
    cfg = {"epochs": args.epochs, "minibatch": args.minibatch, "clip": args.clip,
           "vf_coef": args.vf_coef, "ent_coef": args.ent_coef, "max_grad_norm": args.max_grad_norm}
    print(f"device {device}, obs dim {D}, {sum(p.numel() for p in model.parameters()):,} params, "
          f"canonical {args.canonical}, opponents {dict(zip(opp_names, [round(float(x), 2) for x in w]))} + self-play {args.selfplay}")
    log = open(os.path.join(out, "log.jsonl"), "a")

    # ------------------------------------------------ 1. behaviour cloning
    if it == 0 and args.bc_iters > 0:
        teacher = strat[args.bc_teacher]
        for g in opt.param_groups:
            g["lr"] = args.bc_lr
        print(f"behaviour cloning from {args.bc_teacher}: {args.bc_iters} rollouts of {T * B:,} moves")
        for k in range(args.bc_iters):
            last_v = rollout(env, model, bufs, T, device, teacher=teacher)
            obs_b, mask_b, act_b, _, val_b, rew_b, done_b = bufs
            _, ret = compute_gae(rew_b, done_b, val_b, last_v, args.gamma, args.lam)
            obs, mask, act = obs_b.reshape(T * B, D), mask_b.reshape(T * B, 26), act_b.reshape(-1)
            ret_t = torch.from_numpy(ret.reshape(-1)).to(device)
            model.train()
            idx = np.arange(T * B)
            tot = acc = 0.0; n = 0
            for _ in range(2):
                np.random.shuffle(idx)
                for s in range(0, T * B, args.minibatch):
                    j = torch.from_numpy(idx[s:s + args.minibatch]).to(device)
                    logits, v = model(obs[j], mask[j])
                    loss_pi = Fnn.cross_entropy(logits, act[j])
                    loss = loss_pi + args.vf_coef * 0.5 * ((v - ret_t[j]) ** 2).mean()
                    opt.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    opt.step()
                    tot += loss_pi.item(); acc += (logits.argmax(-1) == act[j]).float().mean().item(); n += 1
            env.pop_episodes()
            print(f"bc {k + 1:3d}/{args.bc_iters} | imitation loss {tot / n:.3f} | "
                  f"move agreement {acc / n:.3f}", flush=True)
            log.write(json.dumps({"bc_iter": k + 1, "bc_loss": tot / n, "bc_acc": acc / n}) + "\n")
        for g in opt.param_groups:
            g["lr"] = args.lr
        ev = evaluate(model, strat["tuned_win"], args.eval_games, args.players, device,
                      objective=args.objective, canonical=args.canonical)
        print(f"after behaviour cloning: EVAL vs tuned_win {ev['win_rate']:.3f}±{ev['ci95']:.3f}, "
              f"round {ev['mean_round']:.2f}", flush=True)
        log.write(json.dumps({"iter": 0, "eval_vs_tuned_win": ev}) + "\n"); log.flush()
        best = ev["win_rate"] if args.objective == "win" else -ev["mean_round"]
        save(os.path.join(out, "best.pt"), model, opt, 0, [], best, args, {"eval": ev})
        pool.append({"state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                     "wr": 0.5})

    # ------------------------------------------------ 2. PPO
    t_ppo = time.time()
    snap = None
    while time.time() < deadline:
        ti = time.time()
        progress = min(1.0, (ti - t_ppo) / max(deadline - t_ppo, 1.0))
        lr = args.lr * (1 - progress * (1 - args.lr_final))
        for g in opt.param_groups:
            g["lr"] = lr
        cfg["ent_coef"] = args.ent_coef + (args.ent_final - args.ent_coef) * progress
        env.shaping = shaping0 * max(0.0, 1 - it / max(args.shaping_decay_iters, 1))
        env.set_probs(probs())
        if pool:
            snap = choose_snapshot(pool)
            frozen.load_state_dict(pool[snap]["state"])

        last_v = rollout(env, model, bufs, T, device)
        obs_b, mask_b, act_b, logp_b, val_b, rew_b, done_b = bufs
        adv, ret = compute_gae(rew_b, done_b, val_b, last_v, args.gamma, args.lam)
        t_roll = time.time() - ti
        model.train()
        batch = (obs_b.reshape(T * B, D), mask_b.reshape(T * B, 26), act_b.reshape(-1),
                 logp_b.reshape(-1), torch.from_numpy(adv.reshape(-1)).to(device),
                 torch.from_numpy(ret.reshape(-1)).to(device),
                 torch.from_numpy(val_b.reshape(-1)).to(device))
        st = ppo_update(model, opt, batch, cfg, device)
        it += 1

        eps = env.pop_episodes()
        aborted = int(eps[:, 4].sum()) if len(eps) else 0
        eps = eps[eps[:, 4] == 0] if len(eps) else eps
        rec = {"iter": it, "time": round(time.time() - t0, 1), "steps_per_s": round(T * B / (time.time() - ti)),
               "rollout_s": round(t_roll, 2), "update_s": round(time.time() - ti - t_roll, 2),
               "lr": lr, "ent_coef": round(cfg["ent_coef"], 5), "shaping": round(env.shaping, 3),
               "episodes": len(eps), **{k: round(v, 4) for k, v in st.items()}}
        for k, name in enumerate(opp_names):
            sel = eps[:, 3] == k
            if sel.sum():
                rec[f"train_win_vs_{name}"] = round(float(eps[sel, 0].mean()), 3)
        sel = eps[:, 3] == -2
        if sel.sum():
            rec["train_win_vs_self"] = round(float(eps[sel, 0].mean()), 3)
            pool[snap]["wr"] = 0.9 * pool[snap]["wr"] + 0.1 * rec["train_win_vs_self"]
        rec["aborted"] = aborted
        if len(eps):
            rec["mean_round"] = round(float(eps[:, 2].sum() / eps[:, 1].sum()), 2)

        if it % args.snapshot_every == 0:
            pool.append({"state": copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}),
                         "wr": 0.5})
            pool = pool[-args.pool_size:]
        if it % args.eval_every == 0:
            model.eval()
            ev = evaluate(model, strat["tuned_win"], args.eval_games, args.players, device,
                          objective=args.objective, canonical=args.canonical)
            rec["eval_vs_tuned_win"] = ev
            score = ev["win_rate"] if args.objective == "win" else -ev["mean_round"]
            if score > best:
                best = score
                save(os.path.join(out, "best.pt"), model, opt, it, [], best, args, {"eval": ev})
        log.write(json.dumps(rec) + "\n"); log.flush()
        ev_s = (f"  EVAL vs tuned_win {rec['eval_vs_tuned_win']['win_rate']:.3f}"
                f"±{rec['eval_vs_tuned_win']['ci95']:.3f}") if "eval_vs_tuned_win" in rec else ""
        print(f"it {it:4d} | {rec['steps_per_s']:>7,} steps/s | lr {lr:.1e} | "
              f"win vs tuned {rec.get('train_win_vs_tuned_win', float('nan')):.3f} "
              f"phase1 {rec.get('train_win_vs_phase1', float('nan')):.3f} "
              f"self {rec.get('train_win_vs_self', float('nan')):.3f} | "
              f"round {rec.get('mean_round', float('nan')):5.1f} | ent {st['entropy']:.2f} kl {st['approx_kl']:.4f}"
              + ev_s, flush=True)
        if it % 5 == 0:
            save(latest, model, opt, it, pool, best, args)
    save(latest, model, opt, it, pool, best, args)


if __name__ == "__main__":
    main()
