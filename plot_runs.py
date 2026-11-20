"""Plot learning curves of one or more training runs.

  python plot_runs.py                 # all runs in runs/
  python plot_runs.py win2p v2        # selected runs
Writes runs/learning_curves.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(name):
    rows = [json.loads(l) for l in open(os.path.join("runs", name, "log.jsonl"))]
    ppo = [r for r in rows if "iter" in r and "time" in r]
    evals = [r for r in rows if "eval_vs_tuned_win" in r]
    return ppo, evals


def main():
    names = sys.argv[1:] or sorted(d for d in os.listdir("runs")
                                   if os.path.exists(os.path.join("runs", d, "log.jsonl")))
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for name in names:
        ppo, evals = load(name)
        if not ppo:
            continue
        t = lambda r: r.get("time", 0) / 60
        ev_t = [0.0 if r["iter"] == 0 else t(r) for r in evals]
        ev_w = [r["eval_vs_tuned_win"]["win_rate"] for r in evals]
        ev_c = [r["eval_vs_tuned_win"]["ci95"] for r in evals]
        line, = ax[0, 0].plot(ev_t, ev_w, marker="o", ms=3, label=name)
        ax[0, 0].fill_between(ev_t, [w - c for w, c in zip(ev_w, ev_c)],
                              [w + c for w, c in zip(ev_w, ev_c)], color=line.get_color(), alpha=0.15)
        r = [x for x in ppo if "mean_round" in x]
        ax[0, 1].plot([t(x) for x in r], [x["mean_round"] for x in r], lw=0.8, label=name)
        ax[1, 0].plot([t(x) for x in ppo], [x["entropy"] for x in ppo], lw=0.8, label=name)
        s = [x for x in ppo if "train_win_vs_self" in x]
        ax[1, 1].plot([t(x) for x in s], [x["train_win_vs_self"] for x in s], lw=0.5, alpha=0.7, label=name)
    ax[0, 0].axhline(0.5, color="grey", ls="--", lw=0.8)
    ax[0, 0].set(title="Eval win rate vs tuned_win (best move, 95% CI)", xlabel="minutes", ylabel="win rate")
    ax[0, 1].set(title="Own mean round score (training games)", xlabel="minutes", ylabel="points")
    ax[1, 0].set(title="Policy entropy", xlabel="minutes")
    ax[1, 1].set(title="Win rate vs self-play snapshot", xlabel="minutes")
    for a in ax.flat:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join("runs", "learning_curves.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
