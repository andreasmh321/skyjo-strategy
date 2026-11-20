"""PPO with action masking for Skyjo.

Network: shared MLP trunk -> policy logits (26 actions, illegal ones masked
to -inf before the softmax) and a value head.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .env import VecEnv

N_ACTIONS = 26


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 512, layers: int = 3):
        super().__init__()
        mods, d = [], obs_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.ReLU()]
            d = hidden
        self.trunk = nn.Sequential(*mods)
        self.pi = nn.Linear(hidden, N_ACTIONS)
        self.v = nn.Linear(hidden, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, math.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.pi.weight, 0.01)
        nn.init.orthogonal_(self.v.weight, 1.0)
        self.config = {"obs_dim": obs_dim, "hidden": hidden, "layers": layers}

    def forward(self, obs, mask):
        h = self.trunk(obs)
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)

    @torch.no_grad()
    def act(self, obs, mask, deterministic=False):
        logits, v = self(obs, mask)
        if deterministic:
            a = logits.argmax(-1)
        else:
            a = torch.distributions.Categorical(logits=logits).sample()
        logp = torch.log_softmax(logits, -1).gather(1, a[:, None]).squeeze(1)
        return a, logp, v


def make_policy_fn(model: ActorCritic, device, deterministic=False):
    """numpy (obs, mask) -> numpy actions, for net opponents in VecEnv."""
    def fn(obs, mask):
        o = torch.from_numpy(obs).to(device)
        m = torch.from_numpy(mask).to(device)
        a, _, _ = model.act(o, m, deterministic)
        return a.cpu().numpy()
    return fn


def load_model(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = ActorCritic(**ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


@torch.no_grad()
def evaluate(model, opponent: dict, n_games=2000, n_players=2, device="cpu",
             deterministic=True, n_envs=500, seed=12345, objective="win", canonical=True):
    """Play `n_games` (each env plays the same number) against `opponent` in all
    other seats. Returns win rate, 95% CI half-width, mean own round score."""
    n_envs = min(n_envs, n_games)
    quota = math.ceil(n_games / n_envs)
    env = VecEnv(n_envs, n_players=n_players, opponents=[opponent], probs=[1.0, 0.0],
                 objective=objective, seed=seed, canonical=canonical)
    taken = np.zeros(n_envs, np.int64)
    rows = []
    guard = 0
    while (taken < quota).any():
        o = torch.from_numpy(env.obs).to(device)
        m = torch.from_numpy(env.masks).to(device)
        a, _, _ = model.act(o, m, deterministic)
        env.step(a.cpu().numpy())
        for b in np.flatnonzero(env.ep_n):
            for j in range(env.ep_n[b]):
                if taken[b] < quota:
                    rows.append(env.ep_out[b, j].copy())
                    taken[b] += 1
        env.ep_n[:] = 0
        guard += 1
        if guard > 2_000_000:
            raise RuntimeError("evaluation did not finish")
    rows = np.array(rows)
    aborted = int(rows[:, 4].sum())
    rows = rows[rows[:, 4] == 0]
    p = rows[:, 0].mean()
    return {"win_rate": float(p), "ci95": float(1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(rows))),
            "mean_round": float(rows[:, 2].sum() / rows[:, 1].sum()), "games": len(rows),
            "aborted": aborted}


def ppo_update(model, opt, batch, cfg, device):
    obs, masks, acts, old_logp, adv, ret, old_v = batch
    n = obs.shape[0]
    idx = np.arange(n)
    stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    k = 0
    for _ in range(cfg["epochs"]):
        np.random.shuffle(idx)
        for s in range(0, n, cfg["minibatch"]):
            j = torch.from_numpy(idx[s:s + cfg["minibatch"]]).to(device)
            logits, v = model(obs[j], masks[j])
            logp_all = torch.log_softmax(logits, -1)
            logp = logp_all.gather(1, acts[j][:, None]).squeeze(1)
            ratio = torch.exp(logp - old_logp[j])
            a = adv[j]
            a = (a - a.mean()) / (a.std() + 1e-8)
            pi_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg["clip"], 1 + cfg["clip"]) * a).mean()
            v_clip = old_v[j] + (v - old_v[j]).clamp(-cfg["clip"], cfg["clip"])
            v_loss = 0.5 * torch.max((v - ret[j]) ** 2, (v_clip - ret[j]) ** 2).mean()
            probs = logp_all.exp()
            ent = -(probs * logp_all.masked_fill(~masks[j], 0.0)).sum(-1).mean()
            loss = pi_loss + cfg["vf_coef"] * v_loss - cfg["ent_coef"] * ent
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            opt.step()
            with torch.no_grad():
                stats["pi_loss"] += pi_loss.item()
                stats["v_loss"] += v_loss.item()
                stats["entropy"] += ent.item()
                stats["approx_kl"] += ((ratio - 1) - torch.log(ratio)).mean().item()
                stats["clip_frac"] += ((ratio - 1).abs() > cfg["clip"]).float().mean().item()
            k += 1
    return {key: v / k for key, v in stats.items()}


def compute_gae(rew, done, val, last_val, gamma, lam):
    T = rew.shape[0]
    adv = np.zeros_like(rew)
    gae = np.zeros(rew.shape[1], np.float32)
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        gae = delta + gamma * lam * nonterm * gae
        adv[t] = gae
    return adv, adv + val
