"""Normalized losses, empirical CVaR, and trajectory-level paired bootstrap.

State losses use k = 1..T, input losses k = 1..T-1 (u(0) is deterministic and is
reported separately). Simultaneous bands use the unstudentized maximum centered
deviation over a = (controller, row, time) within one scenario.
"""

from __future__ import annotations

import numpy as np


def state_losses(mdl, X):
    """X: (M, T+1, n) -> (M, T, n_x) for k = 1..T."""
    Hn = mdl.H / np.linalg.norm(mdl.H, axis=1, keepdims=True)
    bn = mdl.b / np.linalg.norm(mdl.H, axis=1)
    return X[:, 1:, :] @ Hn.T - bn


def input_losses(mdl, U):
    """U: (M, T, m) -> (M, T, n_u) for k = 0..T-1."""
    Gn = mdl.G / np.linalg.norm(mdl.G, axis=1, keepdims=True)
    cn = mdl.c / np.linalg.norm(mdl.G, axis=1)
    return U @ Gn.T - cn


def empirical_cvar(L, eps, axis=0):
    """Exact CVaR of the empirical distribution along `axis`."""
    L = np.moveaxis(L, axis, 0)
    M = L.shape[0]
    t = eps * M
    k = int(np.floor(t))
    kk = min(M, k + 1)
    top = -np.partition(-L, kk - 1, axis=0)[:kk]
    top = -np.sort(-top, axis=0)
    s = top[:k].sum(axis=0) if k > 0 else 0.0
    frac = t - k
    if frac > 0:
        s = s + frac * top[k]
    return s / t


def bootstrap(L, eps, B, level, seed):
    """L: (M, ...) with paired trajectories on axis 0.

    Returns dict with estimate, pointwise percentile bands, simultaneous half-width c.
    """
    rng = np.random.default_rng(seed)
    M = L.shape[0]
    theta = empirical_cvar(L, eps)
    boots = np.empty((B,) + theta.shape)
    for b in range(B):
        idx = rng.integers(0, M, size=M)
        boots[b] = empirical_cvar(L[idx], eps)
    alpha = 1.0 - level
    lo = np.quantile(boots, alpha / 2, axis=0)
    hi = np.quantile(boots, 1 - alpha / 2, axis=0)
    dev = np.abs(boots - theta).reshape(B, -1).max(axis=1)
    c = float(np.quantile(dev, level))
    return {"theta": theta, "pw_lo": lo, "pw_hi": hi, "c": c}


def scenario_summary(mdl, runs, eps, B, level, seed, controllers=("DR", "IF")):
    """Risk statistics for one scenario. runs[name] has X, U, complete."""
    Ls = np.stack([state_losses(mdl, runs[c]["X"]) for c in controllers], axis=1)      # (M,C,T,nx)
    Lu_all = np.stack([input_losses(mdl, runs[c]["U"]) for c in controllers], axis=1)  # (M,C,T,nu)
    u0 = Lu_all[0, :, 0, :]                     # deterministic u(0) losses (C, nu)
    Lu = Lu_all[:, :, 1:, :]
    incomplete = {c: int((~runs[c]["complete"].astype(bool)).sum()) for c in controllers}
    out = {"incomplete": incomplete, "u0_loss_max": u0.max(axis=1).tolist()}
    for tag, L, sd in (("state", Ls, seed), ("input", Lu, seed + 1)):
        if np.isnan(L).any():
            L = np.where(np.isnan(L), np.inf, L)
        bs = bootstrap(L, eps, B, level, sd)
        th, c = bs["theta"], bs["c"]
        worst = th.reshape(len(controllers), -1).max(axis=1)
        out[tag] = {
            "theta": th, "pw_lo": bs["pw_lo"], "pw_hi": bs["pw_hi"], "c": c,
            "worst": worst,
            "worst_lo": (th - c).reshape(len(controllers), -1).max(axis=1),
            "worst_hi": (th + c).reshape(len(controllers), -1).max(axis=1),
            "viol_freq": (L > 0).mean(axis=0),
        }
    return out


def closed_loop_cost(mdl, ref, X, U):
    """Per-trajectory (1/T) sum_k (x-r)'Q(x-r) + u'Ru over k = 0..T-1."""
    dx = X[:, :-1, :] - ref
    sx = np.einsum("mti,ij,mtj->mt", dx, mdl.Q, dx)
    su = np.einsum("mti,ij,mtj->mt", U, mdl.R, U)
    return (sx + su).mean(axis=1)
