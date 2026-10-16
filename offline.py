"""Offline error envelope and constraint tightening (methodology, Sec. III-B/C)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_lyapunov

from quadrotor import Model


def kappa(eps):
    return np.sqrt((1.0 - eps) / eps)


@dataclass
class Tightening:
    rho_w: float
    Sigma_e: np.ndarray      # (T_len, n, n)
    rho_e: np.ndarray        # (T_len,)
    b_t: np.ndarray          # (T_len, n_x) tightened state bounds tilde b_j(t)
    c_t: np.ndarray          # (T_len, n_u) tightened input bounds tilde c_l(t)
    Sigma_inf: np.ndarray
    rho_bar_inf: float
    b_inf: np.ndarray
    c_inf: np.ndarray
    p: int
    gamma_p: float
    rho_lower: float
    gap: float


def power_norms(A_K, count):
    out = np.empty(count)
    M = np.eye(A_K.shape[0])
    for i in range(count):
        out[i] = np.linalg.norm(M, 2)
        M = A_K @ M
    return out


def scan_p(A_K, rho_w, p_max, t_big):
    """Choose p in 1..p_max with gamma_p < 1 minimizing the bound (eq:rho-bar)."""
    norms = power_norms(A_K, max(p_max + 1, t_big))
    best = None
    for p in range(1, p_max + 1):
        g = norms[p]
        if g >= 1.0:
            continue
        bound = norms[:p].sum() / (1.0 - g)
        if best is None or bound < best[2]:
            best = (p, g, bound)
    if best is None:
        raise RuntimeError("no p <= p_max with ||A_K^p|| < 1")
    p, g, s_bar = best
    s_low = norms[:t_big].sum()
    return p, g, rho_w * s_bar, rho_w * s_low


def margins(H, Sigma, rho, eps, c_nom=None, c_rob=None):
    """c_nom sqrt(h' Sigma h) + c_rob rho ||h||, row-wise.

    Defaults are the Gelbrich coefficients c_nom = kappa(eps), c_rob = 1/sqrt(eps).
    """
    quad = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", H, Sigma, H), 0.0))
    nom = (kappa(eps) if c_nom is None else c_nom) * quad
    hn = np.linalg.norm(H, axis=1)
    return nom + (rho * hn / np.sqrt(eps) if c_rob is None else rho * hn * c_rob)


def build_tightening(mdl: Model, rho_w: float, T_len: int, p_max=400, t_big=5000,
                     Sigma_w=None, c_nom=None, c_rob=None):
    n = mdl.n
    A_K, eps = mdl.A_K, mdl.eps
    Sw = mdl.Sigma_w if Sigma_w is None else Sigma_w
    HK = mdl.G @ mdl.K                 # input loss directions (K^T g_l)^T
    norms = power_norms(A_K, T_len)
    Sigma_e = np.zeros((T_len, n, n))
    rho_e = np.zeros(T_len)
    for t in range(1, T_len):
        Sigma_e[t] = A_K @ Sigma_e[t - 1] @ A_K.T + Sw
        rho_e[t] = rho_e[t - 1] + norms[t - 1] * rho_w
    b_t = np.stack([mdl.b - margins(mdl.H, Sigma_e[t], rho_e[t], eps, c_nom, c_rob)
                    for t in range(T_len)])
    c_t = np.stack([mdl.c - margins(HK, Sigma_e[t], rho_e[t], eps, c_nom, c_rob)
                    for t in range(T_len)])

    Sigma_inf = solve_discrete_lyapunov(A_K, Sw)
    p, g, rho_bar, rho_low = scan_p(A_K, rho_w, p_max, t_big)
    gap = 0.0 if rho_w == 0 else (rho_bar - rho_low) / rho_bar
    b_inf = mdl.b - margins(mdl.H, Sigma_inf, rho_bar, eps, c_nom, c_rob)
    c_inf = mdl.c - margins(HK, Sigma_inf, rho_bar, eps, c_nom, c_rob)
    return Tightening(rho_w=rho_w, Sigma_e=Sigma_e, rho_e=rho_e, b_t=b_t, c_t=c_t,
                      Sigma_inf=Sigma_inf, rho_bar_inf=rho_bar, b_inf=b_inf, c_inf=c_inf,
                      p=p, gamma_p=g, rho_lower=rho_low, gap=gap)


def origin_check(tg: Tightening) -> bool:
    return bool(np.all(tg.b_inf > 0) and np.all(tg.c_inf > 0))


def nesting_check(tg: Tightening, tol=1e-12) -> dict:
    """Tightened bounds non-increasing in t and never below the sets at infinity."""
    db = np.diff(tg.b_t, axis=0).max(initial=-np.inf)
    dc = np.diff(tg.c_t, axis=0).max(initial=-np.inf)
    eb = (tg.b_inf[None] - tg.b_t).max()
    ec = (tg.c_inf[None] - tg.c_t).max()
    ok = bool(db <= tol and dc <= tol and eb <= tol and ec <= tol)
    return {"max_increase_b": float(db), "max_increase_c": float(dc),
            "max_below_inf_b": float(eb), "max_below_inf_c": float(ec), "ok": ok}
