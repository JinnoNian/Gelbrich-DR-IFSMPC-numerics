"""Offline tightenings of the comparison controllers.

All share K, P, the cost, and the solver pipeline with DR-MPC; only the tightening
(and the terminal set it induces) differs.

COV  homothetic covariance inflation: IF-SMPC with Sigma_w scaled by
     (1 + rho_w / sqrt(tr Sigma_w))^2, no mean term.
WAS  Gaussian-centered type-1 Wasserstein IF baseline: the ball is centered at the
     nominal Gaussian error law, so the margin is the Gaussian CVaR plus
     rho_W sum_i ||A_K^i|| ||h|| / eps. rho_W gives the DR-MPC steady-state wall margin.
ORC  true-moment tightening oracle for one delta: kappa-CVaR margin with the true
     error mean and covariance; used only in the tightening, not in the cost.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import norm

from offline import Tightening, build_tightening, kappa, power_norms, scan_p

WALL_ROW = 0


def cov_factor(mdl, rho_w):
    return (1.0 + rho_w / np.sqrt(np.trace(mdl.Sigma_w))) ** 2


def cov_tightening(mdl, rho_w, T_len):
    return build_tightening(mdl, 0.0, T_len, Sigma_w=cov_factor(mdl, rho_w) * mdl.Sigma_w)


def gaussian_cvar_coeff(eps):
    return float(norm.pdf(norm.ppf(1.0 - eps)) / eps)


def was_radius(mdl, tg_dr):
    """rho_W such that the steady-state wall margin equals that of DR-MPC (linear in rho_W)."""
    eps = mdl.eps
    h = mdl.H[WALL_ROW]
    m_dr = mdl.b[WALL_ROW] - tg_dr.b_inf[WALL_ROW]
    sig = np.sqrt(h @ tg_dr.Sigma_inf @ h)
    s_bar = tg_dr.rho_bar_inf / tg_dr.rho_w           # p-step bound of sum_i ||A_K^i||
    return float(eps * (m_dr - gaussian_cvar_coeff(eps) * sig) / (s_bar * np.linalg.norm(h)))


def was_tightening(mdl, rho_W, T_len):
    eps = mdl.eps
    return build_tightening(mdl, rho_W, T_len, c_nom=gaussian_cvar_coeff(eps), c_rob=1.0 / eps)


def orc_tightening(mdl, mu, Sigma, T_len, T0=2000, p_max=400, t_big=5000):
    """True-moment margins m(t) = h' mu_e(t) + kappa sqrt(h' Sigma_e(t) h).

    Used bounds: prefix maximum (from m(0) = 0, hence nonnegative). Sets at infinity:
    max of the prefix maximum over t < T0 and a bounded tail for t >= T0.
    """
    n, eps, A_K = mdl.n, mdl.eps, mdl.A_K
    k = kappa(eps)
    HK = mdl.G @ mdl.K
    rows = {"x": mdl.H, "u": HK}
    mu_e = np.zeros(n)
    S = np.zeros((n, n))
    raw = {r: np.empty((T0, H.shape[0])) for r, H in rows.items()}
    Sig_store = np.zeros((T_len, n, n))
    for t in range(T0):
        for r, H in rows.items():
            q = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", H, S, H), 0.0))
            raw[r][t] = H @ mu_e + k * q
        if t < T_len:
            Sig_store[t] = S
        mu_e = A_K @ mu_e + mu
        S = A_K @ S @ A_K.T + Sigma

    p, g, _, _ = scan_p(A_K, 1.0, p_max, t_big)
    head = power_norms(A_K, p).sum()
    tail_sum = g ** (T0 // p) * head / (1.0 - g)
    mu_inf = np.linalg.solve(np.eye(n) - A_K, mu)
    S_inf = solve_discrete_lyapunov(A_K, Sigma)
    env, info = {}, {}
    for r, H in rows.items():
        hn = np.linalg.norm(H, axis=1)
        steady = H @ mu_inf + k * np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", H, S_inf, H), 0.0))
        tail = steady + hn * np.linalg.norm(mu) * tail_sum
        prefix = raw[r].max(axis=0)
        env[r] = np.maximum(np.maximum(prefix, tail), 0.0)
        info[r] = {"transient_excess_max": float((prefix - steady).max()),
                   "raw_min": float(raw[r].min()), "tail_sum_bound": float(tail_sum)}
    run = {r: np.maximum.accumulate(raw[r][:T_len], axis=0) for r in rows}
    tg = Tightening(rho_w=0.0, Sigma_e=Sig_store, rho_e=np.zeros(T_len),
                    b_t=mdl.b - run["x"], c_t=mdl.c - run["u"], Sigma_inf=S_inf,
                    rho_bar_inf=0.0, b_inf=mdl.b - env["x"], c_inf=mdl.c - env["u"],
                    p=p, gamma_p=g, rho_lower=0.0, gap=0.0)
    return tg, info
