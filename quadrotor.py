"""Linearized planar quadrotor in dimensionless deviation coordinates.

State x = (p_x, p_z, theta, v_x, v_z, omega), input u = (T - m g, tau).
Scaled coordinates: xbar = S_x^{-1}(x - x_star), ubar = S_u^{-1}(u - u_star).
All controller quantities (A, B, Sigma_w, K, Q, R, P, constraints) live in the
scaled coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import expm, solve_discrete_are, solve_discrete_lyapunov


@dataclass
class Model:
    A: np.ndarray
    B: np.ndarray
    Sx: np.ndarray
    Su: np.ndarray
    Sigma_w: np.ndarray
    H: np.ndarray          # state rows h_j^T (scaled)
    b: np.ndarray
    G: np.ndarray          # input rows g_l^T (scaled)
    c: np.ndarray
    state_names: list
    input_names: list
    K: np.ndarray
    A_K: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    P: np.ndarray
    x0: np.ndarray         # scaled initial state
    eps: float

    @property
    def n(self):
        return self.A.shape[0]

    @property
    def m(self):
        return self.B.shape[1]


def _discretize(Ac, Bc, Ts):
    n, m = Bc.shape
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    E = expm(M * Ts)
    return E[:n, :n], E[:n, n:]


def physical_dynamics(cfg):
    p = cfg["plant"]
    g, mass, J = p["g"], p["mass"], p["inertia"]
    Ac = np.zeros((6, 6))
    Ac[0, 3] = Ac[1, 4] = Ac[2, 5] = 1.0
    Ac[3, 2] = g                      # horizontal acceleration from tilt
    Bc = np.zeros((6, 2))
    Bc[4, 0] = 1.0 / mass             # thrust deviation
    Bc[5, 1] = 1.0 / J                # torque
    return _discretize(Ac, Bc, p["Ts"])


def _box_rows(lo, hi, n):
    """Rows h^T x <= b for lo_i <= x_i <= hi_i (skip infinite bounds)."""
    H, b = [], []
    for i in range(n):
        if np.isfinite(hi[i]):
            r = np.zeros(n); r[i] = 1.0
            H.append(r); b.append(hi[i])
        if np.isfinite(lo[i]):
            r = np.zeros(n); r[i] = -1.0
            H.append(r); b.append(-lo[i])
    return np.array(H), np.array(b)


def lqr(A, B, Q, R):
    X = solve_discrete_are(A, B, Q, R)
    return -np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A)


def build_model(cfg) -> Model:
    p = cfg["plant"]
    Ad, Bd = physical_dynamics(cfg)
    Sx = np.diag(p["state_scale"])
    Su = np.diag(p["input_scale"])
    Sxi, Sui = np.linalg.inv(Sx), np.linalg.inv(Su)
    A = Sxi @ Ad @ Sx
    B = Sxi @ Bd @ Su

    sig = np.array(p["noise_std_phys"])
    Sigma_w = Sxi @ np.diag(sig ** 2) @ Sxi

    # Physical deviation limits -> scaled limits.
    lo_x = np.array(p["state_lo_phys"], dtype=float) / np.diag(Sx)
    hi_x = np.array(p["state_hi_phys"], dtype=float) / np.diag(Sx)
    H, b = _box_rows(lo_x, hi_x, 6)
    lo_u = np.array(p["input_lo_phys"], dtype=float) / np.diag(Su)
    hi_u = np.array(p["input_hi_phys"], dtype=float) / np.diag(Su)
    G, c = _box_rows(lo_u, hi_u, 2)

    ctrl = cfg["control"]
    QK, RK = np.diag(ctrl["QK_diag"]), np.diag(ctrl["RK_diag"])
    Q, R = np.diag(ctrl["Q_diag"]), np.diag(ctrl["R_diag"])
    K = lqr(A, B, QK, RK)
    A_K = A + B @ K
    # A_K^T P A_K - P = -(Q + K^T R K)
    P = solve_discrete_lyapunov(A_K.T, Q + K.T @ R @ K)

    x0 = np.array(p["x0_phys"], dtype=float) / np.diag(Sx)
    return Model(A=A, B=B, Sx=Sx, Su=Su, Sigma_w=Sigma_w, H=H, b=b, G=G, c=c,
                 state_names=["p_x", "p_z", "theta", "v_x", "v_z", "omega"],
                 input_names=["thrust", "torque"],
                 K=K, A_K=A_K, Q=Q, R=R, P=P, x0=x0, eps=cfg["risk"]["eps"])


def feedback_checks(mdl: Model) -> dict:
    """C = R K + B^T P A_K vanishes when K is the LQR gain of (Q, R)."""
    C = mdl.R @ mdl.K + mdl.B.T @ mdl.P @ mdl.A_K
    rho = max(abs(np.linalg.eigvals(mdl.A_K)))
    return {
        "C_norm": float(np.linalg.norm(C, 2)),
        "RK_norm": float(np.linalg.norm(mdl.R @ mdl.K, 2)),
        "C_ratio": float(np.linalg.norm(C, 2) / np.linalg.norm(mdl.R @ mdl.K, 2)),
        "spectral_radius_AK": float(rho),
        "norm_AK": float(np.linalg.norm(mdl.A_K, 2)),
    }
