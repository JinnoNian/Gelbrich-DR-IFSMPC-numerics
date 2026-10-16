"""Terminal set: maximal A_K-invariant subset of the sets at infinity.

Gilbert--Tan / Kolmanovsky--Gilbert iteration with an explicit convergence
status and an independent LP verification of invariance and inclusion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog


@dataclass
class Polytope:
    H: np.ndarray
    h: np.ndarray


def lp_max(c, H, h):
    """max c^T z s.t. H z <= h. Returns (value, status)."""
    res = linprog(-c, A_ub=H, b_ub=h, bounds=[(None, None)] * H.shape[1], method="highs")
    if res.status == 0:
        return -res.fun, "optimal"
    if res.status == 3:
        return np.inf, "unbounded"
    if res.status == 2:
        return -np.inf, "infeasible"
    return np.nan, f"status{res.status}"


def remove_redundant(P: Polytope, tol=1e-9) -> Polytope:
    keep = np.ones(len(P.h), dtype=bool)
    for i in range(len(P.h)):
        keep[i] = False
        val, st = lp_max(P.H[i], P.H[keep], P.h[keep])
        if st != "optimal" or val > P.h[i] + tol:
            keep[i] = True
    return Polytope(P.H[keep], P.h[keep])


def constraint_polytope(mdl, b_inf, c_inf) -> Polytope:
    """C = tilde X(inf) cap {z : K z in tilde U(inf)}."""
    H = np.vstack([mdl.H, mdl.G @ mdl.K])
    h = np.concatenate([b_inf, c_inf])
    return Polytope(H, h)


def max_invariant_set(A_K, C: Polytope, t_max=200, tol=1e-9):
    """Return (X_f, info). info['status'] is 'converged' or 'max_iter'."""
    O = Polytope(C.H.copy(), C.h.copy())
    Ak_pow = np.eye(A_K.shape[0])
    for t in range(1, t_max + 1):
        Ak_pow = Ak_pow @ A_K
        Hn = C.H @ Ak_pow
        new_rows = []
        for i in range(len(C.h)):
            val, st = lp_max(Hn[i], O.H, O.h)
            if st == "infeasible":
                return None, {"status": "empty", "iterations": t}
            if st != "optimal" or val > C.h[i] + tol:
                new_rows.append(i)
        if not new_rows:
            O = remove_redundant(O)
            return O, {"status": "converged", "iterations": t, "facets": len(O.h)}
        O = Polytope(np.vstack([O.H, Hn[new_rows]]), np.concatenate([O.h, C.h[new_rows]]))
        if len(O.h) > 400:
            O = remove_redundant(O)
    return O, {"status": "max_iter", "iterations": t_max, "facets": len(O.h)}


def verify_terminal(A_K, Xf: Polytope, C: Polytope, tol=1e-7) -> dict:
    """Independent checks: A_K X_f subset X_f and X_f subset C."""
    inv = max(lp_max(Xf.H[i] @ A_K, Xf.H, Xf.h)[0] - Xf.h[i] for i in range(len(Xf.h)))
    inc = max(lp_max(C.H[i], Xf.H, Xf.h)[0] - C.h[i] for i in range(len(C.h)))
    origin = bool(np.all(Xf.h > 0))
    return {"invariance_excess": float(inv), "inclusion_excess": float(inc),
            "origin_interior": origin,
            "ok": bool(inv <= tol and inc <= tol and origin)}
