"""Indirect-feedback MPC QP (eq:mpc) solved with Clarabel, with plan verification.

Decision vector y = [z_0, ..., z_N, v_0, ..., v_{N-1}].
Cost: sum_i (z_i + ebar_i - r)' Q (.) + (v_i + K ebar_i)' R (.) + (z_N + ebar_N - r)' P (.),
with ebar_i = A_K^i e(k). The reference r only enters the objective.

The state rows on the fixed z_0 = z(k) are not passed to the solver; z(k) in X(k) is checked by
`check` with tolerance tau_feas instead. Without these rows the QP is equivalent to (eq:mpc) only
if z(k) in X(k) holds.

A step is accepted only if Clarabel returns "Solved", the solution is finite, and the plan rolled
out from v passes `check` (state rows i = 0..N-1, input rows, terminal set) with tolerance
tau_feas. There is no fallback.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import clarabel
import numpy as np
import scipy.sparse as sp

from offline import Tightening
from quadrotor import Model
from terminal import Polytope


@dataclass
class Plan:
    v: np.ndarray        # (N, m)
    z: np.ndarray        # (N+1, n), rolled out with exact nominal dynamics
    max_violation: float
    dyn_residual: float
    parts: dict = field(default_factory=dict)   # state0, state1, state_rest, input, terminal


@dataclass
class SolveResult:
    solver: str
    status: str
    x: np.ndarray | None
    iterations: int
    r_prim: float
    r_dual: float
    time: float


@dataclass
class StepLog:
    source: str          # "clarabel" or "failed"
    qp_time: float
    total_time: float
    max_violation: float
    dyn_residual: float
    status: str = ""     # raw solver status, e.g. "clarabel:Solved"
    iterations: int = -1
    r_prim: float = np.nan
    r_dual: float = np.nan
    reason: str = ""     # empty if the plan was accepted
    z0_violation: float = np.nan   # max_j h_j' z(k) - b_j(k)
    parts: dict = field(default_factory=dict)
    instance: dict | None = None   # problem data (k, z, e, q, u), only on failure


class IFMPC:
    def __init__(self, mdl: Model, tg: Tightening, Xf: Polytope, N: int, ref: np.ndarray,
                 clarabel_tol=1e-9, tau_feas=1e-5):
        self.mdl, self.tg, self.Xf, self.N = mdl, tg, Xf, N
        self.ref = ref
        self.tau = tau_feas
        self.clarabel_tol = clarabel_tol
        n, m = mdl.n, mdl.m
        self.n, self.m = n, m
        self.nz = n * (N + 1)
        self.nv = m * N
        self.AKpow = np.stack([np.linalg.matrix_power(mdl.A_K, i) for i in range(N + 1)])

        # Hessian
        blocks = [2 * mdl.Q] * N + [2 * mdl.P] + [2 * mdl.R] * N
        self.Pqp = sp.block_diag(blocks, format="csc")

        # Constraint matrix
        rows = []
        Iz0 = sp.hstack([sp.eye(n), sp.csc_matrix((n, self.nz - n + self.nv))])
        rows.append(Iz0)
        dyn = sp.lil_matrix((n * N, self.nz + self.nv))
        for i in range(N):
            r0 = i * n
            dyn[r0:r0 + n, (i + 1) * n:(i + 2) * n] = np.eye(n)
            dyn[r0:r0 + n, i * n:(i + 1) * n] = -mdl.A
            dyn[r0:r0 + n, self.nz + i * m:self.nz + (i + 1) * m] = -mdl.B
        rows.append(dyn.tocsc())
        nx, nu = mdl.H.shape[0], mdl.G.shape[0]
        # State rows on z_1, ..., z_{N-1}; the rows on the fixed z_0 are checked outside the solver.
        ns = N - 1
        Hs = sp.hstack([sp.csc_matrix((nx * ns, n)), sp.kron(sp.eye(ns), mdl.H),
                        sp.csc_matrix((nx * ns, n)), sp.csc_matrix((nx * ns, self.nv))])
        rows.append(Hs)
        Gs = sp.hstack([sp.csc_matrix((nu * N, self.nz)), sp.kron(sp.eye(N), mdl.G)])
        rows.append(Gs)
        nf = Xf.H.shape[0]
        Hf = sp.hstack([sp.csc_matrix((nf, n * N)), sp.csc_matrix(Xf.H), sp.csc_matrix((nf, self.nv))])
        rows.append(Hf)
        self.Aqp = sp.vstack(rows, format="csc")
        self.n_eq = n + n * N
        self.n_state_rows = nx * ns
        self.n_ineq = self.n_state_rows + nu * N + nf
        self.nx, self.nu, self.nf = nx, nu, nf

        self.u = np.zeros(self.n_eq + self.n_ineq)
        self.u[-nf:] = Xf.h

        self.clar = self.make_clarabel()

    # ------------------------------------------------------------------ #
    def make_clarabel(self, tol=None):
        """Clarabel (interior point), rows ordered [equalities; inequalities]."""
        tol = self.clarabel_tol if tol is None else tol
        st = clarabel.DefaultSettings()
        st.verbose = False
        st.presolve_enable = False
        st.tol_gap_abs = st.tol_gap_rel = st.tol_feas = tol
        cones = [clarabel.ZeroConeT(self.n_eq), clarabel.NonnegativeConeT(self.n_ineq)]
        return clarabel.DefaultSolver(sp.triu(self.Pqp, format="csc"),
                                      np.zeros(self.nz + self.nv), self.Aqp,
                                      self.u.copy(), cones, st)

    def qp_data(self, k, z, e):
        """Linear cost q and right-hand side u at time k for nominal state z and measured error e."""
        N, n = self.N, self.n
        ebar = self.AKpow @ e                                    # (N+1, n)
        dz = ebar - self.ref
        qz = np.concatenate([(2 * self.mdl.Q @ dz[:N].T).T.ravel(), 2 * self.mdl.P @ dz[N]])
        qv = (2 * self.mdl.R @ (self.mdl.K @ ebar[:N].T)).T.ravel()
        q = np.concatenate([qz, qv])
        u = self.u.copy()
        u[:n] = z
        s = self.n_eq
        u[s:s + self.n_state_rows] = self.tg.b_t[k + 1:k + N].ravel()
        s += self.n_state_rows
        u[s:s + self.nu * N] = self.tg.c_t[k:k + N].ravel()
        return q, u

    def z0_violation(self, k, z):
        return float(np.max(self.mdl.H @ z - self.tg.b_t[k]))

    def rollout(self, z0, v):
        z = np.empty((self.N + 1, self.n))
        z[0] = z0
        for i in range(self.N):
            z[i + 1] = self.mdl.A @ z[i] + self.mdl.B @ v[i]
        return z

    def check(self, k, z0, v, z_solver=None) -> Plan:
        """Exact rollout of v from z0; tightened state (i = 0..N-1), input and terminal rows."""
        z = self.rollout(z0, v)
        N = self.N
        gs = np.einsum("jn,in->ij", self.mdl.H, z[:N]) - self.tg.b_t[k:k + N]
        vi = np.max(np.einsum("ln,in->il", self.mdl.G, v) - self.tg.c_t[k:k + N])
        vf = np.max(self.Xf.H @ z[N] - self.Xf.h)
        parts = {"state0": float(np.max(gs[0])),
                 "state1": float(np.max(gs[1])) if N > 1 else -np.inf,
                 "state_rest": float(np.max(gs[1:])) if N > 1 else -np.inf,
                 "input": float(vi), "terminal": float(vf)}
        dyn = 0.0 if z_solver is None else float(np.max(np.abs(z_solver - z)))
        mv = float(max(np.max(gs), vi, vf))
        if not np.isfinite(mv):
            mv = np.inf
        return Plan(v=v, z=z, max_violation=mv, dyn_residual=dyn, parts=parts)

    def unpack(self, y):
        return y[self.nz:].reshape(self.N, self.m), y[:self.nz].reshape(self.N + 1, self.n)

    def solve_clarabel(self, q, u, solver=None) -> SolveResult:
        s = self.clar if solver is None else solver
        tq = time.perf_counter()
        s.update(q=q, b=u)
        res = s.solve()
        t = time.perf_counter() - tq
        return SolveResult("clarabel", str(res.status), np.array(res.x), int(res.iterations),
                           float(res.r_prim), float(res.r_dual), t)

    def _instance(self, k, z, e, q, u):
        return {"k": int(k), "z": np.array(z, dtype=float), "e": np.array(e, dtype=float),
                "q": q, "u": u}

    # ------------------------------------------------------------------ #
    def step(self, k, z, e):
        """Return (plan, log). The plan's first input is applied; plan is None if rejected."""
        t0 = time.perf_counter()
        q, u = self.qp_data(k, z, e)
        z0v = self.z0_violation(k, z)
        r = self.solve_clarabel(q, u)
        status = f"clarabel:{r.status}"
        plan, reason = None, ""
        if r.status != "Solved":
            reason = f"status:{r.status}"
        elif r.x is None or not np.all(np.isfinite(r.x)):
            reason = "nonfinite"
        else:
            v, zs = self.unpack(r.x)
            plan = self.check(k, z, v, zs)
            if not plan.max_violation <= self.tau:
                reason = "check"
        mv = plan.max_violation if plan is not None else np.inf
        dyn = plan.dyn_residual if plan is not None else np.inf
        parts = plan.parts if plan is not None else {}
        if reason:
            return None, StepLog("failed", r.time, time.perf_counter() - t0, mv, dyn, status,
                                 r.iterations, r.r_prim, r.r_dual, reason, z0v, parts,
                                 self._instance(k, z, e, q, u))
        return plan, StepLog("clarabel", r.time, time.perf_counter() - t0, mv, dyn, status,
                             r.iterations, r.r_prim, r.r_dual, "", z0v, parts)
