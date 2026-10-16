"""Assemble model, tightenings, terminal sets, and controllers; run offline checks."""

from __future__ import annotations

import numpy as np

from baselines import cov_factor, cov_tightening, orc_tightening, was_radius, was_tightening
from disturbances import law_parameters, wind_direction
from mpc import IFMPC
from offline import build_tightening, nesting_check, origin_check
from quadrotor import build_model, feedback_checks
from terminal import constraint_polytope, max_invariant_set, verify_terminal

CONTROLLERS = ("DR", "IF")


def orc_name(delta):
    return f"ORC_d{delta:g}"


def scenario_controllers(cfg, delta):
    """Controller names simulated in one scenario, in reporting order."""
    return ["DR", "IF", "COV", "WAS", orc_name(delta)]


def label(name):
    return "ORC" if name.startswith("ORC_") else name


def _tightenings(cfg, mdl, T_len, only):
    rho = cfg["ambiguity"]["rho_w"]
    want = lambda name: only is None or name in only
    tight, extra = {}, {}
    tight["DR"] = build_tightening(mdl, rho, T_len)
    tight["IF"] = build_tightening(mdl, 0.0, T_len)
    if want("COV"):
        tight["COV"] = cov_tightening(mdl, rho, T_len)
        extra["COV"] = {"variance_factor": cov_factor(mdl, rho)}
    if want("WAS"):
        rho_W = was_radius(mdl, tight["DR"])
        tight["WAS"] = was_tightening(mdl, rho_W, T_len)
        extra["WAS"] = {"rho_W": rho_W}
    amb = cfg["ambiguity"]
    d = wind_direction(mdl, amb["wind_channels"])
    for dl in cfg["sim"]["delta_grid"]:
        name = orc_name(dl)
        if not want(name):
            continue
        mu, Sigma = law_parameters(mdl.Sigma_w, d, dl * rho, amb["eta"])
        tight[name], info = orc_tightening(mdl, mu, Sigma, T_len, T0=cfg["baselines"]["orc_T0"])
        extra[name] = info
    return {c: tg for c, tg in tight.items() if want(c)}, extra


def build_all(cfg, verbose=False, only=None):
    mdl = build_model(cfg)
    N, T = cfg["control"]["N"], cfg["sim"]["T"]
    T_len = T + N + 2
    ref = np.array(cfg["plant"]["reference_phys"]) / np.diag(mdl.Sx)
    tight, extra = _tightenings(cfg, mdl, T_len, only)
    out = {"model": mdl, "ref": ref, "checks": {"feedback": feedback_checks(mdl)},
           "tight": {}, "Xf": {}, "ctrl": {}}
    for name, tg in tight.items():
        C = constraint_polytope(mdl, tg.b_inf, tg.c_inf)
        chk = {"origin_in_sets_at_infinity": origin_check(tg), "p": tg.p,
               "gamma_p": tg.gamma_p, "rho_bar_inf": tg.rho_bar_inf,
               "rho_lower": tg.rho_lower, "gap": tg.gap, "nesting": nesting_check(tg),
               **extra.get(name, {})}
        Xf, info = (None, {"status": "skipped"})
        if chk["origin_in_sets_at_infinity"]:
            Xf, info = max_invariant_set(mdl.A_K, C)
        chk["terminal"] = info
        if Xf is not None and info["status"] == "converged":
            chk["terminal_verify"] = verify_terminal(mdl.A_K, Xf, C)
        else:
            chk["terminal_verify"] = {"ok": False}
        out["tight"][name], out["Xf"][name] = tg, Xf
        out["checks"][name] = chk
        if chk["terminal_verify"]["ok"] and chk["nesting"]["ok"]:
            out["ctrl"][name] = IFMPC(mdl, tg, Xf, N, ref, **solver_kwargs(cfg))
            out["checks"][name]["qp_constraints"] = int(out["ctrl"][name].Aqp.shape[0])
            out["checks"][name]["qp_variables"] = int(out["ctrl"][name].Aqp.shape[1])
    if verbose:
        for kk, vv in out["checks"].items():
            print(kk, vv)
    return out


def solver_kwargs(cfg):
    """IFMPC solver arguments from cfg['solver']."""
    s = cfg["solver"]
    if s["scheme"] != "clarabel_v3":
        raise ValueError(f"unsupported solver scheme {s['scheme']!r}")
    return {"clarabel_tol": s["clarabel_tol"], "tau_feas": s["tau_feas"]}


def step_record(name, log, **extra):
    """JSON-ready failure record of one step (float64 values kept as Python floats)."""
    rec = {"controller": name, "source": log.source, "reason": log.reason, "status": log.status,
           "iterations": log.iterations, "r_prim": log.r_prim, "r_dual": log.r_dual,
           "max_violation": log.max_violation, "dyn_residual": log.dyn_residual,
           "z0_violation": log.z0_violation, "parts": dict(log.parts)}
    if log.instance is not None:
        rec["instance"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in log.instance.items()}
    rec.update(extra)
    return rec


def initial_plans(prob):
    """Solve at k=0 (e=0) for each controller; return the accepted plans (None if rejected)."""
    mdl = prob["model"]
    plans, logs = {}, {}
    for name, c in prob["ctrl"].items():
        plans[name], logs[name] = c.step(0, mdl.x0.copy(), np.zeros(mdl.n))
    return plans, logs


def initial_failures(prob, plans, logs):
    """Failure records for controllers whose initial plan was not accepted."""
    x0 = prob["model"].x0
    return [step_record(name, logs[name], k=0, x0=np.asarray(x0, dtype=float).tolist())
            for name in prob["ctrl"] if plans[name] is None]


def sensitivity_check(prob, plans, name="DR", n_dirs=6, scale=0.05, seed=0):
    """Max change of v0* over random error directions at test points with slack.

    Reuses the accepted initial plan; every further solve is checked before its plan is read.
    Returns (values, failure record or None).
    """
    rng = np.random.default_rng(seed)
    c = prob["ctrl"][name]
    mdl = prob["model"]
    out = []
    plan0 = plans[name]
    if plan0 is None:
        return None, {"controller": name, "reason": "initial plan not accepted"}
    z_tests = [mdl.x0.copy(), plan0.z[5], plan0.z[10]]
    for ti, z in enumerate(z_tests):
        e0 = np.zeros(mdl.n)
        base, log = c.step(0, z, e0)
        if base is None:
            return None, step_record(name, log, test_point=ti, direction=None,
                                     z=z.tolist(), e=e0.tolist())
        dv = []
        for di in range(n_dirs):
            d = rng.standard_normal(mdl.n)
            d *= scale / np.linalg.norm(d)
            p, log = c.step(0, z, d)
            if p is None:
                return None, step_record(name, log, test_point=ti, direction=di,
                                         z=z.tolist(), e=d.tolist())
            dv.append(np.max(np.abs(p.v[0] - base.v[0])))
        out.append(float(max(dv)))
    return out, None
