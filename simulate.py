"""Paired closed-loop rollouts: all controllers see the same disturbance array."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from problem import CONTROLLERS, build_all, initial_failures, initial_plans, step_record

SOURCE_CODE = {"init": 0, "clarabel": 1, "failed": 4}
RUN_KEYS = ("X", "Z", "U", "source", "qp_time", "total_time", "max_violation", "z0_violation",
            "iterations", "r_prim", "r_dual", "complete")

_PROB = None
_INIT = None
_NAMES = CONTROLLERS
_INIT_FAILURE = None


def default_workers():
    return max(1, (os.cpu_count() or 2) - 2)


def _worker_init(cfg, names, fail_dir=None):
    """Build the controllers in a worker. Failures are stored, not raised: an exception in a
    ProcessPoolExecutor initializer only reaches the parent as BrokenProcessPool."""
    global _PROB, _INIT, _NAMES, _INIT_FAILURE
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _NAMES = tuple(names)
    try:
        _PROB = build_all(cfg, only=_NAMES)
        missing = [c for c in _NAMES if c not in _PROB["ctrl"]]
        plans, logs = initial_plans(_PROB)
        _INIT = plans
        recs = [{"controller": c, "reason": "controller not built"} for c in missing]
        recs += initial_failures(_PROB, plans, logs)
    except Exception as exc:   # recorded and returned by _run_chunk
        recs = [{"reason": "worker init exception", "error": repr(exc)}]
    if recs:
        _INIT_FAILURE = [dict(r, stage="worker_init", pid=os.getpid()) for r in recs]
        if fail_dir is not None:
            p = Path(fail_dir) / f"failures_init_{os.getpid()}.json"
            p.write_text(json.dumps(_INIT_FAILURE, indent=1, default=float))


def rollout_one(ctrl, mdl, init_plan, w, traj=None, name=None):
    """One closed-loop trajectory. Returns (arrays, failure records)."""
    T = w.shape[0]
    n, m = mdl.n, mdl.m
    out = {"X": np.full((T + 1, n), np.nan), "Z": np.full((T + 1, n), np.nan),
           "U": np.full((T, m), np.nan),
           "source": np.full(T, SOURCE_CODE["failed"], dtype=np.int8),
           "qp_time": np.zeros(T), "total_time": np.zeros(T),
           "max_violation": np.full(T, np.nan), "z0_violation": np.full(T, np.nan),
           "iterations": np.full(T, -1, dtype=np.int32),
           "r_prim": np.full(T, np.nan), "r_dual": np.full(T, np.nan)}
    failures = []
    x = mdl.x0.copy()
    z = mdl.x0.copy()
    out["X"][0], out["Z"][0] = x, z
    complete = init_plan is not None
    if not complete:
        failures.append({"controller": name, "traj": traj, "k": 0,
                         "reason": "initial plan not accepted"})
    for k in range(T if complete else 0):
        e = x - z
        if k == 0:
            plan, s = init_plan, "init"
            out["max_violation"][k] = init_plan.max_violation
            out["z0_violation"][k] = ctrl.z0_violation(0, z)
        else:
            plan, log = ctrl.step(k, z, e)
            s = log.source
            out["qp_time"][k], out["total_time"][k] = log.qp_time, log.total_time
            out["max_violation"][k], out["z0_violation"][k] = log.max_violation, log.z0_violation
            out["iterations"][k], out["r_prim"][k], out["r_dual"][k] = (log.iterations, log.r_prim,
                                                                        log.r_dual)
            if s == "failed":
                failures.append(step_record(name, log, traj=traj, k=k))
        out["source"][k] = SOURCE_CODE[s]
        if s == "failed":
            complete = False
            break
        v0 = plan.v[0]
        u = v0 + mdl.K @ e
        out["U"][k] = u
        x = mdl.A @ x + mdl.B @ u + w[k]
        z = mdl.A @ z + mdl.B @ v0
        out["X"][k + 1], out["Z"][k + 1] = x, z
    out["complete"] = complete
    return out, failures


def _run_chunk(job):
    i0, w_chunk = job
    if _INIT_FAILURE is not None:
        return {"runs": None, "failures": list(_INIT_FAILURE)}
    mdl = _PROB["model"]
    runs, failures = {}, []
    for name in _NAMES:
        ctrl = _PROB["ctrl"][name]
        res = []
        for j, w in enumerate(w_chunk):
            r, f = rollout_one(ctrl, mdl, _INIT[name], w, traj=i0 + j, name=name)
            res.append(r)
            failures += f
        runs[name] = {k: np.stack([np.asarray(r[k]) for r in res]) for k in RUN_KEYS}
    return {"runs": runs, "failures": failures}


def run_scenario(cfg, W, controllers=CONTROLLERS, workers=None, chunk=25, fail_dir=None):
    """W: (M, T, n) shared disturbance array.

    Returns (dict of stacked arrays per controller or None, failure records)."""
    controllers = tuple(controllers)
    workers = workers or default_workers()
    jobs = [(i, W[i:i + chunk]) for i in range(0, W.shape[0], chunk)]
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init,
                             initargs=(cfg, controllers, fail_dir)) as ex:
        parts = list(ex.map(_run_chunk, jobs))
    failures = [f for p in parts for f in p["failures"]]
    if any(p["runs"] is None for p in parts):
        return None, failures
    out = {}
    for name in controllers:
        out[name] = {k: np.concatenate([p["runs"][name][k] for p in parts]) for k in RUN_KEYS}
    return out, failures
