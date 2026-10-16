"""Controlled failures of the acceptance rule and of worker initialization.

Usage: python check_failures.py

Cases (tolerance tau_feas from config.json, unchanged):
  status      Clarabel status other than "Solved" with a finite solution
  nonfinite   status "Solved" with a NaN in the solution
  check       status "Solved", finite solution, first input 1.0 beyond its tightened bound
  worker      exception in build_all inside a worker process
For the first three, the rejected step goes through step_record and save_failures as in a run;
the written JSON and npz are read back and compared with the problem data. All output is written
to a temporary directory. Exits with an AssertionError on the first mismatch.
"""

from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np

from mpc import SolveResult
from problem import build_all, step_record
from run_all import CONFIG, ROOT, save_failures
from simulate import run_scenario


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def saved(out_dir, stage, records):
    """Run save_failures, which must stop with SystemExit; return the JSON records and the npz."""
    try:
        save_failures(out_dir, stage, records)
    except SystemExit:
        pass
    else:
        raise AssertionError(f"save_failures did not stop at stage {stage}")
    recs = json.loads((out_dir / f"failures_{stage}.json").read_text())
    return recs, np.load(out_dir / f"failures_{stage}.npz")


def step_cases(cfg, out_dir):
    prob = build_all(cfg, only=("DR",))
    c, mdl = prob["ctrl"]["DR"], prob["model"]
    k, z, e = 0, mdl.x0.copy(), np.zeros(mdl.n)
    q, u = c.qp_data(k, z, e)
    ok = c.solve_clarabel(q, u)
    require(ok.status == "Solved", f"reference solve: {ok.status}")
    plan, log = c.step(k, z, e)
    require(plan is not None and log.source == "clarabel", "reference step not accepted")

    x_nan = ok.x.copy()
    x_nan[c.nz] = np.nan
    x_bad = ok.x.copy()
    v0 = x_bad[c.nz:c.nz + c.m]
    row = 0                                  # G[0] = (1, 0): upper bound on the first input
    require(np.array_equal(mdl.G[row], [1.0, 0.0]), "unexpected input row 0")
    v0[0] = c.tg.c_t[k, row] + 1.0
    cases = (("status", "MaxIterations", ok.x, "status:MaxIterations"),
             ("nonfinite", "Solved", x_nan, "nonfinite"),
             ("check", "Solved", x_bad, "check"))
    for name, status, x, reason in cases:
        c.solve_clarabel = (lambda q_, u_, solver=None, s=status, x=x:
                            SolveResult("clarabel", s, x.copy(), 7, 1e-3, 1e-3, 0.0))
        try:
            plan, log = c.step(k, z, e)
        finally:
            del c.solve_clarabel
        require(plan is None, f"{name}: plan returned")
        require(log.source == "failed" and log.reason == reason,
                f"{name}: source {log.source!r}, reason {log.reason!r}")
        if name == "check":
            require(log.max_violation >= 1.0 - 1e-12, f"check: max_violation {log.max_violation}")
        recs, npz = saved(out_dir, name, [step_record("DR", log, traj=0, k=k)])
        require(len(recs) == 1 and recs[0]["reason"] == reason, f"{name}: JSON reason")
        require(type(recs[0]["instance_k"]) is int and recs[0]["instance_k"] == k,
                f"{name}: instance_k")
        for key, ref in (("z", z), ("e", e), ("q", q), ("u", u)):
            a = npz[f"0_{key}"]
            ref = np.asarray(ref)
            require(a.dtype == np.float64 and ref.dtype == np.float64, f"{name}: dtype of {key}")
            require(a.shape == ref.shape and a.tobytes() == ref.tobytes(), f"{name}: value of {key}")
        print(f"{name}: rejected ({reason}), record and float64 data read back")
    plan, log = c.step(k, z, e)
    require(plan is not None, "step after the checks not accepted")


def worker_case(cfg, out_dir):
    bad = copy.deepcopy(cfg)
    bad["plant"].pop("state_scale")          # KeyError in build_model inside the worker
    W = np.zeros((4, 3, 6))
    runs, failures = run_scenario(bad, W, controllers=("DR",), workers=2, chunk=2,
                                  fail_dir=out_dir)
    require(runs is None and failures, "worker: run did not fail")
    for r in failures:
        require(r.get("stage") == "worker_init" and r.get("reason") == "worker init exception",
                f"worker: record {r}")
        require(isinstance(r.get("error"), str) and r["error"], "worker: empty error")
        require(isinstance(r.get("pid"), int), "worker: pid")
        f = out_dir / f"failures_init_{r['pid']}.json"
        require(f.exists(), f"worker: {f.name} missing")
        require(json.loads(f.read_text())[0]["error"] == r["error"], f"worker: {f.name} content")
    recs, npz = saved(out_dir, "worker", failures)
    require(len(recs) == len(failures) and set(npz.files) == {"records"}, "worker: saved records")
    print(f"worker: {len(failures)} init failure record(s), {failures[0]['error']}")


def main():
    cfg = json.loads((ROOT / CONFIG).read_text())
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        step_cases(cfg, out_dir)
        worker_case(cfg, out_dir)
    print("all failure checks passed")


if __name__ == "__main__":
    main()
