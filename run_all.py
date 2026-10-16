"""Run the pilot or the final experiment.

Usage:
    python run_all.py pilot   # pilot seed, student_t delta=1 only, M=200 (smoke test)
    python run_all.py final   # final seed, all 16 scenarios, M=2000 (results of the paper)

Output in results/<mode>/: one <family>_d<delta>.npz (trajectories) and <...>_summary.npz
(CVaR estimates and bootstrap bounds) per scenario, summary.json, checks.json, meta.json, MANIFEST.
Every scenario must pass the integrity gate before its statistics are computed; otherwise the
failure records are written to results/<mode>/failures_<stage>.json/.npz and the run stops.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from disturbances import disturbance_array, gelbrich, law_parameters, sample_xi, wind_direction
from metrics import closed_loop_cost, scenario_summary
from problem import (build_all, initial_failures, initial_plans, label, scenario_controllers,
                     sensitivity_check)
from simulate import SOURCE_CODE, default_workers, rollout_one, run_scenario

ROOT = Path(__file__).resolve().parent
CONFIG = "config.json"
SEED_KEY = {"pilot": "pilot", "final": "final"}
SIM_OVERRIDE = {"pilot": {"M": 200, "families": ["student_t"], "delta_grid": [1.0]}}
FLOAT32_KEYS = ("X", "Z", "U")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return obj.item()
    return obj


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed_packages():
    """Installed distributions as sorted 'name==version' lines (names normalized)."""
    pkgs = {_norm(d.metadata["Name"]): d.version for d in importlib.metadata.distributions()}
    return sorted(f"{n}=={v}" for n, v in pkgs.items() if n not in ("pip", "setuptools", "wheel"))


def lock_packages():
    lines = (ROOT / "requirements.lock").read_text().splitlines()
    out = []
    for ln in lines:
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        n, v = ln.split("==")
        if _norm(n) not in ("pip", "setuptools", "wheel"):
            out.append(f"{_norm(n)}=={v.strip()}")
    return sorted(out)


def moment_check(cfg, d, n=6, size=400_000, seed=1):
    """Large-sample moments of xi for each family (check only, no correction)."""
    rng = np.random.default_rng(seed)
    amb = cfg["ambiguity"]
    out = {}
    for fam in cfg["sim"]["families"]:
        xi = sample_xi(rng, fam, (size,), n, nu=amb["student_nu"], alpha=amb["mix_alpha"], d_mix=d)
        out[fam] = {"max_abs_mean": float(np.abs(xi.mean(0)).max()),
                    "max_abs_cov_minus_I": float(np.abs(np.cov(xi.T) - np.eye(n)).max())}
    return out


def zero_radius_check(cfg, W):
    """With rho_w = 0 both controllers coincide: compare tightenings and rollouts."""
    c0 = copy.deepcopy(cfg)
    c0["ambiguity"]["rho_w"] = 0.0
    p = build_all(c0, only=("DR", "IF"))
    plans, logs = initial_plans(p)
    failures = initial_failures(p, plans, logs)
    same_tight = bool(np.allclose(p["tight"]["DR"].b_t, p["tight"]["IF"].b_t)
                      and np.allclose(p["tight"]["DR"].c_t, p["tight"]["IF"].c_t))
    if failures:
        return {"same_tightening": same_tight, "max_state_diff": None}, failures
    diffs = []
    for j, w in enumerate(W[:3]):
        a, fa = rollout_one(p["ctrl"]["DR"], p["model"], plans["DR"], w, traj=j, name="DR")
        b, fb = rollout_one(p["ctrl"]["IF"], p["model"], plans["IF"], w, traj=j, name="IF")
        failures += fa + fb
        diffs.append(float(np.nanmax(np.abs(a["X"] - b["X"]))))
    return {"same_tightening": same_tight, "max_state_diff": max(diffs)}, failures


def save_failures(out_dir, stage, records):
    """Failure records: JSON for reading, npz with the float64 problem data. Then stop."""
    records = copy.deepcopy(records)
    arrays = {}
    for i, r in enumerate(records):
        inst = r.pop("instance", None) or {}
        for k, v in inst.items():
            if isinstance(v, list):
                arrays[f"{i}_{k}"] = np.asarray(v, dtype=np.float64)
            else:
                r[f"instance_{k}"] = v
    meta = json.dumps(to_jsonable(records), indent=1, default=float)
    (out_dir / f"failures_{stage}.json").write_text(meta)
    np.savez_compressed(out_dir / f"failures_{stage}.npz", records=np.array(meta), **arrays)
    raise SystemExit(f"{len(records)} failure record(s) at stage '{stage}'; "
                     f"see {out_dir / f'failures_{stage}.json'}")


def integrity_gate(runs, ctrls, tau):
    """Problems that must not occur (empty list if the scenario passes)."""
    bad = []
    for c in ctrls:
        r = runs[c]
        if not r["complete"].all():
            bad.append(f"{c}: {int((~r['complete'].astype(bool)).sum())} incomplete trajectories")
        for k in ("X", "Z", "U"):
            if not np.isfinite(r[k]).all():
                bad.append(f"{c}: non-finite {k}")
        src = r["source"]
        if not (src[:, 0] == SOURCE_CODE["init"]).all() or not (src[:, 1:] == SOURCE_CODE["clarabel"]).all():
            bad.append(f"{c}: steps not accepted from Clarabel")
        mv = r["max_violation"]
        if not (np.isfinite(mv).all() and (mv <= tau).all()):
            bad.append(f"{c}: plan violation above tau_feas or non-finite")
    return bad


def git_state():
    """(commit, dirty). Dirty: any modified or untracked file outside results/ and figures/."""
    rev = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    if rev.returncode != 0:
        return "unknown", True
    st = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all", "--", ".",
                         ":(exclude)results", ":(exclude)figures"],
                        cwd=ROOT, capture_output=True, text=True)
    return rev.stdout.strip(), bool(st.stdout.strip()) or st.returncode != 0


def write_manifest(out_dir, mode, commit, dirty):
    lines = [f"# Cache for results/{mode}", f"# git commit {commit}" + (" (dirty)" if dirty else ""),
             f"# config {CONFIG}",
             "# sha256 of every cached file; trajectory npz files are distributed as release assets"]
    for p in sorted(out_dir.iterdir()):
        if p.is_file() and p.name != "MANIFEST":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}")
    (out_dir / "MANIFEST").write_text("\n".join(lines) + "\n")


def main(mode):
    if mode not in SEED_KEY:
        raise SystemExit(f"unknown mode {mode!r}; use one of {sorted(SEED_KEY)}")
    commit, dirty = git_state()
    cfg = json.loads((ROOT / CONFIG).read_text())
    out_dir = ROOT / "results" / mode
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    if mode in SIM_OVERRIDE:
        cfg = copy.deepcopy(cfg)
        cfg["sim"].update(SIM_OVERRIDE[mode])

    prob = build_all(cfg)
    mdl = prob["model"]
    plans, logs = initial_plans(prob)
    init_fail = initial_failures(prob, plans, logs)
    if init_fail:
        save_failures(out_dir, "initial", init_fail)
    sens, sens_fail = sensitivity_check(prob, plans)
    if sens_fail is not None:
        save_failures(out_dir, "sensitivity", [sens_fail])
    amb, sim, st = cfg["ambiguity"], cfg["sim"], cfg["stats"]
    d = wind_direction(mdl, amb["wind_channels"])
    checks = {
        "offline": prob["checks"],
        "initial_plan_source": {k: v.source for k, v in logs.items()},
        "initial_feasible": {k: v is not None for k, v in plans.items()},
        "initial_plan_violation": {k: v.max_violation for k, v in plans.items()},
        "v0_sensitivity": sens,
        "wind_direction": d,
        "moments": moment_check(cfg, d),
    }
    built = set(prob["ctrl"])
    needed = {c for dl in cfg["sim"]["delta_grid"] for c in scenario_controllers(cfg, dl)}
    if not needed <= built:
        raise SystemExit(f"controllers without verified terminal set: {needed - built}")

    seed_base = cfg["seeds"][SEED_KEY[mode]]
    ss = np.random.SeedSequence(seed_base)
    scen = [(f, dl) for f in sim["families"] for dl in sim["delta_grid"]]
    child = ss.spawn(len(scen))
    summaries = {}
    timing = {}
    for i, (fam, delta) in enumerate(scen):
        tag = f"{fam}_d{delta:g}"
        r = delta * amb["rho_w"]                    # fixed reference radius
        mu, Sigma = law_parameters(mdl.Sigma_w, d, r, amb["eta"])
        G = gelbrich(mu, Sigma, np.zeros(mdl.n), mdl.Sigma_w)
        rng = np.random.default_rng(child[i])
        W = disturbance_array(rng, fam, sim["M"], sim["T"], mu, Sigma,
                              amb["student_nu"], amb["mix_alpha"], d)
        if i == 0:
            checks["zero_radius"], zr_fail = zero_radius_check(cfg, W)
            if zr_fail:
                save_failures(out_dir, "zero_radius", zr_fail)
        names = scenario_controllers(cfg, delta)
        t0 = time.time()
        runs, failures = run_scenario(cfg, W, controllers=names, fail_dir=out_dir)
        t_sim = time.time() - t0
        if runs is None:
            save_failures(out_dir, tag, failures)
        runs = {label(c): r for c, r in runs.items()}
        ctrls = tuple(label(c) for c in names)
        bad = integrity_gate(runs, ctrls, cfg["solver"]["tau_feas"])
        if failures or bad:
            save_failures(out_dir, tag, failures + [{"gate": b} for b in bad])
        e_ref = runs["DR"]["X"] - runs["DR"]["Z"]
        e_diff = max(float(np.nanmax(np.abs((runs[c]["X"] - runs[c]["Z"]) - e_ref)))
                     for c in ctrls[1:])
        t0 = time.time()
        summ = scenario_summary(mdl, runs, mdl.eps, st["B"], st["level"], seed_base + 1000 + i,
                                controllers=ctrls)
        t_boot = time.time() - t0
        costs = {c: closed_loop_cost(mdl, prob["ref"], runs[c]["X"], runs[c]["U"]) for c in ctrls}
        summ["controllers"] = list(ctrls)
        summ["controller_source"] = dict(zip(ctrls, names))
        summ["cost_mean"] = {c: float(costs[c].mean()) for c in ctrls}
        summ["cost_rel_mean"] = float((costs["DR"] / costs["IF"] - 1.0).mean())
        summ["cost_rel_vs_IF"] = {c: float((costs[c] / costs["IF"] - 1.0).mean()) for c in ctrls}
        summ["gelbrich_distance"] = G
        summ["max_e_diff"] = e_diff
        for c in ctrls:
            src = runs[c]["source"]
            k1 = src[:, 1:]
            summ.setdefault("solver", {})[c] = {
                "counts": {name: int((k1 == SOURCE_CODE[name]).sum()) for name in ("clarabel", "failed")},
                "online_steps": int(k1.size),
                "qp_ms_median": float(np.median(runs[c]["qp_time"][:, 1:]) * 1e3),
                "qp_ms_p95": float(np.percentile(runs[c]["qp_time"][:, 1:], 95) * 1e3),
                "step_ms_median": float(np.median(runs[c]["total_time"][:, 1:]) * 1e3),
                "step_ms_p95": float(np.percentile(runs[c]["total_time"][:, 1:], 95) * 1e3),
                "max_plan_violation": float(np.nanmax(runs[c]["max_violation"])),
                "max_z0_violation": float(np.nanmax(runs[c]["z0_violation"])),
                "iterations_median": float(np.median(runs[c]["iterations"][:, 1:])),
                "iterations_max": int(np.max(runs[c]["iterations"][:, 1:])),
            }
        np.savez_compressed(
            out_dir / f"{tag}.npz",
            **{f"{c}_{k}": (v.astype(np.float32) if k in FLOAT32_KEYS else v)
               for c in ctrls for k, v in runs[c].items()},
            **{f"{c}_cost": costs[c] for c in ctrls},
            W=W.astype(np.float32))
        np.savez_compressed(out_dir / f"{tag}_summary.npz",
                            **{f"{t}_{k}": np.asarray(v) for t in ("state", "input")
                               for k, v in summ[t].items()})
        summaries[tag] = {k: v for k, v in summ.items() if k not in ("state", "input")}
        for t in ("state", "input"):
            s = summ[t]
            summaries[tag][t] = {"worst": s["worst"], "worst_lo": s["worst_lo"],
                                 "worst_hi": s["worst_hi"], "c": s["c"],
                                 "max_viol_freq": s["viol_freq"].reshape(len(ctrls), -1).max(axis=1)}
        timing[tag] = {"sim_s": t_sim, "bootstrap_s": t_boot}
        worst = " ".join(f"{c}={w:+.4f}" for c, w in zip(ctrls, summ["state"]["worst"]))
        print(f"{tag}: G={G:.4g} worst state {worst} cost_rel_DR={summ['cost_rel_mean']:+.3f} "
              f"e_diff={e_diff:.1e} sim={t_sim:.0f}s boot={t_boot:.0f}s", flush=True)

    packages = installed_packages()
    meta = {"mode": mode, "config_file": CONFIG, "config": cfg,
            "git_commit": commit, "git_dirty": dirty,
            "python": sys.version, "platform": platform.platform(),
            "cpu_count": os.cpu_count(), "workers": default_workers(),
            "packages": packages, "env_matches_lock": packages == lock_packages(),
            "command": " ".join([Path(sys.executable).name] + sys.argv),
            "total_s": time.time() - t_start, "timing": timing,
            "protocol": ("same seeds and parameters as the paper; every scenario passed the "
                         "integrity gate before statistics were computed")}
    (out_dir / "checks.json").write_text(json.dumps(to_jsonable(checks), indent=1))
    (out_dir / "summary.json").write_text(json.dumps(to_jsonable(summaries), indent=1))
    (out_dir / "meta.json").write_text(json.dumps(to_jsonable(meta), indent=1))
    write_manifest(out_dir, mode, commit, dirty)
    print("total", time.time() - t_start)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "pilot")
