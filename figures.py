"""Publication figures and table numbers (physical units where axes are in meters).

Usage:
    python figures.py final [out_dir]   # Fig. 1, Fig. 2 and the numbers of Table I
    python figures.py pilot [out_dir]   # pilot example: closed-loop figure and numbers only

Reads results/<mode>/ (summary.json, checks.json, meta.json, student_t_d1.npz and
student_t_d1_summary.npz). The run configuration is taken from meta.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from problem import build_all

ROOT = Path(__file__).resolve().parent
COL = {"DR": "#D55E00", "IF": "#0072B2", "COV": "#009E73", "WAS": "#CC79A7", "ORC": "k"}
LAB = {"DR": "DR-MPC", "IF": "IF-SMPC", "COV": "COV", "WAS": "WAS", "ORC": "ORC"}
MODES = ("pilot", "final")
REP = "student_t_d1"
FAM = {"gaussian": "Gaussian", "student_t": "Student-$t$", "laplace": "Laplace",
       "mixture": "Bimodal mixture"}
COLW, TEXTW = 3.5, 7.16

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIX Two Text", "STIXGeneral", "Times New Roman", "Times"],
    "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "lines.linewidth": 1.0, "pdf.fonttype": 42, "savefig.dpi": 400,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
})


def load(mode):
    d = ROOT / "results" / mode
    cfg = json.loads((d / "meta.json").read_text())["config"]
    if mode == "final" and cfg != json.loads((ROOT / "config.json").read_text()):
        raise SystemExit("results/final/meta.json was not produced with config.json")
    return d, cfg, json.loads((d / "summary.json").read_text())


def fig_closed_loop(mode, out):
    d, cfg, _ = load(mode)
    prob = build_all(cfg, only=("DR", "IF"))
    mdl = prob["model"]
    Ts = cfg["plant"]["Ts"]
    sx = np.diag(mdl.Sx)
    x_wall = cfg["plant"]["state_hi_phys"][0]
    z_floor = cfg["plant"]["state_lo_phys"][1]
    raw = np.load(d / f"{REP}.npz")
    summ = np.load(d / f"{REP}_summary.npz")
    n_show = 100
    T = cfg["sim"]["T"]
    t = np.arange(T + 1) * Ts

    fig = plt.figure(figsize=(TEXTW, 1.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34,
                          left=0.06, right=0.995, bottom=0.25, top=0.80)
    ax = fig.add_subplot(gs[0])
    # (a) flight paths; horizontal position relative to the wall, height above the floor.
    for c in ("IF", "DR"):
        X = raw[f"{c}_X"][:n_show].astype(float)
        px = X[:, :, 0] * sx[0] - x_wall
        pz = X[:, :, 1] * sx[1] - z_floor
        for i in range(n_show):
            ax.plot(px[i], pz[i], color=COL[c], lw=0.35, alpha=0.18, rasterized=True)
        ax.plot(np.median(px, 0), np.median(pz, 0), color=COL[c], lw=1.3, label=LAB[c])
        tg = prob["tight"][c]
        bw = tg.b_t[T, 0] * sx[0] - x_wall
        bf = -tg.b_t[T, 3] * sx[1] - z_floor
        ax.plot([bw, bw], [bf, 1.9], color=COL[c], ls="--", lw=0.8)
        ax.plot([-3.7, bw], [bf, bf], color=COL[c], ls="--", lw=0.8)
    ax.add_patch(Rectangle((0, -0.3), 0.3, 2.3, color="0.85", lw=0, zorder=0))
    ax.add_patch(Rectangle((-3.8, -0.3), 4.1, 0.3, color="0.85", lw=0, zorder=0))
    ax.axvline(0, color="k", lw=0.8)
    ax.axhline(0, color="k", lw=0.8)
    ax.plot(raw["DR_X"][0, 0, 0] * sx[0] - x_wall, raw["DR_X"][0, 0, 1] * sx[1] - z_floor,
            "ko", ms=3)
    ax.text(-3.45, 1.72, "start", fontsize=7)
    ax.text(0.04, 1.2, "wall", rotation=90, fontsize=7)
    ax.text(-3.6, -0.2, "floor", fontsize=7)
    ax.set_xlim(-3.7, 0.3)
    ax.set_ylim(-0.3, 1.95)
    ax.set_xlabel("horizontal position to wall [m]")
    ax.set_ylabel("height above floor [m]")
    ax.set_title("(a)", loc="left")

    # inset: steady-state positions near the corner, violations marked
    ins = ax.inset_axes([0.40, 0.43, 0.40, 0.50])
    kss = slice(40, T + 1)
    for c in ("IF", "DR"):
        X = raw[f"{c}_X"][:n_show, kss].astype(float)
        px = (X[:, :, 0] * sx[0] - x_wall).ravel()
        pz = (X[:, :, 1] * sx[1] - z_floor).ravel()
        bad = (px > 0) | (pz < 0)
        ins.scatter(px[~bad], pz[~bad], s=0.6, color=COL[c], alpha=0.25, lw=0, rasterized=True)
        ins.scatter(px[bad], pz[bad], s=3, color="k", marker="x", lw=0.4, rasterized=True)
    ins.add_patch(Rectangle((0, -0.2), 0.2, 1.2, color="0.85", lw=0, zorder=0))
    ins.add_patch(Rectangle((-1.0, -0.2), 1.2, 0.2, color="0.85", lw=0, zorder=0))
    ins.axvline(0, color="k", lw=0.6)
    ins.axhline(0, color="k", lw=0.6)
    ins.set_xlim(-0.75, 0.12)
    ins.set_ylim(-0.08, 0.75)
    ins.tick_params(labelsize=5.5, length=1.5, pad=1)
    for s in ("top", "right"):
        ins.spines[s].set_visible(True)
    ins.set_title(r"$t\geq 4$ s, $\times$: violation", fontsize=5.5, pad=1.5)

    # (b) distance to the wall: actual (solid), nominal (dotted), margin b - tilde b(k) (dashed),
    # matching the line styles of (a)
    ax = fig.add_subplot(gs[1])
    for c in ("IF", "DR"):
        Z = raw[f"{c}_Z"].astype(float)
        X = raw[f"{c}_X"].astype(float)
        dn = (mdl.b[0] - Z[:, :, 0]) * sx[0]
        da = (mdl.b[0] - X[:, :, 0]) * sx[0]
        ax.plot(t, np.median(dn, axis=0), color=COL[c], lw=1.0, ls=":")
        ax.plot(t, np.quantile(da, 0.05, axis=0), color=COL[c], lw=1.1)
        marg = (mdl.b[0] - prob["tight"][c].b_t[:T + 1, 0]) * sx[0]
        ax.plot(t, marg, color=COL[c], ls="--", lw=0.8)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, T * Ts)
    ax.set_ylim(-0.05, 1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance to wall [m]")
    ax.set_title("(b)", loc="left")

    # (c) per-time empirical CVaR of the wall row
    ax = fig.add_subplot(gs[2])
    tt = np.arange(1, T + 1) * Ts
    th, lo, hi = summ["state_theta"], summ["state_pw_lo"], summ["state_pw_hi"]
    for ci, c in enumerate(("DR", "IF")):
        ax.fill_between(tt, lo[ci, :, 0], hi[ci, :, 0], color=COL[c], alpha=0.25, lw=0)
        ax.plot(tt, th[ci, :, 0], color=COL[c], lw=1.0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlim(0, T * Ts)
    ax.set_ylim(-0.6, 0.1)
    ax.set_xlabel("time [s]")
    ax.set_ylabel(r"CVaR$_{0.2}$ of wall loss [m]")
    ax.set_title("(c)", loc="left")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color=COL["DR"], lw=1.3, label=LAB["DR"]),
               Line2D([], [], color=COL["IF"], lw=1.3, label=LAB["IF"]),
               Line2D([], [], color="0.3", lw=1.1, label="actual state"),
               Line2D([], [], color="0.3", ls=":", lw=1.0, label="nominal state"),
               Line2D([], [], color="0.3", ls="--", lw=0.8, label="tightened bound"),
               Patch(color="0.6", alpha=0.4, lw=0, label="(c) 95% interval")]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.53, 1.0),
               handlelength=1.8, columnspacing=1.1)
    fig.savefig(out / "fig_closed_loop.pdf")
    plt.close(fig)


def fig_misspecification(mode, out):
    """2 x 2 panels (one per distribution shape); needs all four shapes and every delta."""
    d, cfg, S = load(mode)
    deltas = cfg["sim"]["delta_grid"]
    fams = cfg["sim"]["families"]
    missing = [f"{f}_d{dl:g}" for f in fams for dl in deltas if f"{f}_d{dl:g}" not in S]
    if len(fams) != 4 or missing:
        print(f"fig_misspecification skipped: needs 4 shapes x {len(deltas)} deltas "
              f"(run has {len(fams)} shape(s), missing {missing})")
        return
    ctrls = controllers(S)
    fig, axes = plt.subplots(2, 2, figsize=(COLW, 1.7), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.20, top=0.86,
                        wspace=0.08, hspace=0.12)
    for ax, fam in zip(axes.ravel(), fams):
        for ci, c in enumerate(ctrls):
            w = [S[f"{fam}_d{dl:g}"]["state"]["worst"][ci] for dl in deltas]
            lo = [S[f"{fam}_d{dl:g}"]["state"]["worst_lo"][ci] for dl in deltas]
            hi = [S[f"{fam}_d{dl:g}"]["state"]["worst_hi"][ci] for dl in deltas]
            orc = c == "ORC"
            ax.fill_between(deltas, lo, hi, color="0.5" if orc else COL[c],
                            alpha=0.15, lw=0)
            ax.plot(deltas, w, color=COL[c], marker=None if orc else "o", ms=2.2,
                    lw=0.7 if orc else 1.0, ls="--" if orc else "-", label=LAB[c],
                    zorder=4 if c in ("DR", "IF") else 3)
        ax.axhline(0, color="k", lw=0.6)
        ax.axvline(1.0, color="0.5", lw=0.6, ls=":")
        ax.text(0.03, 0.385, FAM[fam], transform=ax.transAxes, va="center", ha="left")
        ax.set_xticks(deltas)
    for ax in axes[1]:
        ax.set_xlabel(r"misspecification $\delta$")
    fig.supylabel("worst state CVaR$_{0.2}$", x=0.01, fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=len(ctrls), bbox_to_anchor=(0.57, 1.0),
               handlelength=1.6, columnspacing=1.0)
    fig.savefig(out / "fig_misspecification.pdf")
    plt.close(fig)


def controllers(S):
    return next(iter(S.values()))["controllers"]


def table_numbers(mode):
    d, cfg, S = load(mode)
    checks = json.loads((d / "checks.json").read_text())
    ctrls = controllers(S)
    R = S[REP]
    out = {"representative_scenario": REP, "controllers": ctrls, "table": {}, "pooled": {}}
    for ci, c in enumerate(ctrls):
        out["table"][c] = {
            "worst_state": (R["state"]["worst"][ci], R["state"]["worst_lo"][ci],
                            R["state"]["worst_hi"][ci]),
            "worst_input": (R["input"]["worst"][ci], R["input"]["worst_lo"][ci],
                            R["input"]["worst_hi"][ci]),
            "max_viol_state": R["state"]["max_viol_freq"][ci],
            "max_viol_input": R["input"]["max_viol_freq"][ci],
            "cost_rel_vs_IF": R["cost_rel_vs_IF"][c],
            "step_ms_median": R["solver"][c]["step_ms_median"],
            "qp_ms_median": R["solver"][c]["qp_ms_median"],
        }
        sol = [S[k]["solver"][c] for k in S]
        src = [S[k]["controller_source"][c] for k in S]
        out["pooled"][c] = {
            "worst_state_inball": max(S[k]["state"]["worst"][ci] for k in S
                                      if float(k.split("_d")[1]) <= 1.0),
            "worst_state_by_scenario": {k: (S[k]["state"]["worst"][ci], S[k]["state"]["worst_lo"][ci],
                                            S[k]["state"]["worst_hi"][ci]) for k in S},
            "cost_rel_vs_IF_range": (min(S[k]["cost_rel_vs_IF"][c] for k in S),
                                     max(S[k]["cost_rel_vs_IF"][c] for k in S)),
            "u0_loss_max": max(S[k]["u0_loss_max"][ci] for k in S),
            "step_ms_median": float(np.median([s["step_ms_median"] for s in sol])),
            "step_ms_p95": float(max(s["step_ms_p95"] for s in sol)),
            "qp_ms_median": float(np.median([s["qp_ms_median"] for s in sol])),
            "qp_constraints": sorted({checks["offline"][n]["qp_constraints"] for n in src}),
            "qp_variables": sorted({checks["offline"][n]["qp_variables"] for n in src}),
            "clarabel": sum(s["counts"]["clarabel"] for s in sol),
            "failed": sum(s["counts"]["failed"] for s in sol),
            "incomplete": sum(S[k]["incomplete"][c] for k in S),
            "max_plan_violation": max(s["max_plan_violation"] for s in sol),
            "total_steps": sum(sum(s["counts"].values()) for s in sol),
        }
    out["max_e_diff"] = max(S[k]["max_e_diff"] for k in S)
    out["c_range"] = (min(S[k]["state"]["c"] for k in S), max(S[k]["state"]["c"] for k in S))
    return out


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    if mode not in MODES:
        raise SystemExit(f"unknown mode {mode!r}; use one of {MODES}")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "figures" / mode
    out.mkdir(parents=True, exist_ok=True)
    fig_closed_loop(mode, out)
    fig_misspecification(mode, out)
    nums = table_numbers(mode)
    (out / "table_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    print(json.dumps(nums, indent=1, default=float))
