# Distributionally Robust Indirect-Feedback Stochastic MPC with Gelbrich Ambiguity Sets: numerical results

This repository contains the simulation code and data for the numerical results (Section V) of

> J. Sun, "Distributionally Robust Indirect-Feedback Stochastic MPC with Gelbrich Ambiguity Sets," under review.

The controller (DR-MPC) tightens the constraints of indirect-feedback stochastic MPC offline so that
CVaR constraints hold for every disturbance law whose mean and covariance lie in a Gelbrich ball
around nominal values. The example is a linearized planar quadrotor that flies into the corner
between a wall and the floor. DR-MPC is compared with IF-SMPC, a covariance-inflation baseline
(COV), a type-1 Wasserstein baseline (WAS), and a true-moment oracle (ORC) over 16 scenarios:
four distribution shapes times four levels of moment misspecification.

## Installation

Python 3.11. In the repository root:

```text
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins the four direct dependencies. `requirements.lock` lists every package of
the environment in which the results were verified (see [Reproducibility](#reproducibility));
`pip install -r requirements.lock` installs exactly that environment.

## Pilot (smoke test)

```text
python run_all.py pilot
python figures.py pilot
```

The pilot uses a separate seed, the Student-t scenario with delta = 1 only, and 200 trajectories.
It takes about 31 s on the verification machine. It writes `results/pilot/` and
`figures/pilot/` (the closed-loop figure and `table_numbers.json`). These are example outputs,
not the results of the paper; the misspecification figure needs all 16 scenarios and is skipped.

## Results of the paper

```text
python run_all.py final
python figures.py final
```

`run_all.py final` simulates 16 scenarios x 5 controllers x 2000 trajectories x 80 steps, i.e.
12,640,000 online QP solves (excluding the initial plans and the offline checks). It takes about
68 min on the verification machine with 14 worker processes (`os.cpu_count() - 2`).
`figures.py final` writes to `figures/final/`:

- `fig_closed_loop.pdf`: Fig. 1 (Student-t, delta = 1),
- `fig_misspecification.pdf`: Fig. 2 (all scenarios),
- `table_numbers.json`: the numbers of Table I under `table`; the time column is
  `step_ms_median`, the median time of a complete controller call (QP update, solve and check),
  excluding k = 0.

The results of the final run are included: `results/final/` holds `summary.json`,
`checks.json`, `meta.json`, `MANIFEST` and the 16 `*_summary.npz` files, and `figures/final/`
holds the generated figures. The 16 trajectory files `<family>_d<delta>.npz` (about 67 MB each)
are assets of the [v1.0 release](https://github.com/JinnoNian/Gelbrich-DR-IFSMPC-numerics/releases/tag/v1.0).

### Figures without re-running the simulation

Only `student_t_d1.npz` is needed in addition to the files in the repository:

```text
gh release download v1.0 -p student_t_d1.npz -D results/final
# or: curl -L -o results/final/student_t_d1.npz \
#   https://github.com/JinnoNian/Gelbrich-DR-IFSMPC-numerics/releases/download/v1.0/student_t_d1.npz
cd results/final
grep -E '  (student_t_d1(_summary)?\.npz|summary\.json|checks\.json|meta\.json)$' MANIFEST | shasum -a 256 -c -
cd ../..
python figures.py final
```

### All trajectory data

```text
gh release download v1.0 -p '*.npz' -D results/final
cd results/final
grep -v '^#' MANIFEST | shasum -a 256 -c -
```

### Code version of the results

The second line of `results/final/MANIFEST` is the commit that produced the results. Between that
commit and the tag `v1.0`, only `results/final/`, `figures/final/`, `README.md` and `LICENSE`
changed; the source code, the configuration and the dependency files did not.

## Reproducibility

The results were verified on an Apple M4 Max (macOS 26.6.2) with Python 3.11.16 and the versions
in `requirements.lock`. In this environment, the saved non-timing arrays are byte-identical and
the non-timing statistics in the JSON files are exactly equal to the results used in the paper.
Other environments have not been verified and may give numerical differences; when re-running,
check feasibility and the statistics. Timings depend on the machine.

`meta.json` records the installed packages, whether they match `requirements.lock`
(`env_matches_lock`), the number of workers, the platform and the commit.

## Code and paper

| File | Content | Paper |
|---|---|---|
| `quadrotor.py` | planar hover linearization, zero-order hold, scaling, LQR gain K, terminal cost P | Sec. V-A, V-B |
| `offline.py` | error covariance and Gelbrich radius recursions, steady-state bound, tightened bounds | Sec. III, (14), (15), (18)-(20) |
| `terminal.py` | maximal positively invariant terminal set and its LP verification | (24), Assumption 2 |
| `mpc.py` | online QP, Clarabel, check of the full plan including the state rows at offset 0 | (25), Sec. V-B |
| `baselines.py` | tightenings of COV, WAS and ORC | Sec. V-C |
| `disturbances.py` | true laws at a prescribed Gelbrich distance, the four shapes, wind direction | Sec. V-D |
| `metrics.py` | normalized losses, empirical CVaR, paired trajectory bootstrap | Sec. V-D |
| `problem.py` | assembly of model, tightenings, terminal sets and controllers; offline checks | |
| `simulate.py` | paired closed-loop rollouts in parallel workers | |
| `run_all.py` | experiment driver, integrity gate, output files | |
| `figures.py` | Fig. 1, Fig. 2, Table I | Sec. V-E, V-F |
| `check_failures.py` | controlled failures of the acceptance rule and the failure records | |
| `config.json` | all parameters and seeds | Sec. V |

All controller quantities are in the scaled coordinates `S_x^{-1} x`, `S_u^{-1} u` of Sec. V-B.

### Acceptance rule

A step is accepted only if Clarabel returns `Solved`, the solution is finite, and the plan rolled
out from the returned inputs satisfies all tightened state rows (including offset 0), input rows
and the terminal set with tolerance `tau_feas = 1e-5`. There is no fallback. Before its statistics
are computed, every scenario must pass the integrity gate: all trajectories complete, all values
finite, every online step accepted, every plan violation at most `tau_feas`.

`python check_failures.py` (about 15 s) triggers a non-`Solved` status, a non-finite solution,
a `Solved` and finite but infeasible candidate, and an exception during worker initialization,
and checks that each stops the run with the expected failure record.

## Output files

`results/<mode>/<family>_d<delta>.npz`, per controller `c` in `DR, IF, COV, WAS, ORC`
(T = 80 steps, M trajectories; scaled coordinates):

- `c_X`, `c_Z` (M, T+1, 6), float32: actual and nominal state; `c_U` (M, T, 2), float32: input;
- `c_source` (M, T), int8: 0 initial plan, 1 accepted Clarabel step, 4 rejected step;
- `c_qp_time`, `c_total_time` (M, T), float64 [s]: Clarabel solve and complete controller call;
- `c_max_violation`, `c_z0_violation` (M, T), float64: largest signed constraint value of the
  accepted plan and of the state rows at offset 0;
- `c_iterations` (M, T), int32; `c_r_prim`, `c_r_dual` (M, T), float64: Clarabel diagnostics
  (NaN at k = 0, where the initial plan is used);
- `c_complete` (M,), bool; `c_cost` (M,), float64: closed-loop cost;
- `W` (M, T, 6), float32: the disturbance samples shared by all controllers.

`<family>_d<delta>_summary.npz`: for `state` and `input`, the empirical CVaR `theta`
(controller, time, row), pointwise bounds `pw_lo`, `pw_hi`, simultaneous half-width `c`, worst
values `worst`, `worst_lo`, `worst_hi`, and violation frequencies `viol_freq`.

If a step is rejected, the run stops and writes `failures_<stage>.json` (reason, status, residuals,
constraint values) and `failures_<stage>.npz` with the float64 problem data `<i>_z`, `<i>_e`,
`<i>_q`, `<i>_u` of record `i` (time `instance_k` in the JSON). `q` and `u` are the linear cost and
the right-hand side passed to Clarabel, which together with the fixed matrices of `mpc.py` define
the rejected QP. Worker initialization failures are additionally written to
`failures_init_<pid>.json`.

## License

MIT, see `LICENSE`. Dependencies (installed with pip, not included): NumPy (BSD-3-Clause),
SciPy (BSD-3-Clause), Clarabel (Apache-2.0), Matplotlib (Matplotlib License, PSF-based).
