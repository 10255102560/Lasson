"""Run the complete study; all reported observations are written to CSV/NPZ.

Usage: python run_experiments.py [--output results] [--config experiment_config.json]
No downloaded data, parallel workers, or test-set tuning are involved.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_info, threadpool_limits

from data import make_problem
from lasso import certificate, lipschitz_constant, solve


def reference_fit(d, lam):
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = Lasso(alpha=lam, fit_intercept=False, tol=1e-12,
                      max_iter=200000, selection="cyclic").fit(d.A, d.b)
    x = model.coef_
    r = d.A @ x - d.b
    c = certificate(d.b, x, r, d.A.T @ r / len(d.b), lam, max(1, d.lambda_max))
    if max(c["relative_gap"], c["relative_kkt"]) > 1e-8:
        raise ArithmeticError("Reference solution lacks required accuracy")
    return x, c


def save_csv(out, name, rows):
    pd.DataFrame(rows).to_csv(out / name, index=False, float_format="%.16g")


def run(config_path, out):
    cfg = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    (out / "traces").mkdir(exist_ok=True)
    common = {k: cfg[k] for k in ("p", "sparsity", "n_val", "n_test")}
    options = {k: cfg[k] for k in ("tol", "max_iter")}
    started = perf_counter()
    source_files = ["lasso.py", "data.py", "run_experiments.py", str(config_path)]
    metadata = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    config=cfg, python=sys.version, platform=platform.platform(),
                    machine=platform.machine(), numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__,
                    pandas=pd.__version__, threadpools=threadpool_info(),
                    source_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                   for p in source_files})
    try:
        metadata["cpu"] = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        metadata["cpu"] = platform.processor()
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    benchmark, timing_runs, setup, ref_rows = [], [], [], []

    for rho in cfg["benchmark_rhos"]:
        for seed in cfg["seeds"]:
            d = make_problem(seed, n=300, rho=rho, noise_ratio=.1, **common)
            tic = perf_counter()
            L = lipschitz_constant(d.A)
            L_seconds = perf_counter() - tic
            lam = cfg["benchmark_lambda_ratio"] * d.lambda_max
            xr, cr = reference_fit(d, lam)
            identity = dict(seed=seed, rho=rho, lambda_value=lam,
                            lambda_max=d.lambda_max, L=L)
            setup.append(dict(**identity, L_seconds=L_seconds))
            ref_rows.append(dict(experiment="benchmark", scenario=f"rho{rho}",
                                 **identity, **cr))
            # Warm up BLAS and the Python loop, outside timed repetitions.
            solve(d.A, d.b, lam, L=L, max_iter=8)
            grouped = {m: [] for m in ("ISTA", "FISTA")}
            for rep in range(cfg["timing_repeats"]):
                order = ("ISTA", "FISTA") if rep % 2 == 0 else ("FISTA", "ISTA")
                for method in order:
                    s = solve(d.A, d.b, lam, method, L=L, **options)
                    if not s.converged:
                        raise RuntimeError(f"Unconverged benchmark: {identity}, {method}")
                    grouped[method].append(s)
                    timing_runs.append(dict(**identity, method=method, repeat=rep,
                                            seconds=s.seconds, iterations=s.iterations))
            for method, runs in grouped.items():
                s = runs[-1]
                row = dict(**identity, method=method, iterations=s.iterations,
                           seconds=float(np.median([v.seconds for v in runs])),
                           min_seconds=min(v.seconds for v in runs),
                           max_seconds=max(v.seconds for v in runs),
                           converged=s.converged, **s.metrics, **d.evaluate(s.x),
                           reference_objective=cr["objective"],
                           objective_difference=s.metrics["objective"]-cr["objective"],
                           prediction_difference=float(np.linalg.norm(d.A @ (s.x-xr))
                                                       / max(1, np.linalg.norm(d.A @ xr))))
                benchmark.append(row)
                if seed == cfg["seeds"][0]:
                    traced = solve(d.A, d.b, lam, method, L=L, trace=True, **options)
                    save_csv(out / "traces", f"{method}_rho{rho:.1f}.csv", traced.history)
            print(f"Benchmark rho={rho:g}, seed={seed}: "
                  f"ISTA {grouped['ISTA'][-1].iterations}, "
                  f"FISTA {grouped['FISTA'][-1].iterations} iterations", flush=True)
    save_csv(out, "benchmark.csv", benchmark)
    save_csv(out, "timing_runs.csv", timing_runs)
    save_csv(out, "spectral_setup.csv", setup)

    sweep, selected, baselines, step_rows = [], [], [], []
    for scenario in cfg["scenarios"]:
        params = {k: v for k, v in scenario.items() if k != "name"}
        for seed in cfg["seeds"]:
            d = make_problem(seed, **params, **common)
            L = lipschitz_constant(d.A)
            rows, xs = [], []
            identity = dict(scenario=scenario["name"], seed=seed, **params,
                            lambda_max=d.lambda_max, L=L)
            for ratio in cfg["lambda_ratios"]:
                s = solve(d.A, d.b, ratio*d.lambda_max, "FISTA", L=L, **options)
                row = dict(**identity, lambda_ratio=ratio, lambda_value=ratio*d.lambda_max,
                           method="FISTA", iterations=s.iterations, seconds=s.seconds,
                           converged=s.converged, **s.metrics, **d.evaluate(s.x))
                rows.append(row)
                sweep.append(row)
                xs.append(s.x)
                if not s.converged:
                    save_csv(out, "lambda_sweep.csv", sweep)
                    raise RuntimeError(f"Unconverged path: {identity}, ratio={ratio}")
            # Selection uses validation MSE ONLY; test columns are descriptive.
            chosen = int(np.argmin([r["validation_mse"] for r in rows]))
            best, best_x = rows[chosen], xs[chosen]
            lam = best["lambda_value"]
            si = solve(d.A, d.b, lam, "ISTA", L=L, **options)
            if not si.converged:
                raise RuntimeError(f"Unconverged selected ISTA: {identity}")
            xr, cr = reference_fit(d, lam)
            ref_rows.append(dict(experiment="selection", **identity,
                                 lambda_value=lam, **cr))
            selected.append(dict(**best, ista_iterations=si.iterations,
                                 ista_seconds=si.seconds,
                                 ista_relative_gap=si.metrics["relative_gap"],
                                 ista_relative_kkt=si.metrics["relative_kkt"],
                                 ista_objective_difference=si.metrics["objective"]-cr["objective"],
                                 reference_objective=cr["objective"],
                                 objective_difference=best["objective"]-cr["objective"]))
            if scenario["name"] == "base":
                # Minimum-norm LS is a model baseline, not a Lasso optimizer.
                xls = np.linalg.lstsq(d.A, d.b, rcond=None)[0]
                baselines.append(dict(seed=seed, model="Lasso (validation)", **d.evaluate(best_x)))
                baselines.append(dict(seed=seed, model="Minimum-norm LS", **d.evaluate(xls)))
                baselines.append(dict(seed=seed, model="Intercept only", **d.evaluate(np.zeros(cfg["p"]))))
                for factor in [.25, .5, 1.0]:
                    ss = solve(d.A, d.b, .05*d.lambda_max, "ISTA", L=L,
                               step_factor=factor, trace=(seed == cfg["seeds"][0]), **options)
                    if not ss.converged:
                        raise RuntimeError("Step-size experiment did not converge")
                    step_rows.append(dict(seed=seed, step_factor=factor,
                                          iterations=ss.iterations, seconds=ss.seconds, **ss.metrics))
                    if seed == cfg["seeds"][0]:
                        save_csv(out / "traces", f"step_{factor:.2f}.csv", ss.history)
                if seed == cfg["seeds"][0]:
                    np.savez_compressed(out / "base_problem.npz", **asdict(d))
                    np.savez_compressed(out / "base_solutions.npz", beta_true=d.beta_true,
                                        x_path=np.array(xs), x_selected=best_x,
                                        x_selected_ista=si.x, x_reference=xr, x_ls=xls,
                                        feature_scale=d.feature_scale,
                                        ratios=np.array(cfg["lambda_ratios"]), selected_index=chosen)
            print(f"Scenario {scenario['name']}, seed={seed}: "
                  f"selected ratio={best['lambda_ratio']:g}, "
                  f"test MSE={best['test_mse']:.4f}, nnz={best['nnz']}, "
                  f"path iterations={sum(r['iterations'] for r in rows)}", flush=True)
        save_csv(out, "lambda_sweep.csv", sweep)
        save_csv(out, "selected_models.csv", selected)
        save_csv(out, "baselines.csv", baselines)
        save_csv(out, "step_sizes.csv", step_rows)
        save_csv(out, "reference_checks.csv", ref_rows)
    metadata["total_seconds"] = perf_counter() - started
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    metadata["counts"] = dict(benchmark_problems=len(benchmark)//2,
                              timed_solver_runs=len(timing_runs),
                              lambda_fits=len(sweep), selected_models=len(selected),
                              independent_reference_checks=len(ref_rows))
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Completed in {metadata['total_seconds']:.1f}s; results: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("experiment_config.json"))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.output)
