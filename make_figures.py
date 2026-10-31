"""Create publication figures and aggregate tables from saved observations."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN = "#235789", "#D76A35", "#29836C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.labelcolor": "#263238", "axes.titleweight": "semibold",
                     "axes.grid": True, "grid.alpha": .18,
                     "figure.dpi": 160, "savefig.dpi": 240,
                     "pdf.fonttype": 42, "ps.fonttype": 42})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    plt.close(fig)


def band(ax, df, x, y, label, color, marker="o", log=False):
    a = df.groupby(x)[y].agg(["mean", "std"])
    ax.plot(a.index, a["mean"], marker=marker, color=color, label=label, lw=1.7, ms=4)
    lower = a["mean"] - a["std"]
    if log:
        lower = np.maximum(lower, 1e-12)
    ax.fill_between(a.index, lower, a["mean"] + a["std"], color=color, alpha=.12)
    return a


def main():
    b = pd.read_csv(RES / "benchmark.csv")
    sweeps = pd.read_csv(RES / "lambda_sweep.csv")
    sel = pd.read_csv(RES / "selected_models.csv")
    base = sweeps[sweeps.scenario == "base"].copy()
    base["test_nmse"] = base.test_mse / base.noise_variance
    base["val_nmse"] = base.validation_mse / base.noise_variance
    sel["test_nmse"] = sel.test_mse / sel.noise_variance

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True)
    for col, rho in enumerate([0., .8]):
        ref = b[(b.rho == rho) & (b.seed == 20261031)].reference_objective.iloc[0]
        for method, color in [("ISTA", BLUE), ("FISTA", ORANGE)]:
            h = pd.read_csv(RES / "traces" / f"{method}_rho{rho:.1f}.csv")
            # Reference is numerical; 1e-14 floor is only for log display.
            gap = np.maximum((h.objective - ref) / max(1., abs(ref)), 1e-14)
            axes[0, col].semilogy(h.iteration, gap, color=color, label=method, lw=1.5)
            axes[1, col].semilogy(h.iteration,
                np.maximum(np.maximum(h.relative_gap, h.relative_kkt), 1e-14),
                color=color, label=method, lw=1.5)
        axes[0, col].set_title(rf"Feature correlation $\rho={rho:g}$")
        axes[0, col].set_ylabel(r"$(F(x^k)-F_{\rm ref})/\max(1,|F_{\rm ref}|)$")
        axes[1, col].set_ylabel("Max(normalized KKT, dual gap)")
        axes[1, col].axhline(1e-6, color=GREEN, ls="--", lw=1, label="Stopping tolerance")
        for ax in axes[:, col]:
            ax.set_xlabel("Iteration")
            ax.legend(fontsize=8)
    save(fig, "01_convergence")

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
    for factor, color in [(.25, "#8795A1"), (.5, BLUE), (1., ORANGE)]:
        h = pd.read_csv(RES / "traces" / f"step_{factor:.2f}.csv")
        axes[0].semilogy(h.iteration, np.maximum(h.relative_gap, h.relative_kkt),
                         color=color, label=rf"$\eta={factor:g}/L$", lw=1.7)
    axes[0].axhline(1e-6, color=GREEN, ls="--", lw=1)
    axes[0].set(xlabel="Iteration", ylabel="Max(normalized KKT, dual gap)", title="(a) Step size and convergence")
    axes[0].legend()
    for k, rho in enumerate([0., .8]):
        data = b[b.rho == rho]
        for j, (method, color) in enumerate([("ISTA", BLUE), ("FISTA", ORANGE)]):
            vals = data[data.method == method].seconds * 1000
            pos = k + (j-.5)*.3
            axes[1].bar(pos, vals.mean(), width=.28, color=color, alpha=.86,
                        yerr=vals.std(), capsize=4, label=method if k == 0 else None)
            axes[1].scatter(pos+np.linspace(-.06,.06,len(vals)), vals,
                            s=13, color="#243746", zorder=3)
    axes[1].set_xticks([0,1], [r"$\rho=0$", r"$\rho=0.8$"])
    axes[1].set(ylabel="Solver time (ms)", title="(b) Time: mean ± SD across 5 seeds")
    axes[1].legend()
    save(fig, "02_steps_and_time")

    sol = np.load(RES / "base_solutions.npz")
    truth = sol["beta_true"]
    estimate = sol["x_selected"] / sol["feature_scale"]
    support = np.flatnonzero(truth)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7), constrained_layout=True)
    axes[0].vlines(support, 0, truth[support], color=BLUE, alpha=.55, lw=1)
    axes[0].scatter(support, truth[support], s=25, facecolors="none", edgecolors=BLUE, label="True coefficient")
    nz = np.flatnonzero(np.abs(estimate)>1e-5)
    axes[0].scatter(nz, estimate[nz], s=15, marker="x", color=ORANGE, label="Lasso estimate")
    axes[0].axhline(0, color="#9AA5AF", lw=.7)
    axes[0].set(xlabel="Feature index (zero-based)", ylabel="Coefficient (original scale)", title="(a) Sparse coefficient recovery")
    axes[0].legend(fontsize=8)
    order = support[np.argsort(truth[support])]
    pos = np.arange(len(order))
    axes[1].bar(pos-.19, truth[order], .36, color=BLUE, label="True")
    axes[1].bar(pos+.19, estimate[order], .36, color=ORANGE, label="Lasso")
    axes[1].set_xticks(pos[::2], [str(i) for i in order[::2]], rotation=45)
    axes[1].set(xlabel="True active feature (sorted by coefficient)", ylabel="Coefficient", title="(b) Shrinkage on the true support")
    axes[1].legend(fontsize=8)
    save(fig, "03_recovery")

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.3), constrained_layout=True)
    band(axes[0,0], base, "lambda_ratio", "val_nmse", "Validation MSE", BLUE, log=True)
    band(axes[0,0], base, "lambda_ratio", "test_nmse", "Test MSE", ORANGE, marker="s", log=True)
    axes[0,0].axhline(1, color=GREEN, ls="--", lw=1, label="Noise variance")
    axes[0,0].set(yscale="log", ylabel=r"MSE / $\sigma^2$", title="(a) Prediction error")
    axes[0,0].legend(fontsize=8)
    band(axes[0,1], base, "lambda_ratio", "nnz", "Selected variables", BLUE)
    axes[0,1].axhline(20, color=GREEN, ls="--", lw=1, label="True sparsity = 20")
    axes[0,1].set(ylabel="Number of nonzero coefficients", title="(b) Model size")
    axes[0,1].legend(fontsize=8)
    band(axes[1,0], base, "lambda_ratio", "relative_error", "Relative error", BLUE)
    axes[1,0].set(ylabel=r"$\|\hat\beta-\beta_{\rm true}\|_2 / \|\beta_{\rm true}\|_2$", title="(c) Coefficient error")
    band(axes[1,1], base, "lambda_ratio", "precision", "Precision", BLUE)
    band(axes[1,1], base, "lambda_ratio", "recall", "Recall", ORANGE, marker="s")
    axes[1,1].set(ylabel="Support recovery", ylim=(-.04,1.05), title="(d) Variable selection")
    axes[1,1].legend(fontsize=8)
    for ax in axes.flat:
        ax.set(xscale="log", xlabel=r"$\lambda / \lambda_{\max}$")
    save(fig, "04_regularization")

    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.2), constrained_layout=True)
    studies = [("n", ["n150","base","n600"], "Training samples", "(a) Sample size"),
               ("noise_ratio", ["noise005","base","noise030"], "Noise / signal SD", "(b) Noise level"),
               ("rho", ["base","rho050","rho090"], r"Feature correlation $\rho$", "(c) Correlation")]
    for col, (x, names, label, title) in enumerate(studies):
        df = sel[sel.scenario.isin(names)]
        for row, (y, ylabel, color) in enumerate([
                ("relative_error", "Relative coefficient error", BLUE),
                ("f1", "Support F1 score", ORANGE)]):
            a = df.groupby(x)[y].agg(["mean", "std"])
            axes[row,col].errorbar(a.index, a["mean"], yerr=a["std"],
                                   color=color, marker="o", capsize=4, lw=1.6)
            axes[row,col].set(xlabel=label, ylabel=ylabel)
            if row == 0:
                axes[row,col].set_title(title)
            axes[row,col].set_ylim(bottom=0)
    save(fig, "05_sensitivity")

    b.groupby(["rho","method"])[["iterations","seconds","relative_gap","relative_kkt"]].agg(["mean","std"]).to_csv(RES / "benchmark_summary.csv")
    sel.groupby("scenario")[["relative_error","test_mse","test_nmse","nnz","precision","recall","f1"]].agg(["mean","std"]).to_csv(RES / "sensitivity_summary.csv")
    base.groupby("lambda_ratio")[["relative_error","test_mse","val_nmse","nnz","precision","recall"]].agg(["mean","std"]).to_csv(RES / "regularization_summary.csv")
    print("Generated 5 PDF/PNG figures and 3 aggregate CSV tables.")


if __name__ == "__main__":
    main()
