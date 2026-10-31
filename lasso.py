"""Lasso solved by ISTA/FISTA, with independently computable certificates.

Objective: ||A x - b||^2 / (2 n) + lam * ||x||_1.
The solvers use NumPy matrix-vector products, never a library Lasso solver.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import eigh


def soft_threshold(z, tau):
    """Proximal operator of tau * ||.||_1 (coordinatewise)."""
    return np.sign(z) * np.maximum(np.abs(z) - tau, 0.0)


def lipschitz_constant(A):
    """Largest eigenvalue of the smaller Gram matrix, with rounding cushion."""
    n, p = A.shape
    gram = (A @ A.T if n <= p else A.T @ A) / n
    largest = eigh(gram, subset_by_index=[min(n, p) - 1] * 2,
                   eigvals_only=True, check_finite=False)[0]
    return float(max(0.0, largest) * (1.0 + 1e-12))


def objective(A, b, x, lam):
    residual = A @ x - b
    return float(residual @ residual / (2 * len(b)) + lam * np.abs(x).sum())


def certificate(b, x, residual, gradient, lam, kkt_scale):
    """KKT residual and feasible-dual lower bound, without extra matvecs.

    gradient = A.T @ residual / n; theta = -residual / (n * scaling).
    The scalar scaling makes ||A.T theta||_inf <= lam.
    Exact zeros are used for KKT, independently of the reporting threshold.
    """
    n = len(b)
    primal = float(residual @ residual / (2 * n) + lam * np.abs(x).sum())
    violation = np.maximum(np.abs(gradient) - lam, 0.0)
    active = x != 0.0
    violation[active] = np.abs(gradient[active] + lam * np.sign(x[active]))
    kkt = float(violation.max(initial=0.0))
    dual_scale = max(1.0, float(np.abs(gradient).max(initial=0.0)) / lam)
    theta = -residual / (n * dual_scale)
    dual = float(b @ theta - n / 2 * (theta @ theta))
    raw_gap = primal - dual
    if raw_gap < -1e-9 * max(1.0, abs(primal)):
        raise ArithmeticError("Negative primal-dual gap beyond roundoff")
    gap = max(0.0, raw_gap)
    return dict(objective=primal, dual=dual, dual_gap=gap,
                relative_gap=gap / max(1.0, abs(primal)),
                kkt=kkt, relative_kkt=kkt / kkt_scale)


@dataclass
class Solution:
    x: np.ndarray
    iterations: int
    seconds: float
    converged: bool
    metrics: dict
    history: list


def solve(A, b, lam, method="ISTA", L=None, step_factor=1.0,
          tol=1e-6, max_iter=50000, trace=False):
    """Cold-start solver; stop only when BOTH normalized KKT and gap <= tol.

    FISTA uses the standard (unrestarted) momentum. Because the quadratic
    gradient is affine, gradient(y) is extrapolated from gradients at the two
    consecutive x iterates. Both methods therefore use one A and one A.T
    product per iteration, including the convergence certificate.
    Runtime includes initial gradient, per-iteration certificates, and tracing
    if requested. It excludes L computation and input validation.
    """
    A, b = np.asarray(A, dtype=float), np.asarray(b, dtype=float)
    if A.ndim != 2 or b.ndim != 1 or A.shape[0] != len(b) or not A.size:
        raise ValueError("A must be nonempty 2D, b a compatible vector")
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("A and b must be finite")
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("lam must be positive and finite")
    if method not in ("ISTA", "FISTA"):
        raise ValueError("method must be ISTA or FISTA")
    if not 0 < step_factor <= 1 or tol <= 0 or max_iter < 1:
        raise ValueError("Require 0 < step_factor <= 1, tol > 0, max_iter >= 1")
    L = lipschitz_constant(A) if L is None else float(L)
    if not np.isfinite(L) or L < 0:
        raise ValueError("L must be nonnegative and finite")
    eta = step_factor / (L if L > 0 else 1.0)
    start = perf_counter()
    x = np.zeros(A.shape[1])
    residual = -b.copy()
    grad = A.T @ residual / len(b)
    kkt_scale = max(1.0, float(np.abs(grad).max()))
    y, grad_y, t = x.copy(), grad.copy(), 1.0
    history = []
    metrics = certificate(b, x, residual, grad, lam, kkt_scale)
    if trace:
        history.append(dict(iteration=0, seconds=perf_counter() - start, **metrics))
    iteration = 0
    for iteration in range(1, max_iter + 1):
        if max(metrics["relative_kkt"], metrics["relative_gap"]) <= tol:
            iteration -= 1
            break
        x_new = soft_threshold(y - eta * grad_y, eta * lam)
        residual = A @ x_new - b
        grad_new = A.T @ residual / len(b)
        metrics = certificate(b, x_new, residual, grad_new, lam, kkt_scale)
        if trace:
            history.append(dict(iteration=iteration,
                                seconds=perf_counter() - start, **metrics))
        if method == "FISTA":
            t_new = (1 + np.sqrt(1 + 4 * t * t)) / 2
            momentum = (t - 1) / t_new
            y = x_new + momentum * (x_new - x)
            grad_y = grad_new + momentum * (grad_new - grad)
            t = t_new
        else:
            y, grad_y = x_new, grad_new
        x, grad = x_new, grad_new
    seconds = perf_counter() - start
    converged = max(metrics["relative_kkt"], metrics["relative_gap"]) <= tol
    return Solution(x, iteration, seconds, converged, metrics, history)
