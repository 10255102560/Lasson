"""Seeded Gaussian sparse regression; train-only centering and scaling."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Problem:
    A: np.ndarray
    b: np.ndarray
    A_val: np.ndarray
    y_val: np.ndarray
    A_test: np.ndarray
    y_test: np.ndarray
    signal_test: np.ndarray
    beta_true: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    response_mean: float
    sigma: float

    @property
    def lambda_max(self):
        return float(np.abs(self.A.T @ self.b).max() / len(self.b))

    def evaluate(self, x):
        beta = x / self.feature_scale
        selected = np.abs(beta) > 1e-5
        truth = self.beta_true != 0
        tp = int(np.sum(selected & truth))
        fp = int(np.sum(selected & ~truth))
        fn = int(np.sum(~selected & truth))
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        pred_val = self.A_val @ x + self.response_mean
        pred_test = self.A_test @ x + self.response_mean
        return dict(relative_error=float(np.linalg.norm(beta - self.beta_true)
                                         / np.linalg.norm(self.beta_true)),
                    nnz=int(selected.sum()), tp=tp, fp=fp, fn=fn,
                    precision=precision, recall=recall,
                    f1=2 * precision * recall / max(1e-30, precision + recall),
                    validation_mse=float(np.mean((pred_val - self.y_val)**2)),
                    test_mse=float(np.mean((pred_test - self.y_test)**2)),
                    signal_mse=float(np.mean((pred_test - self.signal_test)**2)),
                    noise_variance=self.sigma**2)


def make_problem(seed, n=300, p=1000, sparsity=20, rho=0.0,
                 noise_ratio=0.1, n_val=300, n_test=1000):
    """Cov(Z_i,Z_j)=rho^|i-j|; sigma = noise_ratio * population signal SD.

    Separate random streams make sample-size/noise comparisons paired by seed.
    Test and validation arrays never affect training preprocessing.
    """
    if not 0 <= rho < 1 or not 0 < sparsity <= p or noise_ratio <= 0:
        raise ValueError("Invalid correlation, sparsity, or noise level")
    streams = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(7)]
    beta = np.zeros(p)
    support = streams[0].choice(p, sparsity, replace=False)
    beta[support] = streams[0].choice([-1, 1], sparsity) * streams[0].uniform(.5, 1.5, sparsity)
    cov_support = rho ** np.abs(support[:, None] - support[None, :])
    sigma = float(noise_ratio * np.sqrt(beta[support] @ cov_support @ beta[support]))

    def design(rng, rows):
        z = rng.standard_normal((rows, p))
        if rho:
            innovation = np.sqrt(1 - rho * rho)
            for j in range(1, p):
                z[:, j] = rho * z[:, j - 1] + innovation * z[:, j]
        return z

    z, zv, zt = (design(r, rows) for r, rows in
                 zip(streams[1:4], [n, n_val, n_test]))
    y = z @ beta + sigma * streams[4].standard_normal(n)
    yv = zv @ beta + sigma * streams[5].standard_normal(n_val)
    st = zt @ beta
    yt = st + sigma * streams[6].standard_normal(n_test)
    mean, scale, ymean = z.mean(axis=0), z.std(axis=0), float(y.mean())
    return Problem(np.ascontiguousarray((z - mean) / scale), y - ymean,
                   np.ascontiguousarray((zv - mean) / scale), yv,
                   np.ascontiguousarray((zt - mean) / scale), yt, st,
                   beta, mean, scale, ymean, sigma)
