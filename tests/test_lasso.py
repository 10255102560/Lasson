"""Numerical checks against closed forms, duality, and an independent solver."""
import unittest
import numpy as np
from sklearn.linear_model import Lasso
from threadpoolctl import threadpool_limits
from lasso import solve, soft_threshold, lipschitz_constant, certificate, objective
from data import make_problem


class LassoTests(unittest.TestCase):
    def test_diagonal_closed_form(self):
        # Nonuniform diagonal prevents an identity-only trivial check.
        A = np.diag([1., 2., 3., 4.])
        b = np.array([2., -.4, -3., .01])
        lam = .12
        d = np.diag(A)
        expected = soft_threshold(d * b / 4, lam) / (d * d / 4)
        for method in ("ISTA", "FISTA"):
            s = solve(A, b, lam, method, tol=1e-10)
            self.assertTrue(s.converged)
            np.testing.assert_allclose(s.x, expected, atol=1e-8)

    def test_zero_solution_at_lambda_max(self):
        d = make_problem(42, n=30, p=60, sparsity=5, n_val=20, n_test=20)
        for method in ("ISTA", "FISTA"):
            s = solve(d.A, d.b, d.lambda_max * (1 + 1e-12), method)
            np.testing.assert_array_equal(s.x, 0)
            self.assertTrue(s.converged)
            self.assertEqual(s.iterations, 0)

    def test_independent_coordinate_descent_and_monotonicity(self):
        d = make_problem(8, n=70, p=100, sparsity=8, rho=.6, n_val=20, n_test=20)
        lam = .04 * d.lambda_max
        ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-13,
                    max_iter=100000).fit(d.A, d.b)
        fref = objective(d.A, d.b, ref.coef_, lam)
        for method in ("ISTA", "FISTA"):
            s = solve(d.A, d.b, lam, method, tol=1e-9, trace=True)
            self.assertTrue(s.converged)
            self.assertLess(abs(s.metrics["objective"] - fref), 1e-8)
            np.testing.assert_allclose(d.A @ s.x, d.A @ ref.coef_, atol=2e-6)
            if method == "ISTA":
                self.assertLessEqual(np.diff([h["objective"] for h in s.history]).max(), 1e-12)

    def test_dual_lower_bound_at_arbitrary_iterate(self):
        d = make_problem(3, n=35, p=60, sparsity=5, n_val=15, n_test=15)
        lam = .1 * d.lambda_max
        ref = Lasso(alpha=lam, fit_intercept=False, tol=1e-13,
                    max_iter=100000).fit(d.A, d.b)
        x = np.random.default_rng(4).normal(size=60)
        residual = d.A @ x - d.b
        gradient = d.A.T @ residual / len(d.b)
        c = certificate(d.b, x, residual, gradient, lam, max(1, d.lambda_max))
        self.assertLessEqual(c["dual"], objective(d.A, d.b, ref.coef_, lam) + 1e-10)
        self.assertGreaterEqual(c["dual_gap"], 0)
        theta = -residual / (len(d.b) * max(1, np.max(abs(gradient)) / lam))
        self.assertLessEqual(np.max(abs(d.A.T @ theta)), lam * (1 + 1e-12))

    def test_preprocessing_and_separate_random_streams(self):
        d = make_problem(5, n=30, p=60, sparsity=5, n_val=20, n_test=20)
        e = make_problem(5, n=30, p=60, sparsity=5, n_val=45, n_test=50)
        np.testing.assert_allclose(d.A.mean(axis=0), 0, atol=1e-14)
        np.testing.assert_allclose(d.A.std(axis=0), 1, atol=1e-14)
        np.testing.assert_array_equal(d.A, e.A)
        np.testing.assert_array_equal(d.b, e.b)
        np.testing.assert_array_equal(d.beta_true, e.beta_true)

    def test_spectral_constant_and_zero_design(self):
        rng = np.random.default_rng(14)
        for shape in [(7, 11), (11, 7)]:
            A = rng.normal(size=shape)
            self.assertAlmostEqual(lipschitz_constant(A),
                                   np.linalg.norm(A, 2)**2 / shape[0], places=10)
        s = solve(np.zeros((3, 5)), np.ones(3), .1)
        self.assertTrue(s.converged)
        np.testing.assert_array_equal(s.x, 0)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        unittest.main()
