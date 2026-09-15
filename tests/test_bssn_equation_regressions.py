import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn.constraints import (
    compute_hamiltonian_constraint,
    compute_momentum_constraint,
    compute_momentum_constraint_and_derivative,
)
from JAX_BSSN.bssn.extrinsic_curvature import (
    evolve_trace_extrinsic_curvature,
    evolve_traceless_extrinsic_curvature,
)
from JAX_BSSN.bssn.geometry import (
    W_FLOOR_VALUE,
    compute_W2_covariant_lapse_hessian,
    compute_W2_ricci,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.derivatives import diff1_field, diff2_field
from JAX_BSSN.evolution.time_evolve import enforce_unit_determinant_conformal_metric, rk4_step
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    determinant_3x3_metric,
    trace_tensor,
    traceless_part,
)


class TestBSSNEquationRegressions(unittest.TestCase):
    def setUp(self):
        self.n = 24
        x = jnp.linspace(-jnp.pi, jnp.pi, self.n, endpoint=False)
        self.dx = float(x[1] - x[0])
        self.X, self.Y, self.Z = jnp.meshgrid(x, x, x, indexing="ij")
        self.shape = (self.n, self.n, self.n)

    def nontrivial_vars(self):
        q = 0.03 * jnp.sin(self.X + self.Y) + 0.02 * jnp.cos(self.Y - self.Z)

        gamma = jnp.zeros((3, 3) + self.shape)
        gamma = gamma.at[0, 0].set(jnp.exp(2.0 * q))
        gamma = gamma.at[1, 1].set(jnp.exp(-q))
        gamma = gamma.at[2, 2].set(jnp.exp(-q))

        inv_gamma = invert_3x3_metric(gamma)

        A_seed = jnp.zeros((3, 3) + self.shape)
        A_seed = A_seed.at[0, 0].set(0.02 * jnp.sin(self.X))
        A_seed = A_seed.at[1, 1].set(0.015 * jnp.cos(self.Y))
        A_seed = A_seed.at[2, 2].set(-0.01 * jnp.sin(self.Z))
        A_seed = A_seed.at[0, 1].set(0.012 * jnp.sin(self.X - self.Y))
        A_seed = A_seed.at[1, 0].set(0.012 * jnp.sin(self.X - self.Y))
        A_seed = A_seed.at[0, 2].set(0.009 * jnp.cos(self.X + self.Z))
        A_seed = A_seed.at[2, 0].set(0.009 * jnp.cos(self.X + self.Z))
        A_ij = traceless_part(A_seed, gamma, inv_gamma)

        W = 1.0 + 0.04 * jnp.sin(self.X - self.Z)
        K = 0.03 * jnp.cos(self.X + self.Y + self.Z)

        metric_derivs = jnp.stack(
            [diff1_field(gamma, d + 2, self.dx) for d in range(3)],
            axis=0,
        )
        christoffel = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
        conformal_connection = jnp.einsum(
            "mn...,imn...->i...", inv_gamma, christoffel
        )

        return BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=A_ij,
            trace_K=K,
            conformal_connection=conformal_connection,
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
        )

    def flat_vars(self, W, lapse):
        dtype = W.dtype
        gamma = jnp.eye(3, dtype=dtype)[:, :, None, None, None] * jnp.ones(
            (3, 3) + self.shape, dtype=dtype
        )

        return BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=jnp.zeros((3, 3) + self.shape, dtype=dtype),
            trace_K=jnp.zeros(self.shape, dtype=dtype),
            conformal_connection=jnp.zeros((3,) + self.shape, dtype=dtype),
            lapse=lapse,
            shift=jnp.zeros((3,) + self.shape, dtype=dtype),
        )

    def denominator_free_W2_ricci_flat(self, W):
        dx = jnp.asarray(self.dx, dtype=W.dtype)
        dWdi = jnp.stack(
            [diff1_field(W, d, dx) for d in range(3)], axis=0
        )
        dWdij = jnp.stack(
            [
                jnp.stack(
                    [
                        diff2_field(W, j, dx)
                        if i == j
                        else diff1_field(dWdi[i], j, dx)
                        for j in range(3)
                    ],
                    axis=0,
                )
                for i in range(3)
            ],
            axis=0,
        )

        laplacian_W = jnp.einsum("ii...->...", dWdij)
        gradient_W_squared = jnp.einsum("i...,i...->...", dWdi, dWdi)
        gamma = jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]

        return (
            W * dWdij
            + gamma * W * laplacian_W
            - 2.0 * gamma * gradient_W_squared
        )

    def denominator_free_W2_lapse_hessian_flat(self, W, lapse):
        dx = jnp.asarray(self.dx, dtype=W.dtype)
        dWdi = jnp.stack(
            [diff1_field(W, d, dx) for d in range(3)], axis=0
        )
        dalphadi = jnp.stack(
            [diff1_field(lapse, d, dx) for d in range(3)], axis=0
        )
        dalphadij = jnp.stack(
            [
                jnp.stack(
                    [
                        diff2_field(lapse, j, dx)
                        if i == j
                        else diff1_field(dalphadi[i], j, dx)
                        for j in range(3)
                    ],
                    axis=0,
                )
                for i in range(3)
            ],
            axis=0,
        )

        gradient_terms = (
            jnp.einsum("i...,j...->ij...", dWdi, dalphadi)
            + jnp.einsum("j...,i...->ij...", dWdi, dalphadi)
        )
        gradient_W_alpha = jnp.einsum("i...,i...->...", dWdi, dalphadi)
        gamma = jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]

        return W**2 * dalphadij + W * (
            gradient_terms - gamma * gradient_W_alpha
        )

    def canonical_divided_W2_sources_flat(self, W, lapse):
        """Return the former divided expressions, only for strictly positive W."""

        dx = jnp.asarray(self.dx, dtype=W.dtype)
        dWdi = jnp.stack(
            [diff1_field(W, d, dx) for d in range(3)], axis=0
        )
        dalphadi = jnp.stack(
            [diff1_field(lapse, d, dx) for d in range(3)], axis=0
        )
        dWdij = jnp.stack(
            [
                jnp.stack(
                    [
                        diff2_field(W, j, dx)
                        if i == j
                        else diff1_field(dWdi[i], j, dx)
                        for j in range(3)
                    ],
                    axis=0,
                )
                for i in range(3)
            ],
            axis=0,
        )
        dalphadij = jnp.stack(
            [
                jnp.stack(
                    [
                        diff2_field(lapse, j, dx)
                        if i == j
                        else diff1_field(dalphadi[i], j, dx)
                        for j in range(3)
                    ],
                    axis=0,
                )
                for i in range(3)
            ],
            axis=0,
        )

        gamma = jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        laplacian_W = jnp.einsum("ii...->...", dWdij)
        gradient_W_squared = jnp.einsum("i...,i...->...", dWdi, dWdi)
        gradient_terms = (
            jnp.einsum("i...,j...->ij...", dWdi, dalphadi)
            + jnp.einsum("j...,i...->ij...", dWdi, dalphadi)
            - gamma * jnp.einsum("i...,i...->...", dWdi, dalphadi)
        )

        ricci = (
            dWdij / W
            + gamma * laplacian_W / W
            - 2.0 * gamma * gradient_W_squared / W**2
        )
        lapse_hessian = dalphadij + gradient_terms / W

        return W**2 * ricci, W**2 * lapse_hessian

    def assert_normalized_allclose(self, actual, expected, rtol, atol):
        expected_array = np.asarray(expected)
        expected_magnitude = float(np.max(np.abs(expected_array)))
        self.assertGreater(
            expected_magnitude / W_FLOOR_VALUE**2,
            1.0e-3,
            "The scaled reference must be nontrivial relative to W_FLOOR_VALUE**2.",
        )
        scale = max(
            expected_magnitude,
            np.finfo(expected_array.dtype).tiny,
        )
        np.testing.assert_allclose(
            np.asarray(actual) / scale,
            expected_array / scale,
            rtol=rtol,
            atol=atol,
        )

    def test_scaled_tensors_match_denominator_free_formulas_across_W_floor(self):
        cases = {
            "above": 4.0,
            "touching": 2.0,
            "straddling": 1.5,
            "below": 0.25,
        }

        for dtype, rtol, atol in (
            (jnp.float64, 1.0e-10, 1.0e-12),
            (jnp.float32, 2.0e-5, 2.0e-6),
        ):
            X = self.X.astype(dtype)
            Y = self.Y.astype(dtype)
            Z = self.Z.astype(dtype)
            lapse = (
                1.0
                + 0.10 * jnp.sin(X)
                + 0.07 * jnp.cos(Y)
                + 0.03 * jnp.sin(X + Z)
            ).astype(dtype)

            for label, amplitude in cases.items():
                with self.subTest(dtype=str(dtype), case=label):
                    dx = jnp.asarray(self.dx, dtype=dtype)
                    scale = jnp.asarray(
                        amplitude * W_FLOOR_VALUE, dtype=dtype
                    )
                    W = scale * (0.75 + 0.25 * jnp.cos(X))
                    vars = self.flat_vars(W, lapse)
                    params = BSSNParameters(
                        dx=dx, dt=0.01, nu=0.0, kappa=0.0
                    )

                    actual_ricci = compute_W2_ricci(vars, params)
                    actual_hessian = compute_W2_covariant_lapse_hessian(
                        vars, params
                    )
                    expected_ricci = self.denominator_free_W2_ricci_flat(W)
                    expected_hessian = (
                        self.denominator_free_W2_lapse_hessian_flat(W, lapse)
                    )

                    self.assertEqual(actual_ricci.dtype, dtype)
                    self.assertEqual(actual_hessian.dtype, dtype)
                    self.assertTrue(bool(jnp.all(jnp.isfinite(actual_ricci))))
                    self.assertTrue(bool(jnp.all(jnp.isfinite(actual_hessian))))
                    self.assert_normalized_allclose(
                        actual_ricci, expected_ricci, rtol, atol
                    )
                    self.assert_normalized_allclose(
                        actual_hessian, expected_hessian, rtol, atol
                    )

                    if label == "above":
                        divided_ricci, divided_hessian = (
                            self.canonical_divided_W2_sources_flat(W, lapse)
                        )
                        self.assert_normalized_allclose(
                            actual_ricci, divided_ricci, rtol, atol
                        )
                        self.assert_normalized_allclose(
                            actual_hessian, divided_hessian, rtol, atol
                        )

                    self.assert_normalized_allclose(
                        actual_ricci,
                        jnp.swapaxes(actual_ricci, 0, 1),
                        rtol,
                        max(atol, 5.0e-5 if dtype == jnp.float32 else atol),
                    )
                    self.assert_normalized_allclose(
                        actual_hessian,
                        jnp.swapaxes(actual_hessian, 0, 1),
                        rtol,
                        max(atol, 5.0e-5 if dtype == jnp.float32 else atol),
                    )

    def test_K_A_and_hamiltonian_use_scaled_sources_at_smooth_puncture(self):
        for dtype, rtol, atol in (
            (jnp.float64, 1.0e-10, 1.0e-12),
            (jnp.float32, 2.0e-5, 2.0e-6),
        ):
            with self.subTest(dtype=str(dtype)):
                dx = jnp.asarray(self.dx, dtype=dtype)
                x = (jnp.arange(self.n, dtype=dtype) - self.n // 2) * dx
                X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
                W = jnp.asarray(4.0 * W_FLOOR_VALUE, dtype=dtype) * (
                    jnp.sin(X / 2.0) ** 2
                    + jnp.sin(Y / 2.0) ** 2
                    + jnp.sin(Z / 2.0) ** 2
                )
                lapse = (
                    1.0
                    + 0.10 * jnp.sin(X)
                    + 0.07 * jnp.cos(Y)
                    + 0.03 * jnp.sin(X + Z)
                ).astype(dtype)
                vars = self.flat_vars(W, lapse)
                zero = jnp.asarray(0.0, dtype=dtype)
                params = BSSNParameters(
                    dx=dx,
                    dt=jnp.asarray(0.01, dtype=dtype),
                    nu=zero,
                    kappa=zero,
                    g=zero,
                    eta=zero,
                )

                self.assertTrue(bool(jnp.any(W == 0.0)))
                self.assertTrue(
                    bool(jnp.any((W > 0.0) & (W < W_FLOOR_VALUE)))
                )
                self.assertTrue(bool(jnp.any(W > W_FLOOR_VALUE)))

                W2_ricci = self.denominator_free_W2_ricci_flat(W)
                W2_hessian = self.denominator_free_W2_lapse_hessian_flat(
                    W, lapse
                )
                expected_K = -jnp.einsum("ii...->...", W2_hessian)
                A_source = lapse * W2_ricci - W2_hessian
                A_source_trace = jnp.einsum("ii...->...", A_source)
                expected_A = (
                    A_source
                    - vars.conformal_metric * A_source_trace / 3.0
                )
                expected_hamiltonian = jnp.einsum("ii...->...", W2_ricci)

                actual_K = evolve_trace_extrinsic_curvature(vars, params)
                actual_A = evolve_traceless_extrinsic_curvature(vars, params)
                actual_hamiltonian = compute_hamiltonian_constraint(vars, params)

                self.assertTrue(bool(jnp.all(jnp.isfinite(actual_K))))
                self.assertTrue(bool(jnp.all(jnp.isfinite(actual_A))))
                self.assertTrue(bool(jnp.all(jnp.isfinite(actual_hamiltonian))))
                self.assert_normalized_allclose(actual_K, expected_K, rtol, atol)
                self.assert_normalized_allclose(actual_A, expected_A, rtol, atol)
                self.assert_normalized_allclose(
                    actual_hamiltonian, expected_hamiltonian, rtol, atol
                )
                self.assert_normalized_allclose(
                    actual_A,
                    jnp.swapaxes(actual_A, 0, 1),
                    rtol,
                    max(atol, 5.0e-5 if dtype == jnp.float32 else atol),
                )

                trace_A = trace_tensor(
                    actual_A, invert_3x3_metric(vars.conformal_metric)
                )
                source_scale = float(jnp.max(jnp.abs(expected_A)))
                normalized_trace = float(jnp.max(jnp.abs(trace_A))) / source_scale
                trace_tolerance = max(
                    rtol, 5.0e-5 if dtype == jnp.float32 else rtol
                )
                self.assertLessEqual(normalized_trace, trace_tolerance)

    def test_momentum_constraint_matches_notes_formula(self):
        vars = self.nontrivial_vars()
        gamma = vars.conformal_metric
        inv_gamma = invert_3x3_metric(gamma)
        A_ij = vars.traceless_K
        W = vars.conformal_factor
        K = vars.trace_K

        params = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0)
        momentum = compute_momentum_constraint(vars, params)

        A_i_up_j = jnp.einsum("jk...,ik...->ij...", inv_gamma, A_ij)
        dA_i_up_j_dk = jnp.stack(
            [diff1_field(A_i_up_j, d + 2, self.dx) for d in range(3)],
            axis=0,
        )
        dA_jk_di = jnp.stack(
            [diff1_field(A_ij, d + 2, self.dx) for d in range(3)],
            axis=0,
        )
        dWdi = jnp.stack([diff1_field(W, d, self.dx) for d in range(3)], axis=0)
        dKdi = jnp.stack([diff1_field(K, d, self.dx) for d in range(3)], axis=0)

        expected = jnp.einsum("jij...->i...", dA_i_up_j_dk)
        expected += -0.5 * jnp.einsum(
            "jk...,ijk...->i...", inv_gamma, dA_jk_di
        )
        expected += -3.0 * jnp.einsum("ij...,j...->i...", A_i_up_j, dWdi) / W
        expected += -(2.0 / 3.0) * dKdi

        np.testing.assert_allclose(momentum, expected, atol=1.0e-12)

    def test_momentum_constraint_does_not_depend_on_evolved_Gamma(self):
        vars = self.nontrivial_vars()
        params = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0)

        shifted_Gamma = vars.conformal_connection + jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.05 * jnp.cos(self.Y),
                0.07 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        vars_with_shifted_Gamma = vars._replace(conformal_connection=shifted_Gamma)

        momentum = compute_momentum_constraint(vars, params)
        shifted_momentum = compute_momentum_constraint(vars_with_shifted_Gamma, params)

        np.testing.assert_allclose(shifted_momentum, momentum, atol=1.0e-12)

    def test_kappa_term_changes_traceless_extrinsic_curvature_rhs(self):
        vars = self.nontrivial_vars()
        params0 = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0, kappa=0.0)
        params1 = params0._replace(kappa=2.5)

        rhs0 = evolve_traceless_extrinsic_curvature(vars, params0)
        rhs1 = evolve_traceless_extrinsic_curvature(vars, params1)

        inv_gamma = invert_3x3_metric(vars.conformal_metric)
        metric_derivs = jnp.stack(
            [
                diff1_field(vars.conformal_metric, d + 2, self.dx)
                for d in range(3)
            ],
            axis=0,
        )
        christoffel = christoffel_symbols_second_kind(inv_gamma, metric_derivs)
        M_i, dMidj = compute_momentum_constraint_and_derivative(vars, params1)

        DjMi = dMidj - jnp.einsum("kij...,k...->ij...", christoffel, M_i)
        DiMj = jnp.swapaxes(DjMi, 0, 1)
        expected_difference = 0.5 * params1.kappa * vars.lapse * (DjMi + DiMj)

        np.testing.assert_allclose(rhs1 - rhs0, expected_difference, atol=4.0e-7)

    def test_rk4_projects_pure_trace_A_before_first_rhs(self):
        gamma = jnp.eye(3)[:, :, None, None, None] * jnp.ones(
            (3, 3) + self.shape
        )
        pure_trace_A = 0.02 * jnp.sin(self.X)
        A_ij = gamma * pure_trace_A

        vars = BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=jnp.ones(self.shape),
            traceless_K=A_ij,
            trace_K=jnp.zeros(self.shape),
            conformal_connection=jnp.zeros((3,) + self.shape),
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
        )
        params = BSSNParameters(
            dx=self.dx, dt=0.01, nu=0.0, kappa=0.0, g=0.0, eta=0.0
        )

        evolved = rk4_step(vars, params)

        np.testing.assert_allclose(evolved.conformal_metric, gamma, atol=1.0e-12)
        trace_A = trace_tensor(
            evolved.traceless_K, invert_3x3_metric(evolved.conformal_metric)
        )
        np.testing.assert_allclose(trace_A, jnp.zeros_like(trace_A), atol=1.0e-12)

    def test_unit_determinant_projection_rescales_only_conformal_metric(self):
        gamma = jnp.eye(3)[:, :, None, None, None] * jnp.ones(
            (3, 3) + self.shape
        )
        gamma = gamma.at[0, 0].set(1.2 + 0.01 * jnp.sin(self.X))
        gamma = gamma.at[1, 1].set(0.9 + 0.01 * jnp.cos(self.Y))
        gamma = gamma.at[2, 2].set(1.1 + 0.01 * jnp.sin(self.Z))
        gamma = gamma.at[0, 1].set(0.02 * jnp.sin(self.X + self.Y))
        gamma = gamma.at[1, 0].set(gamma[0, 1])

        traceless_K = jnp.zeros((3, 3) + self.shape)
        traceless_K = traceless_K.at[0, 0].set(0.01 * jnp.sin(self.X))

        vars = BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=1.0 + 0.03 * jnp.cos(self.X),
            traceless_K=traceless_K,
            trace_K=0.02 * jnp.sin(self.Y),
            conformal_connection=jnp.stack(
                [
                    0.01 * jnp.sin(self.X),
                    0.02 * jnp.sin(self.Y),
                    0.03 * jnp.sin(self.Z),
                ],
                axis=0,
            ),
            lapse=1.0 + 0.01 * jnp.cos(self.Z),
            shift=jnp.zeros((3,) + self.shape),
        )

        projected = enforce_unit_determinant_conformal_metric(vars)

        det_gamma = determinant_3x3_metric(projected.conformal_metric)
        np.testing.assert_allclose(det_gamma, jnp.ones_like(det_gamma), atol=1.0e-12)
        np.testing.assert_allclose(projected.conformal_factor, vars.conformal_factor, atol=0.0)
        np.testing.assert_allclose(projected.traceless_K, vars.traceless_K, atol=0.0)
        np.testing.assert_allclose(projected.trace_K, vars.trace_K, atol=0.0)
        np.testing.assert_allclose(projected.conformal_connection, vars.conformal_connection, atol=0.0)
        np.testing.assert_allclose(projected.lapse, vars.lapse, atol=0.0)
        np.testing.assert_allclose(projected.shift, vars.shift, atol=0.0)

    def test_rk4_enforces_unit_determinant_conformal_metric(self):
        gamma = jnp.eye(3)[:, :, None, None, None] * jnp.ones(
            (3, 3) + self.shape
        )
        gamma = gamma.at[0, 0].set(1.2 + 0.01 * jnp.sin(self.X))
        gamma = gamma.at[1, 1].set(0.9 + 0.01 * jnp.cos(self.Y))
        gamma = gamma.at[2, 2].set(1.1 + 0.01 * jnp.sin(self.Z))
        W = 1.0 + 0.03 * jnp.cos(self.X)

        vars = BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=jnp.zeros((3, 3) + self.shape),
            trace_K=jnp.zeros(self.shape),
            conformal_connection=jnp.zeros((3,) + self.shape),
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
        )
        params = BSSNParameters(
            dx=self.dx, dt=0.01, nu=0.0, kappa=0.0, g=0.0, eta=0.0
        )

        evolved = rk4_step(vars, params)

        det_gamma = determinant_3x3_metric(evolved.conformal_metric)
        np.testing.assert_allclose(det_gamma, jnp.ones_like(det_gamma), atol=1.0e-12)
        np.testing.assert_allclose(evolved.conformal_factor, W, atol=2.0e-4)


if __name__ == "__main__":
    unittest.main()
