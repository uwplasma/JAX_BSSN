import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import (
    BSSNParameters,
    BSSNVariables,
    compute_shift_derivatives,
    evolve_conformal_connection,
    evolve_conformal_metric,
    evolve_lapse,
    evolve_shift,
)
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.evolve import rk4_step


class TestShiftEvolution(unittest.TestCase):
    """Tests for nonzero-shift BSSN evolution terms."""

    def setUp(self):
        self.n = 32
        x = jnp.linspace(-jnp.pi, jnp.pi, self.n, endpoint=False)
        self.dx = float(x[1] - x[0])
        self.X, self.Y, self.Z = jnp.meshgrid(x, x, x, indexing="ij")
        self.shape = (self.n, self.n, self.n)
        self.params = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0, g=0.7, eta=0.2)

    def flat_vars(
        self,
        shift=None,
        conformal_connection=None,
        lapse=None,
        trace_K=None,
        traceless_K=None,
    ):
        gamma = jnp.eye(3)[:, :, None, None, None] * jnp.ones((3, 3) + self.shape)
        W = jnp.ones(self.shape)
        A = jnp.zeros((3, 3) + self.shape) if traceless_K is None else traceless_K
        K = jnp.zeros(self.shape) if trace_K is None else trace_K
        Gamma = (
            jnp.zeros((3,) + self.shape)
            if conformal_connection is None
            else conformal_connection
        )
        alpha = jnp.ones(self.shape) if lapse is None else lapse
        beta = jnp.zeros((3,) + self.shape) if shift is None else shift

        return BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=A,
            trace_K=K,
            conformal_connection=Gamma,
            lapse=alpha,
            shift=beta,
        )

    def test_shift_derivatives_keep_component_and_derivative_axes(self):
        beta = jnp.stack(
            [
                jnp.sin(self.X) + 0.25 * jnp.cos(self.Y),
                0.5 * jnp.sin(self.Y + self.Z),
                0.25 * jnp.cos(self.X - self.Z),
            ],
            axis=0,
        )

        d_beta = compute_shift_derivatives(beta, self.dx)

        self.assertEqual(d_beta.shape, (3, 3) + self.shape)
        np.testing.assert_allclose(d_beta[0, 0], jnp.cos(self.X), atol=6.0e-5)
        np.testing.assert_allclose(d_beta[0, 1], -0.25 * jnp.sin(self.Y), atol=6.0e-5)
        np.testing.assert_allclose(d_beta[1, 1], 0.5 * jnp.cos(self.Y + self.Z), atol=6.0e-5)
        np.testing.assert_allclose(d_beta[1, 2], 0.5 * jnp.cos(self.Y + self.Z), atol=6.0e-5)
        np.testing.assert_allclose(d_beta[2, 0], -0.25 * jnp.sin(self.X - self.Z), atol=6.0e-5)
        np.testing.assert_allclose(d_beta[2, 2], 0.25 * jnp.sin(self.X - self.Z), atol=6.0e-5)

    def test_bssn_parameters_default_to_evolved_shift_and_harmonic_lapse(self):
        params = BSSNParameters()

        self.assertNotIn("f", params._fields)
        self.assertEqual(params.zero_shift, 0)
        self.assertEqual(params.gauge, 0)

    def test_conformal_metric_uses_weighted_lie_derivative_of_shift(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        vars = self.flat_vars(shift=beta)

        dt_gamma = evolve_conformal_metric(vars, self.params)

        d_beta = compute_shift_derivatives(beta, self.dx)
        div_beta = d_beta[0, 0] + d_beta[1, 1] + d_beta[2, 2]
        expected = jnp.zeros((3, 3) + self.shape)
        for i in range(3):
            for j in range(3):
                delta_ij = 1.0 if i == j else 0.0
                expected = expected.at[i, j].set(
                    d_beta[i, j] + d_beta[j, i] - (2.0 / 3.0) * delta_ij * div_beta
                )

        np.testing.assert_allclose(dt_gamma, expected, atol=2.0e-5)

    def test_conformal_connection_includes_second_shift_derivatives(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        vars = self.flat_vars(shift=beta)

        dt_Gamma = evolve_conformal_connection(vars, self.params)

        d_beta = compute_shift_derivatives(beta, self.dx)
        div_beta = d_beta[0, 0] + d_beta[1, 1] + d_beta[2, 2]
        expected = jnp.zeros((3,) + self.shape)
        for i in range(3):
            laplacian_beta_i = sum(
                diff1_field(diff1_field(beta[i], m, self.dx), m, self.dx)
                for m in range(3)
            )
            expected = expected.at[i].set(
                laplacian_beta_i
                + (1.0 / 3.0) * diff1_field(div_beta, i, self.dx)
            )

        np.testing.assert_allclose(dt_Gamma, expected, atol=3.0e-5)

    def test_lapse_rhs_advects_lapse_with_shift(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        alpha = 1.0 + 0.05 * jnp.cos(self.X + self.Y)
        vars = self.flat_vars(shift=beta, lapse=alpha)

        dt_alpha = evolve_lapse(vars, self.params)

        grad_alpha = jnp.stack(
            [diff1_field(alpha, d, self.dx) for d in range(3)],
            axis=0,
        )
        expected = jnp.einsum("i...,i...->...", beta, grad_alpha)

        np.testing.assert_allclose(dt_alpha, expected, atol=2.0e-5)

    def test_lapse_rhs_uses_harmonic_gauge_when_gauge_is_zero(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        alpha = 1.0 + 0.05 * jnp.cos(self.X + self.Y)
        K = 0.02 * jnp.sin(self.X + self.Z)
        params = self.params._replace(gauge=0)
        vars = self.flat_vars(shift=beta, lapse=alpha, trace_K=K)

        dt_alpha = evolve_lapse(vars, params)

        grad_alpha = jnp.stack(
            [diff1_field(alpha, d, self.dx) for d in range(3)],
            axis=0,
        )
        advection = jnp.einsum("i...,i...->...", beta, grad_alpha)
        expected = -alpha**2 * K + advection

        np.testing.assert_allclose(dt_alpha, expected, atol=2.0e-5)

    def test_lapse_rhs_uses_one_plus_log_gauge_when_gauge_is_one(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        alpha = 1.0 + 0.05 * jnp.cos(self.X + self.Y)
        K = 0.02 * jnp.sin(self.X + self.Z)
        params = self.params._replace(gauge=1)
        vars = self.flat_vars(shift=beta, lapse=alpha, trace_K=K)

        dt_alpha = evolve_lapse(vars, params)

        grad_alpha = jnp.stack(
            [diff1_field(alpha, d, self.dx) for d in range(3)],
            axis=0,
        )
        advection = jnp.einsum("i...,i...->...", beta, grad_alpha)
        expected = -2.0 * alpha * K + advection

        np.testing.assert_allclose(dt_alpha, expected, atol=2.0e-5)

    def test_shift_rhs_uses_single_gamma_driver(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        Gamma = jnp.stack(
            [
                0.01 * jnp.cos(self.X),
                0.02 * jnp.cos(self.Y),
                0.03 * jnp.cos(self.Z),
            ],
            axis=0,
        )
        vars = self.flat_vars(shift=beta, conformal_connection=Gamma)

        dt_beta = evolve_shift(vars, self.params)

        d_beta = compute_shift_derivatives(beta, self.dx)
        advection = jnp.einsum("j...,ij...->i...", beta, d_beta)
        expected = (
            self.params.g * Gamma
            + advection
            - self.params.eta * beta
        )

        np.testing.assert_allclose(dt_beta, expected, atol=2.0e-5)

    def test_rk4_keeps_shift_fixed_when_zero_shift_is_one(self):
        beta = jnp.stack(
            [
                0.1 * jnp.sin(self.X),
                0.2 * jnp.sin(self.Y),
                0.3 * jnp.sin(self.Z),
            ],
            axis=0,
        )
        Gamma = jnp.stack(
            [
                0.01 * jnp.cos(self.X),
                0.02 * jnp.cos(self.Y),
                0.03 * jnp.cos(self.Z),
            ],
            axis=0,
        )
        params = self.params._replace(zero_shift=1, g=0.7, eta=0.2, dt=0.001)
        vars = self.flat_vars(shift=beta, conformal_connection=Gamma)

        evolved_vars = rk4_step(vars, params)

        np.testing.assert_allclose(evolved_vars.shift, beta, atol=0.0)

    def test_rk4_evolves_shift_when_zero_shift_is_zero(self):
        Gamma = jnp.stack(
            [
                0.01 * jnp.cos(self.X),
                0.02 * jnp.cos(self.Y),
                0.03 * jnp.cos(self.Z),
            ],
            axis=0,
        )
        params = self.params._replace(zero_shift=0, g=0.7, eta=0.2, dt=0.001)
        vars = self.flat_vars(conformal_connection=Gamma)

        evolved_vars = rk4_step(vars, params)

        max_shift_change = jnp.max(jnp.abs(evolved_vars.shift - vars.shift))
        self.assertGreater(float(max_shift_change), 0.0)


if __name__ == "__main__":
    unittest.main()
