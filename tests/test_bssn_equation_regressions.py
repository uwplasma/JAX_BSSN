import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import (
    BSSNParameters,
    BSSNVariables,
    compute_momentum_constraint,
    evolve_trace_extrinsic_curvature,
    evolve_traceless_extrinsic_curvature,
)
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.evolve import enforce_unit_determinant_conformal_metric, rk4_step
from JAX_BSSN.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    determinant_3x3_metric,
    trace_tensor,
    traceless_part,
)
from tests.helpers import vacuum_matter_fields


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
        rho, stress_tensor, momentum_density = vacuum_matter_fields(
            self.shape, gamma.dtype
        )

        return BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=A_ij,
            trace_K=K,
            conformal_connection=conformal_connection,
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
            rho=rho,
            S_ij=stress_tensor,
            momentum_density=momentum_density,
        )

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
        M_i = compute_momentum_constraint(vars, params1)

        dMidj = jnp.zeros((3,) + M_i.shape)
        for i in range(3):
            for j in range(3):
                dMidj = dMidj.at[i, j].set(diff1_field(M_i[i], j, self.dx))

        DjMi = dMidj - jnp.einsum("kij...,k...->ij...", christoffel, M_i)
        DiMj = jnp.swapaxes(DjMi, 0, 1)
        expected_difference = 0.5 * params1.kappa * vars.lapse * (DjMi + DiMj)

        np.testing.assert_allclose(rhs1 - rhs0, expected_difference, atol=4.0e-7)

    def test_trace_k_rhs_includes_matter_source_when_shift_is_zero(self):
        params = BSSNParameters(dx=self.dx, dt=0.01, nu=0.0, kappa=0.0)
        rho = 0.02 + 0.005 * jnp.cos(self.X)
        stress_trace = 0.03 + 0.004 * jnp.sin(self.Y)
        vars_base = BSSNVariables(
            conformal_metric=jnp.eye(3)[:, :, None, None, None] * jnp.ones((3, 3) + self.shape),
            conformal_factor=jnp.ones(self.shape),
            traceless_K=jnp.zeros((3, 3) + self.shape),
            trace_K=jnp.zeros(self.shape),
            conformal_connection=jnp.zeros((3,) + self.shape),
            lapse=1.0 + 0.01 * jnp.cos(self.Z),
            shift=jnp.zeros((3,) + self.shape),
            rho=jnp.zeros(self.shape),
            S_ij=jnp.zeros((3, 3) + self.shape),
            momentum_density=jnp.zeros((3,) + self.shape),
        )

        stress_tensor = jnp.zeros((3, 3) + self.shape)
        stress_tensor = stress_tensor.at[0, 0].set(stress_trace)

        vars_with_matter = vars_base._replace(rho=rho, S_ij=stress_tensor)

        rhs_base = evolve_trace_extrinsic_curvature(vars_base, params)
        rhs_matter = evolve_trace_extrinsic_curvature(vars_with_matter, params)

        expected_difference = 4.0 * jnp.pi * vars_base.lapse * (rho + stress_trace)
        np.testing.assert_allclose(rhs_matter - rhs_base, expected_difference, atol=1.0e-12)

    def test_rk4_projects_pure_trace_A_before_first_rhs(self):
        gamma = jnp.eye(3)[:, :, None, None, None] * jnp.ones(
            (3, 3) + self.shape
        )
        pure_trace_A = 0.02 * jnp.sin(self.X)
        A_ij = gamma * pure_trace_A
        rho, stress_tensor, momentum_density = vacuum_matter_fields(
            self.shape, gamma.dtype
        )

        vars = BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=jnp.ones(self.shape),
            traceless_K=A_ij,
            trace_K=jnp.zeros(self.shape),
            conformal_connection=jnp.zeros((3,) + self.shape),
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
            rho=rho,
            S_ij=stress_tensor,
            momentum_density=momentum_density,
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
        rho, stress_tensor, momentum_density = vacuum_matter_fields(
            self.shape, gamma.dtype
        )

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
            rho=rho,
            S_ij=stress_tensor,
            momentum_density=momentum_density,
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
        rho, stress_tensor, momentum_density = vacuum_matter_fields(
            self.shape, gamma.dtype
        )

        vars = BSSNVariables(
            conformal_metric=gamma,
            conformal_factor=W,
            traceless_K=jnp.zeros((3, 3) + self.shape),
            trace_K=jnp.zeros(self.shape),
            conformal_connection=jnp.zeros((3,) + self.shape),
            lapse=jnp.ones(self.shape),
            shift=jnp.zeros((3,) + self.shape),
            rho=rho,
            S_ij=stress_tensor,
            momentum_density=momentum_density,
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
