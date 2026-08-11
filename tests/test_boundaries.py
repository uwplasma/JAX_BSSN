import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.boundaries import (
    PERIODIC_BC,
    SUPERGAUSSIAN_BC,
    apply_supergaussian_boundaries,
)
from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.tensor_algebra import (
    determinant_3x3_metric,
    invert_3x3_metric,
    trace_tensor,
)
from tests.helpers import vacuum_matter_fields


class TestSuperGaussianBoundaries(unittest.TestCase):
    def setUp(self):
        self.shape = (12, 10, 8)
        ni, nj, nk = self.shape
        x = jnp.arange(ni)[:, None, None]
        y = jnp.arange(nj)[None, :, None]
        z = jnp.arange(nk)[None, None, :]
        perturbation = 0.1 + 0.01 * x + 0.02 * y + 0.03 * z

        conformal_metric = jnp.zeros((3, 3) + self.shape)
        conformal_metric = conformal_metric.at[0, 0].set(1.0 + perturbation)
        conformal_metric = conformal_metric.at[1, 1].set(1.0 + 2.0 * perturbation)
        conformal_metric = conformal_metric.at[2, 2].set(1.0 + 3.0 * perturbation)
        conformal_metric = conformal_metric.at[0, 1].set(0.05 + perturbation)
        conformal_metric = conformal_metric.at[1, 0].set(0.05 + perturbation)

        traceless_K = jnp.zeros((3, 3) + self.shape)
        traceless_K = traceless_K.at[0, 0].set(0.03 + perturbation)
        traceless_K = traceless_K.at[1, 1].set(-0.02 - perturbation)
        traceless_K = traceless_K.at[2, 2].set(-0.01)
        rho, stress_tensor, momentum_density = vacuum_matter_fields(
            self.shape, conformal_metric.dtype
        )

        self.vars = BSSNVariables(
            conformal_metric=conformal_metric,
            conformal_factor=1.0 + perturbation,
            traceless_K=traceless_K,
            trace_K=0.02 + perturbation,
            conformal_connection=jnp.stack(
                [0.01 + perturbation, 0.02 + perturbation, 0.03 + perturbation],
                axis=0,
            ),
            lapse=1.0 + 0.5 * perturbation,
            shift=jnp.stack(
                [0.04 + perturbation, 0.05 + perturbation, 0.06 + perturbation],
                axis=0,
            ),
            rho=rho,
            S_ij=stress_tensor,
            momentum_density=momentum_density,
        )

    def test_default_boundary_codes_leave_state_unchanged(self):
        params = BSSNParameters(bc_width=3.0, bc_order=4.0, bc_strength=1.0)

        filtered = apply_supergaussian_boundaries(self.vars, params)

        for original, actual in zip(self.vars, filtered):
            np.testing.assert_allclose(actual, original, atol=0.0)

    def test_x_left_boundary_decays_all_variables_to_flat_space(self):
        params = BSSNParameters(
            xl_bc=SUPERGAUSSIAN_BC,
            xr_bc=PERIODIC_BC,
            bc_width=3.0,
            bc_order=4.0,
            bc_strength=1.0,
        )

        filtered = apply_supergaussian_boundaries(self.vars, params)

        np.testing.assert_allclose(filtered.conformal_metric[0, 0, 0], 1.0)
        np.testing.assert_allclose(filtered.conformal_metric[1, 1, 0], 1.0)
        np.testing.assert_allclose(filtered.conformal_metric[2, 2, 0], 1.0)
        np.testing.assert_allclose(filtered.conformal_metric[0, 1, 0], 0.0)
        np.testing.assert_allclose(filtered.conformal_factor[0], 1.0)
        np.testing.assert_allclose(filtered.traceless_K[:, :, 0], 0.0)
        np.testing.assert_allclose(filtered.trace_K[0], 0.0)
        np.testing.assert_allclose(filtered.conformal_connection[:, 0], 0.0)
        np.testing.assert_allclose(filtered.lapse[0], 1.0)
        np.testing.assert_allclose(filtered.shift[:, 0], 0.0)

        np.testing.assert_allclose(filtered.conformal_factor[4:], self.vars.conformal_factor[4:])
        np.testing.assert_allclose(filtered.conformal_metric[:, :, 4:], self.vars.conformal_metric[:, :, 4:])

    def test_right_side_code_only_damps_the_right_side(self):
        params = BSSNParameters(
            xr_bc=SUPERGAUSSIAN_BC,
            bc_width=2.0,
            bc_order=4.0,
            bc_strength=1.0,
        )

        filtered = apply_supergaussian_boundaries(self.vars, params)

        np.testing.assert_allclose(filtered.conformal_factor[-1], 1.0)
        np.testing.assert_allclose(filtered.conformal_factor[0], self.vars.conformal_factor[0])

    def test_rk4_enforces_supergaussian_boundaries_and_trace_free_A(self):
        params = BSSNParameters(
            xl_bc=SUPERGAUSSIAN_BC,
            bc_width=3.0,
            bc_order=4.0,
            bc_strength=1.0,
            dx=0.1,
            dt=0.0001,
            nu=0.0,
            kappa=0.0,
            eta=0.0,
            g=0.0,
        )

        evolved = rk4_step(self.vars, params)

        np.testing.assert_allclose(evolved.conformal_metric[0, 0, 0], 1.0)
        np.testing.assert_allclose(evolved.conformal_metric[0, 1, 0], 0.0)
        np.testing.assert_allclose(evolved.conformal_factor[0], 1.0)
        np.testing.assert_allclose(evolved.lapse[0], 1.0)
        np.testing.assert_allclose(evolved.shift[:, 0], 0.0)

        trace_A = trace_tensor(
            evolved.traceless_K, invert_3x3_metric(evolved.conformal_metric)
        )
        np.testing.assert_allclose(trace_A, jnp.zeros_like(trace_A), atol=1.0e-10)

        det_gamma = determinant_3x3_metric(evolved.conformal_metric)
        np.testing.assert_allclose(det_gamma, jnp.ones_like(det_gamma), atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
