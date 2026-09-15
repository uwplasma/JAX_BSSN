"""Tests for radial Sommerfeld RHS boundary conditions."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.evolution.boundaries import (
    PERIODIC_BC,
    SOMMERFELD_BC,
    apply_sommerfeld_boundaries,
    radial_derivative,
    sommerfeld,
    sommerfeld_boundary_mask,
)
from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs


def flat_bssn_variables(shape, dtype=jnp.float64):
    metric = jnp.eye(3, dtype=dtype)[:, :, None, None, None]
    metric = jnp.broadcast_to(metric, (3, 3) + shape)
    scalar_zero = jnp.zeros(shape, dtype=dtype)
    vector_zero = jnp.zeros((3,) + shape, dtype=dtype)

    return BSSNVariables(
        conformal_metric=metric,
        conformal_factor=jnp.ones(shape, dtype=dtype),
        traceless_K=jnp.zeros_like(metric),
        trace_K=scalar_zero,
        conformal_connection=vector_zero,
        lapse=jnp.ones(shape, dtype=dtype),
        shift=vector_zero,
    )


class TestSommerfeldBoundaries(unittest.TestCase):
    def setUp(self):
        self.shape = (11, 9, 13)
        self.dx = 0.25
        self.params = BSSNParameters(
            dx=self.dx,
            xl_bc=SOMMERFELD_BC,
            xr_bc=SOMMERFELD_BC,
            yl_bc=SOMMERFELD_BC,
            yr_bc=SOMMERFELD_BC,
            zl_bc=SOMMERFELD_BC,
            zr_bc=SOMMERFELD_BC,
            x_min=-5 * self.dx,
            y_min=-4 * self.dx,
            z_min=-6 * self.dx,
        )

        x = self.params.x_min + self.dx * jnp.arange(self.shape[0])
        y = self.params.y_min + self.dx * jnp.arange(self.shape[1])
        z = self.params.z_min + self.dx * jnp.arange(self.shape[2])
        self.X = x[:, None, None]
        self.Y = y[None, :, None]
        self.Z = z[None, None, :]
        self.r = jnp.sqrt(self.X**2 + self.Y**2 + self.Z**2)

    def test_boundary_code_map(self):
        self.assertEqual(PERIODIC_BC, 0)
        self.assertEqual(SOMMERFELD_BC, 1)

    def test_radial_derivative_of_r_squared_at_faces_edges_and_corners(self):
        field = self.X**2 + self.Y**2 + self.Z**2
        expected = 2.0 * self.r

        scalar_derivative = radial_derivative(field, self.params)
        vector_derivative = radial_derivative(
            jnp.stack((field, 2.0 * field, -field), axis=0), self.params
        )
        tensor_derivative = radial_derivative(
            jnp.broadcast_to(field, (3, 3) + self.shape), self.params
        )

        np.testing.assert_allclose(
            scalar_derivative, expected, rtol=0.0, atol=2.0e-14
        )
        np.testing.assert_allclose(
            vector_derivative,
            jnp.stack((expected, 2.0 * expected, -expected), axis=0),
            rtol=0.0,
            atol=4.0e-14,
        )
        np.testing.assert_allclose(
            tensor_derivative,
            jnp.broadcast_to(expected, (3, 3) + self.shape),
            rtol=0.0,
            atol=2.0e-14,
        )

        probes = (
            (0, self.shape[1] // 2, self.shape[2] // 2),
            (0, 0, self.shape[2] // 2),
            (0, 0, 0),
            (-1, -1, -1),
        )
        for probe in probes:
            self.assertAlmostEqual(
                float(scalar_derivative[probe]), float(expected[probe]), places=13
            )

    def test_boundary_mask_is_union_of_active_faces(self):
        params = self.params._replace(
            xl_bc=SOMMERFELD_BC,
            xr_bc=PERIODIC_BC,
            yl_bc=PERIODIC_BC,
            yr_bc=SOMMERFELD_BC,
            zl_bc=PERIODIC_BC,
            zr_bc=PERIODIC_BC,
        )
        mask = np.asarray(sommerfeld_boundary_mask(self.shape, params))

        expected = np.zeros(self.shape, dtype=bool)
        expected[0, :, :] = True
        expected[:, -1, :] = True
        np.testing.assert_array_equal(mask, expected)

    def test_sommerfeld_replaces_rhs_once_on_faces_only(self):
        field = self.X**2 + self.Y**2 + self.Z**2
        rhs = 7.0 * jnp.ones(self.shape, dtype=field.dtype)
        result = sommerfeld(field, rhs, jnp.asarray(0.0), self.params)
        mask = sommerfeld_boundary_mask(self.shape, self.params)
        expected_boundary_rhs = -3.0 * self.r

        np.testing.assert_allclose(
            result[mask], expected_boundary_rhs[mask], rtol=0.0, atol=3.0e-14
        )
        np.testing.assert_array_equal(result[~mask], rhs[~mask])
        self.assertAlmostEqual(
            float(result[0, 0, 0]), float(expected_boundary_rhs[0, 0, 0]), places=13
        )

    def test_flat_space_is_stationary_on_sommerfeld_faces(self):
        vars = flat_bssn_variables(self.shape)
        rhs = BSSNVariables(*(jnp.full_like(field, 3.0) for field in vars))
        result = apply_sommerfeld_boundaries(vars, rhs, self.params)
        mask = np.asarray(sommerfeld_boundary_mask(self.shape, self.params))

        for result_field, rhs_field in zip(result, rhs):
            leading = result_field.ndim - 3
            field_mask = mask.reshape((1,) * leading + self.shape)
            field_mask = np.broadcast_to(field_mask, result_field.shape)
            np.testing.assert_allclose(
                np.asarray(result_field)[field_mask], 0.0, rtol=0.0, atol=5.0e-15
            )
            np.testing.assert_array_equal(
                np.asarray(result_field)[~field_mask], np.asarray(rhs_field)[~field_mask]
            )

    def test_periodic_code_does_not_replace_rhs(self):
        vars = flat_bssn_variables(self.shape)
        rhs = BSSNVariables(*(jnp.full_like(field, 3.0) for field in vars))

        params = self.params._replace(
            xl_bc=PERIODIC_BC,
            xr_bc=PERIODIC_BC,
            yl_bc=PERIODIC_BC,
            yr_bc=PERIODIC_BC,
            zl_bc=PERIODIC_BC,
            zr_bc=PERIODIC_BC,
        )
        result = apply_sommerfeld_boundaries(vars, rhs, params)
        for result_field, rhs_field in zip(result, rhs):
            np.testing.assert_array_equal(result_field, rhs_field)

    def test_assembled_bssn_rhs_keeps_flat_space_stationary(self):
        vars = flat_bssn_variables(self.shape)
        rhs = compute_bssn_rhs(vars, self.params._replace(nu=0.0))

        for field in rhs:
            self.assertTrue(bool(jnp.all(jnp.isfinite(field))))
            self.assertLess(float(jnp.max(jnp.abs(field))), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
