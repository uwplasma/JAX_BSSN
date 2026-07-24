import unittest

import jax.numpy as jnp

from JAX_BSSN.initialization import create_coordinate_arrays, puncture_black_hole_data


class TestPunctureInitialization(unittest.TestCase):
    def test_puncture_initial_data_matches_isotropic_profile(self):
        nx = 8
        dx = 0.5
        mass = 1.7

        vars = puncture_black_hole_data(nx, nx, nx, dx, mass=mass)

        X, Y, Z = create_coordinate_arrays(nx, nx, nx, dx)
        r = jnp.sqrt(X**2 + Y**2 + Z**2)
        psi = 1.0 + mass / (2.0 * r)
        expected_W = psi ** (-2.0)

        self.assertTrue(jnp.allclose(vars.conformal_factor, expected_W, atol=1e-12))
        self.assertTrue(jnp.allclose(vars.lapse, expected_W, atol=1e-12))
        self.assertTrue(jnp.allclose(vars.shift, 0.0, atol=1e-12))
        self.assertTrue(jnp.allclose(vars.traceless_K, 0.0, atol=1e-12))
        self.assertTrue(jnp.allclose(vars.trace_K, 0.0, atol=1e-12))
        self.assertTrue(jnp.allclose(vars.conformal_connection, 0.0, atol=1e-12))

        expected_metric = (
            jnp.eye(3)[:, :, None, None, None] * jnp.ones_like(vars.conformal_metric)
        )
        self.assertTrue(jnp.allclose(vars.conformal_metric, expected_metric, atol=1e-12))


if __name__ == "__main__":
    unittest.main()