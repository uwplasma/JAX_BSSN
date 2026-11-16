"""Analytic gauge-wave spatial convergence and RK4 temporal refinement."""

import unittest

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn.variables import BSSNParameters
from JAX_BSSN.parallel import make_field_mesh, initialize_tiled_bssn, tiled_rk4_step
from tests.initial_data import periodic_gauge_wave_state, gauge_wave_analytic_state, linear_wave_data
from tests.test_parallel import gather_small_field


@unittest.skipUnless(jax.device_count() >= 2, "requires two logical devices")
class TestTiledGaugeWaveConvergence(unittest.TestCase):
    def evolve_wave(self, n, dt, steps):
        x = (jnp.arange(n)/n-.5)[:, None, None]
        zero = jnp.zeros_like(x)
        state = periodic_gauge_wave_state(x, zero, zero, 1/n)
        params = BSSNParameters(dx=1/n, dt=dt, eta=0., g=0., nu=0., zero_shift=1)
        layout = dict(mesh=make_field_mesh((2, 1, 1)), tile_shape=(n//2, 1, 1))
        tiled = initialize_tiled_bssn(state, params, **layout)
        advance = jax.jit(lambda q: tiled_rk4_step(q, params, **layout))
        for _ in range(steps):
            tiled = advance(tiled)
        result = gather_small_field(tiled.conformal_metric, layout["tile_shape"])
        exact = gauge_wave_analytic_state(x, zero, zero, dt*steps)
        return result, exact.conformal_metric

    def test_fourth_order_spatial_convergence_against_analytic_wave(self):
        errors = []
        for n in (16, 32):
            actual, exact = self.evolve_wave(n, 1/n, n)
            errors.append(jnp.sqrt(jnp.mean((actual-exact)**2)))
        order = jnp.log2(errors[0]/errors[1])
        self.assertGreater(float(order), 3.8,
                           f"spatial errors {errors}; observed order {float(order)}")

    def test_fourth_order_temporal_refinement_on_the_same_spatial_grid(self):
        results = [self.evolve_wave(32, .16/steps, steps)[0] for steps in (8, 16, 32)]
        coarse = jnp.sqrt(jnp.mean((results[0]-results[1])**2))
        fine = jnp.sqrt(jnp.mean((results[1]-results[2])**2))
        order = jnp.log2(coarse/fine)
        self.assertGreater(float(order), 3.6,
                           f"temporal differences {coarse}, {fine}; order {float(order)}")

    def test_linear_gravitational_wave_spatial_convergence(self):
        # Keep nonlinear O(amplitude**2) corrections below truncation error.
        amplitude, final_time, dt = 1.e-7, .2, .002
        errors = []
        for n in (16, 32):
            dx = 1/n
            origin = -(n-1)*dx/2
            state = linear_wave_data(n, 1, 1, dx, amplitude=amplitude)
            params = BSSNParameters(dx=dx, dt=dt, eta=0., kappa=0., nu=0., g=0.,
                                     zero_shift=1, gauge=0, x_min=origin)
            layout = dict(mesh=make_field_mesh((2, 1, 1)), tile_shape=(n//2, 1, 1))
            tiled = initialize_tiled_bssn(state, params, **layout)
            advance = jax.jit(lambda q: tiled_rk4_step(q, params, **layout))
            for _ in range(round(final_time/dt)):
                tiled = advance(tiled)
            conformal = gather_small_field(tiled.conformal_metric, layout["tile_shape"])
            W = gather_small_field(tiled.conformal_factor, layout["tile_shape"])
            h_plus = .5*(conformal[1, 1]-conformal[2, 2])/W**2
            x = (origin+jnp.arange(n)*dx)[:, None, None]
            exact = amplitude*jnp.sin(2*jnp.pi*(x-final_time))
            errors.append(jnp.sqrt(jnp.mean(((h_plus-exact)/amplitude)**2)))
        order = jnp.log2(errors[0]/errors[1])
        self.assertGreater(float(order), 3.8,
                           f"linear wave errors {errors}; observed order {float(order)}")


if __name__ == "__main__":
    unittest.main()
