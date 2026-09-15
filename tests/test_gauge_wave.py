import unittest

import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.time_evolve import rk4_step
from tests.initial_data import (
    gauge_wave_analytic_state,
    periodic_coordinate_arrays,
    periodic_gauge_wave_state,
)


class TestGaugeWave(unittest.TestCase):

    def test_gauge_wave_evolution(self):
        domain_length = 1.0
        grid_size = 60
        X, Y, Z = periodic_coordinate_arrays(grid_size, domain_length)
        dx = domain_length / grid_size
        dt = dx

        variables = periodic_gauge_wave_state(
            X, Y, Z, dx, wavelength=domain_length
        )
        parameters = BSSNParameters(
            eta=0.0,
            kappa=0.0,
            g=0.0,
            dx=dx,
            dt=dt,
        )

        for _ in range(50):
            variables = rk4_step(variables, parameters)

        exact = gauge_wave_analytic_state(
            X, Y, Z, 50 * dt, wavelength=domain_length
        )
        relative_errors = (
            jnp.sqrt(jnp.mean(
                ((variables.conformal_metric[0, 0]
                  - exact.conformal_metric[0, 0])
                 / exact.conformal_metric[0, 0]) ** 2
            )),
            jnp.sqrt(jnp.mean(
                ((variables.conformal_metric[1, 1]
                  - exact.conformal_metric[1, 1])
                 / exact.conformal_metric[1, 1]) ** 2
            )),
            jnp.sqrt(jnp.mean(
                ((variables.conformal_metric[2, 2]
                  - exact.conformal_metric[2, 2])
                 / exact.conformal_metric[2, 2]) ** 2
            )),
            jnp.sqrt(jnp.mean(
                ((variables.conformal_factor - exact.conformal_factor)
                 / exact.conformal_factor) ** 2
            )),
        )

        for error, tolerance in zip(
            relative_errors, (5e-5, 5e-5, 5e-5, 1e-4)
        ):
            self.assertLess(error, tolerance)

    def test_convergence(self):
        def run_simulation(grid_size):
            domain_length = 1.0
            X, Y, Z = periodic_coordinate_arrays(
                grid_size, domain_length
            )
            dx = domain_length / grid_size
            variables = periodic_gauge_wave_state(
                X, Y, Z, dx, wavelength=domain_length
            )
            parameters = BSSNParameters(
                eta=0.0,
                kappa=0.0,
                g=0.0,
                dx=dx,
                dt=dx,
                nu=0.0,
                zero_shift=1,
                gauge=0,
            )

            final_time = 1.0
            for _ in range(round(final_time / dx)):
                variables = rk4_step(variables, parameters)

            exact = gauge_wave_analytic_state(
                X, Y, Z, final_time, wavelength=domain_length
            )
            return (
                jnp.sqrt(jnp.mean(
                    (variables.conformal_metric[0, 0]
                     - exact.conformal_metric[0, 0]) ** 2
                )),
                jnp.sqrt(jnp.mean(
                    (variables.conformal_metric[1, 1]
                     - exact.conformal_metric[1, 1]) ** 2
                )),
                jnp.sqrt(jnp.mean(
                    (variables.conformal_metric[2, 2]
                     - exact.conformal_metric[2, 2]) ** 2
                )),
                jnp.sqrt(jnp.mean(
                    (variables.conformal_factor
                     - exact.conformal_factor) ** 2
                )),
            )

        coarse_errors = run_simulation(16)
        fine_errors = run_simulation(32)
        orders = jnp.log2(
            jnp.asarray(coarse_errors) / jnp.asarray(fine_errors)
        )
        field_names = (
            "conformal metric 00",
            "conformal metric 11",
            "conformal metric 22",
            "conformal factor",
        )

        for field_name, coarse_error, fine_error, order in zip(
            field_names, coarse_errors, fine_errors, orders
        ):
            with self.subTest(field=field_name):
                self.assertGreater(
                    order,
                    3.8,
                    msg=(
                        f"{field_name} convergence order {float(order):.6f}; "
                        f"coarse error {float(coarse_error):.6e}, "
                        f"fine error {float(fine_error):.6e}"
                    ),
                )
