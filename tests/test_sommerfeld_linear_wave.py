"""Standalone 3D outgoing spherical-wave validation for Sommerfeld boundaries."""

import unittest

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC, sommerfeld
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.evolution.derivatives import diff1_field, diff6_field


def _boundary_codes(params, direction):
    if direction == 0:
        return params.xl_bc, params.xr_bc
    elif direction == 1:
        return params.yl_bc, params.yr_bc
    else:
        return params.zl_bc, params.zr_bc


def _first_derivative(field, direction, params):
    return diff1_field(
        field,
        direction,
        params.dx,
        *_boundary_codes(params, direction),
    )


def _ko_dissipation(field, params):
    sixth_derivatives = sum(
        diff6_field(
            field,
            direction,
            params.dx,
            *_boundary_codes(params, direction),
        )
        for direction in range(3)
    )
    return params.nu * params.dx**5 * sixth_derivatives / 64.0


@jax.jit
def linear_wave_rhs(state, params):
    """Return ``u_t = Pi`` and ``Pi_t = Laplacian(u)`` with Sommerfeld faces."""

    u, Pi = state
    laplacian_u = sum(
        _first_derivative(
            _first_derivative(u, direction, params), direction, params
        )
        for direction in range(3)
    )

    rhs_u = Pi + _ko_dissipation(u, params)
    rhs_Pi = laplacian_u + _ko_dissipation(Pi, params)

    rhs_u = sommerfeld(u, rhs_u, jnp.asarray(0.0, dtype=u.dtype), params)
    rhs_Pi = sommerfeld(
        Pi, rhs_Pi, jnp.asarray(0.0, dtype=Pi.dtype), params
    )

    return rhs_u, rhs_Pi


@jax.jit
def linear_wave_rk4_step(state, params):
    """Advance the standalone scalar wave by one classical RK4 step."""

    dt = params.dt
    k1 = linear_wave_rhs(state, params)
    stage = tuple(field + 0.5 * dt * rhs for field, rhs in zip(state, k1))

    k2 = linear_wave_rhs(stage, params)
    stage = tuple(field + 0.5 * dt * rhs for field, rhs in zip(state, k2))

    k3 = linear_wave_rhs(stage, params)
    stage = tuple(field + dt * rhs for field, rhs in zip(state, k3))

    k4 = linear_wave_rhs(stage, params)

    return tuple(
        field + dt * (rhs1 + 2.0 * rhs2 + 2.0 * rhs3 + rhs4) / 6.0
        for field, rhs1, rhs2, rhs3, rhs4 in zip(state, k1, k2, k3, k4)
    )


def run_outgoing_spherical_wave(grid_size, nu=0.0):
    """Evolve a Gaussian shell and measure the wave reflected into the interior."""

    half_width = 3.0
    pulse_center = 1.0
    pulse_width = 0.25
    final_time = 4.5
    reflection_start = 3.5
    diagnostic_radius = 1.4

    dx = 2.0 * half_width / (grid_size - 1)
    dt = 0.2 * dx
    coordinate = -half_width + dx * jnp.arange(grid_size, dtype=jnp.float64)
    X, Y, Z = jnp.meshgrid(
        coordinate, coordinate, coordinate, indexing="ij"
    )
    r = jnp.sqrt(X**2 + Y**2 + Z**2)

    profile = jnp.exp(-((r - pulse_center) / pulse_width) ** 2)
    u = profile / r
    Pi = 2.0 * (r - pulse_center) * profile / (pulse_width**2 * r)

    params = BSSNParameters(
        dx=dx,
        dt=dt,
        nu=nu,
        xl_bc=SOMMERFELD_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=SOMMERFELD_BC,
        yr_bc=SOMMERFELD_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-half_width,
        y_min=-half_width,
        z_min=-half_width,
    )

    state = (u, Pi)
    interior = r < diagnostic_radius
    midpoint = grid_size // 2
    probes = {
        "face": (grid_size - 1, midpoint, midpoint),
        "edge": (grid_size - 1, grid_size - 1, midpoint),
        "corner": (grid_size - 1, grid_size - 1, grid_size - 1),
    }
    boundary_peak = {name: 0.0 for name in probes}
    boundary_peak_time = {name: 0.0 for name in probes}
    reflected_rms = []
    reflected_max = []

    time = 0.0
    while time < final_time - 0.5 * dt:
        for name, index in probes.items():
            scaled_amplitude = float(jnp.abs(r[index] * state[0][index]))
            if scaled_amplitude > boundary_peak[name]:
                boundary_peak[name] = scaled_amplitude
                boundary_peak_time[name] = time

        state = linear_wave_rk4_step(state, params)
        time += dt

        if time >= reflection_start:
            reflected_rms.append(
                float(jnp.sqrt(jnp.mean(state[0][interior] ** 2)))
            )
            reflected_max.append(float(jnp.max(jnp.abs(state[0][interior]))))

    jax.block_until_ready(state)

    return {
        "grid_size": grid_size,
        "dx": dx,
        "dt": dt,
        "nu": nu,
        "final_time": time,
        "reflection_start": reflection_start,
        "reflected_rms": max(reflected_rms),
        "reflected_max": max(reflected_max),
        "boundary_peak": boundary_peak,
        "boundary_peak_time": boundary_peak_time,
        "finite": bool(jnp.all(jnp.isfinite(state[0])))
        and bool(jnp.all(jnp.isfinite(state[1]))),
    }


class TestSommerfeldLinearWave(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.no_dissipation = [
            run_outgoing_spherical_wave(grid_size, nu=0.0)
            for grid_size in (20, 28, 36)
        ]
        cls.with_dissipation = run_outgoing_spherical_wave(36, nu=0.1)

    def test_reflection_decreases_under_refinement_without_ko_dissipation(self):
        errors = np.asarray(
            [result["reflected_rms"] for result in self.no_dissipation]
        )
        spacings = np.asarray([result["dx"] for result in self.no_dissipation])
        orders = np.log(errors[:-1] / errors[1:]) / np.log(
            spacings[:-1] / spacings[1:]
        )

        self.assertTrue(all(result["finite"] for result in self.no_dissipation))
        self.assertTrue(np.all(errors[1:] < errors[:-1]), errors)
        self.assertTrue(np.all(orders > 2.0), orders)

    def test_wave_crosses_faces_edges_and_corners_without_face_ordering_jump(self):
        peaks = self.no_dissipation[-1]["boundary_peak"]
        peak_values = np.asarray([peaks["face"], peaks["edge"], peaks["corner"]])

        self.assertTrue(np.all(peak_values > 0.75), peak_values)
        self.assertTrue(np.all(peak_values < 1.05), peak_values)
        self.assertLess(float(np.max(peak_values) - np.min(peak_values)), 0.15)

    def test_lopsided_ko_dissipation_remains_finite_and_reduces_reflection(self):
        without_ko = self.no_dissipation[-1]
        with_ko = self.with_dissipation

        self.assertTrue(with_ko["finite"])
        self.assertEqual(with_ko["nu"], 0.1)
        self.assertLess(with_ko["reflected_rms"], without_ko["reflected_rms"])


if __name__ == "__main__":
    unittest.main()
