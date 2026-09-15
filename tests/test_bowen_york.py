import shutil

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from demos.axisymmetric_bowen_york.axisymmetric_bowen_york import (
    axisymmetric_parameters,
    bowen_york_cartesian_grid,
    bowen_york_plane_data,
)
from demos.axisymmetric_bowen_york.make_movies import (
    _available_specs,
    render_movie,
    track_puncture,
    write_puncture_track,
)
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_rk4_step,
    compact_axisymmetric_state,
    validate_axisymmetric_grid,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.utilities.bowen_york_solver import (
    bowen_york_extrinsic_curvature,
    cell_centered_grid,
    solve_conformal_factor,
)


def test_bowen_york_curvature_is_symmetric_traceless_and_zero_at_zero_momentum():
    grid, _ = cell_centered_grid(num_points=8, half_width=4.0)
    positions = jnp.zeros((1, 3), dtype=grid.dtype)
    momentum = jnp.asarray([[0.2, -0.1, 0.5]], dtype=grid.dtype)

    curvature = bowen_york_extrinsic_curvature(momentum, positions, grid)

    np.testing.assert_allclose(curvature, jnp.swapaxes(curvature, -1, -2))
    np.testing.assert_allclose(
        jnp.trace(curvature, axis1=-2, axis2=-1),
        0.0,
        atol=2.0e-14,
    )
    zero_curvature = bowen_york_extrinsic_curvature(
        jnp.zeros_like(momentum),
        positions,
        grid,
    )
    np.testing.assert_array_equal(zero_curvature, jnp.zeros_like(zero_curvature))


def test_boosted_conformal_factor_converges_with_zero_boundary_correction():
    grid, dx = cell_centered_grid(num_points=10, half_width=5.0)
    psi, u, residual_history = solve_conformal_factor(
        masses=jnp.asarray([1.0]),
        positions=jnp.zeros((1, 3)),
        momenta=jnp.asarray([[0.0, 0.0, 0.2]]),
        grid=grid,
        dx=dx,
        newton_tolerance=1.0e-9,
        cg_tolerance=1.0e-11,
    )

    assert bool(jnp.all(jnp.isfinite(psi)))
    assert float(jnp.min(psi)) > 0.0
    assert residual_history[-1] <= 1.0e-9
    assert residual_history[-1] < residual_history[0]
    for face in (u[0], u[-1], u[:, 0], u[:, -1], u[:, :, 0], u[:, :, -1]):
        np.testing.assert_array_equal(face, jnp.zeros_like(face))


def test_plane_conversion_uses_exact_y_zero_and_bssn_psi_scaling():
    num_rho, num_z, dx = 6, 12, 0.5
    plane_vars, u, residual_history = bowen_york_plane_data(
        mass=1.0,
        momentum_z=0.2,
        num_radial_points=num_rho,
        num_z_points=num_z,
        dx=dx,
        newton_tolerance=1.0e-9,
        cg_tolerance=1.0e-11,
    )
    grid, center_y = bowen_york_cartesian_grid(num_rho, num_z, dx)

    np.testing.assert_array_equal(grid[:, center_y, :, 1], 0.0)
    assert not bool(jnp.any(jnp.all(grid == 0.0, axis=-1)))
    assert u.shape == (2 * num_rho, 2 * num_rho + 1, num_z)
    assert residual_history[-1] <= 1.0e-9

    momenta = jnp.asarray([[0.0, 0.0, 0.2]])
    positions = jnp.zeros((1, 3))
    curvature = bowen_york_extrinsic_curvature(
        momenta,
        positions,
        grid[:, center_y : center_y + 1, :, :],
    )
    curvature = jnp.moveaxis(curvature, (-2, -1), (0, 1))
    expected_traceless_K = plane_vars.conformal_factor[None, None, ...] ** 3 * curvature
    np.testing.assert_allclose(
        plane_vars.traceless_K,
        expected_traceless_K,
        rtol=2.0e-14,
        atol=2.0e-14,
    )


def test_reduced_boosted_plane_compacts_and_takes_one_finite_rk4_step():
    num_rho, num_z, dx = 6, 12, 0.5
    plane_vars, _, _ = bowen_york_plane_data(
        mass=1.0,
        momentum_z=0.1,
        num_radial_points=num_rho,
        num_z_points=num_z,
        dx=dx,
        newton_tolerance=1.0e-8,
        cg_tolerance=1.0e-10,
    )
    vars = compact_axisymmetric_state(plane_vars)
    params = axisymmetric_parameters(num_rho, num_z, 3.0, 3.0, dt=0.02)

    validate_axisymmetric_grid(vars, params)
    assert vars.conformal_factor.shape == (num_rho + 4, 1, num_z)
    advanced = axisymmetric_rk4_step(vars, params)
    jax.block_until_ready(advanced)

    assert all(bool(jnp.all(jnp.isfinite(field))) for field in advanced)