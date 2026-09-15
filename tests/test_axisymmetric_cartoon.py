import jax
import jax.numpy as jnp
import numpy as np
import pytest

from JAX_BSSN.bssn.constraints import ConstraintViolations
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_rk4_step,
    compact_axisymmetric_state,
    compute_axisymmetric_constraint_norms,
    compute_axisymmetric_constraints,
    compute_axisymmetric_rhs,
    expand_axisymmetric_plane,
    fill_axisymmetric_ghosts,
    reconstruct_axisymmetric_support,
    validate_axisymmetric_grid,
)
from JAX_BSSN.cartoon.axisymmetry.reconstruction import (
    AXISYMMETRIC_CENTER,
    AXISYMMETRIC_GHOST_CELLS,
)
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.evolution.derivatives import diff2_field
from JAX_BSSN.evolution.time_evolve import compute_bssn_rhs


jax.config.update("jax_enable_x64", True)


def _params(nrho, nz, dx, z_min=None):
    del nrho
    if z_min is None:
        z_min = -(nz - 1) * dx / 2.0
    return BSSNParameters(
        dx=dx,
        dt=0.05 * dx,
        nu=0.0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=z_min,
        mad_q=1.0,
    )


def _minkowski(nrho=12, nz=17, dx=0.2):
    shape = (nrho + AXISYMMETRIC_GHOST_CELLS, 1, nz)
    scalar = jnp.zeros(shape, dtype=jnp.float64)
    vector = jnp.zeros((3,) + shape, dtype=jnp.float64)
    identity = jnp.eye(3, dtype=jnp.float64)[:, :, None, None, None]
    metric = identity * jnp.ones((3, 3) + shape, dtype=jnp.float64)
    return BSSNVariables(
        metric,
        jnp.ones(shape),
        jnp.zeros_like(metric),
        scalar,
        vector,
        jnp.ones(shape),
        vector,
    )


def _set_positive(field, values):
    return field.at[..., AXISYMMETRIC_GHOST_CELLS :, 0, :].set(values)


def test_grid_validation_enforces_axisymmetric_contract():
    vars = _minkowski()
    params = _params(12, 17, 0.2)
    validate_axisymmetric_grid(vars, params)

    with pytest.raises(ValueError, match="six radial"):
        validate_axisymmetric_grid(_minkowski(nrho=5), params)
    with pytest.raises(ValueError, match="nine z"):
        validate_axisymmetric_grid(_minkowski(nz=8), _params(12, 8, 0.2))
    with pytest.raises(ValueError, match="x-left"):
        validate_axisymmetric_grid(vars, params._replace(xl_bc=SOMMERFELD_BC))
    with pytest.raises(ValueError, match="x-right"):
        validate_axisymmetric_grid(vars, params._replace(xr_bc=PERIODIC_BC))
    with pytest.raises(ValueError, match="y faces"):
        validate_axisymmetric_grid(vars, params._replace(yl_bc=SOMMERFELD_BC))
    with pytest.raises(ValueError, match="mad_q"):
        validate_axisymmetric_grid(vars, params._replace(mad_q=0.5))
    with pytest.raises(ValueError, match="x_min"):
        validate_axisymmetric_grid(vars, params._replace(x_min=0.0))


def test_all_vector_and_tensor_radial_parities_are_exact():
    nrho, nz = 8, 9
    vars = _minkowski(nrho, nz)
    positive_shape = (nrho, nz)
    vector_values = jnp.arange(3 * nrho * nz, dtype=jnp.float64).reshape(
        3, nrho, nz
    )
    tensor_values = jnp.arange(9 * nrho * nz, dtype=jnp.float64).reshape(
        3, 3, nrho, nz
    )
    vars = vars._replace(
        conformal_connection=_set_positive(vars.conformal_connection, vector_values),
        conformal_metric=_set_positive(vars.conformal_metric, tensor_values),
        conformal_factor=_set_positive(
            vars.conformal_factor, jnp.arange(nrho * nz).reshape(positive_shape)
        ),
    )
    filled = fill_axisymmetric_ghosts(vars)
    p = np.asarray((-1.0, -1.0, 1.0))
    np.testing.assert_array_equal(
        filled.conformal_connection[:, :4, 0, :],
        np.asarray(vector_values[:, :4, :])[:, ::-1, :] * p[:, None, None],
    )
    np.testing.assert_array_equal(
        filled.conformal_metric[:, :, :4, 0, :],
        np.asarray(tensor_values[:, :, :4, :])[:, :, ::-1, :]
        * (p[:, None] * p[None, :])[:, :, None, None],
    )
    np.testing.assert_array_equal(
        filled.conformal_factor[:4, 0, :],
        np.asarray(filled.conformal_factor[4:8, 0, :])[::-1, :],
    )


def test_compact_expand_round_trip_preserves_general_twist_components():
    nrho, nz = 8, 9
    full = expand_axisymmetric_plane(fill_axisymmetric_ghosts(_minkowski(nrho, nz)))
    xz = (2 * nrho, 1, nz)
    vector = jnp.arange(3 * np.prod(xz), dtype=jnp.float64).reshape((3,) + xz)
    tensor = jnp.arange(9 * np.prod(xz), dtype=jnp.float64).reshape((3, 3) + xz)
    # The positive half is authoritative; compaction must regenerate every
    # negative component, including azimuthal and twist entries.
    full = full._replace(shift=vector, traceless_K=tensor)
    compact = compact_axisymmetric_state(full)
    expanded = expand_axisymmetric_plane(compact)
    np.testing.assert_array_equal(expanded.shift[:, nrho:, 0, :], vector[:, nrho:, 0, :])
    np.testing.assert_array_equal(
        expanded.traceless_K[:, :, nrho:, 0, :], tensor[:, :, nrho:, 0, :]
    )


def test_reconstructed_reference_plane_equals_filled_compact_state():
    nrho, nz, dx = 16, 17, 0.15
    vars = _minkowski(nrho, nz, dx)
    rho = (jnp.arange(nrho) + 0.5) * dx
    z = (jnp.arange(nz) - (nz - 1) / 2.0) * dx
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    shift = jnp.stack((R * (1 + 0.1 * Z), R * (0.3 - 0.05 * Z), 0.2 + R**2))
    metric_values = jnp.zeros((3, 3, nrho, nz), dtype=jnp.float64)
    metric_values = metric_values.at[0, 0].set(1.0 + 0.02 * R**2)
    metric_values = metric_values.at[1, 1].set(1.0 - 0.01 * R**2)
    metric_values = metric_values.at[2, 2].set(1.0)
    metric_values = metric_values.at[0, 1].set(0.004 * R**2)
    metric_values = metric_values.at[1, 0].set(0.004 * R**2)
    metric_values = metric_values.at[0, 2].set(0.02 * R)
    metric_values = metric_values.at[2, 0].set(0.02 * R)
    metric_values = metric_values.at[1, 2].set(-0.015 * R)
    metric_values = metric_values.at[2, 1].set(-0.015 * R)
    vars = fill_axisymmetric_ghosts(
        vars._replace(
            shift=_set_positive(vars.shift, shift),
            conformal_metric=_set_positive(vars.conformal_metric, metric_values),
        )
    )
    support = reconstruct_axisymmetric_support(vars, _params(nrho, nz, dx))
    for compact_field, support_field in zip(vars, support):
        np.testing.assert_allclose(
            support_field[..., :, AXISYMMETRIC_CENTER, :],
            compact_field[..., :, 0, :],
            atol=2.0e-13,
            rtol=2.0e-13,
        )
        assert bool(jnp.all(jnp.isfinite(support_field)))


def test_nonzero_azimuthal_vector_and_twist_tensor_rotate_correctly():
    nrho, nz, dx = 20, 17, 0.1
    vars = _minkowski(nrho, nz, dx)
    rho = (jnp.arange(nrho) + 0.5) * dx
    z = (jnp.arange(nz) - (nz - 1) / 2.0) * dx
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    vector_ref = jnp.stack((R * (1.0 + 0.1 * Z), 0.3 * R, 0.2 + 0.05 * R**2))
    tensor_ref = jnp.zeros((3, 3, nrho, nz), dtype=jnp.float64)
    tensor_ref = tensor_ref.at[0, 0].set(1.0 + 0.02 * R**2)
    tensor_ref = tensor_ref.at[1, 1].set(1.0 - 0.01 * R**2)
    tensor_ref = tensor_ref.at[2, 2].set(0.8 + 0.01 * R**2)
    tensor_ref = tensor_ref.at[0, 1].set(0.03 * R**2)
    tensor_ref = tensor_ref.at[1, 0].set(0.03 * R**2)
    tensor_ref = tensor_ref.at[0, 2].set(0.04 * R)
    tensor_ref = tensor_ref.at[2, 0].set(0.04 * R)
    tensor_ref = tensor_ref.at[1, 2].set(-0.02 * R)
    tensor_ref = tensor_ref.at[2, 1].set(-0.02 * R)
    vars = fill_axisymmetric_ghosts(
        vars._replace(
            conformal_connection=_set_positive(vars.conformal_connection, vector_ref),
            traceless_K=_set_positive(vars.traceless_K, tensor_ref),
        )
    )
    support = reconstruct_axisymmetric_support(vars, _params(nrho, nz, dx))

    x = (jnp.arange(nrho + 4) - 3.5) * dx
    y = (jnp.arange(9) - 4.0) * dx
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    radius = jnp.sqrt(X**2 + Y**2)
    cosine, sine = X / radius, Y / radius
    z_grid = z[None, None, :]
    radial = radius[:, :, None]
    reference_vector = jnp.stack(
        (
            radial * (1.0 + 0.1 * z_grid),
            0.3 * radial * jnp.ones_like(z_grid),
            (0.2 + 0.05 * radial**2) * jnp.ones_like(z_grid),
        )
    )
    expected_x = cosine[..., None] * reference_vector[0] - sine[..., None] * reference_vector[1]
    expected_y = sine[..., None] * reference_vector[0] + cosine[..., None] * reference_vector[1]
    interior = (slice(None, -5), slice(None), slice(None))
    np.testing.assert_allclose(
        support.conformal_connection[0][interior], expected_x[interior], atol=3e-12
    )
    np.testing.assert_allclose(
        support.conformal_connection[1][interior], expected_y[interior], atol=3e-12
    )
    assert float(jnp.max(jnp.abs(support.traceless_K[0, 1, :-5]))) > 0.0
    assert float(jnp.max(jnp.abs(support.traceless_K[0, 2, :-5]))) > 0.0
    assert float(jnp.max(jnp.abs(support.traceless_K[1, 2, :-5]))) > 0.0


def test_minkowski_rhs_rk4_constraints_and_jit():
    vars = _minkowski(10, 17, 0.2)
    params = _params(10, 17, 0.2)
    rhs = jax.jit(compute_axisymmetric_rhs)(vars, params)
    for field in rhs:
        np.testing.assert_allclose(field, 0.0, atol=2.0e-13)
    evolved = jax.jit(axisymmetric_rk4_step)(vars, params)
    for before, after in zip(vars, evolved):
        np.testing.assert_allclose(after, before, atol=2.0e-13)
    violations = jax.jit(compute_axisymmetric_constraints)(vars, params)
    for field in violations:
        np.testing.assert_allclose(field, 0.0, atol=2.0e-13)
    norms = compute_axisymmetric_constraint_norms(violations)
    for value in norms.values():
        np.testing.assert_allclose(value, 0.0, atol=2.0e-13)


def test_constraint_norms_use_normalized_cylindrical_component_weighting():
    nrho, nz = 8, 11
    shape = (nrho + 4, 1, nz)
    scalar = jnp.ones(shape, dtype=jnp.float64)
    vector = jnp.ones((3,) + shape, dtype=jnp.float64)
    violations = ConstraintViolations(scalar, vector, scalar, scalar, vector)
    norms = compute_axisymmetric_constraint_norms(
        violations, exclude_outer_rho=0, exclude_z=0
    )
    for value in norms.values():
        np.testing.assert_allclose(value, 1.0, atol=2.0e-13)


def test_transverse_second_derivative_is_fourth_order():
    errors = []
    resolutions = (24, 32)
    for nrho in resolutions:
        dx = 2.0 / nrho
        nz = 2 * nrho
        params = _params(nrho, nz, dx)
        vars = _minkowski(nrho, nz, dx)
        rho = (jnp.arange(nrho) + 0.5) * dx
        z = params.z_min + jnp.arange(nz) * dx
        R, Z = jnp.meshgrid(rho, z, indexing="ij")
        values = 1.0 + 0.01 * jnp.exp(-(R**2 + Z**2))
        vars = fill_axisymmetric_ghosts(
            vars._replace(conformal_factor=_set_positive(vars.conformal_factor, values))
        )
        support = reconstruct_axisymmetric_support(vars, params)
        numerical = diff2_field(
            support.conformal_factor, 1, dx, PERIODIC_BC, PERIODIC_BC, 1.0
        )[:, AXISYMMETRIC_CENTER, :]
        expected = -0.02 * jnp.exp(-(R**2 + Z**2))
        error = jnp.max(
            jnp.abs(
                numerical[AXISYMMETRIC_GHOST_CELLS:-4, 4:-4]
                - expected[:-4, 4:-4]
            )
        )
        errors.append(float(error))
    order = np.log(errors[0] / errors[1]) / np.log(resolutions[1] / resolutions[0])
    assert order > 3.8


def test_axisymmetric_scalar_reconstruction_converges_above_fifth_order():
    errors = []
    resolutions = (24, 32)
    for nrho in resolutions:
        dx = 2.0 / nrho
        nz = 2 * nrho
        params = _params(nrho, nz, dx)
        vars = _minkowski(nrho, nz, dx)
        rho = (jnp.arange(nrho) + 0.5) * dx
        z = params.z_min + jnp.arange(nz) * dx
        R, Z = jnp.meshgrid(rho, z, indexing="ij")
        values = 1.0 + 0.01 * jnp.exp(-(R**2 + Z**2))
        vars = fill_axisymmetric_ghosts(
            vars._replace(conformal_factor=_set_positive(vars.conformal_factor, values))
        )
        support = reconstruct_axisymmetric_support(vars, params).conformal_factor
        x = (jnp.arange(nrho + 4) - 3.5) * dx
        y = (jnp.arange(9) - 4.0) * dx
        X, Y, Z3 = jnp.meshgrid(x, y, z, indexing="ij")
        exact = 1.0 + 0.01 * jnp.exp(-(X**2 + Y**2 + Z3**2))
        errors.append(
            float(jnp.max(jnp.abs(support[4:-5, :, 4:-4] - exact[4:-5, :, 4:-4])))
        )
    order = np.log(errors[0] / errors[1]) / np.log(resolutions[1] / resolutions[0])
    assert order > 5.3


def test_three_step_schwarzschild_puncture_regression_is_finite():
    nrho, nz, dx = 10, 17, 0.25
    params = _params(nrho, nz, dx)._replace(
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        dt=0.02 * dx,
    )
    vars = _minkowski(nrho, nz, dx)
    rho = (jnp.arange(nrho) + 0.5) * dx
    z = params.z_min + jnp.arange(nz) * dx
    R, Z = jnp.meshgrid(rho, z, indexing="ij")
    W = (1.0 + 1.0 / (2.0 * jnp.sqrt(R**2 + Z**2))) ** -2
    vars = fill_axisymmetric_ghosts(
        vars._replace(
            conformal_factor=_set_positive(vars.conformal_factor, W),
            lapse=_set_positive(vars.lapse, W),
        )
    )
    advance = jax.jit(lambda state: axisymmetric_rk4_step(state, params))
    for _ in range(3):
        vars = advance(vars)
        assert all(bool(jnp.all(jnp.isfinite(field))) for field in vars)
    violations = compute_axisymmetric_constraints(vars, params)
    assert all(bool(jnp.all(jnp.isfinite(field))) for field in violations)


def _smooth_axisymmetric_state(X, Y, Z):
    rho2 = X**2 + Y**2
    envelope = jnp.exp(-(rho2 + Z**2))
    shape = X.shape
    metric = jnp.eye(3)[:, :, None, None, None] * jnp.ones((3, 3) + shape)
    radial = 0.01 * envelope
    azimuthal = 0.006 * envelope
    vertical = 0.004 * envelope
    shift = jnp.stack(
        (radial * X - azimuthal * Y, radial * Y + azimuthal * X, vertical * Z)
    )
    connection = 0.5 * shift
    return BSSNVariables(
        conformal_metric=metric,
        conformal_factor=1.0 + 0.01 * envelope,
        traceless_K=jnp.zeros_like(metric),
        trace_K=0.005 * envelope,
        conformal_connection=connection,
        lapse=1.0 - 0.02 * envelope,
        shift=shift,
    )


def test_compact_rhs_converges_to_full_3d_axisymmetric_reference():
    errors = []
    resolutions = (24, 32)
    for full_nx in resolutions:
        dx = 4.0 / full_nx
        x = (jnp.arange(full_nx) - (full_nx - 1) / 2.0) * dx
        y = (jnp.arange(full_nx + 1) - full_nx / 2.0) * dx
        z = x
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        full_vars = _smooth_axisymmetric_state(X, Y, Z)
        full_params = BSSNParameters(
            dx=dx,
            dt=0.01 * dx,
            nu=0.0,
            gauge=1,
            x_min=float(x[0]),
            y_min=float(y[0]),
            z_min=float(z[0]),
        )
        X0, Z0 = jnp.meshgrid(x, z, indexing="ij")
        plane_vars = _smooth_axisymmetric_state(
            X0[:, None, :], jnp.zeros_like(X0[:, None, :]), Z0[:, None, :]
        )
        compact_vars = compact_axisymmetric_state(plane_vars)
        compact_params = _params(full_nx // 2, full_nx, dx, float(z[0]))._replace(
            dt=0.01 * dx
        )
        full_rhs = compute_bssn_rhs(full_vars, full_params)
        compact_rhs = compute_axisymmetric_rhs(compact_vars, compact_params)
        field_errors = []
        for full_field, compact_field in zip(full_rhs, compact_rhs):
            reference = full_field[
                (..., slice(full_nx // 2, -4), full_nx // 2, slice(4, -4))
            ]
            candidate = compact_field[
                (..., slice(AXISYMMETRIC_GHOST_CELLS, -4), 0, slice(4, -4))
            ]
            field_errors.append(float(jnp.max(jnp.abs(reference - candidate))))
        errors.append(field_errors)

    coarse, fine = errors
    for coarse_error, fine_error in zip(coarse, fine):
        if coarse_error > 1.0e-12:
            assert fine_error < coarse_error
        else:
            assert fine_error < 1.0e-12
    order = np.log(max(coarse) / max(fine)) / np.log(resolutions[1] / resolutions[0])
    assert order > 3.4
