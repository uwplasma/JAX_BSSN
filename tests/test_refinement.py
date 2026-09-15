import jax
import jax.numpy as jnp
import numpy as np

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from tests.initial_data import (
    create_coordinate_arrays,
    gauge_wave_analytic_state,
    linear_wave_data,
)
from JAX_BSSN.fmr.refinement import (FMRHierarchySpec, FMRPatchSpec, fill_fine_ghosts,
    fine_active_shape, fine_active_slice, fine_active_view, fine_coordinates,
    fmr_rk4_step, prolongate_to_fine, restrict_to_coarse)

jax.config.update("jax_enable_x64", True)


SPEC = FMRPatchSpec((4, 4, 4), (9, 9, 9))


def _reference_interpolate_axis(array, targets, axis):
    """Original six-node gather formulation used as a regression reference."""
    base = np.floor(targets).astype(np.int32)
    offsets = np.arange(-2, 4, dtype=np.int32)
    nodes = base[:, None] + offsets[None, :]
    weights = np.ones(nodes.shape, dtype=array.dtype)
    for k in range(6):
        for m in range(6):
            if k != m:
                weights[:, k] *= (
                    (targets - nodes[:, m])
                    / (nodes[:, k] - nodes[:, m])
                )

    gathered = np.take(array, nodes % array.shape[axis], axis=axis)
    shape = [1] * gathered.ndim
    shape[axis], shape[axis + 1] = targets.size, 6
    return np.sum(gathered * weights.reshape(shape), axis=axis + 1)


def _reference_prolongate(array, spec):
    result = np.asarray(array)
    for direction in range(3):
        target_count = (
            fine_active_shape(spec)[direction] + 2 * spec.ghost_width
        )
        targets = (
            spec.coarse_lo[direction]
            + (np.arange(target_count) - spec.ghost_width) / 2.0
        )
        result = _reference_interpolate_axis(
            result, targets, result.ndim - 3 + direction
        )
    return result


def test_geometry_is_vertex_centered_with_four_guards():
    assert fine_active_shape(SPEC) == (11, 11, 11)
    X, Y, Z = fine_coordinates(SPEC, 0.1, (-0.5, -0.5, -0.5))
    active = fine_active_slice(SPEC)
    np.testing.assert_allclose(X[active][:, 0, 0], -0.1 + np.arange(11) * 0.05)
    np.testing.assert_allclose(Y[active][0, :, 0], -0.1 + np.arange(11) * 0.05)
    np.testing.assert_allclose(Z[active][0, 0, :], -0.1 + np.arange(11) * 0.05)


def test_coordinate_and_linear_wave_arrays_are_strongly_typed():
    coarse_coordinates = create_coordinate_arrays(8, 8, 8, 0.125)
    fine_grid = fine_coordinates(
        FMRPatchSpec((2, 2, 2), (5, 5, 5)),
        0.125,
        (-0.4375,) * 3,
    )
    variables = linear_wave_data(8, 8, 8, 0.125)
    float32_coordinates = create_coordinate_arrays(
        8, 8, 8, np.float32(0.125)
    )

    assert all(not coordinate.weak_type for coordinate in coarse_coordinates)
    assert all(not coordinate.weak_type for coordinate in fine_grid)
    assert all(not field.weak_type for field in variables)
    assert all(coordinate.dtype == jnp.float32
               for coordinate in float32_coordinates)


def test_quintic_prolongation_polynomials_and_leading_axes():
    n = 16
    x = jnp.arange(n, dtype=jnp.float64)
    # Patch and guards stay away from periodic wrapping for this exactness test.
    spec = FMRPatchSpec((5, 5, 5), (8, 8, 8))
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    scalar = X**5 + 2*Y**4 - Z**3 + X*Y
    got = prolongate_to_fine(scalar, spec)
    q = jnp.arange(3, 10.5, 0.5)
    qX, qY, qZ = jnp.meshgrid(q, q, q, indexing="ij")
    expected = qX**5 + 2*qY**4 - qZ**3 + qX*qY
    np.testing.assert_allclose(got, expected, rtol=2e-13, atol=2e-10)
    np.testing.assert_allclose(prolongate_to_fine(jnp.stack([scalar, 2*scalar]), spec)[1], 2*expected)
    tensor = jnp.stack([jnp.stack([scalar, 3*scalar])])
    assert prolongate_to_fine(tensor, spec).shape == (1, 2) + expected.shape


def test_dense_prolongation_matches_periodic_gather_reference():
    spec = FMRPatchSpec((0, 1, 2), (2, 3, 4))
    rng = np.random.default_rng(8147)

    for dtype, tolerance in ((np.float32, 2e-6), (np.float64, 2e-13)):
        scalar = rng.standard_normal((5, 6, 7)).astype(dtype)
        tensor = rng.standard_normal((2, 3, 5, 6, 7)).astype(dtype)
        interpolate = jax.jit(lambda array: prolongate_to_fine(array, spec))

        np.testing.assert_allclose(
            interpolate(jnp.asarray(scalar)),
            _reference_prolongate(scalar, spec),
            rtol=tolerance,
            atol=tolerance,
        )
        np.testing.assert_allclose(
            interpolate(jnp.asarray(tensor)),
            _reference_prolongate(tensor, spec),
            rtol=tolerance,
            atol=tolerance,
        )


def _variables(field):
    return BSSNVariables(field, field + 1, field + 2, field + 3,
                         field + 4, field + 5, field + 6)


def test_ghost_fill_preserves_active_and_fills_faces_edges_corners():
    n = 16
    X, Y, Z = jnp.meshgrid(jnp.arange(n), jnp.arange(n), jnp.arange(n), indexing="ij")
    coarse = _variables((X + 2*Y + 3*Z).astype(jnp.float64))
    shape = tuple(x + 8 for x in fine_active_shape(SPEC))
    fine = _variables(jnp.full(shape, -999.0))
    filled = fill_fine_ghosts(coarse, fine, SPEC)
    expected = prolongate_to_fine(coarse[0], SPEC)
    mask = np.ones(shape, dtype=bool)
    mask[fine_active_slice(SPEC)] = False
    np.testing.assert_allclose(np.asarray(filled[0])[mask], np.asarray(expected)[mask])
    np.testing.assert_array_equal(np.asarray(filled[0][fine_active_slice(SPEC)]), -999.0)


def test_restriction_injects_all_variable_shapes_exactly():
    coarse_shape = (16, 16, 16)
    active_shape = fine_active_shape(SPEC)
    padded = tuple(n + 8 for n in active_shape)
    base = jnp.arange(np.prod(padded), dtype=jnp.float64).reshape(padded)
    coarse = _variables(jnp.zeros(coarse_shape))
    fine = _variables(base)
    result = restrict_to_coarse(coarse, fine, SPEC)
    expected = base[fine_active_slice(SPEC)][::2, ::2, ::2]
    covered = tuple(slice(l, h + 1) for l, h in zip(SPEC.coarse_lo, SPEC.coarse_hi))
    for offset, field in enumerate(result):
        np.testing.assert_array_equal(field[covered], expected + offset)


def test_three_level_transfers_use_parent_ghost_offset_and_restrict_bottom_up():
    root_spec = FMRPatchSpec((3, 3, 3), (8, 8, 8))
    child_spec = FMRPatchSpec((2, 2, 2), (4, 4, 4))
    root_shape = (12, 12, 12)
    level_one_shape = tuple(n + 8 for n in fine_active_shape(root_spec))
    level_two_shape = tuple(n + 8 for n in fine_active_shape(child_spec))
    root = _variables(jnp.zeros(root_shape))
    level_one_base = jnp.arange(np.prod(level_one_shape), dtype=jnp.float64).reshape(
        level_one_shape
    )
    level_one = _variables(level_one_base)
    level_two = _variables(jnp.full(level_two_shape, -7.0))

    filled = fill_fine_ghosts(
        level_one, level_two, child_spec, parent_ghost_width=root_spec.ghost_width
    )
    expected = prolongate_to_fine(
        level_one[0], child_spec, parent_ghost_width=root_spec.ghost_width
    )
    mask = np.ones(level_two_shape, dtype=bool)
    mask[fine_active_slice(child_spec)] = False
    np.testing.assert_allclose(np.asarray(filled[0])[mask], np.asarray(expected)[mask])
    np.testing.assert_array_equal(
        np.asarray(filled[0][fine_active_slice(child_spec)]), -7.0
    )

    finest = _variables(jnp.full(level_two_shape, 42.0))
    restricted_one = restrict_to_coarse(
        level_one, finest, child_spec, parent_ghost_width=root_spec.ghost_width
    )
    restricted_root = restrict_to_coarse(root, restricted_one, root_spec)
    root_covered = tuple(slice(lo, hi + 1)
                         for lo, hi in zip(root_spec.coarse_lo, root_spec.coarse_hi))
    level_one_active = fine_active_view(restricted_one[0], root_spec)
    np.testing.assert_array_equal(restricted_root[0][root_covered],
                                  level_one_active[::2, ::2, ::2])


def test_stage_synchronous_gauge_wave_step_is_finite_and_accurate():
    n, dx = 10, 0.1
    x = -0.5 + jnp.arange(n) * dx
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    spec = FMRPatchSpec((3, 3, 3), (6, 6, 6))
    FX, FY, FZ = fine_coordinates(spec, dx, (-0.5, -0.5, -0.5))
    coarse = gauge_wave_analytic_state(X, Y, Z)
    fine = gauge_wave_analytic_state(FX, FY, FZ)
    dt = 0.005
    coarse_params = BSSNParameters(eta=0, kappa=0, g=0, nu=0, dx=dx,
        dt=dt, zero_shift=1, gauge=0, x_min=-0.5, y_min=-0.5, z_min=-0.5)
    fine_params = coarse_params._replace(dx=dx/2,
        x_min=float(FX[0, 0, 0]), y_min=float(FY[0, 0, 0]), z_min=float(FZ[0, 0, 0]))
    coarse, fine = fmr_rk4_step(
        (coarse, fine), (coarse_params, fine_params), FMRHierarchySpec((spec,))
    )
    assert all(bool(jnp.all(jnp.isfinite(field))) for field in coarse + fine)
    exact = gauge_wave_analytic_state(FX, FY, FZ, dt)
    error = fine_active_view(fine.lapse - exact.lapse, spec)
    assert float(jnp.sqrt(jnp.mean(error**2))) < 2e-7
    determinant = jnp.linalg.det(jnp.moveaxis(fine.conformal_metric, (0, 1), (-2, -1)))
    np.testing.assert_allclose(fine_active_view(determinant, spec), 1.0, atol=2e-12)


def test_stage_synchronous_nonzero_shift_upwind_step_is_finite():
    # Match the preceding test's shapes so a full-suite run can reuse JIT
    # compilations while exercising the coarse MAD and fine D4 upwind paths.
    n, dx = 10, 0.1
    origin = (-0.5,) * 3
    x = origin[0] + jnp.arange(n) * dx
    X, Y, Z = jnp.meshgrid(x, x, x, indexing="ij")
    spec = FMRPatchSpec((3, 3, 3), (6, 6, 6))
    FX, FY, FZ = fine_coordinates(spec, dx, origin)

    shift_components = jnp.asarray((0.125, -0.05, 0.025))

    def with_constant_shift(state):
        shift = jnp.broadcast_to(
            shift_components[:, None, None, None], state.shift.shape
        )
        return state._replace(shift=shift)

    coarse = with_constant_shift(gauge_wave_analytic_state(X, Y, Z))
    fine = with_constant_shift(gauge_wave_analytic_state(FX, FY, FZ))
    coarse_params = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        g=0.0,
        nu=0.0,
        dx=dx,
        dt=0.001,
        zero_shift=0,
        gauge=0,
        x_min=origin[0],
        y_min=origin[1],
        z_min=origin[2],
    )
    fine_params = coarse_params._replace(
        dx=dx / 2,
        x_min=float(FX[0, 0, 0]),
        y_min=float(FY[0, 0, 0]),
        z_min=float(FZ[0, 0, 0]),
    )

    coarse, fine = fmr_rk4_step(
        (coarse, fine), (coarse_params, fine_params), FMRHierarchySpec((spec,))
    )

    assert all(bool(jnp.all(jnp.isfinite(field))) for field in coarse + fine)
    expected_coarse_shift = jnp.broadcast_to(
        shift_components[:, None, None, None], coarse.shift.shape
    )
    expected_fine_shift = jnp.broadcast_to(
        shift_components[:, None, None, None], fine.shift.shape
    )
    np.testing.assert_allclose(coarse.shift, expected_coarse_shift, atol=2e-13)
    np.testing.assert_allclose(fine.shift, expected_fine_shift, atol=2e-13)
