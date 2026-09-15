"""Spatial convergence test for a linear wave crossing an FMR patch."""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import openpmd_api as io

from JAX_BSSN.evolution.boundaries import PERIODIC_BC
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.fmr.refinement import (
    FMRHierarchySpec,
    FMRPatchSpec,
    fine_active_shape,
    fine_active_view,
    fine_coordinates,
    fmr_rk4_step,
)
from tests.initial_data import create_coordinate_arrays, linear_wave_data


jax.config.update("jax_enable_x64", True)


def _h_plus(variables):
    physical_metric_yy = (
        variables.conformal_metric[1, 1] / variables.conformal_factor**2
    )
    physical_metric_zz = (
        variables.conformal_metric[2, 2] / variables.conformal_factor**2
    )
    return 0.5 * (physical_metric_yy - physical_metric_zz)


def _linear_wave_fmr_error(grid_size):
    amplitude = 1.0e-8
    wavelength = 1.0
    final_time = 0.02
    dt = 0.000625
    num_steps = round(final_time / dt)

    dx_coarse = wavelength / grid_size
    dx_fine = dx_coarse / 2.0
    first_coarse_point = -(grid_size - 1) * dx_coarse / 2.0
    coarse_origin = (first_coarse_point,) * 3

    patch_spec = FMRPatchSpec(
        (grid_size // 4,) * 3,
        (3 * grid_size // 4 - 1,) * 3,
    )
    fine_X, fine_Y, fine_Z = fine_coordinates(
        patch_spec, dx_coarse, coarse_origin
    )
    fine_shape = tuple(
        size + 2 * patch_spec.ghost_width
        for size in fine_active_shape(patch_spec)
    )

    coarse = linear_wave_data(
        grid_size,
        grid_size,
        grid_size,
        dx_coarse,
        amplitude=amplitude,
        wavelength=wavelength,
        mad_q=(dx_fine / dx_coarse) ** 4,
    )
    fine = linear_wave_data(
        *fine_shape,
        dx_fine,
        amplitude=amplitude,
        wavelength=wavelength,
    )

    coarse_parameters = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        nu=0.0,
        g=0.0,
        dx=dx_coarse,
        dt=dt,
        zero_shift=1,
        gauge=0,
        xl_bc=PERIODIC_BC,
        xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=first_coarse_point,
        y_min=first_coarse_point,
        z_min=first_coarse_point,
    )
    fine_parameters = coarse_parameters._replace(
        dx=dx_fine,
        x_min=float(fine_X[0, 0, 0]),
        y_min=float(fine_Y[0, 0, 0]),
        z_min=float(fine_Z[0, 0, 0]),
    )

    def evolve_step(_, state):
        return fmr_rk4_step(
            state, (coarse_parameters, fine_parameters),
            FMRHierarchySpec((patch_spec,)),
        )

    evolve = jax.jit(
        lambda coarse_state, fine_state: jax.lax.fori_loop(
            0,
            num_steps,
            evolve_step,
            (coarse_state, fine_state),
        )
    )
    coarse, fine = evolve(coarse, fine)

    coarse_X, _, _ = create_coordinate_arrays(
        grid_size, grid_size, grid_size, dx_coarse
    )
    coarse_exact = amplitude * jnp.sin(
        2.0 * jnp.pi * (coarse_X - final_time) / wavelength
    )
    fine_exact = amplitude * jnp.sin(
        2.0 * jnp.pi * (fine_X - final_time) / wavelength
    )

    coarse_error = (_h_plus(coarse) - coarse_exact) / amplitude
    fine_error = fine_active_view(
        (_h_plus(fine) - fine_exact) / amplitude,
        patch_spec,
    )

    # Use each physical FMR degree of freedom once: native fine points inside
    # the patch and coarse points outside the covered box.
    coarse_uncovered = np.ones((grid_size,) * 3, dtype=bool)
    covered = tuple(
        slice(lower, upper + 1)
        for lower, upper in zip(patch_spec.coarse_lo, patch_spec.coarse_hi)
    )
    coarse_uncovered[covered] = False
    coarse_uncovered = jnp.asarray(coarse_uncovered)

    squared_error = (
        jnp.sum(jnp.where(coarse_uncovered, coarse_error**2, 0.0))
        + jnp.sum(fine_error**2)
    )
    point_count = jnp.sum(coarse_uncovered) + fine_error.size
    return float(jnp.sqrt(squared_error / point_count))


def test_fmr_linear_wave_has_fourth_order_spatial_convergence():
    coarse_grid_size = 12
    fine_grid_size = 16

    coarse_error = _linear_wave_fmr_error(coarse_grid_size)
    fine_error = _linear_wave_fmr_error(fine_grid_size)
    convergence_order = math.log(coarse_error / fine_error) / math.log(
        fine_grid_size / coarse_grid_size
    )

    print(
        "FMR linear-wave spatial errors/order",
        coarse_error,
        fine_error,
        convergence_order,
    )
    assert convergence_order >= 4.0


def test_small_production_fmr_run_writes_readable_visit_collection(tmp_path):
    from demos.linear_wave.linear_wave import run_linear_wave

    base = tmp_path / "linear_wave_fmr"
    _, run = run_linear_wave(
        grid_size=8,
        level_count=3,
        final_time=0.02,
        output_iterations=2,
        output_path=base,
        show_progress=False,
    )

    visit_path = Path(run["output_path"])
    assert visit_path == base.with_suffix(".visit")
    lines = visit_path.read_text().splitlines()
    assert lines[0] == "!NBLOCKS 3"
    assert len(lines) == 9
    assert lines[2].startswith("linear_wave_fmr_level_00_patch_000_")
    assert lines[3].startswith("linear_wave_fmr_level_01_patch_000_")
    assert lines[4].startswith("linear_wave_fmr_level_02_patch_000_")
    assert lines[6].startswith("linear_wave_fmr_level_00_patch_000_")
    assert lines[7].startswith("linear_wave_fmr_level_01_patch_000_")
    assert lines[8].startswith("linear_wave_fmr_level_02_patch_000_")

    for output_index in (0, 1):
        for level in (0, 1, 2):
            path = tmp_path / (
                f"linear_wave_fmr_level_{level:02d}_patch_000_"
                f"{output_index:08d}.h5"
            )
            series = io.Series(str(path), io.Access.read_only)
            iteration = series.iterations[output_index]
            assert set(iteration.meshes) == {
                "h_plus",
                "lapse",
                "shift",
                "K",
                "W",
                "hamiltonian_constraint",
                "momentum_constraint",
            }
            expected_shape = ((8, 8, 8) if level == 0 else
                              (7, 7, 7) if level == 1 else (5, 5, 5))
            assert tuple(
                iteration.meshes["h_plus"][
                    io.Mesh_Record_Component.SCALAR
                ].shape
            ) == expected_shape
            iteration.close()
            series.close()

    assert not base.with_suffix(".h5").exists()
