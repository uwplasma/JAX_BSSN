"""Evolve a plus-polarized linear wave through nested fixed refinement."""

import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from JAX_BSSN.evolution.boundaries import PERIODIC_BC
from JAX_BSSN.bssn.constraints import (
    compute_hamiltonian_constraint,
    compute_momentum_constraint,
)
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    trace_tensor,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.diagnostics.openpmd import FMRPatchSeriesWriter
from JAX_BSSN.evolution.derivatives import diff1_field
from JAX_BSSN.fmr.refinement import (
    FMRHierarchySpec,
    FMRPatchSpec,
    fine_active_shape,
    fine_active_slice,
    fine_active_view,
    fine_coordinates,
    fmr_rk4_step,
)


def create_coordinate_arrays(
    ni: int, nj: int, nk: int, dx: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create vertex-centered Cartesian coordinates for this demonstration."""

    dtype = jnp.result_type(dx, 1.0)
    spacing = jnp.asarray(dx, dtype=dtype)
    half = jnp.asarray(2.0, dtype=dtype)
    x = (
        jnp.arange(ni, dtype=dtype)
        - jnp.asarray(ni - 1, dtype=dtype) / half
    ) * spacing
    y = (
        jnp.arange(nj, dtype=dtype)
        - jnp.asarray(nj - 1, dtype=dtype) / half
    ) * spacing
    z = (
        jnp.arange(nk, dtype=dtype)
        - jnp.asarray(nk - 1, dtype=dtype) / half
    ) * spacing

    return jnp.meshgrid(x, y, z, indexing="ij")


def _compute_conformal_connection(
    conformal_metric: jnp.ndarray, dx: float, mad_q: float = 1.0
) -> jnp.ndarray:
    """Compute conformal connection functions for the initial slice."""

    derivatives = jnp.stack(
        [
            diff1_field(conformal_metric, direction + 2, dx, mad_q=mad_q)
            for direction in range(3)
        ],
        axis=0,
    )
    inverse_metric = invert_3x3_metric(conformal_metric)
    christoffel = christoffel_symbols_second_kind(inverse_metric, derivatives)
    return jnp.einsum("mn..., imn... -> i...", inverse_metric, christoffel)


def linear_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 1.0e-8,
    wavelength: float = 1.0,
    mad_q: float = 1.0,
) -> BSSNVariables:
    """Initialize plus-polarized linear-wave data in Gauss coordinates."""

    shape = (ni, nj, nk)
    X, _, _ = create_coordinate_arrays(ni, nj, nk, dx)

    b0 = amplitude * jnp.sin((2.0 * jnp.pi * X) / wavelength)
    conformal_factor = jnp.power(1.0 - b0**2, -1.0 / 6.0)

    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(conformal_factor**2)
    conformal_metric = conformal_metric.at[1, 1].set(
        (1.0 + b0) * conformal_factor**2
    )
    conformal_metric = conformal_metric.at[2, 2].set(
        (1.0 - b0) * conformal_factor**2
    )

    dbdt0 = -(2.0 * jnp.pi * amplitude / wavelength) * jnp.cos(
        (2.0 * jnp.pi * X) / wavelength
    )
    extrinsic_curvature = jnp.zeros_like(conformal_metric)
    extrinsic_curvature = extrinsic_curvature.at[1, 1].set(-0.5 * dbdt0)
    extrinsic_curvature = extrinsic_curvature.at[2, 2].set(0.5 * dbdt0)

    inverse_metric = invert_3x3_metric(conformal_metric)
    traceless_K = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inverse_metric
    )
    trace_K = conformal_factor**2 * trace_tensor(
        extrinsic_curvature, inverse_metric
    )

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=_compute_conformal_connection(
            conformal_metric, dx, mad_q
        ),
        lapse=jnp.ones(shape),
        shift=jnp.zeros((3,) + shape),
    )


def h_plus(variables):
    """Return the physical plus-polarized metric perturbation."""

    physical_metric_yy = (
        variables.conformal_metric[1, 1] / variables.conformal_factor**2
    )
    physical_metric_zz = (
        variables.conformal_metric[2, 2] / variables.conformal_factor**2
    )
    return 0.5 * (physical_metric_yy - physical_metric_zz)


def diagnostic_fields(variables, parameters):
    """Compute the wave and BSSN constraint fields for one grid level."""

    hamiltonian = compute_hamiltonian_constraint(variables, parameters)
    momentum = compute_momentum_constraint(variables, parameters)

    return {
        "h_plus": h_plus(variables),
        "lapse": variables.lapse,
        "shift": tuple(variables.shift[i] for i in range(3)),
        "K": variables.trace_K,
        "W": variables.conformal_factor,
        "hamiltonian_constraint": hamiltonian,
        "momentum_constraint": tuple(momentum[i] for i in range(3)),
    }


def _rms(field, mask=None):
    if mask is None:
        return jnp.sqrt(jnp.mean(field**2))

    return jnp.sqrt(
        jnp.sum(jnp.where(mask, field**2, 0.0)) / jnp.sum(mask)
    )


def _maximum_absolute(field, mask=None):
    absolute_field = jnp.abs(field)
    if mask is not None:
        absolute_field = jnp.where(mask, absolute_field, 0.0)
    return jnp.max(absolute_field)


def wave_error_diagnostics(
    coarse_fields,
    fine_fields,
    coarse_X,
    fine_X,
    coarse_uncovered,
    patch_spec,
    time,
    amplitude,
    wavelength,
    region_masks=None,
):
    """Measure the numerical wave against the analytic traveling wave."""

    coarse_exact = amplitude * jnp.sin(
        2.0 * jnp.pi * (coarse_X - time) / wavelength
    )
    fine_exact = amplitude * jnp.sin(
        2.0 * jnp.pi * (fine_X - time) / wavelength
    )

    coarse_error = coarse_fields["h_plus"] - coarse_exact
    fine_error = fine_active_view(
        fine_fields["h_plus"] - fine_exact,
        patch_spec,
    )

    hamiltonian_coarse = coarse_fields["hamiltonian_constraint"]
    momentum_coarse = jnp.stack(coarse_fields["momentum_constraint"])
    hamiltonian_fine = fine_active_view(
        fine_fields["hamiltonian_constraint"],
        patch_spec,
    )
    momentum_fine = fine_active_view(
        jnp.stack(fine_fields["momentum_constraint"]),
        patch_spec,
    )

    diagnostics = {
        "coarse_rms": float(_rms(coarse_error, coarse_uncovered)),
        "coarse_max": float(
            _maximum_absolute(coarse_error, coarse_uncovered)
        ),
        "fine_rms": float(_rms(fine_error)),
        "fine_max": float(_maximum_absolute(fine_error)),
        "coarse_hamiltonian_l2": float(
            _rms(hamiltonian_coarse, coarse_uncovered)
        ),
        "fine_hamiltonian_l2": float(_rms(hamiltonian_fine)),
        "coarse_momentum_l2": float(
            _rms(jnp.sqrt(jnp.sum(momentum_coarse**2, axis=0)), coarse_uncovered)
        ),
        "fine_momentum_l2": float(
            _rms(jnp.sqrt(jnp.sum(momentum_fine**2, axis=0)))
        ),
    }
    if region_masks is not None:
        fine_bulk, coarse_bulk, fine_interface, coarse_interface = region_masks
        interface_sum = (
            jnp.sum(jnp.where(fine_interface, fine_error**2, 0.0))
            + jnp.sum(jnp.where(coarse_interface, coarse_error**2, 0.0))
        )
        interface_count = jnp.sum(fine_interface) + jnp.sum(coarse_interface)
        diagnostics.update({
            "fine_region_rms": float(_rms(fine_error, fine_bulk)),
            "coarse_region_rms": float(_rms(coarse_error, coarse_bulk)),
            "interface_region_rms": float(jnp.sqrt(interface_sum / interface_count)),
            "fine_region_max": float(_maximum_absolute(fine_error, fine_bulk)),
            "coarse_region_max": float(_maximum_absolute(coarse_error, coarse_bulk)),
            "interface_region_max": float(jnp.maximum(
                _maximum_absolute(fine_error, fine_interface),
                _maximum_absolute(coarse_error, coarse_interface),
            )),
        })
    return diagnostics


def _print_diagnostics(step, time, diagnostics, amplitude):
    normalization = abs(amplitude)
    print(
        f"step={step:4d} time={time:.8f} "
        f"coarse h+ L2/A={diagnostics['coarse_rms'] / normalization:.6e} "
        f"max/A={diagnostics['coarse_max'] / normalization:.6e} "
        f"fine h+ L2/A={diagnostics['fine_rms'] / normalization:.6e} "
        f"max/A={diagnostics['fine_max'] / normalization:.6e}"
    )
    print(
        " " * 24
        + f"coarse ||H||2={diagnostics['coarse_hamiltonian_l2']:.6e} "
        f"||M||2={diagnostics['coarse_momentum_l2']:.6e} "
        f"fine ||H||2={diagnostics['fine_hamiltonian_l2']:.6e} "
        f"||M||2={diagnostics['fine_momentum_l2']:.6e}"
    )


def _centered_patch(parent_shape):
    """Return a centered child covering approximately half of each parent."""
    bounds = []
    for size in parent_shape:
        covered = size // 2
        if covered < 2:
            raise ValueError("requested hierarchy is too deep for the parent patch")
        lower = (size - covered) // 2
        bounds.append((lower, lower + covered - 1))
    return FMRPatchSpec(tuple(value[0] for value in bounds),
                        tuple(value[1] for value in bounds))


def _composite_wave_diagnostics(level_fields, level_coordinates, hierarchy,
                                time, amplitude, wavelength):
    error_sum = 0.0
    point_count = 0
    maximum = 0.0
    hamiltonian_sum = 0.0
    momentum_sum = 0.0
    for level, (fields, coordinates) in enumerate(
        zip(level_fields, level_coordinates)
    ):
        if level:
            fields = {name: fine_active_view(value, hierarchy.patches[level - 1])
                      if not isinstance(value, tuple) else tuple(
                          fine_active_view(component, hierarchy.patches[level - 1])
                          for component in value)
                      for name, value in fields.items()}
            coordinates = fine_active_view(coordinates,
                                           hierarchy.patches[level - 1])
        mask = np.ones(fields["h_plus"].shape, dtype=bool)
        if level < len(hierarchy.patches):
            child = hierarchy.patches[level]
            covered = tuple(slice(lo, hi + 1)
                            for lo, hi in zip(child.coarse_lo, child.coarse_hi))
            mask[covered] = False
        mask = jnp.asarray(mask)
        exact = amplitude * jnp.sin(
            2.0 * jnp.pi * (coordinates - time) / wavelength
        )
        error = fields["h_plus"] - exact
        momentum = jnp.stack(fields["momentum_constraint"])
        count = jnp.sum(mask)
        error_sum += float(jnp.sum(jnp.where(mask, error**2, 0.0)))
        hamiltonian_sum += float(jnp.sum(jnp.where(
            mask, fields["hamiltonian_constraint"]**2, 0.0
        )))
        momentum_norm = jnp.sum(momentum**2, axis=0)
        momentum_sum += float(jnp.sum(jnp.where(mask, momentum_norm, 0.0)))
        maximum = max(maximum, float(_maximum_absolute(error, mask)))
        point_count += int(count)
    return {
        "composite_rms": math.sqrt(error_sum / point_count),
        "composite_max": maximum,
        "composite_hamiltonian_l2": math.sqrt(hamiltonian_sum / point_count),
        "composite_momentum_l2": math.sqrt(momentum_sum / point_count),
        # Retain concise summary keys used by callers of the demonstration.
        "coarse_rms": math.sqrt(error_sum / point_count),
        "fine_rms": math.sqrt(error_sum / point_count),
    }


def run_linear_wave(
    grid_size=32,
    amplitude=1.0e-8,
    wavelength=1.0,
    cfl=0.25,
    final_time=None,
    output_iterations=17,
    output_path=None,
    show_progress=True,
    use_mad=True,
    level_count=3,
):
    """Run a linear wave through a configurable nested FMR hierarchy."""
    if grid_size % 4 != 0:
        raise ValueError("the centered FMR hierarchy requires grid_size divisible by 4")
    if level_count < 1:
        raise ValueError("level_count must be positive")
    if amplitude == 0.0:
        raise ValueError("the normalized wave diagnostic requires nonzero amplitude")
    final_time = wavelength if final_time is None else final_time
    output_path = (Path(__file__).resolve().parent / "output" / "linear_wave_fmr"
                   if output_path is None else Path(output_path))

    dx_root = wavelength / grid_size
    dx_finest = dx_root / 2 ** (level_count - 1)
    num_steps = math.ceil(final_time / (cfl * dx_finest))
    dt = final_time / num_steps
    if output_iterations < 2 or output_iterations > num_steps + 1:
        raise ValueError("output_iterations must be between 2 and num_steps + 1")
    root_origin = (-(grid_size - 1) * dx_root / 2.0,) * 3

    patches = []
    active_shapes = [(grid_size,) * 3]
    for _ in range(1, level_count):
        patch = _centered_patch(active_shapes[-1])
        patches.append(patch)
        active_shapes.append(fine_active_shape(patch))
    hierarchy = FMRHierarchySpec(tuple(patches), use_mad=use_mad)

    spacings = tuple(dx_root / 2**level for level in range(level_count))
    coordinate_grids = [create_coordinate_arrays(
        grid_size, grid_size, grid_size, dx_root
    )]
    for level, patch in enumerate(patches, start=1):
        parent_active_origin = tuple(
            float(coordinate_grids[level - 1][axis][
                fine_active_slice(patches[level - 2]) if level > 1 else
                (slice(None),) * 3
            ].reshape(-1)[0]) for axis in range(3)
        )
        coordinate_grids.append(fine_coordinates(
            patch, spacings[level - 1], parent_active_origin
        ))

    states = tuple(
        linear_wave_data(*tuple(int(n) for n in grids[0].shape), spacings[level],
                         amplitude=amplitude, wavelength=wavelength,
                         mad_q=(dx_finest / spacings[level])**4
                         if use_mad and level < level_count - 1 else 1.0)
        for level, grids in enumerate(coordinate_grids)
    )
    base_params = BSSNParameters(
        eta=0.0, kappa=0.0, nu=0.0, g=0.0, dx=dx_root, dt=dt,
        zero_shift=1, gauge=0, xl_bc=PERIODIC_BC, xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC, yr_bc=PERIODIC_BC, zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC, x_min=root_origin[0], y_min=root_origin[1],
        z_min=root_origin[2],
    )
    parameters = tuple(
        base_params._replace(
            dx=spacings[level], x_min=float(grids[0][0, 0, 0]),
            y_min=float(grids[1][0, 0, 0]), z_min=float(grids[2][0, 0, 0]),
        ) for level, grids in enumerate(coordinate_grids)
    )
    advance = jax.jit(lambda values: fmr_rk4_step(values, parameters, hierarchy))
    compute_output_fields = jax.jit(lambda values: tuple(
        diagnostic_fields(value, params) for value, params in zip(values, parameters)
    ))
    output_steps = tuple(int(step) for step in np.rint(
        np.linspace(0, num_steps, output_iterations)).astype(int))
    output_indices = {step: index for index, step in enumerate(output_steps)}
    writer = FMRPatchSeriesWriter(output_path, dt=dt)

    active_origins = []
    for level, grids in enumerate(coordinate_grids):
        if level == 0:
            active_origins.append(root_origin)
        else:
            active = fine_active_slice(patches[level - 1])
            active_origins.append(tuple(float(grids[axis][active].reshape(-1)[0])
                                        for axis in range(3)))

    def write_iteration(index, step):
        fields = compute_output_fields(states)
        jax.block_until_ready(fields)
        diagnostics = _composite_wave_diagnostics(
            fields, tuple(grids[0] for grids in coordinate_grids), hierarchy,
            step * dt, amplitude, wavelength,
        )
        writer.write(
            fields, output_index=index, simulation_step=step, time=step * dt,
            spacings=tuple((spacing,) * 3 for spacing in spacings),
            origins=tuple(active_origins),
            ghost_cells=(0,) + tuple(patch.ghost_width for patch in patches),
            hierarchy=hierarchy,
        )
        if show_progress:
            print(f"step={step:4d} time={step*dt:.8f} composite h+ L2/A="
                  f"{diagnostics['composite_rms']/abs(amplitude):.6e}")
        return diagnostics

    final_diagnostics = write_iteration(0, 0)
    progress = tqdm(range(1, num_steps + 1), desc="Evolving linear wave",
                    unit="step", disable=not show_progress)
    try:
        for step in progress:
            states = advance(states)
            if step in output_indices:
                final_diagnostics = write_iteration(output_indices[step], step)
    finally:
        writer.close()
    normalized_error = final_diagnostics["composite_rms"] / abs(amplitude)
    if normalized_error >= 0.05:
        raise RuntimeError(f"composite h_plus RMS error exceeded 5%: {normalized_error}")
    return states, {
        "output_path": writer.visit_path, "num_steps": num_steps,
        "output_steps": output_steps, "final_time": num_steps * dt,
        "hierarchy": hierarchy, "diagnostics": final_diagnostics,
    }


def main():
    run_linear_wave()


if __name__ == "__main__":
    main()
