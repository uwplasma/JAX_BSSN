"""Evolve a plus-polarized linear wave through nested fixed refinement."""

from functools import partial
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from JAX_BSSN.evolution.boundaries import PERIODIC_BC
from JAX_BSSN.bssn.tensor_algebra import (
    invert_3x3_metric,
    trace_tensor,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.diagnostics.openpmd import FMRPatchSeriesWriter, tiled_output_fields
from JAX_BSSN.diagnostics.waves import (
    h_plus, diagnostic_fields, composite_wave_diagnostics as _composite_wave_diagnostics,
)
from JAX_BSSN.fmr import (
    FMRHierarchySpec,
    FMRPatchSpec,
    fine_active_shape,
    fmr_rk4_step,
    initialize_tiled_fmr,
)
from JAX_BSSN.fmr.transfer import active_shape
from JAX_BSSN.parallel import map_tiled


def linear_wave_initial_data(coordinates, local_params, *, amplitude, wavelength):
    """Plus-polarized wave with its analytic conformal connection on owned vertices."""
    x, y, z = coordinates
    one = jnp.ones(local_params.owned_shape)
    wave_number = 2*jnp.pi / wavelength
    h = amplitude*jnp.sin(wave_number*x) + 0*y + 0*z
    dhdx = wave_number*amplitude*jnp.cos(wave_number*x) + 0*y + 0*z
    W = (1 - h*h)**(-1./6.)
    metric = jnp.zeros((3, 3) + one.shape)
    metric = metric.at[0, 0].set(W*W)
    metric = metric.at[1, 1].set((1 + h)*W*W)
    metric = metric.at[2, 2].set((1 - h)*W*W)
    K = jnp.zeros_like(metric).at[1, 1].set(dhdx/2).at[2, 2].set(-dhdx/2)
    inverse = invert_3x3_metric(metric)
    vector = jnp.zeros((3,) + one.shape)
    # det(gamma_tilde)=1 gives Gamma^x=-partial_x gamma_tilde^{xx}.
    connection = vector.at[0].set(2*h*dhdx/3*(1 - h*h)**(-2./3.))
    return BSSNVariables(
        metric, W, W*W*traceless_part(K, metric, inverse),
        W*W*trace_tensor(K, inverse), connection, one, vector,
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
    mesh=None,
):
    """Run a linear wave through a configurable nested FMR hierarchy.

    Pass a field mesh to distribute every patch over those devices; the default
    single-device mesh reproduces the original serial run.
    """
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

    base_params = BSSNParameters(
        eta=0.0, kappa=0.0, nu=0.0, g=0.0, dx=dx_root, dt=dt,
        zero_shift=1, gauge=0, xl_bc=PERIODIC_BC, xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC, yr_bc=PERIODIC_BC, zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC, x_min=root_origin[0], y_min=root_origin[1],
        z_min=root_origin[2],
    )
    # Child spacings, origins and boundary codes are derived from the root.
    states, parameters, layouts = initialize_tiled_fmr(
        partial(linear_wave_initial_data, amplitude=amplitude, wavelength=wavelength),
        base_params, (grid_size,) * 3, hierarchy, mesh=mesh,
    )
    advance = jax.jit(lambda values: fmr_rk4_step(
        values, parameters, hierarchy, layouts=layouts,
    ))

    @jax.jit
    def compute_output_fields(values):
        return tuple(map_tiled(
            diagnostic_fields, state, params, output_template={
                "h_plus": state.lapse, "lapse": state.lapse,
                "shift": tuple(state.shift[i] for i in range(3)),
                "K": state.trace_K, "W": state.conformal_factor,
                "hamiltonian_constraint": state.lapse,
                "momentum_constraint": tuple(state.shift[i] for i in range(3)),
            }, **layout,
        ) for state, params, layout in zip(values, parameters, layouts))

    output_steps = tuple(int(step) for step in np.rint(
        np.linspace(0, num_steps, output_iterations)).astype(int))
    output_indices = {step: index for index, step in enumerate(output_steps)}
    writer = FMRPatchSeriesWriter(output_path, dt=dt)

    shapes = tuple(active_shape(layout) for layout in layouts)
    coordinates = tuple(
        jnp.broadcast_to((params.x_min + jnp.arange(shape[0])*params.dx)[:, None, None], shape)
        for params, shape in zip(parameters, shapes)
    )

    def write_iteration(index, step):
        fields = compute_output_fields(states)
        # Copy owned samples once: the writer and the composite norms share them.
        active = tuple(tiled_output_fields(field, layout)
                       for field, layout in zip(fields, layouts))
        diagnostics = _composite_wave_diagnostics(
            active, coordinates, hierarchy, step * dt, amplitude, wavelength,
            active_fields=True,
        )
        writer.write(
            active, output_index=index, simulation_step=step, time=step * dt,
            spacings=tuple(params.dx for params in parameters),
            origins=tuple((params.x_min, params.y_min, params.z_min)
                          for params in parameters),
            ghost_cells=(0,) * len(active), hierarchy=hierarchy,
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
        "hierarchy": hierarchy, "parameters": parameters, "layouts": layouts,
        "diagnostics": final_diagnostics,
    }


def main():
    run_linear_wave()


if __name__ == "__main__":
    main()
