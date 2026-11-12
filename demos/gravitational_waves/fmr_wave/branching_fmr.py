"""A tiled wave with two sibling patches and a child nested inside the first."""

from functools import partial
import json
import math
from pathlib import Path
import sys
import time

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(__file__).resolve().parents[3]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import jax
import jax.numpy as jnp

from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.bssn.tensor_algebra import invert_3x3_metric, trace_tensor, traceless_part
from JAX_BSSN.diagnostics.openpmd import FMRPatchSeriesWriter, tiled_output_fields
from JAX_BSSN.fmr import (
    FMRHierarchySpec, FMRPatchSpec, hierarchy_depths, hierarchy_parents,
    initialize_tiled_fmr, tiled_fmr_rk4_step,
)
from JAX_BSSN.parallel import make_field_mesh, map_tiled
from JAX_BSSN.fmr.transfer import active_shape
from JAX_BSSN.diagnostics.waves import composite_wave_diagnostics, diagnostic_fields


# Simulation settings. CPU devices are configured in main, before making a mesh.
PLATFORM = "cpu"
TILE_COUNTS = (2, 1, 1)
ROOT_SHAPE = (24, 16, 16)
DX = 1. / ROOT_SHAPE[0]
DT = DX / 32
AMPLITUDE = 1.e-8
WAVELENGTH = 1.
STEPS = 480
OUTPUT_EVERY = 4
OUTPUT_PATH = SCRIPT_DIR / "output" / "branching_fmr"
PROGRESS = True

HIERARCHY = FMRHierarchySpec((
    FMRPatchSpec((3, 4, 4), (9, 11, 11), parent=0),
    FMRPatchSpec((14, 4, 4), (20, 11, 11), parent=0),
    FMRPatchSpec((3, 3, 3), (9, 11, 11), parent=1),
), use_mad=True)
ROOT_PARAMETERS = BSSNParameters(
    eta=0., kappa=0., nu=0., g=0., dx=DX, dt=DT,
    zero_shift=1, gauge=0,
    x_min=-(ROOT_SHAPE[0] - 1)*DX/2,
    y_min=-(ROOT_SHAPE[1] - 1)*DX/2,
    z_min=-(ROOT_SHAPE[2] - 1)*DX/2,
)


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


def run_branching_wave(output_path=None, *, steps=STEPS, mesh=None, amplitude=AMPLITUDE,
                       output_every=OUTPUT_EVERY, progress=False, wavelength=WAVELENGTH,
                       root_shape=ROOT_SHAPE, root_params=ROOT_PARAMETERS,
                       hierarchy=HIERARCHY):
    """Advance a small periodic tree and write native patch samples.

    Pass a field mesh to distribute each patch over those devices. Output is
    currently single-process. Set output_every=None for initial/final output only.
    """
    if steps < 0 or (output_every is not None and output_every <= 0):
        raise ValueError("steps must be nonnegative and output_every must be positive")
    started = time.perf_counter()
    jax.config.update("jax_enable_x64", True)
    params = root_params
    initial = partial(linear_wave_initial_data, amplitude=amplitude, wavelength=wavelength)

    if progress:
        print(f"Initializing {len(hierarchy.patches) + 1} tiled patches (includes JAX compilation).", flush=True)
    states, parameters, layouts = initialize_tiled_fmr(
        initial, params, root_shape, hierarchy, mesh=mesh,
    )
    advance = jax.jit(lambda values: tiled_fmr_rk4_step(
        values, parameters, hierarchy, layouts=layouts,
    ))
    base = OUTPUT_PATH if output_path is None else Path(output_path)
    writer = FMRPatchSeriesWriter(base, dt=params.dt)
    active_shapes = tuple(active_shape(layout) for layout in layouts)
    coordinates = tuple(
        jnp.broadcast_to((p.x_min + jnp.arange(n[0])*p.dx)[:, None, None], n)
        for p, n in zip(parameters, active_shapes)
    )

    @jax.jit
    def output_fields(values):
        return tuple(map_tiled(
            diagnostic_fields, state, p, output_template={
                "h_plus": state.lapse, "lapse": state.lapse,
                "shift": tuple(state.shift[i] for i in range(3)),
                "K": state.trace_K, "W": state.conformal_factor,
                "hamiltonian_constraint": state.lapse,
                "momentum_constraint": tuple(state.shift[i] for i in range(3)),
            }, **layout,
        ) for state, p, layout in zip(values, parameters, layouts))

    parents, depths = hierarchy_parents(hierarchy), hierarchy_depths(hierarchy)
    metadata_path = writer.base_path.with_suffix(".json")
    metadata = {
        "manifest": writer.visit_path.name, "amplitude": amplitude, "dt": params.dt,
        "steps": steps, "output_every": output_every,
        "patches": [
            {"id": i, "parent": parents[i], "depth": depths[i],
             "ordinal": depths[:i].count(depths[i]),
             "origin": [p.x_min, p.y_min, p.z_min], "shape": active_shapes[i],
             "dx": p.dx, "tile_shape": layout["tile_shape"],
             "tile_counts": layout["mesh"].devices.shape,
             "upper_endpoints": layout["upper_endpoints"]}
            for i, (p, layout) in enumerate(zip(parameters, layouts))
        ],
        "history": [],
    }

    def write_output(values, index, step):
        fields = output_fields(values)
        # Copy owned samples once: the writer and composite norms share them.
        active = tuple(tiled_output_fields(field, layout)
                       for field, layout in zip(fields, layouts))
        writer.write(
            active, output_index=index, simulation_step=step, time=step*params.dt,
            spacings=tuple(p.dx for p in parameters),
            origins=tuple((p.x_min, p.y_min, p.z_min) for p in parameters),
            ghost_cells=(0,) * len(active), hierarchy=hierarchy,
        )
        diagnostics = composite_wave_diagnostics(
            active, coordinates, hierarchy, step*params.dt, amplitude, wavelength,
            active_fields=True,
        )
        if not all(math.isfinite(value) for value in diagnostics.values()):
            raise FloatingPointError(f"non-finite wave diagnostics at step {step}")
        metadata["history"].append({"step": step, "time": step*params.dt, **diagnostics})
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        if progress:
            print(f"step {step}/{steps}, t={step*params.dt:.6f}, "
                  f"wave error RMS/A={diagnostics['composite_rms']/abs(amplitude):.3e}, "
                  f"elapsed={time.perf_counter() - started:.1f}s", flush=True)
        return diagnostics

    try:
        diagnostics = write_output(states, 0, 0)
        if progress and steps:
            print("Advancing the wave; the first step includes RK4 compilation.", flush=True)
        index = 1
        for step in range(1, steps + 1):
            states = advance(states)
            if step == steps or (output_every is not None and step % output_every == 0):
                diagnostics = write_output(states, index, step)
                index += 1
    finally:
        writer.close()
    return states, {"output_path": writer.visit_path, "hierarchy": hierarchy,
                    "parameters": parameters, "layouts": layouts,
                    "diagnostics": diagnostics, "metadata_path": metadata_path,
                    "history": metadata["history"]}


def main():
    jax.config.update("jax_platforms", PLATFORM)
    if PLATFORM == "cpu":
        jax.config.update("jax_num_cpu_devices", math.prod(TILE_COUNTS))
    states, result = run_branching_wave(
        OUTPUT_PATH, steps=STEPS, output_every=OUTPUT_EVERY,
        mesh=make_field_mesh(TILE_COUNTS), amplitude=AMPLITUDE, wavelength=WAVELENGTH,
        root_shape=ROOT_SHAPE, root_params=ROOT_PARAMETERS, hierarchy=HIERARCHY,
        progress=PROGRESS,
    )
    print(result["output_path"])
    print(result["metadata_path"])
    print(result["diagnostics"])
    return states, result


if __name__ == "__main__":
    main()
