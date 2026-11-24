"""Evolve an equal-mass binary black hole inspiral and merger with full 3D BSSN.

Two Bowen-York punctures start on the x-axis with tangential and small infall momenta.
The Hamiltonian constraint is solved for the conformal factor (Newton iteration with
conjugate-gradient linear solves), then the moving-puncture gauge (1+log slicing with a
pre-collapsed lapse alpha = W, Gamma-driver shift) is evolved with RK4 on a uniform grid
with Sommerfeld outer faces.

The initial data are solved once by prepare_initial_data in a single process whose grid is
split over all of its GPUs, and saved to <output_dir>/initial_data/*.npy; a single-process
run prepares them automatically. The evolution splits the grid into one tile per device
with JAX_BSSN.parallel, each device memory-mapping only its own tile of the saved data, and
tiles exchange only their guard layers at each RK4 stage, so the run scales across GPUs
and nodes. On several nodes first run prepare_initial_data as one task (srun -n 1
--gpus-per-task=4), then launch one process per GPU (srun --ntasks-per-node=4
--gpus-per-task=1); distributed JAX is initialised automatically for multi-task steps.

Writes output/constraint_l2.txt (time, Hamiltonian L2, momentum L2) every step,
output/initial_data/ (initial data and newton_residual.txt), output/gw_probe_samples.npz (strain h+, hx sampled every
PROBE_INTERVAL at points on the z and x axes, plotted by gw_strain_probe.py) and, every SNAPSHOT_INTERVAL steps, one NumPy file per BSSN
field (used by make_inspiral_movies.py). run_inspiral_merger is
also imported by inspiral_merger_convergence.py.
"""

import math
import os
from pathlib import Path

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")  # limit allocator fragmentation

import jax

if int(os.environ.get("SLURM_STEP_NUM_TASKS", os.environ.get("SLURM_NTASKS", "1"))) > 1:
    jax.distributed.initialize()  # must run before any other JAX or JAX_BSSN call
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from tqdm import tqdm

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC
from JAX_BSSN.parallel import (MESH_AXES, TileLayout, initialize_tiled_bssn, local_parameters,
                               owned_mask, tile_spec, tiled_constraint_norms,
                               tiled_rk4_step)
from JAX_BSSN.utilities.bowen_york_solver import (
    bowen_york_extrinsic_curvature,
    cell_centered_grid,
    solve_conformal_factor,
)


MASS = 0.5                   # bare mass of each puncture (total M = 1)
SEPARATION = 6.0             # initial coordinate separation
TANGENTIAL_MOMENTUM = 0.14   # momentum along y of each puncture (opposite signs)
INFALL_MOMENTUM = 0.01       # momentum along x towards the other puncture
HALF_WIDTH = 12.0            # domain is [-HALF_WIDTH, HALF_WIDTH]^3
NUM_POINTS = 300             # grid points per dimension (each tile count must divide it)
CFL = 0.15                   # dt / dx
FINAL_TIME = 100.0           # evolution time
SNAPSHOT_INTERVAL = 60       # steps between saved full snapshots (large: 27 fields per grid point)
PROBE_FRACTIONS = [0.20, 0.30, 0.40]  # strain probe radii as fractions of the grid width (< 0.5)
PROBE_INTERVAL = 0.5         # time between in-run strain samples at the probes
NEWTON_TOLERANCE = 1.0e-14   # initial-data Newton tolerance
CG_TOLERANCE = 1.0e-14       # initial-data conjugate-gradient tolerance
GUARD_CELLS = 4              # tile guard depth required by the BSSN stencils
OUTPUT_DIR = Path("output")
ROOT = jax.process_index() == 0  # only process 0 prints and writes files


def tile_layout(num_points):
    """
    Split the grid over every device with the most cube-like tiles that divide it.

    Args:
        num_points: Grid points per dimension

    Returns:
        TileLayout with one tile per device and GUARD_CELLS guards
    """
    count = len(jax.devices())
    shapes = [(a, b, count // (a * b)) for a in range(1, count + 1) for b in range(1, count // a + 1)
              if count % (a * b) == 0 and all(num_points % t == 0 for t in (a, b, count // (a * b)))]
    if not shapes:
        raise ValueError(f"No tiling of {num_points}^3 over {count} devices; choose NUM_POINTS divisible by them")
    tiles = min(shapes, key=lambda t: sum(num_points**2 / (t[i] * t[(i + 1) % 3]) for i in range(3)))
    if ROOT:
        print(f"JAX devices: {count} on {jax.process_count()} process(es); tiles {tiles}")
    # jax.make_mesh rejects meshes spanning several GPU nodes, so build the Mesh directly;
    # devices are ordered by process, so node boundaries fall between whole x-slabs
    mesh = Mesh(np.asarray(jax.devices()).reshape(tiles), MESH_AXES, axis_types=(AxisType.Auto,) * 3)
    return TileLayout(mesh, tuple(num_points // t for t in tiles), GUARD_CELLS)


def gather(field, layout):
    """
    Assemble the owned samples of a tiled field into one global host array, one tensor
    component at a time so the all-gather never holds more than one scalar grid per device.

    Args:
        field: Tiled array (components..., tx, ty, tz, nx + 2g, ny + 2g, nz + 2g)
        layout: TileLayout of the field

    Returns:
        NumPy array (components..., tx*nx, ty*ny, tz*nz)
    """
    g, lead = layout.guard_cells, field.ndim - 6
    tiles, owned = field.shape[lead:lead + 3], tuple(n - 2 * g for n in field.shape[lead + 3:])
    result = np.empty(field.shape[:lead] + tuple(t * n for t, n in zip(tiles, owned)), dtype=field.dtype)
    for component in np.ndindex(field.shape[:lead]):
        part = field[component][..., g:-g, g:-g, g:-g]
        part = np.asarray(multihost_utils.process_allgather(part, tiled=True) if jax.process_count() > 1 else part)
        result[component] = part.transpose(0, 3, 1, 4, 2, 5).reshape(result.shape[lead:])
        del part
    return result


def prepare_initial_data(num_points, output_dir=OUTPUT_DIR):
    """
    Solve equal-mass Bowen-York binary puncture data and save it for the tiled evolution.

    The grid is split along x over every local GPU, so the solve needs roughly 1/n of the
    memory per device. Run as a single process.

    Args:
        num_points: Grid points per dimension
        output_dir: Run directory; files go to <output_dir>/initial_data

    Returns:
        Path of the initial-data directory
    """
    directory = Path(output_dir) / "initial_data"
    directory.mkdir(parents=True, exist_ok=True)
    devices = jax.local_devices()
    count = next(n for n in range(len(devices), 0, -1) if num_points % n == 0)
    print(f"[{num_points}^3] solving Bowen-York initial data over {count} device(s) ...")
    grid, dx = cell_centered_grid(num_points=num_points, half_width=HALF_WIDTH)
    grid = jax.device_put(grid, NamedSharding(Mesh(np.asarray(devices[:count]), ("x",)), P("x")))
    positions = jnp.array([[-SEPARATION / 2, 0, 0], [SEPARATION / 2, 0, 0]], dtype=grid.dtype)
    momenta = jnp.array([[INFALL_MOMENTUM, TANGENTIAL_MOMENTUM, 0], [-INFALL_MOMENTUM, -TANGENTIAL_MOMENTUM, 0]],
                        dtype=grid.dtype)
    psi, _, residuals = solve_conformal_factor(
        jnp.array([MASS, MASS], dtype=grid.dtype), positions, momenta, grid, dx,
        newton_tolerance=NEWTON_TOLERANCE, max_newton_iterations=300,
        cg_tolerance=CG_TOLERANCE, max_cg_iterations=5000, verbose=True)
    W = psi**-2
    np.save(directory / "conformal_factor.npy", np.asarray(W))
    # conformally flat, maximal (K = 0) data with A_tilde_ij = psi^-6 A_hat_ij = W^3 A_hat_ij
    A_hat = bowen_york_extrinsic_curvature(momenta, positions, grid)
    A_tilde = np.lib.format.open_memmap(directory / "traceless_K.npy", mode="w+", shape=(3, 3) + W.shape)
    for i in range(3):
        for j in range(3):
            A_tilde[i, j] = np.asarray(W**3 * A_hat[..., i, j])
    A_tilde.flush()
    np.savetxt(directory / "newton_residual.txt", np.asarray(residuals), header="interior Hamiltonian residual RMS")
    print(f"Saved initial data in {directory}")
    return directory


def load_initial_data(directory):
    """
    Describe the saved initial data for initialize_tiled_bssn, which memory-maps each tile.

    Args:
        directory: Directory written by prepare_initial_data

    Returns:
        Mapping from BSSN field name to a .npy path, a callback or False (zero)
    """
    identity = lambda coordinates, p: jnp.broadcast_to(jnp.eye(3)[:, :, None, None, None], (3, 3) + tuple(p.owned_shape))
    W = str(directory / "conformal_factor.npy")
    return {"conformal_metric": identity, "conformal_factor": W, "traceless_K": str(directory / "traceless_K.npy"),
            "trace_K": False, "conformal_connection": False, "lapse": W, "shift": False}


def run_inspiral_merger(num_points=NUM_POINTS, output_dir=OUTPUT_DIR, snapshot_interval=SNAPSHOT_INTERVAL):
    """
    Load (or, in a single process, prepare) the initial data and evolve the binary on tiles.

    Args:
        num_points: Grid points per dimension
        output_dir: Directory for the initial data, constraint history, strain samples and snapshots
        snapshot_interval: Steps between snapshots; 0 disables snapshots

    Returns:
        Tuple (final BSSNVariables of global host arrays, dx, constraint history array of [time, H L2, M L2])
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    directory = output_dir / "initial_data"
    if not (directory / "traceless_K.npy").exists():
        if jax.process_count() > 1:
            raise SystemExit(f"No initial data in {directory}: run prepare_initial_data({num_points}, "
                             f"'{output_dir}') as a single task first")
        prepare_initial_data(num_points, output_dir)
    layout = tile_layout(num_points)
    dx = 2 * HALF_WIDTH / num_points
    num_steps = math.ceil(FINAL_TIME / (CFL * dx))
    x_min = -HALF_WIDTH + dx / 2
    params = BSSNParameters(eta=2.0, kappa=0.002, nu=0.02, g=0.75, dx=dx, dt=FINAL_TIME / num_steps, gauge=1,
                            xl_bc=SOMMERFELD_BC, xr_bc=SOMMERFELD_BC, yl_bc=SOMMERFELD_BC,
                            yr_bc=SOMMERFELD_BC, zl_bc=SOMMERFELD_BC, zr_bc=SOMMERFELD_BC,
                            x_min=x_min, y_min=x_min, z_min=x_min)
    state = initialize_tiled_bssn(load_initial_data(directory), params, layout)  # each device reads only its tile
    advance = jax.jit(lambda s: tiled_rk4_step(s, params, layout), donate_argnums=0)
    centre, g = num_points // 2, layout.guard_cells
    probes = [(axis, int(f * num_points)) for f in PROBE_FRACTIONS for axis in ("z", "x")]
    points = [(centre, centre, centre + r) if axis == "z" else (centre + r, centre, centre) for axis, r in probes]
    # global grid index -> (tile x, tile y, tile z, padded local x, y, z)
    points = [tuple(i // w for i, w in zip(p, layout.tile_shape)) + tuple(g + i % w for i, w in zip(p, layout.tile_shape))
              for p in points]
    sample_every = max(1, round(PROBE_INTERVAL / params.dt))

    def strain(state):
        """
        Leading-order TT strain from the physical metric gamma_ij = W^-2 gamma_tilde_ij at each probe.

        Args:
            state: Tiled BSSNVariables

        Returns:
            Array (probes, 2) of h+ and hx; z-axis: ((g_xx - g_yy)/2, g_xy), x-axis: ((g_yy - g_zz)/2, g_yz)
        """
        values = []
        for (axis, _), point in zip(probes, points):
            g_ij = state.conformal_metric[(slice(None), slice(None)) + point] / state.conformal_factor[point]**2
            a, b = (0, 1) if axis == "z" else (1, 2)  # the two directions transverse to the axis
            values.append(jnp.stack([(g_ij[a, a] - g_ij[b, b]) / 2, g_ij[a, b]]))
        return jnp.stack(values)

    sample = jax.jit(strain, out_shardings=NamedSharding(layout.mesh, P()))  # replicated on every process
    samples = [(0.0, np.asarray(sample(state)))]

    def save_samples():
        """
        Write the strain samples so far to gw_probe_samples.npz (process 0 only).

        Args:
            None

        Returns:
            None
        """
        if ROOT:
            np.savez(output_dir / "gw_probe_samples.npz", t=np.array([t for t, _ in samples]),
                     strain=np.array([h for _, h in samples]), labels=np.array([f"{a}-axis r={r}" for a, r in probes]),
                     radius=np.array([r * dx for _, r in probes]), axis=np.array([a for a, _ in probes]))

    def monitors(local):
        """
        Puncture-gauge monitors over owned samples, reduced across every tile.

        Args:
            local: One device's tiled BSSNVariables (tile axes of size one)

        Returns:
            Tuple (min W, min lapse, max |shift|)
        """
        local = jax.tree.map(lambda a: a.reshape(a.shape[:-6] + a.shape[-3:]), local)
        mask = owned_mask(local.lapse.shape, local_parameters(params, layout))
        low = lambda a: jax.lax.pmin(jnp.min(jnp.where(mask, a, jnp.inf)), MESH_AXES)
        shift = jnp.sqrt(jnp.sum(local.shift**2, axis=0))
        return low(local.conformal_factor), low(local.lapse), jax.lax.pmax(jnp.max(jnp.where(mask, shift, 0.0)), MESH_AXES)

    @jax.jit
    def diagnostics(state):
        """
        Constraint norms and gauge monitors for the tiled state.

        Args:
            state: Tiled BSSNVariables

        Returns:
            Tuple (Hamiltonian L2, momentum L2, min W, min lapse, max |shift|)
        """
        norms = tiled_constraint_norms(state, params, layout)
        gauge = jax.shard_map(monitors, mesh=layout.mesh, in_specs=(jax.tree.map(tile_spec, state),),
                              out_specs=P(), check_vma=False)(state)
        return (norms["hamiltonian_l2"], norms["momentum_l2"], *gauge)

    def save(step):
        """
        Gather every BSSN field and write it with the time for one step (process 0 writes).

        Args:
            step: Step number used in the file names

        Returns:
            None
        """
        for name, field in zip(BSSNVariables._fields, state):
            field = gather(field, layout)  # every process takes part in the gather
            if ROOT:
                np.save(output_dir / f"{name}_step_{step:06d}.npy", field)
        if ROOT:
            np.save(output_dir / f"time_step_{step:06d}.npy", np.asarray(step * params.dt))

    history = [(0.0, *map(float, diagnostics(state)[:2]))]
    if snapshot_interval:
        save(0)
    if ROOT:
        print(f"[{num_points}^3] dx = {dx:.4f}, {num_steps} RK4 steps to t = {FINAL_TIME}")
    progress = tqdm(range(1, num_steps + 1), desc=f"Evolving {num_points}^3", unit="step", disable=not ROOT)
    for step in progress:
        state = advance(state)
        ham, mom, min_W, min_alpha, max_beta = map(float, diagnostics(state))
        history.append((step * params.dt, ham, mom))
        progress.set_postfix(H=f"{ham:.2e}", M=f"{mom:.2e}", min_W=f"{min_W:.2e}",
                             min_lapse=f"{min_alpha:.3f}", max_shift=f"{max_beta:.3f}")
        if snapshot_interval and (step % snapshot_interval == 0 or step == num_steps):
            save(step)
        if step % sample_every == 0:
            samples.append((step * params.dt, np.asarray(sample(state))))
            if len(samples) % 100 == 0:
                save_samples()  # keep partial results if the job hits its time limit
    save_samples()
    history = np.array(history)
    if ROOT:
        np.savetxt(output_dir / "constraint_l2.txt", history, header="time hamiltonian_l2 momentum_l2")
    state = BSSNVariables(*(gather(field, layout) for field in state))
    if ROOT and not all(np.all(np.isfinite(field)) for field in state):
        print("WARNING: the final state contains non-finite values")
    return state, dx, history


if __name__ == "__main__":
    if ROOT:
        print(f"Binary: m = {MASS} each, separation {SEPARATION}, P_t = {TANGENTIAL_MOMENTUM}, "
              f"P_in = {INFALL_MOMENTUM}, domain +/-{HALF_WIDTH}, {NUM_POINTS}^3, t_final = {FINAL_TIME}")
    state, dx, history = run_inspiral_merger()
    if ROOT:
        print(f"Final Hamiltonian L2 = {history[-1, 1]:.4e}, momentum L2 = {history[-1, 2]:.4e}")
        print(f"Saved {OUTPUT_DIR / 'constraint_l2.txt'}, strain samples and snapshots in {OUTPUT_DIR}")
