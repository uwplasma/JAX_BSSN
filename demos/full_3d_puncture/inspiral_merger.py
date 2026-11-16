"""Build quasi-circular Bowen-York binary data and evolve a merger scaffold."""

import math
import os
from collections.abc import Sequence
from pathlib import Path

# Limit pre-allocation fraction to prevent BFC memory fragmentation
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from JAX_BSSN.bssn.constraints import (
    compute_hamiltonian_constraint,
    compute_momentum_constraint,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import SOMMERFELD_BC
from JAX_BSSN.evolution.time_evolve import rk4_step
from JAX_BSSN.utilities.bowen_york_solver import (
    bowen_york_extrinsic_curvature,
    cell_centered_grid,
    solve_conformal_factor,
)


MASS = 0.5
SEPARATION = 6.0
TANGENTIAL_MOMENTUM = 0.14
INFALL_MOMENTUM = 0.01
HALF_WIDTH = 12.0
NUM_POINTS = 300
CFL = 0.15
FINAL_TIME = 100.0
SNAPSHOT_INTERVAL = 60

NEWTON_TOLERANCE = 1.0e-14
MAX_NEWTON_ITERATIONS = 300
CG_TOLERANCE = 1.0e-14
MAX_CG_ITERATIONS = 5000

OUTPUT_DIR = Path("output")
SAVE_SNAPSHOTS = True
SHOW_PROGRESS = True


def select_shard_devices(num_points: int, devices: Sequence[object]) -> tuple[object, ...]:
    """Return the largest usable device subset for x-axis sharding."""

    available_devices = tuple(devices)
    for candidate_count in range(len(available_devices), 1, -1):
        if num_points % candidate_count == 0:
            return available_devices[:candidate_count]
    return ()


def shard_bssn_variables(vars: BSSNVariables, mesh: Mesh) -> BSSNVariables:
    """Shard BSSN grid variables across the device mesh along the first spatial dimension."""
    def _shard_leaf(x):
        if not isinstance(x, (jnp.ndarray, jax.Array)):
            return x
        if x.ndim == 3:
            spec = P('x', None, None)
        elif x.ndim == 4:
            spec = P(None, 'x', None, None)
        elif x.ndim == 5:
            spec = P(None, None, 'x', None, None)
        else:
            spec = P()
        return jax.device_put(x, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map(_shard_leaf, vars)


@jax.jit
def compute_diagnostics_jit(vars: BSSNVariables, params: BSSNParameters):
    """JIT-compiled diagnostic calculation to prevent Device 0 memory spills."""
    hamiltonian = compute_hamiltonian_constraint(vars, params)
    momentum = compute_momentum_constraint(vars, params)
    ham_l2 = jnp.sqrt(jnp.mean(hamiltonian**2))
    mom_l2 = jnp.sqrt(jnp.mean(momentum**2))
    min_W = jnp.min(vars.conformal_factor)
    min_alpha = jnp.min(vars.lapse)
    max_beta = jnp.max(jnp.abs(vars.shift))
    return ham_l2, mom_l2, min_W, min_alpha, max_beta


def save_snapshot(
    vars: BSSNVariables,
    time: float,
    step: int,
    output_dir: Path,
) -> None:
    """Save one synchronized BSSN state as NumPy arrays."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jax.block_until_ready(vars)

    np.save(output_dir / f"time_step_{step:06d}.npy", np.asarray(time))
    for field_name, field in zip(BSSNVariables._fields, vars):
        np.save(output_dir / f"{field_name}_step_{step:06d}.npy", np.asarray(field))


def binary_bowen_york_data(
    mass: float,
    separation: float,
    tangential_momentum: float,
    infall_momentum: float,
    half_width: float,
    num_points: int,
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    verbose: bool = False,
) -> tuple[jnp.ndarray, float, BSSNVariables, jnp.ndarray, list[float]]:
    """Solve equal-mass Bowen-York data for a tangential binary."""

    grid, dx = cell_centered_grid(num_points=num_points, half_width=half_width)
    masses = jnp.asarray([mass, mass], dtype=grid.dtype)
    positions = jnp.asarray(
        [[-0.5 * separation, 0.0, 0.0], [0.5 * separation, 0.0, 0.0]],
        dtype=grid.dtype,
    )
    momenta = jnp.asarray(
        [
            [infall_momentum, tangential_momentum, 0.0],
            [-infall_momentum, -tangential_momentum, 0.0],
        ],
        dtype=grid.dtype,
    )

    psi, u, residual_history = solve_conformal_factor(
        masses,
        positions,
        momenta,
        grid,
        dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=verbose,
    )
    conformal_curvature = bowen_york_extrinsic_curvature(momenta, positions, grid)
    conformal_curvature = jnp.moveaxis(conformal_curvature, (-2, -1), (0, 1))

    W = psi**-2
    shape = W.shape
    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )
    traceless_K = W[None, None, ...] ** 3 * conformal_curvature
    vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=traceless_K,
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )
    return grid, dx, vars, u, residual_history


def run_inspiral_merger(
    mass: float = MASS,
    separation: float = SEPARATION,
    tangential_momentum: float = TANGENTIAL_MOMENTUM,
    infall_momentum: float = INFALL_MOMENTUM,
    half_width: float = HALF_WIDTH,
    num_points: int = NUM_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_interval: int = SNAPSHOT_INTERVAL,
    output_dir: str | Path = "output",
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    show_progress: bool = True,
    save_snapshots: bool = True,
):
    """Solve quasi-circular puncture data and evolve a short merger run."""

    devices = tuple(jax.devices())
    num_devices = len(devices)
    shard_devices = select_shard_devices(num_points, devices)
    if show_progress:
        print(f"JAX active devices ({num_devices}): {devices}")
        if len(shard_devices) > 1 and len(shard_devices) != num_devices:
            print(
                "Sharding inspiral merger state across "
                f"{len(shard_devices)} of {num_devices} devices so Nx={num_points} "
                "divides the x-axis mesh evenly."
            )
        elif num_devices > 1 and not shard_devices:
            print(
                f"Nx={num_points} is not divisible by any multi-GPU device count up to "
                f"{num_devices}; running this case on a single device."
            )

    grid, dx, vars, u, residual_history = binary_bowen_york_data(
        mass,
        separation,
        tangential_momentum,
        infall_momentum,
        half_width,
        num_points,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=show_progress,
    )

    # Shard variables across the largest device subset that evenly tiles Nx.
    if len(shard_devices) > 1:
        mesh = Mesh(np.asarray(shard_devices, dtype=object), axis_names=('x',))
        vars = shard_bssn_variables(vars, mesh)

    target_dt = cfl * dx
    num_steps = max(1, math.ceil(final_time / target_dt)) if target_dt > 0.0 else 1
    dt = final_time / num_steps if final_time > 0.0 else 0.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    constraint_path = output_dir / "constraint_l2.txt"
    residual_path = output_dir / "newton_residual.txt"

    params = BSSNParameters(
        eta=2.0,
        kappa=0.002,
        nu=0.02,
        g=0.75,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=SOMMERFELD_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=SOMMERFELD_BC,
        yr_bc=SOMMERFELD_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-half_width + 0.5 * dx,
        y_min=-half_width + 0.5 * dx,
        z_min=-half_width + 0.5 * dx,
    )

    with residual_path.open("w", encoding="utf-8") as residual_file:
        residual_file.write("# iteration interior_hamiltonian_residual_rms\n")
        for iteration, residual in enumerate(residual_history):
            residual_file.write(f"{iteration:d} {residual:.16e}\n")

    if save_snapshots:
        save_snapshot(vars, time=0.0, step=0, output_dir=output_dir)

    # Rematerialize intermediate stages in RK4 step via jax.checkpoint and donate arg
    advance = jax.jit(
        jax.checkpoint(lambda state: rk4_step(state, params)),
        donate_argnums=0,
    )

    with constraint_path.open("w", encoding="utf-8") as constraint_file:
        constraint_file.write("# time hamiltonian_l2 momentum_l2\n")

        # Initial diagnostic step
        ham_l2, mom_l2, min_W, min_alpha, max_beta = compute_diagnostics_jit(vars, params)
        constraint_file.write(f"0.0000000000000000e+00 {float(ham_l2):.16e} {float(mom_l2):.16e}\n")

        progress = tqdm(
            range(1, num_steps + 1),
            desc="Evolving inspiral merger",
            unit="step",
            disable=not show_progress,
        )
        for step in progress:
            vars = advance(vars)

            time = step * dt

            # Perform JIT diagnostics (keeps operations sharded across all GPUs)
            ham_l2, mom_l2, min_W, min_alpha, max_beta = compute_diagnostics_jit(vars, params)
            constraint_file.write(f"{time:.16e} {float(ham_l2):.16e} {float(mom_l2):.16e}\n")
            constraint_file.flush()

            progress.set_postfix(
                min_W=f"{float(min_W):.6e}",
                min_lapse=f"{float(min_alpha):.6e}",
                max_shift=f"{float(max_beta):.6e}",
            )

            if save_snapshots and (step % snapshot_interval == 0 or step == num_steps):
                save_snapshot(vars, time=time, step=step, output_dir=output_dir)

    final_fields_finite = all(bool(jnp.all(jnp.isfinite(field))) for field in vars)
    return vars, {
        "parameters": params,
        "num_steps": num_steps,
        "final_time": num_steps * dt,
        "grid_shape": tuple(grid.shape[:-1]),
        "elliptic_shape": tuple(u.shape),
        "residual_path": residual_path,
        "constraint_path": constraint_path,
        "final_fields_finite": final_fields_finite,
    }


def main():
    """Build the binary initial data and evolve the inspiral and merger."""

    run_inspiral_merger(
        mass=MASS,
        separation=SEPARATION,
        tangential_momentum=TANGENTIAL_MOMENTUM,
        infall_momentum=INFALL_MOMENTUM,
        half_width=HALF_WIDTH,
        num_points=NUM_POINTS,
        cfl=CFL,
        final_time=FINAL_TIME,
        snapshot_interval=SNAPSHOT_INTERVAL,
        output_dir=OUTPUT_DIR,
        newton_tolerance=NEWTON_TOLERANCE,
        max_newton_iterations=MAX_NEWTON_ITERATIONS,
        cg_tolerance=CG_TOLERANCE,
        max_cg_iterations=MAX_CG_ITERATIONS,
        show_progress=SHOW_PROGRESS,
        save_snapshots=SAVE_SNAPSHOTS,
    )


if __name__ == "__main__":
    main()
