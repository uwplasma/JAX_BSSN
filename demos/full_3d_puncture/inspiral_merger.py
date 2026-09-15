"""Build quasi-circular Bowen-York binary data and evolve a merger scaffold."""

import argparse
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

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
NUM_POINTS = 48
CFL = 0.15
FINAL_TIME = 20.0
SNAPSHOT_INTERVAL = 10

NEWTON_TOLERANCE = 1.0e-10
MAX_NEWTON_ITERATIONS = 12
CG_TOLERANCE = 1.0e-10
MAX_CG_ITERATIONS = 1000


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


def write_constraint_l2(
    constraint_file,
    vars: BSSNVariables,
    params: BSSNParameters,
    time: float,
) -> None:
    """Append the Hamiltonian and momentum constraint L2 norms."""

    hamiltonian = compute_hamiltonian_constraint(vars, params)
    momentum = compute_momentum_constraint(vars, params)
    hamiltonian_l2 = float(jnp.sqrt(jnp.mean(hamiltonian**2)))
    momentum_l2 = float(jnp.sqrt(jnp.mean(momentum**2)))
    constraint_file.write(
        f"{time:.16e} {hamiltonian_l2:.16e} {momentum_l2:.16e}\n"
    )
    constraint_file.flush()


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
):
    """Solve quasi-circular puncture data and evolve a short merger run."""

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

    save_snapshot(vars, time=0.0, step=0, output_dir=output_dir)
    advance = jax.jit(lambda state: rk4_step(state, params))

    with constraint_path.open("w", encoding="utf-8") as constraint_file:
        constraint_file.write("# time hamiltonian_l2 momentum_l2\n")
        write_constraint_l2(constraint_file, vars, params, time=0.0)
        progress = tqdm(
            range(1, num_steps + 1),
            desc="Evolving inspiral merger",
            unit="step",
            disable=not show_progress,
        )
        for step in progress:
            vars = advance(vars)
            jax.block_until_ready(vars)

            time = step * dt
            write_constraint_l2(constraint_file, vars, params, time=time)
            progress.set_postfix(
                min_W=f"{float(jnp.min(vars.conformal_factor)):.6e}",
                min_lapse=f"{float(jnp.min(vars.lapse)):.6e}",
                max_shift=f"{float(jnp.max(jnp.abs(vars.shift))):.6e}",
            )

            if step % snapshot_interval == 0 or step == num_steps:
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


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=MASS)
    parser.add_argument("--separation", type=float, default=SEPARATION)
    parser.add_argument("--tangential-momentum", type=float, default=TANGENTIAL_MOMENTUM)
    parser.add_argument("--infall-momentum", type=float, default=INFALL_MOMENTUM)
    parser.add_argument("--half-width", type=float, default=HALF_WIDTH)
    parser.add_argument("--num-points", type=int, default=NUM_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshot-interval", type=int, default=SNAPSHOT_INTERVAL)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--newton-tolerance", type=float, default=NEWTON_TOLERANCE)
    parser.add_argument("--max-newton-iterations", type=int, default=MAX_NEWTON_ITERATIONS)
    parser.add_argument("--cg-tolerance", type=float, default=CG_TOLERANCE)
    parser.add_argument("--max-cg-iterations", type=int, default=MAX_CG_ITERATIONS)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_inspiral_merger(
        mass=args.mass,
        separation=args.separation,
        tangential_momentum=args.tangential_momentum,
        infall_momentum=args.infall_momentum,
        half_width=args.half_width,
        num_points=args.num_points,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_interval=args.snapshot_interval,
        output_dir=args.output_dir,
        newton_tolerance=args.newton_tolerance,
        max_newton_iterations=args.max_newton_iterations,
        cg_tolerance=args.cg_tolerance,
        max_cg_iterations=args.max_cg_iterations,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()