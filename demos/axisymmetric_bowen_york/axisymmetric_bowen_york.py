"""Evolve a z-boosted Bowen-York puncture with axisymmetric Cartoon."""

import argparse
import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.axisymmetry import (
    axisymmetric_plane_output_fields,
    axisymmetric_rk4_step,
    compact_axisymmetric_state,
    compute_axisymmetric_constraint_norms,
    compute_axisymmetric_constraints,
    validate_axisymmetric_grid,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC
from JAX_BSSN.utilities.bowen_york_solver import (
    bowen_york_extrinsic_curvature,
    solve_conformal_factor,
)


MASS = 1.0
MOMENTUM_Z = 0.5
RHO_MAX = 12.0
Z_HALF_WIDTH = 12.0
NUM_RADIAL_POINTS = 48
NUM_Z_POINTS = 96
CFL = 0.2
FINAL_TIME = 10.0
SNAPSHOT_COUNT = 100

NEWTON_TOLERANCE = 1.0e-10
MAX_NEWTON_ITERATIONS = 12
CG_TOLERANCE = 1.0e-10
MAX_CG_ITERATIONS = 1000

KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75


def axisymmetric_parameters(
    num_radial_points: int,
    num_z_points: int,
    rho_max: float,
    z_half_width: float,
    dt: float,
) -> BSSNParameters:
    """Return parameters for the compact positive-rho Cartoon plane."""

    dx = rho_max / num_radial_points
    dz = 2.0 * z_half_width / num_z_points
    if not jnp.isclose(dx, dz):
        raise ValueError("axisymmetric Cartoon requires equal rho and z spacing")
    if num_z_points % 2:
        raise ValueError("num_z_points must be even so the puncture is off-grid")

    z_min = -(num_z_points - 1) * dx / 2.0
    return BSSNParameters(
        eta=ETA,
        kappa=KAPPA,
        nu=NU,
        g=GAMMA_DRIVER,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=SOMMERFELD_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=SOMMERFELD_BC,
        zr_bc=SOMMERFELD_BC,
        x_min=-3.5 * dx,
        y_min=-4.0 * dx,
        z_min=z_min,
        mad_q=1.0,
    )


def bowen_york_cartesian_grid(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
):
    """Return the full 3D elliptic grid and its exact central y-plane index."""

    num_x = 2 * num_radial_points
    num_y = 2 * num_radial_points + 1

    x = (jnp.arange(num_x, dtype=jnp.float64) - (num_x - 1) / 2.0) * dx
    y = (jnp.arange(num_y, dtype=jnp.float64) - (num_y - 1) / 2.0) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
    grid = jnp.stack((X, Y, Z), axis=-1)

    return grid, num_y // 2


def bowen_york_plane_data(
    mass: float,
    momentum_z: float,
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    verbose: bool = False,
):
    """Solve the 3D constraint and return BSSN data on the exact y=0 plane."""

    grid, center_y = bowen_york_cartesian_grid(
        num_radial_points,
        num_z_points,
        dx,
    )
    masses = jnp.asarray([mass], dtype=grid.dtype)
    positions = jnp.zeros((1, 3), dtype=grid.dtype)
    momenta = jnp.asarray([[0.0, 0.0, momentum_z]], dtype=grid.dtype)

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

    # The odd y dimension includes y=0 exactly; x and z remain half-cell
    # centered so the puncture singularity is never sampled.
    psi_plane = psi[:, center_y : center_y + 1, :]
    plane_grid = grid[:, center_y : center_y + 1, :, :]
    conformal_curvature = bowen_york_extrinsic_curvature(
        momenta,
        positions,
        plane_grid,
    )
    conformal_curvature = jnp.moveaxis(conformal_curvature, (-2, -1), (0, 1))

    W = psi_plane**-2
    shape = W.shape
    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )

    # Bowen-York supplies bar(A)_ij. With gamma_ij = psi^4 delta_ij and K=0,
    # the covariant BSSN variable is tilde(A)_ij = psi^-6 bar(A)_ij.
    traceless_K = psi_plane[None, None, ...] ** -6 * conformal_curvature
    plane_vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=traceless_K,
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )

    return plane_vars, u, residual_history


def run_axisymmetric_bowen_york(
    mass: float = MASS,
    momentum_z: float = MOMENTUM_Z,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    output_dir: str | Path = "output",
    newton_tolerance: float = NEWTON_TOLERANCE,
    max_newton_iterations: int = MAX_NEWTON_ITERATIONS,
    cg_tolerance: float = CG_TOLERANCE,
    max_cg_iterations: int = MAX_CG_ITERATIONS,
    show_progress: bool = True,
):
    """Solve and evolve one z-boosted Bowen-York black hole."""

    dx = rho_max / num_radial_points
    num_steps = max(1, math.ceil(final_time / (cfl * dx)))
    dt = final_time / num_steps
    params = axisymmetric_parameters(
        num_radial_points,
        num_z_points,
        rho_max,
        z_half_width,
        dt,
    )
    output_interval = max(1, num_steps // snapshot_count)

    plane_vars, u, residual_history = bowen_york_plane_data(
        mass,
        momentum_z,
        num_radial_points,
        num_z_points,
        dx,
        newton_tolerance=newton_tolerance,
        max_newton_iterations=max_newton_iterations,
        cg_tolerance=cg_tolerance,
        max_cg_iterations=max_cg_iterations,
        verbose=show_progress,
    )
    vars = compact_axisymmetric_state(plane_vars)
    validate_axisymmetric_grid(vars, params)
    jax.block_until_ready(vars)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    openpmd_path = output_dir / "axisymmetric_bowen_york.h5"
    constraint_path = output_dir / "constraint_norms.txt"
    residual_path = output_dir / "newton_residual.txt"

    with residual_path.open("w", encoding="utf-8") as residual_file:
        residual_file.write("# iteration interior_hamiltonian_residual_rms\n")
        for iteration, residual in enumerate(residual_history):
            residual_file.write(f"{iteration:d} {residual:.16e}\n")

    advance = jax.jit(lambda state: axisymmetric_rk4_step(state, params))
    writer = OpenPMDWriter(
        openpmd_path,
        grid_spacing=(dx, dx, dx),
        grid_global_offset=(
            -(num_radial_points - 0.5) * dx,
            0.0,
            params.z_min,
        ),
        grid_position=(0.0, 0.0, 0.0),
        dt=dt,
        ghost_cells=0,
    )
    norm_names = (
        "hamiltonian_l2",
        "hamiltonian_linf",
        "momentum_l2",
        "momentum_linf",
        "det_gamma_l2",
        "det_gamma_linf",
        "trace_A_l2",
        "trace_A_linf",
        "gamma_l2",
        "gamma_linf",
    )

    def diagnose_and_write(state, step, time, constraint_file):
        violations = compute_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations)
        jax.block_until_ready((violations, norms))
        values = " ".join(f"{float(norms[name]):.16e}" for name in norm_names)
        constraint_file.write(f"{time:.16e} {values}\n")
        constraint_file.flush()
        writer.write(
            axisymmetric_plane_output_fields(
                state,
                violations.hamiltonian,
                violations.momentum,
                det_gamma=violations.det_gamma,
                trace_A=violations.trace_A,
                gamma_condition=violations.gamma_condition,
            ),
            step=step,
            time=time,
        )

    try:
        with constraint_path.open("w", encoding="utf-8") as constraint_file:
            constraint_file.write("# time " + " ".join(norm_names) + "\n")
            diagnose_and_write(vars, 0, 0.0, constraint_file)
            progress = tqdm(
                range(1, num_steps + 1),
                desc="Evolving boosted Bowen-York puncture",
                unit="step",
                disable=not show_progress,
            )
            for step in progress:
                vars = advance(vars)
                jax.block_until_ready(vars)
                time = step * dt
                progress.set_postfix(
                    min_W=f"{float(jnp.min(vars.conformal_factor)):.6e}",
                    min_lapse=f"{float(jnp.min(vars.lapse)):.6e}",
                    max_shift=f"{float(jnp.max(jnp.abs(vars.shift))):.6e}",
                )
                if step % output_interval == 0 or step == num_steps:
                    diagnose_and_write(vars, step, time, constraint_file)
    finally:
        writer.close()

    final_fields_finite = all(bool(jnp.all(jnp.isfinite(field))) for field in vars)
    return vars, {
        "parameters": params,
        "num_steps": num_steps,
        "final_time": num_steps * dt,
        "elliptic_shape": tuple(u.shape),
        "newton_residual_history": tuple(residual_history),
        "openpmd_path": openpmd_path,
        "constraint_path": constraint_path,
        "residual_path": residual_path,
        "final_fields_finite": final_fields_finite,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mass", type=float, default=MASS)
    parser.add_argument("--momentum-z", type=float, default=MOMENTUM_Z)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--newton-tolerance", type=float, default=NEWTON_TOLERANCE)
    parser.add_argument(
        "--max-newton-iterations", type=int, default=MAX_NEWTON_ITERATIONS
    )
    parser.add_argument("--cg-tolerance", type=float, default=CG_TOLERANCE)
    parser.add_argument("--max-cg-iterations", type=int, default=MAX_CG_ITERATIONS)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_axisymmetric_bowen_york(
        mass=args.mass,
        momentum_z=args.momentum_z,
        rho_max=args.rho_max,
        z_half_width=args.z_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        output_dir=args.output_dir,
        newton_tolerance=args.newton_tolerance,
        max_newton_iterations=args.max_newton_iterations,
        cg_tolerance=args.cg_tolerance,
        max_cg_iterations=args.max_cg_iterations,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
