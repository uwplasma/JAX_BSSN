"""Evolve a Schwarzschild puncture with z-axis axisymmetric Cartoon."""

import argparse
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


MASS = 1.0
RHO_MAX = 50.0
Z_HALF_WIDTH = 25.0
NUM_RADIAL_POINTS = 512
NUM_Z_POINTS = 512
CFL = 0.2
FINAL_TIME = 100.0
SNAPSHOT_COUNT = 500

KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75


def axisymmetric_parameters(
    num_radial_points: int,
    num_z_points: int,
    rho_max: float,
    z_half_width: float,
    cfl: float,
) -> BSSNParameters:
    """Return the compact half-plane grid and puncture parameters."""

    dx = rho_max / num_radial_points
    dz = 2.0 * z_half_width / num_z_points
    if not jnp.isclose(dx, dz):
        raise ValueError("axisymmetric Cartoon requires equal rho and z spacing")
    z_min = -(num_z_points - 1) * dx / 2.0
    return BSSNParameters(
        eta=ETA,
        kappa=KAPPA,
        nu=NU,
        g=GAMMA_DRIVER,
        dx=dx,
        dt=cfl * dx,
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


def schwarzschild_plane_data(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    mass: float = 1.0,
) -> BSSNVariables:
    """Return puncture data on the complete signed, cell-centred x-z plane."""

    num_x = 2 * num_radial_points
    x = (jnp.arange(num_x, dtype=jnp.float64) - (num_x - 1) / 2.0) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    radius = jnp.sqrt(x[:, None] ** 2 + z[None, :] ** 2)[:, None, :]
    W = (1.0 + mass / (2.0 * radius)) ** -2
    shape = W.shape
    metric = jnp.eye(3, dtype=W.dtype)[:, :, None, None, None] * jnp.ones(
        (3, 3) + shape, dtype=W.dtype
    )
    return BSSNVariables(
        conformal_metric=metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )


def run_axisymmetric_cartoon_puncture(
    mass: float = MASS,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    output_dir: str | Path = "output",
    show_progress: bool = True,
):
    """Run the axisymmetric puncture and diagnose only written iterations."""

    params = axisymmetric_parameters(
        num_radial_points, num_z_points, rho_max, z_half_width, cfl
    )
    dx, dt = params.dx, params.dt
    num_steps = max(1, int(round(final_time / dt)))
    output_interval = max(1, num_steps // snapshot_count)
    output_dir = Path(output_dir)
    openpmd_path = output_dir / "axisymmetric_cartoon_puncture.h5"
    constraint_path = output_dir / "axisymmetric_constraint_l2.txt"

    vars = compact_axisymmetric_state(
        schwarzschild_plane_data(
            num_radial_points, num_z_points, dx, mass
        )
    )
    validate_axisymmetric_grid(vars, params)
    jax.block_until_ready(vars)
    output_dir.mkdir(parents=True, exist_ok=True)
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

    def diagnose_and_write(state, step, time, constraint_file):
        violations = compute_axisymmetric_constraints(state, params)
        norms = compute_axisymmetric_constraint_norms(violations)
        jax.block_until_ready((violations, norms))
        constraint_file.write(
            f"{time:.16e} {float(norms['hamiltonian_l2']):.16e} "
            f"{float(norms['momentum_l2']):.16e}\n"
        )
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
            constraint_file.write("# time hamiltonian_l2 momentum_l2\n")
            diagnose_and_write(vars, 0, 0.0, constraint_file)
            progress = tqdm(
                range(1, num_steps + 1),
                desc="Evolving axisymmetric Cartoon puncture",
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
        "openpmd_path": openpmd_path,
        "constraint_path": constraint_path,
        "final_fields_finite": final_fields_finite,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_axisymmetric_cartoon_puncture(
        rho_max=args.rho_max,
        z_half_width=args.z_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        output_dir=args.output_dir,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
