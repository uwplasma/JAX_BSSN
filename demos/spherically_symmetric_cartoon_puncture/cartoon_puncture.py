"""Evolve a Schwarzschild puncture with spherical Cartoon reconstruction."""

from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.cartoon.spherical_symmetry import (
    CARTOON_GHOST_CELLS,
    cartoon_axis_output_fields,
    cartoon_rk4_step,
    compact_cartoon_state,
    compute_cartoon_constraint_norms,
    compute_cartoon_constraints,
    validate_cartoon_grid,
)
from JAX_BSSN.diagnostics.openpmd import OpenPMDWriter
from JAX_BSSN.evolution.boundaries import PERIODIC_BC, SOMMERFELD_BC


MASS = 1.0
R_MAX = 100.0
NUM_RADIAL_POINTS = 4000

CFL = 0.2
FINAL_TIME = 100.0
SNAPSHOT_COUNT = 500

KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75


def cartoon_parameters(
    num_radial_points: int,
    r_max: float,
    cfl: float,
) -> BSSNParameters:
    """Return the compact grid and moving-puncture evolution parameters."""

    dx = r_max / num_radial_points

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
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=-(CARTOON_GHOST_CELLS - 0.5) * dx,
        y_min=-CARTOON_GHOST_CELLS * dx,
        z_min=-CARTOON_GHOST_CELLS * dx,
        mad_q=1.0,
    )


def schwarzschild_axis_data(
    num_radial_points: int,
    dx: float,
    mass: float = 1.0,
) -> BSSNVariables:
    """Return time-symmetric puncture data on an even signed x-axis."""

    num_x = 2 * num_radial_points
    x = (
        jnp.arange(num_x, dtype=jnp.float64) - (num_x - 1) / 2.0
    ) * dx
    radius = jnp.abs(x)[:, None, None]
    W = (1.0 + mass / (2.0 * radius)) ** -2
    shape = W.shape

    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )


def run_cartoon_puncture(
    mass: float = MASS,
    r_max: float = R_MAX,
    num_radial_points: int = NUM_RADIAL_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    output_dir: str | Path = "output",
    show_progress: bool = True,
):
    """Run the spherical Cartoon puncture evolution and write diagnostics."""

    params = cartoon_parameters(num_radial_points, r_max, cfl)
    dx = params.dx
    dt = params.dt
    num_steps = max(1, int(round(final_time / dt)))
    output_interval = max(1, num_steps // snapshot_count)

    output_dir = Path(output_dir)
    openpmd_path = output_dir / "cartoon_puncture.h5"
    constraint_path = output_dir / "constraint_l2.txt"

    vars = compact_cartoon_state(
        schwarzschild_axis_data(num_radial_points, dx, mass)
    )
    validate_cartoon_grid(vars, params)
    jax.block_until_ready(vars)

    output_dir.mkdir(parents=True, exist_ok=True)
    advance = jax.jit(lambda state: cartoon_rk4_step(state, params))

    writer = OpenPMDWriter(
        openpmd_path,
        grid_spacing=(dx, dx, dx),
        grid_global_offset=(
            -(num_radial_points - 0.5) * dx,
            0.0,
            0.0,
        ),
        grid_position=(0.0, 0.0, 0.0),
        dt=dt,
        ghost_cells=0,
    )

    try:
        with constraint_path.open("w", encoding="utf-8") as constraint_file:
            constraint_file.write("# time hamiltonian_l2 momentum_l2\n")

            violations = compute_cartoon_constraints(vars, params)
            norms = compute_cartoon_constraint_norms(violations)
            jax.block_until_ready((violations, norms))
            constraint_file.write(
                f"{0.0:.16e} {float(norms['hamiltonian_l2']):.16e} "
                f"{float(norms['momentum_l2']):.16e}\n"
            )
            writer.write(
                cartoon_axis_output_fields(
                    vars,
                    violations.hamiltonian,
                    violations.momentum,
                    det_gamma=violations.det_gamma,
                    trace_A=violations.trace_A,
                    gamma_condition=violations.gamma_condition,
                ),
                step=0,
                time=0.0,
            )

            progress = tqdm(
                range(1, num_steps + 1),
                desc="Evolving Cartoon puncture",
                unit="step",
                disable=not show_progress,
            )
            for step in progress:
                vars = advance(vars)
                jax.block_until_ready(vars)

                time = step * dt
                violations = compute_cartoon_constraints(vars, params)
                norms = compute_cartoon_constraint_norms(violations)
                jax.block_until_ready((violations, norms))
                constraint_file.write(
                    f"{time:.16e} "
                    f"{float(norms['hamiltonian_l2']):.16e} "
                    f"{float(norms['momentum_l2']):.16e}\n"
                )
                constraint_file.flush()

                progress.set_postfix(
                    min_W=f"{float(jnp.min(vars.conformal_factor)):.6e}",
                    min_lapse=f"{float(jnp.min(vars.lapse)):.6e}",
                    max_shift=f"{float(jnp.max(jnp.abs(vars.shift))):.6e}",
                )

                if step % output_interval == 0 or step == num_steps:
                    writer.write(
                        cartoon_axis_output_fields(
                            vars,
                            violations.hamiltonian,
                            violations.momentum,
                            det_gamma=violations.det_gamma,
                            trace_A=violations.trace_A,
                            gamma_condition=violations.gamma_condition,
                        ),
                        step=step,
                        time=time,
                    )
    finally:
        writer.close()

    final_fields_finite = all(
        bool(jnp.all(jnp.isfinite(field))) for field in vars
    )
    print("Spherical Cartoon single-puncture demo")
    print(
        f"Nr={num_radial_points}, dx={dx:.8f}, dt={dt:.8f}, "
        f"steps={num_steps}, final time={num_steps * dt:.8f}"
    )
    print(f"Final fields finite: {final_fields_finite}")

    return vars, {
        "parameters": params,
        "num_steps": num_steps,
        "final_time": num_steps * dt,
        "openpmd_path": openpmd_path,
        "constraint_path": constraint_path,
        "final_fields_finite": final_fields_finite,
    }


def main():
    run_cartoon_puncture()


if __name__ == "__main__":
    main()
