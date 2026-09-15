"""Build and briefly evolve a single Schwarzschild puncture."""

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
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.time_evolve import rk4_step


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
        np.save(
            output_dir / f"{field_name}_step_{step:06d}.npy",
            np.asarray(field),
        )


def write_constraint_l2(
    constraint_file,
    vars: BSSNVariables,
    params: BSSNParameters,
    time: float,
) -> None:
    """Append the Hamiltonian and momentum constraint L2 norms."""

    hamiltonian = compute_hamiltonian_constraint(vars, params)
    hamiltonian_l2 = float(jnp.sqrt(jnp.mean(hamiltonian**2)))

    momentum = compute_momentum_constraint(vars, params)
    momentum_l2 = float(jnp.sqrt(jnp.mean(momentum**2)))

    constraint_file.write(
        f"{time:.16e} {hamiltonian_l2:.16e} {momentum_l2:.16e}\n"
    )
    constraint_file.flush()


def single_puncture_data(
    mass: float = 1.0,
    x_wind: float = 32.0,
    y_wind: float = 32.0,
    z_wind: float = 32.0,
    Nx: int = 64,
    Ny: int = 64,
    Nz: int = 64,
) -> tuple[jnp.ndarray, BSSNVariables]:
    """Return the radius and time-symmetric BSSN data for one puncture."""

    dx = x_wind / Nx
    dy = y_wind / Ny
    dz = z_wind / Nz

    if not (math.isclose(dx, dy) and math.isclose(dx, dz)):
        raise ValueError("x_wind/Nx, y_wind/Ny, and z_wind/Nz must be equal")

    # The coordinate endpoints are half a grid cell inside the specified
    # cell-face domain. For an odd cell count, offset the puncture by half a
    # cell so it remains at a vertex shared by eight cells.
    x = (jnp.arange(Nx, dtype=jnp.float64) - (Nx - 1) / 2.0) * dx
    y = (jnp.arange(Ny, dtype=jnp.float64) - (Ny - 1) / 2.0) * dy
    z = (jnp.arange(Nz, dtype=jnp.float64) - (Nz - 1) / 2.0) * dz
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    puncture_x = 0.5 * dx if Nx % 2 else 0.0
    puncture_y = 0.5 * dy if Ny % 2 else 0.0
    puncture_z = 0.5 * dz if Nz % 2 else 0.0
    R = jnp.sqrt(
        (X - puncture_x) ** 2
        + (Y - puncture_y) ** 2
        + (Z - puncture_z) ** 2
    )

    # Time-symmetric Schwarzschild data with W = psi**(-2). The conformal
    # metric is Cartesian and the physical metric is gamma_ij = W**(-2) delta_ij.
    W = (1.0 + mass / (2.0 * R)) ** (-2)
    shape = R.shape

    conformal_metric = (
        jnp.eye(3, dtype=W.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=W.dtype)
    )

    vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=jnp.zeros((3,) + shape, dtype=W.dtype),
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )

    return R, vars


def main():
    """Run a short moving-puncture-style evolution on a Cartesian grid."""

    mass = 1.0
    x_wind = 16.0 * mass
    y_wind = 16.0 * mass
    z_wind = 16.0 * mass
    Nx = 64
    Ny = 64
    Nz = 64
    cfl = 0.25
    final_time = 5.0 * mass
    snapshot_interval = 10
    output_dir = Path("output")
    constraint_path = output_dir / "constraint_l2.txt"

    dx = x_wind / Nx
    dy = y_wind / Ny
    dz = z_wind / Nz
    num_steps = math.ceil(final_time / (cfl * dx))
    dt = final_time / num_steps

    R, vars = single_puncture_data(
        mass=mass,
        x_wind=x_wind,
        y_wind=y_wind,
        z_wind=z_wind,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
    )
    # JAX arrays are immutable, so this retains the exact initial data while
    # ``vars`` is rebound to the projected RK4 states below.
    initial_vars = vars

    # The finite-difference stencils and boundary conditions remain periodic.
    params = BSSNParameters(
        eta=2.0,
        kappa=0.002,
        nu=0.05,
        g=0.75,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
    )

    minimum_radius = float(jnp.min(R))
    puncture_x = 0.5 * dx if Nx % 2 else 0.0
    puncture_y = 0.5 * dy if Ny % 2 else 0.0
    puncture_z = 0.5 * dz if Nz % 2 else 0.0
    expected_minimum_radius = math.sqrt(dx**2 + dy**2 + dz**2) / 2.0
    samples_origin = bool(jnp.any(R == 0.0))
    initial_fields_finite = all(
        bool(jnp.all(jnp.isfinite(field))) for field in initial_vars
    )

    print("Single-puncture black-hole demo")
    print(f"M = {mass:.6f}")
    print(
        "Cell-face domain = "
        f"x:[{-x_wind / 2.0:.6f}, {x_wind / 2.0:.6f}], "
        f"y:[{-y_wind / 2.0:.6f}, {y_wind / 2.0:.6f}], "
        f"z:[{-z_wind / 2.0:.6f}, {z_wind / 2.0:.6f}]"
    )
    print(
        f"Grid = ({Nx}, {Ny}, {Nz}), "
        f"spacing = ({dx:.6f}, {dy:.6f}, {dz:.6f})"
    )
    print(
        "Puncture location = "
        f"({puncture_x:.12f}, {puncture_y:.12f}, {puncture_z:.12f})"
    )
    print(
        f"CFL <= {cfl:.6f}, dt = {dt:.6f}, "
        f"steps = {num_steps}, final time = {final_time:.6f}"
    )
    print(
        f"Snapshots = {output_dir}/ every {snapshot_interval} steps "
        "plus the initial and final states"
    )
    print(f"Constraint history = {constraint_path} every step")
    print(
        f"Minimum R = {minimum_radius:.12f} "
        f"(expected {expected_minimum_radius:.12f})"
    )
    print(f"R=0 sampled: {samples_origin}")
    print(
        "Initial W range = "
        f"[{float(jnp.min(initial_vars.conformal_factor)):.12f}, "
        f"{float(jnp.max(initial_vars.conformal_factor)):.12f}]"
    )
    print(
        "Initial lapse range = "
        f"[{float(jnp.min(initial_vars.lapse)):.12f}, "
        f"{float(jnp.max(initial_vars.lapse)):.12f}]"
    )
    print(f"Initial fields finite: {initial_fields_finite}")

    center_x = Nx // 2
    center_y = Ny // 2
    center_z = Nz // 2
    face_centers = {
        "x-": (0, center_y, center_z),
        "x+": (-1, center_y, center_z),
        "y-": (center_x, 0, center_z),
        "y+": (center_x, -1, center_z),
        "z-": (center_x, center_y, 0),
        "z+": (center_x, center_y, -1),
    }

    print("Initial outer-face values:")
    for face, index in face_centers.items():
        print(
            f"  {face}: W={float(initial_vars.conformal_factor[index]):.12f}, "
            f"lapse={float(initial_vars.lapse[index]):.12f}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_snapshot(vars, time=0.0, step=0, output_dir=output_dir)

    with constraint_path.open("w", encoding="utf-8") as constraint_file:
        constraint_file.write("# time hamiltonian_l2 momentum_l2\n")
        write_constraint_l2(constraint_file, vars, params, time=0.0)

        del R, initial_vars

        progress_bar = tqdm(
            range(1, num_steps + 1),
            desc="Evolving single puncture",
            unit="step",
        )
        for step in progress_bar:
            vars = rk4_step(vars, params)
            jax.block_until_ready(vars)

            time = step * dt
            write_constraint_l2(constraint_file, vars, params, time=time)

            minimum_W = float(jnp.min(vars.conformal_factor))
            minimum_lapse = float(jnp.min(vars.lapse))
            maximum_shift = float(jnp.max(jnp.abs(vars.shift)))
            progress_bar.set_postfix(
                min_W=f"{minimum_W:.6e}",
                min_lapse=f"{minimum_lapse:.6e}",
                max_shift=f"{maximum_shift:.6e}",
            )

            if step % snapshot_interval == 0 or step == num_steps:
                save_snapshot(vars, time=time, step=step, output_dir=output_dir)

    final_fields_finite = all(
        bool(jnp.all(jnp.isfinite(field))) for field in vars
    )

    print("Final outer-face values:")
    for face, index in face_centers.items():
        print(
            f"  {face}: W={float(vars.conformal_factor[index]):.12f}, "
            f"lapse={float(vars.lapse[index]):.12f}"
        )

    print(f"Final fields finite: {final_fields_finite}")

    lapse_slice_1D = vars.lapse[center_x, center_y, :]
    shift_slice_1D = vars.shift[center_x, center_y, :]
    conformal_factor_slice_1D = vars.conformal_factor[center_x, center_y, :]
    grid_z = (jnp.arange(Nz, dtype=jnp.float64) - (Nz - 1) / 2.0) * dz

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.subplot(3, 1, 1)
    plt.plot(grid_z, lapse_slice_1D, label="lapse")
    plt.title("Final lapse along z-axis")
    plt.xlabel("z")
    plt.ylabel("lapse")
    plt.grid()
    plt.subplot(3, 1, 2)
    plt.plot(grid_z, shift_slice_1D, label="shift")
    plt.title("Final shift along z-axis")
    plt.xlabel("z")
    plt.ylabel("shift")
    plt.grid()
    plt.subplot(3, 1, 3)
    plt.plot(grid_z, conformal_factor_slice_1D, label="conformal factor")
    plt.title("Final conformal factor along z-axis")
    plt.xlabel("z")
    plt.ylabel("conformal factor")
    plt.grid()
    plt.tight_layout()
    plt.savefig("single_puncture_final_fields.png")


if __name__ == "__main__":
    main()
