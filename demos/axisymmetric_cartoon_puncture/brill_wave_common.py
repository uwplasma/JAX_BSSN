"""Shared helpers for axisymmetric gravitational-wave demos."""

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


KAPPA = 0.002
ETA = 2.0
NU = 0.02
GAMMA_DRIVER = 0.75

BRILL_TOLERANCE = 1.0e-10
BRILL_MAX_ITERATIONS = 20000
BRILL_RELAXATION = 0.8


def axisymmetric_parameters(
    num_radial_points: int,
    num_z_points: int,
    rho_max: float,
    z_half_width: float,
    dt: float,
    zero_shift: int = 0,
) -> BSSNParameters:
    """Return parameters for the compact positive-rho Cartoon plane."""

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
        dt=dt,
        zero_shift=zero_shift,
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


def positive_rho_z_grid(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return cell-centered positive-rho and signed-z arrays."""

    rho = (jnp.arange(num_radial_points, dtype=jnp.float64) + 0.5) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    Rho, Z = jnp.meshgrid(rho, z, indexing="ij")
    return rho, z, Rho, Z


def brill_wave_profile(
    rho: jnp.ndarray,
    z: jnp.ndarray,
    amplitude: float,
    rho_width: float,
    z_width: float,
    center_rho: float = 0.0,
    center_z: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return q, dq/drho, dq/dz, and q_rr + q_zz for a Brill pulse."""

    if rho_width <= 0.0 or z_width <= 0.0:
        raise ValueError("Brill-wave widths must be positive")

    shifted_rho = rho - center_rho
    shifted_z = z - center_z
    u = shifted_rho**2 / rho_width**2
    v = shifted_z**2 / z_width**2
    envelope = jnp.exp(-(u + v))
    q = amplitude * u * envelope

    dq_drho = (
        2.0
        * amplitude
        * shifted_rho
        / rho_width**2
        * envelope
        * (1.0 - u)
    )
    dq_dz = -2.0 * shifted_z / z_width**2 * q

    d2q_drho2 = (
        2.0
        * amplitude
        / rho_width**2
        * envelope
        * (1.0 - 5.0 * u + 2.0 * u**2)
    )
    d2q_dz2 = q * (4.0 * shifted_z**2 / z_width**4 - 2.0 / z_width**2)

    return q, dq_drho, dq_dz, d2q_drho2 + d2q_dz2


def solve_brill_conformal_factor(
    q_laplacian: jnp.ndarray,
    dx: float,
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    boundary_value: float = 1.0,
    verbose: bool = False,
) -> tuple[jnp.ndarray, list[float]]:
    """Solve the time-symmetric Brill Hamiltonian constraint by relaxation."""

    num_radial_points, _ = q_laplacian.shape
    rho = (jnp.arange(num_radial_points, dtype=q_laplacian.dtype) + 0.5) * dx
    rho_east = rho + 0.5 * dx
    rho_west = rho - 0.5 * dx

    radial_east = rho_east[:, None] / (rho[:, None] * dx**2)
    radial_west = rho_west[:, None] / (rho[:, None] * dx**2)
    axial = jnp.full_like(q_laplacian, 1.0 / dx**2)
    center = -(radial_east + radial_west + 2.0 * axial) + 0.25 * q_laplacian

    psi = jnp.ones_like(q_laplacian) * boundary_value
    residual_history = []

    for iteration in range(max_iterations):
        padded = jnp.pad(
            psi,
            ((1, 1), (1, 1)),
            mode="constant",
            constant_values=boundary_value,
        )
        east = padded[2:, 1:-1]
        west = padded[:-2, 1:-1]
        north = padded[1:-1, 2:]
        south = padded[1:-1, :-2]

        neighbor_sum = (
            radial_east * east
            + radial_west * west
            + axial * north
            + axial * south
        )
        updated = jnp.maximum(-neighbor_sum / center, 1.0e-12)
        psi_next = relaxation * updated + (1.0 - relaxation) * psi

        delta = float(jnp.max(jnp.abs(psi_next - psi)))
        residual_history.append(delta)
        psi = psi_next

        if verbose and (iteration == 0 or (iteration + 1) % 100 == 0):
            print(
                "Brill solve iteration "
                f"{iteration + 1:5d}: max update = {delta:.3e}"
            )
        if delta <= tolerance:
            break
    else:
        raise RuntimeError(
            "Brill-wave Hamiltonian solve did not converge: "
            f"max update {delta:.3e} exceeds {tolerance:.3e}."
        )

    return psi, residual_history


def _reflect_even(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate((values[::-1, :], values), axis=0)


def _reflect_odd(values: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate((-values[::-1, :], values), axis=0)


def build_brill_wave_plane_data(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    amplitude: float,
    rho_width: float,
    z_width: float,
    center_rho: float = 0.0,
    center_z: float = 0.0,
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    verbose: bool = False,
) -> tuple[BSSNVariables, dict[str, object]]:
    """Return axisymmetric Brill-wave initial data on the signed x-z plane."""

    _, _, rho_grid, z_grid = positive_rho_z_grid(
        num_radial_points, num_z_points, dx
    )
    q, dq_drho, dq_dz, q_laplacian = brill_wave_profile(
        rho_grid,
        z_grid,
        amplitude,
        rho_width,
        z_width,
        center_rho=center_rho,
        center_z=center_z,
    )
    psi, residual_history = solve_brill_conformal_factor(
        q_laplacian,
        dx,
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        verbose=verbose,
    )

    W_positive = psi**-2 * jnp.exp(-2.0 * q / 3.0)
    gamma_xx_positive = jnp.exp(2.0 * q / 3.0)
    gamma_yy_positive = jnp.exp(-4.0 * q / 3.0)
    gamma_zz_positive = gamma_xx_positive
    gamma_x_positive = 2.0 / 3.0 * jnp.exp(-2.0 * q / 3.0) * dq_drho
    gamma_z_positive = 2.0 / 3.0 * jnp.exp(-2.0 * q / 3.0) * dq_dz

    W = _reflect_even(W_positive)[:, None, :]
    gamma_xx = _reflect_even(gamma_xx_positive)[:, None, :]
    gamma_yy = _reflect_even(gamma_yy_positive)[:, None, :]
    gamma_zz = _reflect_even(gamma_zz_positive)[:, None, :]
    Gamma_x = _reflect_odd(gamma_x_positive)[:, None, :]
    Gamma_z = _reflect_even(gamma_z_positive)[:, None, :]

    shape = W.shape
    conformal_metric = jnp.zeros((3, 3) + shape, dtype=W.dtype)
    conformal_metric = conformal_metric.at[0, 0].set(gamma_xx)
    conformal_metric = conformal_metric.at[1, 1].set(gamma_yy)
    conformal_metric = conformal_metric.at[2, 2].set(gamma_zz)

    conformal_connection = jnp.zeros((3,) + shape, dtype=W.dtype)
    conformal_connection = conformal_connection.at[0].set(Gamma_x)
    conformal_connection = conformal_connection.at[2].set(Gamma_z)

    plane_vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=conformal_connection,
        lapse=jnp.ones(shape, dtype=W.dtype),
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )

    return plane_vars, {
        "residual_history": tuple(residual_history),
        "elliptic_shape": tuple(psi.shape),
        "max_abs_q": float(jnp.max(jnp.abs(q))),
        "min_psi": float(jnp.min(psi)),
        "center_rho": center_rho,
    }


def build_distorted_puncture_plane_data(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    mass: float,
    amplitude: float,
    rho_width: float,
    z_width: float,
    center_rho: float = 0.0,
    center_z: float = 0.0,
) -> tuple[BSSNVariables, dict[str, object]]:
    """Return a puncture with a superposed Brill-wave distortion."""

    x = (jnp.arange(2 * num_radial_points, dtype=jnp.float64) - num_radial_points + 0.5) * dx
    z = (
        jnp.arange(num_z_points, dtype=jnp.float64)
        - (num_z_points - 1) / 2.0
    ) * dx
    rho = jnp.abs(x)[:, None]
    z_grid = z[None, :]

    q, dq_drho, dq_dz, _ = brill_wave_profile(
        rho,
        z_grid,
        amplitude,
        rho_width,
        z_width,
        center_rho=center_rho,
        center_z=center_z,
    )
    radius = jnp.sqrt(rho**2 + z_grid**2)

    W = (1.0 + mass / (2.0 * radius)) ** -2 * jnp.exp(-2.0 * q / 3.0)
    gamma_xx = jnp.exp(2.0 * q / 3.0)
    gamma_yy = jnp.exp(-4.0 * q / 3.0)
    gamma_zz = gamma_xx
    sign_x = jnp.sign(x)[:, None]
    Gamma_x = sign_x * (2.0 / 3.0) * jnp.exp(-2.0 * q / 3.0) * dq_drho
    Gamma_z = (2.0 / 3.0) * jnp.exp(-2.0 * q / 3.0) * dq_dz

    W = W[:, None, :]
    gamma_xx = gamma_xx[:, None, :]
    gamma_yy = gamma_yy[:, None, :]
    gamma_zz = gamma_zz[:, None, :]
    Gamma_x = Gamma_x[:, None, :]
    Gamma_z = Gamma_z[:, None, :]

    shape = W.shape
    conformal_metric = jnp.zeros((3, 3) + shape, dtype=W.dtype)
    conformal_metric = conformal_metric.at[0, 0].set(gamma_xx)
    conformal_metric = conformal_metric.at[1, 1].set(gamma_yy)
    conformal_metric = conformal_metric.at[2, 2].set(gamma_zz)

    conformal_connection = jnp.zeros((3,) + shape, dtype=W.dtype)
    conformal_connection = conformal_connection.at[0].set(Gamma_x)
    conformal_connection = conformal_connection.at[2].set(Gamma_z)

    plane_vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=W,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=W.dtype),
        conformal_connection=conformal_connection,
        lapse=W,
        shift=jnp.zeros((3,) + shape, dtype=W.dtype),
    )

    return plane_vars, {
        "approximate_superposition": True,
        "max_abs_q": float(jnp.max(jnp.abs(q))),
        "min_radius": float(jnp.min(radius)),
        "center_rho": center_rho,
    }


def evolve_axisymmetric_demo(
    plane_vars: BSSNVariables,
    *,
    rho_max: float,
    z_half_width: float,
    num_radial_points: int,
    num_z_points: int,
    cfl: float,
    final_time: float,
    snapshot_count: int,
    output_dir: str | Path,
    output_stem: str,
    progress_label: str,
    zero_shift: int = 0,
    show_progress: bool = True,
) -> tuple[BSSNVariables, dict[str, object]]:
    """Evolve a signed x-z reference plane with the axisymmetric Cartoon code."""

    dx = rho_max / num_radial_points
    target_dt = cfl * dx
    num_steps = max(1, math.ceil(final_time / target_dt)) if target_dt > 0.0 else 1
    dt = final_time / num_steps if final_time > 0.0 else 0.0
    params = axisymmetric_parameters(
        num_radial_points,
        num_z_points,
        rho_max,
        z_half_width,
        dt,
        zero_shift=zero_shift,
    )
    output_interval = max(1, num_steps // max(snapshot_count, 1))

    vars = compact_axisymmetric_state(plane_vars)
    validate_axisymmetric_grid(vars, params)
    jax.block_until_ready(vars)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    openpmd_path = output_dir / f"{output_stem}.h5"
    constraint_path = output_dir / f"{output_stem}_constraint_norms.txt"

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
    advance = jax.jit(lambda state: axisymmetric_rk4_step(state, params))
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
                desc=progress_label,
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