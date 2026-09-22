"""Evolve a robust axisymmetric Teukolsky-wave collapse case."""

import argparse
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from JAX_BSSN.bssn.constraints import compute_hamiltonian_constraint
from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
)
from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.evolution.derivatives import diff1_field

from brill_wave_common import (
    BRILL_MAX_ITERATIONS,
    BRILL_RELAXATION,
    BRILL_TOLERANCE,
    _reflect_even,
    _reflect_odd,
    axisymmetric_parameters,
    evolve_axisymmetric_demo,
    positive_rho_z_grid,
    solve_brill_conformal_factor,
)


AMPLITUDE = 0.0018
RHO_MAX = 12.0
Z_HALF_WIDTH = 12.0
NUM_RADIAL_POINTS = 192
NUM_Z_POINTS = 384
CFL = 0.2
FINAL_TIME = 40.0
SNAPSHOT_COUNT = 160

WIDTH = 0.5
CENTER_RADIUS = 2.0


def _teukolsky_seed(radius, amplitude, width, center_radius):
    gaussian_plus = jnp.exp(-((radius + center_radius) / width) ** 2)
    gaussian_minus = jnp.exp(-((radius - center_radius) / width) ** 2)
    return -amplitude * radius / width * (gaussian_plus + gaussian_minus)


def _teukolsky_derivatives(radius, amplitude, width, center_radius):
    def scalar_seed(r):
        return _teukolsky_seed(r, amplitude, width, center_radius)

    first = jax.grad(scalar_seed)
    second = jax.grad(first)
    third = jax.grad(second)
    fourth = jax.grad(third)
    radii = radius.reshape(-1)
    values = jax.vmap(
        lambda r: jnp.asarray(
            [scalar_seed(r), first(r), second(r), third(r), fourth(r)],
            dtype=radius.dtype,
        )
    )(radii)
    return tuple(values[:, index].reshape(radius.shape) for index in range(5))


def _teukolsky_positive_conformal_metric(
    rho: jnp.ndarray,
    z: jnp.ndarray,
    amplitude: float,
    width: float,
    center_radius: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return the positive-rho Teukolsky conformal metric components."""

    radius = jnp.sqrt(rho**2 + z**2)
    sin_theta = rho / radius
    cos_theta = z / radius

    seed, first, second, third, fourth = _teukolsky_derivatives(
        radius,
        amplitude,
        width,
        center_radius,
    )
    radius2 = radius**2
    radius3 = radius**3
    radius4 = radius**4
    radius5 = radius**5

    radial_a = 3.0 * (second / radius3 + 3.0 * first / radius4 + 3.0 * seed / radius5)
    radial_b = -(
        third / radius2
        + 3.0 * second / radius3
        + 6.0 * first / radius4
        + 6.0 * seed / radius5
    )
    radial_c = 0.25 * (
        fourth / radius
        + 2.0 * third / radius2
        + 9.0 * second / radius3
        + 21.0 * first / radius4
        + 21.0 * seed / radius5
    )

    sin2 = sin_theta**2
    cos2 = cos_theta**2
    sin_cos = sin_theta * cos_theta

    f_rr = 2.0 - 3.0 * sin2
    f_rtheta = -3.0 * sin_cos
    f_theta_theta_1 = 3.0 * sin2
    f_phi_phi_1 = -3.0 * sin2
    f_theta_theta_2 = -1.0
    f_phi_phi_2 = 3.0 * sin2 - 1.0

    gamma_rr = 1.0 + radial_a * f_rr
    gamma_rtheta = radius * radial_b * f_rtheta
    gamma_theta_theta = radius2 * (
        1.0 + radial_c * f_theta_theta_1 + radial_a * f_theta_theta_2
    )
    gamma_phi_phi = radius2 * sin2 * (
        1.0 + radial_c * f_phi_phi_1 + radial_a * f_phi_phi_2
    )

    gamma_rho_rho = (
        gamma_rr * sin2
        + 2.0 * gamma_rtheta * sin_cos / radius
        + gamma_theta_theta * cos2 / radius2
    )
    gamma_zz = (
        gamma_rr * cos2
        - 2.0 * gamma_rtheta * sin_cos / radius
        + gamma_theta_theta * sin2 / radius2
    )
    gamma_rho_z = (
        gamma_rr * sin_cos
        + gamma_rtheta * (cos2 - sin2) / radius
        - gamma_theta_theta * sin_cos / radius2
    )
    gamma_yy = gamma_phi_phi / rho**2

    determinant = gamma_yy * (gamma_rho_rho * gamma_zz - gamma_rho_z**2)
    scale = determinant ** (-1.0 / 3.0)
    return (
        gamma_rho_rho * scale,
        gamma_rho_z * scale,
        gamma_yy * scale,
        gamma_zz * scale,
        seed,
    )


def _contracted_christoffel(conformal_metric, params):
    metric_derivatives = jnp.stack(
        [
            diff1_field(
                conformal_metric,
                direction + 2,
                params.dx,
                *((params.xl_bc, params.xr_bc) if direction == 0 else (params.yl_bc, params.yr_bc) if direction == 1 else (params.zl_bc, params.zr_bc)),
                mad_q=params.mad_q,
            )
            for direction in range(3)
        ],
        axis=0,
    )
    inverse_metric = invert_3x3_metric(conformal_metric)
    christoffel = christoffel_symbols_second_kind(inverse_metric, metric_derivatives)
    return jnp.einsum("jk...,ijk...->i...", inverse_metric, christoffel)


def build_teukolsky_wave_plane_data(
    num_radial_points: int,
    num_z_points: int,
    dx: float,
    amplitude: float,
    width: float,
    center_radius: float,
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    verbose: bool = False,
) -> tuple[BSSNVariables, dict[str, object]]:
    """Return axisymmetric quadrupolar Teukolsky-wave data on the signed x-z plane."""

    _, _, rho_grid, z_grid = positive_rho_z_grid(
        num_radial_points, num_z_points, dx
    )
    gamma_rho_rho_pos, gamma_rho_z_pos, gamma_yy_pos, gamma_zz_pos, seed = (
        _teukolsky_positive_conformal_metric(
            rho_grid,
            z_grid,
            amplitude,
            width,
            center_radius,
        )
    )

    gamma_xx = _reflect_even(gamma_rho_rho_pos)[:, None, :]
    gamma_xz = _reflect_odd(gamma_rho_z_pos)[:, None, :]
    gamma_yy = _reflect_even(gamma_yy_pos)[:, None, :]
    gamma_zz = _reflect_even(gamma_zz_pos)[:, None, :]
    shape = gamma_xx.shape

    conformal_metric = jnp.zeros((3, 3) + shape, dtype=gamma_xx.dtype)
    conformal_metric = conformal_metric.at[0, 0].set(gamma_xx)
    conformal_metric = conformal_metric.at[0, 2].set(gamma_xz)
    conformal_metric = conformal_metric.at[1, 1].set(gamma_yy)
    conformal_metric = conformal_metric.at[2, 0].set(gamma_xz)
    conformal_metric = conformal_metric.at[2, 2].set(gamma_zz)

    rho_max = num_radial_points * dx
    z_half_width = 0.5 * num_z_points * dx
    provisional_params = axisymmetric_parameters(
        num_radial_points,
        num_z_points,
        rho_max,
        z_half_width,
        0.0,
        zero_shift=1,
    )
    conformal_connection = _contracted_christoffel(conformal_metric, provisional_params)
    provisional_vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=jnp.ones(shape, dtype=gamma_xx.dtype),
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=gamma_xx.dtype),
        conformal_connection=conformal_connection,
        lapse=jnp.ones(shape, dtype=gamma_xx.dtype),
        shift=jnp.zeros((3,) + shape, dtype=gamma_xx.dtype),
    )
    hamiltonian = compute_hamiltonian_constraint(provisional_vars, provisional_params)
    psi_positive, residual_history = solve_brill_conformal_factor(
        -0.5 * hamiltonian[num_radial_points:, 0, :],
        dx,
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        verbose=verbose,
    )
    conformal_factor = _reflect_even(psi_positive**-2)[:, None, :]
    plane_vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=jnp.zeros_like(conformal_metric),
        trace_K=jnp.zeros(shape, dtype=gamma_xx.dtype),
        conformal_connection=conformal_connection,
        lapse=jnp.ones(shape, dtype=gamma_xx.dtype),
        shift=jnp.zeros((3,) + shape, dtype=gamma_xx.dtype),
    )
    return plane_vars, {
        "residual_history": tuple(residual_history),
        "elliptic_shape": tuple(psi_positive.shape),
        "approximate_flat_laplacian_constraint_solve": True,
        "max_abs_seed": float(jnp.max(jnp.abs(seed))),
        "max_abs_pre_solve_hamiltonian": float(jnp.max(jnp.abs(hamiltonian))),
        "min_psi": float(jnp.min(psi_positive)),
        "center_radius": center_radius,
        "width": width,
    }


def run_teukolsky_wave_common(
    amplitude: float = AMPLITUDE,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    width: float = WIDTH,
    center_radius: float = CENTER_RADIUS,
    output_dir: str | Path = "output",
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    show_progress: bool = True,
):
    """Build and evolve one axisymmetric Teukolsky wave with zero shift."""

    dx = rho_max / num_radial_points
    plane_vars, initial_data = build_teukolsky_wave_plane_data(
        num_radial_points,
        num_z_points,
        dx,
        amplitude,
        width,
        center_radius,
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        verbose=show_progress,
    )
    vars, evolution = evolve_axisymmetric_demo(
        plane_vars,
        rho_max=rho_max,
        z_half_width=z_half_width,
        num_radial_points=num_radial_points,
        num_z_points=num_z_points,
        cfl=cfl,
        final_time=final_time,
        snapshot_count=snapshot_count,
        output_dir=output_dir,
        output_stem="teukolsky_wave_common",
        progress_label="Evolving Teukolsky wave",
        zero_shift=1,
        show_progress=show_progress,
    )

    residual_path = Path(output_dir) / "teukolsky_wave_initial_residual.txt"
    with residual_path.open("w", encoding="utf-8") as residual_file:
        residual_file.write("# iteration max_update\n")
        for iteration, residual in enumerate(initial_data["residual_history"]):
            residual_file.write(f"{iteration:d} {residual:.16e}\n")

    return vars, {
        **initial_data,
        **evolution,
        "amplitude": amplitude,
        "width": width,
        "center_radius": center_radius,
        "residual_path": residual_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--width", type=float, default=WIDTH)
    parser.add_argument("--center-radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tolerance", type=float, default=BRILL_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=BRILL_MAX_ITERATIONS)
    parser.add_argument("--relaxation", type=float, default=BRILL_RELAXATION)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_teukolsky_wave_common(
        amplitude=args.amplitude,
        rho_max=args.rho_max,
        z_half_width=args.z_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        width=args.width,
        center_radius=args.center_radius,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()