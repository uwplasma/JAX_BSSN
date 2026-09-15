"""Evolve the analytic harmonic gauge wave on a periodic Cartesian grid."""

import math

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
from tqdm import tqdm

from JAX_BSSN.bssn.tensor_algebra import (
    invert_3x3_metric,
    trace_tensor,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import PERIODIC_BC
from JAX_BSSN.evolution.time_evolve import rk4_step


def create_coordinate_arrays(
    ni: int, nj: int, nk: int, dx: float
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create vertex-centered Cartesian coordinates for this demonstration."""

    dtype = jnp.result_type(dx, 1.0)
    spacing = jnp.asarray(dx, dtype=dtype)
    half = jnp.asarray(2.0, dtype=dtype)
    axes = [
        (
            jnp.arange(size, dtype=dtype)
            - jnp.asarray(size - 1, dtype=dtype) / half
        )
        * spacing
        for size in (ni, nj, nk)
    ]
    return jnp.meshgrid(*axes, indexing="ij")


def gauge_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """Initialize the analytic harmonic gauge wave at time zero."""

    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    return gauge_wave_analytic_state(
        X, Y, Z, 0.0, amplitude, wavelength
    )


def gauge_wave_analytic_state(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    t: float = 0.0,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """Evaluate the exact positive-x harmonic gauge-wave solution."""

    del Y, Z
    shape = X.shape
    phase = (2.0 * jnp.pi * (X - t)) / wavelength
    H0 = 1.0 - amplitude * jnp.sin(phase)

    lapse = jnp.sqrt(H0)
    shift = jnp.zeros((3,) + shape)
    conformal_factor = jnp.power(H0, -1.0 / 6.0)

    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(
        H0 * conformal_factor**2
    )
    conformal_metric = conformal_metric.at[1, 1].set(
        conformal_factor**2
    )
    conformal_metric = conformal_metric.at[2, 2].set(
        conformal_factor**2
    )

    extrinsic_curvature = jnp.zeros_like(conformal_metric)
    K_xx = -(jnp.pi * amplitude / wavelength) * jnp.cos(
        phase
    ) / jnp.sqrt(H0)
    extrinsic_curvature = extrinsic_curvature.at[0, 0].set(K_xx)

    inverse_metric = invert_3x3_metric(conformal_metric)
    traceless_K = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inverse_metric
    )
    trace_K = conformal_factor**2 * trace_tensor(
        extrinsic_curvature, inverse_metric
    )

    dHdx = -(2.0 * jnp.pi * amplitude / wavelength) * jnp.cos(phase)
    conformal_connection = jnp.zeros((3,) + shape)
    conformal_connection = conformal_connection.at[0].set(
        (2.0 / 3.0) * H0 ** (-5.0 / 3.0) * dHdx
    )

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )


def run_gauge_wave(
    grid_size: int = 60,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
    cfl: float = 1.0,
    final_time: float | None = None,
    show_progress: bool = True,
):
    """Run the periodic gauge-wave demonstration and report final errors."""

    if final_time is None:
        final_time = wavelength
    dx = wavelength / grid_size
    num_steps = math.ceil(final_time / (cfl * dx))
    dt = final_time / num_steps
    coordinates = create_coordinate_arrays(
        grid_size, grid_size, grid_size, dx
    )
    variables = gauge_wave_analytic_state(
        *coordinates, amplitude=amplitude, wavelength=wavelength
    )
    first_point = -(grid_size - 1) * dx / 2.0
    parameters = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        nu=0.0,
        g=0.0,
        dx=dx,
        dt=dt,
        zero_shift=1,
        gauge=0,
        xl_bc=PERIODIC_BC,
        xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=first_point,
        y_min=first_point,
        z_min=first_point,
    )
    advance = jax.jit(lambda state: rk4_step(state, parameters))

    for _ in tqdm(
        range(num_steps),
        desc="Evolving gauge wave",
        unit="step",
        disable=not show_progress,
    ):
        variables = advance(variables)

    exact = gauge_wave_analytic_state(
        *coordinates,
        t=num_steps * dt,
        amplitude=amplitude,
        wavelength=wavelength,
    )
    errors = {
        field: float(
            jnp.sqrt(jnp.mean((numerical - analytic) ** 2))
        )
        for field, numerical, analytic in zip(
            BSSNVariables._fields, variables, exact
        )
    }
    print("Gauge-wave demo")
    print(
        f"grid={grid_size}^3 dx={dx:.8f}, dt={dt:.8f}, "
        f"steps={num_steps}, final time={num_steps * dt:.8f}"
    )
    print(
        "final RMS errors: "
        + ", ".join(f"{name}={error:.6e}" for name, error in errors.items())
    )
    return variables, {
        "parameters": parameters,
        "num_steps": num_steps,
        "final_time": num_steps * dt,
        "errors": errors,
    }


def main():
    run_gauge_wave()


if __name__ == "__main__":
    main()
