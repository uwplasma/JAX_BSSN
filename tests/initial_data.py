"""Initial-data helpers owned by the test suite."""

from typing import Tuple

import jax.numpy as jnp

from JAX_BSSN.bssn.tensor_algebra import (
    christoffel_symbols_second_kind,
    invert_3x3_metric,
    trace_tensor,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNVariables
from JAX_BSSN.evolution.derivatives import diff1_field


def create_coordinate_arrays(
    ni: int, nj: int, nk: int, dx: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create vertex-centered Cartesian coordinate arrays for tests."""

    dtype = jnp.result_type(dx, 1.0)
    spacing = jnp.asarray(dx, dtype=dtype)
    half = jnp.asarray(2.0, dtype=dtype)
    x = (
        jnp.arange(ni, dtype=dtype)
        - jnp.asarray(ni - 1, dtype=dtype) / half
    ) * spacing
    y = (
        jnp.arange(nj, dtype=dtype)
        - jnp.asarray(nj - 1, dtype=dtype) / half
    ) * spacing
    z = (
        jnp.arange(nk, dtype=dtype)
        - jnp.asarray(nk - 1, dtype=dtype) / half
    ) * spacing

    return jnp.meshgrid(x, y, z, indexing="ij")


def _compute_conformal_connection(
    conformal_metric: jnp.ndarray, dx: float, mad_q: float = 1.0
) -> jnp.ndarray:
    """Compute conformal connection functions from a conformal metric."""

    derivs = jnp.stack(
        [
            diff1_field(conformal_metric, direction + 2, dx, mad_q=mad_q)
            for direction in range(3)
        ],
        axis=0,
    )
    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    christoffel_2 = christoffel_symbols_second_kind(
        inv_conformal_metric, derivs
    )
    return jnp.einsum(
        "mn..., imn... -> i...", inv_conformal_metric, christoffel_2
    )


def gauge_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """Initialize harmonic gauge-wave data."""

    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)
    return gauge_wave_analytic_state(
        X, Y, Z, 0.0, amplitude, wavelength
    )


def periodic_coordinate_arrays(
    grid_size: int, domain_length: float = 1.0
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Create the endpoint-excluded periodic grid used by gauge-wave tests."""

    axis = jnp.linspace(
        -domain_length / 2.0,
        domain_length / 2.0,
        grid_size,
        endpoint=False,
    )
    return jnp.meshgrid(axis, axis, axis, indexing="ij")


def periodic_gauge_wave_state(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """Build the historical periodic gauge-wave regression initial state."""

    del Y, Z
    shape = X.shape
    phase = 2.0 * jnp.pi * X / wavelength
    H = amplitude * jnp.sin(phase)
    physical_metric = jnp.zeros((3, 3) + shape)
    physical_metric = physical_metric.at[0, 0].set(1.0 - H)
    physical_metric = physical_metric.at[1, 1].set(jnp.ones_like(H))
    physical_metric = physical_metric.at[2, 2].set(jnp.ones_like(H))

    conformal_factor = jnp.power(1.0 - H, -1.0 / 6.0)
    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(
        jnp.power(1.0 - H, 2.0 / 3.0)
    )
    conformal_metric = conformal_metric.at[1, 1].set(
        jnp.power(1.0 - H, -1.0 / 3.0)
    )
    conformal_metric = conformal_metric.at[2, 2].set(
        jnp.power(1.0 - H, -1.0 / 3.0)
    )

    lapse = jnp.sqrt(1.0 - H)
    derivatives = jnp.stack(
        [diff1_field(conformal_metric, direction + 2, dx)
         for direction in range(3)],
        axis=0,
    )
    inverse_conformal_metric = invert_3x3_metric(conformal_metric)
    christoffel = christoffel_symbols_second_kind(
        inverse_conformal_metric, derivatives
    )
    conformal_connection = jnp.einsum(
        "mn..., imn... -> i...", inverse_conformal_metric, christoffel
    )

    extrinsic_curvature = jnp.zeros_like(physical_metric)
    extrinsic_curvature = extrinsic_curvature.at[0, 0].set(
        -amplitude
        * jnp.pi
        * jnp.cos(phase)
        / (wavelength * lapse)
    )
    inverse_physical_metric = invert_3x3_metric(physical_metric)
    trace_K = jnp.einsum(
        "mn..., mn... -> ...", inverse_physical_metric, extrinsic_curvature
    )
    traceless_K = conformal_factor**2 * (
        extrinsic_curvature - physical_metric * trace_K / 3.0
    )

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=jnp.zeros((3,) + shape),
    )


def gauge_wave_analytic_state(
    X: jnp.ndarray,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    t: float = 0.0,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """Evaluate the exact harmonic gauge wave on coordinate arrays."""

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

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    traceless_extrinsic_curvature = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inv_conformal_metric
    )
    trace_extrinsic_curvature = conformal_factor**2 * trace_tensor(
        extrinsic_curvature, inv_conformal_metric
    )

    dHdx = -(2.0 * jnp.pi * amplitude / wavelength) * jnp.cos(phase)
    conformal_connection = jnp.zeros((3,) + shape)
    conformal_connection = conformal_connection.at[0].set(
        (2.0 / 3.0) * H0 ** (-5.0 / 3.0) * dHdx
    )

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_extrinsic_curvature,
        trace_K=trace_extrinsic_curvature,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )


def linear_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 1.0e-8,
    wavelength: float = 1.0,
    mad_q: float = 1.0,
) -> BSSNVariables:
    """Initialize plus-polarized linear-wave data in Gauss coordinates."""

    shape = (ni, nj, nk)
    X, _, _ = create_coordinate_arrays(ni, nj, nk, dx)

    d = wavelength
    b0 = amplitude * jnp.sin((2.0 * jnp.pi * X) / d)

    g_xx = jnp.ones_like(X)
    g_yy = 1.0 + b0
    g_zz = 1.0 - b0

    conformal_factor = jnp.power(1.0 - b0**2, -1.0 / 6.0)

    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(
        g_xx * conformal_factor**2
    )
    conformal_metric = conformal_metric.at[1, 1].set(
        g_yy * conformal_factor**2
    )
    conformal_metric = conformal_metric.at[2, 2].set(
        g_zz * conformal_factor**2
    )

    dbdt0 = -(2.0 * jnp.pi * amplitude / d) * jnp.cos(
        (2.0 * jnp.pi * X) / d
    )

    extrinsic_curvature = jnp.zeros_like(conformal_metric)
    extrinsic_curvature = extrinsic_curvature.at[1, 1].set(-0.5 * dbdt0)
    extrinsic_curvature = extrinsic_curvature.at[2, 2].set(0.5 * dbdt0)

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    traceless_K = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inv_conformal_metric
    )
    trace_K = conformal_factor**2 * trace_tensor(
        extrinsic_curvature, inv_conformal_metric
    )
    conformal_connection = _compute_conformal_connection(
        conformal_metric, dx, mad_q
    )

    lapse = jnp.ones(shape)
    shift = jnp.zeros((3,) + shape)

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )
