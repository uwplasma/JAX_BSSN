"""
Initial data setup for numerical relativity simulations.

This module provides analytic initial data for wave tests, static matter tests,
and puncture black holes in the BSSN variables used by this codebase.
"""

from typing import Tuple

import jax.numpy as jnp
import numpy as np
from scipy.special import j0, j1

from JAX_BSSN.bssn import BSSNVariables, BSSNParameters
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.tensor_algebra import (
    invert_3x3_metric,
    determinant_3x3_metric,
    christoffel_symbols_second_kind,
    trace_tensor,
    traceless_part,
)


# NOTE: INITIAL DATA IS ALIGNED WITH THE HARMONIC GAUGE FORMULATION.


def setup_simulation_parameters():
    """Set up default simulation parameters for harmonic gauge evolution."""
    # Grid parameters
    ni, nj, nk = 64, 64, 64  # Grid size
    dx = 0.1                 # Grid spacing
    dt = dx                  # Time step (CFL ~ 1 for gauge-wave tests)
    t_final = 1.0            # Final time

    # BSSN parameters (harmonic gauge, zero shift)
    bssn_params = BSSNParameters(
        eta=0.0,
        kappa=0.025,
        g=0.0,
        dx=dx,
        dt=dt,
        nu=0.25,  # Kreiss-Oliger dissipation coefficient
    )


    return {
        "grid": (ni, nj, nk, dx),
        "evolution": (dt, t_final),
        "bssn_params": bssn_params,
    }


def create_coordinate_arrays(
    ni: int, nj: int, nk: int, dx: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Create coordinate arrays for the computational grid.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing (assumed uniform)

    Returns:
        Tuple of (x, y, z) coordinate arrays
    """
    x = jnp.arange(ni) * dx - (ni - 1) * dx / 2
    y = jnp.arange(nj) * dx - (nj - 1) * dx / 2
    z = jnp.arange(nk) * dx - (nk - 1) * dx / 2

    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    return X, Y, Z


def _compute_conformal_connection(conformal_metric: jnp.ndarray, dx: float) -> jnp.ndarray:
    """Compute conformal connection functions from a conformal metric."""
    derivs = jnp.stack(
        [diff1_field(conformal_metric, d + 2, dx) for d in range(3)], axis=0
    )
    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    christoffel_2 = christoffel_symbols_second_kind(inv_conformal_metric, derivs)
    return jnp.einsum("mn..., imn... -> i...", inv_conformal_metric, christoffel_2)


def gauge_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """
    Initialize harmonic gauge wave initial data.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Gauge wave amplitude
        wavelength: Wave wavelength

    Returns:
        BSSN variables for the gauge wave.
    """
    shape = (ni, nj, nk)
    X, _, _ = create_coordinate_arrays(ni, nj, nk, dx)

    d = wavelength

    def H(x_: jnp.ndarray, t_: float) -> jnp.ndarray:
        return 1.0 - amplitude * jnp.sin((2.0 * jnp.pi * (x_ - t_)) / d)

    H0 = H(X, 0.0)

    lapse = jnp.sqrt(H0)
    shift = jnp.zeros((3,) + shape)

    conformal_factor = jnp.power(H0, -1.0 / 6.0)

    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(H0 * conformal_factor**2)
    conformal_metric = conformal_metric.at[1, 1].set(conformal_factor**2)
    conformal_metric = conformal_metric.at[2, 2].set(conformal_factor**2)

    extrinsic_curvature = jnp.zeros_like(conformal_metric)
    K_xx = -(jnp.pi * amplitude / d) * jnp.cos((2.0 * jnp.pi * X) / d) / jnp.sqrt(H0)
    extrinsic_curvature = extrinsic_curvature.at[0, 0].set(K_xx)

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    traceless_extrinsic_curvature = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inv_conformal_metric
    )
    trace_extrinsic_curvature = (
        conformal_factor**2 * trace_tensor(extrinsic_curvature, inv_conformal_metric)
    )

    derivs = jnp.stack(
        [diff1_field(conformal_metric, d_ + 2, dx) for d_ in range(3)], axis=0
    )
    christoffel_2 = christoffel_symbols_second_kind(inv_conformal_metric, derivs)
    conformal_connection = jnp.einsum(
        "mn..., imn... -> i...", inv_conformal_metric, christoffel_2
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


def gowdy_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 0.1,
    wavelength: float = 1.0,
    t0: float = 1.0,
) -> BSSNVariables:
    """
    Initialize polarized Gowdy wave initial data (Q = 0) at a fixed time t0.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Wave amplitude
        wavelength: Wave wavelength in the z-direction
        t0: Initial time for the Gowdy solution

    Returns:
        BSSN variables for the Gowdy wave.
    """
    del amplitude, wavelength

    shape = (ni, nj, nk)
    _, _, Z = create_coordinate_arrays(ni, nj, nk, dx)

    def J0(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(j0(np.asarray(x)))

    def J1(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(j1(np.asarray(x)))

    two_pi = 2.0 * jnp.pi
    J0_t = J0(two_pi * t0)
    J1_t = J1(two_pi * t0)

    P = J0_t * jnp.cos(two_pi * Z)
    dPdt = -two_pi * J1_t * jnp.cos(two_pi * Z)
    dPdz = -two_pi * J0_t * jnp.sin(two_pi * Z)

    cos2 = jnp.cos(two_pi * Z) ** 2
    term1 = -two_pi * t0 * J0_t * J1_t * cos2
    term2 = 2.0 * (jnp.pi**2) * (t0**2) * (J0_t**2 + J1_t**2)

    J0_2pi = J0(two_pi)
    J1_2pi = J1(two_pi)
    c0 = 0.5 * ((two_pi**2) * (J0_2pi**2 + J1_2pi**2) - two_pi * J0_2pi * J1_2pi)
    lam = term1 + term2 - c0

    g_xx = t0 * jnp.exp(P)
    g_yy = t0 * jnp.exp(-P)
    g_zz = t0 ** (-0.5) * jnp.exp(0.5 * lam)

    physical_metric = jnp.zeros((3, 3) + shape)
    physical_metric = physical_metric.at[0, 0].set(g_xx)
    physical_metric = physical_metric.at[1, 1].set(g_yy)
    physical_metric = physical_metric.at[2, 2].set(g_zz)

    lam_t = t0 * (dPdt**2 + dPdz**2)
    K_xx = -0.5 * (t0**0.25) * jnp.exp(-0.25 * lam) * jnp.exp(P) * (1.0 + t0 * dPdt)
    K_yy = -0.5 * (t0**0.25) * jnp.exp(-0.25 * lam) * jnp.exp(-P) * (1.0 - t0 * dPdt)
    K_zz = 0.25 * (t0 ** (-0.25)) * jnp.exp(0.25 * lam) * (t0**(-1.0) - lam_t)

    extrinsic_curvature = jnp.zeros_like(physical_metric)
    extrinsic_curvature = extrinsic_curvature.at[0, 0].set(K_xx)
    extrinsic_curvature = extrinsic_curvature.at[1, 1].set(K_yy)
    extrinsic_curvature = extrinsic_curvature.at[2, 2].set(K_zz)

    det_gamma = determinant_3x3_metric(physical_metric)
    conformal_factor = det_gamma ** (-1.0 / 6.0)
    conformal_metric = physical_metric * conformal_factor**2

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    traceless_K = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inv_conformal_metric
    )
    trace_K = conformal_factor**2 * trace_tensor(extrinsic_curvature, inv_conformal_metric)
    conformal_connection = _compute_conformal_connection(conformal_metric, dx)

    lapse = t0 ** (-0.25) * jnp.exp(0.25 * lam)
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


def linear_wave_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    amplitude: float = 1.0e-8,
    wavelength: float = 1.0,
) -> BSSNVariables:
    """
    Initialize linear-wave data in Gauss coordinates from the notebook test setup.

    Args:
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        amplitude: Wave amplitude A
        wavelength: Domain length d in b = A sin(2pi(x-t)/d)

    Returns:
        BSSN variables for the linear wave.
    """
    shape = (ni, nj, nk)
    X, _, _ = create_coordinate_arrays(ni, nj, nk, dx)

    d = wavelength
    b0 = amplitude * jnp.sin((2.0 * jnp.pi * X) / d)

    g_xx = jnp.ones_like(X)
    g_yy = 1.0 + b0
    g_zz = 1.0 - b0

    conformal_factor = jnp.power(1.0 - b0**2, -1.0 / 6.0)

    conformal_metric = jnp.zeros((3, 3) + shape)
    conformal_metric = conformal_metric.at[0, 0].set(g_xx * conformal_factor**2)
    conformal_metric = conformal_metric.at[1, 1].set(g_yy * conformal_factor**2)
    conformal_metric = conformal_metric.at[2, 2].set(g_zz * conformal_factor**2)

    dbdt0 = -(2.0 * jnp.pi * amplitude / d) * jnp.cos((2.0 * jnp.pi * X) / d)

    extrinsic_curvature = jnp.zeros_like(conformal_metric)
    # Sign convention used in evolution code: K_ij = -(1/2) * d_t g_ij for alpha=1, beta=0.
    extrinsic_curvature = extrinsic_curvature.at[1, 1].set(-0.5 * dbdt0)
    extrinsic_curvature = extrinsic_curvature.at[2, 2].set(0.5 * dbdt0)

    inv_conformal_metric = invert_3x3_metric(conformal_metric)
    traceless_K = conformal_factor**2 * traceless_part(
        extrinsic_curvature, conformal_metric, inv_conformal_metric
    )
    trace_K = conformal_factor**2 * trace_tensor(extrinsic_curvature, inv_conformal_metric)
    conformal_connection = _compute_conformal_connection(conformal_metric, dx)

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


def puncture_black_hole_data(
    ni: int,
    nj: int,
    nk: int,
    dx: float,
    mass: float = 1.0,
    lapse_puncture: bool = True,
    center_between_points: bool = True,
    min_radius: float | None = None,
) -> BSSNVariables:
    """
    Initialize isotropic puncture black hole data.

    The physical 3-metric is

        g_ij = psi^4 delta_ij,  psi = 1 + M / (2 r),

    while the BSSN conformal metric is flat and the code's conformal factor is

        W = psi^-2 = (1 + M / (2 r))^-2.

    Following the working puncture setup described upstream, the initial lapse is
    also given a puncture profile alpha = W.  The extrinsic curvature, trace K,
    conformal connection, and shift are all initially zero.
    """
    shape = (ni, nj, nk)
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)

    if center_between_points:
        if ni % 2 == 1:
            X = X + 0.5 * dx
        if nj % 2 == 1:
            Y = Y + 0.5 * dx
        if nk % 2 == 1:
            Z = Z + 0.5 * dx

    r = jnp.sqrt(X**2 + Y**2 + Z**2)
    r_floor = 0.5 * dx if min_radius is None else min_radius
    r_safe = jnp.maximum(r, r_floor)

    psi = 1.0 + mass / (2.0 * r_safe)
    conformal_factor = psi ** (-2.0)
    lapse = conformal_factor if lapse_puncture else jnp.ones(shape)
    shift = jnp.zeros((3,) + shape)

    conformal_metric = (
        jnp.eye(3, dtype=conformal_factor.dtype)[:, :, None, None, None]
        * jnp.ones((3, 3) + shape, dtype=conformal_factor.dtype)
    )
    traceless_K = jnp.zeros_like(conformal_metric)
    trace_K = jnp.zeros(shape, dtype=conformal_factor.dtype)
    conformal_connection = jnp.zeros((3,) + shape, dtype=conformal_factor.dtype)

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
    )


def get_initial_data(
    data_type: str, ni: int, nj: int, nk: int, dx: float, **kwargs
) -> BSSNVariables:
    """
    Get initial data of specified type.

    Args:
        data_type: Type of initial data ('gauge_wave', 'gowdy_wave', 'linear_wave', 'puncture_black_hole')
        ni, nj, nk: Grid dimensions
        dx: Grid spacing
        **kwargs: Additional parameters for specific data types

    Returns:
        BSSN variables for the specified initial data
    """
    if data_type in {"gauge_wave", "gauge"}:
        return gauge_wave_data(ni, nj, nk, dx, **kwargs)
    if data_type in {"gowdy_wave", "gowdy"}:
        return gowdy_wave_data(ni, nj, nk, dx, **kwargs)
    if data_type in {"linear_wave", "linear"}:
        return linear_wave_data(ni, nj, nk, dx, **kwargs)
    if data_type in {"puncture_black_hole", "puncture", "black_hole"}:
        return puncture_black_hole_data(ni, nj, nk, dx, **kwargs)

    raise ValueError(f"Unknown initial data type: {data_type}")


def compute_adm_mass(vars: BSSNVariables, dx: float) -> float:
    """
    Compute ADM mass using surface integral at infinity (simplified).

    Args:
        vars: BSSN variables
        dx: Grid spacing

    Returns:
        Approximate ADM mass
    """
    boundary_average = (
        jnp.mean(vars.conformal_factor[0, :, :])
        + jnp.mean(vars.conformal_factor[-1, :, :])
        + jnp.mean(vars.conformal_factor[:, 0, :])
        + jnp.mean(vars.conformal_factor[:, -1, :])
        + jnp.mean(vars.conformal_factor[:, :, 0])
        + jnp.mean(vars.conformal_factor[:, :, -1])
    ) / 6

    return jnp.maximum(0.0, boundary_average - 1.0)


def compute_adm_momentum(vars: BSSNVariables, dx: float) -> jnp.ndarray:
    """
    Compute ADM momentum (simplified).

    Args:
        vars: BSSN variables
        dx: Grid spacing

    Returns:
        3-vector of ADM momentum components
    """
    momentum = jnp.zeros(3)

    for i in range(3):
        for j in range(3):
            momentum = momentum.at[i].add(jnp.sum(vars.traceless_K[i, j]))

    return momentum * dx**3
