"""BSSN state and parameter definitions."""

from typing import NamedTuple, Tuple

import jax.numpy as jnp


class BSSNVariables(NamedTuple):
    """Container for BSSN evolution variables."""
    conformal_metric: jnp.ndarray      # γ_ij (3x3 symmetric)
    conformal_factor: jnp.ndarray      # W or φ
    traceless_K: jnp.ndarray          # A_ij (3x3 traceless)
    trace_K: jnp.ndarray              # K (scalar)
    conformal_connection: jnp.ndarray  # Γ^i (3-vector)
    lapse: jnp.ndarray                # α (scalar)
    shift: jnp.ndarray                # β^i (3-vector)


class BSSNParameters(NamedTuple):
    """Parameters for BSSN evolution."""
    eta: float = 2.0          # Damping parameter for shift evolution
    kappa: float = 0.0        # Constraint damping parameter
    nu: float = 0.002          # Kreiss-Oliger dissipation coefficient
    g: float = 0.75           # Gamma driver shift parameter
    dx: float = 0.1           # Grid spacing
    dt: float = 0.001         # Time step
    zero_shift: int = 0       # If 1, hold the shift fixed during RK stages
    gauge: int = 0            # 0 = harmonic slicing, 1 = 1+log slicing
    xl_bc: int = 0            # x-left boundary code: 0 periodic, 1 Sommerfeld
    xr_bc: int = 0            # x-right boundary code: 0 periodic, 1 Sommerfeld
    yl_bc: int = 0            # y-left boundary code: 0 periodic, 1 Sommerfeld
    yr_bc: int = 0            # y-right boundary code: 0 periodic, 1 Sommerfeld
    zl_bc: int = 0            # z-left boundary code: 0 periodic, 1 Sommerfeld
    zr_bc: int = 0            # z-right boundary code: 0 periodic, 1 Sommerfeld
    x_min: float = 0.0        # Coordinate at the first x grid point
    y_min: float = 0.0        # Coordinate at the first y grid point
    z_min: float = 0.0        # Coordinate at the first z grid point
    mad_q: float = 1.0        # D4 weight in the MAD first/second derivatives


def get_boundary_codes(
    params: BSSNParameters, spatial_direction: int
) -> Tuple[int, int]:
    """Return the left and right face codes for one physical direction."""

    if spatial_direction == 0:
        return params.xl_bc, params.xr_bc
    elif spatial_direction == 1:
        return params.yl_bc, params.yr_bc
    else:
        return params.zl_bc, params.zr_bc
