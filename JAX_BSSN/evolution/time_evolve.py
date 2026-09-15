from JAX_BSSN.bssn.conformal_connection import evolve_conformal_connection
from JAX_BSSN.bssn.extrinsic_curvature import (
    evolve_trace_extrinsic_curvature,
    evolve_traceless_extrinsic_curvature,
)
from JAX_BSSN.bssn.shift_and_lapse import evolve_lapse, evolve_shift
from JAX_BSSN.bssn.spatial_metric import (
    evolve_conformal_factor,
    evolve_conformal_metric,
)
from JAX_BSSN.bssn.tensor_algebra import (
    determinant_3x3_metric,
    invert_3x3_metric,
    traceless_part,
)
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import apply_sommerfeld_boundaries
from jax import jit
import jax


# NOTE: FULLY TESTED AND FUNCTIONAL AS OF DEC 3RD 2025


def evolve_shift_or_freeze(vars: BSSNVariables, params: BSSNParameters):
    """Return the shift RHS, or zero it when the zero-shift gauge is selected."""

    return jax.lax.cond(
        params.zero_shift == 0,
        lambda _: evolve_shift(vars, params),
        lambda _: 0.0 * vars.shift,
        operand=None,
    )


@jit
def compute_bssn_rhs(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Assemble the bulk BSSN RHS and apply active Sommerfeld faces once."""

    rhs = BSSNVariables(
        conformal_metric=evolve_conformal_metric(vars, params),
        conformal_factor=evolve_conformal_factor(vars, params),
        traceless_K=evolve_traceless_extrinsic_curvature(vars, params),
        trace_K=evolve_trace_extrinsic_curvature(vars, params),
        conformal_connection=evolve_conformal_connection(vars, params),
        lapse=evolve_lapse(vars, params),
        shift=evolve_shift_or_freeze(vars, params),
    )

    return apply_sommerfeld_boundaries(vars, rhs, params)


@jit
def enforce_unit_determinant_conformal_metric(
    vars: BSSNVariables,
) -> BSSNVariables:
    """Rescale the conformal metric so det(gamma_tilde) is one."""

    det_gamma = determinant_3x3_metric(vars.conformal_metric)
    conformal_metric = vars.conformal_metric * det_gamma ** (-1.0 / 3.0)

    return BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=vars.conformal_factor,
        traceless_K=vars.traceless_K,
        trace_K=vars.trace_K,
        conformal_connection=vars.conformal_connection,
        lapse=vars.lapse,
        shift=vars.shift,
    )


@jit
def eliminate_trace_A(vars: BSSNVariables) -> BSSNVariables:
    """Project A_ij onto its trace-free part with the current conformal metric."""

    inv_gamma = invert_3x3_metric(vars.conformal_metric)
    traceless_K = traceless_part(
        vars.traceless_K, vars.conformal_metric, inv_gamma
    )

    return BSSNVariables(
        conformal_metric=vars.conformal_metric,
        conformal_factor=vars.conformal_factor,
        traceless_K=traceless_K,
        trace_K=vars.trace_K,
        conformal_connection=vars.conformal_connection,
        lapse=vars.lapse,
        shift=vars.shift,
    )


@jit
def enforce_algebraic_constraints(vars: BSSNVariables) -> BSSNVariables:
    """Enforce the determinant-one and trace-free BSSN constraints."""

    vars = enforce_unit_determinant_conformal_metric(vars)
    vars = eliminate_trace_A(vars)

    return vars


@jit
def rk4_step(vars: BSSNVariables, params: BSSNParameters) -> BSSNVariables:
    """
    Perform one RK4 timestep with Kreiss-Oliger dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        Kreiss-Oliger dissipation is controlled by params.nu.
        
    Returns:
        Updated BSSN variables
    """
    dt = params.dt
    vars = enforce_algebraic_constraints(vars)
    
    # k1 time derivatives
    k1 = compute_bssn_rhs(vars, params)

    # k2 - midpoint with k1
    mid_vars = enforce_algebraic_constraints(BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k1[0],
        conformal_factor=vars.conformal_factor + 0.5 * dt * k1[1],
        traceless_K=vars.traceless_K + 0.5 * dt * k1[2],
        trace_K=vars.trace_K + 0.5 * dt * k1[3],
        conformal_connection=vars.conformal_connection + 0.5 * dt * k1[4],
        lapse=vars.lapse + 0.5 * dt * k1[5],
        shift=vars.shift + 0.5 * dt * k1[6]
    ))

    # k2 time derivatives
    k2 = compute_bssn_rhs(mid_vars, params)

    # k3 - midpoint with k2
    mid_vars = enforce_algebraic_constraints(BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k2[0],
        conformal_factor=vars.conformal_factor + 0.5 * dt * k2[1],
        traceless_K=vars.traceless_K + 0.5 * dt * k2[2],
        trace_K=vars.trace_K + 0.5 * dt * k2[3],
        conformal_connection=vars.conformal_connection + 0.5 * dt * k2[4],
        lapse=vars.lapse + 0.5 * dt * k2[5],
        shift=vars.shift + 0.5 * dt * k2[6]
    ))

    # k3 time derivatives
    k3 = compute_bssn_rhs(mid_vars, params)

    # k4 - endpoint with k3
    end_vars = enforce_algebraic_constraints(BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * k3[0],
        conformal_factor=vars.conformal_factor + dt * k3[1],
        traceless_K=vars.traceless_K + dt * k3[2],
        trace_K=vars.trace_K + dt * k3[3],
        conformal_connection=vars.conformal_connection + dt * k3[4],
        lapse=vars.lapse + dt * k3[5],
        shift=vars.shift + dt * k3[6]
    ))

    # k4 time derivatives
    k4 = compute_bssn_rhs(end_vars, params)

    # Final RK4 update
    new_vars = enforce_algebraic_constraints(BSSNVariables(
        conformal_metric=vars.conformal_metric + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        conformal_factor=vars.conformal_factor + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        traceless_K=vars.traceless_K + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        trace_K=vars.trace_K + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
        conformal_connection=vars.conformal_connection + (dt / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
        lapse=vars.lapse + (dt / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]),
        shift=vars.shift + (dt / 6.0) * (k1[6] + 2 * k2[6] + 2 * k3[6] + k4[6])
    ))

    return new_vars
