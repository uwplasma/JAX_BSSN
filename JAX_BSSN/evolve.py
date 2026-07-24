from JAX_BSSN.bssn import (
    BSSNVariables, BSSNParameters, 
    evolve_conformal_metric, evolve_conformal_factor,
    evolve_traceless_extrinsic_curvature, evolve_trace_extrinsic_curvature,
    evolve_conformal_connection, evolve_lapse, evolve_shift
)
from JAX_BSSN.tensor_algebra import (
    determinant_3x3_metric,
    invert_3x3_metric,
    traceless_part,
)
from JAX_BSSN.boundaries import apply_supergaussian_boundaries
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
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
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
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
    )


@jit
def enforce_boundaries_and_trace_free_A(
    vars: BSSNVariables, params: BSSNParameters
) -> BSSNVariables:
    """Apply boundary filtering and enforce BSSN algebraic constraints."""

    vars = apply_supergaussian_boundaries(vars, params)
    vars = enforce_unit_determinant_conformal_metric(vars)
    vars = eliminate_trace_A(vars)

    return vars


@jit
def rk4_step(vars: BSSNVariables, params: BSSNParameters, ko_sigma: float = None) -> BSSNVariables:
    """
    Perform one RK4 timestep with Kreiss-Oliger dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Optional Kreiss-Oliger dissipation coefficient override.
        
    Returns:
        Updated BSSN variables
    """
    if ko_sigma is not None:
        params = params._replace(nu=ko_sigma)

    dt = params.dt
    vars = enforce_boundaries_and_trace_free_A(vars, params)
    
    # k1 time derivatives
    dt_gamma = evolve_conformal_metric(vars, params)
    dt_W = evolve_conformal_factor(vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(vars, params)
    dt_K = evolve_trace_extrinsic_curvature(vars, params)
    dt_Gamma = evolve_conformal_connection(vars, params)
    dt_alpha = evolve_lapse(vars, params)
    dt_beta = evolve_shift_or_freeze(vars, params)

    k1 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # k2 - midpoint with k1
    mid_vars = enforce_boundaries_and_trace_free_A(BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k1[0],
        conformal_factor=vars.conformal_factor + 0.5 * dt * k1[1],
        traceless_K=vars.traceless_K + 0.5 * dt * k1[2],
        trace_K=vars.trace_K + 0.5 * dt * k1[3],
        conformal_connection=vars.conformal_connection + 0.5 * dt * k1[4],
        lapse=vars.lapse + 0.5 * dt * k1[5],
        shift=vars.shift + 0.5 * dt * k1[6],
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
    ), params)

    # k2 time derivatives
    dt_gamma = evolve_conformal_metric(mid_vars, params)
    dt_W = evolve_conformal_factor(mid_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(mid_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(mid_vars, params)
    dt_Gamma = evolve_conformal_connection(mid_vars, params)
    dt_alpha = evolve_lapse(mid_vars, params)
    dt_beta = evolve_shift_or_freeze(mid_vars, params)

    k2 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # k3 - midpoint with k2
    mid_vars = enforce_boundaries_and_trace_free_A(BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k2[0],
        conformal_factor=vars.conformal_factor + 0.5 * dt * k2[1],
        traceless_K=vars.traceless_K + 0.5 * dt * k2[2],
        trace_K=vars.trace_K + 0.5 * dt * k2[3],
        conformal_connection=vars.conformal_connection + 0.5 * dt * k2[4],
        lapse=vars.lapse + 0.5 * dt * k2[5],
        shift=vars.shift + 0.5 * dt * k2[6],
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
    ), params)

    # k3 time derivatives
    dt_gamma = evolve_conformal_metric(mid_vars, params)
    dt_W = evolve_conformal_factor(mid_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(mid_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(mid_vars, params)
    dt_Gamma = evolve_conformal_connection(mid_vars, params)
    dt_alpha = evolve_lapse(mid_vars, params)
    dt_beta = evolve_shift_or_freeze(mid_vars, params)

    k3 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # k4 - endpoint with k3
    end_vars = enforce_boundaries_and_trace_free_A(BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * k3[0],
        conformal_factor=vars.conformal_factor + dt * k3[1],
        traceless_K=vars.traceless_K + dt * k3[2],
        trace_K=vars.trace_K + dt * k3[3],
        conformal_connection=vars.conformal_connection + dt * k3[4],
        lapse=vars.lapse + dt * k3[5],
        shift=vars.shift + dt * k3[6],
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
    ), params)

    # k4 time derivatives
    dt_gamma = evolve_conformal_metric(end_vars, params)
    dt_W = evolve_conformal_factor(end_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(end_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(end_vars, params)
    dt_Gamma = evolve_conformal_connection(end_vars, params)
    dt_alpha = evolve_lapse(end_vars, params)
    dt_beta = evolve_shift_or_freeze(end_vars, params)

    k4 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # Final RK4 update
    new_vars = enforce_boundaries_and_trace_free_A(BSSNVariables(
        conformal_metric=vars.conformal_metric + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        conformal_factor=vars.conformal_factor + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        traceless_K=vars.traceless_K + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        trace_K=vars.trace_K + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
        conformal_connection=vars.conformal_connection + (dt / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
        lapse=vars.lapse + (dt / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]),
        shift=vars.shift + (dt / 6.0) * (k1[6] + 2 * k2[6] + 2 * k3[6] + k4[6]),
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
    ), params)

    return new_vars
