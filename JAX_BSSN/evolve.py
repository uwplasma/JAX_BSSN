from JAX_BSSN.bssn import (
    BSSNVariables, BSSNParameters, 
    evolve_conformal_metric, evolve_conformal_factor,
    evolve_traceless_extrinsic_curvature, evolve_trace_extrinsic_curvature,
    evolve_conformal_connection, evolve_lapse, evolve_shift, evolve_rho
)
from jax import jit
import jax


# NOTE: FULLY TESTED AND FUNCTIONAL AS OF DEC 3RD 2025


@jit
def rk4_step(vars: BSSNVariables, params: BSSNParameters) -> BSSNVariables:
    """
    Perform one RK4 timestep with Kreiss-Oliger dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Kreiss-Oliger dissipation coefficient
        
    Returns:
        Updated BSSN variables
    """
    dt = params.dt
    
    # k1 time derivatives
    dt_gamma = evolve_conformal_metric(vars, params)
    dt_W = evolve_conformal_factor(vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(vars, params)
    dt_K = evolve_trace_extrinsic_curvature(vars, params)
    dt_Gamma = evolve_conformal_connection(vars, params)
    dt_alpha = evolve_lapse(vars, params)
    dt_beta = evolve_shift(vars, params)
    dt_rho = evolve_rho(vars, params) # NEW

    k1 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta, dt_rho]

    # k2 - midpoint with k1
    mid_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k1[0],
        conformal_factor=vars.conformal_factor + 0.5 * dt * k1[1],
        traceless_K=vars.traceless_K + 0.5 * dt * k1[2],
        trace_K=vars.trace_K + 0.5 * dt * k1[3],
        conformal_connection=vars.conformal_connection + 0.5 * dt * k1[4],
        lapse=vars.lapse + 0.5 * dt * k1[5],
        shift=vars.shift + 0.5 * dt * k1[6],
        rho=vars.rho + 0.5 * dt * k1[7] # NEW
    )

    # k2 time derivatives
    dt_gamma = evolve_conformal_metric(mid_vars, params)
    dt_W = evolve_conformal_factor(mid_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(mid_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(mid_vars, params)
    dt_Gamma = evolve_conformal_connection(mid_vars, params)
    dt_alpha = evolve_lapse(mid_vars, params)
    dt_beta = evolve_shift(mid_vars, params)
    dt_rho = evolve_rho(mid_vars, params) # NEW

    k2 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta, dt_rho]

    # k3 - midpoint with k2
    mid_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + 0.5 * dt * k2[0],
        conformal_factor=vars.conformal_factor + 0.5 * dt * k2[1],
        traceless_K=vars.traceless_K + 0.5 * dt * k2[2],
        trace_K=vars.trace_K + 0.5 * dt * k2[3],
        conformal_connection=vars.conformal_connection + 0.5 * dt * k2[4],
        lapse=vars.lapse + 0.5 * dt * k2[5],
        shift=vars.shift + 0.5 * dt * k2[6],
        rho=vars.rho + 0.5 * dt * k2[7] # NEW
    )

    # k3 time derivatives
    dt_gamma = evolve_conformal_metric(mid_vars, params)
    dt_W = evolve_conformal_factor(mid_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(mid_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(mid_vars, params)
    dt_Gamma = evolve_conformal_connection(mid_vars, params)
    dt_alpha = evolve_lapse(mid_vars, params)
    dt_beta = evolve_shift(mid_vars, params)
    dt_rho = evolve_rho(mid_vars, params) # NEW

    k3 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta, dt_rho]

    # k4 - endpoint with k3
    end_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + dt * k3[0],
        conformal_factor=vars.conformal_factor + dt * k3[1],
        traceless_K=vars.traceless_K + dt * k3[2],
        trace_K=vars.trace_K + dt * k3[3],
        conformal_connection=vars.conformal_connection + dt * k3[4],
        lapse=vars.lapse + dt * k3[5],
        shift=vars.shift + dt * k3[6],
        rho=vars.rho + dt * k3[7] # NEW
    )

    # k4 time derivatives
    dt_gamma = evolve_conformal_metric(end_vars, params)
    dt_W = evolve_conformal_factor(end_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(end_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(end_vars, params)
    dt_Gamma = evolve_conformal_connection(end_vars, params)
    dt_alpha = evolve_lapse(end_vars, params)
    dt_beta = evolve_shift(end_vars, params)
    dt_rho = evolve_rho(end_vars, params) # NEW

    k4 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta, dt_rho]

    # Final RK4 update
    new_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        conformal_factor=vars.conformal_factor + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        traceless_K=vars.traceless_K + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2]),
        trace_K=vars.trace_K + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
        conformal_connection=vars.conformal_connection + (dt / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
        lapse=vars.lapse + (dt / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]),
        shift=vars.shift + (dt / 6.0) * (k1[6] + 2 * k2[6] + 2 * k3[6] + k4[6]),
        rho=vars.rho + (dt / 6.0) * (k1[7] + 2 * k2[7] + 2 * k3[7] + k4[7]) # NEW
    )

    return new_vars