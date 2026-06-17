from JAX_BSSN.bssn import (
    BSSNVariables, BSSNParameters, 
    evolve_conformal_metric, evolve_conformal_factor,
    evolve_traceless_extrinsic_curvature, evolve_trace_extrinsic_curvature,
    evolve_conformal_connection, evolve_lapse, evolve_shift
)
from JAX_BSSN.tensor_algebra import invert_3x3_metric, traceless_part, determinant_3x3_metric
from jax import jit
import jax


# NOTE: FULLY TESTED AND FUNCTIONAL AS OF DEC 3RD 2025


@jit
def rk4_step(vars: BSSNVariables, params: BSSNParameters, ko_sigma: float = None) -> BSSNVariables:
    """
    Perform one RK4 timestep with Kreiss-Oliger dissipation.
    
    Args:
        vars: Current BSSN variables
        params: Evolution parameters
        ko_sigma: Optional Kreiss-Oliger dissipation coefficient override
        
    Returns:
        Updated BSSN variables
    """
    if ko_sigma is not None:
        params = params._replace(nu=ko_sigma)

    dt = params.dt
    
    # k1 time derivatives
    dt_gamma = evolve_conformal_metric(vars, params)
    dt_W = evolve_conformal_factor(vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(vars, params)
    dt_K = evolve_trace_extrinsic_curvature(vars, params)
    dt_Gamma = evolve_conformal_connection(vars, params)
    dt_alpha = evolve_lapse(vars, params)
    dt_beta = evolve_shift(vars, params)

    k1 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # k2 - midpoint with k1
    mid_vars = BSSNVariables(
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
    )

    # k2 time derivatives
    dt_gamma = evolve_conformal_metric(mid_vars, params)
    dt_W = evolve_conformal_factor(mid_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(mid_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(mid_vars, params)
    dt_Gamma = evolve_conformal_connection(mid_vars, params)
    dt_alpha = evolve_lapse(mid_vars, params)
    dt_beta = evolve_shift(mid_vars, params)

    k2 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # k3 - midpoint with k2
    mid_vars = BSSNVariables(
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
    )

    # k3 time derivatives
    dt_gamma = evolve_conformal_metric(mid_vars, params)
    dt_W = evolve_conformal_factor(mid_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(mid_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(mid_vars, params)
    dt_Gamma = evolve_conformal_connection(mid_vars, params)
    dt_alpha = evolve_lapse(mid_vars, params)
    dt_beta = evolve_shift(mid_vars, params)

    k3 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # k4 - endpoint with k3
    end_vars = BSSNVariables(
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
    )

    # k4 time derivatives
    dt_gamma = evolve_conformal_metric(end_vars, params)
    dt_W = evolve_conformal_factor(end_vars, params)
    dt_A = evolve_traceless_extrinsic_curvature(end_vars, params)
    dt_K = evolve_trace_extrinsic_curvature(end_vars, params)
    dt_Gamma = evolve_conformal_connection(end_vars, params)
    dt_alpha = evolve_lapse(end_vars, params)
    dt_beta = evolve_shift(end_vars, params)

    k4 = [dt_gamma, dt_W, dt_A, dt_K, dt_Gamma, dt_alpha, dt_beta]

    # Final RK4 update
    new_traceless_K = vars.traceless_K + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    new_vars = BSSNVariables(
        conformal_metric=vars.conformal_metric + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]),
        conformal_factor=vars.conformal_factor + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]),
        traceless_K=new_traceless_K,
        trace_K=vars.trace_K + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3]),
        conformal_connection=vars.conformal_connection + (dt / 6.0) * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4]),
        lapse=vars.lapse + (dt / 6.0) * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5]),
        shift=vars.shift + (dt / 6.0) * (k1[6] + 2 * k2[6] + 2 * k3[6] + k4[6]),
        rho=vars.rho,
        S_ij=vars.S_ij,
        momentum_density=vars.momentum_density,
    )

    # Enforce tracelessness explicitly on the updated extrinsic curvature.
    # Then normalize the conformal metric so that det(\tilde{\gamma}) = 1.
    # This keeps the algebraic BSSN constraints satisfied after the RK4 step.
    # Compute inverse and make A_ij traceless first (uniform scaling preserves tracelessness,
    # but we compute the traceless projection before and after normalization to be safe).
    inv_gamma = invert_3x3_metric(new_vars.conformal_metric)
    enforced_traceless_K = traceless_part(new_vars.traceless_K, new_vars.conformal_metric, inv_gamma)

    # Normalize conformal metric determinant to 1
    det_gamma = determinant_3x3_metric(new_vars.conformal_metric)
    # det_gamma has shape (ni,nj,nk). compute scale factor s = det^{-1/3}
    scale = det_gamma ** (-1.0 / 3.0)
    # Broadcast scale to (3,3,ni,nj,nk)
    scale = scale[None, None, ...]
    normalized_metric = new_vars.conformal_metric * scale

    # Recompute traceless K w.r.t. normalized metric to be consistent
    inv_gamma_norm = invert_3x3_metric(normalized_metric)
    enforced_traceless_K = traceless_part(enforced_traceless_K, normalized_metric, inv_gamma_norm)

    return BSSNVariables(
        conformal_metric=normalized_metric,
        conformal_factor=new_vars.conformal_factor,
        traceless_K=enforced_traceless_K,
        trace_K=new_vars.trace_K,
        conformal_connection=new_vars.conformal_connection,
        lapse=new_vars.lapse,
        shift=new_vars.shift,
        rho=new_vars.rho,
        S_ij=new_vars.S_ij,
        momentum_density=new_vars.momentum_density,
    )