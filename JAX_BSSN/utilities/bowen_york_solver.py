"""Bowen-York puncture initial data on a cell-centered Cartesian grid."""

import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg


def cell_centered_grid(num_points=48, half_width=12.0):
    """Return a cubic cell-centered grid and its uniform grid spacing."""

    dx = 2.0 * half_width / num_points
    axis = -half_width + (jnp.arange(num_points, dtype=jnp.float64) + 0.5) * dx
    X, Y, Z = jnp.meshgrid(axis, axis, axis, indexing="ij")
    grid = jnp.stack((X, Y, Z), axis=-1)

    return grid, dx


def brill_lindquist_conformal_factor(masses, positions, grid):
    """Compute ``psi_BL = 1 + sum_a m_a / (2 r_a)``."""

    masses = jnp.asarray(masses, dtype=grid.dtype)
    positions = jnp.asarray(positions, dtype=grid.dtype)

    displacement = grid[None, ...] - positions[:, None, None, None, :]
    radius_squared = jnp.einsum("a...i,a...i->a...", displacement, displacement)
    radius = jnp.sqrt(radius_squared)
    puncture_terms = masses[:, None, None, None] / (2.0 * radius)

    return 1.0 + jnp.einsum("a...->...", puncture_terms)


def bowen_york_extrinsic_curvature(momenta, positions, grid):
    """Compute the conformal Bowen-York curvature for boosted punctures."""

    momenta = jnp.asarray(momenta, dtype=grid.dtype)
    positions = jnp.asarray(positions, dtype=grid.dtype)

    displacement = grid[None, ...] - positions[:, None, None, None, :]
    radius_squared = jnp.einsum("a...i,a...i->a...", displacement, displacement)
    radius = jnp.sqrt(radius_squared)
    normal = displacement / radius[..., None]

    momentum_dot_normal = jnp.einsum("ai,a...i->a...", momenta, normal)
    momentum_normal = jnp.einsum("ai,a...j->a...ij", momenta, normal)
    normal_momentum = jnp.einsum("a...i,aj->a...ij", normal, momenta)
    normal_normal = jnp.einsum("a...i,a...j->a...ij", normal, normal)

    identity = jnp.eye(3, dtype=grid.dtype)
    transverse_metric = identity - normal_normal
    curvature = 3.0 / (2.0 * radius[..., None, None] ** 2) * (
        momentum_normal
        + normal_momentum
        - transverse_metric * momentum_dot_normal[..., None, None]
    )

    return jnp.einsum("a...ij->...ij", curvature)


@jax.jit
def contract_conformal_curvature(conformal_curvature):
    """Compute the flat-metric contraction ``A_tilde_ij A_tilde^ij``."""

    return jnp.einsum(
        "...ij,...ij->...",
        conformal_curvature,
        conformal_curvature,
    )


@jax.jit
def laplacian(interior_field, dx):
    """Apply the seven-point Laplacian with zero values outside the interior."""

    field = jnp.pad(interior_field, 1)

    return (
        field[2:, 1:-1, 1:-1]
        + field[:-2, 1:-1, 1:-1]
        + field[1:-1, 2:, 1:-1]
        + field[1:-1, :-2, 1:-1]
        + field[1:-1, 1:-1, 2:]
        + field[1:-1, 1:-1, :-2]
        - 6.0 * interior_field
    ) / dx**2


@jax.jit
def hamiltonian_source(psi, curvature_squared):
    """Compute ``S(psi) = A_tilde_ij A_tilde^ij psi^-7 / 8``."""

    return 1.0 / 8.0 * curvature_squared * psi**-7


@jax.jit
def linearized_hamiltonian_source(delta_u, psi_background, curvature_squared):
    """Compute ``delta S = -7 A_tilde^2 psi_background^-8 delta_u / 8``."""

    return -7.0 / 8.0 * curvature_squared * psi_background**-8 * delta_u


@jax.jit
def linearized_hamiltonian_operator(
    delta_u,
    psi_background,
    curvature_squared,
    dx,
):
    """Apply ``delta F = Laplacian(delta_u) + delta S``."""

    linearized_source = linearized_hamiltonian_source(
        delta_u,
        psi_background,
        curvature_squared,
    )

    return laplacian(delta_u, dx) + linearized_source


@jax.jit
def hamiltonian_residual(u, psi_bl, conformal_curvature, dx):
    """Evaluate the nonlinear Hamiltonian residual at interior grid points."""

    u_interior = u[1:-1, 1:-1, 1:-1]
    psi_interior = psi_bl[1:-1, 1:-1, 1:-1] + u_interior

    curvature_squared = contract_conformal_curvature(conformal_curvature)
    curvature_squared_interior = curvature_squared[1:-1, 1:-1, 1:-1]
    source = hamiltonian_source(psi_interior, curvature_squared_interior)

    return laplacian(u_interior, dx) + source


def solve_conformal_factor(
    masses,
    positions,
    momenta,
    grid,
    dx,
    newton_tolerance=1.0e-10,
    max_newton_iterations=12,
    cg_tolerance=1.0e-10,
    max_cg_iterations=1000,
    verbose=False,
):
    """Solve the puncture Hamiltonian constraint with Newton iterations.

    The singular Brill-Lindquist part is analytic. The numerical unknown is
    the regular correction ``u``, fixed to zero on all six outer grid layers.
    Each Newton correction is obtained with matrix-free conjugate gradients
    applied to the negative linearized Hamiltonian operator.

    Returns
    -------
    psi : jax.Array
        Total conformal factor ``psi_bl + u`` on the full grid.
    u : jax.Array
        Regular correction, including its zero-valued outer boundary.
    residual_history : list[float]
        Interior RMS residual before each Newton correction and after the
        final correction.
    """

    psi_bl = brill_lindquist_conformal_factor(masses, positions, grid)
    conformal_curvature = bowen_york_extrinsic_curvature(momenta, positions, grid)
    curvature_squared = contract_conformal_curvature(conformal_curvature)

    psi_bl_interior = psi_bl[1:-1, 1:-1, 1:-1]
    curvature_squared_interior = curvature_squared[1:-1, 1:-1, 1:-1]

    u_n = jnp.zeros_like(psi_bl_interior)
    residual_history = []

    for iteration in range(max_newton_iterations + 1):
        # Re-linearize about psi_n = psi_BL + u_n after every correction.
        psi_n = psi_bl_interior + u_n
        source_n = hamiltonian_source(psi_n, curvature_squared_interior)
        residual_n = laplacian(u_n, dx) + source_n

        residual_rms = float(jnp.sqrt(jnp.mean(residual_n**2)))
        residual_history.append(residual_rms)

        if verbose:
            print(f"Newton iteration {iteration:2d}: RMS residual = {residual_rms:.3e}")

        if residual_rms <= newton_tolerance:
            break

        if iteration == max_newton_iterations:
            raise RuntimeError(
                "Newton-Raphson solve did not converge: "
                f"RMS residual {residual_rms:.3e} exceeds {newton_tolerance:.3e}."
            )

        def negative_linearized_operator(delta_u):
            return -linearized_hamiltonian_operator(
                delta_u,
                psi_n,
                curvature_squared_interior,
                dx,
            )

        # -[Laplacian(delta_u) + delta S_n] = F(u_n) is SPD for psi_n > 0.
        delta_u, _ = cg(
            negative_linearized_operator,
            residual_n,
            tol=cg_tolerance,
            atol=0.0,
            maxiter=max_cg_iterations,
        )
        u_n = u_n + delta_u
        jax.block_until_ready(u_n)

    u = jnp.pad(u_n, 1)
    psi = psi_bl + u

    return psi, u, residual_history
