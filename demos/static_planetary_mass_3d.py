import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from functools import partial
from jax import jit, lax

from JAX_BSSN import setup_jax_config
setup_jax_config(enable_x64=True, verbose=True)

from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.bssn import BSSNVariables, BSSNParameters, compute_momentum_constraint
from JAX_BSSN.errors import compute_hamiltonian_constraint
from JAX_BSSN.derivatives import diff1_field
from JAX_BSSN.tensor_algebra import (
    invert_3x3_metric,
    christoffel_symbols_second_kind,
    trace_tensor,
    traceless_part,
)

"""Evolve flat spacetime + spherical mass density in 3D BSSN.

Initialises Minkowski initial data (alpha=1, flat metric, K=0)
with a caller-supplied density field.

Params:
- nx: number of grid points along each axis (total grid is nx^3)
- L: physical size of the domain along each axis (total domain is L^3)
- dt_factor: factor to multiply the CFL time step by (e.g. 0.5 for safety)
- Nt: total number of time steps to evolve
- snapshot_stride: how often to save snapshots of the lapse for plotting
- rho_field_fn: function that takes the 1-D coordinate array and returns the mass-density field in geometric units
- show_progress: whether to display a progress bar during evolution
"""
def run_static_mass_3d(R_sphere, rho_geom, nx, L, dt_factor, Nt, snapshot_stride, rho_field_fn,
                    show_progress=True):
    dx = L / nx
    dt = dt_factor * dx
    coords = -L/2 + (jnp.arange(nx) + 0.5) * dx  # cell-centred

    # Build the density on this grid via the user-supplied function
    rho_field = rho_field_fn(coords, R_sphere, rho_geom)

    x3d, y3d, z3d = jnp.meshgrid(coords, coords, coords, indexing='ij')
    r3d = jnp.sqrt(x3d**2 + y3d**2 + z3d**2)
    r3d = jnp.maximum(r3d, 1e-30)

    # Flat-space BSSN initial data
    alpha = jnp.ones((nx, nx, nx))
    beta  = jnp.zeros((3, nx, nx, nx))
    conformal_factor = jnp.ones((nx, nx, nx))

    gamma = jnp.zeros((3, 3, nx, nx, nx))
    gamma = gamma.at[0, 0].set(1.0)
    gamma = gamma.at[1, 1].set(1.0)
    gamma = gamma.at[2, 2].set(1.0)

    traceless_K = jnp.zeros((3, 3, nx, nx, nx))
    trace_K     = jnp.zeros((nx, nx, nx))
    conformal_connection = jnp.zeros((3, nx, nx, nx))

    vars = BSSNVariables(
        conformal_metric=gamma,
        conformal_factor=conformal_factor,
        traceless_K=traceless_K,
        trace_K=trace_K,
        conformal_connection=conformal_connection,
        lapse=alpha,
        shift=beta,
        rho=rho_field,
    )

    params = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        nu=0.25,
        f=1.0,      # 1+log slicing -> lapse settles to ~ 1 + Phi/c^2
        g=0.0,
        dx=dx,
        dt=dt,
    )

    # Snapshot storage
    times_list = [0.0]
    lapse_snapshots = [vars.lapse]

    iterator = range(1, Nt + 1)
    if show_progress:
        iterator = tqdm(iterator, leave=False)

    for step in iterator:
        vars = rk4_step(vars, params)
        if (step % snapshot_stride == 0) or (step == Nt):
            times_list.append(step * dt)
            lapse_snapshots.append(vars.lapse)
    
    return (
        jnp.asarray(times_list),
        lapse_snapshots,
        coords,
        r3d,
        rho_field,
        dx,
    )

"""Create spherical density field centered at origin."""
"""Params:
- coords: 1D array of coordinate values along one axis (assuming uniform grid)
- R_sphere: radius of the spherical mass distribution
- rho_geom: uniform density within the sphere (geometric units)"""
def make_rho_field(coords, R_sphere, rho_geom):
    # Create the 3D grid from the 1D coordinate arrays provided by the solver
    X, Y, Z = jnp.meshgrid(coords, coords, coords, indexing='ij')
    
    # Calculate radial distance from the center (0,0,0)
    r_squared = X**2 + Y**2 + Z**2
    
    # Apply the density condition
    rho_field = jnp.where(r_squared < R_sphere**2, rho_geom, 0.0)
    return rho_field

if __name__ == "__main__":
    print("Running Earth simulation")
    # Initialize parameters for the static mass evolution
    # Earth density (SI): 5513 kg/m^3
    #
    # To convert mass density to geometric units (1/m^2):
    #   rho_geom = (G / c^2) * rho_SI
    #
    # G = 6.67430e-11 m^3 kg^-1 s^-2
    # c = 2.99792e8   m s^-1
    # G/c^2 = 7.426e-28 m/kg

    G_SI = 6.67430e-11          # m^3 kg^-1 s^-2
    c_SI = 2.99792458e8         # m s^-1

    # G/c^2 in m/kg
    G_over_c2_m = G_SI / (c_SI**2) 

    R_sphere = 6.3817e6         # meters (neutron star)
    rho_SI = 5513.0             # kg/m^3

    # rho_geom has units of 1/m^2
    rho_geom = G_over_c2_m * rho_SI   

    M_sphere = (4.0/3.0) * jnp.pi * R_sphere**3 * rho_SI   # kg
    M_geom   = G_over_c2_m * M_sphere # Dimensionless mass in units of meters

    print(f'Earth density (SI):        {rho_SI:.4f} kg/m^3')
    print(f'G/c^2 (m/kg):              {G_over_c2_m:.4e}')
    print(f'rho_geom (1/m^2):          {rho_geom:.4e}')
    print(f'Sphere mass:               {float(M_sphere):.4e} kg')
    print(f'Phi/c^2 at surface ~ GM/(c^2 R) = {float(G_over_c2_m * M_sphere / R_sphere):.4e}')

    # Run the static-mass evolution
    L_domain = 10.0 * R_sphere    # ~25.5 million meters
    nx_mass  = 172                 
    dt_fac   = 0.4                # CFL factor
    Nt_mass  = 500               # 2*10^3 timesteps
    snap_stride = 25

    print(f'Domain: [{-L_domain/2:.2f}, {L_domain/2:.2f}] m   ({nx_mass}^3 grid)')
    print(f'dx = {L_domain/nx_mass:.4f} m,  dt = {dt_fac * L_domain/nx_mass:.4e} m')
    print(f'Nt = {Nt_mass},  rho_geom = {rho_geom:.4e} m^-2')

    times_mass, lapse_snaps, coords_mass, r3d_mass, rho_field_mass, dx_mass = run_static_mass_3d(
        R_sphere=R_sphere,
        rho_geom=rho_geom,
        nx=nx_mass,
        L=L_domain,
        dt_factor=dt_fac,
        Nt=Nt_mass,
        snapshot_stride=snap_stride,
        rho_field_fn=make_rho_field,
        show_progress=True,
    )

    print(f'\nEvolution finished: {len(times_mass)} snapshots over t=[0, {float(times_mass[-1]):.4f}] m')

    # ---- PLOTTING SECTION (in-memory) ----
    # Use the in-memory arrays produced by the evolution (no file I/O)
    print("\nGenerating plots (in-memory)...")
    R_sphere_plot = float(R_sphere)
    rho_geom_plot = float(rho_geom)
    # times_mass, lapse_snaps, r3d_mass are already in memory from the run

    def newtonian_potential_f64(r, M, R):
        inside  = -M / (2.0 * R) * (3.0 - r**2 / R**2)
        outside = -M / r
        return jnp.where(r <= R, inside, outside)

    # Keep calculations in meters (simulation units)
    M_geom_sphere_f64 = (4.0/3.0) * jnp.pi * R_sphere_plot**3 * rho_geom_plot
    r_ref = jnp.linspace(0.01, 2.0 * R_sphere_plot, 500)
    Phi_ref_f64 = newtonian_potential_f64(r_ref, M_geom_sphere_f64, R_sphere_plot)
    alpha_dev_newton = Phi_ref_f64   

    # BSSN radial lapse profile processing (keep in meters)
    alpha_final = jnp.asarray(lapse_snaps[-1], dtype=jnp.float64)
    r_flat = jnp.asarray(r3d_mass).ravel()
    a_flat = alpha_final.ravel()

    n_bins = 60
    r_edges = jnp.linspace(0, float(r_flat.max()), n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    bin_idx = jnp.digitize(r_flat, r_edges) - 1
    alpha_radial = jnp.array([
        a_flat[bin_idx == i].mean() if jnp.any(bin_idx == i) else jnp.nan
        for i in range(n_bins)
    ])
    bssn_dev = (alpha_radial**2 - 1.0)/2

    # ---- Plots ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: lapse deviation (alpha - 1)
    axes[0].plot(r_ref, alpha_dev_newton, 'k--', lw=2,
                 label=r'Newtonian $\Phi(r)$')
    axes[0].plot(r_centers, bssn_dev, '-', ms=4, lw=1.2,
                 label=f'BSSN $\\frac{{1}}{{2}}(\\alpha^2 - 1)$ (t = {float(times_mass[-1]):.2e} m)')
    axes[0].axvline(float(R_sphere_plot), ls=':', color='grey', label=f'R = {float(R_sphere_plot):.2e} m')
    axes[0].set_xlabel('r  [m]')
    axes[0].set_ylabel(r'$\alpha - 1$')
    axes[0].set_title('Lapse deviation vs Newtonian potential')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    # Right: zoom — force y-axis to Newtonian scale
    Phi_surface = newtonian_potential_f64(np.array([float(R_sphere_plot)]),
                                           M_geom_sphere_f64, float(R_sphere_plot))[0]
    y_lo = 1.5 * Phi_surface  # ~-1.5e-9 (negative, so 1.5× makes it more negative)
    y_hi = -0.3 * Phi_surface  # small positive offset
    axes[1].plot(r_ref, alpha_dev_newton, 'k--', lw=2, label=r'Newtonian $\Phi(r)$')
    axes[1].plot(r_centers, bssn_dev, 'o-', ms=4, lw=1.2, color='tab:red',
                 label=f'BSSN $\\alpha - 1$')
    axes[1].axvline(float(R_sphere_plot), ls=':', color='grey')
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].set_xlabel('r  [m]')
    axes[1].set_ylabel(r'$\alpha - 1$')
    axes[1].set_title(f'Zoomed to Newtonian scale  (Φ surface ≈ {Phi_surface:.2e})')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    plt.savefig("planetary_mass_plot.png", dpi=300)
    print("Plot saved as planetary_mass_plot.png")

    # ---- Poisson potential vs BSSN lapse comparison ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(r_ref, alpha_dev_newton, 'k--', lw=2,
             label=r'Poisson potential $\Phi(r)$')
    ax2.plot(r_centers, bssn_dev, 'o-', ms=4, lw=1.2,
             label=f'BSSN $\\alpha - 1$ (t = {float(times_mass[-1]):.2e} m)')
    ax2.axvline(float(R_sphere_plot), ls=':', color='grey',
                label=f'R = {float(R_sphere_plot):.2e} m')
    ax2.set_xlabel('r  [m]')
    ax2.set_ylabel(r'$\Phi(r) / \alpha - 1$')
    ax2.set_title('Poisson potential vs BSSN lapse deviation')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    fig2.tight_layout()
    fig2.savefig("poisson_vs_bssn_lapse.png", dpi=300)
    print("Plot saved as poisson_vs_bssn_lapse.png")

    @partial(jit, static_argnames=("tol", "max_iter"))
    def solve_poisson_with_conjugate_gradient(rho, phi, world, tol=1e-8, max_iter=5000):
        """
        Solve Poisson's equation ∇^2 φ = 4πρ with periodic boundary conditions.

        The periodic Laplacian is computed with rolls, and the source is mean-subtracted
        so the equation is solvable on a periodic domain.

        Args:
            rho (ndarray): Mass density field with shape (Nx, Ny, Nz).
            phi (ndarray): Initial guess for potential with shape (Nx, Ny, Nz).
            world (dict): Simulation metadata with ``dx``, ``dy``, and ``dz``.
            tol (float): Residual tolerance.
            max_iter (int): Maximum number of conjugate gradient iterations.

        Returns:
            ndarray: Periodic potential field with shape (Nx, Ny, Nz).
        """
        dx = world["dx"]
        dy = world["dy"]
        dz = world["dz"]

        def lapl(field):
            return (
                (jnp.roll(field, -1, axis=0) + jnp.roll(field, 1, axis=0) - 2.0 * field) / (dx * dx)
                + (jnp.roll(field, -1, axis=1) + jnp.roll(field, 1, axis=1) - 2.0 * field) / (dy * dy)
                + (jnp.roll(field, -1, axis=2) + jnp.roll(field, 1, axis=2) - 2.0 * field) / (dz * dz)
            )

        phi = phi
        source = 4.0 * jnp.pi * (rho - jnp.mean(rho))
        r = source - lapl(phi)
        p = r

        def body_fun(state):
            phi, r, p, k = state
            Ap = lapl(p)
            alpha = jnp.sum(r * r) / jnp.sum(p * Ap)
            phi_next = phi + alpha * p
            r_next = r - alpha * Ap
            beta = jnp.sum(r_next * r_next) / jnp.sum(r * r)
            p_next = r_next + beta * p
            return phi_next, r_next, p_next, k + 1

        def cond_fun(state):
            _, r, _, k = state
            norm_r = jnp.sum(r * r)
            return jnp.logical_and(k < max_iter, norm_r > tol**2)

        phi, _, _, _ = lax.while_loop(
            cond_fun,
            body_fun,
            (phi, r, p, 0),
        )
        return phi

    def run_cg_convergence_test():
        """Test CG convergence by comparing against a periodic analytic solution."""
        L = 1.0
        kx = ky = kz = 1
        dx_values = [1.0 / n for n in (16, 24, 32, 48, 64)]
        errors = []

        for dx in dx_values:
            nx = int(round(L / dx))
            x = jnp.linspace(0.0, L, nx, endpoint=False)
            X, Y, Z = jnp.meshgrid(x, x, x, indexing='ij')

            phi_exact = jnp.sin(2.0 * jnp.pi * kx * X / L)
            phi_exact *= jnp.sin(2.0 * jnp.pi * ky * Y / L)
            phi_exact *= jnp.sin(2.0 * jnp.pi * kz * Z / L)

            rho = -3.0 * jnp.pi * phi_exact
            phi0 = jnp.zeros_like(phi_exact)
            world = {'dx': dx, 'dy': dx, 'dz': dx}

            phi_num = solve_poisson_with_conjugate_gradient(rho, phi0, world,
                                                            tol=1e-10, max_iter=2000)

            errors.append(float(jnp.sqrt(jnp.mean((phi_num - phi_exact) ** 2))))

        dxs = np.array(dx_values)
        errors = np.array(errors)
        slope, intercept = np.polyfit(np.log(dxs), np.log(errors), 1)
        fit_line = np.exp(intercept) * dxs**slope

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.loglog(dxs, errors, 'o-', label='CG L2 error')
        ax3.loglog(dxs, fit_line, '--', color='grey',
                   label=f'fit slope = {slope:.2f}')
        ax3.set_xlabel(r'$\Delta x$')
        ax3.set_ylabel(r'$L_2$ error')
        ax3.set_title('Conjugate gradient Poisson convergence')
        ax3.grid(True, which='both', alpha=0.3)
        ax3.legend(fontsize=10)
        fig3.tight_layout()
        fig3.savefig('cg_convergence_test.png', dpi=300)
        print(f'CG convergence test saved as cg_convergence_test.png')
        print(f'  dx: {dxs}')
        print(f'  error: {errors}')
        print(f'  measured order = {slope:.2f}')

        return dxs, errors, slope

    print(f'\n--- Precision diagnostic ---')
    print(f'Newtonian Φ(R_surface) = {float(Phi_surface):.4e}')
    bssn_surface_idx = jnp.argmin(jnp.abs(r_centers - float(R_sphere_plot)))
    print(f'BSSN (α-1) at surface  = {bssn_dev[bssn_surface_idx]:.4e}')
    print(f'float64 epsilon at 1.0 = {jnp.finfo(jnp.float64).eps:.2e}')
    print(f'Signal / float64 eps   = {abs(float(Phi_surface)) / jnp.finfo(jnp.float64).eps:.1f}')

    run_cg_convergence_test()