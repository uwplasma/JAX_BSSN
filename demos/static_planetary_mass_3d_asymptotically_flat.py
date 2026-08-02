import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from demos.static_planetary_mass_3d import (
    run_static_mass_3d,
    make_rho_field,
    newtonian_potential_uniform_sphere,
    radial_bin_average,
    extract_metric_potential,
    shift_potential_zero_mode,
)

"""Asymptotically-flat comparison for the static planetary mass demo.

The underlying BSSN evolution stencil in JAX_BSSN remains periodic. This
script keeps that evolution path unchanged, then compares the extracted metric
potential against an isolated-sphere Newtonian reference by anchoring the
metric potential to zero on the outer boundary.
"""


if __name__ == "__main__":
    print("Running Earth simulation with asymptotically-flat comparison")

    G_SI = 6.67430e-11
    c_SI = 2.99792458e8
    G_over_c2_m = G_SI / (c_SI**2)

    R_sphere = 6.3817e6
    rho_SI = 5513.0
    rho_geom = G_over_c2_m * rho_SI

    M_sphere = (4.0 / 3.0) * jnp.pi * R_sphere**3 * rho_SI
    M_geom = G_over_c2_m * M_sphere

    L_domain = 10.0 * R_sphere
    nx_mass = 172
    dt_fac = 0.4
    Nt_mass = 2000
    snap_stride = 100
    n_bins = 60

    print(f"Earth density (SI):        {rho_SI:.4f} kg/m^3")
    print(f"G/c^2 (m/kg):              {G_over_c2_m:.4e}")
    print(f"rho_geom (1/m^2):          {rho_geom:.4e}")
    print(f"Sphere mass:               {float(M_sphere):.4e} kg")
    print(f"Asymptotic comparison box: {L_domain:.4e} m")
    print(f"dx = {L_domain / nx_mass:.4f} m, dt = {dt_fac * L_domain / nx_mass:.4e} m")

    times_mass, lapse_snaps, coords_mass, r3d_mass, rho_field_mass, dx_mass, final_vars = run_static_mass_3d(
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

    print(f"\nEvolution finished: {len(times_mass)} snapshots over t=[0, {float(times_mass[-1]):.4f}] m")
    print("\nGenerating asymptotically-flat comparison plots...")

    alpha_final, spatial_trace, phi_metric_raw = extract_metric_potential(final_vars)
    phi_metric_asymptotic, boundary_offset = shift_potential_zero_mode(phi_metric_raw, "boundary_mean")

    r_centers, phi_metric_asymptotic_radial, bin_idx = radial_bin_average(
        phi_metric_asymptotic,
        r3d_mass,
        n_bins=n_bins,
    )
    _, phi_metric_raw_radial, _ = radial_bin_average(phi_metric_raw, r3d_mass, n_bins=n_bins)

    r_ref = jnp.linspace(0.01, 2.0 * float(R_sphere), 500)
    Phi_ref_isolated = newtonian_potential_uniform_sphere(r_ref, M_geom, R_sphere)
    Phi_surface = float(newtonian_potential_uniform_sphere(
        jnp.asarray([float(R_sphere)]),
        M_geom,
        R_sphere,
    )[0])

    surface_idx = int(jnp.argmin(jnp.abs(r_centers - float(R_sphere))))
    y_lo = 1.5 * Phi_surface
    y_hi = -0.3 * Phi_surface

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(r_ref, Phi_ref_isolated, 'k--', lw=2,
                 label=r'Isolated Newtonian $\Phi_{\mathrm{iso}}$')
    axes[0].plot(r_centers, phi_metric_asymptotic_radial, '-', ms=4, lw=1.2,
                 label=r'BSSN metric $\Phi_{\mathrm{metric}} - \langle \Phi \rangle_{\partial \Omega}$')
    axes[0].axvline(float(R_sphere), ls=':', color='grey', label=f'R = {float(R_sphere):.2e} m')
    axes[0].set_xlabel('r  [m]')
    axes[0].set_ylabel(r'$\Phi$')
    axes[0].set_title('Asymptotically-flat comparison to isolated Newtonian potential')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    axes[1].plot(r_ref, Phi_ref_isolated, 'k--', lw=2,
                 label=r'Isolated Newtonian $\Phi_{\mathrm{iso}}$')
    axes[1].plot(r_centers, phi_metric_asymptotic_radial, 'o-', ms=4, lw=1.2, color='tab:blue',
                 label=r'Boundary-shifted BSSN metric')
    axes[1].axvline(float(R_sphere), ls=':', color='grey')
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].set_xlabel('r  [m]')
    axes[1].set_ylabel(r'$\Phi$')
    axes[1].set_title(f'Zoomed isolated comparison  (Φ surface ≈ {Phi_surface:.2e})')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig('asymptotically_flat_planetary_mass_plot.png', dpi=300)
    print('Plot saved as asymptotically_flat_planetary_mass_plot.png')

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(r_ref, Phi_ref_isolated, 'k--', lw=2,
             label=r'Isolated Newtonian $\Phi_{\mathrm{iso}}$')
    ax2.plot(r_centers, phi_metric_raw_radial, ':', ms=4, lw=1.0,
             label=r'Raw BSSN metric $\Phi_{\mathrm{metric}}$')
    ax2.plot(r_centers, phi_metric_asymptotic_radial, 'o-', ms=4, lw=1.2,
             label=r'Boundary-shifted BSSN metric $\Phi_{\mathrm{metric}}$')
    ax2.axvline(float(R_sphere), ls=':', color='grey', label=f'R = {float(R_sphere):.2e} m')
    ax2.set_xlabel('r  [m]')
    ax2.set_ylabel(r'$\Phi$')
    ax2.set_title('Asymptotically-flat boundary anchoring diagnostic')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    fig2.tight_layout()
    fig2.savefig('asymptotically_flat_boundary_diagnostic.png', dpi=300)
    print('Plot saved as asymptotically_flat_boundary_diagnostic.png')

    alpha_radial = jnp.array([
        float(jnp.asarray(alpha_final).ravel()[bin_idx == i].mean()) if jnp.any(bin_idx == i) else jnp.nan
        for i in range(n_bins)
    ])
    bssn_dev = (alpha_radial**2 - 1.0) / 2.0
    spatial_trace_radial = jnp.array([
        float(spatial_trace.ravel()[bin_idx == i].mean()) if jnp.any(bin_idx == i) else jnp.nan
        for i in range(n_bins)
    ])

    print("\n--- Asymptotically-flat diagnostic ---")
    print(f"Isolated Newtonian Φ(R_surface) = {Phi_surface:.4e}")
    print(f"BSSN 1/2(α²-1) at surface       = {bssn_dev[surface_idx]:.4e}")
    print(f"BSSN 1/6 tr(δg) at surface      = {spatial_trace_radial[surface_idx] / 6.0:.4e}")
    print(f"Boundary offset removed         = {float(boundary_offset):.4e}")
    print(f"Boundary-shifted Φ at surface   = {phi_metric_asymptotic_radial[surface_idx]:.4e}")
    print(f"float64 epsilon at 1.0          = {jnp.finfo(jnp.float64).eps:.2e}")
