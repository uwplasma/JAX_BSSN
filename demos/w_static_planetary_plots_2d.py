import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

# Enable 64-bit precision to match the simulation
jax.config.update("jax_enable_x64", True)

# Load the saved data
data = jnp.load("simulation_results_2d.npz")
times_mass = data['times_mass']
lapse_snaps = data['lapse_snaps']
r3d_mass = data['r3d_mass']
R_sphere = float(data['R_sphere'])
rho_geom = float(data['rho_geom'])

def newtonian_potential_f64(r, M, R):
    inside  = -M / (2.0 * R) * (3.0 - r**2 / R**2)
    outside = -M / r
    return jnp.where(r <= R, inside, outside)

# Keep calculations in Mm (simulation units)
M_geom_sphere_f64 = (4.0/3.0) * jnp.pi * R_sphere**3 * rho_geom
r_ref = jnp.linspace(0.01, 2.0 * R_sphere, 500)
Phi_ref_f64 = newtonian_potential_f64(r_ref, M_geom_sphere_f64, R_sphere)
alpha_dev_newton = Phi_ref_f64   

# BSSN radial lapse profile processing (keep in Mm)
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
bssn_dev = alpha_radial - 1.0

# ---- Plots ----
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: lapse deviation (alpha - 1)
axes[0].plot(r_ref, alpha_dev_newton, 'k--', lw=2,
             label=r'Newtonian $\Phi(r)$')
axes[0].plot(r_centers, bssn_dev, 'o-', ms=4, lw=1.2,
             label=f'BSSN $\\alpha - 1$ (t = {float(times_mass[-1]):.2e} m)')
axes[0].axvline(float(R_sphere), ls=':', color='grey', label=f'R = {float(R_sphere):.2e} m')
axes[0].set_xlabel('r  [m]')
axes[0].set_ylabel(r'$\alpha - 1$')
axes[0].set_title('Lapse deviation vs Newtonian potential')
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
axes[0].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# Right: zoom — force y-axis to Newtonian scale
Phi_surface = newtonian_potential_f64(np.array([float(R_sphere)]),
                                       M_geom_sphere_f64, float(R_sphere))[0]
y_lo = 1.5 * Phi_surface  # ~-1.5e-9 (negative, so 1.5× makes it more negative)
y_hi = -0.3 * Phi_surface  # small positive offset
axes[1].plot(r_ref, alpha_dev_newton, 'k--', lw=2, label=r'Newtonian $\Phi(r)$')
axes[1].plot(r_centers, bssn_dev, 'o-', ms=4, lw=1.2, color='tab:red',
             label=f'BSSN $\\alpha - 1$')
axes[1].axvline(float(R_sphere), ls=':', color='grey')
axes[1].set_ylim(y_lo, y_hi)
axes[1].set_xlabel('r  [m]')
axes[1].set_ylabel(r'$\alpha - 1$')
axes[1].set_title(f'Zoomed to Newtonian scale  (Φ surface ≈ {Phi_surface:.2e})')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)
axes[1].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

plt.savefig("planetary_mass_plot_2d.png", dpi=300)
print("Plot saved as planetary_mass_plot_2d.png")

print(f'\n--- Precision diagnostic ---')
print(f'Newtonian Φ(R_surface) = {float(Phi_surface):.4e}')
bssn_surface_idx = jnp.argmin(jnp.abs(r_centers - float(R_sphere)))
print(f'BSSN (α-1) at surface  = {bssn_dev[bssn_surface_idx]:.4e}')
print(f'float64 epsilon at 1.0 = {jnp.finfo(jnp.float64).eps:.2e}')
print(f'Signal / float64 eps   = {abs(float(Phi_surface)) / jnp.finfo(jnp.float64).eps:.1f}')
