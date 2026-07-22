import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters, BSSNVariables, compute_physical_metric
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.tensor_algebra import invert_3x3_metric, trace_tensor


def make_rho_field(coords, radius, rho_geom):
    """Create a uniform spherical density field centered at the origin."""
    x3d, y3d, z3d = jnp.meshgrid(coords, coords, coords, indexing="ij")
    r_squared = x3d**2 + y3d**2 + z3d**2
    return jnp.where(r_squared < radius**2, rho_geom, 0.0)


def invert_periodic_laplacian_zero_mean(source, dx):
    """Invert the periodic Laplacian on the zero-mean subspace."""
    source_np = np.asarray(source, dtype=np.float64)
    source_np = source_np - source_np.mean()

    n_x, n_y, n_z = source_np.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n_y, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(n_z, d=dx)
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k_squared = kx3d * kx3d + ky3d * ky3d + kz3d * kz3d

    source_hat = np.fft.fftn(source_np)
    solution_hat = np.zeros_like(source_hat)
    nonzero_mask = k_squared > 0.0
    solution_hat[nonzero_mask] = -source_hat[nonzero_mask] / k_squared[nonzero_mask]

    return jnp.asarray(np.fft.ifftn(solution_hat).real)


def solve_periodic_hamiltonian_constraint(
    rho_field,
    dx,
    tol=1e-13,
    max_iter=100,
    relaxation=0.6,
    k_sign=-1.0,
):
    """Solve the periodic conformally flat CMC Hamiltonian constraint."""
    psi = jnp.ones_like(rho_field, dtype=jnp.float64)
    k_value = jnp.array(0.0, dtype=jnp.float64)

    for _ in range(max_iter):
        psi_power = psi**5
        k_squared = 24.0 * jnp.pi * jnp.mean(rho_field * psi_power) / jnp.mean(psi_power)
        source = (k_squared / 12.0 - 2.0 * jnp.pi * rho_field) * psi_power
        source = source - jnp.mean(source)

        psi_target = 1.0 + invert_periodic_laplacian_zero_mean(source, dx)
        psi_next = (1.0 - relaxation) * psi + relaxation * psi_target
        psi_next = jnp.maximum(psi_next, 1e-14)

        max_update = jnp.max(jnp.abs(psi_next - psi))
        psi = psi_next
        k_value = k_sign * jnp.sqrt(jnp.maximum(k_squared, 0.0))
        if float(max_update) < tol:
            break

    return psi, k_value


def build_periodic_hamiltonian_initial_data(rho_field, dx):
    """Construct periodic conformally flat CMC initial data."""
    psi, k_value = solve_periodic_hamiltonian_constraint(rho_field, dx)

    conformal_factor = psi**(-2)
    conformal_metric = jnp.zeros((3, 3) + rho_field.shape)
    conformal_metric = conformal_metric.at[0, 0].set(1.0)
    conformal_metric = conformal_metric.at[1, 1].set(1.0)
    conformal_metric = conformal_metric.at[2, 2].set(1.0)

    traceless_k = jnp.zeros((3, 3) + rho_field.shape)
    trace_k = jnp.full(rho_field.shape, k_value)
    conformal_connection = jnp.zeros((3,) + rho_field.shape)
    shift = jnp.zeros((3,) + rho_field.shape)

    phi_init = 0.5 * (1.0 - psi**4)
    lapse = jnp.sqrt(jnp.maximum(1.0 + 2.0 * phi_init, 1e-14))

    return lapse, shift, conformal_factor, conformal_metric, traceless_k, trace_k, conformal_connection


def run_static_mass_3d(radius, rho_geom, nx, domain_size, dt_factor, steps, show_progress=True):
    """Evolve periodic CMC initial data for a uniform spherical density field."""
    dx = domain_size / nx
    dt = dt_factor * dx
    coords = -domain_size / 2.0 + (jnp.arange(nx) + 0.5) * dx
    rho_field = make_rho_field(coords, radius, rho_geom)

    x3d, y3d, z3d = jnp.meshgrid(coords, coords, coords, indexing="ij")
    r3d = jnp.maximum(jnp.sqrt(x3d**2 + y3d**2 + z3d**2), 1e-30)

    lapse, shift, conformal_factor, conformal_metric, traceless_k, trace_k, conformal_connection = (
        build_periodic_hamiltonian_initial_data(rho_field, dx)
    )

    vars = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_k,
        trace_K=trace_k,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
        rho=rho_field,
    )
    params = BSSNParameters(
        eta=0.0,
        kappa=1.0,
        nu=0.25,
        f=1.0,
        g=0.0,
        dx=dx,
        dt=dt,
    )

    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, leave=False)

    for _ in iterator:
        vars = rk4_step(vars, params)

    return {
        "dx": dx,
        "dt": dt,
        "final_time": steps * dt,
        "r3d": r3d,
        "rho_field": rho_field,
        "vars": vars,
    }


def radial_bin_average(field, r3d, n_bins=60):
    """Average a 3D field over spherical radial bins."""
    field_flat = jnp.asarray(field).ravel()
    r_flat = jnp.asarray(r3d).ravel()
    r_edges = jnp.linspace(0.0, float(r_flat.max()), n_bins + 1)
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    bin_idx = jnp.clip(jnp.digitize(r_flat, r_edges) - 1, 0, n_bins - 1)
    radial_profile = jnp.array(
        [
            float(field_flat[bin_idx == i].mean()) if jnp.any(bin_idx == i) else jnp.nan
            for i in range(n_bins)
        ]
    )
    return r_centers, radial_profile, bin_idx


def extract_metric_potential(final_vars):
    """Compute the weak-field metric potential from the final physical metric."""
    alpha_final = jnp.asarray(final_vars.lapse, dtype=jnp.float64)
    final_physical_metric = compute_physical_metric(
        final_vars.conformal_metric,
        final_vars.conformal_factor,
    )
    reference_metric = jnp.eye(3)[..., jnp.newaxis, jnp.newaxis, jnp.newaxis]
    reference_factor = jnp.ones_like(final_vars.conformal_factor)
    initial_physical_metric = compute_physical_metric(reference_metric, reference_factor)
    delta_g_ij = final_physical_metric - initial_physical_metric
    inv_initial_metric = invert_3x3_metric(initial_physical_metric)
    spatial_trace = trace_tensor(delta_g_ij, inv_initial_metric)
    return alpha_final, spatial_trace, spatial_trace / 6.0


def extract_lapse_potential(alpha_final):
    """Weak-field lapse potential Phi ~= 1/2 (alpha^2 - 1)."""
    return 0.5 * (alpha_final**2 - 1.0)


def shift_potential_zero_mode(phi_field):
    """Remove the volume-average zero mode from a periodic potential."""
    offset = jnp.mean(phi_field)
    return phi_field - offset, offset


def solve_poisson_periodic_reference(rho_field, dx):
    """Solve the periodic Poisson reference equation with zero-mean source."""
    source = 4.0 * jnp.pi * (rho_field - jnp.mean(rho_field))
    return invert_periodic_laplacian_zero_mean(source, dx)


def newtonian_potential_uniform_sphere(r, mass_geom, radius):
    """Analytic isolated Newtonian potential for a uniform sphere."""
    inside = -mass_geom / (2.0 * radius) * (3.0 - r**2 / radius**2)
    outside = -mass_geom / r
    return jnp.where(r <= radius, inside, outside)


def plot_diagnostics(result, radius, mass_geom, n_bins, output_prefix):
    """Build radial diagnostics and save the comparison plots."""
    alpha_final, spatial_trace, phi_metric_raw = extract_metric_potential(result["vars"])
    phi_lapse_raw = extract_lapse_potential(alpha_final)
    phi_metric_periodic, metric_zero_mode = shift_potential_zero_mode(phi_metric_raw)
    phi_lapse_periodic, lapse_zero_mode = shift_potential_zero_mode(phi_lapse_raw)
    phi_poisson_periodic = solve_poisson_periodic_reference(result["rho_field"], result["dx"])
    phi_poisson_periodic, poisson_zero_mode = shift_potential_zero_mode(phi_poisson_periodic)

    r_centers, phi_metric_periodic_radial, bin_idx = radial_bin_average(
        phi_metric_periodic,
        result["r3d"],
        n_bins=n_bins,
    )
    _, phi_lapse_periodic_radial, _ = radial_bin_average(phi_lapse_periodic, result["r3d"], n_bins=n_bins)
    _, phi_metric_raw_radial, _ = radial_bin_average(phi_metric_raw, result["r3d"], n_bins=n_bins)
    _, phi_poisson_periodic_radial, _ = radial_bin_average(
        phi_poisson_periodic,
        result["r3d"],
        n_bins=n_bins,
    )

    r_ref = jnp.linspace(0.01, float(r_centers[-1]), 500)
    phi_ref_isolated = newtonian_potential_uniform_sphere(r_ref, mass_geom, radius)
    surface_idx = int(jnp.argmin(jnp.abs(r_centers - float(radius))))
    phi_surface = float(phi_poisson_periodic_radial[surface_idx])
    y_lo = 1.5 * phi_surface
    y_hi = -0.3 * phi_surface

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(r_centers, phi_poisson_periodic_radial, "k--", lw=2, label=r"Periodic Poisson $\Phi_{\mathrm{per}}$")
    axes[0].plot(r_centers, phi_lapse_periodic_radial, lw=1.2, label=r"BSSN lapse $\frac{1}{2}(\alpha^2 - 1) - \langle \Phi \rangle$")
    axes[0].plot(r_centers, phi_metric_periodic_radial, ":", lw=1.0, label=r"BSSN metric $\Phi_{\mathrm{metric}} - \langle \Phi \rangle$")
    axes[0].axvline(float(radius), ls=":", color="grey", label=f"R = {float(radius):.2e} m")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel(r"$\Phi$")
    axes[0].set_title("Periodic BSSN potentials vs periodic Poisson reference")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))

    axes[1].plot(r_centers, phi_poisson_periodic_radial, "k--", lw=2, label=r"Periodic Poisson $\Phi_{\mathrm{per}}$")
    axes[1].plot(r_centers, phi_lapse_periodic_radial, "o-", ms=4, lw=1.2, color="tab:blue", label="BSSN lapse shifted to zero mean")
    axes[1].plot(r_centers, phi_metric_periodic_radial, ":", ms=4, lw=1.0, color="tab:orange", label="BSSN metric shifted to zero mean")
    axes[1].axvline(float(radius), ls=":", color="grey")
    axes[1].set_ylim(y_lo, y_hi)
    axes[1].set_xlabel("r [m]")
    axes[1].set_ylabel(r"$\Phi$")
    axes[1].set_title(f"Periodic zoom near r = R (Phi surface ~= {phi_surface:.2e})")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_potentials.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(r_centers, phi_poisson_periodic_radial, "k--", lw=2, label=r"Periodic Poisson $\Phi_{\mathrm{per}}$")
    ax2.plot(r_centers, phi_lapse_periodic_radial, lw=1.2, label=r"Zero-mean BSSN lapse $\Phi$")
    ax2.plot(r_centers, phi_metric_raw_radial, ":", lw=1.0, label=r"Raw BSSN metric $\Phi_{\mathrm{metric}}$")
    ax2.plot(r_centers, phi_metric_periodic_radial, "o-", ms=4, lw=1.2, label=r"Zero-mean BSSN metric $\Phi_{\mathrm{metric}}$")
    ax2.plot(r_ref, phi_ref_isolated, color="tab:green", lw=1.0, alpha=0.8, label=r"Isolated Newtonian $\Phi_{\mathrm{iso}}$")
    ax2.axvline(float(radius), ls=":", color="grey", label=f"R = {float(radius):.2e} m")
    ax2.set_xlabel("r [m]")
    ax2.set_ylabel(r"$\Phi$")
    ax2.set_title("Periodic zero-mode diagnostic")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_zero_mode.png", dpi=300)

    alpha_radial = jnp.array(
        [
            float(jnp.asarray(alpha_final).ravel()[bin_idx == i].mean()) if jnp.any(bin_idx == i) else jnp.nan
            for i in range(n_bins)
        ]
    )
    spatial_trace_radial = jnp.array(
        [
            float(spatial_trace.ravel()[bin_idx == i].mean()) if jnp.any(bin_idx == i) else jnp.nan
            for i in range(n_bins)
        ]
    )

    return {
        "surface_idx": surface_idx,
        "phi_poisson_periodic_radial": phi_poisson_periodic_radial,
        "bssn_dev": (alpha_radial**2 - 1.0) / 2.0,
        "spatial_trace_radial": spatial_trace_radial,
        "lapse_zero_mode": lapse_zero_mode,
        "metric_zero_mode": metric_zero_mode,
        "poisson_zero_mode": poisson_zero_mode,
        "rms_lapse": jnp.sqrt(jnp.mean((phi_lapse_periodic - phi_poisson_periodic) ** 2)),
        "rms_metric": jnp.sqrt(jnp.mean((phi_metric_periodic - phi_poisson_periodic) ** 2)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Periodic static planetary mass demo")
    parser.add_argument("--radius", type=float, default=6.3817e6, help="Sphere radius in meters")
    parser.add_argument("--rho-si", type=float, default=5513.0, help="Mass density in kg/m^3")
    parser.add_argument("--domain-factor", type=float, default=10.0, help="Domain width in units of the radius")
    parser.add_argument("--nx", type=int, default=172, help="Grid points per dimension")
    parser.add_argument("--dt-factor", type=float, default=0.4, help="Time step as a fraction of dx")
    parser.add_argument("--steps", type=int, default=2000, help="Number of RK4 steps")
    parser.add_argument("--radial-bins", type=int, default=60, help="Number of radial bins for plots")
    parser.add_argument("--output-prefix", default="periodic_planetary_mass", help="Output filename prefix")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    return parser.parse_args()


def main():
    setup_jax_config(enable_x64=True, verbose=True)
    args = parse_args()

    g_si = 6.67430e-11
    c_si = 2.99792458e8
    g_over_c2_m = g_si / (c_si**2)

    radius = args.radius
    rho_si = args.rho_si
    rho_geom = g_over_c2_m * rho_si
    mass_si = (4.0 / 3.0) * jnp.pi * radius**3 * rho_si
    mass_geom = g_over_c2_m * mass_si
    domain_size = args.domain_factor * radius

    print("Running periodic static planetary mass demo")
    print(f"rho_SI                     = {rho_si:.4f} kg/m^3")
    print(f"rho_geom                   = {rho_geom:.4e} 1/m^2")
    print(f"sphere mass                = {float(mass_si):.4e} kg")
    print(f"domain width               = {domain_size:.4e} m")
    print(f"dx                         = {domain_size / args.nx:.4f} m")
    print(f"dt                         = {args.dt_factor * domain_size / args.nx:.4e} m")

    coords = -domain_size / 2.0 + (jnp.arange(args.nx) + 0.5) * (domain_size / args.nx)
    rho_field = make_rho_field(coords, radius, rho_geom)
    psi_init, k_init = solve_periodic_hamiltonian_constraint(rho_field, domain_size / args.nx)
    print(f"periodic CMC initial K     = {float(k_init):.4e} m^-1")
    print(f"max |psi - 1|              = {float(jnp.max(jnp.abs(psi_init - 1.0))):.4e}")

    result = run_static_mass_3d(
        radius=radius,
        rho_geom=rho_geom,
        nx=args.nx,
        domain_size=domain_size,
        dt_factor=args.dt_factor,
        steps=args.steps,
        show_progress=not args.no_progress,
    )
    print(f"evolution finished at t    = {float(result['final_time']):.4f} m")

    diagnostics = plot_diagnostics(
        result,
        radius=radius,
        mass_geom=mass_geom,
        n_bins=args.radial_bins,
        output_prefix=args.output_prefix,
    )

    surface_idx = diagnostics["surface_idx"]
    print("\n--- Periodic diagnostic ---")
    print(f"Periodic Poisson Phi(R)    = {float(diagnostics['phi_poisson_periodic_radial'][surface_idx]):.4e}")
    print(f"BSSN 1/2(alpha^2 - 1)      = {float(diagnostics['bssn_dev'][surface_idx]):.4e}")
    print(f"BSSN 1/6 tr(delta g)       = {float(diagnostics['spatial_trace_radial'][surface_idx] / 6.0):.4e}")
    print(f"lapse zero mode removed    = {float(diagnostics['lapse_zero_mode']):.4e}")
    print(f"metric zero mode removed   = {float(diagnostics['metric_zero_mode']):.4e}")
    print(f"poisson zero mode removed  = {float(diagnostics['poisson_zero_mode']):.4e}")
    print(f"RMS(lapse - Poisson)       = {float(diagnostics['rms_lapse']):.4e}")
    print(f"RMS(metric - Poisson)      = {float(diagnostics['rms_metric']):.4e}")
    print(f"saved                      = {args.output_prefix}_potentials.png")
    print(f"saved                      = {args.output_prefix}_zero_mode.png")


if __name__ == "__main__":
    main()