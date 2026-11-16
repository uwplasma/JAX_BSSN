"""Recover the Newtonian potential of a static uniform-density planet with periodic BSSN."""

import math
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from JAX_BSSN.bssn.geometry import compute_physical_metric
from JAX_BSSN.bssn.tensor_algebra import invert_3x3_metric, trace_tensor
from JAX_BSSN.bssn.variables import BSSNParameters, BSSNVariables
from JAX_BSSN.evolution.boundaries import PERIODIC_BC
from JAX_BSSN.evolution.time_evolve import rk4_step


G_SI = 6.67430e-11
C_SI = 2.99792458e8

RADIUS = 6.3817e6           # sphere radius [m]
RHO_SI = 5513.0             # mass density [kg / m^3]
DOMAIN_FACTOR = 10.0        # periodic box width in units of RADIUS
NUM_POINTS = 256            # grid points per dimension
CFL = 0.4                   # dt / dx
FINAL_TIME = 2.97e8         # evolution time [m]
RADIAL_BINS = 60            # radial bins used for the plots
OUTPUT_DIR = Path("output")
OUTPUT_PREFIX = "periodic_planetary_mass"
SHOW_PROGRESS = True


def make_rho_field(coords, radius, rho_geom):
    """
    Create a uniform spherical density field centered at the origin.

    Args:
        coords: 1D cell-centered coordinates [m], shared by x, y and z
        radius: Sphere radius [m]
        rho_geom: Density inside the sphere in geometric units [1/m^2]

    Returns:
        Density field with shape (n, n, n), rho_geom inside the sphere and zero outside
    """
    x3d, y3d, z3d = jnp.meshgrid(coords, coords, coords, indexing="ij")
    r_squared = x3d**2 + y3d**2 + z3d**2
    return jnp.where(r_squared < radius**2, rho_geom, 0.0)


def invert_periodic_laplacian_zero_mean(source, dx):
    """
    Invert the periodic Laplacian on the zero-mean subspace.

    Args:
        source: 3D right-hand side; its mean is removed before inverting
        dx: Grid spacing [m]

    Returns:
        Zero-mean periodic solution u of laplacian(u) = source
    """
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
    """
    Solve the periodic conformally flat CMC Hamiltonian constraint.

    Args:
        rho_field: 3D density field in geometric units [1/m^2]
        dx: Grid spacing [m]
        tol: Stop once the max change in psi between iterations drops below this
        max_iter: Maximum number of fixed-point iterations
        relaxation: Under-relaxation weight for the psi update (0 to 1)
        k_sign: Sign of the constant trace K (-1 for an expanding slice)

    Returns:
        Tuple (psi, K) of the conformal factor field and the constant trace of K
    """
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
    """
    Construct periodic conformally flat CMC initial data.

    Args:
        rho_field: 3D density field in geometric units [1/m^2]
        dx: Grid spacing [m]

    Returns:
        Tuple (lapse, shift, W, conformal metric, A_ij, K, Gamma^i) of initial BSSN fields
    """
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


def run_static_mass_3d(radius, rho_geom, nx, domain_size, cfl, final_time, show_progress=True):
    """
    Evolve periodic CMC initial data for a uniform spherical density field.

    Args:
        radius: Sphere radius [m]
        rho_geom: Density inside the sphere in geometric units [1/m^2]
        nx: Grid points per dimension
        domain_size: Periodic box width [m]
        cfl: Time step as a fraction of dx
        final_time: Total evolution time [m]
        show_progress: If True, show a tqdm progress bar

    Returns:
        Dict with dx, dt, final_time, r3d, rho_field and the final BSSN variables
    """
    dx = domain_size / nx
    steps = math.ceil(final_time / (cfl * dx))
    dt = final_time / steps
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
    )
    params = BSSNParameters(
        eta=0.0,
        kappa=1.0,
        nu=0.25,
        g=0.0,
        dx=dx,
        dt=dt,
        zero_shift=1,
        gauge=1,
        xl_bc=PERIODIC_BC,
        xr_bc=PERIODIC_BC,
        yl_bc=PERIODIC_BC,
        yr_bc=PERIODIC_BC,
        zl_bc=PERIODIC_BC,
        zr_bc=PERIODIC_BC,
        x_min=float(coords[0]),
        y_min=float(coords[0]),
        z_min=float(coords[0]),
        rho=rho_field,
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
    """
    Average a 3D field over spherical radial bins.

    Args:
        field: 3D field to average
        r3d: 3D radius from the sphere center [m], same shape as field
        n_bins: Number of radial bins between 0 and max(r3d)

    Returns:
        Tuple (bin centers, binned average, flat bin index of every grid point)
    """
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
    """
    Compute the weak-field metric potential from the final physical metric.

    Args:
        final_vars: BSSN variables at the end of the evolution

    Returns:
        Tuple (lapse, trace of the physical metric perturbation, metric potential = trace / 6)
    """
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
    """
    Weak-field lapse potential Phi ~= 1/2 (alpha^2 - 1).

    Args:
        alpha_final: Lapse field at the end of the evolution

    Returns:
        Weak-field potential 1/2 (alpha^2 - 1)
    """
    return 0.5 * (alpha_final**2 - 1.0)


def shift_potential_zero_mode(phi_field):
    """
    Remove the volume-average zero mode from a periodic potential.

    Args:
        phi_field: 3D periodic potential

    Returns:
        Tuple (zero-mean potential, removed mean)
    """
    offset = jnp.mean(phi_field)
    return phi_field - offset, offset


def solve_poisson_periodic_reference(rho_field, dx):
    """
    Solve the periodic Poisson reference equation with zero-mean source.

    Args:
        rho_field: 3D density field in geometric units [1/m^2]
        dx: Grid spacing [m]

    Returns:
        Zero-mean periodic Newtonian potential
    """
    source = 4.0 * jnp.pi * (rho_field - jnp.mean(rho_field))
    return invert_periodic_laplacian_zero_mean(source, dx)


def newtonian_potential_uniform_sphere(r, mass_geom, radius):
    """
    Analytic isolated Newtonian potential for a uniform sphere.

    Args:
        r: Radius from the sphere center [m]
        mass_geom: Sphere mass in geometric units [m]
        radius: Sphere radius [m]

    Returns:
        Isolated (non-periodic) Newtonian potential at r
    """
    inside = -mass_geom / (2.0 * radius) * (3.0 - r**2 / radius**2)
    outside = -mass_geom / r
    return jnp.where(r <= radius, inside, outside)


def plot_diagnostics(result, radius, mass_geom, n_bins, output_prefix):
    """
    Build radial diagnostics and save the comparison plots.

    Args:
        result: Output dict from run_static_mass_3d
        radius: Sphere radius [m]
        mass_geom: Sphere mass in geometric units [m]
        n_bins: Number of radial bins for the plots
        output_prefix: Path prefix for the saved potentials and zero-mode PNGs

    Returns:
        Dict of surface values, removed zero modes and RMS differences from Poisson
    """
    output_prefix = str(output_prefix)
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


def main():
    """Evolve the static planet and compare BSSN potentials with Poisson."""

    g_over_c2_m = G_SI / (C_SI**2)
    radius = RADIUS
    rho_geom = g_over_c2_m * RHO_SI
    mass_si = (4.0 / 3.0) * jnp.pi * radius**3 * RHO_SI
    mass_geom = g_over_c2_m * mass_si
    domain_size = DOMAIN_FACTOR * radius
    dx = domain_size / NUM_POINTS
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = OUTPUT_DIR / OUTPUT_PREFIX

    print("Running periodic static planetary mass demo")
    print(f"rho_SI                     = {RHO_SI:.4f} kg/m^3")
    print(f"rho_geom                   = {rho_geom:.4e} 1/m^2")
    print(f"sphere mass                = {float(mass_si):.4e} kg")
    print(f"domain width               = {domain_size:.4e} m")
    print(f"grid                       = {NUM_POINTS}^3")
    print(f"dx                         = {dx:.4f} m")

    coords = -domain_size / 2.0 + (jnp.arange(NUM_POINTS) + 0.5) * dx
    rho_field = make_rho_field(coords, radius, rho_geom)
    psi_init, k_init = solve_periodic_hamiltonian_constraint(rho_field, dx)
    print(f"periodic CMC initial K     = {float(k_init):.4e} m^-1")
    print(f"max |psi - 1|              = {float(jnp.max(jnp.abs(psi_init - 1.0))):.4e}")

    result = run_static_mass_3d(
        radius=radius,
        rho_geom=rho_geom,
        nx=NUM_POINTS,
        domain_size=domain_size,
        cfl=CFL,
        final_time=FINAL_TIME,
        show_progress=SHOW_PROGRESS,
    )
    print(f"evolution finished at t    = {float(result['final_time']):.4e} m")

    diagnostics = plot_diagnostics(
        result,
        radius=radius,
        mass_geom=mass_geom,
        n_bins=RADIAL_BINS,
        output_prefix=output_prefix,
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
    print(f"saved                      = {output_prefix}_potentials.png")
    print(f"saved                      = {output_prefix}_zero_mode.png")


if __name__ == "__main__":
    main()
