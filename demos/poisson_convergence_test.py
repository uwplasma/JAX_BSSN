import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def make_rho_field(coords, radius, rho_geom):
    """Create a uniform spherical density field centered at the origin."""
    x3d, y3d, z3d = jnp.meshgrid(coords, coords, coords, indexing="ij")
    r_squared = x3d**2 + y3d**2 + z3d**2
    return jnp.where(r_squared < radius**2, rho_geom, 0.0)


def invert_periodic_laplacian_zero_mean(source, dx):
    """Invert the periodic Laplacian on the zero-mean subspace via FFT."""
    source_np = np.asarray(source, dtype=np.float64)
    source_np = source_np - source_np.mean()

    n_x, n_y, n_z = source_np.shape
    kx = 2.0 * np.pi * np.fft.fftfreq(n_x, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n_y, d=dx)
    kz = 2.0 * np.pi * np.fft.fftfreq(n_z, d=dx)
    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k_squared = kx3d**2 + ky3d**2 + kz3d**2

    source_hat = np.fft.fftn(source_np)
    solution_hat = np.zeros_like(source_hat)
    nonzero_mask = k_squared > 0.0
    solution_hat[nonzero_mask] = -source_hat[nonzero_mask] / k_squared[nonzero_mask]

    return jnp.asarray(np.fft.ifftn(solution_hat).real)


def solve_poisson_periodic_reference(rho_field, dx):
    """Solve periodic Poisson equation ∇²Φ = 4π (ρ - <ρ>)."""
    source = 4.0 * jnp.pi * (rho_field - jnp.mean(rho_field))
    return invert_periodic_laplacian_zero_mean(source, dx)


def downsample_3d(field_fine, ratio):
    """Downsample fine grid data by integer factor via index slicing."""
    return field_fine[::ratio, ::ratio, ::ratio]


def run_convergence_test(resolutions, domain_size, radius, rho_geom):
    """Perform convergence testing against the highest resolution solution."""
    nx_fine = max(resolutions)
    dx_fine = domain_size / nx_fine
    coords_fine = -domain_size / 2.0 + (jnp.arange(nx_fine) + 0.5) * dx_fine
    rho_fine = make_rho_field(coords_fine, radius, rho_geom)

    print(f"Computing finest reference grid: Nx = {nx_fine}")
    phi_fine = solve_poisson_periodic_reference(rho_fine, dx_fine)

    errors_l2 = []
    dx_list = []
    coarse_resolutions = [nx for nx in resolutions if nx < nx_fine]

    for nx in coarse_resolutions:
        dx = domain_size / nx
        coords = -domain_size / 2.0 + (jnp.arange(nx) + 0.5) * dx
        rho = make_rho_field(coords, radius, rho_geom)

        phi_num = solve_poisson_periodic_reference(rho, dx)

        ratio = nx_fine // nx
        phi_ref_downsampled = downsample_3d(phi_fine, ratio)

        # RMS L2 norm error
        l2_err = float(jnp.sqrt(jnp.mean((phi_num - phi_ref_downsampled) ** 2)))
        errors_l2.append(l2_err)
        dx_list.append(dx)

        print(f"Nx = {nx:3d} | dx = {dx:.3e} | L2 Error = {l2_err:.5e}")

    return np.array(dx_list), np.array(errors_l2)


def plot_convergence(dx_list, errors_l2, output_file="poisson_convergence.png"):
    """Plot log-log error vs grid spacing to demonstrate convergence order."""
    plt.figure(figsize=(7, 5))
    plt.loglog(dx_list, errors_l2, "o-", label=r"Measured Error $|| \Phi - \Phi_{\mathrm{fine}} ||_2$")

    # Ideal 2nd order reference scaling line
    ref_2nd = errors_l2[-1] * (dx_list / dx_list[-1]) ** 2
    plt.loglog(dx_list, ref_2nd, "k--", label=r"2nd Order Scaling ($\mathcal{O}(\Delta x^2)$)")

    plt.xlabel(r"Grid Spacing $\Delta x$ [m]")
    plt.ylabel(r"$L_2$ Error Norm")
    plt.title("Poisson Solver Grid Convergence Test")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Convergence plot saved to {output_file}")


def main():
    resolutions = [32, 64, 128, 256]
    domain_size = 10.0 * 6.3817e6
    radius = 6.3817e6
    rho_geom = 4.09e-18

    dx_list, errors_l2 = run_convergence_test(resolutions, domain_size, radius, rho_geom)
    plot_convergence(dx_list, errors_l2)


if __name__ == "__main__":
    main()