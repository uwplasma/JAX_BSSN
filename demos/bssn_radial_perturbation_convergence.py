import argparse
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters, BSSNVariables
from JAX_BSSN.evolve import rk4_step


def build_radially_perturbed_initial_data(coords, amplitude=1e-3, sigma=1.0):
    """Construct BSSN initial data with a smooth Gaussian radial conformal perturbation."""
    x3d, y3d, z3d = jnp.meshgrid(coords, coords, coords, indexing="ij")
    r3d = jnp.sqrt(x3d**2 + y3d**2 + z3d**2)

    pulse = amplitude * jnp.exp(-(r3d**2) / (sigma**2))
    conformal_factor = 1.0 / (1.0 + pulse)

    conformal_metric = jnp.zeros((3, 3) + r3d.shape, dtype=jnp.float64)
    conformal_metric = conformal_metric.at[0, 0].set(1.0)
    conformal_metric = conformal_metric.at[1, 1].set(1.0)
    conformal_metric = conformal_metric.at[2, 2].set(1.0)

    traceless_k = jnp.zeros((3, 3) + r3d.shape, dtype=jnp.float64)
    trace_k = jnp.zeros(r3d.shape, dtype=jnp.float64)
    conformal_connection = jnp.zeros((3,) + r3d.shape, dtype=jnp.float64)
    lapse = jnp.ones(r3d.shape, dtype=jnp.float64)
    shift = jnp.zeros((3,) + r3d.shape, dtype=jnp.float64)
    rho_field = jnp.zeros(r3d.shape, dtype=jnp.float64)
    stress_tensor = jnp.zeros((3, 3) + r3d.shape, dtype=jnp.float64)
    momentum_density = jnp.zeros((3,) + r3d.shape, dtype=jnp.float64)

    return (
        lapse,
        shift,
        conformal_factor,
        conformal_metric,
        traceless_k,
        trace_k,
        conformal_connection,
        rho_field,
        stress_tensor,
        momentum_density,
    )


def run_perturbed_bssn_simulation(
    nx,
    domain_size,
    dt_factor,
    target_time,
    amplitude,
    sigma,
    eta,
    gamma_driver,
    ko_dissipation,
    gauge,
    use_supergaussian_boundaries,
    bc_width,
    bc_order,
    bc_strength,
):
    """Evolve a radially perturbed metric forward to target_time using BSSN RK4."""
    dx = domain_size / nx
    dt = dt_factor * dx
    steps = int(np.round(target_time / dt))

    coords = -domain_size / 2.0 + (jnp.arange(nx) + 0.5) * dx

    (
        lapse,
        shift,
        conformal_factor,
        conformal_metric,
        traceless_k,
        trace_k,
        conformal_connection,
        rho_field,
        stress_tensor,
        momentum_density,
    ) = build_radially_perturbed_initial_data(coords, amplitude=amplitude, sigma=sigma)

    vars_bssn = BSSNVariables(
        conformal_metric=conformal_metric,
        conformal_factor=conformal_factor,
        traceless_K=traceless_k,
        trace_K=trace_k,
        conformal_connection=conformal_connection,
        lapse=lapse,
        shift=shift,
        rho=rho_field,
        S_ij=stress_tensor,
        momentum_density=momentum_density,
    )

    boundary_code = 1 if use_supergaussian_boundaries else 0

    params = BSSNParameters(
        eta=eta,
        kappa=1.0,
        nu=ko_dissipation,
        g=gamma_driver,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=gauge,
        xl_bc=boundary_code,
        xr_bc=boundary_code,
        yl_bc=boundary_code,
        yr_bc=boundary_code,
        zl_bc=boundary_code,
        zr_bc=boundary_code,
        bc_width=bc_width,
        bc_order=bc_order,
        bc_strength=bc_strength,
    )

    print(f"Evolving Nx = {nx:3d} | dx = {dx:.4e} | steps = {steps}")
    for _ in range(steps):
        vars_bssn = rk4_step(vars_bssn, params)

    return dx, vars_bssn


def downsample_3d(field_fine, ratio):
    """Downsample fine grid slice data by integer factor."""
    return field_fine[::ratio, ::ratio, ::ratio]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convergence test for a radially perturbed BSSN metric with shift evolution"
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[32, 64, 128],
        help="Grid sizes to run, with the largest used as the reference solution",
    )
    parser.add_argument("--domain-size", type=float, default=10.0, help="Domain length")
    parser.add_argument("--dt-factor", type=float, default=0.25, help="Time step factor relative to dx")
    parser.add_argument("--target-time", type=float, default=0.5, help="Target evolution time")
    parser.add_argument("--amplitude", type=float, default=1e-3, help="Radial perturbation amplitude")
    parser.add_argument("--sigma", type=float, default=1.0, help="Radial perturbation width")
    parser.add_argument("--eta", type=float, default=1.0, help="Gamma-driver damping parameter")
    parser.add_argument("--gamma-driver", type=float, default=0.75, help="Shift-driver coupling g")
    parser.add_argument("--ko", type=float, default=0.25, help="Kreiss-Oliger dissipation coefficient")
    parser.add_argument(
        "--gauge",
        type=int,
        default=1,
        choices=[0, 1],
        help="0 = harmonic slicing, 1 = 1+log slicing",
    )
    parser.add_argument(
        "--periodic-boundaries",
        action="store_true",
        help="Disable super-Gaussian boundary damping and use periodic boundaries",
    )
    parser.add_argument("--bc-width", type=float, default=8.0, help="Super-Gaussian boundary width in grid cells")
    parser.add_argument("--bc-order", type=float, default=4.0, help="Super-Gaussian boundary exponent")
    parser.add_argument("--bc-strength", type=float, default=1.0, help="Super-Gaussian boundary strength")
    parser.add_argument("--output", default="bssn_shift_convergence.png", help="Output plot filename")
    return parser.parse_args()


def main():
    setup_jax_config(enable_x64=True, verbose=False)
    args = parse_args()

    resolutions = sorted(args.resolutions)
    domain_size = args.domain_size
    dt_factor = args.dt_factor
    target_time = args.target_time
    amplitude = args.amplitude
    sigma = args.sigma
    use_supergaussian_boundaries = not args.periodic_boundaries

    print("--- BSSN Radially Perturbed Metric Convergence Test (Shift Vector) ---")
    print(
        f"Gauge = {'1+log' if args.gauge == 1 else 'harmonic'} | "
        f"shift driver g = {args.gamma_driver:.3f} | eta = {args.eta:.3f} | "
        f"boundaries = {'super-Gaussian' if use_supergaussian_boundaries else 'periodic'}"
    )

    nx_fine = max(resolutions)
    coarse_resolutions = [nx for nx in resolutions if nx < nx_fine]

    print(f"\n[1/2] Computing finest reference solution on Nx = {nx_fine}...")
    _, fine_vars = run_perturbed_bssn_simulation(
        nx_fine,
        domain_size,
        dt_factor,
        target_time,
        amplitude,
        sigma,
        args.eta,
        args.gamma_driver,
        args.ko,
        args.gauge,
        use_supergaussian_boundaries,
        args.bc_width,
        args.bc_order,
        args.bc_strength,
    )

    fine_shift_x = fine_vars.shift[0]

    dx_list = []
    l2_errors = []

    print(f"\n[2/2] Running convergence evaluations across coarse grids...")
    for nx in coarse_resolutions:
        dx, coarse_vars = run_perturbed_bssn_simulation(
            nx,
            domain_size,
            dt_factor,
            target_time,
            amplitude,
            sigma,
            args.eta,
            args.gamma_driver,
            args.ko,
            args.gauge,
            use_supergaussian_boundaries,
            args.bc_width,
            args.bc_order,
            args.bc_strength,
        )

        coarse_shift_x = coarse_vars.shift[0]
        ratio = nx_fine // nx
        fine_shift_x_downsampled = downsample_3d(fine_shift_x, ratio)

        error_field = coarse_shift_x - fine_shift_x_downsampled
        l2_err = float(jnp.sqrt(jnp.mean(error_field**2)))
        l2_err = max(l2_err, 1e-16)

        dx_list.append(dx)
        l2_errors.append(l2_err)

        print(f"-> Nx = {nx:3d} | dx = {dx:.4e} | L2 Error (Shift_x) = {l2_err:.6e}")

    dx_arr = np.array(dx_list)
    err_arr = np.array(l2_errors)

    plt.figure(figsize=(7, 5))
    plt.loglog(dx_arr, err_arr, "o-", label=r"Measured Shift Error $||\beta^x - \beta^x_{\mathrm{fine}}||_2$")

    ref_2nd = err_arr[-1] * (dx_arr / dx_arr[-1]) ** 2
    ref_4th = err_arr[-1] * (dx_arr / dx_arr[-1]) ** 4

    plt.loglog(dx_arr, ref_2nd, "k--", alpha=0.6, label=r"2nd Order ($\mathcal{O}(\Delta x^2)$)")
    plt.loglog(dx_arr, ref_4th, "r:", alpha=0.8, label=r"4th Order ($\mathcal{O}(\Delta x^4)$)")

    plt.xlabel(r"Grid Spacing $\Delta x$")
    plt.ylabel(r"$L_2$ Error Norm ($\beta^x$)")
    plt.title("BSSN Convergence Test (Shift Vector $\\beta^x$)")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    out_file = args.output
    plt.savefig(out_file, dpi=300)
    print(f"\nSuccess! Convergence plot saved as '{out_file}'.")


if __name__ == "__main__":
    main()