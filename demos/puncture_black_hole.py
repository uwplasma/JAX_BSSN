import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.errors import compute_all_constraints, compute_constraint_norms
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.initialization import create_coordinate_arrays, get_initial_data


def puncture_coordinates(ni, nj, nk, dx, center_between_points=True):
    """Build the coordinate arrays used by the puncture initialization."""
    X, Y, Z = create_coordinate_arrays(ni, nj, nk, dx)

    if center_between_points:
        if ni % 2 == 1:
            X = X + 0.5 * dx
        if nj % 2 == 1:
            Y = Y + 0.5 * dx
        if nk % 2 == 1:
            Z = Z + 0.5 * dx

    return X, Y, Z


def radial_bin_average(field, radius, n_bins):
    """Average a 3D scalar field over spherical shells."""
    r_flat = radius.ravel()
    f_flat = field.ravel()
    r_max = float(jnp.max(r_flat))
    edges = jnp.linspace(0.0, r_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = jnp.clip(jnp.digitize(r_flat, edges) - 1, 0, n_bins - 1)

    values = jnp.array(
        [
            float(jnp.mean(f_flat[bin_idx == i])) if jnp.any(bin_idx == i) else jnp.nan
            for i in range(n_bins)
        ]
    )
    return centers, values


def shift_magnitude(shift):
    """Return |beta| on the grid."""
    return jnp.sqrt(jnp.sum(shift**2, axis=0))


def parse_args():
    parser = argparse.ArgumentParser(description="Evolve a single isotropic puncture black hole with 1+log slicing and Gamma-driver shift")
    parser.add_argument("--nx", type=int, default=64, help="Grid points per dimension")
    parser.add_argument("--domain-radius", type=float, default=8.0, help="Half-width L of the cubic domain [-L, L)")
    parser.add_argument("--mass", type=float, default=1.0, help="Puncture mass M")
    parser.add_argument("--final-time", type=float, default=2.0, help="Final evolution time")
    parser.add_argument("--dt-factor", type=float, default=0.1, help="Time step factor relative to dx")
    parser.add_argument("--eta", type=float, default=2.0, help="Gamma-driver damping parameter")
    parser.add_argument("--gamma-driver", type=float, default=0.75, help="Gamma-driver source coefficient g")
    parser.add_argument("--ko", type=float, default=0.25, help="Kreiss-Oliger dissipation coefficient")
    parser.add_argument("--bc-width", type=float, default=8.0, help="Super-Gaussian boundary width in grid cells")
    parser.add_argument("--bc-order", type=float, default=4.0, help="Super-Gaussian exponent")
    parser.add_argument("--bc-strength", type=float, default=1.0, help="Super-Gaussian damping strength")
    parser.add_argument("--snapshot-every", type=int, default=10, help="Snapshot cadence in time steps")
    parser.add_argument("--radial-bins", type=int, default=80, help="Number of spherical radial bins")
    parser.add_argument("--output-prefix", default="puncture_black_hole", help="Output filename prefix")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    return parser.parse_args()


def main():
    setup_jax_config(enable_x64=True, verbose=True)
    args = parse_args()

    nx = args.nx
    dx = 2.0 * args.domain_radius / nx
    dt = args.dt_factor * dx
    num_steps = int(round(args.final_time / dt))

    X, Y, Z = puncture_coordinates(nx, nx, nx, dx, center_between_points=True)
    radius = jnp.sqrt(X**2 + Y**2 + Z**2)
    min_radius = float(jnp.min(radius))

    params = BSSNParameters(
        eta=args.eta,
        kappa=0.0,
        nu=args.ko,
        g=args.gamma_driver,
        dx=dx,
        dt=dt,
        zero_shift=0,
        gauge=1,
        xl_bc=1,
        xr_bc=1,
        yl_bc=1,
        yr_bc=1,
        zl_bc=1,
        zr_bc=1,
        bc_width=args.bc_width,
        bc_order=args.bc_order,
        bc_strength=args.bc_strength,
    )
    vars = get_initial_data(
        "puncture_black_hole",
        nx,
        nx,
        nx,
        dx,
        mass=args.mass,
        lapse_puncture=True,
        center_between_points=True,
    )

    print(f"nx = {nx}, dx = {dx:.6e}, dt = {dt:.6e}, Nt = {num_steps}")
    print(f"mass = {args.mass:.6e}, domain radius = {args.domain_radius:.6e}")
    print(f"min sampled radius = {min_radius:.6e}")

    initial_W = vars.conformal_factor
    initial_alpha = vars.lapse

    times = [0.0]
    lapse_mins = [float(jnp.min(vars.lapse))]
    shift_max = [float(jnp.max(shift_magnitude(vars.shift)))]
    ham_l2 = []

    initial_constraints = compute_all_constraints(vars, params)
    initial_norms = compute_constraint_norms(initial_constraints)
    ham_l2.append(float(initial_norms["hamiltonian_l2"]))

    iterator = range(1, num_steps + 1)
    if not args.no_progress:
        iterator = tqdm(iterator)

    for step in iterator:
        vars = rk4_step(vars, params)
        if step % args.snapshot_every == 0 or step == num_steps:
            norms = compute_constraint_norms(compute_all_constraints(vars, params))
            times.append(step * dt)
            lapse_mins.append(float(jnp.min(vars.lapse)))
            shift_max.append(float(jnp.max(shift_magnitude(vars.shift))))
            ham_l2.append(float(norms["hamiltonian_l2"]))

    final_W = vars.conformal_factor
    final_alpha = vars.lapse
    final_shift = shift_magnitude(vars.shift)

    r_centers, W_initial_radial = radial_bin_average(initial_W, radius, args.radial_bins)
    _, W_final_radial = radial_bin_average(final_W, radius, args.radial_bins)
    _, alpha_initial_radial = radial_bin_average(initial_alpha, radius, args.radial_bins)
    _, alpha_final_radial = radial_bin_average(final_alpha, radius, args.radial_bins)
    _, shift_final_radial = radial_bin_average(final_shift, radius, args.radial_bins)

    psi_exact = 1.0 + args.mass / (2.0 * jnp.maximum(r_centers, 0.5 * dx))
    W_exact = psi_exact ** (-2.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(r_centers, W_exact, "k--", lw=1.5, label="Initial exact W")
    axes[0, 0].plot(r_centers, W_initial_radial, lw=1.2, label="Initial W")
    axes[0, 0].plot(r_centers, W_final_radial, lw=1.2, label="Final W")
    axes[0, 0].set_xlabel("r")
    axes[0, 0].set_ylabel("W")
    axes[0, 0].set_title("Conformal factor radial profile")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(r_centers, alpha_initial_radial, lw=1.2, label="Initial alpha")
    axes[0, 1].plot(r_centers, alpha_final_radial, lw=1.2, label="Final alpha")
    axes[0, 1].set_xlabel("r")
    axes[0, 1].set_ylabel("alpha")
    axes[0, 1].set_title("Lapse radial profile")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(r_centers, shift_final_radial, lw=1.2)
    axes[1, 0].set_xlabel("r")
    axes[1, 0].set_ylabel("|beta|")
    axes[1, 0].set_title("Final shift magnitude")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(times, ham_l2, lw=1.2)
    axes[1, 1].set_xlabel("t")
    axes[1, 1].set_ylabel("Hamiltonian L2")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Constraint history")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{args.output_prefix}_diagnostics.png", dpi=300)

    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 4))
    axes2[0].plot(times, lapse_mins, lw=1.2)
    axes2[0].set_xlabel("t")
    axes2[0].set_ylabel("min(alpha)")
    axes2[0].set_title("Minimum lapse")
    axes2[0].grid(True, alpha=0.3)

    axes2[1].plot(times, shift_max, lw=1.2)
    axes2[1].set_xlabel("t")
    axes2[1].set_ylabel("max(|beta|)")
    axes2[1].set_title("Maximum shift magnitude")
    axes2[1].grid(True, alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(f"{args.output_prefix}_timeseries.png", dpi=300)

    print(f"Final min(alpha) = {lapse_mins[-1]:.6e}")
    print(f"Final max(|beta|) = {shift_max[-1]:.6e}")
    print(f"Final Hamiltonian L2 = {ham_l2[-1]:.6e}")
    print(f"saved = {args.output_prefix}_diagnostics.png")
    print(f"saved = {args.output_prefix}_timeseries.png")


if __name__ == "__main__":
    main()