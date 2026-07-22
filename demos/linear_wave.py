import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters, compute_momentum_constraint, compute_physical_metric
from JAX_BSSN.errors import compute_hamiltonian_constraint
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.initialization import create_coordinate_arrays, get_initial_data


def linear_wave_b(x, t, amplitude, wavelength):
    """Analytic linear-wave profile b(x, t)."""
    return amplitude * jnp.sin((2.0 * jnp.pi * (x - t)) / wavelength)


def l2_norm_over_x(history, dx):
    """Compute the 1D L2 norm over x for a history of profiles."""
    return jnp.sqrt(dx * jnp.sum(history**2, axis=1))


def linf_norm_over_x(history):
    """Compute the 1D Linf norm over x for a history of profiles."""
    return jnp.max(jnp.abs(history), axis=1)


def restrict_history_cell_centered(history, refinement_ratio):
    """Restrict cell-centered 1D data by repeated pair averaging."""
    out = history
    ratio = int(refinement_ratio)
    while ratio > 1:
        out = 0.5 * (out[:, 0::2] + out[:, 1::2])
        ratio //= 2
    return out


def sample_linear_wave_state(vars):
    """Extract the physical fields tracked by the demo from the BSSN state."""
    physical_metric = compute_physical_metric(vars.conformal_metric, vars.conformal_factor)
    return {
        "gyy": physical_metric[1, 1, :, 0, 0],
        "gzz": physical_metric[2, 2, :, 0, 0],
        "conformal_factor": vars.conformal_factor[:, 0, 0],
        "lapse": vars.lapse[:, 0, 0],
    }


def analytic_histories(x, times, amplitude, wavelength):
    """Build the analytic linear-wave histories on the stored sample times."""
    phase = (2.0 * jnp.pi / wavelength) * (x[jnp.newaxis, :] - times[:, jnp.newaxis])
    b_exact = amplitude * jnp.sin(phase)
    return {
        "gyy": 1.0 + b_exact,
        "gzz": 1.0 - b_exact,
        "conformal_factor": jnp.power(1.0 - b_exact**2, -1.0 / 6.0),
        "lapse": jnp.ones_like(b_exact),
    }


def run_linear_wave(nx, amplitude, wavelength, final_crossings, snapshot_every_crossings, show_progress=True):
    """Run the 1D linear-wave demo on an x line embedded in the 3D code."""
    dx = wavelength / nx
    dt = dx / 4.0
    x3d, _, _ = create_coordinate_arrays(nx, 1, 1, dx)
    x = x3d[:, 0, 0]

    params = BSSNParameters(
        eta=0.0,
        kappa=0.0,
        nu=0.0,
        f=0.0,
        g=0.0,
        dx=dx,
        dt=dt,
    )
    vars = get_initial_data(
        "linear_wave",
        nx,
        1,
        1,
        dx,
        amplitude=amplitude,
        wavelength=wavelength,
    )

    steps_per_crossing = int(round(wavelength / dt))
    total_steps = int(round(final_crossings * steps_per_crossing))
    snapshot_stride = max(1, int(round(snapshot_every_crossings * steps_per_crossing)))

    sample = sample_linear_wave_state(vars)
    times = [0.0]
    gyy_hist = [sample["gyy"]]
    gzz_hist = [sample["gzz"]]
    cf_hist = [sample["conformal_factor"]]
    lapse_hist = [sample["lapse"]]
    ham_l2_hist = [jnp.sqrt(jnp.mean(compute_hamiltonian_constraint(vars, params) ** 2))]
    mom_l2_hist = [jnp.sqrt(jnp.mean(compute_momentum_constraint(vars, params) ** 2))]

    iterator = range(1, total_steps + 1)
    if show_progress:
        iterator = tqdm(iterator, leave=False)

    for step in iterator:
        vars = rk4_step(vars, params)
        if step % snapshot_stride == 0 or step == total_steps:
            sample = sample_linear_wave_state(vars)
            times.append(step * dt)
            gyy_hist.append(sample["gyy"])
            gzz_hist.append(sample["gzz"])
            cf_hist.append(sample["conformal_factor"])
            lapse_hist.append(sample["lapse"])
            ham_l2_hist.append(jnp.sqrt(jnp.mean(compute_hamiltonian_constraint(vars, params) ** 2)))
            mom_l2_hist.append(jnp.sqrt(jnp.mean(compute_momentum_constraint(vars, params) ** 2)))

    times = jnp.asarray(times)
    exact = analytic_histories(x, times, amplitude, wavelength)
    return {
        "nx": nx,
        "dx": dx,
        "dt": dt,
        "Nt": total_steps,
        "snapshot_stride": snapshot_stride,
        "x": x,
        "times": times,
        "gyy": jnp.asarray(gyy_hist),
        "gzz": jnp.asarray(gzz_hist),
        "conformal_factor": jnp.asarray(cf_hist),
        "lapse": jnp.asarray(lapse_hist),
        "ham_l2": jnp.asarray(ham_l2_hist),
        "mom_l2": jnp.asarray(mom_l2_hist),
        "gyy_exact": exact["gyy"],
        "gzz_exact": exact["gzz"],
        "conformal_factor_exact": exact["conformal_factor"],
        "lapse_exact": exact["lapse"],
    }


def self_convergence_factor(coarse_hist, mid_hist, fine_hist, dx_coarse, ratio_mid, ratio_fine, den_floor=1e-30):
    """Compute the self-convergence factor and observed order."""
    mid_on_coarse = restrict_history_cell_centered(mid_hist, ratio_mid)
    fine_on_coarse = restrict_history_cell_centered(fine_hist, ratio_fine)

    diff_c_m = l2_norm_over_x(coarse_hist - mid_on_coarse, dx_coarse)
    diff_m_f = l2_norm_over_x(mid_on_coarse - fine_on_coarse, dx_coarse)
    valid = diff_m_f > den_floor
    cf = jnp.where(valid, diff_c_m / diff_m_f, jnp.nan)
    p = jnp.where(valid, jnp.log2(cf), jnp.nan)
    return cf, p


def plot_linear_wave_diagnostics(results, amplitude, wavelength, output_prefix):
    """Save profile, error, and convergence plots for the demo."""
    grid_sizes = sorted(results)
    profile_time = float(min(0.5, results[grid_sizes[0]]["times"][-1]))
    profile_idx = int(jnp.argmin(jnp.abs(results[grid_sizes[0]]["times"] - profile_time)))
    log_floor = jnp.finfo(jnp.float64).tiny

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    final_linf_errors = []

    for nx in grid_sizes:
        run = results[nx]
        gzz_error = run["gzz"] - run["gzz_exact"]
        cf_error = run["conformal_factor"] - run["conformal_factor_exact"]

        gzz_linf = linf_norm_over_x(gzz_error)
        cf_linf = jnp.maximum(linf_norm_over_x(cf_error), log_floor)
        final_linf_errors.append(float(gzz_linf[-1]))

        axes[0, 0].plot(run["times"], gzz_linf, marker=".", linewidth=1, label=f"nx={nx}")
        axes[0, 1].plot(run["times"], cf_linf, marker=".", linewidth=1, label=f"nx={nx}")
        axes[1, 0].plot(run["times"], run["ham_l2"], marker=".", linewidth=1, label=f"nx={nx}")
        axes[1, 1].plot(run["times"], run["mom_l2"], marker=".", linewidth=1, label=f"nx={nx}")

    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")

    axes[0, 0].set_ylabel(r"$||g_{zz}^{num} - g_{zz}^{exact}||_{\infty}$")
    axes[0, 0].set_title(r"$g_{zz}$ error")
    axes[0, 1].set_ylabel(r"$||W^{num} - W^{exact}||_{\infty}$")
    axes[0, 1].set_title("Conformal-factor error")
    axes[1, 0].set_ylabel(r"$||\mathcal{H}||_2$")
    axes[1, 0].set_title("Hamiltonian constraint")
    axes[1, 1].set_ylabel(r"$||\mathcal{M}||_2$")
    axes[1, 1].set_title("Momentum constraint")

    for ax in axes.ravel():
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_errors.png", dpi=300)

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    dense_x = jnp.linspace(float(results[grid_sizes[0]]["x"][0]), float(results[grid_sizes[0]]["x"][-1]), 2000)
    dense_gzz = 1.0 - linear_wave_b(dense_x, profile_time, amplitude, wavelength)
    for nx in grid_sizes:
        run = results[nx]
        ax2.plot(run["x"], run["gzz"][profile_idx] - 1.0, marker=".", linewidth=1, label=f"nx={nx}")
    ax2.plot(dense_x, dense_gzz - 1.0, "k--", linewidth=1.5, label="Exact")
    ax2.set_xlabel("x")
    ax2.set_ylabel(r"$g_{zz} - 1$")
    ax2.set_title(f"Linear-wave profile at t={profile_time:.3f}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig(f"{output_prefix}_profile.png", dpi=300)

    convergence = None
    if len(grid_sizes) >= 3:
        coarse, mid, fine = grid_sizes[:3]
        cf, order = self_convergence_factor(
            results[coarse]["gzz"],
            results[mid]["gzz"],
            results[fine]["gzz"],
            results[coarse]["dx"],
            mid // coarse,
            fine // coarse,
        )
        convergence = {
            "cf": cf,
            "order": order,
            "coarse": coarse,
            "mid": mid,
            "fine": fine,
        }

        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
        axes3[0].plot(results[coarse]["times"], cf, marker=".", linewidth=1)
        axes3[0].axhline(4.0, color="k", linestyle="--", linewidth=1, label="Exact 2nd order")
        axes3[0].set_xlabel("Time")
        axes3[0].set_ylabel("Convergence factor")
        axes3[0].set_title("Self-convergence factor")
        axes3[0].grid(True, alpha=0.3)
        axes3[0].legend()

        axes3[1].plot(results[coarse]["times"], order, marker=".", linewidth=1)
        axes3[1].axhline(2.0, color="k", linestyle="--", linewidth=1, label="Exact 2nd order")
        axes3[1].set_xlabel("Time")
        axes3[1].set_ylabel("Observed order")
        axes3[1].set_title(r"$p = \log_2(\mathrm{CF})$")
        axes3[1].grid(True, alpha=0.3)
        axes3[1].legend()
        fig3.tight_layout()
        fig3.savefig(f"{output_prefix}_convergence.png", dpi=300)

    return {
        "final_linf_errors": final_linf_errors,
        "convergence": convergence,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Linear-wave demo using the gauge-wave style evolution path")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[50, 100, 200], help="Grid sizes to evolve")
    parser.add_argument("--amplitude", type=float, default=1.0e-8, help="Wave amplitude")
    parser.add_argument("--wavelength", type=float, default=1.0, help="Wave wavelength")
    parser.add_argument("--final-crossings", type=float, default=1.0, help="Run time in crossing times")
    parser.add_argument("--snapshot-every", type=float, default=0.1, help="Snapshot cadence in crossing times")
    parser.add_argument("--output-prefix", default="linear_wave", help="Output filename prefix")
    parser.add_argument("--no-progress", action="store_true", help="Disable the progress bar")
    return parser.parse_args()


def main():
    setup_jax_config(enable_x64=True, verbose=True)
    args = parse_args()

    results = {}
    for nx in args.grid_sizes:
        print(f"Running linear wave with nx={nx}")
        results[nx] = run_linear_wave(
            nx=nx,
            amplitude=args.amplitude,
            wavelength=args.wavelength,
            final_crossings=args.final_crossings,
            snapshot_every_crossings=args.snapshot_every,
            show_progress=not args.no_progress,
        )

    plot_summary = plot_linear_wave_diagnostics(
        results,
        amplitude=args.amplitude,
        wavelength=args.wavelength,
        output_prefix=args.output_prefix,
    )

    print("\nFinal gzz Linf errors:")
    for nx, err in zip(sorted(results), plot_summary["final_linf_errors"]):
        run = results[nx]
        print(f"  nx={nx}: dt={float(run['dt']):.6e}, Nt={run['Nt']}, error={err:.6e}")

    convergence = plot_summary["convergence"]
    if convergence is not None:
        coarse = convergence["coarse"]
        mid = convergence["mid"]
        fine = convergence["fine"]
        valid = jnp.isfinite(convergence["order"])
        mean_order = float(jnp.mean(convergence["order"][valid])) if jnp.any(valid) else float("nan")
        print(f"Observed self-convergence order over nx={coarse},{mid},{fine}: {mean_order:.3f}")

    print(f"saved = {args.output_prefix}_errors.png")
    print(f"saved = {args.output_prefix}_profile.png")
    if convergence is not None:
        print(f"saved = {args.output_prefix}_convergence.png")


if __name__ == "__main__":
    main()