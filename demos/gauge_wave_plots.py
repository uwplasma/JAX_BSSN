import argparse

import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm

from JAX_BSSN import setup_jax_config
from JAX_BSSN.bssn import BSSNParameters
from JAX_BSSN.errors import compute_hamiltonian_constraint
from JAX_BSSN.evolve import rk4_step
from JAX_BSSN.initialization import gauge_wave_data


def analytic_gauge_wave(x, t, wavelength, amplitude):
    """Return analytic conformal-metric components and conformal factor."""

    h = amplitude * jnp.sin((2.0 * jnp.pi * (x - t)) / wavelength)
    one_minus_h = 1.0 - h
    return {
        "g00": jnp.power(one_minus_h, 2.0 / 3.0),
        "g11": jnp.power(one_minus_h, -1.0 / 3.0),
        "g22": jnp.power(one_minus_h, -1.0 / 3.0),
        "W": jnp.power(one_minus_h, -1.0 / 6.0),
    }


def run_gauge_wave(nx, steps, wavelength, amplitude, show_progress=True):
    """Evolve the gauge-wave test problem and collect diagnostics."""

    x = jnp.linspace(-wavelength / 2.0, wavelength / 2.0, nx, endpoint=False)
    dx = float(x[1] - x[0])
    dt = dx
    vars = gauge_wave_data(nx, nx, nx, dx, amplitude=amplitude, wavelength=wavelength)
    params = BSSNParameters(eta=0.0, kappa=0.0, g=0.0, dx=dx, dt=dt)

    ham_history = [jnp.sqrt(jnp.mean(compute_hamiltonian_constraint(vars, params) ** 2))]
    iterator = range(steps)
    if show_progress:
        iterator = tqdm(iterator, leave=False)

    for _ in iterator:
        vars = rk4_step(vars, params)
        ham_history.append(jnp.sqrt(jnp.mean(compute_hamiltonian_constraint(vars, params) ** 2)))

    final_time = steps * dt
    exact = analytic_gauge_wave(x, final_time, wavelength, amplitude)
    return {
        "nx": nx,
        "x": x,
        "dx": dx,
        "dt": dt,
        "time": final_time,
        "hamiltonian_l2": jnp.asarray(ham_history),
        "g00": vars.conformal_metric[0, 0, :, 0, 0],
        "g11": vars.conformal_metric[1, 1, :, 0, 0],
        "g22": vars.conformal_metric[2, 2, :, 0, 0],
        "W": vars.conformal_factor[:, 0, 0],
        "g00_exact": exact["g00"],
        "g11_exact": exact["g11"],
        "g22_exact": exact["g22"],
        "W_exact": exact["W"],
    }


def relative_l2_error(numerical, exact):
    return jnp.sqrt(jnp.mean(((numerical - exact) / exact) ** 2))


def save_profile_plot(run, output_prefix):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    profiles = [
        ("g00", r"$\tilde{\gamma}_{xx}$"),
        ("g11", r"$\tilde{\gamma}_{yy}$"),
        ("g22", r"$\tilde{\gamma}_{zz}$"),
        ("W", r"$W$"),
    ]

    for ax, (key, label) in zip(axes.ravel(), profiles):
        ax.plot(run["x"], run[key], linewidth=1.5, label="Numerical")
        ax.plot(run["x"], run[f"{key}_exact"], "k--", linewidth=1.0, label="Analytic")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes[1]:
        ax.set_xlabel("x")

    fig.suptitle(f"Gauge wave profiles at t={run['time']:.3f}, nx={run['nx']}")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_profiles.png", dpi=300)


def save_diagnostics_plot(runs, output_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for run in runs:
        times = jnp.arange(run["hamiltonian_l2"].shape[0]) * run["dt"]
        axes[0].plot(times, run["hamiltonian_l2"], marker=".", linewidth=1, label=f"nx={run['nx']}")

        error = jnp.abs(run["W"] - run["W_exact"])
        axes[1].plot(run["x"], error, linewidth=1.2, label=f"nx={run['nx']}")

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(r"$||\mathcal{H}||_2$")
    axes[0].set_title("Hamiltonian constraint")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_yscale("log")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$|W^{num} - W^{exact}|$")
    axes[1].set_title("Final conformal-factor error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(f"{output_prefix}_diagnostics.png", dpi=300)


def save_convergence_plot(runs, output_prefix):
    dxs = jnp.asarray([run["dx"] for run in runs])
    g00_errors = jnp.asarray([relative_l2_error(run["g00"], run["g00_exact"]) for run in runs])
    g11_errors = jnp.asarray([relative_l2_error(run["g11"], run["g11_exact"]) for run in runs])
    g22_errors = jnp.asarray([relative_l2_error(run["g22"], run["g22_exact"]) for run in runs])
    w_errors = jnp.asarray([relative_l2_error(run["W"], run["W_exact"]) for run in runs])

    fig, ax = plt.subplots(figsize=(7, 5))
    for label, values in [
        (r"$\tilde{\gamma}_{xx}$", g00_errors),
        (r"$\tilde{\gamma}_{yy}$", g11_errors),
        (r"$\tilde{\gamma}_{zz}$", g22_errors),
        (r"$W$", w_errors),
    ]:
        ax.loglog(dxs, values, marker="o", linewidth=1.2, label=label)

    reference = w_errors[-1] * (dxs / dxs[-1]) ** 4
    ax.loglog(dxs, reference, "k--", linewidth=1.0, label="4th-order reference")
    ax.set_xlabel("dx")
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Gauge-wave convergence")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{output_prefix}_convergence.png", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate gauge-wave diagnostic plots.")
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=[30, 50, 70])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--wavelength", type=float, default=1.0)
    parser.add_argument("--amplitude", type=float, default=0.1)
    parser.add_argument("--output-prefix", default="gauge_wave")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    setup_jax_config(enable_x64=True, verbose=True)
    args = parse_args()

    runs = []
    for nx in args.grid_sizes:
        print(f"Running gauge wave with nx={nx}")
        runs.append(
            run_gauge_wave(
                nx=nx,
                steps=args.steps,
                wavelength=args.wavelength,
                amplitude=args.amplitude,
                show_progress=not args.no_progress,
            )
        )

    save_profile_plot(runs[-1], args.output_prefix)
    save_diagnostics_plot(runs, args.output_prefix)
    save_convergence_plot(runs, args.output_prefix)

    print(f"saved = {args.output_prefix}_profiles.png")
    print(f"saved = {args.output_prefix}_diagnostics.png")
    print(f"saved = {args.output_prefix}_convergence.png")


if __name__ == "__main__":
    main()