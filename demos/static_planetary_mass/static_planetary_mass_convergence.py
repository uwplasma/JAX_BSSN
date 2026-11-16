"""Grid self-convergence test for the static planetary mass demo.

Runs static_planetary_mass.py at three resolutions in a 2:1 ladder with the same
physical setup and final time. Each run is compared with the next finer one, block-
averaged onto the coarser cells (the correct restriction for a cell-centred grid), for
the BSSN lapse potential and for the periodic Poisson reference. The observed order is
p = log2(||Phi_c - R Phi_m|| / ||Phi_m - R Phi_f||). The planet's sharp surface limits
the attainable order, so the Poisson curve shows what the source itself permits.

Writes output/static_planetary_mass_convergence.png and prints the error table.
"""

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import static_planetary_mass as planet


RESOLUTIONS = [64, 128, 256]  # grid points per dimension, each twice the previous
FIELDS = {"lapse": r"BSSN lapse $\frac{1}{2}(\alpha^2-1)$", "poisson": r"Periodic Poisson $\Phi_{\mathrm{per}}$"}


def restrict(field):
    """
    Block-average a cell-centred field onto the grid with half the points per axis.

    Args:
        field: 3D field (2n, 2n, 2n)

    Returns:
        Restricted field (n, n, n)
    """
    n = field.shape[0] // 2
    return field.reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5))


if __name__ == "__main__":
    print(f"Self-convergence over {RESOLUTIONS}^3 to t = {planet.FINAL_TIME:.3e} m")
    runs = []
    for num_points in RESOLUTIONS:
        result = planet.run_static_mass(num_points)
        runs.append({"dx": result["dx"], **{name: result[name] for name in FIELDS}})
        print(f"[{num_points}^3] RMS(lapse - Poisson) = "
              f"{float(jnp.sqrt(jnp.mean((result['lapse'] - result['poisson'])**2))):.4e}")
        del result

    dx = np.array([run["dx"] for run in runs[:-1]])
    errors = {name: np.array([float(jnp.sqrt(jnp.mean((coarse[name] - restrict(fine[name]))**2)))
                              for coarse, fine in zip(runs[:-1], runs[1:])]) for name in FIELDS}
    print("\nfield     " + "".join(f"  ||{a}-R{b}||" for a, b in zip(RESOLUTIONS[:-1], RESOLUTIONS[1:])) + "   order p")
    for name, error in errors.items():
        order = np.log2(error[:-1] / error[1:])
        print(f"{name:<9} " + "".join(f"  {e:>14.4e}" for e in error) + "   " + ", ".join(f"{p:.2f}" for p in order))

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, label in FIELDS.items():
        ax.loglog(dx, errors[name], "o-", label=rf"{label}: $||\Phi_h - R\,\Phi_{{h/2}}||_2$")
    ax.loglog(dx, errors["lapse"][-1] * (dx / dx[-1])**2, "k--", label=r"2nd order ($\mathcal{O}(\Delta x^2)$)")
    ax.set(xlabel=r"Grid spacing $\Delta x$ [m]", ylabel=r"$L_2$ error", title="Static planetary mass self-convergence")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    planet.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(planet.OUTPUT_DIR / "static_planetary_mass_convergence.png", dpi=300)
    print(f"Saved {planet.OUTPUT_DIR / 'static_planetary_mass_convergence.png'}")
