"""Three-resolution self-convergence test for the inspiral merger demo.

Runs inspiral_merger.py at three grid sizes in a 2:1 ladder with the same physics and
final time (no snapshots). At the final time each run's conformal factor W and lapse are
compared with the next finer run, block-averaged onto the coarser cells (the correct
restriction for the cell-centred grid). The observed order is
p = log2(||u_c - R u_m|| / ||u_m - R u_f||). The punctures are not smooth, so expect
below the scheme's formal order near them.

Writes output/convergence/inspiral_merger_convergence.png (constraint histories and
self-convergence errors) and prints the error table.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import inspiral_merger as binary


GRID_SIZES = [128, 256, 512]           # grid points per dimension, each twice the previous (divisible by the tiles)
FIELDS = {"conformal_factor": r"$W$", "lapse": r"$\alpha$"}
OUTPUT_DIR = Path("output") / "convergence"


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
    if binary.ROOT:
        print(f"Self-convergence over {GRID_SIZES}^3 to t = {binary.FINAL_TIME}")
    runs = []
    for num_points in GRID_SIZES:
        state, dx, history = binary.run_inspiral_merger(num_points, OUTPUT_DIR / f"nx_{num_points:04d}", 0)
        runs.append({"dx": dx, "history": history, **{name: np.asarray(getattr(state, name)) for name in FIELDS}})
        del state

    if not binary.ROOT:  # only process 0 reports on a multi-process run
        raise SystemExit
    dx = np.array([run["dx"] for run in runs[:-1]])
    errors = {name: np.array([np.sqrt(np.mean((coarse[name] - restrict(fine[name]))**2))
                              for coarse, fine in zip(runs[:-1], runs[1:])]) for name in FIELDS}
    print("\nfield              " + "".join(f"  ||{a}-R{b}||" for a, b in zip(GRID_SIZES[:-1], GRID_SIZES[1:])) + "   order p")
    for name, error in errors.items():
        print(f"{name:<18} " + "".join(f"  {e:>14.4e}" for e in error) + f"   {np.log2(error[0] / error[1]):.2f}")
    for num_points, run in zip(GRID_SIZES, runs):
        print(f"[{num_points}^3] final Hamiltonian L2 = {run['history'][-1, 1]:.4e}, momentum L2 = {run['history'][-1, 2]:.4e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    for num_points, run in zip(GRID_SIZES, runs):
        line, = ax1.semilogy(run["history"][:, 0], run["history"][:, 1], label=rf"$N={num_points}$ Hamiltonian")
        ax1.semilogy(run["history"][:, 0], run["history"][:, 2], "--", color=line.get_color(), label=rf"$N={num_points}$ momentum")
    ax1.set(xlabel=r"$t/M$", ylabel=r"$L_2$ norm", title="Constraint histories")
    for name, label in FIELDS.items():
        ax2.loglog(dx, errors[name], "o-", label=f"{label} (p = {np.log2(errors[name][0] / errors[name][1]):.2f})")
    ax2.loglog(dx, errors["conformal_factor"][-1] * (dx / dx[-1])**4, "k:", label="4th order")
    ax2.set(xlabel=r"$\Delta x$", ylabel=r"$||u_h - R\,u_{h/2}||_2$", title="Final-time self-convergence")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "inspiral_merger_convergence.png", dpi=200)
    print(f"Saved {OUTPUT_DIR / 'inspiral_merger_convergence.png'}")
