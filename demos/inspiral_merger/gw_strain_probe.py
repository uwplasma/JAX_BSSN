"""Extract approximate gravitational-wave strain h+, hx at fixed points from inspiral snapshots.

Reads the conformal metric, conformal factor and time snapshots written by
inspiral_merger.py (memory-mapped, so only the probe points are read), rebuilds the
physical metric gamma_ij = W^-2 gamma_tilde_ij at each probe, and reads off the
transverse-traceless part along the propagation direction:

    z-axis probe (n = z):  h+ = (g_xx - g_yy) / 2,  hx = g_xy
    x-axis probe (n = x):  h+ = (g_yy - g_zz) / 2,  hx = g_yz

This leading-order estimate is gauge dependent at finite radius and, for probes inside a
wavelength of the source, mixed with the near-zone field (Psi4 on distant spheres is the
gauge-invariant alternative), but it shows the outgoing merger signal.

Writes output/gw_strain_probe.png and output/gw_strain_probe.npz (t and every probe's
h+ and hx, keyed hplus_<axis>-axis_r=<grid points>).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SNAPSHOT_DIR = Path("output")         # directory written by inspiral_merger.py
HALF_WIDTH = 12.0                     # must match inspiral_merger.py
PROBE_FRACTIONS = [0.20, 0.30, 0.40]  # probe radius as a fraction of the grid width (< 0.5)
RETARDED_TIME = True                  # plot r h against t - r instead of h against t
OUTPUT_DIR = Path("output")


def probe_strain(gamma_tilde, W, index, axis):
    """
    Return h+ and hx from the physical metric at one grid point.

    Args:
        gamma_tilde: Memory-mapped conformal metric (3, 3, n, n, n)
        W: Memory-mapped conformal factor (n, n, n)
        index: (i, j, k) grid index of the probe
        axis: Propagation direction, "z" or "x"

    Returns:
        Tuple (h_plus, h_cross)
    """
    g = np.asarray(gamma_tilde[(slice(None), slice(None)) + index]) / float(W[index])**2
    return (0.5 * (g[0, 0] - g[1, 1]), g[0, 1]) if axis == "z" else (0.5 * (g[1, 1] - g[2, 2]), g[1, 2])


if __name__ == "__main__":
    steps = sorted(int(path.stem.rsplit("_", 1)[-1]) for path in SNAPSHOT_DIR.glob("time_step_*.npy"))
    if not steps:
        raise SystemExit(f"No inspiral snapshots found in {SNAPSHOT_DIR}")
    load = lambda name, step: np.load(SNAPSHOT_DIR / f"{name}_step_{step:06d}.npy", mmap_mode="r")
    n = load("conformal_factor", steps[0]).shape[0]
    dx = 2 * HALF_WIDTH / n
    probes = {f"{axis}-axis r={int(f * n)}": (axis, int(f * n), (n // 2, n // 2, n // 2 + int(f * n)) if axis == "z"
                                              else (n // 2 + int(f * n), n // 2, n // 2))
              for f in PROBE_FRACTIONS for axis in ("z", "x")}
    print(f"{len(steps)} snapshots on a {n}^3 grid (dx = {dx:.4f}); {len(probes)} probes")

    t, strain = [], {label: [] for label in probes}
    for count, step in enumerate(steps, 1):
        gamma_tilde, W = load("conformal_metric", step), load("conformal_factor", step)
        t.append(float(load("time", step)))
        for label, (axis, _, index) in probes.items():
            strain[label].append(probe_strain(gamma_tilde, W, index, axis))
        print(f"\rreading snapshot {count}/{len(steps)} (t = {t[-1]:.2f})", end="", flush=True)
    print()
    t = np.array(t)
    strain = {label: np.array(values) for label, values in strain.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUTPUT_DIR / "gw_strain_probe.npz", t=t,
             **{f"{kind}_{label.replace(' ', '_')}": strain[label][:, column]
                for label in probes for column, kind in enumerate(("hplus", "hcross"))})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for label, (axis, r, _) in probes.items():
        col, radius = (0 if axis == "z" else 1), r * dx
        u, scale = (t - radius, radius) if RETARDED_TIME else (t, 1.0)
        for row in (0, 1):
            ax[row, col].plot(u, scale * strain[label][:, row], label=label)
    prefix = "r h" if RETARDED_TIME else "h"
    for col, axis in enumerate(("z", "x")):
        ax[0, col].set(title=f"Probes on the {axis}-axis", ylabel=f"${prefix}_+$")
        ax[1, col].set(xlabel=r"Retarded time $t - r$" if RETARDED_TIME else "t (code units)", ylabel=rf"${prefix}_\times$")
        for a in ax[:, col]:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle("Approximate GW strain from the physical metric")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "gw_strain_probe.png", dpi=150)
    print(f"Saved {OUTPUT_DIR / 'gw_strain_probe.png'} and {OUTPUT_DIR / 'gw_strain_probe.npz'}")
