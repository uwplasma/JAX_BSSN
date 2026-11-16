"""Extract approximate GW strain h+, hx at fixed probe points from inspiral snapshots.

Reads the ``conformal_metric_step_*.npy``, ``conformal_factor_step_*.npy`` and
``time_step_*.npy`` files written by ``inspiral_merger.py``, rebuilds the physical
metric gamma_ij = W**(-2) gamma_tilde_ij at each probe point, and reads off the
transverse-traceless part along the propagation direction:

    z-axis probe (n = z):  h+ = (g_xx - g_yy) / 2,  hx = g_xy
    x-axis probe (n = x):  h+ = (g_yy - g_zz) / 2,  hx = g_yz

This leading-order metric-perturbation estimate is gauge dependent at finite
radius (Psi4 is the gauge-invariant alternative), but it shows the chirp.
"""

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SNAPSHOT_DIR = Path("output")       # directory holding the inspiral_merger.py snapshots
PROBE_FRACTIONS = [0.20, 0.30, 0.40]  # probe radius as a fraction of the grid width
HALF_WIDTH = 12.0                   # half width of the inspiral domain (code units)
RETARDED_TIME = True                # plot r*h against t - r instead of h against t
W_FLOOR = 1.0e-12                   # floor on W before dividing
OUTPUT_DIR = Path("output")
OUTPUT_PREFIX = "gw_strain_probe"


def step_of(path):
    """
    Return the integer step number encoded in a snapshot filename.

    Args:
        path: Snapshot path ending in ``_step_XXXXXX.npy``

    Returns:
        Step number as an int
    """
    return int(re.search(r"(\d+)\.npy$", path.name).group(1))


def find_snapshots(snapshot_dir):
    """
    Collect the snapshot steps that have a metric, conformal factor and time.

    Args:
        snapshot_dir: Directory containing the inspiral snapshot files

    Returns:
        Sorted list of (step, metric path, conformal factor path, time path)
    """
    metric = {step_of(p): p for p in snapshot_dir.glob("conformal_metric_step_*.npy")}
    factor = {step_of(p): p for p in snapshot_dir.glob("conformal_factor_step_*.npy")}
    times = {step_of(p): p for p in snapshot_dir.glob("time_step_*.npy")}
    steps = sorted(set(metric) & set(factor) & set(times))
    return [(s, metric[s], factor[s], times[s]) for s in steps]


def build_probes(num_points, fractions):
    """
    Place probe points on the +z and +x axes through the grid centre.

    Args:
        num_points: Grid points per dimension
        fractions: Probe radii as fractions of the grid width

    Returns:
        Dict mapping a label to (axis name, radius in grid cells, index triple)
    """
    centre = np.array([num_points // 2] * 3)
    probes = {}
    for fraction in fractions:
        r = int(fraction * num_points)
        probes[f"z-axis r={r}"] = ("z", r, tuple(centre + [0, 0, r]))
        probes[f"x-axis r={r}"] = ("x", r, tuple(centre + [r, 0, 0]))
    return probes


def physical_metric_at(conformal_metric, conformal_factor, index):
    """
    Rebuild the 3x3 physical metric at one grid point.

    Args:
        conformal_metric: Memory-mapped conformal metric with shape (3, 3, n, n, n)
        conformal_factor: Memory-mapped W with shape (n, n, n)
        index: (i, j, k) grid index of the probe

    Returns:
        3x3 NumPy array gamma_ij = gamma_tilde_ij / W**2
    """
    i, j, k = index
    W = max(float(conformal_factor[i, j, k]), W_FLOOR)
    return np.asarray(conformal_metric[:, :, i, j, k], dtype=np.float64) / W**2


def strain_from_metric(g, axis):
    """
    Read h+ and hx off the physical metric for a wave travelling along an axis.

    Args:
        g: 3x3 physical metric at the probe
        axis: Propagation direction, "z" or "x"

    Returns:
        Tuple (h_plus, h_cross)
    """
    if axis == "z":
        return 0.5 * (g[0, 0] - g[1, 1]), g[0, 1]
    return 0.5 * (g[1, 1] - g[2, 2]), g[1, 2]


def extract_strain(snapshots, probes):
    """
    Sample h+ and hx at every probe for every snapshot.

    Args:
        snapshots: Output of find_snapshots
        probes: Output of build_probes

    Returns:
        Tuple (times, h_plus dict, h_cross dict) keyed by probe label
    """
    times = []
    h_plus = {label: [] for label in probes}
    h_cross = {label: [] for label in probes}
    for _, metric_path, factor_path, time_path in snapshots:
        conformal_metric = np.load(metric_path, mmap_mode="r")
        conformal_factor = np.load(factor_path, mmap_mode="r")
        times.append(float(np.load(time_path)))
        for label, (axis, _, index) in probes.items():
            g = physical_metric_at(conformal_metric, conformal_factor, index)
            hp, hx = strain_from_metric(g, axis)
            h_plus[label].append(hp)
            h_cross[label].append(hx)
    return np.array(times), h_plus, h_cross


def plot_strain(times, h_plus, h_cross, probes, dx, output_prefix):
    """
    Plot h+ and hx for the z-axis and x-axis probes and save the figure.

    Args:
        times: Snapshot times (code units)
        h_plus: Dict of h+ series keyed by probe label
        h_cross: Dict of hx series keyed by probe label
        probes: Output of build_probes
        dx: Grid spacing (code units), used for the retarded-time axis
        output_prefix: Path prefix for the saved PNG

    Returns:
        None
    """
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for col, axis in enumerate(["z", "x"]):
        for label, (probe_axis, r, _) in probes.items():
            if probe_axis != axis:
                continue
            radius = r * dx
            if RETARDED_TIME:
                t_plot, scale = times - radius, radius
            else:
                t_plot, scale = times, 1.0
            ax[0, col].plot(t_plot, scale * np.array(h_plus[label]), label=label)
            ax[1, col].plot(t_plot, scale * np.array(h_cross[label]), label=label)
        prefix = "r h" if RETARDED_TIME else "h"
        ax[0, col].set_title(f"Probes on the {axis}-axis")
        ax[0, col].set_ylabel(f"${prefix}_+$")
        ax[1, col].set_ylabel(rf"${prefix}_\times$")
        ax[1, col].set_xlabel(r"Retarded time $t - r$" if RETARDED_TIME else "t (code units)")
        for a in ax[:, col]:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle("Approximate GW strain from the physical metric")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}.png", dpi=150)


def main():
    """Extract probe strains from the inspiral snapshots and save plot and data."""

    snapshots = find_snapshots(SNAPSHOT_DIR)
    if not snapshots:
        raise SystemExit(f"No inspiral snapshots found in {SNAPSHOT_DIR}")
    print(f"{len(snapshots)} snapshots with metric, conformal factor and time")

    num_points = np.load(snapshots[0][2], mmap_mode="r").shape[0]
    dx = 2.0 * HALF_WIDTH / num_points
    probes = build_probes(num_points, PROBE_FRACTIONS)

    times, h_plus, h_cross = extract_strain(snapshots, probes)
    print(f"t from {times[0]} to {times[-1]}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_prefix = OUTPUT_DIR / OUTPUT_PREFIX
    np.savez(
        f"{output_prefix}.npz",
        t=times,
        **{f"hplus_{k.replace(' ', '_')}": np.array(v) for k, v in h_plus.items()},
        **{f"hcross_{k.replace(' ', '_')}": np.array(v) for k, v in h_cross.items()},
    )
    plot_strain(times, h_plus, h_cross, probes, dx, output_prefix)
    print(f"saved {output_prefix}.png and {output_prefix}.npz")


if __name__ == "__main__":
    main()
