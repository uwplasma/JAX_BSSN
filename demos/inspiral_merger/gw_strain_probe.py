"""Plot the approximate gravitational-wave strain h+, hx sampled at fixed probes during the inspiral run.

inspiral_merger.py samples the physical metric gamma_ij = W^-2 gamma_tilde_ij every
PROBE_INTERVAL at points on the z and x axes and reads off the transverse-traceless part
along the propagation direction:

    z-axis probe (n = z):  h+ = (g_xx - g_yy) / 2,  hx = g_xy
    x-axis probe (n = x):  h+ = (g_yy - g_zz) / 2,  hx = g_yz

This leading-order estimate is gauge dependent at finite radius and, for probes inside a
wavelength of the source (about 20 M at merger), mixed with the near-zone field, so place
the probes at r >~ 20 M for comparison with analytic merger-ringdown waveforms.

Reads <INPUT_DIR>/gw_probe_samples.npz and writes <INPUT_DIR>/gw_strain_probe.png and
<INPUT_DIR>/gw_strain_probe.npz (t and every probe's h+ and hx, keyed hplus_<axis>-axis_r=<grid points>).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


INPUT_DIR = Path("output")  # output directory of inspiral_merger.py
RETARDED_TIME = True        # plot r h against t - r instead of h against t


def plot_strain(input_dir):
    """
    Plot r h+ and r hx at every probe and re-save them keyed by probe.

    Args:
        input_dir: Output directory of inspiral_merger.py holding gw_probe_samples.npz

    Returns:
        None
    """
    input_dir = Path(input_dir)
    data = np.load(input_dir / "gw_probe_samples.npz")
    t, strain, labels, radius, axes = data["t"], data["strain"], data["labels"], data["radius"], data["axis"]
    print(f"{len(t)} strain samples to t = {t[-1]:.2f} at {len(labels)} probes (r = {sorted(set(float(r) for r in radius.round(2)))} M)")

    np.savez(input_dir / "gw_strain_probe.npz", t=t,
             **{f"{kind}_{label.replace(' ', '_')}": strain[:, p, column]
                for p, label in enumerate(labels) for column, kind in enumerate(("hplus", "hcross"))})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for p, (label, r, axis) in enumerate(zip(labels, radius, axes)):
        u, scale = (t - r, r) if RETARDED_TIME else (t, 1.0)
        for row in (0, 1):
            ax[row, 0 if axis == "z" else 1].plot(u, scale * strain[:, p, row], label=f"{label} ({r:.1f} M)")
    prefix = "r h" if RETARDED_TIME else "h"
    for col, axis in enumerate(("z", "x")):
        ax[0, col].set(title=f"Probes on the {axis}-axis", ylabel=f"${prefix}_+$")
        ax[1, col].set(xlabel=r"Retarded time $t - r$" if RETARDED_TIME else "t (code units)", ylabel=rf"${prefix}_\times$")
        for a in ax[:, col]:
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
    fig.suptitle("Approximate GW strain from the physical metric")
    fig.tight_layout()
    fig.savefig(input_dir / "gw_strain_probe.png", dpi=150)
    print(f"Saved {input_dir / 'gw_strain_probe.png'} and {input_dir / 'gw_strain_probe.npz'}")


if __name__ == "__main__":
    plot_strain(INPUT_DIR)
