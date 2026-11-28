"""Render MP4 movies of the inspiral merger from its NumPy snapshots.

For W, the lapse, trace K and the shift magnitude, each movie shows the y = 0, z = 0 and
x = 0 slices through the grid centre at every snapshot written by inspiral_merger.py.
Snapshots are memory-mapped so only the three slice planes are read from disk. The colour
range is fixed over the whole run (symmetric when the field changes sign). Requires ffmpeg.

Writes output/movies/{conformal_factor,lapse,trace_K,shift_magnitude}.mp4.
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


INPUT_DIR = Path("output")         # snapshot directory written by inspiral_merger.py
MOVIE_DIR = INPUT_DIR / "movies"
HALF_WIDTH = 12.0                  # must match inspiral_merger.py
FPS = 5
DPI = 110
MAX_FRAMES = None                  # render only the first N snapshots; None renders all
MOVIES = {"conformal_factor": "Conformal factor W", "lapse": "Lapse", "trace_K": "Trace of extrinsic curvature K",
          "shift_magnitude": "Shift magnitude"}
PLANES = (("y = 0", r"$x/M$", r"$z/M$"), ("z = 0", r"$x/M$", r"$y/M$"), ("x = 0", r"$y/M$", r"$z/M$"))


def load_planes(name, step):
    """
    Read the three central slices of one snapshot field.

    Args:
        name: BSSN field name, or "shift_magnitude" for |beta|
        step: Snapshot step number

    Returns:
        List of three 2D arrays (y = 0, z = 0 and x = 0 planes), transposed for imshow
    """
    field = np.load(INPUT_DIR / f"{'shift' if name == 'shift_magnitude' else name}_step_{step:06d}.npy", mmap_mode="r")
    n = field.shape[-1] // 2
    planes = [field[..., :, n, :], field[..., :, :, n], field[..., n, :, :]]
    if name == "shift_magnitude":
        planes = [np.sqrt(np.sum(np.square(plane), axis=0)) for plane in planes]
    return [np.array(plane).T for plane in planes]


def render_movie(name, title, steps):
    """
    Render one three-panel slice movie over every snapshot.

    Args:
        name: Field name (key of MOVIES)
        title: Figure title prefix
        steps: Sorted snapshot step numbers

    Returns:
        Path of the written MP4
    """
    frames = [load_planes(name, step) for step in steps]
    values = np.concatenate([plane[np.isfinite(plane)] for frame in frames for plane in frame])
    lo, hi = (values.min(), values.max()) if values.size else (-1.0, 1.0)
    cmap = "viridis"
    if lo < 0 < hi:
        lo, hi, cmap = -max(-lo, hi), max(-lo, hi), "RdBu_r"
    if np.isclose(lo, hi):
        lo, hi = lo - 1e-12, hi + 1e-12
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    images = []
    for ax, plane, (label, xlabel, ylabel) in zip(axes, frames[0], PLANES):
        images.append(ax.imshow(plane, origin="lower", extent=(-HALF_WIDTH, HALF_WIDTH) * 2, cmap=cmap,
                                vmin=lo, vmax=hi, interpolation="nearest"))
        ax.set(title=f"{label} slice", xlabel=xlabel, ylabel=ylabel)
        fig.colorbar(images[-1], ax=ax, shrink=0.82)
    suptitle = fig.suptitle("")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = MOVIE_DIR / f"{name}.mp4"
    writer = FFMpegWriter(fps=FPS, codec="libx264", bitrate=3000,
                          extra_args=["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p"])
    with writer.saving(fig, output, dpi=DPI):
        for index, (step, frame) in enumerate(zip(steps, frames), 1):
            finite = all(np.isfinite(plane).all() for plane in frame)
            for image, plane in zip(images, frame):
                image.set_data(plane)
            time = float(np.load(INPUT_DIR / f"time_step_{step:06d}.npy"))
            suptitle.set_text(f"{title}, step {step}, t = {time:.3f} M" + ("" if finite else " (NON-FINITE DATA)"))
            suptitle.set_color("black" if finite else "crimson")
            writer.grab_frame()
            print(f"\r{output.name}: frame {index}/{len(steps)}", end="", flush=True)
    print()
    plt.close(fig)
    return output


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required to write MP4 movies; install it and put it on PATH")
    steps = sorted(int(path.stem.rsplit("_", 1)[-1]) for path in INPUT_DIR.glob("time_step_*.npy"))[:MAX_FRAMES]
    if not steps:
        raise SystemExit(f"No snapshots found in {INPUT_DIR}")
    MOVIE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Rendering {len(steps)} snapshots from {INPUT_DIR}")
    for name, title in MOVIES.items():
        print(f"Saved {render_movie(name, title, steps)}")
