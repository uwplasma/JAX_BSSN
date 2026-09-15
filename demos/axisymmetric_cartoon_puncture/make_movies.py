"""Render every saved axisymmetric BSSN field as an x-z-plane MP4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import openpmd_api as io


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "axisymmetric_cartoon_puncture.h5"
)
DEFAULT_MOVIE_DIR = DEFAULT_INPUT.parent / "movies"
FPS = 5
DPI = 110


@dataclass(frozen=True)
class MovieSpec:
    output_name: str
    title: str
    components: tuple[tuple[str, str | None, str], ...]


MOVIES = (
    MovieSpec("conformal_factor_W.mp4", "Conformal factor W", (("W", None, r"$W$"),)),
    MovieSpec("trace_K.mp4", "Trace of extrinsic curvature K", (("K", None, r"$K$"),)),
    MovieSpec("lapse.mp4", "Lapse", (("lapse", None, r"$\alpha$"),)),
    MovieSpec("shift.mp4", "Shift", tuple(("shift", c, rf"$\beta^{c}$") for c in "xyz")),
    MovieSpec(
        "conformal_connection.mp4",
        "Conformal connection functions",
        tuple(("conformal_connection", c, rf"$\widetilde{{\Gamma}}^{c}$") for c in "xyz"),
    ),
    MovieSpec(
        "conformal_metric.mp4",
        "Conformal spatial metric",
        tuple((f"conformal_metric_{c}", None, rf"$\widetilde{{\gamma}}_{{{c}}}$") for c in ("xx", "xy", "xz", "yy", "yz", "zz")),
    ),
    MovieSpec(
        "traceless_K.mp4",
        "Conformal traceless extrinsic curvature",
        tuple((f"traceless_K_{c}", None, rf"$\widetilde{{A}}_{{{c}}}$") for c in ("xx", "xy", "xz", "yy", "yz", "zz")),
    ),
    MovieSpec("hamiltonian_constraint.mp4", "Hamiltonian constraint", (("hamiltonian_constraint", None, r"$\mathcal{H}$"),)),
    MovieSpec("momentum_constraint.mp4", "Momentum constraint", tuple(("momentum_constraint", c, rf"$\mathcal{{M}}_{c}$") for c in "xyz")),
    MovieSpec("det_gamma_constraint.mp4", "Unit-determinant constraint", (("det_gamma_constraint", None, r"$\det(\widetilde{\gamma})-1$"),)),
    MovieSpec("trace_A_constraint.mp4", "Trace-free constraint", (("trace_A_constraint", None, r"$\widetilde{\gamma}^{ij}\widetilde{A}_{ij}$"),)),
    MovieSpec("gamma_constraint.mp4", "Conformal-connection constraint", tuple(("gamma_constraint", c, rf"$\mathcal{{G}}^{c}$") for c in "xyz")),
)


def _component(mesh, name: str | None):
    return mesh[name] if name is not None else mesh[next(iter(mesh))]


def _read_plane(series, iteration, mesh_name: str, component_name: str | None):
    component = _component(iteration.meshes[mesh_name], component_name)
    pending = component.load_chunk()
    series.flush()
    return np.asarray(pending)[:, 0, :].T.copy()


def _limits(path: Path, steps: list[int], mesh_name: str, component_name: str | None):
    minimum, maximum = np.inf, -np.inf
    series = io.Series(str(path), io.Access.read_only)
    for step in steps:
        iteration = series.iterations[step]
        values = _read_plane(series, iteration, mesh_name, component_name)
        finite = values[np.isfinite(values)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
        iteration.close()
    series.close()
    if not np.isfinite(minimum):
        return -1.0, 1.0, "RdBu_r"
    if np.isclose(minimum, maximum):
        pad = max(abs(minimum), 1.0) * 0.05
        return minimum - pad, maximum + pad, "viridis"
    if minimum < 0.0 < maximum:
        bound = max(abs(minimum), abs(maximum))
        return -bound, bound, "RdBu_r"
    return minimum, maximum, "viridis"


def _coordinates(mesh, component):
    coords = []
    for axis in (0, 2):
        coords.append(
            float(mesh.grid_global_offset[axis])
            + (np.arange(component.shape[axis]) + float(component.position[axis]))
            * float(mesh.grid_spacing[axis])
        )
    return coords


def _available_specs(path: Path):
    series = io.Series(str(path), io.Access.read_only)
    steps = list(series.iterations)
    if not steps:
        series.close()
        raise ValueError(f"no iterations found in {path}")
    iteration = series.iterations[steps[0]]
    available, missing = [], []
    for spec in MOVIES:
        present = all(
            mesh_name in iteration.meshes
            and (name is None or name in iteration.meshes[mesh_name])
            for mesh_name, name, _ in spec.components
        )
        (available if present else missing).append(spec)
    first_mesh, first_name, _ = available[0].components[0]
    x, z = _coordinates(iteration.meshes[first_mesh], _component(iteration.meshes[first_mesh], first_name))
    iteration.close()
    series.close()
    return steps, x, z, tuple(available), tuple(missing)


def render_movie(path: Path, output_dir: Path, spec: MovieSpec, steps: list[int], x, z, fps: int, dpi: int):
    limits = [_limits(path, steps, mesh, component) for mesh, component, _ in spec.components]
    count = len(spec.components)
    columns = 1 if count == 1 else 2
    rows = math.ceil(count / columns)
    figure, axes_grid = plt.subplots(rows, columns, figsize=(7.5 * columns, 6.0 * rows), squeeze=False)
    axes = axes_grid.ravel()
    images = []
    extent = (x[0] - 0.5 * (x[1] - x[0]), x[-1] + 0.5 * (x[1] - x[0]), z[0] - 0.5 * (z[1] - z[0]), z[-1] + 0.5 * (z[1] - z[0]))

    for axis, (_, _, label), (vmin, vmax, cmap) in zip(axes, spec.components, limits):
        image = axis.imshow(np.zeros((z.size, x.size)), origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest", aspect="equal")
        axis.set_xlabel(r"$x/M$")
        axis.set_ylabel(r"$z/M$")
        axis.set_title(label)
        figure.colorbar(image, ax=axis, shrink=0.85)
        images.append(image)
    for axis in axes[count:]:
        axis.set_visible(False)
    title = figure.suptitle("")
    figure.tight_layout(rect=(0, 0, 1, 0.96))

    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / spec.output_name
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=3000, extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    series = io.Series(str(path), io.Access.read_only)
    with writer.saving(figure, output, dpi=dpi):
        for frame, step in enumerate(steps, 1):
            iteration = series.iterations[step]
            invalid = False
            for image, (mesh, component, _) in zip(images, spec.components):
                plane = _read_plane(series, iteration, mesh, component)
                invalid |= not np.isfinite(plane).all()
                image.set_data(np.where(np.isfinite(plane), plane, np.nan))
            title.set_text(f"{spec.title} -- step {step}, t={float(iteration.time):.6f} M" + (" -- NON-FINITE DATA" if invalid else ""))
            title.set_color("crimson" if invalid else "black")
            writer.grab_frame()
            iteration.close()
            print(f"\r{spec.output_name}: frame {frame}/{len(steps)}", end="", flush=True)
    print()
    series.close()
    plt.close(figure)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument("--max-frames", type=int, default=None, help="render only the first N snapshots (useful for quick checks)")
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"axisymmetric output not found: {args.input}")
    steps, x, z, specs, missing = _available_specs(args.input)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        steps = steps[: args.max_frames]
    output_dir = args.output_dir or args.input.parent / "movies"
    print(f"Rendering {len(steps)} snapshots from {args.input}")
    if missing:
        print("Skipping absent fields: " + ", ".join(spec.title for spec in missing))
    for spec in specs:
        print(f"Scanning fixed color limits for {spec.title}")
        output = render_movie(args.input, output_dir, spec, steps, x, z, args.fps, args.dpi)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
