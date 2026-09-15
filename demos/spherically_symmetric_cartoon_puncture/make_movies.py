"""Render every saved spherical Cartoon BSSN field as an MP4."""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np
import openpmd_api as io


DATA_PATH = Path(__file__).resolve().parent / "output" / "cartoon_puncture.h5"
MOVIE_DIR = DATA_PATH.parent / "movies"
DISPLAY_RADIUS = 20.0
FPS = 5
DPI = 120


@dataclass(frozen=True)
class MovieSpec:
    """Describe one logical BSSN variable and its plotted components."""

    output_name: str
    title: str
    mesh_components: tuple[tuple[str, str | None], ...]
    labels: tuple[str, ...]


MOVIES = (
    MovieSpec("conformal_factor_W.mp4", "Conformal factor W", (("W", None),), (r"$W$",)),
    MovieSpec("trace_K.mp4", "Trace of extrinsic curvature K", (("K", None),), (r"$K$",)),
    MovieSpec("lapse.mp4", "Lapse", (("lapse", None),), (r"$\alpha$",)),
    MovieSpec(
        "shift.mp4",
        "Shift",
        (("shift", "x"), ("shift", "y"), ("shift", "z")),
        (r"$\beta^x$", r"$\beta^y$", r"$\beta^z$"),
    ),
    MovieSpec(
        "conformal_connection.mp4",
        "Conformal connection functions",
        (
            ("conformal_connection", "x"),
            ("conformal_connection", "y"),
            ("conformal_connection", "z"),
        ),
        (r"$\tilde{\Gamma}^x$", r"$\tilde{\Gamma}^y$", r"$\tilde{\Gamma}^z$"),
    ),
    MovieSpec(
        "conformal_metric.mp4",
        "Conformal spatial metric",
        tuple((f"conformal_metric_{name}", None) for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
        tuple(rf"$\tilde{{\gamma}}_{{{name}}}$" for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
    ),
    MovieSpec(
        "traceless_K.mp4",
        "Conformal traceless extrinsic curvature",
        tuple((f"traceless_K_{name}", None) for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
        tuple(rf"$\tilde{{A}}_{{{name}}}$" for name in ("xx", "xy", "xz", "yy", "yz", "zz")),
    ),
    MovieSpec(
        "hamiltonian_constraint.mp4",
        "Hamiltonian constraint",
        (("hamiltonian_constraint", None),),
        (r"$\mathcal{H}$",),
    ),
    MovieSpec(
        "momentum_constraint.mp4",
        "Momentum constraint",
        (
            ("momentum_constraint", "x"),
            ("momentum_constraint", "y"),
            ("momentum_constraint", "z"),
        ),
        (r"$\mathcal{M}_x$", r"$\mathcal{M}_y$", r"$\mathcal{M}_z$"),
    ),
    MovieSpec(
        "det_gamma_constraint.mp4",
        "Unit-determinant constraint",
        (("det_gamma_constraint", None),),
        (r"$\det(\tilde{\gamma})-1$",),
    ),
    MovieSpec(
        "trace_A_constraint.mp4",
        "Trace-free constraint",
        (("trace_A_constraint", None),),
        (r"$\tilde{\gamma}^{ij}\tilde{A}_{ij}$",),
    ),
    MovieSpec(
        "gamma_constraint.mp4",
        "Conformal-connection constraint",
        (
            ("gamma_constraint", "x"),
            ("gamma_constraint", "y"),
            ("gamma_constraint", "z"),
        ),
        (r"$\mathcal{G}^x$", r"$\mathcal{G}^y$", r"$\mathcal{G}^z$"),
    ),
)


def _component_name(mesh, requested_name: str | None):
    """Return an openPMD component key, including its special scalar key."""

    if requested_name is not None:
        return requested_name
    return next(iter(mesh))


def available_movie_specs(path: Path) -> tuple[tuple[MovieSpec, ...], tuple[MovieSpec, ...]]:
    """Separate movie specifications present in the first saved iteration."""

    series = io.Series(str(path), io.Access.read_only)
    steps = list(series.iterations)
    if not steps:
        series.close()
        raise ValueError(f"No iterations found in {path}")
    iteration = series.iterations[steps[0]]
    available = []
    missing = []
    for spec in MOVIES:
        present = True
        for mesh_name, requested_component in spec.mesh_components:
            if mesh_name not in iteration.meshes:
                present = False
                break
            if (
                requested_component is not None
                and requested_component not in iteration.meshes[mesh_name]
            ):
                present = False
                break
        (available if present else missing).append(spec)
    iteration.close()
    series.close()
    return tuple(available), tuple(missing)


def load_snapshots(path: Path, movie_specs: tuple[MovieSpec, ...] = MOVIES):
    """Load the requested 1D state fields and their coordinates from a series."""

    if not path.is_file():
        raise FileNotFoundError(f"Cartoon output not found: {path}")

    series = io.Series(str(path), io.Access.read_only)
    steps = list(series.iterations)
    if not steps:
        series.close()
        raise ValueError(f"No iterations found in {path}")

    requested = tuple(
        dict.fromkeys(
            component for movie in movie_specs for component in movie.mesh_components
        )
    )
    times = []
    snapshots = {component: [] for component in requested}
    x = None

    for step in steps:
        iteration = series.iterations[step]
        times.append(float(iteration.time))
        pending = {}

        for mesh_name, requested_component in requested:
            mesh = iteration.meshes[mesh_name]
            component = mesh[_component_name(mesh, requested_component)]
            pending[(mesh_name, requested_component)] = component.load_chunk()

            if x is None and mesh_name == "W":
                nx = component.shape[0]
                x = (
                    float(mesh.grid_global_offset[0])
                    + (np.arange(nx) + float(component.position[0]))
                    * float(mesh.grid_spacing[0])
                )

        series.flush()
        for component, array in pending.items():
            snapshots[component].append(np.array(array[:, 0, 0], copy=True))
        iteration.close()

    series.close()
    arrays = {component: np.stack(values) for component, values in snapshots.items()}
    return np.asarray(steps), np.asarray(times), x, arrays


def padded_limits(values: np.ndarray) -> tuple[float, float]:
    """Return finite, fixed plot limits across all frames for one component."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0

    lower = float(np.min(finite))
    upper = float(np.max(finite))
    scale = max(abs(lower), abs(upper), 1.0e-12)
    if np.isclose(lower, upper, rtol=1.0e-12, atol=1.0e-15):
        return lower - 0.05 * scale, upper + 0.05 * scale
    padding = 0.05 * (upper - lower)
    return lower - padding, upper + padding


def subplot_shape(component_count: int) -> tuple[int, int]:
    """Return a compact panel layout for a scalar, vector, or symmetric tensor."""

    if component_count == 1:
        return 1, 1
    if component_count == 3:
        return 3, 1
    if component_count == 6:
        return 3, 2
    raise ValueError(f"Unsupported component count: {component_count}")


def render_movie(
    spec: MovieSpec,
    steps: np.ndarray,
    times: np.ndarray,
    x: np.ndarray,
    snapshots: dict,
    view: np.ndarray,
    display_radius: float = DISPLAY_RADIUS,
    movie_dir: Path = MOVIE_DIR,
    fps: int = FPS,
    dpi: int = DPI,
) -> Path:
    """Render one BSSN variable, with one panel per independent component."""

    rows, columns = subplot_shape(len(spec.mesh_components))
    figure, axes_grid = plt.subplots(
        rows,
        columns,
        figsize=(10.0, 4.8 if rows == 1 else 9.0),
        squeeze=False,
        sharex=True,
    )
    axes = axes_grid.ravel()
    lines = []
    warnings = []

    for axis, component, label in zip(axes, spec.mesh_components, spec.labels):
        values = snapshots[component][:, view]
        line, = axis.plot(x[view], values[0], linewidth=1.5)
        axis.set_xlim(float(x[view][0]), float(x[view][-1]))
        axis.set_ylim(*padded_limits(values))
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        warning = axis.text(
            0.5,
            0.5,
            "NON-FINITE DATA",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color="crimson",
            fontsize=14,
            fontweight="bold",
            visible=False,
        )
        lines.append(line)
        warnings.append(warning)

    for axis in axes[-columns:]:
        axis.set_xlabel(r"$x/M$")

    frame_title = figure.suptitle("")
    figure.text(
        0.5,
        0.01,
        f"Central signed Cartoon axis, |x| <= {display_radius:g} M",
        ha="center",
        fontsize=9,
        color="0.35",
    )
    figure.tight_layout(rect=(0.0, 0.035, 1.0, 0.94))

    movie_dir.mkdir(parents=True, exist_ok=True)
    output_path = movie_dir / spec.output_name
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=2200,
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        metadata={"title": spec.title},
    )

    with writer.saving(figure, output_path, dpi=dpi):
        for frame, (step, time) in enumerate(zip(steps, times)):
            invalid_samples = 0
            total_samples = 0
            for line, warning, component in zip(lines, warnings, spec.mesh_components):
                values = snapshots[component][frame, view]
                finite = np.isfinite(values)
                line.set_data(x[view], np.where(finite, values, np.nan))
                warning.set_visible(not np.all(finite))
                invalid_samples += int(np.count_nonzero(~finite))
                total_samples += values.size

            status = ""
            if invalid_samples:
                status = f" -- NON-FINITE {invalid_samples}/{total_samples} samples"
                frame_title.set_color("crimson")
            else:
                frame_title.set_color("black")
            frame_title.set_text(
                f"{spec.title} -- step {int(step)}, t = {time:.6f} M{status}"
            )
            writer.grab_frame()

    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    """Return command-line controls without changing the demo defaults."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--display-radius", type=float, default=DISPLAY_RADIUS)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--dpi", type=int, default=DPI)
    return parser.parse_args()


def main() -> None:
    """Render one movie for each logical BSSN state variable."""

    args = parse_args()
    movie_dir = args.output_dir or args.input.parent / "movies"
    movie_specs, unavailable_specs = available_movie_specs(args.input)
    steps, times, x, snapshots = load_snapshots(args.input, movie_specs)
    view = np.abs(x) <= args.display_radius
    if not np.any(view):
        raise ValueError(
            f"No grid points lie inside |x| <= {args.display_radius}"
        )

    first_bad = None
    for frame in range(len(steps)):
        if any(not np.isfinite(values[frame]).all() for values in snapshots.values()):
            first_bad = frame
            break

    print(f"Loaded {len(steps)} snapshots from {args.input}")
    if unavailable_specs:
        print(
            "Skipped fields absent from this older file: "
            + ", ".join(spec.title for spec in unavailable_specs)
        )
    if first_bad is not None:
        print(
            "First non-finite saved state: "
            f"step {steps[first_bad]}, t={times[first_bad]:.16g} M"
        )

    for spec in movie_specs:
        output_path = render_movie(
            spec,
            steps,
            times,
            x,
            snapshots,
            view,
            display_radius=args.display_radius,
            movie_dir=movie_dir,
            fps=args.fps,
            dpi=args.dpi,
        )
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
