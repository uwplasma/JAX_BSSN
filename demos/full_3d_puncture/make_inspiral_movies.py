"""Render 2D slice movies from the inspiral-merger NumPy snapshots."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


DEFAULT_INPUT = Path(__file__).resolve().parent / "output"
DEFAULT_MOVIE_DIR = DEFAULT_INPUT / "movies"
FPS = 5
DPI = 110


def require_ffmpeg() -> str:
    """Return the ffmpeg executable or raise a clear prerequisite error."""

    executable = shutil.which("ffmpeg")
    if executable is None:
        raise RuntimeError(
            "ffmpeg is required to render MP4 movies. Install ffmpeg and make "
            "sure it is available on PATH."
        )
    return executable


@dataclass(frozen=True)
class SliceSpec:
    output_name: str
    field_name: str
    title: str


MOVIES = (
    SliceSpec("conformal_factor_W.mp4", "conformal_factor", "Conformal factor W"),
    SliceSpec("lapse.mp4", "lapse", "Lapse"),
    SliceSpec("trace_K.mp4", "trace_K", "Trace of extrinsic curvature K"),
)


def _snapshot_steps(path: Path) -> list[int]:
    steps = sorted(
        int(entry.stem.rsplit("_", 1)[-1])
        for entry in path.glob("time_step_*.npy")
    )
    if not steps:
        raise ValueError(f"no snapshot files found in {path}")
    return steps


def _load_scalar(path: Path, field_name: str, step: int) -> np.ndarray:
    return np.load(path / f"{field_name}_step_{step:06d}.npy")


def _load_vector_magnitude(path: Path, field_name: str, step: int) -> np.ndarray:
    vector = np.load(path / f"{field_name}_step_{step:06d}.npy")
    return np.sqrt(np.sum(vector**2, axis=0))


def _slice_planes(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = values.shape
    return (
        values[:, ny // 2, :].T.copy(),
        values[:, :, nz // 2].T.copy(),
        values[nx // 2, :, :].T.copy(),
    )


def _limits(path: Path, steps: list[int], field_name: str):
    minimum, maximum = np.inf, -np.inf
    for step in steps:
        values = _load_scalar(path, field_name, step)
        finite = values[np.isfinite(values)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
    if not np.isfinite(minimum):
        return -1.0, 1.0, "RdBu_r"
    if np.isclose(minimum, maximum):
        pad = max(abs(minimum), 1.0) * 0.05
        return minimum - pad, maximum + pad, "viridis"
    if minimum < 0.0 < maximum:
        bound = max(abs(minimum), abs(maximum))
        return -bound, bound, "RdBu_r"
    return minimum, maximum, "viridis"


def _format_extent(half_width: float, shape: tuple[int, int]) -> tuple[float, float, float, float]:
    nx, nz = shape
    dx = 2.0 * half_width / nx
    dz = 2.0 * half_width / nz
    return (
        -half_width,
        half_width,
        -half_width,
        half_width,
    )


def render_scalar_movie(
    input_dir: Path,
    output_dir: Path,
    spec: SliceSpec,
    steps: list[int],
    half_width: float,
    fps: int,
    dpi: int,
):
    limits = _limits(input_dir, steps, spec.field_name)
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    axes = axes.ravel()
    plane_labels = (
        (r"$x/M$", r"$z/M$", "y=0 slice"),
        (r"$x/M$", r"$y/M$", "z=0 slice"),
        (r"$y/M$", r"$z/M$", "x=0 slice"),
    )
    first = _load_scalar(input_dir, spec.field_name, steps[0])
    planes = _slice_planes(first)
    extent = _format_extent(half_width, (first.shape[0], first.shape[2]))

    images = []
    for axis, plane, (xlabel, ylabel, subtitle) in zip(axes, planes, plane_labels):
        image = axis.imshow(
            plane,
            origin="lower",
            extent=extent,
            cmap=limits[2],
            vmin=limits[0],
            vmax=limits[1],
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle)
        figure.colorbar(image, ax=axis, shrink=0.82)
        images.append(image)

    title = figure.suptitle("")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3000,
        extra_args=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    output = output_dir / spec.output_name
    with writer.saving(figure, output, dpi=dpi):
        for frame, step in enumerate(steps, 1):
            values = _load_scalar(input_dir, spec.field_name, step)
            invalid = not np.isfinite(values).all()
            for image, plane in zip(images, _slice_planes(values)):
                image.set_data(np.where(np.isfinite(plane), plane, np.nan))
            time = float(np.load(input_dir / f"time_step_{step:06d}.npy"))
            title.set_text(
                f"{spec.title} -- step {step}, t={time:.6f} M"
                + (" -- NON-FINITE DATA" if invalid else "")
            )
            title.set_color("crimson" if invalid else "black")
            writer.grab_frame()
            print(f"\r{spec.output_name}: frame {frame}/{len(steps)}", end="", flush=True)
    print()
    plt.close(figure)
    return output


def render_shift_magnitude_movie(
    input_dir: Path,
    output_dir: Path,
    steps: list[int],
    half_width: float,
    fps: int,
    dpi: int,
):
    figure, axes = plt.subplots(1, 3, figsize=(16, 5), squeeze=False)
    axes = axes.ravel()
    plane_labels = (
        (r"$x/M$", r"$z/M$", "y=0 slice"),
        (r"$x/M$", r"$y/M$", "z=0 slice"),
        (r"$y/M$", r"$z/M$", "x=0 slice"),
    )
    magnitude = _load_vector_magnitude(input_dir, "shift", steps[0])
    planes = _slice_planes(magnitude)
    finite = magnitude[np.isfinite(magnitude)]
    if finite.size:
        vmax = float(finite.max())
    else:
        vmax = 1.0
    if np.isclose(vmax, 0.0):
        vmax = 1.0e-12
    extent = _format_extent(half_width, (magnitude.shape[0], magnitude.shape[2]))

    images = []
    for axis, plane, (xlabel, ylabel, subtitle) in zip(axes, planes, plane_labels):
        image = axis.imshow(
            plane,
            origin="lower",
            extent=extent,
            cmap="magma",
            vmin=0.0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_title(subtitle)
        figure.colorbar(image, ax=axis, shrink=0.82)
        images.append(image)

    title = figure.suptitle("")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3000,
        extra_args=[
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
        ],
    )
    output = output_dir / "shift_magnitude.mp4"
    with writer.saving(figure, output, dpi=dpi):
        for frame, step in enumerate(steps, 1):
            magnitude = _load_vector_magnitude(input_dir, "shift", step)
            invalid = not np.isfinite(magnitude).all()
            for image, plane in zip(images, _slice_planes(magnitude)):
                image.set_data(np.where(np.isfinite(plane), plane, np.nan))
            time = float(np.load(input_dir / f"time_step_{step:06d}.npy"))
            title.set_text(
                f"Shift magnitude -- step {step}, t={time:.6f} M"
                + (" -- NON-FINITE DATA" if invalid else "")
            )
            title.set_color("crimson" if invalid else "black")
            writer.grab_frame()
            print(f"\rshift_magnitude.mp4: frame {frame}/{len(steps)}", end="", flush=True)
    print()
    plt.close(figure)
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MOVIE_DIR)
    parser.add_argument("--half-width", type=float, default=12.0)
    parser.add_argument("--fps", type=int, default=FPS)
    parser.add_argument("--dpi", type=int, default=DPI)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="render only the first N snapshots",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(f"snapshot directory not found: {args.input_dir}")
    require_ffmpeg()
    steps = _snapshot_steps(args.input_dir)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        steps = steps[: args.max_frames]
    print(f"Rendering {len(steps)} inspiral snapshots from {args.input_dir}")
    for spec in MOVIES:
        output = render_scalar_movie(
            args.input_dir,
            args.output_dir,
            spec,
            steps,
            args.half_width,
            args.fps,
            args.dpi,
        )
        print(f"Wrote {output}")
    output = render_shift_magnitude_movie(
        args.input_dir,
        args.output_dir,
        steps,
        args.half_width,
        args.fps,
        args.dpi,
    )
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()