"""Render MP4 movies from the saved single-puncture BSSN snapshots."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import numpy as np


DATA_DIR = Path(__file__).resolve().parent / "output"
DOMAIN_EXTENT = (-8.0, 8.0, -8.0, 8.0)
FPS = 3
DPI = 120


def snapshot_steps(data_dir: Path) -> list[int]:
    """Return the ordered steps shared by the requested fields and times."""

    field_names = ("lapse", "shift", "conformal_factor", "time")
    steps_by_field = {}

    for field_name in field_names:
        steps_by_field[field_name] = {
            int(path.stem.rsplit("_step_", 1)[1])
            for path in data_dir.glob(f"{field_name}_step_*.npy")
        }

    lapse_steps = steps_by_field["lapse"]
    if not lapse_steps:
        raise FileNotFoundError(f"No lapse snapshots found in {data_dir}")

    if any(steps != lapse_steps for steps in steps_by_field.values()):
        raise ValueError("Lapse, shift, conformal-factor, and time steps must match")

    return sorted(lapse_steps)


def load_plane(data_dir: Path, field_name: str, step: int) -> np.ndarray:
    """Load the central x-y plane from one scalar snapshot."""

    field = np.load(
        data_dir / f"{field_name}_step_{step:06d}.npy",
        mmap_mode="r",
    )
    center_z = field.shape[-1] // 2
    return np.array(field[:, :, center_z]).T


def load_shift_planes(data_dir: Path, step: int) -> np.ndarray:
    """Load the three central x-y shift-component planes."""

    shift = np.load(
        data_dir / f"shift_step_{step:06d}.npy",
        mmap_mode="r",
    )
    center_z = shift.shape[-1] // 2
    return np.array(shift[:, :, :, center_z]).transpose(0, 2, 1)


def load_time(data_dir: Path, step: int) -> float:
    """Load the physical time associated with one snapshot."""

    return float(np.load(data_dir / f"time_step_{step:06d}.npy"))


def scalar_color_limits(
    data_dir: Path,
    field_name: str,
    steps: list[int],
) -> tuple[float, float]:
    """Find fixed color limits across all central slices of one field."""

    minimum = np.inf
    maximum = -np.inf

    for step in steps:
        plane = load_plane(data_dir, field_name, step)
        minimum = min(minimum, float(np.min(plane)))
        maximum = max(maximum, float(np.max(plane)))

    return minimum, maximum


def shift_color_limits(data_dir: Path, steps: list[int]) -> np.ndarray:
    """Find one fixed symmetric color scale for each shift component."""

    maximum_magnitudes = np.zeros(3)

    for step in steps:
        planes = load_shift_planes(data_dir, step)
        maximum_magnitudes = np.maximum(
            maximum_magnitudes,
            np.max(np.abs(planes), axis=(1, 2)),
        )

    return maximum_magnitudes


def render_scalar_movie(
    data_dir: Path,
    steps: list[int],
    field_name: str,
    field_label: str,
    output_name: str,
) -> None:
    """Render one scalar central-slice movie with fixed color limits."""

    color_minimum, color_maximum = scalar_color_limits(
        data_dir,
        field_name,
        steps,
    )

    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(
        load_plane(data_dir, field_name, steps[0]),
        origin="lower",
        extent=DOMAIN_EXTENT,
        cmap="viridis",
        vmin=color_minimum,
        vmax=color_maximum,
        interpolation="nearest",
    )
    axis.set_xlabel(r"$x/M$")
    axis.set_ylabel(r"$y/M$")
    axis.set_aspect("equal")
    title = axis.set_title("")
    figure.colorbar(image, ax=axis, label=field_label)
    figure.tight_layout()

    output_path = data_dir / output_name
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )

    with writer.saving(figure, output_path, dpi=DPI):
        for frame, step in enumerate(steps, start=1):
            time = load_time(data_dir, step)
            image.set_data(load_plane(data_dir, field_name, step))
            title.set_text(f"{field_label}, t = {time:.6f} M")
            writer.grab_frame()
            print(f"\r{output_name}: frame {frame}/{len(steps)}", end="", flush=True)

    print()
    plt.close(figure)


def render_shift_movie(data_dir: Path, steps: list[int]) -> None:
    """Render the three central-slice shift components in one movie."""

    color_limits = shift_color_limits(data_dir, steps)
    component_labels = (r"$\beta^x$", r"$\beta^y$", r"$\beta^z$")
    initial_planes = load_shift_planes(data_dir, steps[0])

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = []

    for component, (axis, component_label) in enumerate(
        zip(axes, component_labels)
    ):
        image = axis.imshow(
            initial_planes[component],
            origin="lower",
            extent=DOMAIN_EXTENT,
            cmap="RdBu_r",
            vmin=-color_limits[component],
            vmax=color_limits[component],
            interpolation="nearest",
        )
        axis.set_title(component_label)
        axis.set_xlabel(r"$x/M$")
        axis.set_ylabel(r"$y/M$")
        axis.set_aspect("equal")
        figure.colorbar(image, ax=axis, label=component_label)
        images.append(image)

    title = figure.suptitle("")
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))

    output_name = "shift.mp4"
    output_path = data_dir / output_name
    writer = FFMpegWriter(
        fps=FPS,
        codec="libx264",
        extra_args=["-pix_fmt", "yuv420p"],
    )

    with writer.saving(figure, output_path, dpi=DPI):
        for frame, step in enumerate(steps, start=1):
            time = load_time(data_dir, step)
            planes = load_shift_planes(data_dir, step)

            for component, image in enumerate(images):
                image.set_data(planes[component])

            title.set_text(f"Shift components, t = {time:.6f} M")
            writer.grab_frame()
            print(f"\r{output_name}: frame {frame}/{len(steps)}", end="", flush=True)

    print()
    plt.close(figure)


def main() -> None:
    """Render lapse, shift, and conformal-factor MP4 movies."""

    steps = snapshot_steps(DATA_DIR)
    print(f"Rendering {len(steps)} snapshots from {DATA_DIR}")

    render_scalar_movie(
        DATA_DIR,
        steps,
        field_name="lapse",
        field_label="Lapse",
        output_name="lapse.mp4",
    )
    render_shift_movie(DATA_DIR, steps)
    render_scalar_movie(
        DATA_DIR,
        steps,
        field_name="conformal_factor",
        field_label="Conformal factor W",
        output_name="conformal_factor.mp4",
    )


if __name__ == "__main__":
    main()
