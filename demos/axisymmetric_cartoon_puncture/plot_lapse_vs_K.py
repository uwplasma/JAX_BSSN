"""Plot final lapse versus trace K along an axisymmetric x cut."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io


DEFAULT_INPUT = (
    Path(__file__).resolve().parent
    / "output"
    / "axisymmetric_cartoon_puncture.h5"
)
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "lapse_vs_K_x_cut.png"


def _scalar_component(mesh):
    return mesh[next(iter(mesh))]


def _coordinate(mesh, component, axis: int) -> np.ndarray:
    return (
        float(mesh.grid_global_offset[axis])
        + (np.arange(component.shape[axis]) + float(component.position[axis]))
        * float(mesh.grid_spacing[axis])
    )


def load_final_x_cut(
    path: Path,
    z_cut: float = 0.0,
) -> tuple[int, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Load lapse and K along x at the saved z plane nearest ``z_cut``."""

    if not path.is_file():
        raise FileNotFoundError(f"axisymmetric output not found: {path}")

    series = io.Series(str(path), io.Access.read_only)
    steps = list(series.iterations)
    if not steps:
        series.close()
        raise ValueError(f"no iterations found in {path}")

    step = steps[-1]
    iteration = series.iterations[step]
    lapse_mesh = iteration.meshes["lapse"]
    K_mesh = iteration.meshes["K"]
    lapse_component = _scalar_component(lapse_mesh)
    K_component = _scalar_component(K_mesh)

    x = _coordinate(lapse_mesh, lapse_component, 0)
    z = _coordinate(lapse_mesh, lapse_component, 2)
    z_index = int(np.argmin(np.abs(z - z_cut)))
    actual_z = float(z[z_index])

    extent = [0, 0, z_index]
    shape = [lapse_component.shape[0], 1, 1]
    lapse_pending = lapse_component.load_chunk(extent, shape)
    K_pending = K_component.load_chunk(extent, shape)
    time = float(iteration.time)
    series.flush()

    lapse = np.asarray(lapse_pending)[:, 0, 0].copy()
    trace_K = np.asarray(K_pending)[:, 0, 0].copy()
    iteration.close()
    series.close()

    finite = np.isfinite(x) & np.isfinite(trace_K) & np.isfinite(lapse)
    if not np.all(finite):
        raise ValueError(
            f"final iteration {step} contains {np.count_nonzero(~finite)} "
            "non-finite samples on the requested x cut"
        )
    return step, time, actual_z, x, trace_K, lapse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--z-cut",
        type=float,
        default=0.0,
        help="z coordinate of the requested x cut (nearest saved plane is used)",
    )
    parser.add_argument(
        "--full-x",
        action="store_true",
        help="use the full signed x cut instead of the default x >= 0 half",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step, time, actual_z, x, trace_K, lapse = load_final_x_cut(
        args.input, args.z_cut
    )
    if not args.full_x:
        view = x >= 0.0
        x, trace_K, lapse = x[view], trace_K[view], lapse[view]

    figure, axis = plt.subplots(figsize=(10.0, 8.0))
    axis.plot(trace_K, lapse, linewidth=2.0)
    axis.set_xlabel(r"$K$", fontsize=18)
    axis.set_ylabel(r"$\alpha$", fontsize=18)
    axis.tick_params(labelsize=15)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(
        f"Wrote {args.output} from step {step}, t={time:.16g}, "
        f"z={actual_z:.16g}, using {x.size} x-cut samples"
    )


if __name__ == "__main__":
    main()
