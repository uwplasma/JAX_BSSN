"""Plot the final spherical-Cartoon lapse as a function of trace K."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import openpmd_api as io


DEFAULT_INPUT = Path(__file__).resolve().parent / "output" / "cartoon_puncture.h5"
DEFAULT_OUTPUT = DEFAULT_INPUT.parent / "lapse_vs_K.png"


def _scalar_component(mesh):
    return mesh[next(iter(mesh))]


def load_final_positive_axis(path: Path) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Load K and lapse on the positive radial axis at the final iteration."""

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
    lapse_pending = lapse_component.load_chunk()
    K_pending = K_component.load_chunk()

    time = float(iteration.time)
    dx = float(lapse_mesh.grid_spacing[0])
    offset = float(lapse_mesh.grid_global_offset[0])
    position = float(lapse_component.position[0])
    series.flush()

    lapse = np.array(lapse_pending[:, 0, 0], copy=True)
    trace_K = np.array(K_pending[:, 0, 0], copy=True)
    radius = offset + (np.arange(lapse.size) + position) * dx
    positive = radius >= 0.0

    iteration.close()
    series.close()

    lapse = lapse[positive]
    trace_K = trace_K[positive]
    finite = np.isfinite(lapse) & np.isfinite(trace_K)
    if not np.all(finite):
        raise ValueError(
            f"final iteration {step} contains {np.count_nonzero(~finite)} "
            "non-finite positive-axis K/lapse samples"
        )
    return step, time, trace_K, lapse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step, time, trace_K, lapse = load_final_positive_axis(args.input)

    figure, axis = plt.subplots(figsize=(10.0, 8.0))
    axis.plot(trace_K, lapse, linewidth=2.0)
    axis.set_xlabel("K", fontsize=18)
    axis.set_ylabel(r"$\alpha$", fontsize=18)
    axis.tick_params(labelsize=15)
    figure.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(
        f"Wrote {args.output} from step {step}, t={time:.16g}, "
        f"using {trace_K.size} positive-axis samples"
    )


if __name__ == "__main__":
    main()
