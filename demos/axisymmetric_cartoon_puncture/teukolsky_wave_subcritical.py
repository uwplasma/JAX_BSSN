"""Evolve a subcritical axisymmetric Teukolsky wave that disperses."""

import argparse
from pathlib import Path

from teukolsky_wave_common import (
    CENTER_RADIUS,
    CFL,
    ELLIPTIC_MAX_ITERATIONS,
    ELLIPTIC_RELAXATION,
    ELLIPTIC_TOLERANCE,
    FINAL_TIME,
    NUM_RADIAL_POINTS,
    NUM_Z_POINTS,
    RHO_MAX,
    SNAPSHOT_COUNT,
    WIDTH,
    Z_HALF_WIDTH,
    run_teukolsky_wave_case,
)


AMPLITUDE = 0.00148


def run_teukolsky_wave_subcritical(
    amplitude: float = AMPLITUDE,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    width: float = WIDTH,
    center_radius: float = CENTER_RADIUS,
    output_dir: str | Path = "output",
    tolerance: float = ELLIPTIC_TOLERANCE,
    max_iterations: int = ELLIPTIC_MAX_ITERATIONS,
    relaxation: float = ELLIPTIC_RELAXATION,
    show_progress: bool = True,
):
    """Run the subcritical Teukolsky wave that remains dispersive."""

    return run_teukolsky_wave_case(
        amplitude=amplitude,
        rho_max=rho_max,
        z_half_width=z_half_width,
        num_radial_points=num_radial_points,
        num_z_points=num_z_points,
        cfl=cfl,
        final_time=final_time,
        snapshot_count=snapshot_count,
        width=width,
        center_radius=center_radius,
        output_dir=output_dir,
        output_stem="teukolsky_wave_subcritical",
        progress_label="Evolving subcritical Teukolsky wave",
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        show_progress=show_progress,
    )


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amplitude", type=float, default=AMPLITUDE)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--width", type=float, default=WIDTH)
    parser.add_argument("--center-radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--tolerance", type=float, default=ELLIPTIC_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=ELLIPTIC_MAX_ITERATIONS)
    parser.add_argument("--relaxation", type=float, default=ELLIPTIC_RELAXATION)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_teukolsky_wave_subcritical(
        amplitude=args.amplitude,
        rho_max=args.rho_max,
        z_half_width=args.z_half_width,
        num_radial_points=args.num_rho,
        num_z_points=args.num_z,
        cfl=args.cfl,
        final_time=args.final_time,
        snapshot_count=args.snapshots,
        width=args.width,
        center_radius=args.center_radius,
        output_dir=args.output_dir,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
        relaxation=args.relaxation,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()