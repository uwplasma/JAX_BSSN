"""Run the near-critical axisymmetric Teukolsky-wave pair from the literature."""

import argparse
from pathlib import Path

from teukolsky_wave_common import (
    BRILL_MAX_ITERATIONS,
    BRILL_RELAXATION,
    BRILL_TOLERANCE,
    CENTER_RADIUS,
    CFL,
    FINAL_TIME,
    NUM_RADIAL_POINTS,
    NUM_Z_POINTS,
    RHO_MAX,
    SNAPSHOT_COUNT,
    WIDTH,
    Z_HALF_WIDTH,
    run_teukolsky_wave_common,
)


SUBCRITICAL_AMPLITUDE = 0.00148
SUPERCRITICAL_AMPLITUDE = 0.00164


def run_teukolsky_wave_critical(
    subcritical_amplitude: float = SUBCRITICAL_AMPLITUDE,
    supercritical_amplitude: float = SUPERCRITICAL_AMPLITUDE,
    rho_max: float = RHO_MAX,
    z_half_width: float = Z_HALF_WIDTH,
    num_radial_points: int = NUM_RADIAL_POINTS,
    num_z_points: int = NUM_Z_POINTS,
    cfl: float = CFL,
    final_time: float = FINAL_TIME,
    snapshot_count: int = SNAPSHOT_COUNT,
    width: float = WIDTH,
    center_radius: float = CENTER_RADIUS,
    output_dir: str | Path = "output/teukolsky_critical",
    tolerance: float = BRILL_TOLERANCE,
    max_iterations: int = BRILL_MAX_ITERATIONS,
    relaxation: float = BRILL_RELAXATION,
    show_progress: bool = True,
):
    """Run the highest subcritical and lowest supercritical Teukolsky amplitudes."""

    output_dir = Path(output_dir)
    sub_dir = output_dir / "subcritical"
    sup_dir = output_dir / "supercritical"
    _, sub_info = run_teukolsky_wave_common(
        amplitude=subcritical_amplitude,
        rho_max=rho_max,
        z_half_width=z_half_width,
        num_radial_points=num_radial_points,
        num_z_points=num_z_points,
        cfl=cfl,
        final_time=final_time,
        snapshot_count=snapshot_count,
        width=width,
        center_radius=center_radius,
        output_dir=sub_dir,
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        show_progress=show_progress,
    )
    _, sup_info = run_teukolsky_wave_common(
        amplitude=supercritical_amplitude,
        rho_max=rho_max,
        z_half_width=z_half_width,
        num_radial_points=num_radial_points,
        num_z_points=num_z_points,
        cfl=cfl,
        final_time=final_time,
        snapshot_count=snapshot_count,
        width=width,
        center_radius=center_radius,
        output_dir=sup_dir,
        tolerance=tolerance,
        max_iterations=max_iterations,
        relaxation=relaxation,
        show_progress=show_progress,
    )

    summary_path = output_dir / "critical_summary.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write("# axisymmetric Teukolsky-wave near-critical pair\n")
        summary_file.write("# label amplitude final_time final_fields_finite constraint_path\n")
        summary_file.write(
            f"subcritical {subcritical_amplitude:.16e} {sub_info['final_time']:.16e} "
            f"{int(sub_info['final_fields_finite'])} {sub_info['constraint_path']}\n"
        )
        summary_file.write(
            f"supercritical {supercritical_amplitude:.16e} {sup_info['final_time']:.16e} "
            f"{int(sup_info['final_fields_finite'])} {sup_info['constraint_path']}\n"
        )

    return {
        "subcritical": sub_info,
        "supercritical": sup_info,
        "summary_path": summary_path,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subcritical-amplitude", type=float, default=SUBCRITICAL_AMPLITUDE)
    parser.add_argument("--supercritical-amplitude", type=float, default=SUPERCRITICAL_AMPLITUDE)
    parser.add_argument("--rho-max", type=float, default=RHO_MAX)
    parser.add_argument("--z-half-width", type=float, default=Z_HALF_WIDTH)
    parser.add_argument("--num-rho", type=int, default=NUM_RADIAL_POINTS)
    parser.add_argument("--num-z", type=int, default=NUM_Z_POINTS)
    parser.add_argument("--cfl", type=float, default=CFL)
    parser.add_argument("--final-time", type=float, default=FINAL_TIME)
    parser.add_argument("--snapshots", type=int, default=SNAPSHOT_COUNT)
    parser.add_argument("--width", type=float, default=WIDTH)
    parser.add_argument("--center-radius", type=float, default=CENTER_RADIUS)
    parser.add_argument("--output-dir", type=Path, default=Path("output/teukolsky_critical"))
    parser.add_argument("--tolerance", type=float, default=BRILL_TOLERANCE)
    parser.add_argument("--max-iterations", type=int, default=BRILL_MAX_ITERATIONS)
    parser.add_argument("--relaxation", type=float, default=BRILL_RELAXATION)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    run_teukolsky_wave_critical(
        subcritical_amplitude=args.subcritical_amplitude,
        supercritical_amplitude=args.supercritical_amplitude,
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