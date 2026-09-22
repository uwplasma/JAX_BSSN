"""Render MP4 movies for an axisymmetric Teukolsky-wave output file."""

from __future__ import annotations

import argparse
from pathlib import Path

from make_movies import DPI, FPS, _available_specs, render_movie, require_ffmpeg


DEFAULT_INPUT = (
    Path(__file__).resolve().parent / "output" / "teukolsky_wave_common.h5"
)
DEFAULT_MOVIE_DIR = DEFAULT_INPUT.parent / "teukolsky_movies"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MOVIE_DIR)
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
    if not args.input.is_file():
        raise FileNotFoundError(f"Teukolsky-wave output not found: {args.input}")
    require_ffmpeg()

    steps, x, z, specs, missing = _available_specs(args.input)
    if args.max_frames is not None:
        if args.max_frames < 1:
            raise ValueError("--max-frames must be positive")
        steps = steps[: args.max_frames]

    print(f"Rendering {len(steps)} Teukolsky-wave snapshots from {args.input}")
    if missing:
        print("Skipping absent fields: " + ", ".join(spec.title for spec in missing))

    for spec in specs:
        print(f"Scanning fixed color limits for {spec.title}")
        output = render_movie(
            args.input,
            args.output_dir,
            spec,
            steps,
            x,
            z,
            args.fps,
            args.dpi,
        )
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()