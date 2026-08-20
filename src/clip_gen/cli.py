from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .detection import detect_shots
from .manifest import read_manifest, write_manifest
from .sampling import extract_representative_frames
from .splitting import split_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clip-gen",
        description="Detect video shots and prepare representative frames for editorial selection.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect shots and write a manifest.")
    _add_detection_arguments(detect_parser)
    detect_parser.add_argument(
        "--manifest", type=Path, default=Path("shot-manifest.json"), help="Output JSON path."
    )

    prepare_parser = subparsers.add_parser(
        "prepare", help="Detect shots and extract representative frames."
    )
    _add_detection_arguments(prepare_parser)
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clip-gen-output"),
        help="Directory for the manifest and frame samples.",
    )
    prepare_parser.add_argument(
        "--samples-per-shot", type=int, default=3, help="Frames to extract from each shot."
    )

    split_parser = subparsers.add_parser(
        "split", help="Render every manifest shot as a frame-accurate MP4 clip."
    )
    split_parser.add_argument("video", type=Path)
    split_parser.add_argument("--manifest", type=Path, required=True)
    split_parser.add_argument(
        "--output-dir", type=Path, default=Path("clip-gen-output/clips")
    )
    split_parser.add_argument("--overwrite", action="store_true")

    serve_parser = subparsers.add_parser(
        "serve", help="Run the web UI for uploading and preparing videos."
    )
    serve_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("clip-gen-output/web"),
        help="Directory holding one folder per upload.",
    )
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)

    return parser


def _add_detection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=3.0,
        help="Relative change required for a cut; higher values produce fewer cuts.",
    )
    parser.add_argument(
        "--minimum-shot-length",
        type=float,
        default=0.5,
        help="Minimum seconds between detected cuts.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide detection progress.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "detect":
            manifest = _detect_from_args(args)
            write_manifest(manifest, args.manifest)
            print(f"Detected {len(manifest.shots)} shots; wrote {args.manifest}")
            return 0

        if args.command == "prepare":
            manifest_path = args.output_dir / "shot-manifest.json"
            frames_dir = args.output_dir / "frames"
            manifest = _detect_from_args(args)
            manifest = extract_representative_frames(
                args.video,
                manifest,
                output_dir=frames_dir,
                samples_per_shot=args.samples_per_shot,
            )
            write_manifest(manifest, manifest_path)
            print(
                f"Prepared {len(manifest.shots)} shots with frame samples; "
                f"wrote {manifest_path}"
            )
            return 0

        if args.command == "serve":
            try:
                from .web import serve
            except ImportError as error:
                raise RuntimeError(
                    "The web UI needs extra packages. Install them with: uv sync --extra web"
                ) from error
            print(f"clip-gen UI on http://{args.host}:{args.port}")
            serve(output_dir=args.output_dir, host=args.host, port=args.port)
            return 0

        if args.command == "split":
            manifest = read_manifest(args.manifest)
            outputs = split_video(
                args.video,
                manifest,
                output_dir=args.output_dir,
                overwrite=args.overwrite,
            )
            print(f"Rendered {len(outputs)} clips to {args.output_dir}")
            return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        parser.exit(1, f"clip-gen: error: {error}\n")

    return 1


def _detect_from_args(args: argparse.Namespace):
    return detect_shots(
        args.video,
        adaptive_threshold=args.adaptive_threshold,
        minimum_shot_length=args.minimum_shot_length,
        show_progress=not args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
