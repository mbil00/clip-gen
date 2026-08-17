from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .models import Manifest


def split_video(
    video_path: Path,
    manifest: Manifest,
    *,
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[Path, ...]:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg is required for splitting. Install it and ensure 'ffmpeg' is on PATH."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    for shot in manifest.shots:
        output_path = output_dir / f"{shot.id}.mp4"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}")

        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y" if overwrite else "-n",
            "-ss",
            f"{shot.start:.6f}",
            "-i",
            str(video_path),
            "-t",
            f"{shot.duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-movflags",
            "+faststart",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
        subprocess.run(command, check=True)
        output_paths.append(output_path)

    return tuple(output_paths)
