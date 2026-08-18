from __future__ import annotations

import shutil
import subprocess
import tempfile
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

        with tempfile.NamedTemporaryFile(
            prefix=f".{shot.id}-", suffix=".mp4", dir=output_dir, delete=False
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)

        command = _build_ffmpeg_command(
            ffmpeg=ffmpeg,
            video_path=video_path,
            shot_start=shot.start,
            shot_duration=shot.duration,
            output_path=temporary_path,
        )
        try:
            subprocess.run(command, check=True)
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"Output already exists: {output_path}")
            temporary_path.replace(output_path)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"FFmpeg failed while rendering {shot.id}") from error
        finally:
            temporary_path.unlink(missing_ok=True)

        output_paths.append(output_path)

    return tuple(output_paths)


def _build_ffmpeg_command(
    *,
    ffmpeg: str,
    video_path: Path,
    shot_start: float,
    shot_duration: float,
    output_path: Path,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{shot_start:.6f}",
        "-i",
        str(video_path),
        "-t",
        f"{shot_duration:.6f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
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
