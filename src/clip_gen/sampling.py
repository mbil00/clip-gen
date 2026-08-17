from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2

from .models import Manifest, Sample, Shot


def sample_timestamps(shot: Shot, count: int) -> tuple[float, ...]:
    if count < 1:
        raise ValueError("sample count must be at least one")
    if shot.duration <= 0:
        return ()

    return tuple(
        round(shot.start + shot.duration * position / (count + 1), 6)
        for position in range(1, count + 1)
    )


def extract_representative_frames(
    video_path: Path,
    manifest: Manifest,
    *,
    output_dir: Path,
    samples_per_shot: int = 3,
    jpeg_quality: int = 90,
) -> Manifest:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")

    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    sampled_shots: list[Shot] = []
    try:
        for shot in manifest.shots:
            samples: list[Sample] = []
            for number, timestamp in enumerate(
                sample_timestamps(shot, samples_per_shot), start=1
            ):
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(
                        f"Could not decode frame at {timestamp:.3f}s from {video_path}"
                    )

                filename = f"{shot.id}-{number:02d}.jpg"
                image_path = output_dir / filename
                written = cv2.imwrite(
                    str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                if not written:
                    raise RuntimeError(f"Could not write image: {image_path}")

                samples.append(Sample(timestamp=timestamp, image=str(image_path)))

            sampled_shots.append(replace(shot, samples=tuple(samples)))
    finally:
        capture.release()

    return manifest.with_shots(tuple(sampled_shots))
