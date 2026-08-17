from __future__ import annotations

from pathlib import Path

from scenedetect import AdaptiveDetector, detect

from .models import Manifest, Shot


def detect_shots(
    video_path: Path,
    *,
    adaptive_threshold: float = 3.0,
    minimum_shot_length: float = 0.5,
    show_progress: bool = True,
) -> Manifest:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if adaptive_threshold <= 0:
        raise ValueError("adaptive_threshold must be greater than zero")
    if minimum_shot_length < 0:
        raise ValueError("minimum_shot_length cannot be negative")

    scene_list = detect(
        str(video_path),
        AdaptiveDetector(
            adaptive_threshold=adaptive_threshold,
            min_scene_len=minimum_shot_length,
        ),
        show_progress=show_progress,
        start_in_scene=True,
    )

    shots = tuple(
        Shot(
            id=f"shot-{index:04d}",
            start=round(start.seconds, 6),
            end=round(end.seconds, 6),
        )
        for index, (start, end) in enumerate(scene_list, start=1)
        if end.seconds > start.seconds
    )

    return Manifest(
        source=str(video_path.resolve()),
        detector={
            "name": "adaptive",
            "adaptive_threshold": adaptive_threshold,
            "minimum_shot_length": minimum_shot_length,
        },
        shots=shots,
    )
