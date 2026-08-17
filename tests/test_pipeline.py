from pathlib import Path

import cv2
import numpy as np

from clip_gen.detection import detect_shots
from clip_gen.sampling import extract_representative_frames


def test_detects_cut_and_extracts_samples(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.avi"
    writer = cv2.VideoWriter(
        str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 64)
    )
    assert writer.isOpened()
    for value in (0, 255):
        for _ in range(10):
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()

    manifest = detect_shots(video, show_progress=False)
    prepared = extract_representative_frames(
        video,
        manifest,
        output_dir=tmp_path / "frames",
        samples_per_shot=3,
    )

    assert len(prepared.shots) == 2
    assert all(len(shot.samples) == 3 for shot in prepared.shots)
    assert all(Path(sample.image).is_file() for shot in prepared.shots for sample in shot.samples)
