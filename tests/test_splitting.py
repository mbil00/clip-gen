from pathlib import Path

import pytest

from clip_gen.models import Manifest
from clip_gen.splitting import split_video


def test_split_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    manifest = Manifest(source=str(video), detector={}, shots=())
    monkeypatch.setattr("clip_gen.splitting.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="FFmpeg is required"):
        split_video(video, manifest, output_dir=tmp_path / "clips")
