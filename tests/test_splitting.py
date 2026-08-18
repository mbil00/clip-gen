import subprocess
from pathlib import Path

import pytest

from clip_gen.models import Manifest, Shot
from clip_gen.splitting import _build_ffmpeg_command, split_video


def test_split_requires_ffmpeg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    manifest = Manifest(source=str(video), detector={}, shots=())
    monkeypatch.setattr("clip_gen.splitting.shutil.which", lambda _: None)

    with pytest.raises(RuntimeError, match="FFmpeg is required"):
        split_video(video, manifest, output_dir=tmp_path / "clips")


def test_ffmpeg_command_pads_odd_dimensions(tmp_path: Path) -> None:
    command = _build_ffmpeg_command(
        ffmpeg="ffmpeg",
        video_path=tmp_path / "source.mov",
        shot_start=1.25,
        shot_duration=2.5,
        output_path=tmp_path / "shot.mp4",
    )

    filter_index = command.index("-vf")
    assert command[filter_index + 1] == "pad=ceil(iw/2)*2:ceil(ih/2)*2"


def test_failed_render_removes_temporary_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mov"
    video.touch()
    manifest = Manifest(
        source=str(video),
        detector={},
        shots=(Shot(id="shot-0001", start=0.0, end=1.0),),
    )
    output_dir = tmp_path / "clips"
    monkeypatch.setattr("clip_gen.splitting.shutil.which", lambda _: "/usr/bin/ffmpeg")

    def fail_render(command: list[str], *, check: bool) -> None:
        Path(command[-1]).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr("clip_gen.splitting.subprocess.run", fail_render)

    with pytest.raises(RuntimeError, match="shot-0001"):
        split_video(video, manifest, output_dir=output_dir)

    assert list(output_dir.iterdir()) == []
