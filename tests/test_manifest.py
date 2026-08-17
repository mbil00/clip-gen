from pathlib import Path

from clip_gen.manifest import read_manifest, write_manifest
from clip_gen.models import Manifest, Sample, Shot


def test_manifest_round_trip(tmp_path: Path) -> None:
    manifest = Manifest(
        source="source.mp4",
        detector={"name": "adaptive"},
        shots=(
            Shot(
                id="shot-0001",
                start=1.25,
                end=3.75,
                samples=(Sample(timestamp=2.5, image="frames/shot-0001-01.jpg"),),
            ),
        ),
    )
    path = tmp_path / "manifest.json"

    write_manifest(manifest, path)

    assert read_manifest(path) == manifest


def test_relative_source_is_resolved_from_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "work" / "manifest.json"
    manifest = Manifest(source="../source.mp4", detector={}, shots=())

    assert manifest.source_path(manifest_path) == (tmp_path / "source.mp4").resolve()
