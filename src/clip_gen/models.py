from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    timestamp: float
    image: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Sample:
        return cls(timestamp=float(value["timestamp"]), image=str(value["image"]))


@dataclass(frozen=True)
class Shot:
    id: str
    start: float
    end: float
    samples: tuple[Sample, ...] = field(default_factory=tuple)

    @property
    def duration(self) -> float:
        return self.end - self.start

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Shot:
        return cls(
            id=str(value["id"]),
            start=float(value["start"]),
            end=float(value["end"]),
            samples=tuple(Sample.from_dict(item) for item in value.get("samples", [])),
        )


@dataclass(frozen=True)
class Manifest:
    source: str
    detector: dict[str, Any]
    shots: tuple[Shot, ...]
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Manifest:
        return cls(
            source=str(value["source"]),
            detector=dict(value.get("detector", {})),
            shots=tuple(Shot.from_dict(item) for item in value["shots"]),
            schema_version=int(value.get("schema_version", 1)),
        )

    def with_shots(self, shots: tuple[Shot, ...]) -> Manifest:
        return Manifest(
            source=self.source,
            detector=self.detector,
            shots=shots,
            schema_version=self.schema_version,
        )

    def source_path(self, manifest_path: Path) -> Path:
        source = Path(self.source)
        if source.is_absolute():
            return source
        return (manifest_path.parent / source).resolve()
