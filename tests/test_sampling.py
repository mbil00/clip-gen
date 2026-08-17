import pytest

from clip_gen.models import Shot
from clip_gen.sampling import sample_timestamps


def test_samples_are_evenly_distributed_inside_shot() -> None:
    shot = Shot(id="shot-0001", start=10.0, end=14.0)

    assert sample_timestamps(shot, 3) == (11.0, 12.0, 13.0)


def test_one_sample_uses_middle_of_shot() -> None:
    shot = Shot(id="shot-0001", start=3.0, end=8.0)

    assert sample_timestamps(shot, 1) == (5.5,)


def test_sample_count_must_be_positive() -> None:
    shot = Shot(id="shot-0001", start=0.0, end=1.0)

    with pytest.raises(ValueError, match="at least one"):
        sample_timestamps(shot, 0)
