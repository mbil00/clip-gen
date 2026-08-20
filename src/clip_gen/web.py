from __future__ import annotations

import json
import shutil
import threading
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from .describe import ModelConfig, describe_shot, load_env
from .detection import detect_shots
from .manifest import write_manifest
from .models import Manifest, Sample, Shot
from .sampling import extract_representative_frames
from .splitting import split_video

INDEX_PATH = Path(__file__).with_name("index.html")
MANIFEST_NAME = "shot-manifest.json"
DESCRIPTIONS_NAME = "descriptions.json"


@dataclass
class Job:
    """A single upload-and-prepare run. Kept in memory; output lives on disk."""

    id: str
    filename: str
    directory: Path
    settings: dict[str, Any]
    status: str = "running"
    stage: str = "queued"
    error: str | None = None
    manifest: dict[str, Any] | None = None
    descriptions: dict[str, dict[str, Any]] = field(default_factory=dict)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def to_dict(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "filename": self.filename,
                "settings": self.settings,
                "status": self.status,
                "stage": self.stage,
                "error": self.error,
                "manifest": self.manifest,
                "descriptions": dict(self.descriptions),
            }

    def update(self, **changes: Any) -> None:
        with self.lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def set_description(self, shot_id: str, **fields: Any) -> None:
        with self.lock:
            self.descriptions[shot_id] = {"shot_id": shot_id, **fields}
            snapshot = dict(self.descriptions)
        path = self.directory / DESCRIPTIONS_NAME
        path.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")

    def shots(self) -> list[Shot]:
        with self.lock:
            manifest = self.manifest
        if manifest is None:
            return []
        return [Shot.from_dict(shot) for shot in manifest["shots"]]


def create_app(output_dir: Path) -> FastAPI:
    load_env(Path.cwd() / ".env")
    app = FastAPI(title="clip-gen")
    jobs: dict[str, Job] = {}
    root = output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    def get_job(job_id: str) -> Job:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job")
        return job

    @app.get("/", response_class=HTMLResponse)
    def index() -> FileResponse:
        return FileResponse(INDEX_PATH)

    @app.post("/api/jobs")
    async def create_job(
        video: Annotated[UploadFile, File()],
        adaptive_threshold: Annotated[float, Form()] = 3.0,
        minimum_shot_length: Annotated[float, Form()] = 0.5,
        samples_per_shot: Annotated[int, Form()] = 3,
        render_clips: Annotated[bool, Form()] = False,
    ) -> dict[str, Any]:
        job_id = uuid.uuid4().hex[:8]
        directory = root / job_id
        directory.mkdir(parents=True)

        suffix = Path(video.filename or "source.mp4").suffix or ".mp4"
        video_path = directory / f"source{suffix}"
        with video_path.open("wb") as handle:
            shutil.copyfileobj(video.file, handle)
        await video.close()

        job = Job(
            id=job_id,
            filename=video.filename or video_path.name,
            directory=directory,
            settings={
                "adaptive_threshold": adaptive_threshold,
                "minimum_shot_length": minimum_shot_length,
                "samples_per_shot": samples_per_shot,
                "render_clips": render_clips,
            },
        )
        jobs[job_id] = job

        thread = threading.Thread(target=_run_job, args=(job, video_path), daemon=True)
        thread.start()
        return job.to_dict()

    @app.get("/api/jobs/{job_id}")
    def read_job(job_id: str) -> dict[str, Any]:
        return get_job(job_id).to_dict()

    @app.post("/api/jobs/{job_id}/describe")
    def describe(job_id: str, shot_ids: Annotated[list[str], Body(embed=True)]) -> dict[str, Any]:
        job = get_job(job_id)
        known = {shot.id: shot for shot in job.shots()}
        unknown = [shot_id for shot_id in shot_ids if shot_id not in known]
        if unknown:
            raise HTTPException(status_code=400, detail=f"Unknown shots: {', '.join(unknown)}")

        for shot_id in shot_ids:
            job.set_description(shot_id, status="pending", text=None, error=None, model=None)
        selected = [known[shot_id] for shot_id in shot_ids]
        thread = threading.Thread(target=_describe_shots, args=(job, selected), daemon=True)
        thread.start()
        return job.to_dict()

    @app.get("/api/model")
    def read_model() -> dict[str, Any]:
        return ModelConfig.from_env().check()

    @app.get("/jobs/{job_id}/files/{relative_path:path}")
    def read_file(job_id: str, relative_path: str) -> FileResponse:
        job = get_job(job_id)
        target = (job.directory / relative_path).resolve()
        if job.directory.resolve() not in target.parents or not target.is_file():
            raise HTTPException(status_code=404, detail="Unknown file")
        return FileResponse(target)

    return app


def _run_job(job: Job, video_path: Path) -> None:
    settings = job.settings
    try:
        job.update(stage="Detecting shots")
        manifest = detect_shots(
            video_path,
            adaptive_threshold=settings["adaptive_threshold"],
            minimum_shot_length=settings["minimum_shot_length"],
            show_progress=False,
        )

        job.update(stage=f"Extracting frames from {len(manifest.shots)} shots")
        manifest = extract_representative_frames(
            video_path,
            manifest,
            output_dir=job.directory / "frames",
            samples_per_shot=settings["samples_per_shot"],
        )
        manifest = _relative_to_job(manifest, job.directory, video_path)

        if settings["render_clips"]:
            job.update(stage=f"Rendering {len(manifest.shots)} clips")
            split_video(
                video_path,
                manifest,
                output_dir=job.directory / "clips",
                overwrite=True,
            )

        write_manifest(manifest, job.directory / MANIFEST_NAME)
        job.update(stage="Done", status="done", manifest=manifest.to_dict())
    except Exception as error:  # noqa: BLE001 - surfaced to the browser as-is
        job.update(stage="Failed", status="error", error=str(error))


def _describe_shots(job: Job, shots: list[Shot]) -> None:
    """Describe the selected shots one at a time so the model server is never overloaded."""
    config = ModelConfig.from_env()
    for shot in shots:
        job.set_description(shot.id, status="running", text=None, error=None, model=None)
        try:
            text, model = describe_shot(shot, job.directory, config)
            job.set_description(shot.id, status="done", text=text, error=None, model=model)
        except Exception as error:  # noqa: BLE001 - surfaced to the browser as-is
            job.set_description(shot.id, status="error", text=None, error=str(error), model=None)


def _relative_to_job(manifest: Manifest, directory: Path, video_path: Path) -> Manifest:
    """Rewrite absolute paths as job-relative ones so the manifest is portable and servable."""

    def relative(path: str) -> str:
        return str(Path(path).resolve().relative_to(directory.resolve()))

    shots = tuple(
        replace(
            shot,
            samples=tuple(
                Sample(timestamp=sample.timestamp, image=relative(sample.image))
                for sample in shot.samples
            ),
        )
        for shot in manifest.shots
    )
    return Manifest(
        source=relative(str(video_path)),
        detector=manifest.detector,
        shots=shots,
        schema_version=manifest.schema_version,
    )


def serve(*, output_dir: Path, host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(output_dir), host=host, port=port, log_level="info")
