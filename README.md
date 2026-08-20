# clip-gen

`clip-gen` prepares long videos for short-form editing. It detects camera cuts, records each
shot as a timestamp range, and extracts representative frames for visual analysis. Physical
video clips are rendered only when they are actually needed.

The project deliberately separates editorial metadata from media rendering. This avoids
creating large collections of intermediate files, prevents repeated quality loss, and keeps
the original video as the single source of truth.

## Requirements

- Python 3.11 or newer
- [UV](https://docs.astral.sh/uv/)
- [FFmpeg](https://ffmpeg.org/download.html), only for the optional `split` command and for
  rendering clips from the web UI

On macOS, FFmpeg can be installed with:

```bash
brew install ffmpeg
```

## Setup

```bash
git clone https://github.com/mbil00/clip-gen.git
cd clip-gen
uv sync
```

Run the CLI through UV:

```bash
uv run clip-gen --help
```

## Web UI

The quickest way to try a video is the browser UI. It uploads a file, runs detection and frame
extraction with the settings you choose, and shows the resulting manifest with every timestamp
and captured frame.

```bash
uv sync --extra web
uv run clip-gen serve
```

Then open <http://127.0.0.1:8000>. Each run gets its own folder under `clip-gen-output/web/`:

```text
clip-gen-output/web/3533a019/
├── source.mp4
├── shot-manifest.json
├── frames/
└── clips/          # only when "render clip MP4s" is checked
```

Paths inside these manifests are relative to the job folder, so the whole folder can be moved or
handed to another tool as a unit. The UI is a local single-user POC: jobs run in a background
thread, there is no authentication, and the job list is not kept across restarts.

### Describe scenes with a vision model

Once a run finishes, each shot has a checkbox. Select the interesting ones, press **Describe**, and
every selected shot's frames are sent to a vision-language model as one ordered set. Descriptions
appear on the page as they complete and are saved to `descriptions.json` in the job folder.

The model is any OpenAI-compatible endpoint, configured in `.env`:

```bash
cp .env.example .env
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `CLIP_GEN_MODEL_URL` | `http://127.0.0.1:8080/v1` | Base URL of the endpoint. |
| `CLIP_GEN_MODEL` | *(empty)* | Model name. When empty, clip-gen asks the endpoint which model is loaded and uses that. |
| `CLIP_GEN_API_KEY` | *(empty)* | Sent as a bearer token when set. Local servers do not need it. |
| `CLIP_GEN_MAX_TOKENS` | `600` | Upper bound on description length. |

Real environment variables take precedence over `.env`, so a single run can be pointed elsewhere:

```bash
CLIP_GEN_MODEL_URL=http://127.0.0.1:11434/v1 CLIP_GEN_MODEL=qwen2.5vl uv run clip-gen serve
```

For local analysis on Apple Silicon, install the runtime and model as described in
[Local model analysis](docs/local-model-analysis.md), then start the server before `clip-gen serve`:

```bash
mlx_vlm.server --model ~/models/Qwen3-VL-30B-A3B-Instruct-4bit --port 8080
```

Keep the model server running across jobs. Loading the checkpoint is the expensive part; once it is
resident, a three-frame shot takes a few seconds. The toolbar above the shot list shows which model
the endpoint reports, or a warning when it cannot be reached.

Note that the model must be vision-capable. The text-only `Qwen3-30B-A3B` cannot read frames; the
`Qwen3-VL-30B-A3B-Instruct-4bit` checkpoint above is its vision sibling. Shots are described one at
a time, and the wording of the request lives in `PROMPT` in `src/clip_gen/describe.py` — change it
there to ask for a different kind of description.

Results accumulate in `descriptions.json` next to the manifest, keyed by shot:

```json
{
  "shot-0005": {
    "shot_id": "shot-0005",
    "status": "done",
    "text": "The shot displays a static title card with the text \"THREE RODENTS\"...",
    "error": null,
    "model": "/Users/me/models/Qwen3-VL-30B-A3B-Instruct-4bit"
  }
}
```

Descriptions are free text at this stage. The structured analysis record in
[Local model analysis](docs/local-model-analysis.md) is the intended next step, and is what later
selection passes should consume.

### HTTP endpoints

The UI is a thin client over a small JSON API, which is also convenient for scripting:

| Endpoint | Purpose |
| --- | --- |
| `POST /api/jobs` | Multipart upload plus settings. Returns a job id and starts the run. |
| `GET /api/jobs/{id}` | Status, stage, manifest, and any descriptions. Poll this. |
| `POST /api/jobs/{id}/describe` | Body `{"shot_ids": ["shot-0002"]}`. Queues those shots. |
| `GET /api/model` | Reports the configured endpoint and whether it answers. |
| `GET /jobs/{id}/files/{path}` | Serves frames, clips, and the manifest from the job folder. |

## Prepare a video

The normal workflow detects shots and saves three representative JPEGs from each shot:

```bash
uv run clip-gen prepare input.mp4 --output-dir clip-gen-output
```

The result is:

```text
clip-gen-output/
├── shot-manifest.json
└── frames/
    ├── shot-0001-01.jpg
    ├── shot-0001-02.jpg
    ├── shot-0001-03.jpg
    └── ...
```

Each shot in the manifest contains its exact source range and its frame samples:

```json
{
  "id": "shot-0001",
  "start": 0.0,
  "end": 4.36,
  "samples": [
    {
      "timestamp": 1.09,
      "image": "clip-gen-output/frames/shot-0001-01.jpg"
    }
  ]
}
```

To detect timestamps without extracting images:

```bash
uv run clip-gen detect input.mp4 --manifest shot-manifest.json
```

## Render individual clips

Splitting is optional because most downstream processing can operate on timestamp ranges and
the original file. When separate MP4s are required:

```bash
uv run clip-gen split input.mp4 \
  --manifest clip-gen-output/shot-manifest.json \
  --output-dir clip-gen-output/clips
```

The command re-encodes each range with H.264 and AAC. Re-encoding is slower than stream copying,
but it produces accurate cuts even when a boundary does not coincide with a source keyframe.

## How detection works

`clip-gen` uses PySceneDetect's adaptive content detector. It measures color and brightness
changes between adjacent frames, then evaluates each change relative to a rolling local average.
This makes it less likely that fast camera motion will be mistaken for an edit.

The defaults are intended as a useful baseline:

```text
adaptive threshold:   3.0
minimum shot length:  0.5 seconds
samples per shot:     3
```

Videos vary, so detection should be reviewed before expensive model analysis. If real cuts are
missing, lower the adaptive threshold. If camera movement creates false cuts, raise it:

```bash
uv run clip-gen prepare input.mp4 --adaptive-threshold 2.5
uv run clip-gen prepare input.mp4 --adaptive-threshold 4.0
```

A cut detector finds edits that already exist in the source. It cannot subdivide a long,
continuous camera take according to its meaning. Long shots should be divided into overlapping
candidate windows later, while preserving their original shot identifier.

Fades and dissolves are also different from hard cuts. Material that uses them heavily may need
a second detection pass with PySceneDetect's threshold detector or purpose-built transition
detection.

## Production pipeline

A reliable short-form generation pipeline uses the manifest as its timeline:

1. Detect hard cuts and review the boundary statistics on representative source material.
2. Divide long shots into short, overlapping candidate windows.
3. Analyze several frames, or the short video window itself, with a vision-capable model.
4. Store structured descriptions, quality signals, relevance scores, and preferred subranges.
5. Filter unusable candidates and remove near-duplicates with visual embeddings.
6. Select a varied sequence under an exact duration constraint. Use deterministic code to enforce
   timing even if a language model proposes the editorial order.
7. Move cut points slightly to nearby music beats without changing the underlying selection.
8. Render once from the original media, then review the completed short for repetition, awkward
   transitions, crop failures, and encoding errors.

Steps 1 and 3 have a working first pass in the [web UI](#web-ui): shot detection with frame
extraction, and free-text descriptions of selected shots from a local vision model. The remaining
steps, including the structured records that selection depends on, are not implemented yet.

The vision-analysis stage should evaluate motion as well as appearance. A single frame cannot
show a reveal, gesture, camera move, or developing action, so video-capable analysis is preferred.
When only image input is available, use multiple ordered frames from each candidate.

See [Local model analysis](docs/local-model-analysis.md) for the recommended Apple Silicon model,
installation instructions, output schema, and division of work between the vision model and
deterministic computer-vision checks.

## Development

```bash
uv sync --dev --extra web
uv run ruff check .
uv run pytest
```

The `web` extra installs FastAPI, Uvicorn, and python-multipart. It is optional so that the CLI and
library stay usable without them; `clip-gen serve` explains what to install when they are missing.
The UI is a single static file, `src/clip_gen/index.html`, with no build step.
