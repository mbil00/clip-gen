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
- [FFmpeg](https://ffmpeg.org/download.html), only for the optional `split` command

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

The vision-analysis stage should evaluate motion as well as appearance. A single frame cannot
show a reveal, gesture, camera move, or developing action, so video-capable analysis is preferred.
When only image input is available, use multiple ordered frames from each candidate.

See [Local model analysis](docs/local-model-analysis.md) for the recommended Apple Silicon model,
installation instructions, output schema, and division of work between the vision model and
deterministic computer-vision checks.

## Development

```bash
uv sync --dev
uv run ruff check .
uv run pytest
```
