# Local model analysis

The recommended local analyzer is
[`mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit`](https://huggingface.co/mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit),
run with [MLX-VLM](https://github.com/Blaizzy/mlx-vlm) on Apple Silicon.

Qwen3-VL is suitable for this pipeline because it can reason over several ordered images, describe
motion and visual progression, follow an editorial brief, and identify a useful subrange within a
candidate. The 4-bit MLX checkpoint keeps local inference practical while retaining the larger
model's capabilities.

Use the instruct checkpoint for extraction and scoring. A thinking checkpoint adds latency and is
not necessary for the per-candidate analysis pass.

## Install the runtime

Install MLX-VLM as a UV-managed tool so its large inference dependency set stays separate from
`clip-gen`:

```bash
uv tool install mlx-vlm
```

Download the model into a predictable local directory:

```bash
mkdir -p ~/models
uvx --from huggingface_hub hf download \
  mlx-community/Qwen3-VL-30B-A3B-Instruct-4bit \
  --local-dir ~/models/Qwen3-VL-30B-A3B-Instruct-4bit
```

Model weights should not be placed in this repository or committed to Git. The Hugging Face cache
inside the model directory supports interrupted-download resumption.

For faster iteration on machines with less unified memory, use
`mlx-community/Qwen3-VL-8B-Instruct-4bit` with the same pipeline and prompt.

## Analyze ordered frames

Start with several ordered frames rather than sending each image independently. Three samples are
enough for a basic shot; use five to eight for a longer candidate or one containing fast action.

```bash
mlx_vlm.generate \
  --model ~/models/Qwen3-VL-30B-A3B-Instruct-4bit \
  --image \
    clip-gen-output/frames/shot-0001-01.jpg \
    clip-gen-output/frames/shot-0001-02.jpg \
    clip-gen-output/frames/shot-0001-03.jpg \
  --prompt "$(cat prompt.txt)" \
  --max-tokens 600 \
  --temperature 0
```

The prompt must state that the images are chronological frames from one candidate and include their
source timestamps. Without that context, a model may interpret them as unrelated photographs.

```text
You are analyzing one candidate for a short-form video.

The attached images are chronological frames from the same candidate:
- image 1: 18.40 seconds
- image 2: 19.20 seconds
- image 3: 20.00 seconds

Editorial goal:
Create an energetic vertical travel montage. Prefer human activity, motion,
scenic reveals, and clear visual progression.

Describe only what is supported by the frames. Return one JSON object matching
the requested schema. Scores range from 0.0 to 1.0. The preferred range uses
offsets relative to the start of this candidate.
```

For the initial integration, ordered images are preferable to direct video input: the pipeline
already controls frame timestamps, visual-token use is predictable, and inputs are easy to cache
and reproduce. Direct video can be added later for candidates where denser temporal information is
valuable.

## Analysis record

Validate model output before writing it to the timeline. A useful record contains observations,
problems, scores, and the best subrange:

```json
{
  "summary": "A cyclist emerges from a forest and rides toward the camera.",
  "subjects": ["cyclist", "bicycle", "forest"],
  "action": "The cyclist approaches rapidly.",
  "camera_motion": "Mostly static with slight handheld movement.",
  "visual_progression": "The distant subject enters and becomes the focal point.",
  "preferred_range": {
    "start_offset": 0.8,
    "end_offset": 3.4,
    "reason": "The cyclist is clearly visible and moving toward the viewer."
  },
  "scores": {
    "goal_relevance": 0.84,
    "visual_interest": 0.88,
    "energy": 0.79,
    "vertical_crop_suitability": 0.73
  },
  "problems": ["minor camera shake"],
  "recommended": true
}
```

Use a Pydantic model for the schema. Reject and retry responses that contain invalid JSON, missing
fields, out-of-range scores, or preferred timestamps outside the candidate.

Keep the original observations alongside the scores. Scores from separate model calls are not
perfectly calibrated, so later selection should be able to revisit the evidence rather than depend
only on a single number.

## Keep deterministic checks outside the model

The vision-language model should handle meaning and editorial judgment. Use ordinary computer
vision for properties that can be measured directly:

| Signal | Suggested method |
| --- | --- |
| Blur | Variance of the Laplacian |
| Exposure and clipping | Luminance histogram |
| Motion intensity | Optical flow |
| Near-duplicate candidates | SigLIP or DINOv2 embeddings |
| Shot boundaries | PySceneDetect |
| Exact duration limits | Timeline code |

Combining measured signals with model observations makes filtering cheaper, more stable, and easier
to debug. Run inexpensive checks first and avoid model inference for unusable candidates.

## Processing strategy

Load the model once and process candidates sequentially or in small batches. Repeatedly loading a
30B checkpoint will dominate the runtime. Cache every result using a key derived from:

- the source video fingerprint;
- candidate start and end timestamps;
- sampled frame timestamps;
- model identifier and quantization;
- prompt and schema version.

Set temperature to zero, but do not assume that generation is perfectly deterministic across
runtime or model upgrades. Record the model and runtime versions in each analysis artifact.

After all candidates have structured records, a separate selection pass can build a varied sequence
under the exact duration budget. The selection pass should operate on compact records rather than
loading all source frames again.
