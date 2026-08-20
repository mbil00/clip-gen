from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .models import Shot

DEFAULT_BASE_URL = "http://127.0.0.1:8080/v1"
DEFAULT_MAX_TOKENS = 600
REQUEST_TIMEOUT = 600

PROMPT = """You are describing one shot from a video, for an editor choosing material for a \
short-form edit.

The attached images are chronological frames from this single shot:
{frame_lines}

Describe what happens across the shot in 2-4 sentences: the subjects, the action and how it \
progresses, the setting, and any camera movement. Describe only what the frames support, and do \
not guess at the title or source of the footage. Finish with one short sentence on how usable the \
shot looks for a highlight reel."""


def load_env(path: Path) -> None:
    """Read a minimal KEY=value .env file. Existing environment variables win."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = ""
    api_key: str = ""
    max_tokens: int = DEFAULT_MAX_TOKENS

    @classmethod
    def from_env(cls) -> ModelConfig:
        return cls(
            base_url=os.environ.get("CLIP_GEN_MODEL_URL", DEFAULT_BASE_URL).rstrip("/"),
            model=os.environ.get("CLIP_GEN_MODEL", ""),
            api_key=os.environ.get("CLIP_GEN_API_KEY", ""),
            max_tokens=int(os.environ.get("CLIP_GEN_MAX_TOKENS", DEFAULT_MAX_TOKENS)),
        )

    def resolve_model(self) -> str:
        """Fall back to whatever model the endpoint already has loaded."""
        if self.model:
            return self.model
        listing = self._get("/models")
        models = listing.get("data") or []
        if not models:
            raise RuntimeError(f"No model available at {self.base_url}")
        return str(models[0]["id"])

    def check(self) -> dict[str, object]:
        try:
            return {"url": self.base_url, "model": self.resolve_model(), "reachable": True}
        except RuntimeError as error:
            return {"url": self.base_url, "model": self.model, "reachable": False,
                    "detail": str(error)}

    def complete(self, content: list[dict[str, object]], model: str) -> str:
        response = self._post(
            "/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": self.max_tokens,
                "temperature": 0.0,
            },
        )
        try:
            text = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as error:
            raise RuntimeError(f"Unexpected response from the model: {response}") from error
        if not text or not text.strip():
            raise RuntimeError("The model returned an empty description")
        return str(text).strip()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _get(self, path: str) -> dict:
        return self._send(urllib.request.Request(f"{self.base_url}{path}", headers=self._headers()))

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}", data=json.dumps(payload).encode(), headers=self._headers()
        )
        return self._send(request)

    def _send(self, request: urllib.request.Request) -> dict:
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"Model endpoint returned {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise RuntimeError(
                f"Could not reach the model at {self.base_url} ({error}). Start it with: "
                "mlx_vlm.server --model ~/models/Qwen3-VL-30B-A3B-Instruct-4bit --port 8080"
            ) from error


def describe_shot(shot: Shot, directory: Path, config: ModelConfig) -> tuple[str, str]:
    """Send a shot's frames to the vision model and return (description, model name)."""
    if not shot.samples:
        raise RuntimeError(f"{shot.id} has no extracted frames to describe")

    frame_lines = "\n".join(
        f"- image {number}: {sample.timestamp:.2f} seconds"
        for number, sample in enumerate(shot.samples, start=1)
    )
    content: list[dict[str, object]] = [
        {"type": "text", "text": PROMPT.format(frame_lines=frame_lines)}
    ]
    for sample in shot.samples:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _data_url(directory / sample.image)},
            }
        )

    model = config.resolve_model()
    return config.complete(content, model), model


def _data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode()
    return f"data:image/jpeg;base64,{encoded}"
