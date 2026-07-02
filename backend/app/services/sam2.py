from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.core.config import get_settings

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class FrameBox:
    """Per-frame bbox for one tracked dancer, returned by SAM2 server."""

    frame: int
    x1: int
    y1: int
    x2: int
    y2: int
    score: float


class Sam2Error(RuntimeError):
    """Raised when the configured backend is unavailable or returned bad data."""


class Sam2NotConfigured(Sam2Error):
    """Raised when no backend is configured; surface as HTTP 503."""


class Sam2Client(ABC):
    @abstractmethod
    def track_video(
        self,
        video_path: Path,
        clicks: list[dict],
    ) -> dict[str, list[FrameBox]]:
        """Send the video + clicks to SAM2, return per-name list of FrameBox.

        clicks: list of {"name": str, "key_frame": int, "x": int, "y": int}
        Returns: {name: [FrameBox, ...]} — frames where SAM2 lost the dancer are
        simply omitted (existing interpolation fills small gaps).
        """


class DisabledClient(Sam2Client):
    def track_video(self, video_path, clicks):
        raise Sam2NotConfigured(
            "SAM2 backend is not configured. "
            "Set FORMATIONAI_SAM2_BACKEND=http and FORMATIONAI_SAM2_URL."
        )


class HttpClient(Sam2Client):
    """Multipart POST: video file + clicks_json to a remote SAM2 server.

    Expected server contract (you control this — see scripts/mock_sam2.py):

      POST {url}
      multipart/form-data:
        video: <video file>
        clicks_json: '[{"name": "...", "key_frame": 0, "x": 320, "y": 180}, ...]'

      Response (application/json):
      {
        "tracks": {
          "Yeji": [
            {"frame": 0, "bbox": [x1, y1, x2, y2], "score": 0.95},
            ...
          ],
          ...
        }
      }

    Coords are pixel-space in the source video. Missing frames per name = SAM2
    lost the dancer briefly; we leave the gap and let interpolation fill.
    """

    def __init__(self, url: str, timeout_sec: float) -> None:
        self._url = url
        self._timeout = timeout_sec

    def track_video(self, video_path, clicks):
        clicks_json = json.dumps(clicks)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                with video_path.open("rb") as f:
                    files = {"video": (video_path.name, f, "video/mp4")}
                    data = {"clicks_json": clicks_json}
                    resp = client.post(self._url, files=files, data=data)
                    resp.raise_for_status()
                payload = resp.json()
        except httpx.HTTPError as exc:
            raise Sam2Error(f"SAM2 HTTP call failed: {exc}") from exc
        except ValueError as exc:
            raise Sam2Error(f"SAM2 bad JSON: {exc}") from exc

        tracks_raw = payload.get("tracks", {})
        if not isinstance(tracks_raw, dict):
            raise Sam2Error(f"SAM2 'tracks' must be an object, got {type(tracks_raw)}")

        result: dict[str, list[FrameBox]] = {}
        for name, frames in tracks_raw.items():
            if not isinstance(frames, list):
                logger.warning("SAM2: skip non-list tracks for %r", name)
                continue
            out: list[FrameBox] = []
            for entry in frames:
                if not isinstance(entry, dict):
                    continue
                bbox = entry.get("bbox")
                frame = entry.get("frame")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                if not isinstance(frame, int):
                    continue
                score = float(entry.get("score", 1.0))
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                if x2 <= x1 or y2 <= y1:
                    continue
                out.append(FrameBox(frame=frame, x1=x1, y1=y1, x2=x2, y2=y2, score=score))
            out.sort(key=lambda fb: fb.frame)
            result[name] = out
        return result


def build_client() -> Sam2Client:
    settings = get_settings()
    backend = settings.sam2_backend
    if backend == "http":
        if not settings.sam2_url:
            raise Sam2NotConfigured(
                "FORMATIONAI_SAM2_BACKEND=http but FORMATIONAI_SAM2_URL is empty."
            )
        return HttpClient(settings.sam2_url, settings.sam2_timeout_sec)
    return DisabledClient()
