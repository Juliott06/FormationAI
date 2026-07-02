from __future__ import annotations

import base64
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import httpx

from app.core.config import get_settings

logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class BBoxPx:
    """Bounding box in pixel coordinates of the source frame, (x1, y1, x2, y2)."""

    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


class LocateAnythingError(RuntimeError):
    """Raised when the configured backend is unavailable or returned bad data."""


class LocateAnythingNotConfigured(LocateAnythingError):
    """Raised when no backend is configured; surface as HTTP 503."""


class LocateAnythingClient(ABC):
    @abstractmethod
    def identify(
        self,
        frame_jpeg: bytes,
        frame_width: int,
        frame_height: int,
        prompts: dict[str, str],
    ) -> dict[str, BBoxPx]:
        """Return {name: BBoxPx} for prompts the model located.

        Names with no detection are omitted. Coordinates are in the source frame's
        pixel space (not the 0..1000 normalized space LocateAnything uses natively).
        """


class DisabledClient(LocateAnythingClient):
    def identify(self, frame_jpeg, frame_width, frame_height, prompts):
        raise LocateAnythingNotConfigured(
            "LocateAnything backend is not configured. "
            "Set FORMATIONAI_LOCATEANYTHING_BACKEND=http and FORMATIONAI_LOCATEANYTHING_URL."
        )


class HttpClient(LocateAnythingClient):
    """POSTs frame + prompts to a remote LocateAnything server.

    Expected server contract (you control this — see scripts/mock_locateanything.py):
      POST {url}
      body: {"image_b64": "<jpeg base64>", "prompts": {"name": "desc", ...}}
      reply: {"boxes": {"name": [x1, y1, x2, y2], ...}}  # coords 0..1000 normalized
    """

    def __init__(self, url: str, timeout_sec: float) -> None:
        self._url = url
        # One persistent connection-pooling client: identify() is called once
        # per video frame, and a fresh client per call would pay a TCP (+TLS)
        # handshake thousands of times per clip.
        self._client = httpx.Client(timeout=timeout_sec)

    def identify(self, frame_jpeg, frame_width, frame_height, prompts):
        payload = {
            "image_b64": base64.b64encode(frame_jpeg).decode("ascii"),
            "prompts": prompts,
        }
        try:
            resp = self._client.post(self._url, json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPError as exc:
            raise LocateAnythingError(f"LocateAnything HTTP call failed: {exc}") from exc
        except ValueError as exc:
            raise LocateAnythingError(f"LocateAnything bad JSON: {exc}") from exc

        boxes_raw = data.get("boxes", {})
        result: dict[str, BBoxPx] = {}
        for name, coords in boxes_raw.items():
            if not (isinstance(coords, list) and len(coords) == 4):
                logger.warning("LocateAnything: skip malformed box for %r: %r", name, coords)
                continue
            x1 = int(round(coords[0] / 1000.0 * frame_width))
            y1 = int(round(coords[1] / 1000.0 * frame_height))
            x2 = int(round(coords[2] / 1000.0 * frame_width))
            y2 = int(round(coords[3] / 1000.0 * frame_height))
            x1, x2 = max(0, min(x1, x2)), min(frame_width, max(x1, x2))
            y1, y2 = max(0, min(y1, y2)), min(frame_height, max(y1, y2))
            if x2 > x1 and y2 > y1:
                result[name] = BBoxPx(x1, y1, x2, y2)
        return result


def build_client() -> LocateAnythingClient:
    settings = get_settings()
    backend = settings.locateanything_backend
    if backend == "http":
        if not settings.locateanything_url:
            raise LocateAnythingNotConfigured(
                "FORMATIONAI_LOCATEANYTHING_BACKEND=http but FORMATIONAI_LOCATEANYTHING_URL is empty."
            )
        return HttpClient(settings.locateanything_url, settings.locateanything_timeout_sec)
    return DisabledClient()
