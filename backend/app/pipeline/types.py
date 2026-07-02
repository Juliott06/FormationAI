from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class Detection:
    bbox: tuple[int, int, int, int]
    anchor_px: tuple[int, int]
    confidence: float


@dataclass(slots=True)
class TrackedDetection:
    track_id: int
    bbox: tuple[int, int, int, int]
    anchor_px: tuple[int, int]
    confidence: float

