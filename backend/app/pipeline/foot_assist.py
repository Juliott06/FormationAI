"""Derive per-frame FOOT positions for click-tracked dancers using YOLO boxes.

Why this exists: users click a dancer's torso, and CoTracker tracks that torso
point reliably. But in a typical practice-room video the camera sits at roughly
chest height, which puts torso points near the image horizon — and a point at
the horizon barely moves in the image as the dancer moves toward/away from the
camera. The depth signal lives in the FEET (far below the horizon). Observed on
a real 9-dancer clip: tracked torso points spanned only ~94 px vertically over
a clip where dancers visibly crossed the whole floor.

The hybrid: CoTracker answers "who is where" (identity via the tracked point),
YOLO person detection answers "where are their feet" (bbox bottom). Per sampled
frame, each tracked point is matched to the detection box containing it, and
the box's bottom-center becomes the dancer's foot anchor. Between samples (and
when a dancer has no matching box) the last known point→foot offset is carried,
so a missed detection degrades gracefully instead of dropping depth.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

from app.pipeline.types import Detection
from app.services.cotracker import TrackedPoint


logger = logging.getLogger("uvicorn.error")

Point = tuple[int, int]


class _Detector(Protocol):
    def detect(self, frame_bgr: object) -> list[Detection]: ...


def representative_points(
    tracks: list[list[TrackedPoint]],
    click_names: list[str],
) -> dict[str, list[Point | None]]:
    """Per unique name, the median visible tracked point per frame (None when
    no point of that dancer is visible). Mirrors the multi-click semantics of
    tracked_points_to_frames."""
    n_frames = max((len(t) for t in tracks), default=0)
    by_name_indices: dict[str, list[int]] = {}
    for idx, name in enumerate(click_names):
        by_name_indices.setdefault(name, []).append(idx)

    out: dict[str, list[Point | None]] = {}
    for name, idxs in by_name_indices.items():
        series: list[Point | None] = []
        for fi in range(n_frames):
            xs: list[int] = []
            ys: list[int] = []
            for ci in idxs:
                if fi < len(tracks[ci]):
                    pt = tracks[ci][fi]
                    if pt is not None and pt.visible:
                        xs.append(pt.x)
                        ys.append(pt.y)
            if not xs:
                series.append(None)
                continue
            xs.sort()
            ys.sort()
            series.append((xs[len(xs) // 2], ys[len(ys) // 2]))
        out[name] = series
    return out


def match_points_to_boxes(
    points: dict[str, Point],
    boxes: list[Detection],
) -> dict[str, Point]:
    """Match each named point to the best person box and return foot anchors
    (bbox bottom-center). Pure function.

    A box is a candidate for a point if the point lies inside it (with a small
    tolerance). Each box is claimed by at most one name — nearest point-to-
    center wins — so two dancers crossing don't both snap to one box."""
    candidates: list[tuple[float, str, int]] = []
    for name, (px, py) in points.items():
        for bi, det in enumerate(boxes):
            x, y, w, h = det.bbox
            tol_x = max(int(w * 0.15), 8)
            if not (x - tol_x <= px <= x + w + tol_x and y <= py <= y + h):
                continue
            cx = x + w / 2.0
            cy = y + h / 2.0
            dist_sq = (px - cx) ** 2 + (py - cy) ** 2
            candidates.append((dist_sq, name, bi))
    candidates.sort()

    matched: dict[str, Point] = {}
    used_boxes: set[int] = set()
    for _dist, name, bi in candidates:
        if name in matched or bi in used_boxes:
            continue
        x, y, w, h = boxes[bi].bbox
        matched[name] = (x + w // 2, y + h)
        used_boxes.add(bi)
    return matched


def compute_foot_anchors(
    video_path: Path,
    rep_points: dict[str, list[Point | None]],
    *,
    detector: _Detector,
    sample_every: int = 3,
) -> dict[str, list[Point | None]]:
    """For each dancer and frame, the estimated FOOT position.

    On sampled frames, feet come from matched YOLO boxes. On other frames (and
    on sample frames where the match failed), the dancer's most recent
    point→foot offset is applied to their tracked point. Dancers who never
    match a box get their raw tracked points back (graceful no-op)."""
    import cv2

    n_frames = max((len(s) for s in rep_points.values()), default=0)
    if n_frames == 0:
        return {}

    sample_every = max(int(sample_every), 1)
    offsets: dict[str, tuple[int, int]] = {}  # name -> (dx, dy) point→foot
    feet: dict[str, list[Point | None]] = {
        name: [None] * n_frames for name in rep_points
    }
    match_count = 0
    sample_count = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    try:
        for fi in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            frame_points = {
                name: series[fi]
                for name, series in rep_points.items()
                if fi < len(series) and series[fi] is not None
            }
            if fi % sample_every == 0 and frame_points:
                sample_count += 1
                boxes = detector.detect(frame)
                matched = match_points_to_boxes(frame_points, boxes)  # type: ignore[arg-type]
                match_count += len(matched)
                for name, foot in matched.items():
                    px, py = frame_points[name]  # type: ignore[misc]
                    offsets[name] = (foot[0] - px, foot[1] - py)
            for name, pt in frame_points.items():
                if pt is None:
                    continue
                dx, dy = offsets.get(name, (0, 0))
                feet[name][fi] = (pt[0] + dx, pt[1] + dy)
    finally:
        cap.release()

    total_named = sum(1 for s in rep_points.values() for p in s if p is not None)
    logger.info(
        "foot assist: %d sampled frames, %d box matches, %d/%d dancer-frames anchored",
        sample_count,
        match_count,
        sum(1 for s in feet.values() for p in s if p is not None),
        total_named,
    )
    return feet
