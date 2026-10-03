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


# A tracked point is "on" a box when it sits inside it horizontally (with a
# little tolerance) and vertically above the box's lower edge zone — a torso
# point near the very bottom of a box is almost always a back-row dancer
# whose torso happens to overlap a front dancer's legs.
_MAX_POINT_FRACTION_IN_BOX = 0.85
# With a known point->foot offset, a box is accepted only if its bottom-center
# lies within this many box-heights of the predicted foot.
_FOOT_GATE_BOX_HEIGHTS = 0.45
# Median filter width (in SAMPLES, ~1s at the default sample_every=3)
# applied to each dancer's offset series. The offset only changes slowly (body
# scale with depth, posture), so a wide window costs little.
# Kills single-sample glitches: jumps (box bottom leaves the floor), a wrong
# box claimed for one sample during a crossing, a detection clipped by an arm.
_OFFSET_MEDIAN_SAMPLES = 9


def match_points_to_boxes(
    points: dict[str, Point],
    boxes: list[Detection],
    offsets: dict[str, tuple[int, int]] | None = None,
) -> dict[str, Point]:
    """Foot anchor (matched box's bottom-center) per name; see _assign_boxes."""
    out: dict[str, Point] = {}
    for name, bi in _assign_boxes(points, boxes, offsets).items():
        x, y, w, h = boxes[bi].bbox
        out[name] = (x + w // 2, y + h)
    return out


def _assign_boxes(
    points: dict[str, Point],
    boxes: list[Detection],
    offsets: dict[str, tuple[int, int]] | None = None,
) -> dict[str, int]:
    """Match each named point to the best person box and return foot anchors
    (bbox bottom-center). Pure function.

    A box is a candidate for a point if the point lies inside it (with a small
    tolerance, and not in the bottom ~15% where only legs are). Assignment is
    a global optimum (Hungarian), each box claimed by at most one name.

    Cost: when the dancer's current point->foot offset is known, the distance
    between the PREDICTED foot and the box's bottom-center (in box heights) —
    this is what separates a back-row dancer from the front-row dancer whose
    box overlaps their torso: the boxes contain the same torso point, but only
    one has its bottom where this dancer's feet are. Candidates outside
    _FOOT_GATE_BOX_HEIGHTS are rejected. Without an offset, falls back to
    point-to-box-center distance."""
    if not points or not boxes:
        return {}
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    names = list(points)
    big = 1e6
    cost = np.full((len(names), len(boxes)), big, dtype=np.float64)
    for ni, name in enumerate(names):
        px, py = points[name]
        off = offsets.get(name) if offsets else None
        for bi, det in enumerate(boxes):
            x, y, w, h = det.bbox
            if h <= 0 or w <= 0:
                continue
            tol_x = max(int(w * 0.15), 8)
            if not (x - tol_x <= px <= x + w + tol_x and y <= py <= y + h):
                continue
            if (py - y) / h > _MAX_POINT_FRACTION_IN_BOX:
                continue
            fx, fy = x + w / 2.0, float(y + h)
            if off is not None:
                d = ((px + off[0] - fx) ** 2 + (py + off[1] - fy) ** 2) ** 0.5 / h
                if d > _FOOT_GATE_BOX_HEIGHTS:
                    continue
                cost[ni, bi] = d
            else:
                cx, cy = x + w / 2.0, y + h / 2.0
                cost[ni, bi] = (((px - cx) / w) ** 2 + ((py - cy) / h) ** 2) ** 0.5
    rows, cols = linear_sum_assignment(cost)
    return {
        names[ni]: int(bi) for ni, bi in zip(rows, cols) if cost[ni, bi] < big
    }


def _median_filter(values: list[float], width: int) -> list[float]:
    if width <= 1 or len(values) < 3:
        return list(values)
    half = width // 2
    out: list[float] = []
    for i in range(len(values)):
        window = sorted(values[max(0, i - half) : i + half + 1])
        out.append(window[len(window) // 2])
    return out


def _smooth_offset_series(
    samples: list[tuple[int, int, int]], n_frames: int
) -> list[tuple[int, int] | None]:
    """samples: (frame, dx, dy) observations, frame-ascending. Returns a per-
    frame offset: median-filtered, linearly interpolated between samples and
    held flat before the first / after the last sample."""
    out: list[tuple[int, int] | None] = [None] * n_frames
    if not samples:
        return out
    frames = [s[0] for s in samples]
    dxs = _median_filter([float(s[1]) for s in samples], _OFFSET_MEDIAN_SAMPLES)
    dys = _median_filter([float(s[2]) for s in samples], _OFFSET_MEDIAN_SAMPLES)
    k = 0
    for fi in range(n_frames):
        while k + 1 < len(frames) and frames[k + 1] <= fi:
            k += 1
        if fi <= frames[0]:
            dx, dy = dxs[0], dys[0]
        elif k + 1 >= len(frames):
            dx, dy = dxs[-1], dys[-1]
        else:
            t = (fi - frames[k]) / max(frames[k + 1] - frames[k], 1)
            dx = dxs[k] + t * (dxs[k + 1] - dxs[k])
            dy = dys[k] + t * (dys[k + 1] - dys[k])
        out[fi] = (int(round(dx)), int(round(dy)))
    return out


def compute_foot_anchors(
    video_path: Path,
    rep_points: dict[str, list[Point | None]],
    *,
    detector: _Detector,
    sample_every: int = 3,
    initial_offsets: dict[str, tuple[int, int]] | None = None,
) -> dict[str, list[Point | None]]:
    """For each dancer and frame, the estimated FOOT position (None where the
    dancer isn't tracked, or where no foot estimate exists at all — callers
    then fall back to their own estimate).

    On sampled frames, YOLO boxes are matched to tracked points; each match
    yields a point->foot offset. The offset series is median-filtered and
    interpolated, then applied to the tracked point on every frame.

    initial_offsets: per-dancer offset measured at the key frame (from the box
    under the user's click). Seeds matching so the very first samples already
    use the foot-prediction gate, and is the fallback for a dancer YOLO never
    matches. A dancer with neither gets None — NOT their raw tracked point,
    which would put a torso where the feet belong and shove them to the back
    of the stage."""
    import cv2

    n_frames = max((len(s) for s in rep_points.values()), default=0)
    if n_frames == 0:
        return {}

    sample_every = max(int(sample_every), 1)
    current: dict[str, tuple[int, int]] = dict(initial_offsets or {})
    samples: dict[str, list[tuple[int, int, int]]] = {name: [] for name in rep_points}
    match_count = 0
    sample_count = 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    try:
        for fi in range(n_frames):
            frame_points = {
                name: series[fi]
                for name, series in rep_points.items()
                if fi < len(series) and series[fi] is not None
            }
            if fi % sample_every != 0 or not frame_points:
                if not cap.grab():
                    break
                continue
            ok, frame = cap.read()
            if not ok:
                break
            sample_count += 1
            boxes = detector.detect(frame)
            matched = match_points_to_boxes(frame_points, boxes, current)  # type: ignore[arg-type]
            match_count += len(matched)
            for name, foot in matched.items():
                px, py = frame_points[name]  # type: ignore[misc]
                off = (foot[0] - px, foot[1] - py)
                current[name] = off
                samples[name].append((fi, off[0], off[1]))
    finally:
        cap.release()

    feet: dict[str, list[Point | None]] = {}
    for name, series in rep_points.items():
        offs = _smooth_offset_series(samples[name], n_frames)
        seed = (initial_offsets or {}).get(name)
        out: list[Point | None] = [None] * n_frames
        for fi in range(min(n_frames, len(series))):
            pt = series[fi]
            if pt is None:
                continue
            off = offs[fi] if offs[fi] is not None else seed
            if off is None:
                continue
            out[fi] = (pt[0] + off[0], pt[1] + off[1])
        feet[name] = out

    total_named = sum(1 for s in rep_points.values() for p in s if p is not None)
    logger.info(
        "foot assist: %d sampled frames, %d box matches, %d/%d dancer-frames anchored",
        sample_count,
        match_count,
        sum(1 for s in feet.values() for p in s if p is not None),
        total_named,
    )
    return feet


def match_clicks_to_boxes(
    click_points: dict[str, list[Point]],
    boxes: list[Detection],
) -> dict[str, tuple[int, int, int, int]]:
    """At the key frame: the person box under each dancer's clicks.

    Uses the median click as the dancer's point. Pure function."""
    reps: dict[str, Point] = {}
    for name, pts in click_points.items():
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        reps[name] = (xs[len(xs) // 2], ys[len(ys) // 2])
    if not reps or not boxes:
        return {}
    return {name: boxes[bi].bbox for name, bi in _assign_boxes(reps, boxes).items()}
