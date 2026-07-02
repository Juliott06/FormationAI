"""Pipeline that runs CoTracker3 on a video and folds its tracks into
the same FramePositions format the rest of the app uses.

Differs from sam2_pipeline:
- CoTracker returns POINT tracks (not masks/boxes). We synthesize a small bbox
  around each tracked point for compatibility with downstream code that
  expects bbox (e.g. appearance histogram extraction is skipped since the
  point may not be on the dancer's torso).
- Per-point `visible` flag tells us when CoTracker thinks the dancer is
  occluded; we drop those frames so existing interpolation can fill them.
- Identity-recovery post-processing stages are SKIPPED — CoTracker already
  binds tracks to specific click points so identity is deterministic.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from app.core.config import get_settings
from app.pipeline.pose import normalize_stage_proxy
from app.pipeline.processor import finalize_frames, inspect_video_file
from app.schemas.jobs import (
    CoordinateSpaceMetadata,
    DancerClick,
    DancerPosition,
    DetectionSummary,
    FramePositions,
    PositionsResult,
    VideoMetadata,
    unique_click_names,
)
from app.services.cotracker import TrackedPoint, build_client


logger = logging.getLogger("uvicorn.error")


_SYNTHETIC_BBOX_W = 80
_SYNTHETIC_BBOX_H = 160


def tracked_points_to_frames(
    tracks: list[list[TrackedPoint]],
    clicks: list[DancerClick],
    video_meta: VideoMetadata,
) -> list[FramePositions]:
    """Convert per-click CoTracker point tracks into per-frame DancerPosition lists.

    Multi-click semantics: clicks with the same `name` represent multiple
    tracking points on one dancer (e.g. head + torso + foot). Per frame, we
    emit one DancerPosition per UNIQUE name if ANY of that name's points is
    visible.

    Stage anchor (depth/width) is computed by PROJECTING each visible point
    to its estimated foot using the per-click foot offset measured at the
    key frame (foot_offset_i = key_frame_foot_y - key_frame_y_i). The median
    of projected feet across visible points is the dancer's stage anchor.
    This keeps depth stable when only the head/torso is visible — the head
    point's y is mapped to where the foot would be.

    Pure function — easy to unit test."""
    # Roster order: first occurrence of each name is its track_id
    unique_names = unique_click_names(clicks)
    name_to_id: dict[str, int] = {n: idx + 1 for idx, n in enumerate(unique_names)}
    # Map name → indices into the parallel tracks/clicks lists
    name_to_indices: dict[str, list[int]] = {n: [] for n in unique_names}
    for idx, click in enumerate(clicks):
        name_to_indices[click.name].append(idx)

    # Per-click foot offset: how much further down (in image y) is the foot
    # vs this click's key-frame y. For a single-click dancer this is 0.
    foot_offset_per_click: dict[int, int] = {}
    for name in unique_names:
        click_idxs = name_to_indices[name]
        foot_y_at_keyframe = max(clicks[ci].y for ci in click_idxs)
        for ci in click_idxs:
            foot_offset_per_click[ci] = foot_y_at_keyframe - clicks[ci].y

    frames: list[FramePositions] = []
    for fi in range(video_meta.frame_count):
        dancers: list[DancerPosition] = []
        for name in unique_names:
            projected_ys: list[int] = []
            xs: list[int] = []
            for click_idx in name_to_indices[name]:
                if fi >= len(tracks[click_idx]):
                    continue
                pt = tracks[click_idx][fi]
                if pt is None or not pt.visible:
                    continue
                projected_ys.append(pt.y + foot_offset_per_click[click_idx])
                xs.append(pt.x)
            if not projected_ys:
                continue
            projected_ys.sort()
            xs.sort()
            ay = projected_ys[len(projected_ys) // 2]
            ax = xs[len(xs) // 2]
            ay = max(0, min(video_meta.height - 1, ay))
            ax = max(0, min(video_meta.width - 1, ax))
            half_w = _SYNTHETIC_BBOX_W // 2
            half_h = _SYNTHETIC_BBOX_H // 2
            bx = max(0, ax - half_w)
            by = max(0, ay - _SYNTHETIC_BBOX_H + half_h // 2)
            bw = min(_SYNTHETIC_BBOX_W, video_meta.width - bx)
            bh = min(_SYNTHETIC_BBOX_H, video_meta.height - by)
            anchor_px = (ax, ay)
            x, y = normalize_stage_proxy(anchor_px, video_meta.width, video_meta.height)
            dancers.append(
                DancerPosition(
                    id=name_to_id[name],
                    bbox=[bx, by, max(bw, 1), max(bh, 1)],
                    anchor_px=list(anchor_px),
                    x=x,
                    y=y,
                    confidence=1.0,
                )
            )
        frames.append(
            FramePositions(
                frame=fi,
                timestamp_sec=round(fi / max(video_meta.fps, 1.0), 3),
                dancers=dancers,
            )
        )
    return frames


def process_video_with_cotracker(
    *,
    job_id: str,
    video_path: Path,
    clicks: list[DancerClick],
    key_frame: int,
    progress_callback: Callable[[int, int], None],
) -> PositionsResult:
    """Run the CoTracker click-to-track pipeline end to end."""
    video_meta = inspect_video_file(video_path)
    if not clicks:
        raise ValueError("CoTracker pipeline requires at least one click")

    click_payload = [
        {"name": c.name, "key_frame": key_frame, "x": c.x, "y": c.y} for c in clicks
    ]

    client = build_client()
    logger.info(
        "CoTracker pipeline: job=%s frames=%d clicks=%d key_frame=%d",
        job_id, video_meta.frame_count, len(clicks), key_frame,
    )
    progress_callback(0, max(video_meta.frame_count, 1))

    t0 = time.perf_counter()
    tracks = client.track_video(video_path, click_payload)
    logger.info(
        "CoTracker returned %d point tracks in %.1fs",
        len(tracks), time.perf_counter() - t0,
    )

    frames = tracked_points_to_frames(tracks, clicks, video_meta)

    dancers_per_frame_total = sum(len(f.dancers) for f in frames)
    max_dancers_in_frame = max((len(f.dancers) for f in frames), default=0)
    frames_with_detections = sum(1 for f in frames if f.dancers)

    progress_callback(video_meta.frame_count, max(video_meta.frame_count, 1))

    formations = finalize_frames(frames, fps=video_meta.fps)

    expected = len(unique_click_names(clicks))
    frames_below_expected = sum(1 for f in frames if len(f.dancers) < expected)
    frames_meeting_expected = sum(1 for f in frames if len(f.dancers) >= expected)

    total_frames = len(frames)
    return PositionsResult(
        job_id=job_id,
        video=video_meta,
        coordinate_space=CoordinateSpaceMetadata(
            image_anchor_px="pixel anchor point in the source frame",
            normalized_stage_proxy=(
                "auto-fit top-down proxy: y axis stretched to the observed anchor band "
                "(10th-90th percentile), occupying the middle 50% of the canvas; "
                "not floor-plane calibrated"
            ),
        ),
        summary=DetectionSummary(
            expected_dancer_count=expected,
            unique_track_ids=expected,
            max_dancers_in_frame=max_dancers_in_frame,
            average_dancers_per_frame=round(
                dancers_per_frame_total / max(total_frames, 1), 3
            ),
            frames_with_detections=frames_with_detections,
            frames_below_expected=frames_below_expected,
            frames_meeting_expected=frames_meeting_expected,
        ),
        frames=frames,
        formations=formations,
    )
