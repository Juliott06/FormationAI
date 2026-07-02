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
    unique_click_names,
)
from app.services.sam2 import FrameBox, Sam2Error, build_client as build_sam2_client
from app.schemas.jobs import VideoMetadata

logger = logging.getLogger("uvicorn.error")


def tracks_to_frames(
    tracks: dict[str, list[FrameBox]],
    clicks: list[DancerClick],
    video_meta: VideoMetadata,
) -> list[FramePositions]:
    """Convert per-name SAM2 tracks into per-frame DancerPosition lists.

    Pure function — easy to unit test. Missing (name, frame) pairs are simply
    omitted; downstream interpolation fills small gaps.

    Multi-click semantics: SAM2 cannot use multiple click points per name on
    the server side, so we collapse duplicate names to the first occurrence
    here. (CoTracker is the path that actually supports multi-click.)"""
    names = unique_click_names(clicks)
    first_click_by_name = {}
    for click in clicks:
        first_click_by_name.setdefault(click.name, click)
    unique_clicks: list[DancerClick] = [first_click_by_name[n] for n in names]
    name_to_id: dict[str, int] = {n: idx + 1 for idx, n in enumerate(names)}
    by_name_frame: dict[str, dict[int, FrameBox]] = {
        name: {fb.frame: fb for fb in fblist} for name, fblist in tracks.items()
    }
    frames: list[FramePositions] = []
    for fi in range(video_meta.frame_count):
        dancers: list[DancerPosition] = []
        for click in unique_clicks:
            fb = by_name_frame.get(click.name, {}).get(fi)
            if fb is None:
                continue
            w = max(fb.x2 - fb.x1, 1)
            h = max(fb.y2 - fb.y1, 1)
            anchor_px = ((fb.x1 + fb.x2) // 2, fb.y2)
            x, y = normalize_stage_proxy(anchor_px, video_meta.width, video_meta.height)
            dancers.append(
                DancerPosition(
                    id=name_to_id[click.name],
                    bbox=[fb.x1, fb.y1, w, h],
                    anchor_px=list(anchor_px),
                    x=x,
                    y=y,
                    confidence=round(min(max(fb.score, 0.0), 1.0), 4),
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


def process_video_with_sam2(
    *,
    job_id: str,
    video_path: Path,
    clicks: list[DancerClick],
    key_frame: int,
    progress_callback: Callable[[int, int], None],
) -> PositionsResult:
    """Run the SAM2 click-to-track pipeline end to end.

    Single SAM2 server call returns per-name per-frame boxes; we convert to
    FramePositions and run the subset of post-processing that still applies
    (identity-recovery passes are skipped because SAM2 already gives clean
    per-dancer tracks)."""
    settings = get_settings()
    video_meta = inspect_video_file(video_path)
    if not clicks:
        raise ValueError("SAM2 pipeline requires at least one click")

    # SAM2 server expects unique names per click — multi-click is a CoTracker-only
    # feature. Send the first occurrence per name.
    seen_payload_names: set[str] = set()
    click_payload: list[dict] = []
    for c in clicks:
        if c.name in seen_payload_names:
            continue
        seen_payload_names.add(c.name)
        click_payload.append({"name": c.name, "key_frame": key_frame, "x": c.x, "y": c.y})

    expected = len(unique_click_names(clicks))

    sam2_client = build_sam2_client()
    logger.info(
        "SAM2 pipeline: job=%s frames=%d clicks=%d key_frame=%d url=%s",
        job_id,
        video_meta.frame_count,
        len(clicks),
        key_frame,
        settings.sam2_url,
    )
    progress_callback(0, max(video_meta.frame_count, 1))

    call_start = time.perf_counter()
    try:
        tracks = sam2_client.track_video(video_path, click_payload)
    except Sam2Error:
        raise
    logger.info(
        "SAM2 call returned %d named tracks in %.1fs",
        len(tracks),
        time.perf_counter() - call_start,
    )

    frames = tracks_to_frames(tracks, clicks, video_meta)

    dancers_per_frame_total = sum(len(f.dancers) for f in frames)
    max_dancers_in_frame = max((len(f.dancers) for f in frames), default=0)
    frames_with_detections = sum(1 for f in frames if f.dancers)

    progress_callback(video_meta.frame_count, max(video_meta.frame_count, 1))

    formations = finalize_frames(frames, fps=video_meta.fps)

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
