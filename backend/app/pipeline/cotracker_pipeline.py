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
from app.pipeline.foot_assist import (
    compute_foot_anchors,
    match_clicks_to_boxes,
    representative_points,
)
from app.pipeline.homography import compute_stage_homography, project_to_stage
from app.pipeline.pose import normalize_stage_proxy
from app.pipeline.processor import (
    apply_stage_rescue,
    finalize_frames,
    inspect_video_file,
)
from app.pipeline.yolo_detector import YoloPersonDetector
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
    homography: list[list[float]] | None = None,
    foot_anchors: dict[str, list[tuple[int, int] | None]] | None = None,
    click_foot_offsets: list[int | None] | None = None,
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

    homography: if provided, the foot anchor is mapped through it to true
    top-down floor coords; otherwise falls back to raw camera-space proxy.

    foot_anchors: optional per-name per-frame FOOT positions from foot-assist
    (YOLO bbox bottoms). When present for a (name, frame), it overrides the
    click-offset estimate — tracked torso points near the camera's horizon
    carry almost no depth signal, so real feet matter for depth accuracy.

    click_foot_offsets: optional per-click y offset to the dancer's feet,
    measured at the key frame from the person box under the click. Overrides
    the "lowest click is the foot" guess, which for a single torso click means
    offset 0 — i.e. the torso treated as the feet, putting the dancer far too
    deep on the stage.

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
            if click_foot_offsets is not None and ci < len(click_foot_offsets):
                measured = click_foot_offsets[ci]
                if measured is not None:
                    foot_offset_per_click[ci] = measured

    frames: list[FramePositions] = []
    for fi in range(video_meta.frame_count):
        dancers: list[DancerPosition] = []
        for name in unique_names:
            assist = None
            if foot_anchors is not None:
                series = foot_anchors.get(name)
                if series is not None and fi < len(series):
                    assist = series[fi]
            if assist is not None:
                ax, ay = assist
            else:
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
            if homography is not None:
                x, y = project_to_stage(homography, anchor_px)
            else:
                x, y = normalize_stage_proxy(
                    anchor_px, video_meta.width, video_meta.height
                )
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


_KEY_FRAME_DETECT_OFFSETS = (-4, -2, 0, 2, 4)


def _read_frames_sequential(video_path: Path, indices: list[int]) -> dict[int, object]:
    """Decode frames from the start and keep the requested indices. Matches
    how CoTracker reads the video (sequentially) rather than trusting
    CAP_PROP_POS_FRAMES seeking."""
    import cv2

    wanted = sorted(i for i in set(indices) if i >= 0)
    out: dict[int, object] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    try:
        for fi in range(wanted[-1] + 1):
            if fi in wanted:
                ok, frame = cap.read()
                if not ok:
                    break
                out[fi] = frame
            elif not cap.grab():
                break
    finally:
        cap.release()
    return out


def _key_frame_boxes(
    video_path: Path,
    key_frame: int,
    clicks: list[DancerClick],
    detector: YoloPersonDetector,
) -> dict[str, tuple[int, int, int, int]]:
    """Person box under each dancer's clicks around the key frame.

    Detects on a few frames either side of the key frame (dancers barely move
    in ±4 frames) and takes each dancer's per-coordinate median box, so one
    missed or merged detection doesn't leave a dancer without a box."""
    frames = _read_frames_sequential(
        video_path, [key_frame + o for o in _KEY_FRAME_DETECT_OFFSETS]
    )
    by_name: dict[str, list[tuple[int, int]]] = {}
    for c in clicks:
        by_name.setdefault(c.name, []).append((c.x, c.y))
    found: dict[str, list[tuple[int, int, int, int]]] = {}
    for frame in frames.values():
        for name, box in match_clicks_to_boxes(by_name, detector.detect(frame)).items():
            found.setdefault(name, []).append(box)
    out: dict[str, tuple[int, int, int, int]] = {}
    for name, boxes in found.items():
        cols = [sorted(b[i] for b in boxes) for i in range(4)]
        out[name] = tuple(col[len(col) // 2] for col in cols)  # type: ignore[assignment]
    return out


# Auto support points, as fractions of the dancer's key-frame box height
# measured from the top: upper chest and hips, on the box's vertical centre
# line. Tracked alongside the user's click(s); per frame the median of the
# visible points is used, so one point sliding onto a neighbour during a
# crossing (the classic CoTracker failure) is outvoted instead of dragging
# the dancer's dot across the stage.
_SUPPORT_POINT_FRACTIONS = (0.30, 0.55)


def add_support_points(
    clicks: list[DancerClick],
    key_boxes: dict[str, tuple[int, int, int, int]],
    video_meta: VideoMetadata,
) -> list[DancerClick]:
    """Return clicks + extra same-name points inside each matched dancer's box.

    A support point is skipped if it lands within 8% of the box height of an
    existing click (no value in tracking the same pixel twice). Pure."""
    out = list(clicks)
    for name, (bx, by, bw, bh) in key_boxes.items():
        existing = [(c.x, c.y) for c in out if c.name == name]
        cx = min(max(bx + bw // 2, 0), video_meta.width - 1)
        # A box whose centre line is far from the user's click probably spans
        # two overlapping people — its centre may be on the wrong dancer.
        xs = sorted(ex for ex, _ in existing)
        if xs and abs(xs[len(xs) // 2] - cx) > 0.3 * bw:
            continue
        for frac in _SUPPORT_POINT_FRACTIONS:
            py = min(max(int(round(by + frac * bh)), 0), video_meta.height - 1)
            if any(abs(ex - cx) < 0.08 * bh and abs(ey - py) < 0.08 * bh for ex, ey in existing):
                continue
            out.append(DancerClick(name=name, x=cx, y=py))
            existing.append((cx, py))
    return out


def process_video_with_cotracker(
    *,
    job_id: str,
    video_path: Path,
    clicks: list[DancerClick],
    key_frame: int,
    progress_callback: Callable[[int, int], None],
    stage_corners: list[list[int]] | None = None,
) -> PositionsResult:
    """Run the CoTracker click-to-track pipeline end to end."""
    video_meta = inspect_video_file(video_path)
    if not clicks:
        raise ValueError("CoTracker pipeline requires at least one click")

    homography = compute_stage_homography(
        stage_corners, (video_meta.width, video_meta.height)
    ) if stage_corners else None
    if stage_corners and homography is None:
        logger.warning(
            "CoTracker: stage_corners provided but homography was unusable "
            "(degenerate quad); falling back to camera-space view."
        )

    settings = get_settings()
    detector = None
    key_boxes: dict[str, tuple[int, int, int, int]] = {}
    if settings.cotracker_foot_assist:
        detector = YoloPersonDetector(
            model_name=settings.yolo_model_name,
            confidence_threshold=settings.yolo_confidence_threshold,
            iou_threshold=settings.yolo_iou_threshold,
            image_size=settings.yolo_image_size,
            max_detections=settings.yolo_max_detections,
            device=settings.yolo_device,
            tracker_config=settings.tracker_config,
        )
        try:
            key_boxes = _key_frame_boxes(video_path, key_frame, clicks, detector)
            logger.info(
                "CoTracker: matched %d/%d dancers to a person box at key frame %d",
                len(key_boxes), len(unique_click_names(clicks)), key_frame,
            )
        except Exception:
            logger.exception("key-frame person detection failed; continuing without it")
            key_boxes = {}

    track_clicks = list(clicks)
    if settings.cotracker_auto_points and key_boxes:
        track_clicks = add_support_points(clicks, key_boxes, video_meta)
        logger.info(
            "CoTracker: added %d auto support points (%d user clicks)",
            len(track_clicks) - len(clicks), len(clicks),
        )
    click_foot_offsets = [
        (key_boxes[c.name][1] + key_boxes[c.name][3] - c.y) if c.name in key_boxes else None
        for c in track_clicks
    ]

    click_payload = [
        {"name": c.name, "key_frame": key_frame, "x": c.x, "y": c.y} for c in track_clicks
    ]

    client = build_client()
    logger.info(
        "CoTracker pipeline: job=%s frames=%d clicks=%d key_frame=%d calibrated=%s",
        job_id, video_meta.frame_count, len(track_clicks), key_frame, homography is not None,
    )
    progress_callback(0, max(video_meta.frame_count, 1))

    t0 = time.perf_counter()
    tracks = client.track_video(video_path, click_payload)
    logger.info(
        "CoTracker returned %d point tracks in %.1fs",
        len(tracks), time.perf_counter() - t0,
    )

    foot_anchors = None
    if detector is not None:
        try:
            rep = representative_points(tracks, [c.name for c in track_clicks])
            initial_offsets: dict[str, tuple[int, int]] = {}
            for name, (bx, by, bw, bh) in key_boxes.items():
                series = rep.get(name)
                pt = series[key_frame] if series and key_frame < len(series) else None
                if pt is not None:
                    initial_offsets[name] = (bx + bw // 2 - pt[0], by + bh - pt[1])
            t1 = time.perf_counter()
            foot_anchors = compute_foot_anchors(
                video_path,
                rep,
                detector=detector,
                sample_every=settings.foot_assist_sample_every,
                initial_offsets=initial_offsets,
            )
            logger.info("foot assist done in %.1fs", time.perf_counter() - t1)
        except Exception:
            logger.exception(
                "foot assist failed — falling back to tracked points for anchors"
            )
            foot_anchors = None

    frames = tracked_points_to_frames(
        tracks, track_clicks, video_meta, homography, foot_anchors, click_foot_offsets
    )
    if homography is not None:
        apply_stage_rescue(frames)

    dancers_per_frame_total = sum(len(f.dancers) for f in frames)
    max_dancers_in_frame = max((len(f.dancers) for f in frames), default=0)
    frames_with_detections = sum(1 for f in frames if f.dancers)

    progress_callback(video_meta.frame_count, max(video_meta.frame_count, 1))

    calibrated = homography is not None
    formations = finalize_frames(
        frames,
        fps=video_meta.fps,
        refit_y=not calibrated,
        dedup=False,
        smooth=True,
    )

    expected = len(unique_click_names(clicks))
    frames_below_expected = sum(1 for f in frames if len(f.dancers) < expected)
    frames_meeting_expected = sum(1 for f in frames if len(f.dancers) >= expected)

    total_frames = len(frames)
    coord_note = (
        "perspective-corrected top-down floor coords from user-marked stage corners"
        if calibrated
        else (
            "auto-fit top-down proxy: y axis stretched to the observed anchor band "
            "(10th-90th percentile); not floor-plane calibrated"
        )
    )
    return PositionsResult(
        job_id=job_id,
        video=video_meta,
        coordinate_space=CoordinateSpaceMetadata(
            image_anchor_px="pixel anchor point in the source frame",
            normalized_stage_proxy=coord_note,
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
        stage_calibrated=calibrated,
    )
