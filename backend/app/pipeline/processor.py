from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger("uvicorn.error")

_MAX_REASONABLE_DISPLACEMENT_PX = 60.0

from app.core.config import get_settings
from app.pipeline.debug_video import DebugVideoRenderer
from app.pipeline.pose import normalize_stage_proxy
from app.pipeline.appearance import crop_outfit_histogram, merge_by_appearance
from app.pipeline.labels import (
    assign_unlabeled_to_anchors,
    build_cooccurring_pairs as _labels_build_cooccurring,
)
from app.pipeline.named_matcher import match_names_to_yolo
from app.pipeline.templates import fit_template, generate_templates_for_count
from app.pipeline.types import TrackedDetection
from app.pipeline.yolo_detector import YoloPersonDetector
from app.schemas.jobs import (
    CoordinateSpaceMetadata,
    DancerPosition,
    DancerRosterEntry,
    DetectionSummary,
    Formation,
    FormationDancerPosition,
    FramePositions,
    PositionsResult,
    VideoMetadata,
)
from app.services.locate_anything import (
    BBoxPx,
    LocateAnythingError,
    build_client as build_locateanything_client,
)


def _cap_to_expected_count(
    tracked: list[TrackedDetection], cap: int | None
) -> list[TrackedDetection]:
    if cap is None or len(tracked) <= cap:
        return tracked
    return sorted(tracked, key=lambda t: t.confidence, reverse=True)[:cap]


_STAGE_Y_MIN = 0.25
_STAGE_Y_MAX = 0.75


def _aggregate_formation_dancers(
    frames: list[FramePositions], start_idx: int, end_idx: int
) -> list[FormationDancerPosition]:
    sums: dict[int, list[float]] = {}
    for f_idx in range(start_idx, end_idx + 1):
        for d in frames[f_idx].dancers:
            entry = sums.setdefault(d.id, [0.0, 0.0, 0])
            entry[0] += d.x
            entry[1] += d.y
            entry[2] += 1
    return [
        FormationDancerPosition(
            id=did, x=round(sx / count, 6), y=round(sy / count, 6)
        )
        for did, (sx, sy, count) in sorted(sums.items())
    ]


def _make_formation(
    index: int, frames: list[FramePositions], start_idx: int, end_idx: int
) -> Formation:
    return Formation(
        index=index,
        start_frame=frames[start_idx].frame,
        end_frame=frames[end_idx].frame,
        start_time_sec=frames[start_idx].timestamp_sec,
        end_time_sec=frames[end_idx].timestamp_sec,
        duration_sec=round(
            frames[end_idx].timestamp_sec - frames[start_idx].timestamp_sec, 3
        ),
        dancers=_aggregate_formation_dancers(frames, start_idx, end_idx),
    )


def _avg_position_distance(
    left: list[FormationDancerPosition],
    right: list[FormationDancerPosition],
) -> float:
    left_map = {d.id: (d.x, d.y) for d in left}
    right_map = {d.id: (d.x, d.y) for d in right}
    common = set(left_map) & set(right_map)
    if not common:
        return 0.0
    total = 0.0
    for did in common:
        lx, ly = left_map[did]
        rx, ry = right_map[did]
        total += ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5
    return total / len(common)


def _snap_formations_to_grid(
    formations: list[Formation], grid_step: float
) -> None:
    if grid_step <= 0:
        return
    for formation in formations:
        for d in formation.dancers:
            d.x = round(round(d.x / grid_step) * grid_step, 6)
            d.y = round(round(d.y / grid_step) * grid_step, 6)


def _snap_formations_to_templates(
    formations: list[Formation], threshold: float
) -> None:
    if threshold <= 0:
        return
    import numpy as np

    for formation in formations:
        n = len(formation.dancers)
        if n < 3:
            continue
        observed = np.array([[d.x, d.y] for d in formation.dancers])
        templates = generate_templates_for_count(n)
        best_rmse = float("inf")
        best_snapped = None
        best_name = None
        for tmpl in templates:
            snapped, rmse = fit_template(tmpl, observed)
            if rmse < best_rmse:
                best_rmse = rmse
                best_snapped = snapped
                best_name = tmpl.name
        if best_snapped is not None and best_rmse <= threshold:
            for i, d in enumerate(formation.dancers):
                d.x = round(float(best_snapped[i, 0]), 6)
                d.y = round(float(best_snapped[i, 1]), 6)
            formation.shape_name = best_name
            logger.info(
                "formation %d snapped to %s (rmse=%.3f)",
                formation.index,
                best_name,
                best_rmse,
            )


def _dedup_close_dancers_per_formation(
    formations: list[Formation], merge_distance: float
) -> None:
    if merge_distance <= 0:
        return
    for formation in formations:
        kept: list[FormationDancerPosition] = []
        for candidate in sorted(formation.dancers, key=lambda d: d.id):
            duplicate = False
            for existing in kept:
                dx = candidate.x - existing.x
                dy = candidate.y - existing.y
                if (dx * dx + dy * dy) ** 0.5 < merge_distance:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(candidate)
        formation.dancers = kept


def apply_stage_rescue(frames: list[FramePositions]) -> None:
    """Fix calibrated stage coords that fell outside the canvas.

    project_to_stage is unclamped, so a badly marked floor quad (e.g. one that
    doesn't contain the ground under the dancers) yields coords outside [0,1].
    This computes one uniform-scale affine over ALL positions in the clip and
    applies it, then clamps — dancers stay in true proportion instead of piling
    up on a canvas edge. No-op when the quad was marked well."""
    from app.pipeline.homography import compute_rescue_affine

    points = [(d.x, d.y) for f in frames for d in f.dancers]
    if not points:
        return
    s, ox, oy = compute_rescue_affine(points)
    if s == 1.0 and ox == 0.0 and oy == 0.0:
        # Still clamp: individual outliers within an otherwise-inside set.
        for f in frames:
            for d in f.dancers:
                d.x = round(min(max(d.x, 0.0), 1.0), 6)
                d.y = round(min(max(d.y, 0.0), 1.0), 6)
        return
    logger.info(
        "stage rescue applied: scale=%.3f offset=(%.3f, %.3f) — the marked floor "
        "quad did not contain all dancer positions",
        s, ox, oy,
    )
    for f in frames:
        for d in f.dancers:
            d.x = round(min(max(s * d.x + ox, 0.0), 1.0), 6)
            d.y = round(min(max(s * d.y + oy, 0.0), 1.0), 6)


def finalize_frames(
    frames: list[FramePositions],
    *,
    fps: float,
    refit_y: bool = True,
    frame_height: int = 1080,
) -> "list[Formation]":
    """Shared post-processing tail: fill detection gaps, refit stage Y, segment
    into formations, and clean the formations up.

    Every producer of FramePositions — the YOLO pipeline, the SAM2/CoTracker
    click pipelines, and the merge/swap/label rebuild path — MUST call this
    single function so all outputs go through identical stages. (These tails
    were previously copy-pasted per caller and drifted: the YOLO path lost
    template snapping and the rebuild path lost the stage-Y refit.)

    refit_y: the vertical-stretch heuristic that compensates for camera
    perspective. Pass False when x,y are already true top-down floor coords
    (a stage homography was applied) — refitting would re-distort them.

    frame_height: source video height, used to normalize movement so the
    formation threshold means the same thing at every resolution/fps."""
    settings = get_settings()
    _interpolate_missing_dancers(frames, settings.interpolation_max_gap_frames)
    if refit_y:
        _refit_stage_y(frames)
    formations = _segment_formations(
        frames,
        fps=fps,
        movement_threshold_px=settings.formation_movement_threshold_px,
        smoothing_window=settings.formation_smoothing_window,
        min_duration_sec=settings.formation_min_duration_sec,
        frame_height=frame_height,
    )
    formations = _split_by_position_change(
        formations,
        frames,
        window=settings.formation_split_window_frames,
        split_threshold=settings.formation_split_threshold,
    )
    _snap_formations_to_grid(formations, settings.formation_snap_grid_step)
    _dedup_close_dancers_per_formation(formations, settings.formation_dedup_distance)
    _snap_formations_to_templates(formations, settings.formation_template_snap_threshold)
    return formations


def _rebuild_positions_result(
    base: PositionsResult, new_frames: list[FramePositions]
) -> PositionsResult:
    # If the job was calibrated, x,y are already true top-down coords — don't
    # re-apply the perspective-compensation stretch.
    new_formations = finalize_frames(
        new_frames,
        fps=base.video.fps,
        refit_y=not base.stage_calibrated,
        frame_height=base.video.height,
    )

    unique_ids = {d.id for f in new_frames for d in f.dancers}
    counts = [len(f.dancers) for f in new_frames]
    expected = base.summary.expected_dancer_count
    new_summary = DetectionSummary(
        expected_dancer_count=expected,
        unique_track_ids=len(unique_ids),
        max_dancers_in_frame=max(counts, default=0),
        average_dancers_per_frame=round(sum(counts) / max(len(counts), 1), 3),
        frames_with_detections=sum(1 for c in counts if c > 0),
        frames_below_expected=(
            sum(1 for c in counts if c < expected) if expected is not None else None
        ),
        frames_meeting_expected=(
            sum(1 for c in counts if c >= expected) if expected is not None else None
        ),
    )

    return PositionsResult(
        job_id=base.job_id,
        video=base.video,
        coordinate_space=base.coordinate_space,
        summary=new_summary,
        frames=new_frames,
        formations=new_formations,
        stage_calibrated=base.stage_calibrated,
    )


def apply_id_merge(
    positions: PositionsResult, *, keep_id: int, remove_id: int
) -> PositionsResult:
    new_frames: list[FramePositions] = []
    for frame in positions.frames:
        seen: set[int] = set()
        new_dancers: list[DancerPosition] = []
        for d in frame.dancers:
            mapped_id = keep_id if d.id == remove_id else d.id
            if mapped_id in seen:
                continue
            seen.add(mapped_id)
            if d.id == remove_id:
                new_dancers.append(d.model_copy(update={"id": keep_id}))
            else:
                new_dancers.append(d)
        new_frames.append(
            FramePositions(
                frame=frame.frame,
                timestamp_sec=frame.timestamp_sec,
                dancers=new_dancers,
            )
        )
    return _rebuild_positions_result(positions, new_frames)


def apply_labels(
    positions: PositionsResult,
    histograms_raw: dict[int, list[float]],
    labels: dict[int, str],
) -> PositionsResult:
    """Apply labels as anchors: rename unlabeled tracks to whichever labeled
    track they most resemble (subject to cooccurring guard)."""
    import numpy as np

    histograms = {tid: np.array(vec, dtype=np.float32) for tid, vec in histograms_raw.items()}
    cooccurring = _labels_build_cooccurring(positions.frames)
    rename = assign_unlabeled_to_anchors(histograms, labels, cooccurring)

    new_frames: list[FramePositions] = []
    for frame in positions.frames:
        seen: set[int] = set()
        new_dancers: list[DancerPosition] = []
        for d in frame.dancers:
            mapped_id = d.id
            while mapped_id in rename:
                mapped_id = rename[mapped_id]
            if mapped_id in seen:
                continue
            seen.add(mapped_id)
            if mapped_id != d.id:
                new_dancers.append(d.model_copy(update={"id": mapped_id}))
            else:
                new_dancers.append(d)
        new_frames.append(
            FramePositions(
                frame=frame.frame,
                timestamp_sec=frame.timestamp_sec,
                dancers=new_dancers,
            )
        )

    return _rebuild_positions_result(positions, new_frames)


def apply_id_swap(
    positions: PositionsResult, *, id_a: int, id_b: int, from_frame: int
) -> PositionsResult:
    swap_map = {id_a: id_b, id_b: id_a}
    new_frames: list[FramePositions] = []
    for frame in positions.frames:
        new_dancers: list[DancerPosition] = []
        for d in frame.dancers:
            if frame.frame >= from_frame and d.id in swap_map:
                new_dancers.append(d.model_copy(update={"id": swap_map[d.id]}))
            else:
                new_dancers.append(d)
        new_frames.append(
            FramePositions(
                frame=frame.frame,
                timestamp_sec=frame.timestamp_sec,
                dancers=new_dancers,
            )
        )
    return _rebuild_positions_result(positions, new_frames)


def _split_by_position_change(
    formations: list[Formation],
    frames: list[FramePositions],
    *,
    window: int,
    split_threshold: float,
) -> list[Formation]:
    if split_threshold <= 0 or window <= 0:
        return formations

    frame_to_idx = {f.frame: i for i, f in enumerate(frames)}
    result: list[Formation] = []
    for formation in formations:
        start_idx = frame_to_idx.get(formation.start_frame)
        end_idx = frame_to_idx.get(formation.end_frame)
        if (
            start_idx is None
            or end_idx is None
            or (end_idx - start_idx + 1) < 2 * window
        ):
            result.append(formation)
            continue

        best_split = -1
        best_score = 0.0
        for split_idx in range(start_idx + window - 1, end_idx - window + 1):
            left = _aggregate_formation_dancers(frames, start_idx, split_idx)
            right = _aggregate_formation_dancers(frames, split_idx + 1, end_idx)
            score = _avg_position_distance(left, right)
            if score > best_score:
                best_score = score
                best_split = split_idx

        if best_score >= split_threshold and best_split > 0:
            logger.info(
                "split formation %d at frame %d (avg dancer drift=%.3f, threshold=%.3f)",
                formation.index,
                frames[best_split].frame,
                best_score,
                split_threshold,
            )
            result.append(_make_formation(0, frames, start_idx, best_split))
            result.append(_make_formation(0, frames, best_split + 1, end_idx))
        else:
            result.append(formation)

    for i, f in enumerate(result):
        f.index = i
    return result


# Movement thresholds are expressed in px/frame at this reference format.
# Measured movement is normalized to it so the same physical motion reads the
# same number whether the upload is 720p@30fps or 1080p@24fps — otherwise the
# threshold silently tightens on smaller/faster-fps videos (a 720p/30fps clip
# produced ~54% of the px/frame of the 1080p/24fps clip the threshold was
# tuned on, merging a whole song into one formation).
_MOVEMENT_REF_HEIGHT = 1080.0
_MOVEMENT_REF_FPS = 24.0


def _segment_formations(
    frames: list[FramePositions],
    fps: float,
    *,
    movement_threshold_px: float,
    smoothing_window: int,
    min_duration_sec: float,
    frame_height: int = 1080,
) -> list[Formation]:
    if len(frames) < 2:
        return []

    norm = (_MOVEMENT_REF_HEIGHT / max(frame_height, 1)) * (
        max(fps, 1.0) / _MOVEMENT_REF_FPS
    )
    raw_movement: list[float] = [0.0]
    for i in range(1, len(frames)):
        prev_by_id = {d.id: d.anchor_px for d in frames[i - 1].dancers}
        distances: list[float] = []
        for d in frames[i].dancers:
            if d.id in prev_by_id:
                dx = d.anchor_px[0] - prev_by_id[d.id][0]
                dy = d.anchor_px[1] - prev_by_id[d.id][1]
                raw_dist = (dx * dx + dy * dy) ** 0.5 * norm
                distances.append(min(raw_dist, _MAX_REASONABLE_DISPLACEMENT_PX))
        raw_movement.append(sum(distances) / len(distances) if distances else float("inf"))

    half = max(smoothing_window // 2, 1)
    smoothed: list[float] = []
    for i in range(len(raw_movement)):
        lo = max(0, i - half)
        hi = min(len(raw_movement), i + half + 1)
        window = raw_movement[lo:hi]
        smoothed.append(sum(window) / len(window))

    def _stats(values: list[float]) -> str:
        finite = sorted(v for v in values if v != float("inf"))
        if not finite:
            return "n/a"
        n = len(finite)
        return (
            f"min={finite[0]:.1f} "
            f"p25={finite[min(int(0.25 * n), n - 1)]:.1f} "
            f"p50={finite[min(int(0.50 * n), n - 1)]:.1f} "
            f"p75={finite[min(int(0.75 * n), n - 1)]:.1f} "
            f"p90={finite[min(int(0.90 * n), n - 1)]:.1f} "
            f"max={finite[-1]:.1f}"
        )

    below = sum(1 for v in smoothed if v < movement_threshold_px)
    logger.info("formation movement raw      (px/frame): %s", _stats(raw_movement[1:]))
    logger.info(
        "formation movement smoothed (px/frame): %s | %d/%d below threshold=%.1f",
        _stats(smoothed),
        below,
        len(smoothed),
        movement_threshold_px,
    )

    min_frame_count = max(int(round(fps * min_duration_sec)), 1)
    formations: list[Formation] = []
    i = 0
    while i < len(smoothed):
        if smoothed[i] >= movement_threshold_px:
            i += 1
            continue
        start = i
        while i < len(smoothed) and smoothed[i] < movement_threshold_px:
            i += 1
        end = i - 1
        if (end - start + 1) < min_frame_count:
            continue
        formations.append(_make_formation(len(formations), frames, start, end))

    return formations


def _apply_id_renames(
    frames: list[FramePositions], rename: dict[int, int]
) -> None:
    if not rename:
        return
    for f in frames:
        seen: set[int] = set()
        kept: list[DancerPosition] = []
        for d in f.dancers:
            new_id = d.id
            while new_id in rename:
                new_id = rename[new_id]
            if new_id in seen:
                continue
            seen.add(new_id)
            d.id = new_id
            kept.append(d)
        f.dancers = kept


def _build_cooccurring_pairs(
    frames: list[FramePositions],
) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for f in frames:
        ids = [d.id for d in f.dancers]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] != ids[j]:
                    pairs.add(frozenset((ids[i], ids[j])))
    return pairs


def _recover_id_continuity(
    frames: list[FramePositions],
    *,
    max_gap_frames: int,
    max_distance: float,
    histograms_by_id: dict[int, Any] | None = None,
    appearance_threshold: float = 0.88,
    appearance_distance_boost: float = 3.0,
) -> int:
    if max_gap_frames <= 0 or max_distance <= 0 or len(frames) < 2:
        return 0

    track_info: dict[int, list[tuple[int, float, float]]] = {}
    for i, f in enumerate(frames):
        for d in f.dancers:
            track_info.setdefault(d.id, []).append((i, d.x, d.y))

    mean_hists: dict[int, Any] = {}
    if histograms_by_id:
        import numpy as np
        for tid, hists in histograms_by_id.items():
            valid = [h for h in hists if h is not None and getattr(h, "size", 0) > 0]
            if valid:
                mean_hists[tid] = np.mean(valid, axis=0)

    def _appearance_sim(a: int, b: int) -> float:
        ha = mean_hists.get(a)
        hb = mean_hists.get(b)
        if ha is None or hb is None:
            return 0.0
        import numpy as np
        denom = float(np.linalg.norm(ha) * np.linalg.norm(hb))
        if denom < 1e-9:
            return 0.0
        return float(np.dot(ha, hb) / denom)

    sorted_by_start = sorted(track_info.keys(), key=lambda tid: track_info[tid][0][0])
    rename: dict[int, int] = {}
    used_olds: set[int] = set()

    for new_id in sorted_by_start:
        new_first_frame, nfx, nfy = track_info[new_id][0]
        best_old: int | None = None
        best_score = float("inf")
        for old_id in track_info:
            if old_id == new_id or old_id in used_olds:
                continue
            old_last_frame, olx, oly = track_info[old_id][-1]
            gap = new_first_frame - old_last_frame
            if gap <= 0 or gap > max_gap_frames:
                continue
            dist = ((olx - nfx) ** 2 + (oly - nfy) ** 2) ** 0.5
            app_sim = _appearance_sim(old_id, new_id)
            effective_limit = max_distance
            if app_sim >= appearance_threshold:
                effective_limit = max_distance * appearance_distance_boost
            if dist > effective_limit:
                continue
            score = dist - app_sim * max_distance * 0.5 + gap * 0.0005
            if score < best_score:
                best_score = score
                best_old = old_id
        if best_old is not None:
            rename[new_id] = best_old
            used_olds.add(best_old)

    if not rename:
        return 0

    _apply_id_renames(frames, rename)

    logger.info("recovered %d ID continuit%s", len(rename), "y" if len(rename) == 1 else "ies")
    for new_id, old_id in rename.items():
        logger.info("  ID %d -> %d (re-acquired track)", new_id, old_id)
    return len(rename)


def _auto_fix_id_swaps(
    frames: list[FramePositions],
    *,
    jump_threshold: float,
    advantage_ratio: float,
    max_passes: int = 5,
) -> int:
    if jump_threshold <= 0 or advantage_ratio <= 0 or len(frames) < 2:
        return 0

    def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    total = 0
    for _pass in range(max_passes):
        applied = 0
        for i in range(1, len(frames)):
            prev_by_id = {d.id: (d.x, d.y) for d in frames[i - 1].dancers}
            cur_by_id = {d.id: (d.x, d.y) for d in frames[i].dancers}
            common = sorted(set(prev_by_id) & set(cur_by_id))
            jumpers = [
                did for did in common
                if _dist(prev_by_id[did], cur_by_id[did]) > jump_threshold
            ]
            if len(jumpers) < 2:
                continue

            best_pair: tuple[int, int] | None = None
            best_gain = 0.0
            for j in range(len(jumpers)):
                for k in range(j + 1, len(jumpers)):
                    a_id, b_id = jumpers[j], jumpers[k]
                    a_old, b_old = prev_by_id[a_id], prev_by_id[b_id]
                    a_new, b_new = cur_by_id[a_id], cur_by_id[b_id]
                    no_swap_d = _dist(a_old, a_new) + _dist(b_old, b_new)
                    swap_d = _dist(a_old, b_new) + _dist(b_old, a_new)
                    if swap_d > 0 and no_swap_d > advantage_ratio * swap_d:
                        gain = no_swap_d - swap_d
                        if gain > best_gain:
                            best_gain = gain
                            best_pair = (a_id, b_id)

            if best_pair is not None:
                a_id, b_id = best_pair
                swap_map = {a_id: b_id, b_id: a_id}
                for later_frame in frames[i:]:
                    for d in later_frame.dancers:
                        if d.id in swap_map:
                            d.id = swap_map[d.id]
                applied += 1
                logger.info(
                    "auto-swap at frame %d: IDs %d <-> %d (gain=%.3f)",
                    frames[i].frame, a_id, b_id, best_gain,
                )
        total += applied
        if applied == 0:
            break
    if total > 0:
        logger.info("auto-fixed %d ID swap(s) total", total)
    return total


def _detect_gradual_swaps(
    frames: list[FramePositions],
    *,
    proximity: float,
    advantage: float,
    min_overlap_frames: int = 30,
    max_passes: int = 4,
) -> int:
    if proximity <= 0 or advantage <= 0 or len(frames) < min_overlap_frames:
        return 0

    def _trajectory_length(points: dict[int, tuple[float, float]]) -> float:
        ordered = sorted(points.items())
        total = 0.0
        for i in range(1, len(ordered)):
            _, (x1, y1) = ordered[i - 1]
            _, (x2, y2) = ordered[i]
            total += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        return total

    total_swaps = 0
    for _pass in range(max_passes):
        tracks: dict[int, dict[int, tuple[float, float]]] = {}
        for i, f in enumerate(frames):
            for d in f.dancers:
                tracks.setdefault(d.id, {})[i] = (d.x, d.y)

        applied_this_pass = 0
        track_ids = sorted(tracks.keys())
        for j in range(len(track_ids)):
            for k in range(j + 1, len(track_ids)):
                a_id, b_id = track_ids[j], track_ids[k]
                a_pts, b_pts = tracks[a_id], tracks[b_id]
                common = sorted(set(a_pts) & set(b_pts))
                if len(common) < min_overlap_frames:
                    continue

                min_dist = float("inf")
                cross_frame = -1
                for f in common:
                    ax, ay = a_pts[f]
                    bx, by = b_pts[f]
                    d = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                    if d < min_dist:
                        min_dist = d
                        cross_frame = f

                if min_dist > proximity:
                    continue
                if cross_frame <= common[0] + 3 or cross_frame >= common[-1] - 3:
                    continue

                orig = _trajectory_length(a_pts) + _trajectory_length(b_pts)
                swapped_a = {f: (b_pts[f] if f > cross_frame else a_pts[f]) for f in a_pts if f <= cross_frame or f in b_pts}
                swapped_b = {f: (a_pts[f] if f > cross_frame else b_pts[f]) for f in b_pts if f <= cross_frame or f in a_pts}
                new_total = _trajectory_length(swapped_a) + _trajectory_length(swapped_b)

                if new_total > 0 and orig > advantage * new_total:
                    swap_map = {a_id: b_id, b_id: a_id}
                    for f_idx in range(cross_frame + 1, len(frames)):
                        for d in frames[f_idx].dancers:
                            if d.id in swap_map:
                                d.id = swap_map[d.id]
                    logger.info(
                        "gradual swap at frame %d: IDs %d <-> %d (orig=%.3f -> swap=%.3f)",
                        frames[cross_frame].frame, a_id, b_id, orig, new_total,
                    )
                    applied_this_pass += 1
                    break
            if applied_this_pass > 0:
                break

        total_swaps += applied_this_pass
        if applied_this_pass == 0:
            break

    if total_swaps > 0:
        logger.info("auto-fixed %d gradual swap(s)", total_swaps)
    return total_swaps


def _interpolate_missing_dancers(
    frames: list[FramePositions], max_gap: int
) -> None:
    if max_gap <= 0 or len(frames) < 3:
        return

    presences_by_id: dict[int, list[tuple[int, DancerPosition]]] = {}
    for i, f in enumerate(frames):
        for d in f.dancers:
            presences_by_id.setdefault(d.id, []).append((i, d))

    for did, presences in presences_by_id.items():
        for k in range(len(presences) - 1):
            prev_idx, prev_d = presences[k]
            next_idx, next_d = presences[k + 1]
            gap = next_idx - prev_idx - 1
            if gap <= 0 or gap > max_gap:
                continue
            span = next_idx - prev_idx
            for fill_idx in range(prev_idx + 1, next_idx):
                t = (fill_idx - prev_idx) / span
                bbox = [
                    int(round(prev_d.bbox[i] + t * (next_d.bbox[i] - prev_d.bbox[i])))
                    for i in range(4)
                ]
                anchor_px = [
                    int(round(prev_d.anchor_px[0] + t * (next_d.anchor_px[0] - prev_d.anchor_px[0]))),
                    int(round(prev_d.anchor_px[1] + t * (next_d.anchor_px[1] - prev_d.anchor_px[1]))),
                ]
                frames[fill_idx].dancers.append(
                    DancerPosition(
                        id=did,
                        bbox=bbox,
                        anchor_px=anchor_px,
                        x=round(prev_d.x + t * (next_d.x - prev_d.x), 6),
                        y=round(prev_d.y + t * (next_d.y - prev_d.y), 6),
                        confidence=round(min(prev_d.confidence, next_d.confidence), 4),
                    )
                )


def _refit_stage_y(frames: list[FramePositions]) -> None:
    anchor_ys = [d.anchor_px[1] for f in frames for d in f.dancers]
    if not anchor_ys:
        return
    sorted_ys = sorted(anchor_ys)
    n = len(sorted_ys)
    if n >= 10:
        lo_idx = int(n * 0.10)
        hi_idx = max(int(n * 0.90) - 1, lo_idx + 1)
        floor_top_px = sorted_ys[lo_idx]
        floor_bot_px = sorted_ys[hi_idx]
    else:
        floor_top_px = sorted_ys[0]
        floor_bot_px = sorted_ys[-1]
    floor_span = max(floor_bot_px - floor_top_px, 1)
    target_span = _STAGE_Y_MAX - _STAGE_Y_MIN
    for frame in frames:
        for dancer in frame.dancers:
            normalized = (dancer.anchor_px[1] - floor_top_px) / floor_span
            normalized = min(max(normalized, 0.0), 1.0)
            stretched = _STAGE_Y_MIN + normalized * target_span
            dancer.y = round(1.0 - stretched, 6)


def inspect_video_file(video_path: Path) -> VideoMetadata:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    duration_sec = (frame_count / fps) if fps > 0 else 0.0
    return VideoMetadata(
        filename=video_path.name,
        fps=round(fps, 3),
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=round(duration_sec, 3),
    )


def process_video(
    *,
    job_id: str,
    video_path: Path,
    progress_callback: Callable[[int, int], None],
    debug_video_path: Path | None = None,
    expected_dancer_count: int | None = None,
    roster: list[DancerRosterEntry] | None = None,
    histograms_out: dict[int, list[float]] | None = None,
) -> PositionsResult:
    settings = get_settings()
    video_meta = inspect_video_file(video_path)
    detector = YoloPersonDetector(
        model_name=settings.yolo_model_name,
        confidence_threshold=settings.yolo_confidence_threshold,
        iou_threshold=settings.yolo_iou_threshold,
        image_size=settings.yolo_image_size,
        max_detections=settings.yolo_max_detections,
        device=settings.yolo_device,
        tracker_config=settings.tracker_config,
    )

    la_enabled = bool(roster) and settings.locateanything_backend != "disabled"
    la_client = build_locateanything_client() if la_enabled else None
    roster_list: list[DancerRosterEntry] = roster or []
    roster_names: list[str] = [entry.name for entry in roster_list]
    roster_prompts: dict[str, str] = {entry.name: entry.hint for entry in roster_list}
    last_known_pos: dict[str, tuple[int, int]] = {}
    la_failures = 0
    if la_enabled:
        if expected_dancer_count is None:
            expected_dancer_count = len(roster_names)
        logger.info(
            "LocateAnything ENABLED for job %s — roster=%d, backend=%s, url=%s",
            job_id,
            len(roster_names),
            settings.locateanything_backend,
            settings.locateanything_url,
        )

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")

    frames: list[FramePositions] = []
    frame_index = 0
    all_track_ids: set[int] = set()
    dancers_per_frame_total = 0
    max_dancers_in_frame = 0
    frames_with_detections = 0
    frames_below_expected = 0
    frames_meeting_expected = 0
    histograms_by_id: dict[int, list] = {}
    appearance_enabled = settings.appearance_merge_threshold > 0
    debug_renderer = (
        DebugVideoRenderer(
            output_path=debug_video_path,
            fps=video_meta.fps,
            frame_size=(video_meta.width, video_meta.height),
        )
        if debug_video_path is not None
        else None
    )

    loop_start = time.perf_counter()
    last_log_count = 0
    last_log_time = loop_start

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = int(round((frame_index / max(video_meta.fps, 1.0)) * 1000))
            yolo_tracked = detector.track(frame, timestamp_ms)
            yolo_tracked = _cap_to_expected_count(yolo_tracked, expected_dancer_count)

            if la_enabled and la_client is not None:
                name_boxes: dict[str, BBoxPx] = {}
                ok_enc, jpeg = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85]
                )
                if ok_enc:
                    try:
                        name_boxes = la_client.identify(
                            bytes(jpeg),
                            video_meta.width,
                            video_meta.height,
                            roster_prompts,
                        )
                    except LocateAnythingError as exc:
                        la_failures += 1
                        if la_failures <= 3:
                            logger.warning(
                                "LocateAnything call failed at frame %d: %s",
                                frame_index,
                                exc,
                            )
                        if la_failures >= 10:
                            logger.error(
                                "LocateAnything failed %d times — disabling for rest of clip; "
                                "reverting to YOLO-only.",
                                la_failures,
                            )
                            la_enabled = False
                if la_enabled:
                    tracked, last_known_pos = match_names_to_yolo(
                        name_boxes,
                        yolo_tracked,
                        last_known_pos,
                        roster_names,
                    )
                else:
                    tracked = yolo_tracked
            else:
                tracked = yolo_tracked

            if (
                appearance_enabled
                and settings.appearance_sample_every > 0
                and frame_index % settings.appearance_sample_every == 0
            ):
                for item in tracked:
                    hist = crop_outfit_histogram(frame, item.bbox)
                    if hist is not None:
                        histograms_by_id.setdefault(item.track_id, []).append(hist)
            dancers = []
            for item in tracked:
                all_track_ids.add(item.track_id)
                x, y = normalize_stage_proxy(
                    item.anchor_px,
                    video_meta.width,
                    video_meta.height,
                )
                dancers.append(
                    DancerPosition(
                        id=item.track_id,
                        bbox=list(item.bbox),
                        anchor_px=list(item.anchor_px),
                        x=x,
                        y=y,
                        confidence=item.confidence,
                    )
                )

            dancer_count = len(dancers)
            dancers_per_frame_total += dancer_count
            max_dancers_in_frame = max(max_dancers_in_frame, dancer_count)
            if dancer_count > 0:
                frames_with_detections += 1
            if expected_dancer_count is not None:
                if dancer_count < expected_dancer_count:
                    frames_below_expected += 1
                else:
                    frames_meeting_expected += 1

            if debug_renderer is not None:
                debug_renderer.write_frame(frame, dancers)

            frames.append(
                FramePositions(
                    frame=frame_index,
                    timestamp_sec=round(frame_index / max(video_meta.fps, 1.0), 3),
                    dancers=dancers,
                )
            )
            frame_index += 1
            progress_callback(frame_index, video_meta.frame_count)

            if frame_index - last_log_count >= 50:
                now = time.perf_counter()
                fps = (frame_index - last_log_count) / max(now - last_log_time, 1e-6)
                logger.info(
                    "processing %d/%d frames | %.1f fps last batch | %.1fs elapsed",
                    frame_index,
                    video_meta.frame_count,
                    fps,
                    now - loop_start,
                )
                last_log_count = frame_index
                last_log_time = now
    finally:
        if debug_renderer is not None:
            debug_renderer.close()
        capture.release()

    if appearance_enabled and histograms_by_id:
        cooccurring = _build_cooccurring_pairs(frames)
        rename = merge_by_appearance(
            histograms_by_id,
            threshold=settings.appearance_merge_threshold,
            target_count=expected_dancer_count,
            cooccurring_pairs=cooccurring,
        )
        _apply_id_renames(frames, rename)
        if rename:
            logger.info("applied %d appearance-based merge(s)", len(rename))

    if histograms_out is not None and histograms_by_id:
        import numpy as np
        for tid, hists in histograms_by_id.items():
            valid = [h for h in hists if h is not None and getattr(h, "size", 0) > 0]
            if not valid:
                continue
            final_tid = tid
            histograms_out[final_tid] = np.mean(valid, axis=0).tolist()

    _recover_id_continuity(
        frames,
        max_gap_frames=settings.id_continuity_max_gap_frames,
        max_distance=settings.id_continuity_max_distance,
        histograms_by_id=histograms_by_id if appearance_enabled else None,
    )
    _auto_fix_id_swaps(
        frames,
        jump_threshold=settings.auto_swap_jump_threshold,
        advantage_ratio=settings.auto_swap_advantage_ratio,
    )
    _detect_gradual_swaps(
        frames,
        proximity=settings.gradual_swap_proximity,
        advantage=settings.gradual_swap_advantage,
    )
    formations = finalize_frames(
        frames, fps=video_meta.fps, frame_height=video_meta.height
    )

    total_frames = len(frames)
    return PositionsResult(
        job_id=job_id,
        video=video_meta,
        coordinate_space=CoordinateSpaceMetadata(
            image_anchor_px="pixel anchor point in the source frame",
            normalized_stage_proxy="auto-fit top-down proxy: y axis stretched to the observed anchor band (10th-90th percentile), occupying the middle 50% of the canvas; not floor-plane calibrated",
        ),
        summary=DetectionSummary(
            expected_dancer_count=expected_dancer_count,
            unique_track_ids=len(all_track_ids),
            max_dancers_in_frame=max_dancers_in_frame,
            average_dancers_per_frame=round(dancers_per_frame_total / max(total_frames, 1), 3),
            frames_with_detections=frames_with_detections,
            frames_below_expected=frames_below_expected if expected_dancer_count is not None else None,
            frames_meeting_expected=frames_meeting_expected if expected_dancer_count is not None else None,
        ),
        frames=frames,
        formations=formations,
    )
