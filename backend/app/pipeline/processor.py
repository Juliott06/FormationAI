from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable


logger = logging.getLogger("uvicorn.error")


from app.core.config import get_settings
from app.pipeline.debug_video import DebugVideoRenderer
from app.pipeline.pose import normalize_stage_proxy
from app.pipeline.appearance import crop_outfit_histogram, merge_by_appearance
from app.pipeline.labels import (
    assign_unlabeled_to_anchors,
    build_cooccurring_pairs as _labels_build_cooccurring,
)
from app.pipeline.named_matcher import match_names_to_yolo
from app.pipeline.templates import (
    fit_template,
    formation_spread,
    generate_templates_for_count,
)
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


# A 2-D shape (V, rows, grid, triangle) fitted with one axis squashed to
# under this extent (stage units) has degenerated into a line — e.g. a V with
# zero depth IS a line, and won on noise over the real "Line" template.
_MIN_2D_EXTENT = 0.06


def _collapsed_2d_shape(template_points, snapped) -> bool:
    t_range = template_points.max(axis=0) - template_points.min(axis=0)
    if (t_range < 1e-9).any():
        return False  # genuinely 1-D template (Line / Vertical line)
    s_range = snapped.max(axis=0) - snapped.min(axis=0)
    return bool((s_range < _MIN_2D_EXTENT).any())


def _snap_formations_to_templates(
    formations: list[Formation],
    threshold: float,
    relative_threshold: float = 0.25,
) -> None:
    """Snap each formation to its best-fitting named shape when the fit is
    good both absolutely (rmse <= threshold, stage units) and relative to the
    formation's own size (rmse / spread <= relative_threshold). The relative
    test stops a tight cluster from "matching" every template — with only an
    absolute threshold, any formation smaller than ~0.1 stage units snapped
    to whatever shape came out first."""
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
            if _collapsed_2d_shape(tmpl.points, snapped):
                continue
            if rmse < best_rmse:
                best_rmse = rmse
                best_snapped = snapped
                best_name = tmpl.name
        spread = formation_spread(observed)
        if (
            best_snapped is not None
            and best_rmse <= threshold
            and spread > 1e-6
            and best_rmse / spread <= relative_threshold
        ):
            for i, d in enumerate(formation.dancers):
                d.x = round(float(best_snapped[i, 0]), 6)
                d.y = round(float(best_snapped[i, 1]), 6)
            formation.shape_name = best_name
            logger.info(
                "formation %d snapped to %s (rmse=%.3f, relative=%.2f)",
                formation.index,
                best_name,
                best_rmse,
                best_rmse / spread,
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


# Fastest believable stage movement, canvas px/second (800x450 canvas): about
# half the stage width per second, a dancer sprinting to a new spot.
_MAX_STAGE_SPEED = 350.0
# Glitches shorter than about half this window are removed by the rolling
# median; real moves to a new spot (monotonic) pass through it intact.
_SPIKE_WINDOW_SEC = 1.0


def _rolling_median(vals: list[float], half: int) -> list[float]:
    out = []
    for i in range(len(vals)):
        w = sorted(vals[max(0, i - half) : i + half + 1])
        out.append(w[len(w) // 2])
    return out


def _speed_limited(xs: list[float], ys: list[float], max_step: float) -> tuple[list[float], list[float]]:
    """Clamp per-frame movement (canvas px) to max_step, run forward and
    backward and averaged so a clamp doesn't drag the path late."""

    def one_pass(order: list[int]) -> tuple[list[float], list[float]]:
        ox, oy = list(xs), list(ys)
        prev = order[0]
        for i in order[1:]:
            dx = (ox[i] - ox[prev]) * 800.0
            dy = (oy[i] - oy[prev]) * 450.0
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_step:
                k = max_step / dist
                ox[i] = ox[prev] + (ox[i] - ox[prev]) * k
                oy[i] = oy[prev] + (oy[i] - oy[prev]) * k
            prev = i
        return ox, oy

    n = len(xs)
    fx, fy = one_pass(list(range(n)))
    bx, by = one_pass(list(range(n - 1, -1, -1)))
    return [(a + b) / 2 for a, b in zip(fx, bx)], [(a + b) / 2 for a, b in zip(fy, by)]


def smooth_trajectories(
    frames: list[FramePositions], *, fps: float, window_sec: float = 0.5
) -> None:
    """Make each dancer's stage path physically plausible, in place.

    Per contiguous run of frames where the dancer is present:
    1. ~1s rolling median — removes glitches that go out and come back, e.g.
       a back-row dancer's dot shooting to the back for a few frames while
       someone passes in front of her;
    2. speed cap (_MAX_STAGE_SPEED) — nobody crosses half the stage in a
       fraction of a second;
    3. centred moving average of ~window_sec — removes residual jitter.
    anchor_px is left raw."""
    fps = max(fps, 1.0)
    half = max(int(round(window_sec * fps / 2)), 1)
    med_half = max(int(round(_SPIKE_WINDOW_SEC * fps / 2)), 1)
    max_step = _MAX_STAGE_SPEED / fps
    runs: dict[int, list[list[DancerPosition]]] = {}
    last_seen: dict[int, int] = {}
    for fi, f in enumerate(frames):
        for d in f.dancers:
            if last_seen.get(d.id) == fi - 1:
                runs[d.id][-1].append(d)
            else:
                runs.setdefault(d.id, []).append([d])
            last_seen[d.id] = fi

    def _box(vals: list[float]) -> list[float]:
        prefix = [0.0]
        for v in vals:
            prefix.append(prefix[-1] + v)
        out = []
        for i in range(len(vals)):
            lo, hi = max(0, i - half), min(len(vals), i + half + 1)
            out.append((prefix[hi] - prefix[lo]) / (hi - lo))
        return out

    for dancer_runs in runs.values():
        for run in dancer_runs:
            if len(run) < 3:
                continue
            xs = _rolling_median([d.x for d in run], med_half)
            ys = _rolling_median([d.y for d in run], med_half)
            xs, ys = _speed_limited(xs, ys, max_step)
            xs, ys = _box(xs), _box(ys)
            for d, x, y in zip(run, xs, ys):
                d.x = round(x, 6)
                d.y = round(y, 6)


def finalize_frames(
    frames: list[FramePositions],
    *,
    fps: float,
    refit_y: bool = True,
    dedup: bool = True,
    smooth: bool = False,
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

    dedup: drop near-coincident dancers from formations. Only for automatic
    tracking, where two IDs on one spot are usually one person split in two.
    Click-tracked dancers are distinct people by construction — deduping them
    made a dancer silently vanish from the formation whenever two stood close
    (and the template fit then ran on the wrong dancer count).

    smooth: temporally smooth each dancer's stage position (see
    smooth_trajectories). Done after the y refit, which recomputes y from the
    raw anchors and would otherwise discard it."""
    settings = get_settings()
    _interpolate_missing_dancers(frames, settings.interpolation_max_gap_frames)
    if refit_y:
        _refit_stage_y(frames)
    if smooth:
        smooth_trajectories(frames, fps=fps)
    formations = _segment_formations(
        frames,
        fps=fps,
        movement_threshold=settings.formation_movement_threshold,
        smoothing_window=settings.formation_smoothing_window,
        min_duration_sec=settings.formation_min_duration_sec,
    )
    formations = _split_by_position_change(
        formations,
        frames,
        window=settings.formation_split_window_frames,
        split_threshold=settings.formation_split_threshold,
    )
    _snap_formations_to_grid(formations, settings.formation_snap_grid_step)
    if dedup:
        _dedup_close_dancers_per_formation(formations, settings.formation_dedup_distance)
    _snap_formations_to_templates(
        formations,
        settings.formation_template_snap_threshold,
        settings.formation_template_snap_relative_threshold,
    )
    return formations


def _rebuild_positions_result(
    base: PositionsResult, new_frames: list[FramePositions]
) -> PositionsResult:
    # If the job was calibrated, x,y are already true top-down coords — don't
    # re-apply the perspective-compensation stretch.
    unique_ids = {d.id for f in new_frames for d in f.dancers}
    expected = base.summary.expected_dancer_count
    click_tracked = expected is not None and len(unique_ids) <= expected
    new_formations = finalize_frames(
        new_frames,
        fps=base.video.fps,
        refit_y=not base.stage_calibrated,
        # Fragmented automatic tracks still need dedup; a click-tracked job
        # (one ID per expected dancer) must keep every dancer.
        dedup=not click_tracked,
        smooth=click_tracked,
    )

    counts = [len(f.dancers) for f in new_frames]
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


# Movement is measured on the STAGE canvas (800x450 px, the size every view
# draws) as each dancer's displacement across a short window, relative to the
# group and normalized by the formation's size. Previously it was image-pixel motion of the anchor per frame:
# far-away dancers move only a few image pixels per metre, so a whole back-row
# walk could fall under the threshold, while per-frame anchor jitter (~4px on
# synthetic tests) sat right under it. The windowed stage measure is
# depth-independent and the window cancels frame-to-frame jitter.
_STAGE_CANVAS_W = 800.0
_STAGE_CANVAS_H = 450.0
_MOVEMENT_WINDOW_SEC = 0.17  # half-width of the displacement window
_NOISE_FLOOR_MARGIN = 1.8
_GROUP_DRIFT_WEIGHT = 0.35
_MAX_THRESHOLD_FRACTION_OF_P90 = 0.4


def _segment_formations(
    frames: list[FramePositions],
    fps: float,
    *,
    movement_threshold: float,
    smoothing_window: int,
    min_duration_sec: float,
) -> list[Formation]:
    """Split the clip into held formations: runs of frames whose average
    dancer speed RELATIVE TO THE GROUP (shared drift down-weighted), in
    formation sizes per second, stays under movement_threshold."""
    if len(frames) < 2:
        return []

    fps = max(fps, 1.0)
    k = max(int(round(_MOVEMENT_WINDOW_SEC * fps)), 1)
    by_frame = [{d.id: (d.x, d.y) for d in f.dancers} for f in frames]
    # Speeds are divided by the clip's typical formation size (median RMS
    # radius in canvas px), making the threshold independent of how zoomed
    # the stage is: the same choreography mapped through user corners vs the
    # auto floor calibration differed ~3x in canvas px/s, and a fixed px
    # threshold fragmented the zoomed one.
    spreads: list[float] = []
    for pos in by_frame:
        if len(pos) < 2:
            continue
        mx = sum(x for x, _ in pos.values()) / len(pos)
        my = sum(y for _, y in pos.values()) / len(pos)
        spreads.append(
            (sum(((x - mx) * _STAGE_CANVAS_W) ** 2 + ((y - my) * _STAGE_CANVAS_H) ** 2
                 for x, y in pos.values()) / len(pos)) ** 0.5
        )
    spreads.sort()
    size = max(spreads[len(spreads) // 2], 20.0) if spreads else 100.0
    raw_movement: list[float] = []
    for i in range(len(frames)):
        lo_i, hi_i = max(0, i - k), min(len(frames) - 1, i + k)
        span = hi_i - lo_i
        if span == 0:
            raw_movement.append(float("inf"))
            continue
        before, after = by_frame[lo_i], by_frame[hi_i]
        common = [did for did in after if did in before]
        if not common:
            raw_movement.append(float("inf"))
            continue
        # Down-weight the group's shared drift: dancers grooving forward or
        # sideways together (constant in real choreography) shouldn't break a
        # formation, but the whole shape travelling across the stage still
        # should, so drift counts at _GROUP_DRIFT_WEIGHT instead of fully.
        disp = [
            ((after[did][0] - before[did][0]) * _STAGE_CANVAS_W,
             (after[did][1] - before[did][1]) * _STAGE_CANVAS_H)
            for did in common
        ]
        mx = sum(dx for dx, _ in disp) / len(disp) if len(disp) > 1 else 0.0
        my = sum(dy for _, dy in disp) / len(disp) if len(disp) > 1 else 0.0
        drift = (mx * mx + my * my) ** 0.5 * _GROUP_DRIFT_WEIGHT
        speeds = [
            (((dx - mx) ** 2 + (dy - my) ** 2) ** 0.5 + drift) * fps / span / size
            for dx, dy in disp
        ]
        raw_movement.append(sum(speeds) / len(speeds))

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
            f"min={finite[0]:.2f} "
            f"p25={finite[min(int(0.25 * n), n - 1)]:.2f} "
            f"p50={finite[min(int(0.50 * n), n - 1)]:.2f} "
            f"p75={finite[min(int(0.75 * n), n - 1)]:.2f} "
            f"p90={finite[min(int(0.90 * n), n - 1)]:.2f} "
            f"max={finite[-1]:.2f}"
        )

    # Adapt to this clip's noise floor: the quietest stretches are holds, so
    # their speed is pure jitter (tracking noise, grooving in place). The
    # threshold rises to clear it by a margin, capped well below real moves.
    # A fixed threshold flickered on/off and chopped holds into pieces when a
    # clip's jitter sat near it (e.g. a stage zoomed in on the dancers).
    finite_sorted = sorted(v for v in smoothed if v != float("inf"))
    if len(finite_sorted) >= 20:
        p20 = finite_sorted[int(0.2 * len(finite_sorted))]
        p90 = finite_sorted[int(0.9 * len(finite_sorted))]
        adaptive = min(_NOISE_FLOOR_MARGIN * p20, _MAX_THRESHOLD_FRACTION_OF_P90 * p90)
        movement_threshold = max(movement_threshold, adaptive)
    below = sum(1 for v in smoothed if v < movement_threshold)
    logger.info("formation movement raw      (sizes/s): %s", _stats(raw_movement))
    logger.info(
        "formation movement smoothed (sizes/s): %s | %d/%d below threshold=%.2f",
        _stats(smoothed),
        below,
        len(smoothed),
        movement_threshold,
    )

    min_frame_count = max(int(round(fps * min_duration_sec)), 1)
    formations: list[Formation] = []
    i = 0
    while i < len(smoothed):
        if smoothed[i] >= movement_threshold:
            i += 1
            continue
        start = i
        while i < len(smoothed) and smoothed[i] < movement_threshold:
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
        frames, fps=video_meta.fps
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
