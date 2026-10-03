"""Detection-anchored identity: keep each clicked dancer on the right PERSON.

CoTracker follows pixels. When a dancer is fully hidden behind someone, her
points have nothing to follow and latch onto whoever is in front — observed
on a real clip: two of Blonde2's three points stuck to Dark after a crossing,
and from then on Blonde2's dot followed Dark. Nothing noticed or recovered.

This pass runs a person detector on every frame and assigns detections to
dancers (Hungarian) using three cues together:

- motion: where the dancer should be given her recent movement (people don't
  teleport) — gated so a dancer can't jump onto someone across the stage;
- appearance: colour histograms of hair / top / pants, learned at the click
  frame and slowly updated on unoccluded frames;
- point votes: which dancer's tracked points fall inside the box.

Points that keep sitting inside ANOTHER dancer's assigned box are declared
defected and ignored from then on, so a latched point stops voting for (and
dragging) the wrong person. A dancer with no box this frame (hidden, or merged
into someone else's box) coasts on her remaining trusted points.

Outputs per-dancer foot anchors (box bottom; x from trusted torso points when
available, which unlike the box centre doesn't swing with extended arms), the
cleaned tracks, and unoccluded box samples for floor auto-calibration.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.pipeline.types import Detection
from app.services.cotracker import TrackedPoint


logger = logging.getLogger("uvicorn.error")

Point = tuple[int, int]
Box = tuple[int, int, int, int]

# Cost weights / gates (motion distances are in units of the dancer's box height).
# Motion is capped so it can't by itself lock a dancer onto the wrong person:
# a hard tight gate meant that once a dancer grabbed a neighbour's box, her
# real box looked "too far" forever (observed on a real clip). Appearance
# carries the most weight; it separates differently dressed dancers clearly.
_W_MOTION = 0.8
_MOTION_CAP = 1.2
_W_APPEAR = 2.5
_W_POINTS = 0.6
_W_SCALE = 0.3
_MAX_COST = 3.0
_MOTION_GATE = 1.0  # box heights, base
_MOTION_GATE_GROWTH = 0.04  # extra per frame the dancer has been lost
_MOTION_GATE_MAX = 2.0  # a hidden dancer reappears near where she vanished
_MAX_SPEED = 0.06  # box heights per frame (~1.8 body heights/s at 30fps)
_VEL_DAMP = 0.7  # per-frame velocity decay while coasting blind
_DEFECT_FRAMES = 8  # consecutive frames inside another dancer's box
_CLEAN_IOU = 0.08  # box overlapping others less than this = unoccluded
_TEMPLATE_RATE = 0.05


class _Detector(Protocol):
    def detect(self, frame_bgr: object) -> list[Detection]: ...


def appearance_feature(frame_bgr, box: Box):
    """Concatenated HSV histograms of head / top / legs bands of a person box
    (central 60% of the width, to keep background out). L1-normalized per
    band, so each band weighs the same."""
    import cv2
    import numpy as np

    h_img, w_img = frame_bgr.shape[:2]
    x, y, w, h = box
    x0 = max(int(x + 0.2 * w), 0)
    x1 = min(int(x + 0.8 * w), w_img)
    feats = []
    for a, b in ((0.0, 0.18), (0.18, 0.5), (0.5, 0.95)):
        y0 = max(int(y + a * h), 0)
        y1 = min(int(y + b * h), h_img)
        if x1 - x0 < 2 or y1 - y0 < 2:
            feats.append(np.full(72, 1.0 / 72, np.float32))
            continue
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 3, 3], [0, 180, 0, 256, 0, 256])
        hist = hist.flatten().astype(np.float32)
        feats.append(hist / max(float(hist.sum()), 1e-6))
    return np.concatenate(feats)


def appearance_distance(a, b) -> float:
    """Hellinger distance averaged over the 3 bands: 0 = identical, 1 = disjoint."""
    import numpy as np

    return float(np.sqrt(max(0.0, 1.0 - float(np.sqrt(a * b).sum()) / 3.0)))


def _clamp_vel(vx: float, vy: float, h: float) -> tuple[float, float]:
    """Limit predicted speed: a coasting estimate must not run away (observed:
    a hidden dancer's prediction flew off-screen and then grabbed boxes on the
    far side of the stage)."""
    lim = _MAX_SPEED * max(h, 1.0)
    sp = (vx * vx + vy * vy) ** 0.5
    if sp <= lim:
        return vx, vy
    return vx * lim / sp, vy * lim / sp


def _is_identity(m) -> bool:
    import numpy as np

    return bool(np.allclose(np.asarray(m), np.eye(3)))


def _invert(m) -> list[list[float]]:
    import numpy as np

    inv = np.linalg.inv(np.asarray(m, dtype=np.float64))
    return (inv / inv[2, 2]).tolist()


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _inside(px: float, py: float, box: Box, shrink: float = 0.05) -> bool:
    x, y, w, h = box
    return (x + shrink * w <= px <= x + (1 - shrink) * w) and (y <= py <= y + h)


@dataclass
class _Dancer:
    name: str
    point_idx: list[int]
    foot: tuple[float, float]
    vel: tuple[float, float] = (0.0, 0.0)
    h: float = 100.0
    template_key: object = None
    template: object = None
    lost: int = 0
    # point -> (dx, dy) from point to foot, measured when last matched
    offsets: dict[int, tuple[float, float]] = field(default_factory=dict)


@dataclass
class IdentityResult:
    """feet / tracks / box_samples are in REFERENCE (key-frame) camera
    coordinates when camera motion was compensated, else raw image coords."""

    feet: dict[str, list[Point | None]]
    tracks: list[list[TrackedPoint]]
    box_samples: list[tuple[float, float]]
    reassigned_frames: int
    defected_points: int
    camera_moving: bool = False
    feet_image: dict[str, list[Point | None]] | None = None  # raw image coords, for overlays



def assign_detections(
    dancers: list[_Dancer],
    boxes: list[Box],
    feats: list,
    point_positions: dict[int, tuple[float, float]],
    trusted: list[bool],
):
    """Hungarian assignment of boxes to dancers. Returns {dancer_index: box_index}."""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    if not dancers or not boxes:
        return {}
    big = 1e6
    cost = np.full((len(dancers), len(boxes)), big)
    for di, d in enumerate(dancers):
        px = d.foot[0] + d.vel[0]
        py = d.foot[1] + d.vel[1]
        gate = min(_MOTION_GATE + _MOTION_GATE_GROWTH * d.lost, _MOTION_GATE_MAX)
        pts = [point_positions[i] for i in d.point_idx if trusted[i] and i in point_positions]
        for bi, (box, feat) in enumerate(zip(boxes, feats)):
            x, y, w, h = box
            fx, fy = x + w / 2.0, float(y + h)
            ref_h = max(d.h, 1.0)
            motion = ((fx - px) ** 2 + (fy - py) ** 2) ** 0.5 / ref_h
            if motion > gate:
                continue
            app = 0.5 * appearance_distance(d.template_key, feat) + 0.5 * appearance_distance(
                d.template, feat
            )
            if pts:
                vote = sum(1 for (qx, qy) in pts if _inside(qx, qy, box)) / len(pts)
            else:
                vote = 0.5
            scale = abs(float(np.log(max(h, 1) / ref_h)))
            c = _W_MOTION * min(motion, _MOTION_CAP) + _W_APPEAR * app + _W_POINTS * (1 - vote) + _W_SCALE * scale
            if c <= _MAX_COST:
                cost[di, bi] = c
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols) if cost[r, c] < big}


def run_identity_tracking(
    video_path: Path,
    tracks: list[list[TrackedPoint]],
    point_names: list[str],
    key_frame: int,
    key_boxes: dict[str, Box],
    *,
    detector: _Detector,
    detect_every: int = 1,
    compensate_camera: bool = True,
) -> IdentityResult:
    import cv2

    from app.pipeline.camera_motion import CameraMotionEstimator, apply_h, camera_is_moving

    n_frames = max((len(t) for t in tracks), default=0)
    names = list(dict.fromkeys(point_names))
    trusted = [True] * len(tracks)
    defect_run = [0] * len(tracks)
    defect_at: list[int | None] = [None] * len(tracks)
    feet: dict[str, list[Point | None]] = {n: [None] * n_frames for n in names}
    # (frame, centre_x, top_y, bottom_y) of unoccluded matched boxes
    raw_samples: list[tuple[int, float, float, float]] = []
    reassigned = 0
    estimator = None

    dancers: list[_Dancer] = []
    for n in names:
        idx = [i for i, pn in enumerate(point_names) if pn == n]
        if n in key_boxes:
            x, y, w, h = key_boxes[n]
            foot = (x + w / 2.0, float(y + h))
            hh = float(h)
        else:
            pts = [tracks[i][key_frame] for i in idx if key_frame < len(tracks[i])]
            foot = (
                sorted(p.x for p in pts)[len(pts) // 2] if pts else 0.0,
                max((p.y for p in pts), default=0) + 100.0,
            )
            hh = 200.0
        dancers.append(_Dancer(name=n, point_idx=idx, foot=foot, h=hh))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    if compensate_camera:
        estimator = CameraMotionEstimator(
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        )
    last_boxes: list[Box] = []
    try:
        for fi in range(n_frames):
            if fi < key_frame:
                if not cap.grab():
                    break
                continue
            process = (fi - key_frame) % max(detect_every, 1) == 0
            if process or estimator is not None:
                ok, raw_frame = cap.read()
                frame = raw_frame if process else None
            else:
                ok = cap.grab()
                raw_frame = frame = None
            if not ok:
                break

            point_pos: dict[int, tuple[float, float]] = {}
            for i, tr in enumerate(tracks):
                if fi < len(tr) and tr[fi].visible:
                    point_pos[i] = (float(tr[fi].x), float(tr[fi].y))

            if fi == key_frame and frame is not None:
                for d in dancers:
                    box = key_boxes.get(d.name)
                    if box is None:
                        continue
                    feat = appearance_feature(frame, box)
                    d.template_key = feat
                    d.template = feat.copy()
                    for i in d.point_idx:
                        if i in point_pos:
                            d.offsets[i] = (d.foot[0] - point_pos[i][0], d.foot[1] - point_pos[i][1])

            # Track the camera BEFORE assigning: if it panned/zoomed since the
            # last frame, move every dancer's predicted position (and scale)
            # into this frame's view. Otherwise a zoom makes everyone appear
            # to rush outward and predictions point at the wrong person.
            if estimator is not None:
                estimator.add_frame(raw_frame, last_boxes)
                step = estimator.steps[-1]  # maps this frame -> previous frame
                if fi > key_frame and not _is_identity(step):
                    inv = _invert(step)
                    zoom = abs(inv[0][0] * inv[1][1] - inv[0][1] * inv[1][0]) ** 0.5
                    for d in dancers:
                        nx, ny = apply_h(inv, d.foot[0], d.foot[1])
                        vx, vy = apply_h(inv, d.foot[0] + d.vel[0], d.foot[1] + d.vel[1])
                        d.foot = (nx, ny)
                        d.vel = (vx - nx, vy - ny)
                        d.h *= zoom

            assignment: dict[int, int] = {}
            boxes: list[Box] = []
            if frame is not None:
                boxes = [det.bbox for det in detector.detect(frame)]
                last_boxes = boxes
                feats = [appearance_feature(frame, b) for b in boxes]
                ready = [d for d in dancers if d.template_key is not None]
                ready_idx = [di for di, d in enumerate(dancers) if d.template_key is not None]
                sub = assign_detections(ready, boxes, feats, point_pos, trusted)
                assignment = {ready_idx[k]: v for k, v in sub.items()}

                # Point defection: a trusted point sitting inside another
                # dancer's assigned box (and not its own) for several frames.
                owner_box = {di: boxes[bi] for di, bi in assignment.items()}
                for di, d in enumerate(dancers):
                    for i in d.point_idx:
                        if not trusted[i] or i not in point_pos:
                            continue
                        qx, qy = point_pos[i]
                        own = owner_box.get(di)
                        in_own = own is not None and _inside(qx, qy, own, 0.0)
                        in_other = any(
                            _inside(qx, qy, b) for oj, b in owner_box.items() if oj != di
                        )
                        if in_other and not in_own:
                            defect_run[i] += 1
                            if defect_run[i] >= _DEFECT_FRAMES:
                                trusted[i] = False
                                defect_at[i] = fi - _DEFECT_FRAMES + 1
                        elif in_own:
                            defect_run[i] = 0

            for di, d in enumerate(dancers):
                trusted_pts = [
                    (i, point_pos[i]) for i in d.point_idx if trusted[i] and i in point_pos
                ]
                if di in assignment:
                    x, y, w, h = boxes[assignment[di]]
                    fy = float(y + h)
                    inside_pts = [p for _, p in trusted_pts if _inside(p[0], p[1], (x, y, w, h))]
                    if inside_pts:
                        xs = sorted(p[0] for p in inside_pts)
                        fx = xs[len(xs) // 2]
                    else:
                        fx = x + w / 2.0
                    if d.lost > 0:
                        reassigned += 1
                    nv = (fx - d.foot[0], fy - d.foot[1])
                    d.vel = _clamp_vel(0.5 * d.vel[0] + 0.5 * nv[0], 0.5 * d.vel[1] + 0.5 * nv[1], d.h)
                    d.foot = (fx, fy)
                    d.h = 0.8 * d.h + 0.2 * h
                    d.lost = 0
                    for i, p in trusted_pts:
                        d.offsets[i] = (fx - p[0], fy - p[1])
                    clean = all(
                        _iou((x, y, w, h), b) < _CLEAN_IOU
                        for bj, b in enumerate(boxes)
                        if bj != assignment[di]
                    )
                    if clean:
                        raw_samples.append((fi, x + w / 2.0, float(y), fy))
                        feat = appearance_feature(frame, (x, y, w, h))
                        d.template = (1 - _TEMPLATE_RATE) * d.template + _TEMPLATE_RATE * feat
                    feet[d.name][fi] = (int(round(fx)), int(round(fy)))
                    continue

                # No box (hidden / merged / not a detection frame): coast on
                # trusted points + their last point->foot offsets.
                d.lost += 1
                est = [
                    (p[0] + d.offsets[i][0], p[1] + d.offsets[i][1])
                    for i, p in trusted_pts
                    if i in d.offsets
                ]
                px_, py_ = d.foot[0] + d.vel[0], d.foot[1] + d.vel[1]
                if est:
                    xs = sorted(e[0] for e in est)
                    ys = sorted(e[1] for e in est)
                    fx, fy = xs[len(xs) // 2], ys[len(ys) // 2]
                    # Points that put her implausibly far from where she was
                    # are probably latched onto someone else — don't follow.
                    if ((fx - px_) ** 2 + (fy - py_) ** 2) ** 0.5 > 0.5 * d.h:
                        est = []
                if est:
                    nv = (fx - d.foot[0], fy - d.foot[1])
                    d.vel = _clamp_vel(0.5 * d.vel[0] + 0.5 * nv[0], 0.5 * d.vel[1] + 0.5 * nv[1], d.h)
                    d.foot = (fx, fy)
                    feet[d.name][fi] = (int(round(fx)), int(round(fy)))
                else:
                    d.vel = (d.vel[0] * _VEL_DAMP, d.vel[1] * _VEL_DAMP)
                    d.foot = (d.foot[0] + d.vel[0], d.foot[1] + d.vel[1])
    finally:
        cap.release()

    # Camera motion: express everything in the key frame's view so the floor
    # mapping (solved in that view) stays valid while the camera pans/zooms.
    mats = estimator.to_reference(0) if estimator is not None else []
    moving = bool(mats) and camera_is_moving(
        mats, (int(estimator.size[0] / estimator.scale), int(estimator.size[1] / estimator.scale))
    )

    def to_ref(fi: int, x: float, y: float) -> tuple[float, float]:
        k = fi - key_frame
        if not moving or k < 0 or k >= len(mats):
            return x, y
        return apply_h(mats[k], x, y)

    feet_image = {name: list(series) for name, series in feet.items()}
    if moving:
        for name, series in feet.items():
            feet[name] = [
                None if p is None else tuple(int(round(v)) for v in to_ref(fi, p[0], p[1]))  # type: ignore[misc]
                for fi, p in enumerate(series)
            ]
    box_samples: list[tuple[float, float]] = []
    for fi, cx, top, bottom in raw_samples:
        _, ty = to_ref(fi, cx, top)
        _, by = to_ref(fi, cx, bottom)
        box_samples.append((by, by - ty))

    cleaned: list[list[TrackedPoint]] = []
    for i, tr in enumerate(tracks):
        cut = defect_at[i]
        out_tr = []
        for p in tr:
            x, y = (p.x, p.y)
            if moving:
                rx, ry = to_ref(p.frame, p.x, p.y)
                x, y = int(round(rx)), int(round(ry))
            visible = p.visible and (cut is None or p.frame < cut)
            out_tr.append(p if (x, y, visible) == (p.x, p.y, p.visible) else TrackedPoint(p.frame, x, y, visible))
        cleaned.append(out_tr)
    defected = sum(1 for t in trusted if not t)
    if moving:
        logger.info("identity: camera pans/zooms — positions compensated to the key-frame view")
    logger.info(
        "identity: %d dancers, %d points defected (ignored after latching onto "
        "someone else), %d re-acquisitions after occlusion, %d clean box samples",
        len(dancers), defected, reassigned, len(box_samples),
    )
    return IdentityResult(feet, cleaned, box_samples, reassigned, defected, moving, feet_image)
