"""End-to-end check of the click-to-track pipeline on a synthetic clip.

Renders a fake practice-room video (perspective camera, 6 coloured dancers
moving through known formations, with crossings), then runs the REAL
CoTracker pipeline with the two ML models swapped for simulators that
reproduce their failure modes:

- CoTracker: follows each query point with pixel noise, reports it invisible
  while a nearer dancer covers it, and — the classic failure — a point that
  stays covered for a while can latch onto the occluder for good.
- YOLO: ground-truth person boxes with jitter, random misses, and boxes of
  heavily overlapping people merged into one (as NMS does).

Because the true floor positions are known, it reports how far the rendered
stage positions are from the truth, whether the floor proportions survive,
and which formations were snapped. Needs no model weights or network.

Usage (from backend/):
    python -m scripts.synthetic_check --out ../synthetic_out
"""
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------- scene ----

W, H = 1280, 720
FPS = 24.0
FOCAL = 0.8 * W
CAM_POS = np.array([0.0, 2.0, -7.0])  # x right, y up, z into the room (metres)
PITCH_DEG = 7.0  # camera tilted down
FLOOR_W, FLOOR_D = 8.0, 6.0  # floor x in [-4, 4], z in [0, 6] (z=0 = front)
PERSON_H, PERSON_W = 1.65, 0.45
COLORS = [
    (60, 60, 230), (60, 200, 60), (230, 120, 40),
    (40, 200, 230), (200, 60, 200), (230, 230, 230),
]
NAMES = ["Ana", "Bo", "Cy", "Di", "Ed", "Fi"]

_c, _s = math.cos(math.radians(PITCH_DEG)), math.sin(math.radians(PITCH_DEG))
# world -> camera: rotate about x so the optical axis points down by PITCH.
_R = np.array([[1, 0, 0], [0, _c, _s], [0, -_s, _c]], dtype=np.float64)


def project(p: np.ndarray) -> tuple[float, float, float]:
    """World point -> (u, v, depth)."""
    cam = _R @ (p - CAM_POS)
    x, y, z = cam
    return W / 2 + FOCAL * x / z, H / 2 - FOCAL * y / z, z


def floor_pt(x: float, z: float) -> np.ndarray:
    return np.array([x, 0.0, z])


# ---------------------------------------------------------- choreography ----

# Formations in floor metres (x, z). Each list is indexed by SLOT; the
# dancer->slot assignment changes between formations to force crossings.
FORMATIONS: list[tuple[str, list[tuple[float, float]]]] = [
    ("Line", [(-3.0, 3.0), (-1.8, 3.0), (-0.6, 3.0), (0.6, 3.0), (1.8, 3.0), (3.0, 3.0)]),
    ("V", [(-0.6, 1.2), (0.6, 1.2), (-1.8, 2.8), (1.8, 2.8), (-3.0, 4.4), (3.0, 4.4)]),
    ("Two rows", [(-2.4, 2.0), (0.0, 2.0), (2.4, 2.0), (-2.4, 4.5), (0.0, 4.5), (2.4, 4.5)]),
    ("Inverted V", [(-0.6, 4.8), (0.6, 4.8), (-1.8, 3.2), (1.8, 3.2), (-3.0, 1.6), (3.0, 1.6)]),
    ("Triangle", [(0.0, 1.2), (-1.0, 2.8), (1.0, 2.8), (-2.0, 4.4), (0.0, 4.4), (2.0, 4.4)]),
]
ASSIGNMENTS = [
    [0, 1, 2, 3, 4, 5],
    [4, 2, 0, 1, 3, 5],
    [1, 5, 3, 0, 2, 4],
    [0, 3, 5, 2, 4, 1],
    [2, 1, 4, 5, 0, 3],
]
HOLD_SEC, MOVE_SEC, LEAD_SEC = 3.0, 1.5, 1.0


def _ease(t: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * t)


def choreography() -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Return positions[frame, dancer, (x,z)] and [(name, start, end)] holds."""
    per_formation = []
    for (name, slots), assign in zip(FORMATIONS, ASSIGNMENTS):
        per_formation.append(np.array([slots[assign[d]] for d in range(6)]))
    hold_f, move_f, lead_f = int(HOLD_SEC * FPS), int(MOVE_SEC * FPS), int(LEAD_SEC * FPS)
    frames: list[np.ndarray] = []
    holds: list[tuple[str, int, int]] = []
    for i, pos in enumerate(per_formation):
        start = len(frames)
        n_hold = hold_f + (lead_f if i == 0 else 0)
        for k in range(n_hold):
            # small sway so holds aren't perfectly frozen
            sway = 0.03 * math.sin(2 * math.pi * (len(frames) / FPS) * 0.8)
            frames.append(pos + np.array([sway, 0.0]))
        holds.append((FORMATIONS[i][0], start, len(frames) - 1))
        if i + 1 < len(per_formation):
            nxt = per_formation[i + 1]
            for k in range(move_f):
                t = _ease((k + 1) / (move_f + 1))
                frames.append(pos + (nxt - pos) * t)
    return np.stack(frames), holds


# ------------------------------------------------------------- rendering ----


@dataclass
class Body:
    dancer: int
    depth: float
    box: tuple[int, int, int, int]  # x, y, w, h
    foot: tuple[float, float]


def bodies_at(pos: np.ndarray) -> list[Body]:
    out = []
    for d in range(pos.shape[0]):
        x, z = pos[d]
        fu, fv, depth = project(floor_pt(x, z))
        hu, hv, _ = project(np.array([x, PERSON_H, z]))
        h = fv - hv
        w = h * PERSON_W / PERSON_H
        out.append(Body(d, depth, (int(fu - w / 2), int(hv), int(w), int(h)), (fu, fv)))
    return out


def render_background() -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (70, 70, 80)  # wall
    poly = [project(floor_pt(x, z))[:2] for x, z in [(-6, -2), (6, -2), (6, 6), (-6, 6)]]
    cv2.fillPoly(img, [np.array(poly, np.int32)], (95, 110, 125))
    for x in np.arange(-6, 6.01, 1.0):
        a, b = project(floor_pt(x, -2))[:2], project(floor_pt(x, 6))[:2]
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (80, 92, 105), 1)
    for z in np.arange(-2, 6.01, 1.0):
        a, b = project(floor_pt(-6, z))[:2], project(floor_pt(6, z))[:2]
        cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (80, 92, 105), 1)
    return img


def render_frame(bg: np.ndarray, bodies: list[Body]) -> np.ndarray:
    img = bg.copy()
    for b in sorted(bodies, key=lambda b: -b.depth):  # far first
        x, y, w, h = b.box
        col = COLORS[b.dancer]
        head_r = max(int(w * 0.28), 2)
        cv2.circle(img, (x + w // 2, y + head_r), head_r, (150, 180, 220), -1)
        cv2.rectangle(img, (x + w // 8, y + 2 * head_r), (x + w - w // 8, y + int(h * 0.55)), col, -1)
        cv2.rectangle(img, (x + w // 5, y + int(h * 0.55)), (x + w - w // 5, y + h), (40, 40, 50), -1)
    return img


# -------------------------------------------------------------- simulators ----

_BITS, _CELL = 12, 8


def stamp(img: np.ndarray, fi: int) -> None:
    """Frame index as black/white 8px cells in the top-left corner (survives
    lossy encoding)."""
    for b in range(_BITS):
        val = 255 if (fi >> b) & 1 else 0
        img[0:_CELL, b * _CELL:(b + 1) * _CELL] = val


def read_stamp(img: np.ndarray) -> int:
    fi = 0
    for b in range(_BITS):
        cell = img[2:_CELL - 2, b * _CELL + 2:(b + 1) * _CELL - 2]
        if cell.mean() > 127:
            fi |= 1 << b
    return fi


def covered_by(bodies: list[Body], dancer: int, pt: tuple[float, float]) -> int | None:
    me = bodies[dancer]
    for b in bodies:
        if b.dancer == dancer or b.depth >= me.depth:
            continue
        x, y, w, h = b.box
        if x <= pt[0] <= x + w and y <= pt[1] <= y + h:
            return b.dancer
    return None


class SimCoTracker:
    """Follows a query point as a fixed fraction of its dancer's box."""

    def __init__(self, all_bodies: list[list[Body]], seed: int = 1) -> None:
        self.all_bodies = all_bodies
        self.rng = random.Random(seed)

    def track_video(self, video_path, clicks):
        from app.services.cotracker import TrackedPoint

        out = []
        for ci, c in enumerate(clicks):
            kf = c["key_frame"]
            bodies = self.all_bodies[kf]
            owner = None
            for b in sorted(bodies, key=lambda b: b.depth):
                x, y, w, h = b.box
                if x <= c["x"] <= x + w and y <= c["y"] <= y + h:
                    owner = b
                    break
            if owner is None:
                owner = min(bodies, key=lambda b: abs(b.box[0] + b.box[2] / 2 - c["x"]))
            fx = (c["x"] - owner.box[0]) / max(owner.box[2], 1)
            fy = (c["y"] - owner.box[1]) / max(owner.box[3], 1)
            target = owner.dancer
            covered_run = 0
            pts = []
            for fi, bodies in enumerate(self.all_bodies):
                b = bodies[target]
                x, y, w, h = b.box
                px = x + fx * w + self.rng.gauss(0, 1.5)
                py = y + fy * h + self.rng.gauss(0, 1.5)
                occ = covered_by(bodies, target, (px, py)) if fi >= kf else None
                covered_run = covered_run + 1 if occ is not None else 0
                # Long occlusion: sometimes the point latches onto the occluder.
                if occ is not None and covered_run == 8 and self.rng.random() < 0.35:
                    ob = bodies[occ]
                    fx = (px - ob.box[0]) / max(ob.box[2], 1)
                    fy = (py - ob.box[1]) / max(ob.box[3], 1)
                    target = occ
                    occ = None
                visible = fi >= kf and occ is None
                pts.append(TrackedPoint(fi, int(round(px)), int(round(py)), visible))
            out.append(pts)
        return out


class SimDetector:
    def __init__(self, all_bodies: list[list[Body]], seed: int = 2) -> None:
        self.all_bodies = all_bodies
        self.rng = random.Random(seed)
        self.frame_lookup: dict[bytes, int] = {}

    def detect(self, frame_bgr):
        from app.pipeline.types import Detection

        fi = read_stamp(frame_bgr)
        bodies = self.all_bodies[min(fi, len(self.all_bodies) - 1)]
        boxes = []
        for b in bodies:
            if self.rng.random() < 0.1:
                continue
            x, y, w, h = b.box
            j = lambda s: int(round(self.rng.gauss(0, 0.03 * s)))
            boxes.append([x + j(w), y + j(h), max(w + j(w), 4), max(h + j(h), 8)])
        merged: list[list[int]] = []
        for bx in boxes:
            for m in merged:
                ix = max(0, min(bx[0] + bx[2], m[0] + m[2]) - max(bx[0], m[0]))
                iy = max(0, min(bx[1] + bx[3], m[1] + m[3]) - max(bx[1], m[1]))
                inter = ix * iy
                union = bx[2] * bx[3] + m[2] * m[3] - inter
                if union > 0 and inter / union > 0.55:
                    x0, y0 = min(bx[0], m[0]), min(bx[1], m[1])
                    x1 = max(bx[0] + bx[2], m[0] + m[2])
                    y1 = max(bx[1] + bx[3], m[1] + m[3])
                    m[:] = [x0, y0, x1 - x0, y1 - y0]
                    break
            else:
                merged.append(bx)
        return [
            Detection(bbox=tuple(m), anchor_px=(m[0] + m[2] // 2, m[1] + m[3]), confidence=0.9)
            for m in merged
        ]


# ------------------------------------------------------------------ main ----


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("../synthetic_out"))
    ap.add_argument("--key-frame", type=int, default=6)
    ap.add_argument("--no-corners", action="store_true")
    ap.add_argument("--seed", type=int, default=1, help="simulated tracker failure seed")
    ap.add_argument("--no-identity", action="store_true", help="disable identity correction")
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    positions, holds = choreography()
    all_bodies = [bodies_at(p) for p in positions]
    bg = render_background()
    video = out / "synthetic.mp4"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for fi, bodies in enumerate(all_bodies):
        img = render_frame(bg, bodies)
        stamp(img, fi)  # lets SimDetector know which frame it was handed
        vw.write(img)
    vw.release()

    import app.pipeline.cotracker_pipeline as cp
    from app.pipeline.homography import project_to_stage
    from app.schemas.jobs import DancerClick

    kf = args.key_frame
    clicks = []
    for b in all_bodies[kf]:
        x, y, w, h = b.box
        clicks.append(DancerClick(name=NAMES[b.dancer], x=x + w // 2, y=y + int(0.38 * h)))
    corners = None
    if not args.no_corners:
        rng = random.Random(5)
        corners = []
        for x, z in [(-4, 0), (4, 0), (4, 6), (-4, 6)]:
            u, v, _ = project(floor_pt(x, z))
            corners.append([int(u + rng.uniform(-4, 4)), int(v + rng.uniform(-3, 3))])

    from app.core.config import get_settings

    if args.no_identity:
        get_settings().identity_tracking = False
    cp.build_client = lambda: SimCoTracker(all_bodies, seed=args.seed)
    cp.YoloPersonDetector = lambda **kw: SimDetector(all_bodies)  # type: ignore[assignment]
    result = cp.process_video_with_cotracker(
        job_id="synthetic",
        video_path=video,
        clicks=clicks,
        key_frame=kf,
        progress_callback=lambda a, b: None,
        stage_corners=corners,
    )
    (out / "positions.json").write_text(result.model_dump_json())

    # Ground truth in stage coords: the true feet through the SAME homography
    # (so margins / aspect handling are identical) — any remaining error is
    # tracking/anchoring error. Floor proportions are checked separately.
    if corners is not None:
        from app.pipeline.homography import compute_stage_homography

        hom = compute_stage_homography(corners, (W, H))
        hom_exact = compute_stage_homography(
            [[round(project(floor_pt(x, z))[0]), round(project(floor_pt(x, z))[1])]
             for x, z in [(-4, 0), (4, 0), (4, 6), (-4, 6)]],
            (W, H),
        )

        def gt_stage(fi: int, d: int) -> tuple[float, float]:
            return project_to_stage(hom_exact, all_bodies[fi][d].foot)

        # Floor proportions: canvas distance across vs deep, in screen units
        bl = project_to_stage(hom, project(floor_pt(-4, 6))[:2])
        br = project_to_stage(hom, project(floor_pt(4, 6))[:2])
        fl = project_to_stage(hom, project(floor_pt(-4, 0))[:2])
        width_px = abs(br[0] - bl[0]) * 800
        depth_px = abs(bl[1] - fl[1]) * 450
        floor_aspect = width_px / max(depth_px, 1e-6)
    else:
        floor_aspect = float("nan")

        def gt_stage(fi: int, d: int) -> tuple[float, float]:
            return float("nan"), float("nan")

    name_to_dancer = {n: i for i, n in enumerate(NAMES)}
    id_to_dancer = {i + 1: name_to_dancer[n] for i, n in enumerate(dict.fromkeys(c.name for c in clicks))}
    errs = []
    depth_errs = []
    wrong_identity = 0
    for f in result.frames:
        gts = {d: gt_stage(f.frame, d) for d in range(len(NAMES))}
        for dp in f.dancers:
            gx, gy = gts[id_to_dancer[dp.id]]
            errs.append(math.hypot((dp.x - gx) * 800, (dp.y - gy) * 450))
            depth_errs.append(abs(dp.y - gy) * 450)
            # Identity error: the dot is nearer some OTHER dancer's true spot.
            nearest = min(gts, key=lambda d: math.hypot((dp.x - gts[d][0]) * 800, (dp.y - gts[d][1]) * 450))
            if nearest != id_to_dancer[dp.id]:
                wrong_identity += 1
    errs_np = np.array(errs)
    summary = {
        "frames": len(result.frames),
        "key_frame": kf,
        "frames_before_key_with_dancers": sum(1 for f in result.frames[:kf] if f.dancers),
        "coverage_pct": round(100 * sum(len(f.dancers) for f in result.frames[kf:])
                              / (6 * (len(result.frames) - kf)), 1),
        "stage_err_px_median": round(float(np.median(errs_np)), 1),
        "stage_err_px_p90": round(float(np.percentile(errs_np, 90)), 1),
        "stage_err_px_max": round(float(errs_np.max()), 1),
        "depth_err_px_median": round(float(np.median(depth_errs)), 1),
        "pct_dancer_frames_off_by_gt_40px": round(100 * float((errs_np > 40).mean()), 1),
        "pct_dancer_frames_wrong_identity": round(100 * wrong_identity / max(len(errs), 1), 2),
        "floor_aspect_on_canvas": round(floor_aspect, 3),
        "floor_aspect_true": FLOOR_W / FLOOR_D,
        "true_holds": [(n, round(s / FPS, 1), round(e / FPS, 1)) for n, s, e in holds],
        "detected_formations": [
            (fm.shape_name, round(fm.start_time_sec, 1), round(fm.end_time_sec, 1), len(fm.dancers))
            for fm in result.formations
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    from app.pipeline.stage_renderer import concat_side_by_side, render_stage_video

    labels = {i + 1: n for i, n in enumerate(dict.fromkeys(c.name for c in clicks))}
    stage = render_stage_video(result, out / "stage.mp4", labels=labels)
    concat_side_by_side(video, stage, out / "side_by_side.mp4")


if __name__ == "__main__":
    main()
