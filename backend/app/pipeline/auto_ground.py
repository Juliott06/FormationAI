"""Top-down floor mapping WITHOUT marked corners, calibrated from the dancers.

Most practice videos don't show the floor's corners. But people are a known
ruler: dancers are roughly the same height, and in a pinhole camera a person's
height in pixels is proportional to how far their feet are below the horizon:

    h_px = k * (v_foot - v_horizon),     k = person_height / camera_height

Fitting that line to the person boxes YOLO already finds gives the horizon row
and camera height (in person-height units). With an assumed focal length, any
foot pixel then maps to a point on the floor — a ground-plane homography — so
dancers get true relative positions in both width and depth, no clicks needed.

Assumptions: square pixels, principal point at the image centre, no camera
roll, a typical phone focal length (0.8 x image width). Depth spacing scales
with the focal length (±12% wrong guess ≈ ±12% depth), width doesn't depend
on it at all.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass


logger = logging.getLogger("uvicorn.error")

FOCAL_WIDTHS = 0.8
# Need this much vertical spread of foot positions (fraction of frame height)
# to trust a fitted horizon; otherwise assume a level camera (horizon at the
# image centre), which practice-room tripods usually are.
_MIN_FOOT_SPREAD = 0.06


@dataclass(frozen=True)
class GroundModel:
    horizon_v: float
    k: float  # person_height / camera_height
    focal: float
    cx: float
    cy: float
    fitted_horizon: bool

    def to_homography(self) -> list[list[float]]:
        """3x3 matrix mapping image (u, v, 1) to floor (X, Z, 1), in units of
        person height; X right, Z away from the camera."""
        f, cx, cy = self.focal, self.cx, self.cy
        theta = math.atan2(cy - self.horizon_v, f)  # pitch down (>0)
        c, s = math.cos(theta), math.sin(theta)
        cam_h = 1.0 / self.k
        # Ray d = ((u-cx)/f, (v-cy)/f, 1). Level-frame components:
        #   x = d_x, y_down = d_y c + s, z = -d_y s + c.
        # Floor hit at y_down = cam_h: X = cam_h x / y_down, Z = cam_h z / y_down.
        # Written as rows over (u, v, 1) (common 1/f factor cancels):
        row_x = [cam_h, 0.0, -cam_h * cx]
        row_z = [0.0, -cam_h * s, cam_h * (cy * s + f * c)]
        row_w = [0.0, c, -cy * c + f * s]
        return [row_x, row_z, row_w]


def fit_ground_model(
    samples: list[tuple[float, float]],
    frame_size: tuple[int, int],
) -> GroundModel | None:
    """samples: (foot_v, box_height_px) of person boxes. Returns None if there
    is nothing usable."""
    import numpy as np

    w, h = frame_size
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    focal = FOCAL_WIDTHS * w
    pts = np.array([(v, bh) for v, bh in samples if bh > 8], dtype=np.float64)
    if len(pts) < 5:
        return None
    v, hh = pts[:, 0], pts[:, 1]

    model = None
    if np.percentile(v, 90) - np.percentile(v, 10) >= _MIN_FOOT_SPREAD * h:
        keep = np.ones(len(v), dtype=bool)
        for _ in range(4):
            a, b = np.polyfit(v[keep], hh[keep], 1)
            resid = hh - (a * v + b)
            mad = np.median(np.abs(resid[keep] - np.median(resid[keep]))) + 1e-6
            # Crouches / jumps / merged boxes sit far off the line.
            keep = np.abs(resid - np.median(resid[keep])) < 2.5 * 1.4826 * mad
            if keep.sum() < 5:
                break
        if a > 1e-3:
            horizon = -b / a
            # Horizon must be above the feet and not absurdly far up.
            if -1.5 * h < horizon < np.percentile(v, 5) - 0.05 * h:
                model = GroundModel(horizon, float(a), focal, cx, cy, True)
    if model is None:
        below = v > cy + 0.03 * h
        if below.sum() < 5:
            return None
        k = float(np.median(hh[below] / (v[below] - cy)))
        model = GroundModel(cy, k, focal, cx, cy, False)
    logger.info(
        "auto ground: horizon v=%.0f (%s), camera height=%.2f person heights",
        model.horizon_v,
        "fitted" if model.fitted_horizon else "assumed level",
        1.0 / model.k,
    )
    return model


def fit_floor_to_stage(
    points: list[tuple[float, float]], margin: float = 0.12
) -> tuple[float, float, float, float]:
    """Uniform-scale map from floor (X, Z) to normalized stage (x, y).

    Returns (s, ox, oy) style parameters as (sx, sy, ox, oy) with
    x = sx*X + ox, y = sy*Z + oy, where sx*800 == sy*450 so one metre is the
    same length on screen across and in depth. The 2nd-98th percentile box of
    the dancers' positions is fitted inside the margin and centred."""
    from app.pipeline.homography import CANVAS_ASPECT

    xs = sorted(p[0] for p in points)
    zs = sorted(p[1] for p in points)
    trim = int(len(xs) * 0.02) if len(xs) >= 50 else 0
    x0, x1 = xs[trim], xs[-1 - trim]
    z0, z1 = zs[trim], zs[-1 - trim]
    span_x = max(x1 - x0, 1e-6)
    span_z = max(z1 - z0, 1e-6)
    avail = 1.0 - 2 * margin
    # Canvas units: x spans CANVAS_ASPECT "screen units", y spans 1.
    scale = min(avail * CANVAS_ASPECT / span_x, avail / span_z)  # screen units / metre
    sx = scale / CANVAS_ASPECT
    sy = scale
    ox = 0.5 - sx * (x0 + x1) / 2.0
    oy = 0.5 - sy * (z0 + z1) / 2.0
    return sx, sy, ox, oy
