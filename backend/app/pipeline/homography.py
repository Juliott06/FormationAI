"""Perspective (homography) mapping from camera image space to a top-down floor.

A fixed camera sees the dance floor in perspective: a rectangle on the floor
appears as a trapezoid in the image, so dancers at the back look compressed and
shifted. If the user marks the four floor corners, we can solve the homography
that maps image pixels back to a true top-down rectangle and place each dancer's
FOOT position on the floor plane correctly.

Coordinate conventions:
- Input corners are image pixels (x right, y down), any order — we classify them
  into back-left / back-right / front-right / front-left by geometry.
- Output stage coords are normalized: x in [0,1] left->right, y in [0,1] where
  y=1 is the BACK of the stage and y=0 the FRONT (matches the frontend, which
  renders at cy = (1 - y) * height).
"""
from __future__ import annotations


Corner = tuple[float, float]

# The marked floor quad maps to the middle of stage space, not the exact edges.
# Users eyeball the corners, and dancers routinely step slightly outside the
# marked floor — with a hard 0..1 mapping those feet clamp onto the canvas
# border and look cut off. The margin keeps them visible.
STAGE_MARGIN = 0.1


def _order_corners(corners: list[Corner]) -> list[Corner]:
    """Sort 4 image-space points into [back-left, back-right, front-right, front-left].

    Image y grows downward, so the two smallest-y points are the back edge."""
    pts = sorted(corners, key=lambda p: p[1])  # by image y: back (top) first
    back = sorted(pts[:2], key=lambda p: p[0])  # left, right
    front = sorted(pts[2:], key=lambda p: p[0])  # left, right
    back_left, back_right = back
    front_left, front_right = front
    return [back_left, back_right, front_right, front_left]


def compute_stage_homography(corners: list[list[int]] | list[Corner]) -> list[list[float]] | None:
    """Return a 3x3 homography (list of lists) mapping image px -> stage (x,y) in
    [0,1], or None if the corners are unusable (not exactly 4, or degenerate).

    Uses the standard 4-point DLT; solved with numpy so the result is a plain
    nested list that round-trips through JSON."""
    import numpy as np

    if corners is None or len(corners) != 4:
        return None
    src = _order_corners([(float(x), float(y)) for x, y in corners])
    # Destination in stage space: back-left, back-right, front-right, front-left,
    # inset by STAGE_MARGIN (see comment above).
    lo, hi = STAGE_MARGIN, 1.0 - STAGE_MARGIN
    dst = [(lo, hi), (hi, hi), (hi, lo), (lo, lo)]

    # Reject degenerate quads (near-collinear / zero area via the shoelace formula).
    area = 0.0
    for i in range(4):
        x0, y0 = src[i]
        x1, y1 = src[(i + 1) % 4]
        area += x0 * y1 - x1 * y0
    if abs(area) < 1.0:  # < 1 square pixel of floor => unusable
        return None

    # Solve H (up to scale) from 4 correspondences: 8 equations, 8 unknowns.
    a_rows = []
    b_vals = []
    for (sx, sy), (dx, dy) in zip(src, dst):
        a_rows.append([sx, sy, 1, 0, 0, 0, -sx * dx, -sy * dx])
        b_vals.append(dx)
        a_rows.append([0, 0, 0, sx, sy, 1, -sx * dy, -sy * dy])
        b_vals.append(dy)
    a = np.asarray(a_rows, dtype=np.float64)
    b = np.asarray(b_vals, dtype=np.float64)
    try:
        h = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(h)):
        return None
    return [
        [float(h[0]), float(h[1]), float(h[2])],
        [float(h[3]), float(h[4]), float(h[5])],
        [float(h[6]), float(h[7]), 1.0],
    ]


def project_to_stage(
    h: list[list[float]], anchor_px: tuple[int, int]
) -> tuple[float, float]:
    """Map an image-space foot anchor through the homography to stage (x,y).

    UNCLAMPED: values outside [0,1] mean the point lies outside the marked
    floor quad. Callers must run compute_rescue_affine over the full set of
    projected points and then clamp — clamping here would silently pile
    out-of-quad dancers onto the canvas border (observed: a user marked only
    the empty floor in FRONT of the dancers and every dancer rendered as a
    flat line glued to the back edge).

    Pure math (no numpy) so it is cheap per-dancer-per-frame."""
    x, y = float(anchor_px[0]), float(anchor_px[1])
    denom = h[2][0] * x + h[2][1] * y + h[2][2]
    if abs(denom) < 1e-9:
        return 0.5, 0.5
    u = (h[0][0] * x + h[0][1] * y + h[0][2]) / denom
    v = (h[1][0] * x + h[1][1] * y + h[1][2]) / denom
    return round(u, 6), round(v, 6)


def compute_rescue_affine(
    points: list[tuple[float, float]],
) -> tuple[float, float, float]:
    """Fit out-of-bounds stage points back onto the canvas.

    Returns (s, ox, oy) for x' = s*x + ox, y' = s*y + oy.

    The scale is UNIFORM (same for x and y) so the floor's proportions are
    preserved — per-axis stretching is exactly the distortion the homography
    exists to remove. Scale is capped at 1: a correctly marked floor is left
    untouched; a wrong quad is zoomed out / recentered until everything fits
    inside the STAGE_MARGIN box."""
    if not points:
        return 1.0, 0.0, 0.0
    lo, hi = STAGE_MARGIN, 1.0 - STAGE_MARGIN
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    if min_x >= 0.0 and max_x <= 1.0 and min_y >= 0.0 and max_y <= 1.0:
        return 1.0, 0.0, 0.0  # everything already on-canvas — trust the quad

    span = max(max_x - min_x, max_y - min_y, 1e-6)
    s = min(1.0, (hi - lo) / span)
    # Center the (uniformly scaled) bounding box on the stage.
    ox = 0.5 - s * (min_x + max_x) / 2.0
    oy = 0.5 - s * (min_y + max_y) / 2.0
    return s, ox, oy
