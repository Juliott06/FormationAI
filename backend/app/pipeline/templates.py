from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class FormationTemplate:
    name: str
    points: np.ndarray  # shape (N, 2), canonical positions roughly in [-1, 1]


def generate_templates_for_count(n: int) -> list[FormationTemplate]:
    if n < 2:
        return []

    templates: list[FormationTemplate] = []

    # Single horizontal line
    line = np.array([[i / (n - 1) * 2 - 1, 0.0] for i in range(n)])
    templates.append(FormationTemplate("Line", line))

    # Vertical line (front to back relative to camera)
    vline = np.array([[0.0, i / (n - 1) * 2 - 1] for i in range(n)])
    templates.append(FormationTemplate("Vertical line", vline))

    # Two rows (front + back)
    front_n = (n + 1) // 2
    back_n = n - front_n
    front = [[(i + 0.5) / front_n * 2 - 1, -0.5] for i in range(front_n)]
    back = [[(i + 0.5) / back_n * 2 - 1, 0.5] for i in range(back_n)]
    templates.append(FormationTemplate("Two rows", np.array(front + back)))

    # Grid (rows × cols closest to square)
    cols = max(1, int(round(n ** 0.5)))
    rows = (n + cols - 1) // cols
    if rows >= 2 and cols >= 2:
        grid_pts = []
        for r in range(rows):
            in_row = min(cols, n - r * cols)
            # A partial last row is centred, not left-aligned.
            offset = (cols - in_row) / 2.0
            for c in range(in_row):
                x = ((c + offset) / max(cols - 1, 1)) * 2 - 1
                y = (r / max(rows - 1, 1)) * 2 - 1
                grid_pts.append([x, y])
        templates.append(
            FormationTemplate(f"{rows}x{cols} grid", np.array(grid_pts))
        )

    # V shape (point at the front). Symmetric: odd counts put one dancer at
    # the apex, even counts a pair; the rest pair up outward/backward.
    pairs = n // 2
    has_apex = n % 2 == 1
    levels = pairs + (1 if has_apex else 0)
    v_list: list[list[float]] = []
    if has_apex:
        v_list.append([0.0, -1.0])
    for k in range(pairs):
        level = k + (1 if has_apex else 0)
        y = -1.0 + 2.0 * level / max(levels - 1, 1)
        x = (k + 1) / pairs if has_apex else (k + 0.5) / pairs
        v_list.append([-x, y])
        v_list.append([x, y])
    v_pts = np.array(v_list)
    templates.append(FormationTemplate("V", v_pts))

    # Centre front / centre back: one dancer in the middle ahead of (or
    # behind) a row of everyone else — the K-pop "centre position" layout.
    if n >= 4:
        row = [[(i / (n - 2)) * 2 - 1, 0.5] for i in range(n - 1)]
        templates.append(FormationTemplate("Centre front", np.array([[0.0, -0.5]] + row)))
        flipped = [[x, -y] for x, y in row]
        templates.append(FormationTemplate("Centre back", np.array([[0.0, 0.5]] + flipped)))
    if n == 4:
        templates.append(
            FormationTemplate("Diamond", np.array([[0.0, -1.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]))
        )

    # Inverted V (point at the back)
    inv_v = v_pts.copy()
    inv_v[:, 1] *= -1
    templates.append(FormationTemplate("Inverted V", inv_v))

    # Diamond — only nicely fits odd counts (1+3+5+..., or pyramid)
    # For 9: 1+3+5 = 9 (triangle, point up)
    # For 6: 1+2+3 = 6 (triangle)
    accum = 0
    diamond_rows: list[int] = []
    row_size = 1
    while accum + row_size <= n:
        diamond_rows.append(row_size)
        accum += row_size
        row_size += 1
    if accum == n and len(diamond_rows) >= 2:
        triangle_pts = []
        max_row = len(diamond_rows)
        for r_idx, row_count in enumerate(diamond_rows):
            y = (r_idx / max(max_row - 1, 1)) * 2 - 1
            for c_idx in range(row_count):
                x = (c_idx - (row_count - 1) / 2) / max(max_row - 1, 1) * 2
                triangle_pts.append([x, y])
        tri = np.array(triangle_pts)
        templates.append(FormationTemplate("Triangle", tri))
        inv_tri = tri.copy()
        inv_tri[:, 1] *= -1
        templates.append(FormationTemplate("Inverted triangle", inv_tri))

    return templates


def _axis_aligned_fit(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    """Best fit of `source` onto `target` (same order) using translation and a
    separate NON-NEGATIVE scale per axis — deliberately no rotation.

    Formations are named relative to the audience: a line tilted 30 degrees is
    not a "Line", and a V rotated 180 degrees is an inverted V. The previous
    similarity (Procrustes) fit allowed any rotation, so tilted shapes snapped
    to "Line" and got drawn tilted, and V / Inverted V were interchangeable.
    Per-axis scale lets a wide shallow V still be a V. Returns the fitted
    points and RMSE (in target units)."""
    src_c = source - source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    tgt_c = target - tgt_mean
    fitted = np.empty_like(tgt_c)
    for axis in range(2):
        denom = float((src_c[:, axis] ** 2).sum())
        scale = float((src_c[:, axis] * tgt_c[:, axis]).sum() / denom) if denom > 1e-12 else 0.0
        fitted[:, axis] = max(scale, 0.0) * src_c[:, axis]
    fitted += tgt_mean
    rmse = float(np.sqrt(((fitted - target) ** 2).sum(axis=1).mean()))
    return fitted, rmse


def formation_spread(observed: np.ndarray) -> float:
    """RMS distance of the points from their centroid — the formation's size,
    used to judge fit error relative to how big the formation is."""
    centered = observed - observed.mean(axis=0)
    return float(np.sqrt((centered ** 2).sum(axis=1).mean()))


def fit_template(
    template: FormationTemplate, observed: np.ndarray, *, max_iters: int = 6
) -> tuple[np.ndarray, float]:
    """Iteratively fit template points to observed points (same N).
    Returns (snapped_positions in observed order, RMSE).

    Alternates optimal assignment (Hungarian) with the axis-aligned fit,
    starting from the template stretched to the observed bounding box."""
    n = observed.shape[0]
    if template.points.shape[0] != n or n < 2:
        return observed.copy(), float("inf")

    obs_mean = observed.mean(axis=0)
    obs_half_range = (observed.max(axis=0) - observed.min(axis=0)) / 2.0
    tmpl_centered = template.points - template.points.mean(axis=0)
    tmpl_half_range = (tmpl_centered.max(axis=0) - tmpl_centered.min(axis=0)) / 2.0
    scale = np.where(tmpl_half_range > 1e-9, obs_half_range / np.maximum(tmpl_half_range, 1e-9), 0.0)
    snapped = tmpl_centered * scale + obs_mean

    best_rmse = float("inf")
    best = snapped
    for _ in range(max_iters):
        diff = observed[:, None, :] - snapped[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=2))
        obs_idx, tmpl_idx = linear_sum_assignment(cost)
        reorder = np.zeros(n, dtype=int)
        reorder[obs_idx] = tmpl_idx
        ordered_template = tmpl_centered[reorder]
        new_snapped, rmse = _axis_aligned_fit(ordered_template, observed)
        if rmse + 1e-9 >= best_rmse:
            break
        best_rmse = rmse
        best = new_snapped
        snapped = new_snapped

    return best, best_rmse
