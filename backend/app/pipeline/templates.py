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
            for c in range(cols):
                if len(grid_pts) >= n:
                    break
                x = (c / max(cols - 1, 1)) * 2 - 1 if cols > 1 else 0.0
                y = (r / max(rows - 1, 1)) * 2 - 1 if rows > 1 else 0.0
                grid_pts.append([x, y])
        templates.append(
            FormationTemplate(f"{rows}x{cols} grid", np.array(grid_pts))
        )

    # V shape (point at the front)
    arm_n = (n + 1) // 2
    other_n = n - arm_n
    left_arm = [
        [-i / max(arm_n - 1, 1), -(arm_n - 1 - i) / max(arm_n - 1, 1)]
        for i in range(arm_n)
    ]
    right_arm = [
        [i / max(other_n, 1), -(other_n - i) / max(other_n, 1)]
        for i in range(1, other_n + 1)
    ]
    v_pts = np.array(left_arm + right_arm)
    templates.append(FormationTemplate("V", v_pts))

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
        templates.append(FormationTemplate("Triangle", np.array(triangle_pts)))

    return templates


def _procrustes_fit(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    """Find best rigid-similarity transform (rotation + uniform scale + translation)
    that maps source onto target. Returns transformed source and RMSE distance."""
    src_mean = source.mean(axis=0)
    tgt_mean = target.mean(axis=0)
    src_c = source - src_mean
    tgt_c = target - tgt_mean

    H = src_c.T @ tgt_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    src_norm_sq = float((src_c ** 2).sum())
    scale = float(S.sum() / src_norm_sq) if src_norm_sq > 1e-9 else 1.0

    transformed = scale * src_c @ R.T + tgt_mean
    rmse = float(np.sqrt(((transformed - target) ** 2).sum(axis=1).mean()))
    return transformed, rmse


def fit_template(
    template: FormationTemplate, observed: np.ndarray, *, max_iters: int = 4
) -> tuple[np.ndarray, float]:
    """Iteratively fit template points to observed points (same N).
    Returns (snapped_positions in observed order, RMSE)."""
    n = observed.shape[0]
    if template.points.shape[0] != n or n < 2:
        return observed.copy(), float("inf")

    # Initialize: center template on observed and rough-scale to match spread.
    obs_mean = observed.mean(axis=0)
    obs_spread = float(np.linalg.norm(observed - obs_mean))
    tmpl_centered = template.points - template.points.mean(axis=0)
    tmpl_spread = float(np.linalg.norm(tmpl_centered))
    if tmpl_spread > 1e-9 and obs_spread > 1e-9:
        snapped = tmpl_centered * (obs_spread / tmpl_spread) + obs_mean
    else:
        snapped = tmpl_centered + obs_mean

    best_rmse = float("inf")
    for _ in range(max_iters):
        # Assign each observed dancer to the nearest snapped template point.
        diff = observed[:, None, :] - snapped[None, :, :]
        cost = np.sqrt((diff ** 2).sum(axis=2))
        obs_idx, tmpl_idx = linear_sum_assignment(cost)

        # Reorder template points to match observed[i]'s order.
        reorder = np.zeros(n, dtype=int)
        reorder[obs_idx] = tmpl_idx
        ordered_template = snapped[reorder]

        # Procrustes alignment in observed space.
        new_snapped, rmse = _procrustes_fit(ordered_template, observed)
        if rmse + 1e-6 >= best_rmse:
            snapped = new_snapped
            best_rmse = rmse
            break
        snapped = new_snapped
        best_rmse = rmse

    return snapped, best_rmse
