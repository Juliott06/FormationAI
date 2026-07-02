from __future__ import annotations

import logging
from typing import Any

import numpy as np


logger = logging.getLogger("uvicorn.error")


def crop_outfit_histogram(frame: Any, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
    """Extract an HSV histogram from the upper-body region of a detection.

    Upper half emphasises shirt/jacket color (less affected by leg motion blur
    or shadow at the floor than full-body crops)."""
    import cv2

    x, y, w, h = bbox
    if w < 8 or h < 16:
        return None
    crop = frame[y : y + h, x : x + w]
    if crop.size == 0:
        return None
    upper = crop[: max(h // 2, 1)]
    if upper.size == 0:
        return None

    hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
    # Coarse 2D histogram over hue & saturation; ignore value to be lighting-tolerant.
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L2)
    return hist.flatten().astype(np.float32)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-9:
        return 0.0
    return float(np.dot(a, b) / denom)


def merge_by_appearance(
    histograms_by_id: dict[int, list[np.ndarray]],
    *,
    threshold: float,
    target_count: int | None = None,
    cooccurring_pairs: set[frozenset[int]] | None = None,
) -> dict[int, int]:
    """Build a rename map (remove_id -> keep_id) for tracks with very similar
    average outfit color. Stops merging once unique-id count would drop below
    `target_count`. Skips any pair whose IDs ever appear in the same frame
    (those are definitely different real dancers)."""
    if threshold <= 0 or not histograms_by_id:
        return {}

    mean_hists: dict[int, np.ndarray] = {}
    for tid, hists in histograms_by_id.items():
        valid = [h for h in hists if h is not None and h.size > 0]
        if valid:
            mean_hists[tid] = np.mean(valid, axis=0)

    track_ids = sorted(mean_hists.keys())
    pairs: list[tuple[int, int, float]] = []
    for i, a in enumerate(track_ids):
        for b in track_ids[i + 1 :]:
            sim = _cosine_similarity(mean_hists[a], mean_hists[b])
            if sim >= threshold:
                pairs.append((a, b, sim))
    pairs.sort(key=lambda p: p[2], reverse=True)

    rename: dict[int, int] = {}
    active_ids = set(track_ids)
    skipped_cooccurring = 0

    for a, b, sim in pairs:
        if target_count is not None and len(active_ids) <= target_count:
            break
        if cooccurring_pairs is not None and frozenset((a, b)) in cooccurring_pairs:
            skipped_cooccurring += 1
            continue
        ea = a
        while ea in rename:
            ea = rename[ea]
        eb = b
        while eb in rename:
            eb = rename[eb]
        if ea == eb:
            continue
        if cooccurring_pairs is not None and frozenset((ea, eb)) in cooccurring_pairs:
            skipped_cooccurring += 1
            continue
        keep, remove = (ea, eb) if ea < eb else (eb, ea)
        rename[remove] = keep
        active_ids.discard(remove)
        logger.info(
            "appearance merge: ID %d -> %d (cosine similarity=%.3f)", remove, keep, sim
        )

    if skipped_cooccurring > 0:
        logger.info(
            "appearance merge skipped %d co-occurring pair(s) (different real dancers)",
            skipped_cooccurring,
        )

    return rename
