from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import linear_sum_assignment

from app.pipeline.appearance import _cosine_similarity
from app.schemas.jobs import FramePositions


logger = logging.getLogger("uvicorn.error")


def build_cooccurring_pairs(frames: list[FramePositions]) -> set[frozenset[int]]:
    pairs: set[frozenset[int]] = set()
    for f in frames:
        ids = [d.id for d in f.dancers]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                if ids[i] != ids[j]:
                    pairs.add(frozenset((ids[i], ids[j])))
    return pairs


def assign_unlabeled_to_anchors(
    histograms: dict[int, np.ndarray],
    labels: dict[int, str],
    cooccurring: set[frozenset[int]],
    *,
    similarity_threshold: float = 0.85,
) -> dict[int, int]:
    """For each unlabeled track ID, find the most similar labeled anchor and
    return a rename map (unlabeled_id -> labeled_anchor_id).

    Respects cooccurring constraint: never merges two IDs that share a frame.
    Skips matches below similarity threshold (low-confidence stay unlabeled)."""
    anchor_ids = sorted(labels.keys())
    if not anchor_ids:
        return {}

    unlabeled_ids = [tid for tid in histograms if tid not in labels]
    if not unlabeled_ids:
        return {}

    rename: dict[int, int] = {}
    for tid in unlabeled_ids:
        if tid not in histograms:
            continue
        best_anchor = None
        best_sim = similarity_threshold
        for anchor_id in anchor_ids:
            if anchor_id not in histograms:
                continue
            if frozenset((tid, anchor_id)) in cooccurring:
                continue
            sim = _cosine_similarity(histograms[tid], histograms[anchor_id])
            if sim > best_sim:
                best_sim = sim
                best_anchor = anchor_id
        if best_anchor is not None:
            rename[tid] = best_anchor
            logger.info(
                "label match: ID %d -> %d (%s, similarity=%.3f)",
                tid, best_anchor, labels[best_anchor], best_sim,
            )
    return rename
