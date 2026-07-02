from __future__ import annotations

from app.pipeline.types import TrackedDetection
from app.pipeline.utils import box_iou, xywh_to_xyxy
from app.services.locate_anything import BBoxPx


def _bottom_center_xyxy(box: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, y2)


def _bbox_xyxy_to_xywh(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return (x1, y1, max(x2 - x1, 1), max(y2 - y1, 1))


def match_names_to_yolo(
    name_boxes: dict[str, BBoxPx],
    yolo_tracked: list[TrackedDetection],
    last_known_pos: dict[str, tuple[int, int]],
    roster: list[str],
    *,
    iou_threshold: float = 0.3,
    fallback_max_distance_px: float = 200.0,
) -> tuple[list[TrackedDetection], dict[str, tuple[int, int]]]:
    """Map roster names to per-frame TrackedDetections.

    Strategy:
      1. Greedy IoU pairing of LocateAnything name->box with YOLO detections.
         Highest-IoU pair wins, no double-claim of either side. The matched
         detection carries the LocateAnything box (tighter / text-grounded).
      2. Names still unplaced fall back: look up their last_known_pos and pick
         the nearest UNCLAIMED YOLO box within fallback_max_distance_px.
      3. Names with no LA box and no last_known_pos (or no nearby YOLO box) are
         simply not placed this frame — existing interpolation fills the gap.
      4. Returns (placed_detections, new_last_known_pos) — caller passes the new
         dict back in on the next frame.

    track_id is stable across the whole clip: roster.index(name) + 1 (1-based to
    satisfy DancerPosition.id >= 1).
    """
    name_to_id: dict[str, int] = {name: idx + 1 for idx, name in enumerate(roster)}
    yolo_xyxy = [xywh_to_xyxy(d.bbox) for d in yolo_tracked]

    placed: dict[str, TrackedDetection] = {}
    claimed_yolo: set[int] = set()
    new_last_known: dict[str, tuple[int, int]] = {}

    pairings: list[tuple[float, str, int]] = []
    for name, la_box in name_boxes.items():
        if name not in name_to_id:
            continue
        la_xyxy = la_box.as_tuple()
        for yi, yxyxy in enumerate(yolo_xyxy):
            iou = box_iou(la_xyxy, yxyxy)
            if iou >= iou_threshold:
                pairings.append((iou, name, yi))
    pairings.sort(reverse=True)

    for iou, name, yi in pairings:
        if name in placed or yi in claimed_yolo:
            continue
        la_xyxy = name_boxes[name].as_tuple()
        anchor = _bottom_center_xyxy(la_xyxy)
        placed[name] = TrackedDetection(
            track_id=name_to_id[name],
            bbox=_bbox_xyxy_to_xywh(la_xyxy),
            anchor_px=anchor,
            confidence=round(min(max(iou, 0.0), 1.0), 4),
        )
        claimed_yolo.add(yi)
        new_last_known[name] = anchor

    for name in roster:
        if name in placed:
            continue
        last = last_known_pos.get(name)
        if last is None:
            continue
        best_idx: int | None = None
        best_dist_sq: float = fallback_max_distance_px * fallback_max_distance_px
        for yi, yd in enumerate(yolo_tracked):
            if yi in claimed_yolo:
                continue
            cx, cy = yd.anchor_px
            dist_sq = (cx - last[0]) ** 2 + (cy - last[1]) ** 2
            if dist_sq <= best_dist_sq:
                best_dist_sq = dist_sq
                best_idx = yi
        if best_idx is None:
            continue
        fallback_det = yolo_tracked[best_idx]
        placed[name] = TrackedDetection(
            track_id=name_to_id[name],
            bbox=fallback_det.bbox,
            anchor_px=fallback_det.anchor_px,
            confidence=fallback_det.confidence,
        )
        claimed_yolo.add(best_idx)
        new_last_known[name] = fallback_det.anchor_px

    for name, last in last_known_pos.items():
        if name in new_last_known:
            continue
        if name in name_to_id:
            new_last_known[name] = last

    return list(placed.values()), new_last_known
