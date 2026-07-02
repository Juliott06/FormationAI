from __future__ import annotations

from app.pipeline.named_matcher import match_names_to_yolo
from app.pipeline.types import TrackedDetection
from app.pipeline.utils import box_iou
from app.services.locate_anything import BBoxPx


def _yolo(track_id: int, x: int, y: int, w: int, h: int, conf: float = 0.9) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        bbox=(x, y, w, h),
        anchor_px=(x + w // 2, y + h),
        confidence=conf,
    )


def test_box_iou_basic():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (100, 100, 110, 110)) == 0.0
    # half-overlap on x axis
    assert abs(box_iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-6


def test_high_iou_la_match_assigns_name_to_yolo_track():
    yolo = [_yolo(track_id=42, x=100, y=200, w=80, h=160)]
    # LA box mostly overlaps the YOLO box
    name_boxes = {"Yeji": BBoxPx(105, 205, 175, 355)}
    roster = ["Yeji", "Lia"]

    placed, last_known = match_names_to_yolo(name_boxes, yolo, {}, roster)

    assert len(placed) == 1
    det = placed[0]
    assert det.track_id == 1  # Yeji is index 0 → id 1
    assert "Yeji" in last_known


def test_low_iou_la_box_does_not_match_then_no_fallback_yields_unplaced():
    yolo = [_yolo(track_id=7, x=0, y=0, w=20, h=20)]
    # LA box nowhere near YOLO box; no last_known
    name_boxes = {"Yeji": BBoxPx(500, 500, 560, 600)}
    roster = ["Yeji"]

    placed, _ = match_names_to_yolo(name_boxes, yolo, {}, roster)

    # LA box doesn't match YOLO; no fallback because no last_known_pos → unplaced
    assert placed == []


def test_fallback_via_last_known_position():
    # LA returns nothing for Yeji; her last known position is near a YOLO box
    yolo = [_yolo(track_id=5, x=300, y=400, w=50, h=100)]  # anchor at (325, 500)
    last_known = {"Yeji": (320, 495)}  # close to YOLO anchor
    roster = ["Yeji"]

    placed, new_last_known = match_names_to_yolo({}, yolo, last_known, roster)

    assert len(placed) == 1
    assert placed[0].track_id == 1
    # The YOLO box is reused for the fallback
    assert placed[0].bbox == (300, 400, 50, 100)
    assert new_last_known["Yeji"] == (325, 500)


def test_fallback_skipped_when_yolo_too_far():
    yolo = [_yolo(track_id=5, x=900, y=900, w=50, h=100)]  # anchor (925, 1000)
    last_known = {"Yeji": (10, 10)}  # very far away
    roster = ["Yeji"]

    placed, new_last_known = match_names_to_yolo(
        {}, yolo, last_known, roster, fallback_max_distance_px=100.0
    )

    assert placed == []
    # last_known is carried forward for future frames even if not placed
    assert new_last_known["Yeji"] == (10, 10)


def test_two_names_overlap_same_yolo_box_greedy_winner_takes():
    # One YOLO box; two LA names overlap it; the higher-IoU name wins
    yolo = [_yolo(track_id=1, x=0, y=0, w=100, h=100)]
    name_boxes = {
        "Yeji": BBoxPx(0, 0, 100, 100),   # full overlap, IoU 1.0
        "Lia": BBoxPx(50, 0, 150, 100),   # partial overlap
    }
    roster = ["Yeji", "Lia"]

    placed, _ = match_names_to_yolo(name_boxes, yolo, {}, roster)

    placed_by_name = {p.track_id: p for p in placed}
    # Yeji (id=1) wins the YOLO box; Lia has no other box and no last_known → unplaced
    assert 1 in placed_by_name
    assert 2 not in placed_by_name


def test_yolo_not_in_roster_is_ignored():
    yolo = [_yolo(track_id=99, x=0, y=0, w=50, h=50)]
    name_boxes = {"Stranger": BBoxPx(0, 0, 50, 50)}  # not in roster
    roster = ["Yeji"]

    placed, last_known = match_names_to_yolo(name_boxes, yolo, {}, roster)

    assert placed == []
    assert last_known == {}


def test_last_known_carried_forward_for_unplaced_names():
    # Yeji was at (100, 100) before; this frame has no LA box and no nearby YOLO
    last_known = {"Yeji": (100, 100), "Lia": (500, 500)}
    yolo: list[TrackedDetection] = []
    roster = ["Yeji", "Lia"]

    placed, new_last_known = match_names_to_yolo({}, yolo, last_known, roster)

    assert placed == []
    # Carried forward so the next frame still has a fallback target
    assert new_last_known["Yeji"] == (100, 100)
    assert new_last_known["Lia"] == (500, 500)


def test_la_match_uses_la_box_not_yolo_box():
    # LA box and YOLO box overlap but are not identical — placed should use LA box
    yolo = [_yolo(track_id=42, x=100, y=100, w=100, h=200)]
    name_boxes = {"Yeji": BBoxPx(110, 110, 195, 290)}  # slightly inside
    roster = ["Yeji"]

    placed, _ = match_names_to_yolo(name_boxes, yolo, {}, roster)

    assert len(placed) == 1
    # bbox stored as xywh, derived from LA's xyxy
    assert placed[0].bbox[0] == 110
    assert placed[0].bbox[1] == 110
    assert placed[0].bbox[2] == 85  # 195 - 110
    assert placed[0].bbox[3] == 180  # 290 - 110
