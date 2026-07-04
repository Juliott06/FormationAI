from __future__ import annotations

from app.pipeline.foot_assist import match_points_to_boxes, representative_points
from app.pipeline.types import Detection
from app.services.cotracker import TrackedPoint


def _box(x: int, y: int, w: int, h: int, conf: float = 0.9) -> Detection:
    return Detection(bbox=(x, y, w, h), anchor_px=(x + w // 2, y + h), confidence=conf)


def test_point_inside_box_matches_bbox_bottom_center():
    # Torso point at (500, 400) inside a person box (450, 300, 100, 300):
    # foot anchor must be the box's bottom-center (500, 600), NOT the point.
    boxes = [_box(450, 300, 100, 300)]
    matched = match_points_to_boxes({"Mina": (500, 400)}, boxes)
    assert matched == {"Mina": (500, 600)}


def test_each_box_claimed_once_nearest_point_wins():
    # Two dancers crossing: both points inside one box; only the nearer-to-
    # center point claims it, the other stays unmatched (keeps its offset).
    boxes = [_box(400, 300, 100, 300)]  # center (450, 450)
    points = {"A": (455, 460), "B": (405, 310)}  # A is much nearer the center
    matched = match_points_to_boxes(points, boxes)
    assert "A" in matched
    assert "B" not in matched


def test_point_outside_all_boxes_is_unmatched():
    boxes = [_box(100, 100, 80, 200)]
    matched = match_points_to_boxes({"Mina": (900, 500)}, boxes)
    assert matched == {}


def test_two_dancers_two_boxes_both_match():
    boxes = [_box(100, 300, 100, 300), _box(700, 300, 100, 300)]
    points = {"L": (150, 450), "R": (750, 450)}
    matched = match_points_to_boxes(points, boxes)
    assert matched["L"] == (150, 600)
    assert matched["R"] == (750, 600)


def test_representative_points_median_of_visible_multiclick():
    # One dancer, two click points; frame 0 both visible, frame 1 only one.
    tracks = [
        [TrackedPoint(0, 100, 200, True), TrackedPoint(1, 110, 210, False)],
        [TrackedPoint(0, 120, 240, True), TrackedPoint(1, 130, 250, True)],
    ]
    rep = representative_points(tracks, ["Mina", "Mina"])
    assert rep["Mina"][0] == (120, 240)  # median of two points (upper median)
    assert rep["Mina"][1] == (130, 250)  # only the visible one


def test_representative_points_none_when_all_hidden():
    tracks = [[TrackedPoint(0, 100, 200, False)]]
    rep = representative_points(tracks, ["Mina"])
    assert rep["Mina"][0] is None
