from __future__ import annotations

from app.pipeline.cotracker_pipeline import tracked_points_to_frames
from app.schemas.jobs import DancerClick, VideoMetadata
from app.services.cotracker import TrackedPoint


def _meta(frame_count: int = 3, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        filename="test.mp4",
        fps=30.0,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=round(frame_count / 30.0, 3),
    )


def _pt(frame: int, x: int, y: int, visible: bool = True) -> TrackedPoint:
    return TrackedPoint(frame=frame, x=x, y=y, visible=visible)


def test_single_click_per_dancer_no_projection():
    # One click on D1 at foot (928, 800). All frames visible.
    clicks = [DancerClick(name="D1", x=928, y=800)]
    tracks = [[_pt(0, 928, 800), _pt(1, 930, 802), _pt(2, 932, 804)]]
    frames = tracked_points_to_frames(tracks, clicks, _meta(3))
    assert len(frames) == 3
    for fi, f in enumerate(frames):
        assert len(f.dancers) == 1
        d = f.dancers[0]
        # No projection: anchor y == tracked y
        assert d.anchor_px[1] == [800, 802, 804][fi]
        assert d.id == 1


def test_label_order_follows_unique_name_first_occurrence():
    # Multi-click on D1, then single click on D2. D1 should be id=1, D2 id=2.
    clicks = [
        DancerClick(name="D1", x=928, y=400),  # head
        DancerClick(name="D1", x=928, y=600),  # torso
        DancerClick(name="D1", x=928, y=800),  # foot
        DancerClick(name="D2", x=500, y=900),
    ]
    tracks = [
        [_pt(0, 928, 400)],
        [_pt(0, 928, 600)],
        [_pt(0, 928, 800)],
        [_pt(0, 500, 900)],
    ]
    frames = tracked_points_to_frames(tracks, clicks, _meta(1))
    ids = sorted(d.id for d in frames[0].dancers)
    assert ids == [1, 2]


def test_multi_click_with_foot_visible_uses_foot_y():
    # D1: head 400, torso 600, foot 800 at key frame. All visible at frame 0.
    clicks = [
        DancerClick(name="D1", x=928, y=400),
        DancerClick(name="D1", x=928, y=600),
        DancerClick(name="D1", x=928, y=800),
    ]
    # Frame 0: all three at exactly their key-frame positions.
    tracks = [
        [_pt(0, 928, 400)],
        [_pt(0, 928, 600)],
        [_pt(0, 928, 800)],
    ]
    frames = tracked_points_to_frames(tracks, clicks, _meta(1))
    d = frames[0].dancers[0]
    # Projected feet: head→800, torso→800, foot→800. Median = 800.
    assert d.anchor_px[1] == 800


def test_multi_click_with_only_head_visible_projects_to_foot():
    # D1: head 400, foot 800 at key frame. foot_offset for head = 400.
    # Frame 0: only head visible at y=420. Projected foot = 420 + 400 = 820.
    clicks = [
        DancerClick(name="D1", x=928, y=400),  # head
        DancerClick(name="D1", x=928, y=800),  # foot
    ]
    tracks = [
        [_pt(0, 928, 420, visible=True)],   # head visible
        [_pt(0, 928, 850, visible=False)],  # foot occluded
    ]
    frames = tracked_points_to_frames(tracks, clicks, _meta(1))
    d = frames[0].dancers[0]
    # Projected foot from head: 420 + (800-400) = 820, NOT just 420
    assert d.anchor_px[1] == 820


def test_multi_click_with_only_foot_visible_uses_foot_y():
    clicks = [
        DancerClick(name="D1", x=928, y=400),
        DancerClick(name="D1", x=928, y=800),
    ]
    tracks = [
        [_pt(0, 928, 420, visible=False)],
        [_pt(0, 928, 810, visible=True)],
    ]
    d = tracked_points_to_frames(tracks, clicks, _meta(1))[0].dancers[0]
    # Foot offset for foot click = 0, so projected = 810
    assert d.anchor_px[1] == 810


def test_no_visible_points_drops_dancer():
    clicks = [DancerClick(name="D1", x=928, y=800)]
    tracks = [[_pt(0, 928, 800, visible=False)]]
    frames = tracked_points_to_frames(tracks, clicks, _meta(1))
    assert frames[0].dancers == []


def test_median_x_across_visible_points():
    # 3 visible points at different x. Median should pick middle.
    clicks = [
        DancerClick(name="D1", x=900, y=400),
        DancerClick(name="D1", x=920, y=600),
        DancerClick(name="D1", x=940, y=800),
    ]
    tracks = [
        [_pt(0, 800, 400)],
        [_pt(0, 920, 600)],
        [_pt(0, 1100, 800)],
    ]
    d = tracked_points_to_frames(tracks, clicks, _meta(1))[0].dancers[0]
    # xs sorted: [800, 920, 1100], median index 1 = 920
    assert d.anchor_px[0] == 920


def test_two_dancers_each_with_multi_click():
    # D1: 2 points. D2: 2 points. Identity must stay separate.
    clicks = [
        DancerClick(name="D1", x=300, y=400),
        DancerClick(name="D1", x=300, y=800),
        DancerClick(name="D2", x=1200, y=400),
        DancerClick(name="D2", x=1200, y=800),
    ]
    tracks = [
        [_pt(0, 300, 410)],
        [_pt(0, 300, 810)],
        [_pt(0, 1200, 410)],
        [_pt(0, 1200, 810)],
    ]
    f = tracked_points_to_frames(tracks, clicks, _meta(1))[0]
    by_id = {d.id: d for d in f.dancers}
    assert by_id[1].anchor_px[0] == 300  # D1 stays on left
    assert by_id[2].anchor_px[0] == 1200  # D2 stays on right
