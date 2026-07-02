from __future__ import annotations

from app.pipeline.sam2_pipeline import tracks_to_frames
from app.schemas.jobs import DancerClick, VideoMetadata
from app.services.sam2 import FrameBox


def _meta(frame_count: int = 5, width: int = 1920, height: int = 1080) -> VideoMetadata:
    return VideoMetadata(
        filename="test.mp4",
        fps=30.0,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_sec=round(frame_count / 30.0, 3),
    )


def _click(name: str, x: int = 100, y: int = 200) -> DancerClick:
    return DancerClick(name=name, x=x, y=y)


def test_full_tracks_produce_one_dancer_per_frame_per_name():
    clicks = [_click("Yeji"), _click("Lia")]
    tracks = {
        "Yeji": [FrameBox(frame=i, x1=100, y1=200, x2=180, y2=400, score=0.9) for i in range(3)],
        "Lia": [FrameBox(frame=i, x1=300, y1=200, x2=380, y2=400, score=0.9) for i in range(3)],
    }
    meta = _meta(frame_count=3)

    frames = tracks_to_frames(tracks, clicks, meta)

    assert len(frames) == 3
    for f in frames:
        assert len(f.dancers) == 2
        ids = {d.id for d in f.dancers}
        assert ids == {1, 2}  # Yeji=1, Lia=2 (roster order)


def test_missing_frames_for_a_name_are_omitted_not_zeroed():
    clicks = [_click("Yeji")]
    # SAM2 only returned 2 of the 5 frames
    tracks = {
        "Yeji": [
            FrameBox(frame=0, x1=100, y1=200, x2=180, y2=400, score=0.9),
            FrameBox(frame=4, x1=110, y1=200, x2=190, y2=400, score=0.9),
        ],
    }
    meta = _meta(frame_count=5)

    frames = tracks_to_frames(tracks, clicks, meta)

    assert len(frames) == 5
    assert len(frames[0].dancers) == 1
    assert frames[1].dancers == []  # SAM2 lost her here
    assert frames[2].dancers == []
    assert frames[3].dancers == []
    assert len(frames[4].dancers) == 1


def test_bbox_converted_xyxy_to_xywh_and_anchor_is_bottom_center():
    clicks = [_click("Yeji")]
    tracks = {
        "Yeji": [FrameBox(frame=0, x1=100, y1=200, x2=180, y2=400, score=0.9)],
    }
    meta = _meta(frame_count=1)

    frames = tracks_to_frames(tracks, clicks, meta)
    dancer = frames[0].dancers[0]

    # bbox stored as [x, y, w, h]
    assert dancer.bbox == [100, 200, 80, 200]
    # anchor is bottom-center: ((100+180)/2, 400)
    assert dancer.anchor_px == [140, 400]


def test_track_id_is_roster_index_plus_one_in_order():
    clicks = [_click("Yeji"), _click("Lia"), _click("Ryujin")]
    tracks = {
        "Yeji": [FrameBox(frame=0, x1=0, y1=0, x2=10, y2=10, score=0.9)],
        "Lia": [FrameBox(frame=0, x1=20, y1=0, x2=30, y2=10, score=0.9)],
        "Ryujin": [FrameBox(frame=0, x1=40, y1=0, x2=50, y2=10, score=0.9)],
    }
    meta = _meta(frame_count=1)

    frames = tracks_to_frames(tracks, clicks, meta)

    by_id = {d.id: d for d in frames[0].dancers}
    assert by_id[1].confidence == 0.9  # Yeji
    assert 2 in by_id  # Lia
    assert 3 in by_id  # Ryujin


def test_extra_tracks_not_in_roster_are_ignored():
    clicks = [_click("Yeji")]
    tracks = {
        "Yeji": [FrameBox(frame=0, x1=0, y1=0, x2=10, y2=10, score=0.9)],
        "Stranger": [FrameBox(frame=0, x1=20, y1=0, x2=30, y2=10, score=0.9)],
    }
    meta = _meta(frame_count=1)

    frames = tracks_to_frames(tracks, clicks, meta)
    assert len(frames[0].dancers) == 1
    assert frames[0].dancers[0].id == 1


def test_empty_tracks_produces_empty_dancers_per_frame():
    clicks = [_click("Yeji")]
    meta = _meta(frame_count=3)

    frames = tracks_to_frames({}, clicks, meta)

    assert len(frames) == 3
    assert all(f.dancers == [] for f in frames)


def test_confidence_clamped_to_unit_interval():
    clicks = [_click("Yeji")]
    tracks = {
        "Yeji": [
            FrameBox(frame=0, x1=0, y1=0, x2=10, y2=10, score=1.5),  # too high
            FrameBox(frame=1, x1=0, y1=0, x2=10, y2=10, score=-0.2),  # too low
        ],
    }
    meta = _meta(frame_count=2)

    frames = tracks_to_frames(tracks, clicks, meta)
    assert frames[0].dancers[0].confidence == 1.0
    assert frames[1].dancers[0].confidence == 0.0
