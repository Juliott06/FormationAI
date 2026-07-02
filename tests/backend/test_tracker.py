from pathlib import Path

from app.pipeline.pose import (
    LandmarkPoint,
    compute_anchor_point,
    deduplicate_detections,
    generate_tile_regions,
    normalize_stage_proxy,
)
from app.pipeline.tracker import MultiPersonTracker
from app.pipeline.types import Detection
from app.schemas.jobs import (
    CoordinateSpaceMetadata,
    DancerPosition,
    DetectionSummary,
    FramePositions,
    PositionsResult,
    VideoMetadata,
)


def make_tracker() -> MultiPersonTracker:
    return MultiPersonTracker(
        max_missed_frames=10,
        distance_weight=1.0,
        area_weight=150.0,
        max_cost=400.0,
    )


def build_landmarks() -> list[LandmarkPoint]:
    return [LandmarkPoint(x=0.5, y=0.5, visibility=0.0) for _ in range(33)]


def test_tracker_preserves_ids_across_small_motion() -> None:
    tracker = make_tracker()
    first = tracker.update(
        [Detection(bbox=(10, 20, 50, 100), anchor_px=(35, 120), confidence=0.9)]
    )
    second = tracker.update(
        [Detection(bbox=(13, 22, 50, 100), anchor_px=(38, 122), confidence=0.88)]
    )

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id == 1


def test_tracker_expires_after_max_missed_frames() -> None:
    tracker = make_tracker()
    initial = tracker.update(
        [Detection(bbox=(10, 20, 50, 100), anchor_px=(35, 120), confidence=0.9)]
    )

    for _ in range(11):
        tracker.update([])

    new_detection = tracker.update(
        [Detection(bbox=(10, 20, 50, 100), anchor_px=(35, 120), confidence=0.9)]
    )

    assert initial[0].track_id == 1
    assert new_detection[0].track_id == 2


def test_anchor_prefers_ankles_then_hips_then_bbox() -> None:
    landmarks = build_landmarks()
    landmarks[27] = LandmarkPoint(x=0.2, y=0.9, visibility=0.95)
    landmarks[28] = LandmarkPoint(x=0.4, y=0.9, visibility=0.97)
    anchor = compute_anchor_point(landmarks, (100, 200, 80, 160), 1000, 1000, 0.5)
    assert anchor == (300, 900)

    landmarks = build_landmarks()
    landmarks[23] = LandmarkPoint(x=0.3, y=0.6, visibility=0.95)
    landmarks[24] = LandmarkPoint(x=0.5, y=0.6, visibility=0.94)
    anchor = compute_anchor_point(landmarks, (100, 200, 80, 160), 1000, 1000, 0.5)
    assert anchor == (400, 600)

    landmarks = build_landmarks()
    anchor = compute_anchor_point(landmarks, (100, 200, 80, 160), 1000, 1000, 0.5)
    assert anchor == (140, 360)


def test_normalized_stage_proxy() -> None:
    x, y = normalize_stage_proxy((140, 390), 1920, 1080)
    assert x == 0.072917
    assert y == 0.638889


def test_generate_tile_regions_with_overlap() -> None:
    tiles = generate_tile_regions(1920, 1080, rows=2, cols=3, overlap=0.2)

    assert len(tiles) == 6
    assert tiles[0].x0 == 0
    assert tiles[0].y0 == 0
    assert tiles[0].x1 > 640
    assert tiles[0].y1 > 540
    assert tiles[-1].x1 == 1920
    assert tiles[-1].y1 == 1080


def test_deduplicate_detections_prefers_highest_confidence_overlap() -> None:
    merged = deduplicate_detections(
        [
            Detection(bbox=(100, 100, 80, 160), anchor_px=(140, 250), confidence=0.95),
            Detection(bbox=(105, 102, 78, 158), anchor_px=(143, 248), confidence=0.82),
            Detection(bbox=(500, 120, 85, 170), anchor_px=(540, 280), confidence=0.91),
        ],
        iou_threshold=0.3,
    )

    assert len(merged) == 2
    assert merged[0].confidence == 0.95
    assert merged[1].anchor_px == (540, 280)


def test_detector_internal_timestamps_are_monotonic() -> None:
    from app.pipeline.pose import MediaPipePoseDetector

    detector = MediaPipePoseDetector(
        model_path=Path("unused.task"),
        num_poses=16,
        visibility_threshold=0.5,
    )

    first = detector._next_inference_timestamp_ms(0)
    second = detector._next_inference_timestamp_ms(0)
    third = detector._next_inference_timestamp_ms(33)
    fourth = detector._next_inference_timestamp_ms(33)

    assert first == 0
    assert second == 1
    assert third == 33
    assert fourth == 34


def test_schema_serialization_contract() -> None:
    payload = PositionsResult(
        job_id="job-1",
        video=VideoMetadata(
            filename="clip.mp4",
            fps=30.0,
            frame_count=10,
            width=1920,
            height=1080,
            duration_sec=0.333,
        ),
        coordinate_space=CoordinateSpaceMetadata(
            image_anchor_px="pixel anchor point in the source frame",
            normalized_stage_proxy="heuristic top-down proxy derived from image-space anchors",
        ),
        summary=DetectionSummary(
            expected_dancer_count=9,
            unique_track_ids=1,
            max_dancers_in_frame=1,
            average_dancers_per_frame=1.0,
            frames_with_detections=1,
            frames_below_expected=1,
            frames_meeting_expected=0,
        ),
        frames=[
            FramePositions(
                frame=0,
                timestamp_sec=0.0,
                dancers=[
                    DancerPosition(
                        id=1,
                        bbox=[100, 200, 80, 190],
                        anchor_px=[140, 390],
                        x=0.31,
                        y=0.22,
                        confidence=0.95,
                    )
                ],
            )
        ],
    )

    dumped = payload.model_dump(mode="json")
    assert dumped["job_id"] == "job-1"
    assert dumped["frames"][0]["dancers"][0]["anchor_px"] == [140, 390]
