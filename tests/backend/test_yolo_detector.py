from app.pipeline.yolo_detector import RawBox, raw_boxes_to_detections


def test_raw_boxes_to_detections_uses_bottom_center_anchor() -> None:
    detections = raw_boxes_to_detections(
        [
            RawBox(xyxy=(100.2, 50.4, 180.7, 210.9), confidence=0.9321),
        ],
        frame_width=640,
        frame_height=360,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.bbox == (100, 50, 81, 161)
    assert detection.anchor_px == (140, 211)
    assert detection.confidence == 0.9321


def test_raw_boxes_are_clipped_to_frame_bounds() -> None:
    detections = raw_boxes_to_detections(
        [
            RawBox(xyxy=(-20.0, -15.0, 700.0, 500.0), confidence=1.2),
        ],
        frame_width=640,
        frame_height=360,
    )

    detection = detections[0]
    assert detection.bbox == (0, 0, 639, 359)
    assert detection.anchor_px == (319, 359)
    assert detection.confidence == 1.0
