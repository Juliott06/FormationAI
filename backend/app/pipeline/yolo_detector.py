from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.pipeline.types import Detection, TrackedDetection


@dataclass(slots=True)
class RawBox:
    xyxy: tuple[float, float, float, float]
    confidence: float


def _bottom_center_anchor(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y, width, height = bbox
    return x + (width // 2), y + height


def _clip_bbox(
    xyxy: tuple[float, float, float, float],
    *,
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = xyxy
    left = min(max(int(round(x1)), 0), max(frame_width - 1, 0))
    top = min(max(int(round(y1)), 0), max(frame_height - 1, 0))
    right = min(max(int(round(x2)), 0), max(frame_width - 1, 0))
    bottom = min(max(int(round(y2)), 0), max(frame_height - 1, 0))
    return (
        left,
        top,
        max(right - left, 1),
        max(bottom - top, 1),
    )


def raw_boxes_to_detections(
    raw_boxes: list[RawBox],
    *,
    frame_width: int,
    frame_height: int,
) -> list[Detection]:
    detections: list[Detection] = []
    for raw in raw_boxes:
        bbox = _clip_bbox(raw.xyxy, frame_width=frame_width, frame_height=frame_height)
        detections.append(
            Detection(
                bbox=bbox,
                anchor_px=_bottom_center_anchor(bbox),
                confidence=round(min(max(raw.confidence, 0.0), 1.0), 4),
            )
        )
    return detections


class YoloPersonDetector:
    def __init__(
        self,
        *,
        model_name: str,
        confidence_threshold: float,
        iou_threshold: float,
        image_size: int,
        max_detections: int,
        device: str,
        tracker_config: str,
    ) -> None:
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.image_size = image_size
        self.max_detections = max_detections
        self.device = device
        self.tracker_config = self._resolve_tracker_config(tracker_config)
        self._model: Any | None = None

    @staticmethod
    def _resolve_tracker_config(name: str) -> str:
        local_path = Path(__file__).parent / "trackers" / name
        if local_path.is_file():
            return str(local_path)
        return name

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        from ultralytics import YOLO  # Imported lazily for lightweight test imports.

        self._model = YOLO(self.model_name)

    def track(self, frame_bgr: Any, timestamp_ms: int) -> list[TrackedDetection]:
        del timestamp_ms  # Ultralytics tracker uses internal frame counters.
        self._ensure_model()
        assert self._model is not None

        results = self._model.track(
            source=frame_bgr,
            persist=True,
            classes=[0],
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.image_size,
            max_det=self.max_detections,
            device=self.device,
            tracker=self.tracker_config,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            return []

        xyxy_rows = boxes.xyxy.cpu().tolist()
        conf_rows = boxes.conf.cpu().tolist()
        id_rows = boxes.id.int().cpu().tolist()

        frame_height, frame_width = frame_bgr.shape[:2]
        tracked: list[TrackedDetection] = []
        for row, confidence, track_id in zip(xyxy_rows, conf_rows, id_rows, strict=True):
            bbox = _clip_bbox(
                (float(row[0]), float(row[1]), float(row[2]), float(row[3])),
                frame_width=frame_width,
                frame_height=frame_height,
            )
            tracked.append(
                TrackedDetection(
                    track_id=int(track_id),
                    bbox=bbox,
                    anchor_px=_bottom_center_anchor(bbox),
                    confidence=round(min(max(float(confidence), 0.0), 1.0), 4),
                )
            )
        return tracked
