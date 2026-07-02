from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from pathlib import Path
from typing import Any

from app.pipeline.types import Detection


LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_ANKLE = 27
RIGHT_ANKLE = 28


@dataclass(slots=True)
class LandmarkPoint:
    x: float
    y: float
    visibility: float


@dataclass(slots=True)
class TileRegion:
    x0: int
    y0: int
    x1: int
    y1: int


def _to_pixel(point: LandmarkPoint, frame_width: int, frame_height: int) -> tuple[int, int]:
    x = min(max(int(round(point.x * frame_width)), 0), frame_width - 1)
    y = min(max(int(round(point.y * frame_height)), 0), frame_height - 1)
    return x, y


def _is_visible(point: LandmarkPoint | None, visibility_threshold: float) -> bool:
    return point is not None and point.visibility >= visibility_threshold


def _mean_pixel(
    points: list[LandmarkPoint],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    avg = LandmarkPoint(
        x=float(sum(xs) / len(xs)),
        y=float(sum(ys) / len(ys)),
        visibility=float(sum(point.visibility for point in points) / len(points)),
    )
    return _to_pixel(avg, frame_width, frame_height)


def compute_anchor_point(
    landmarks: list[LandmarkPoint],
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
    visibility_threshold: float,
) -> tuple[int, int]:
    left_ankle = landmarks[LEFT_ANKLE] if len(landmarks) > LEFT_ANKLE else None
    right_ankle = landmarks[RIGHT_ANKLE] if len(landmarks) > RIGHT_ANKLE else None
    if _is_visible(left_ankle, visibility_threshold) and _is_visible(right_ankle, visibility_threshold):
        return _mean_pixel([left_ankle, right_ankle], frame_width, frame_height)

    left_hip = landmarks[LEFT_HIP] if len(landmarks) > LEFT_HIP else None
    right_hip = landmarks[RIGHT_HIP] if len(landmarks) > RIGHT_HIP else None
    if _is_visible(left_hip, visibility_threshold) and _is_visible(right_hip, visibility_threshold):
        return _mean_pixel([left_hip, right_hip], frame_width, frame_height)

    x, y, width, height = bbox
    return (x + width // 2, y + height)


def normalize_stage_proxy(anchor_px: tuple[int, int], frame_width: int, frame_height: int) -> tuple[float, float]:
    x = anchor_px[0] / max(frame_width, 1)
    y = 1.0 - (anchor_px[1] / max(frame_height, 1))
    return round(x, 6), round(y, 6)


def generate_tile_regions(
    frame_width: int,
    frame_height: int,
    *,
    rows: int,
    cols: int,
    overlap: float,
) -> list[TileRegion]:
    if rows <= 0 or cols <= 0:
        return []

    overlap = min(max(overlap, 0.0), 0.9)
    cell_width = frame_width / cols
    cell_height = frame_height / rows
    horizontal_pad = (cell_width * overlap) / 2
    vertical_pad = (cell_height * overlap) / 2

    tiles: list[TileRegion] = []
    for row in range(rows):
        for col in range(cols):
            x0 = max(0, floor((col * cell_width) - horizontal_pad))
            y0 = max(0, floor((row * cell_height) - vertical_pad))
            x1 = min(frame_width, ceil(((col + 1) * cell_width) + horizontal_pad))
            y1 = min(frame_height, ceil(((row + 1) * cell_height) + vertical_pad))
            tiles.append(TileRegion(x0=x0, y0=y0, x1=max(x1, x0 + 1), y1=max(y1, y0 + 1)))
    return tiles


def _bbox_bottom_right(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    return bbox[0] + bbox[2], bbox[1] + bbox[3]


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_x2, left_y2 = _bbox_bottom_right(left)
    right_x2, right_y2 = _bbox_bottom_right(right)
    inter_x0 = max(left[0], right[0])
    inter_y0 = max(left[1], right[1])
    inter_x1 = min(left_x2, right_x2)
    inter_y1 = min(left_y2, right_y2)

    inter_width = max(0, inter_x1 - inter_x0)
    inter_height = max(0, inter_y1 - inter_y0)
    inter_area = inter_width * inter_height
    if inter_area == 0:
        return 0.0

    left_area = left[2] * left[3]
    right_area = right[2] * right[3]
    union_area = max(left_area + right_area - inter_area, 1)
    return inter_area / union_area


def deduplicate_detections(
    detections: list[Detection],
    *,
    iou_threshold: float,
) -> list[Detection]:
    if not detections:
        return []

    ordered = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[Detection] = []

    for candidate in ordered:
        duplicate = False
        for existing in kept:
            if _bbox_iou(candidate.bbox, existing.bbox) >= iou_threshold:
                duplicate = True
                break

            max_extent = max(
                candidate.bbox[2],
                candidate.bbox[3],
                existing.bbox[2],
                existing.bbox[3],
                1,
            )
            dx = candidate.anchor_px[0] - existing.anchor_px[0]
            dy = candidate.anchor_px[1] - existing.anchor_px[1]
            anchor_distance_sq = (dx * dx) + (dy * dy)
            if anchor_distance_sq <= (0.25 * max_extent) ** 2:
                duplicate = True
                break

        if not duplicate:
            kept.append(candidate)

    return kept


class MediaPipePoseDetector:
    def __init__(
        self,
        model_path: Path,
        num_poses: int,
        visibility_threshold: float,
        *,
        tile_rows: int = 1,
        tile_cols: int = 1,
        tile_overlap: float = 0.0,
        include_full_frame: bool = True,
        dedup_iou_threshold: float = 0.3,
    ) -> None:
        self.model_path = model_path
        self.num_poses = num_poses
        self.visibility_threshold = visibility_threshold
        self.tile_rows = max(tile_rows, 1)
        self.tile_cols = max(tile_cols, 1)
        self.tile_overlap = tile_overlap
        self.include_full_frame = include_full_frame
        self.dedup_iou_threshold = dedup_iou_threshold
        self._landmarker: Any | None = None
        self._mediapipe: Any | None = None
        self._last_inference_timestamp_ms = -1

    def _ensure_landmarker(self) -> None:
        if self._landmarker is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MediaPipe Pose Landmarker model not found: {self.model_path}"
            )

        import mediapipe as mp  # Imported lazily to keep tests lightweight.
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(self.model_path))
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=self.num_poses,
        )
        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._mediapipe = mp

    def detect(self, frame_bgr: Any, timestamp_ms: int) -> list[Detection]:
        self._ensure_landmarker()
        assert self._landmarker is not None
        assert self._mediapipe is not None

        frame_height, frame_width = frame_bgr.shape[:2]
        detections: list[Detection] = []

        if self.include_full_frame:
            detections.extend(
                self._detect_single_region(
                    frame_bgr,
                    timestamp_ms,
                    offset_x=0,
                    offset_y=0,
                    full_frame_width=frame_width,
                    full_frame_height=frame_height,
                )
            )

        tile_regions = generate_tile_regions(
            frame_width,
            frame_height,
            rows=self.tile_rows,
            cols=self.tile_cols,
            overlap=self.tile_overlap,
        )
        if self.tile_rows > 1 or self.tile_cols > 1:
            for tile in tile_regions:
                crop = frame_bgr[tile.y0:tile.y1, tile.x0:tile.x1]
                detections.extend(
                    self._detect_single_region(
                        crop,
                        timestamp_ms,
                        offset_x=tile.x0,
                        offset_y=tile.y0,
                        full_frame_width=frame_width,
                        full_frame_height=frame_height,
                    )
                )

        return deduplicate_detections(
            detections,
            iou_threshold=self.dedup_iou_threshold,
        )

    def _detect_single_region(
        self,
        frame_bgr: Any,
        timestamp_ms: int,
        *,
        offset_x: int,
        offset_y: int,
        full_frame_width: int,
        full_frame_height: int,
    ) -> list[Detection]:
        frame_height, frame_width = frame_bgr.shape[:2]
        frame_rgb = frame_bgr[:, :, ::-1]
        mp_image = self._mediapipe.Image(
            image_format=self._mediapipe.ImageFormat.SRGB,
            data=frame_rgb,
        )
        inference_timestamp_ms = self._next_inference_timestamp_ms(timestamp_ms)
        result = self._landmarker.detect_for_video(mp_image, inference_timestamp_ms)
        detections = self._result_to_detections(
            result.pose_landmarks,
            frame_width,
            frame_height,
            self.visibility_threshold,
        )
        if offset_x == 0 and offset_y == 0:
            return detections

        offset_detections: list[Detection] = []
        for detection in detections:
            offset_detections.append(
                Detection(
                    bbox=(
                        min(detection.bbox[0] + offset_x, full_frame_width - 1),
                        min(detection.bbox[1] + offset_y, full_frame_height - 1),
                        detection.bbox[2],
                        detection.bbox[3],
                    ),
                    anchor_px=(
                        min(detection.anchor_px[0] + offset_x, full_frame_width - 1),
                        min(detection.anchor_px[1] + offset_y, full_frame_height - 1),
                    ),
                    confidence=detection.confidence,
                )
            )
        return offset_detections

    def _next_inference_timestamp_ms(self, requested_timestamp_ms: int) -> int:
        if requested_timestamp_ms <= self._last_inference_timestamp_ms:
            requested_timestamp_ms = self._last_inference_timestamp_ms + 1
        self._last_inference_timestamp_ms = requested_timestamp_ms
        return requested_timestamp_ms

    @staticmethod
    def _result_to_detections(
        pose_landmarks: list[list[Any]],
        frame_width: int,
        frame_height: int,
        visibility_threshold: float,
    ) -> list[Detection]:
        detections: list[Detection] = []

        for pose in pose_landmarks:
            landmarks = [
                LandmarkPoint(
                    x=float(landmark.x),
                    y=float(landmark.y),
                    visibility=float(getattr(landmark, "visibility", 1.0)),
                )
                for landmark in pose
            ]
            visible_points = [
                point
                for point in landmarks
                if point.visibility >= visibility_threshold
            ]
            if not visible_points:
                continue

            xs = [point.x for point in visible_points]
            ys = [point.y for point in visible_points]
            min_x = min(max(int(min(xs) * frame_width), 0), frame_width - 1)
            min_y = min(max(int(min(ys) * frame_height), 0), frame_height - 1)
            max_x = min(max(int(max(xs) * frame_width), 0), frame_width - 1)
            max_y = min(max(int(max(ys) * frame_height), 0), frame_height - 1)
            bbox = (
                min_x,
                min_y,
                max(max_x - min_x, 1),
                max(max_y - min_y, 1),
            )
            anchor_px = compute_anchor_point(
                landmarks,
                bbox,
                frame_width,
                frame_height,
                visibility_threshold,
            )
            confidence = round(
                min(
                    1.0,
                    max(
                        0.0,
                        float(sum(point.visibility for point in visible_points) / len(visible_points)),
                    ),
                ),
                4,
            )
            detections.append(
                Detection(
                    bbox=bbox,
                    anchor_px=anchor_px,
                    confidence=confidence,
                )
            )

        return detections
