from __future__ import annotations

from dataclasses import dataclass
from math import dist

from app.pipeline.types import Detection, TrackedDetection

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - exercised only in lightweight environments.
    linear_sum_assignment = None


@dataclass(slots=True)
class TrackState:
    track_id: int
    bbox: tuple[int, int, int, int]
    anchor_px: tuple[int, int]
    missed_frames: int = 0


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(bbox[2], 0) * max(bbox[3], 0)


class MultiPersonTracker:
    def __init__(
        self,
        *,
        max_missed_frames: int,
        distance_weight: float,
        area_weight: float,
        max_cost: float,
    ) -> None:
        self.max_missed_frames = max_missed_frames
        self.distance_weight = distance_weight
        self.area_weight = area_weight
        self.max_cost = max_cost
        self._tracks: list[TrackState] = []
        self._next_track_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedDetection]:
        if not self._tracks:
            return self._create_initial_tracks(detections)

        if not detections:
            self._increment_misses_for_all_tracks()
            self._drop_stale_tracks()
            return []

        costs = self._build_cost_matrix(detections)
        track_indices, detection_indices = assign_detections(costs)

        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()
        assignments: list[TrackedDetection] = []

        for track_idx, detection_idx in zip(track_indices, detection_indices, strict=True):
            cost = costs[track_idx][detection_idx]
            if cost > self.max_cost:
                continue

            track = self._tracks[track_idx]
            detection = detections[detection_idx]
            track.bbox = detection.bbox
            track.anchor_px = detection.anchor_px
            track.missed_frames = 0
            matched_tracks.add(track_idx)
            matched_detections.add(detection_idx)
            assignments.append(
                TrackedDetection(
                    track_id=track.track_id,
                    bbox=detection.bbox,
                    anchor_px=detection.anchor_px,
                    confidence=detection.confidence,
                )
            )

        for idx, track in enumerate(self._tracks):
            if idx not in matched_tracks:
                track.missed_frames += 1

        for idx, detection in enumerate(detections):
            if idx not in matched_detections:
                new_track = TrackState(
                    track_id=self._next_track_id,
                    bbox=detection.bbox,
                    anchor_px=detection.anchor_px,
                )
                self._next_track_id += 1
                self._tracks.append(new_track)
                assignments.append(
                    TrackedDetection(
                        track_id=new_track.track_id,
                        bbox=detection.bbox,
                        anchor_px=detection.anchor_px,
                        confidence=detection.confidence,
                    )
                )

        self._drop_stale_tracks()
        assignments.sort(key=lambda item: item.track_id)
        return assignments

    def _create_initial_tracks(self, detections: list[Detection]) -> list[TrackedDetection]:
        tracked: list[TrackedDetection] = []
        for detection in detections:
            track = TrackState(
                track_id=self._next_track_id,
                bbox=detection.bbox,
                anchor_px=detection.anchor_px,
            )
            self._next_track_id += 1
            self._tracks.append(track)
            tracked.append(
                TrackedDetection(
                    track_id=track.track_id,
                    bbox=detection.bbox,
                    anchor_px=detection.anchor_px,
                    confidence=detection.confidence,
                )
            )
        return tracked

    def _increment_misses_for_all_tracks(self) -> None:
        for track in self._tracks:
            track.missed_frames += 1

    def _drop_stale_tracks(self) -> None:
        self._tracks = [
            track
            for track in self._tracks
            if track.missed_frames <= self.max_missed_frames
        ]

    def _build_cost_matrix(self, detections: list[Detection]) -> list[list[float]]:
        matrix: list[list[float]] = []
        for track in self._tracks:
            row: list[float] = []
            track_area = _bbox_area(track.bbox)
            for detection in detections:
                detection_area = _bbox_area(detection.bbox)
                area_ratio = abs(track_area - detection_area) / max(track_area, detection_area, 1)
                row.append(
                    (dist(track.anchor_px, detection.anchor_px) * self.distance_weight)
                    + (area_ratio * self.area_weight)
                )
            matrix.append(row)
        return matrix


def assign_detections(costs: list[list[float]]) -> tuple[list[int], list[int]]:
    if linear_sum_assignment is not None:
        track_indices, detection_indices = linear_sum_assignment(costs)
        return list(track_indices), list(detection_indices)

    assignments: list[tuple[int, int, float]] = []
    for track_index, row in enumerate(costs):
        for detection_index, cost in enumerate(row):
            assignments.append((track_index, detection_index, cost))
    assignments.sort(key=lambda item: item[2])

    used_tracks: set[int] = set()
    used_detections: set[int] = set()
    selected_tracks: list[int] = []
    selected_detections: list[int] = []

    for track_index, detection_index, _ in assignments:
        if track_index in used_tracks or detection_index in used_detections:
            continue
        used_tracks.add(track_index)
        used_detections.add(detection_index)
        selected_tracks.append(track_index)
        selected_detections.append(detection_index)

    return selected_tracks, selected_detections
