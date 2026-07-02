from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


JobStatus = Literal["queued", "awaiting_clicks", "processing", "completed", "failed"]


class VideoMetadata(BaseModel):
    filename: str
    fps: float = Field(ge=0)
    frame_count: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    duration_sec: float = Field(ge=0)


class DancerRosterEntry(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    hint: str = Field(
        min_length=1,
        max_length=200,
        description='Visual hint for LocateAnything, e.g. "dancer in white top with black pants"',
    )


class DancerClick(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    x: int = Field(ge=0, description="X pixel coordinate in the key frame")
    y: int = Field(ge=0, description="Y pixel coordinate in the key frame")


class ClickSeedRequest(BaseModel):
    key_frame: int = Field(ge=0, description="Frame index the clicks were made on")
    clicks: list[DancerClick] = Field(min_length=1)


class JobRecord(BaseModel):
    job_id: str
    status: JobStatus
    video_meta: VideoMetadata
    expected_dancer_count: int | None = Field(default=None, ge=1)
    roster: list[DancerRosterEntry] = Field(default_factory=list)
    clicks: list[DancerClick] = Field(default_factory=list)
    key_frame: int = Field(default=0, ge=0)
    processed_frames: int = Field(default=0, ge=0)
    total_frames: int = Field(default=0, ge=0)
    error: str | None = None


class UploadResponse(BaseModel):
    job_id: str
    status: JobStatus
    video_meta: VideoMetadata
    expected_dancer_count: int | None = Field(default=None, ge=1)
    roster: list[DancerRosterEntry] = Field(default_factory=list)


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    processed_frames: int = Field(ge=0)
    total_frames: int = Field(ge=0)
    progress: float = Field(ge=0, le=1)
    error: str | None = None
    video_meta: VideoMetadata
    expected_dancer_count: int | None = Field(default=None, ge=1)


class DetectionSummary(BaseModel):
    expected_dancer_count: int | None = Field(default=None, ge=1)
    unique_track_ids: int = Field(ge=0)
    max_dancers_in_frame: int = Field(ge=0)
    average_dancers_per_frame: float = Field(ge=0)
    frames_with_detections: int = Field(ge=0)
    frames_below_expected: int | None = Field(default=None, ge=0)
    frames_meeting_expected: int | None = Field(default=None, ge=0)


class PositionsSummaryResponse(BaseModel):
    job_id: str
    status: JobStatus
    video_meta: VideoMetadata
    summary: DetectionSummary


class DancerPosition(BaseModel):
    id: int = Field(ge=1)
    bbox: list[int] = Field(min_length=4, max_length=4)
    anchor_px: list[int] = Field(min_length=2, max_length=2)
    x: float
    y: float
    confidence: float = Field(ge=0, le=1)


class FramePositions(BaseModel):
    frame: int = Field(ge=0)
    timestamp_sec: float = Field(ge=0)
    dancers: list[DancerPosition]


class CoordinateSpaceMetadata(BaseModel):
    image_anchor_px: str
    normalized_stage_proxy: str


class FormationDancerPosition(BaseModel):
    id: int = Field(ge=1)
    x: float
    y: float


class Formation(BaseModel):
    index: int = Field(ge=0)
    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_time_sec: float = Field(ge=0)
    end_time_sec: float = Field(ge=0)
    duration_sec: float = Field(ge=0)
    dancers: list[FormationDancerPosition]
    shape_name: str | None = None


class MergeIdsRequest(BaseModel):
    keep_id: int = Field(ge=1)
    remove_id: int = Field(ge=1)


class SwapIdsRequest(BaseModel):
    id_a: int = Field(ge=1)
    id_b: int = Field(ge=1)
    from_frame: int = Field(ge=0)


class CompareToReferenceRequest(BaseModel):
    reference_filename: str = Field(
        min_length=1,
        description="Filename (no path) under backend/data/references/",
    )
    reference_side: Literal["left", "right"] = Field(
        description="Which half of the side-by-side reference is the formation animation"
    )


class CompareMetrics(BaseModel):
    detected_formation_count: int = Field(ge=0)
    unique_track_ids: int = Field(ge=0)
    expected_dancer_count: int = Field(ge=0)
    id_stability_score: float = Field(
        ge=0,
        description="unique_track_ids / expected_dancer_count; closer to 1.0 = better",
    )
    longest_missing_gap_sec: float = Field(
        ge=0,
        description="Longest stretch (in seconds) where a well-established dancer goes missing within their active span; only considers IDs that exist in >10% of frames so orphan tracks don't inflate the metric",
    )
    full_count_formations: int = Field(
        ge=0,
        description="Number of formations whose dancer count equals expected_dancer_count",
    )
    frames_with_expected_count: int = Field(
        ge=0,
        description="Number of frames where the visible dancer count >= expected_dancer_count",
    )
    pct_frames_with_expected: float = Field(
        ge=0,
        description="Percentage of frames hitting the expected dancer count (frames_with_expected_count / total_frames)",
    )
    avg_dancers_per_frame: float = Field(ge=0)


class CompareToReferenceResponse(BaseModel):
    comparison_video_url: str
    metrics: CompareMetrics


class ReferenceFile(BaseModel):
    filename: str
    size_bytes: int = Field(ge=0)


class ReferencesListResponse(BaseModel):
    files: list[ReferenceFile]


class PositionsResult(BaseModel):
    job_id: str
    video: VideoMetadata
    coordinate_space: CoordinateSpaceMetadata
    summary: DetectionSummary
    frames: list[FramePositions]
    formations: list[Formation] = Field(default_factory=list)
