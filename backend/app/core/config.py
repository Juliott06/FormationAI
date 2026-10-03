from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POSE_MODEL = PROJECT_ROOT / "models" / "pose_landmarker_heavy.task"
DEFAULT_DATA_DIR = PROJECT_ROOT / "backend" / "data"
DEFAULT_ENV_FILE = PROJECT_ROOT / "backend" / ".env"


class Settings(BaseSettings):
    app_name: str = "FormationAI Backend"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    jobs_dir: Path = Field(default=DEFAULT_DATA_DIR / "jobs")
    references_dir: Path = Field(default=DEFAULT_DATA_DIR / "references")
    pose_model_path: Path = Field(default=DEFAULT_POSE_MODEL)
    yolo_model_name: str = "yolo26s.pt"
    yolo_confidence_threshold: float = 0.15
    yolo_iou_threshold: float = 0.45
    yolo_image_size: int = 1280
    yolo_max_detections: int = 20
    yolo_device: str = "cpu"
    tracker_config: str = "botsort_reid.yaml"
    # Deprecated (image px/frame); segmentation now uses
    # formation_movement_threshold. Kept so old .env files still load.
    formation_movement_threshold_px: float = 15.0
    # Average dancer speed relative to the group (shared drift down-weighted), in
    # formation sizes (RMS radius) per second, below which the dancers count
    # as holding a formation. Raised automatically above a clip's own jitter.
    formation_movement_threshold: float = 0.25
    formation_smoothing_window: int = 9
    formation_min_duration_sec: float = 0.5
    formation_split_window_frames: int = 10
    formation_split_threshold: float = 0.08
    formation_snap_grid_step: float = 0.05
    formation_dedup_distance: float = 0.03
    formation_template_snap_threshold: float = 0.08
    # Fit error relative to the formation's own size (RMS radius). 0.25 means
    # dancers sit on average within a quarter of the formation's radius of
    # the ideal shape.
    formation_template_snap_relative_threshold: float = 0.25
    interpolation_max_gap_frames: int = 60
    auto_swap_jump_threshold: float = 0.08
    auto_swap_advantage_ratio: float = 2.5
    gradual_swap_proximity: float = 0.10
    gradual_swap_advantage: float = 1.25
    id_continuity_max_gap_frames: int = 150
    id_continuity_max_distance: float = 0.15
    appearance_merge_threshold: float = 0.92
    appearance_sample_every: int = 3
    generate_debug_video: bool = True
    allowed_video_extensions: tuple[str, ...] = (".mp4", ".mov", ".m4v", ".avi")
    locateanything_backend: Literal["disabled", "http"] = "disabled"
    locateanything_url: str | None = None
    locateanything_timeout_sec: float = 60.0
    sam2_backend: Literal["disabled", "http"] = "disabled"
    sam2_url: str | None = None
    sam2_timeout_sec: float = 600.0
    cotracker_backend: Literal["disabled", "local"] = "disabled"
    cotracker_resize_width: int = 960
    # Foot assist: match tracked click points to YOLO person boxes so depth
    # comes from real feet (bbox bottoms) instead of torso points, which sit
    # near the camera horizon and carry almost no depth signal.
    cotracker_foot_assist: bool = True
    foot_assist_sample_every: int = 3
    # Auto support points: track 2 extra points (chest + hips, from the person
    # box under each click) per dancer and take the median, so one point
    # drifting onto a neighbour is outvoted. Costs ~CoTracker time per point.
    cotracker_auto_points: bool = True
    # Without 4 marked floor corners, build the top-down view from the
    # dancers' apparent heights (see pipeline/auto_ground.py) instead of the
    # raw camera view.
    auto_ground_calibration: bool = True

    model_config = SettingsConfigDict(
        env_prefix="FORMATIONAI_",
        case_sensitive=False,
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.jobs_dir.mkdir(parents=True, exist_ok=True)
    settings.references_dir.mkdir(parents=True, exist_ok=True)
    return settings
