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
    formation_movement_threshold_px: float = 15.0
    formation_smoothing_window: int = 9
    formation_min_duration_sec: float = 0.5
    formation_split_window_frames: int = 10
    formation_split_threshold: float = 0.08
    formation_snap_grid_step: float = 0.05
    formation_dedup_distance: float = 0.03
    formation_template_snap_threshold: float = 0.08
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
