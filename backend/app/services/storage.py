from __future__ import annotations

import json
import time
from pathlib import Path

from app.core.config import get_settings
from app.schemas.jobs import DancerRosterEntry, JobRecord, PositionsResult, VideoMetadata


class JobNotFoundError(FileNotFoundError):
    pass


class JobStore:
    def __init__(self, jobs_dir: Path | None = None) -> None:
        settings = get_settings()
        self.jobs_dir = jobs_dir or settings.jobs_dir
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def create_job(
        self,
        job_id: str,
        video_meta: VideoMetadata,
        *,
        expected_dancer_count: int | None = None,
        roster: list[DancerRosterEntry] | None = None,
        initial_status: str = "queued",
    ) -> Path:
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=False)
        record = JobRecord(
            job_id=job_id,
            status=initial_status,  # type: ignore[arg-type]
            video_meta=video_meta,
            expected_dancer_count=expected_dancer_count,
            roster=roster or [],
            processed_frames=0,
            total_frames=video_meta.frame_count,
        )
        self._write_json(job_dir / "job.json", record.model_dump(mode="json"))
        return job_dir

    def job_dir(self, job_id: str) -> Path:
        path = self.jobs_dir / job_id
        if not path.exists():
            raise JobNotFoundError(f"Unknown job: {job_id}")
        return path

    def save_upload(self, job_id: str, source_bytes: bytes, filename: str) -> Path:
        upload_path = self.job_dir(job_id) / filename
        upload_path.write_bytes(source_bytes)
        return upload_path

    def upload_path(self, job_id: str) -> Path:
        """Path to the original uploaded video for this job."""
        job = self.load_job(job_id)
        path = self.job_dir(job_id) / job.video_meta.filename
        if not path.exists():
            raise FileNotFoundError(f"Original upload missing for job {job_id}: {path}")
        return path

    def extract_frame_jpeg(
        self, job_id: str, frame_index: int, *, quality: int = 85
    ) -> bytes:
        """Read frame N from the stored upload and return JPEG bytes."""
        import cv2

        video_path = self.upload_path(job_id)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video file: {video_path}")
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                raise FileNotFoundError(
                    f"Could not read frame {frame_index} from {video_path}"
                )
            ok, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
            )
            if not ok:
                raise RuntimeError(f"JPEG encode failed for frame {frame_index}")
            return bytes(buf)
        finally:
            capture.release()

    def load_job(self, job_id: str) -> JobRecord:
        try:
            raw = self._read_json(self.job_dir(job_id) / "job.json")
        except (FileNotFoundError, json.JSONDecodeError) as exc:
            raise JobNotFoundError(
                f"Job {job_id} is missing or corrupted (likely from a crashed earlier run)"
            ) from exc
        return JobRecord.model_validate(raw)

    def update_job(self, job_id: str, **changes: object) -> JobRecord:
        record = self.load_job(job_id)
        updated = record.model_copy(update=changes)
        self._write_json(
            self.job_dir(job_id) / "job.json",
            updated.model_dump(mode="json"),
        )
        return updated

    def save_positions(self, job_id: str, positions: PositionsResult) -> Path:
        path = self.job_dir(job_id) / "positions.json"
        self._write_json(path, positions.model_dump(mode="json"))
        return path

    def load_positions(self, job_id: str) -> PositionsResult:
        raw = self._read_json(self.job_dir(job_id) / "positions.json")
        return PositionsResult.model_validate(raw)

    def debug_video_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "debug_overlay.mp4"

    def find_debug_video(self, job_id: str) -> Path | None:
        try:
            job_dir = self.job_dir(job_id)
        except JobNotFoundError:
            return None
        for ext in (".mp4", ".avi"):
            candidate = job_dir / f"debug_overlay{ext}"
            if candidate.exists() and candidate.stat().st_size > 0:
                return candidate
        return None

    def positions_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "positions.json"

    def histograms_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "histograms.json"

    def labels_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "labels.json"

    def save_histograms(self, job_id: str, histograms: dict[int, list[float]]) -> Path:
        path = self.histograms_path(job_id)
        self._write_json(path, {str(k): v for k, v in histograms.items()})
        return path

    def load_histograms(self, job_id: str) -> dict[int, list[float]]:
        path = self.histograms_path(job_id)
        if not path.exists():
            return {}
        raw = self._read_json(path)
        return {int(k): v for k, v in raw.items()}

    def save_labels(self, job_id: str, labels: dict[int, str]) -> Path:
        path = self.labels_path(job_id)
        self._write_json(path, {str(k): v for k, v in labels.items()})
        return path

    def load_labels(self, job_id: str) -> dict[int, str]:
        path = self.labels_path(job_id)
        if not path.exists():
            return {}
        raw = self._read_json(path)
        return {int(k): v for k, v in raw.items()}

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        last_error: Exception | None = None
        for attempt in range(8):
            try:
                tmp_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        raise last_error if last_error else RuntimeError("rename failed")

    @staticmethod
    def _read_json(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(path)
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            raise FileNotFoundError(f"{path} is empty")
        return json.loads(content)
