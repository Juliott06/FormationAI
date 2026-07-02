from __future__ import annotations

import shutil
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from app.schemas.jobs import (
    CoordinateSpaceMetadata,
    DancerPosition,
    DetectionSummary,
    FramePositions,
    PositionsResult,
    VideoMetadata,
)
from app.services.storage import JobStore


def _override_jobs_dir(tmp_path: Path) -> Path:
    jobs_dir = tmp_path / "jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)
    return jobs_dir


def test_upload_status_and_positions_flow(monkeypatch, tmp_path: Path) -> None:
    jobs_dir = _override_jobs_dir(tmp_path)

    from app.api import routes
    from app.core import config
    from app.services import job_runner

    settings = config.get_settings()
    original_jobs_dir = settings.jobs_dir
    original_sam2_backend = settings.sam2_backend
    original_cotracker_backend = settings.cotracker_backend
    settings.jobs_dir = jobs_dir
    settings.sam2_backend = "disabled"
    settings.cotracker_backend = "disabled"

    def fake_inspect_video_file(video_path: Path) -> VideoMetadata:
        return VideoMetadata(
            filename=video_path.name,
            fps=30.0,
            frame_count=3,
            width=640,
            height=360,
            duration_sec=0.1,
        )

    def fake_run_processing_job(job_id: str) -> None:
        store = JobStore(jobs_dir=jobs_dir)
        job = store.load_job(job_id)
        store.update_job(job_id, status="processing", processed_frames=2, total_frames=3, error=None)
        store.save_positions(
            job_id,
            PositionsResult(
                job_id=job_id,
                video=job.video_meta,
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
                                bbox=[10, 20, 30, 40],
                                anchor_px=[25, 60],
                                x=0.039063,
                                y=0.833333,
                                confidence=0.9,
                            )
                        ],
                    )
                ],
            ),
        )
        store.debug_video_path(job_id).write_bytes(b"fake-mp4-data")
        store.update_job(job_id, status="completed", processed_frames=3, total_frames=3, error=None)

    monkeypatch.setattr(routes, "inspect_video_file", fake_inspect_video_file)
    monkeypatch.setattr(job_runner, "run_processing_job", fake_run_processing_job)
    monkeypatch.setattr(routes, "run_processing_job", fake_run_processing_job)

    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("dance_practice.mp4", b"fake-video-bytes", "video/mp4")},
            data={"expected_dancer_count": "9"},
        )
        assert response.status_code == 202
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == "queued"
        assert data["expected_dancer_count"] == 9

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["status"] == "completed"
        assert status_payload["progress"] == 1.0
        assert status_payload["expected_dancer_count"] == 9

        positions_response = client.get(f"/api/v1/jobs/{job_id}/positions")
        assert positions_response.status_code == 200
        positions_payload = positions_response.json()
        assert positions_payload["job_id"] == job_id
        assert positions_payload["summary"]["expected_dancer_count"] == 9
        assert positions_payload["frames"][0]["dancers"][0]["id"] == 1

        summary_response = client.get(f"/api/v1/jobs/{job_id}/summary")
        assert summary_response.status_code == 200
        summary_payload = summary_response.json()
        assert summary_payload["job_id"] == job_id
        assert summary_payload["summary"]["unique_track_ids"] == 1

        positions_file_response = client.get(f"/api/v1/jobs/{job_id}/positions-file")
        assert positions_file_response.status_code == 200
        assert positions_file_response.headers["content-type"].startswith("application/json")
        assert b'"job_id"' in positions_file_response.content

        debug_video_response = client.get(f"/api/v1/jobs/{job_id}/debug-video")
        assert debug_video_response.status_code == 200
        assert debug_video_response.headers["content-type"] == "video/mp4"
        assert debug_video_response.content == b"fake-mp4-data"
    finally:
        settings.jobs_dir = original_jobs_dir
        settings.sam2_backend = original_sam2_backend
        settings.cotracker_backend = original_cotracker_backend
        shutil.rmtree(jobs_dir, ignore_errors=True)


def test_upload_rejects_invalid_extension(tmp_path: Path) -> None:
    jobs_dir = _override_jobs_dir(tmp_path)
    from app.core import config

    settings = config.get_settings()
    original_jobs_dir = settings.jobs_dir
    settings.jobs_dir = jobs_dir

    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("notes.txt", b"not-a-video", "text/plain")},
        )
        assert response.status_code == 400
    finally:
        settings.jobs_dir = original_jobs_dir
        shutil.rmtree(jobs_dir, ignore_errors=True)


def test_positions_conflict_when_job_not_complete(tmp_path: Path) -> None:
    jobs_dir = _override_jobs_dir(tmp_path)
    from app.core import config

    settings = config.get_settings()
    original_jobs_dir = settings.jobs_dir
    settings.jobs_dir = jobs_dir

    try:
        store = JobStore(jobs_dir=jobs_dir)
        video_meta = VideoMetadata(
            filename="dance_practice.mp4",
            fps=30.0,
            frame_count=5,
            width=640,
            height=360,
            duration_sec=0.166,
        )
        job_id = "job-incomplete"
        store.create_job(job_id, video_meta)
        store.save_upload(job_id, b"video", video_meta.filename)

        client = TestClient(create_app())
        response = client.get(f"/api/v1/jobs/{job_id}/positions")
        assert response.status_code == 409

        summary_response = client.get(f"/api/v1/jobs/{job_id}/summary")
        assert summary_response.status_code == 409

        positions_file_response = client.get(f"/api/v1/jobs/{job_id}/positions-file")
        assert positions_file_response.status_code == 409

        debug_response = client.get(f"/api/v1/jobs/{job_id}/debug-video")
        assert debug_response.status_code == 409
    finally:
        settings.jobs_dir = original_jobs_dir
        shutil.rmtree(jobs_dir, ignore_errors=True)


def test_processing_failure_sets_failed_status(monkeypatch, tmp_path: Path) -> None:
    jobs_dir = _override_jobs_dir(tmp_path)

    from app.api import routes
    from app.core import config
    from app.services import job_runner

    settings = config.get_settings()
    original_jobs_dir = settings.jobs_dir
    original_sam2_backend = settings.sam2_backend
    original_cotracker_backend = settings.cotracker_backend
    settings.jobs_dir = jobs_dir
    settings.sam2_backend = "disabled"
    settings.cotracker_backend = "disabled"

    def fake_inspect_video_file(video_path: Path) -> VideoMetadata:
        return VideoMetadata(
            filename=video_path.name,
            fps=30.0,
            frame_count=3,
            width=640,
            height=360,
            duration_sec=0.1,
        )

    def fake_run_processing_job(job_id: str) -> None:
        store = JobStore(jobs_dir=jobs_dir)
        store.update_job(job_id, status="processing", processed_frames=1, total_frames=3, error=None)
        store.update_job(job_id, status="failed", error="synthetic failure")

    monkeypatch.setattr(routes, "inspect_video_file", fake_inspect_video_file)
    monkeypatch.setattr(job_runner, "run_processing_job", fake_run_processing_job)
    monkeypatch.setattr(routes, "run_processing_job", fake_run_processing_job)

    try:
        client = TestClient(create_app())
        response = client.post(
            "/api/v1/jobs/upload",
            files={"file": ("dance_practice.mp4", b"fake-video-bytes", "video/mp4")},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

        status_response = client.get(f"/api/v1/jobs/{job_id}")
        assert status_response.status_code == 200
        status_payload = status_response.json()
        assert status_payload["status"] == "failed"
        assert status_payload["error"] == "synthetic failure"
    finally:
        settings.jobs_dir = original_jobs_dir
        settings.sam2_backend = original_sam2_backend
        settings.cotracker_backend = original_cotracker_backend
        shutil.rmtree(jobs_dir, ignore_errors=True)


def test_debug_video_500_when_missing(tmp_path: Path) -> None:
    jobs_dir = _override_jobs_dir(tmp_path)
    from app.core import config

    settings = config.get_settings()
    original_jobs_dir = settings.jobs_dir
    settings.jobs_dir = jobs_dir

    try:
        store = JobStore(jobs_dir=jobs_dir)
        video_meta = VideoMetadata(
            filename="dance_practice.mp4",
            fps=30.0,
            frame_count=5,
            width=640,
            height=360,
            duration_sec=0.166,
        )
        job_id = "job-no-debug"
        store.create_job(job_id, video_meta)
        store.save_upload(job_id, b"video", video_meta.filename)
        store.update_job(job_id, status="completed")
        store.save_positions(
            job_id,
            PositionsResult(
                job_id=job_id,
                video=video_meta,
                coordinate_space=CoordinateSpaceMetadata(
                    image_anchor_px="pixel anchor point in the source frame",
                    normalized_stage_proxy="heuristic top-down proxy derived from image-space anchors",
                ),
                summary=DetectionSummary(
                    expected_dancer_count=None,
                    unique_track_ids=0,
                    max_dancers_in_frame=0,
                    average_dancers_per_frame=0.0,
                    frames_with_detections=0,
                    frames_below_expected=None,
                    frames_meeting_expected=None,
                ),
                frames=[],
            ),
        )

        client = TestClient(create_app())
        response = client.get(f"/api/v1/jobs/{job_id}/debug-video")
        assert response.status_code == 500
    finally:
        settings.jobs_dir = original_jobs_dir
        shutil.rmtree(jobs_dir, ignore_errors=True)
