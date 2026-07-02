from __future__ import annotations

from app.core.config import get_settings
from app.pipeline.cotracker_pipeline import process_video_with_cotracker
from app.pipeline.processor import process_video
from app.pipeline.sam2_pipeline import process_video_with_sam2
from app.schemas.jobs import unique_click_names
from app.services.storage import JobStore


def run_processing_job(job_id: str) -> None:
    settings = get_settings()
    store = JobStore()
    job = store.load_job(job_id)
    job_dir = store.job_dir(job_id)
    video_path = job_dir / job.video_meta.filename
    store.update_job(job_id, status="processing", processed_frames=0, error=None)

    sam2_enabled = settings.sam2_backend != "disabled" and bool(job.clicks)
    cotracker_enabled = (
        settings.cotracker_backend != "disabled" and bool(job.clicks) and not sam2_enabled
    )
    histograms_out: dict[int, list[float]] = {}

    try:
        # Labels follow unique-name first-occurrence order, matching how the
        # cotracker/sam2 pipelines assign track_ids. Multi-click on one dancer
        # produces duplicate names; they collapse to a single track_id.
        click_labels = {
            idx + 1: name for idx, name in enumerate(unique_click_names(job.clicks))
        }

        if sam2_enabled:
            positions = process_video_with_sam2(
                job_id=job_id,
                video_path=video_path,
                clicks=job.clicks,
                key_frame=job.key_frame,
                progress_callback=lambda processed, total: store.update_job(
                    job_id,
                    status="processing",
                    processed_frames=processed,
                    total_frames=total,
                    error=None,
                ),
            )
            store.save_labels(job_id, click_labels)
        elif cotracker_enabled:
            positions = process_video_with_cotracker(
                job_id=job_id,
                video_path=video_path,
                clicks=job.clicks,
                key_frame=job.key_frame,
                progress_callback=lambda processed, total: store.update_job(
                    job_id,
                    status="processing",
                    processed_frames=processed,
                    total_frames=total,
                    error=None,
                ),
            )
            store.save_labels(job_id, click_labels)
        else:
            debug_path = job_dir / "debug_overlay.mp4" if settings.generate_debug_video else None
            positions = process_video(
                job_id=job_id,
                video_path=video_path,
                debug_video_path=debug_path,
                expected_dancer_count=job.expected_dancer_count,
                roster=job.roster,
                histograms_out=histograms_out,
                progress_callback=lambda processed, total: store.update_job(
                    job_id,
                    status="processing",
                    processed_frames=processed,
                    total_frames=total,
                    error=None,
                ),
            )
            if histograms_out:
                store.save_histograms(job_id, histograms_out)
            if job.roster:
                roster_labels = {idx + 1: entry.name for idx, entry in enumerate(job.roster)}
                store.save_labels(job_id, roster_labels)

        store.save_positions(job_id, positions)
        store.update_job(
            job_id,
            status="completed",
            processed_frames=positions.video.frame_count,
            total_frames=positions.video.frame_count,
            error=None,
        )
    except Exception as exc:
        store.update_job(job_id, status="failed", error=str(exc))
