"""Run one iteration through the CoTracker3 pipeline against a reference.

Usage:
    .venv/Scripts/python.exe backend/scripts/iterate_cotracker.py \\
        --raw "backend/data/references/<raw>.mp4" \\
        --reference "backend/data/references/<reference>.mp4" \\
        --side left \\
        --iter 6 \\
        --clicks-json '[{"name":"D1","x":928,"y":523,"key_frame":93},...]' \\
        --clip-seconds 30

Forces SAM2 off + CoTracker local on, regardless of .env.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def trim_video(src: Path, dst: Path, seconds: float) -> int:
    import cv2

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dst), fourcc, fps, (w, h))
    target = int(seconds * fps)
    written = 0
    try:
        while written < target:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            written += 1
    finally:
        cap.release()
        writer.release()
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--side", required=True, choices=["left", "right"])
    parser.add_argument("--iter", type=int, required=True)
    parser.add_argument("--clicks-json", required=True,
                        help='JSON list of {"name","x","y","key_frame"}')
    parser.add_argument("--clip-seconds", type=float, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("iterate_cotracker")

    from app.api.routes import _compute_compare_metrics
    from app.core.config import get_settings
    from app.pipeline.cotracker_pipeline import process_video_with_cotracker
    from app.pipeline.processor import inspect_video_file
    from app.pipeline.stage_renderer import concat_side_by_side, render_stage_video
    from app.schemas.jobs import DancerClick
    from app.services.storage import JobStore

    raw_path = Path(args.raw).resolve()
    ref_path = Path(args.reference).resolve()

    settings = get_settings()
    original_sam2 = settings.sam2_backend
    original_la = settings.locateanything_backend
    original_cotracker = settings.cotracker_backend
    settings.sam2_backend = "disabled"
    settings.locateanything_backend = "disabled"
    settings.cotracker_backend = "local"

    try:
        process_input = raw_path
        key_frame = 0
        clicks_raw = json.loads(args.clicks_json)
        if args.clip_seconds:
            trimmed = raw_path.parent / f"_trim_{int(args.clip_seconds)}s.mp4"
            n = trim_video(raw_path, trimmed, args.clip_seconds)
            logger.info("trimmed %d frames into %s", n, trimmed)
            process_input = trimmed

        clicks: list[DancerClick] = [DancerClick(**c) for c in clicks_raw]
        if clicks_raw:
            key_frame = int(clicks_raw[0].get("key_frame", 0))

        job_id = f"iter{args.iter:02d}cot_{uuid4().hex[:8]}"
        store = JobStore()
        video_meta = inspect_video_file(process_input)
        video_meta = video_meta.model_copy(update={"filename": process_input.name})
        store.create_job(job_id, video_meta, expected_dancer_count=len(clicks))
        shutil.copyfile(process_input, store.job_dir(job_id) / process_input.name)

        job_dir = store.job_dir(job_id)
        video_path = job_dir / process_input.name

        def progress_cb(processed: int, total: int) -> None:
            if processed == 0 or processed == total or processed % 200 == 0:
                logger.info("progress: %d/%d", processed, total)

        logger.info(
            "starting CoTracker pipeline: %d frames, %d clicks, key_frame=%d",
            video_meta.frame_count, len(clicks), key_frame,
        )
        t0 = time.perf_counter()
        positions = process_video_with_cotracker(
            job_id=job_id,
            video_path=video_path,
            clicks=clicks,
            key_frame=key_frame,
            progress_callback=progress_cb,
        )
        elapsed = time.perf_counter() - t0
        logger.info("pipeline done in %.1fs", elapsed)

        store.save_positions(job_id, positions)
        # Dedupe by unique-name first-occurrence so labels line up with the
        # track_ids the pipeline assigns (multi-click collapses to one id).
        unique_click_names: list[str] = []
        for c in clicks:
            if c.name not in unique_click_names:
                unique_click_names.append(c.name)
        click_labels = {idx + 1: name for idx, name in enumerate(unique_click_names)}
        store.save_labels(job_id, click_labels)
        store.update_job(
            job_id, status="completed",
            processed_frames=positions.video.frame_count,
            total_frames=positions.video.frame_count, error=None,
        )

        stage_path = job_dir / "stage_view.mp4"
        stage_path = render_stage_video(
            positions, stage_path, width=800, height=450, labels=click_labels,
        )

        import cv2
        ref_cap = cv2.VideoCapture(str(ref_path))
        ref_w = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ref_h = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ref_cap.release()
        half = ref_w // 2
        crop = (0, 0, half, ref_h) if args.side == "left" else (ref_w - half, 0, half, ref_h)

        comparisons_dir = BACKEND_DIR / "data" / "references" / "comparisons"
        comparisons_dir.mkdir(parents=True, exist_ok=True)
        out_path = comparisons_dir / f"iter_{args.iter:02d}.mp4"
        out_path = concat_side_by_side(ref_path, stage_path, out_path, left_crop=crop)

        metrics = _compute_compare_metrics(positions, video_meta.fps)

        print("=" * 64)
        print(f"Iteration {args.iter:02d} (CoTracker) job={job_id}")
        print(f"  Processing time:      {elapsed:.1f}s ({video_meta.frame_count} frames)")
        print(f"  Comparison video:     {out_path}")
        print("-" * 64)
        print("Metrics:")
        print(f"  detected_formation_count:  {metrics.detected_formation_count}")
        print(f"  unique_track_ids:          {metrics.unique_track_ids} (expected {metrics.expected_dancer_count})")
        print(f"  id_stability_score:        {metrics.id_stability_score:.2f}")
        print(f"  longest_missing_gap_sec:   {metrics.longest_missing_gap_sec:.2f}")
        print(f"  full_count_formations:     {metrics.full_count_formations} of {metrics.detected_formation_count}")
        print(f"  frames_with_expected:      {metrics.frames_with_expected_count} ({metrics.pct_frames_with_expected}%)")
        print(f"  avg_dancers_per_frame:     {metrics.avg_dancers_per_frame}")
        print("=" * 64)
    finally:
        settings.sam2_backend = original_sam2
        settings.locateanything_backend = original_la
        settings.cotracker_backend = original_cotracker

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
