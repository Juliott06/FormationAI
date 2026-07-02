from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, Response
from pydantic import ValidationError

from app.core.config import get_settings
from app.pipeline.processor import apply_id_merge, apply_id_swap, apply_labels, inspect_video_file
from app.pipeline.stage_renderer import concat_side_by_side, render_stage_video
from app.schemas.jobs import (
    ClickSeedRequest,
    CompareMetrics,
    CompareToReferenceRequest,
    CompareToReferenceResponse,
    DancerRosterEntry,
    JobStatusResponse,
    MergeIdsRequest,
    PositionsResult,
    PositionsSummaryResponse,
    ReferenceFile,
    ReferencesListResponse,
    SwapIdsRequest,
    UploadResponse,
)
from app.services.job_runner import run_processing_job
from app.services.storage import JobNotFoundError, JobStore


_REFERENCES_DIR = Path(__file__).resolve().parents[3] / "data" / "references"
_REFERENCE_EXTS = {".mp4", ".mov", ".m4v", ".avi"}


def _references_dir() -> Path:
    _REFERENCES_DIR.mkdir(parents=True, exist_ok=True)
    return _REFERENCES_DIR


def _safe_reference_path(filename: str) -> Path:
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="reference_filename must be a bare filename (no path separators)",
        )
    candidate = _references_dir() / filename
    if candidate.suffix.lower() not in _REFERENCE_EXTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"reference file must be one of {sorted(_REFERENCE_EXTS)}",
        )
    if not candidate.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference file not found: {filename}",
        )
    return candidate


def _compute_compare_metrics(
    positions: PositionsResult, fps: float
) -> CompareMetrics:
    expected = positions.summary.expected_dancer_count or positions.summary.max_dancers_in_frame
    if expected <= 0:
        expected = max((len(f.dancers) for f in positions.frames), default=1)

    total_frames = len(positions.frames)
    if total_frames == 0:
        return CompareMetrics(
            detected_formation_count=len(positions.formations),
            unique_track_ids=0,
            expected_dancer_count=expected,
            id_stability_score=0.0,
            longest_missing_gap_sec=0.0,
            full_count_formations=0,
            frames_with_expected_count=0,
            pct_frames_with_expected=0.0,
            avg_dancers_per_frame=0.0,
        )

    # Per-id presence: first/last frame seen + total frame count
    first_seen: dict[int, int] = {}
    last_seen: dict[int, int] = {}
    id_total_count: dict[int, int] = {}
    for f in positions.frames:
        for d in f.dancers:
            id_total_count[d.id] = id_total_count.get(d.id, 0) + 1
            if d.id not in first_seen:
                first_seen[d.id] = f.frame
            last_seen[d.id] = f.frame

    unique_ids = list(id_total_count.keys())
    id_stability = round(len(unique_ids) / max(expected, 1), 3)

    # Longest missing gap: only counted for well-established IDs (>10% of frames),
    # within their active span. Filters out orphan tracks that briefly existed
    # before being merged or recovered.
    presence_threshold = max(1, total_frames // 10)
    longest_gap_frames = 0
    for tid, count in id_total_count.items():
        if count < presence_threshold:
            continue
        active_first = first_seen[tid]
        active_last = last_seen[tid]
        present_frames: set[int] = set()
        for f in positions.frames:
            if active_first <= f.frame <= active_last:
                if any(d.id == tid for d in f.dancers):
                    present_frames.add(f.frame)
        run = 0
        for f_idx in range(active_first, active_last + 1):
            if f_idx in present_frames:
                run = 0
            else:
                run += 1
                if run > longest_gap_frames:
                    longest_gap_frames = run
    longest_gap_sec = round(longest_gap_frames / max(fps, 1.0), 2)

    full_count_formations = sum(
        1 for form in positions.formations if len(form.dancers) >= expected
    )

    frames_with_expected = sum(
        1 for f in positions.frames if len(f.dancers) >= expected
    )
    pct_frames_with_expected = round(frames_with_expected * 100.0 / total_frames, 1)
    avg_dancers = round(
        sum(len(f.dancers) for f in positions.frames) / total_frames, 2
    )

    return CompareMetrics(
        detected_formation_count=len(positions.formations),
        unique_track_ids=len(unique_ids),
        expected_dancer_count=expected,
        id_stability_score=id_stability,
        longest_missing_gap_sec=longest_gap_sec,
        full_count_formations=full_count_formations,
        frames_with_expected_count=frames_with_expected,
        pct_frames_with_expected=pct_frames_with_expected,
        avg_dancers_per_frame=avg_dancers,
    )


router = APIRouter()


@router.post(
    "/jobs/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    expected_dancer_count: int | None = Form(default=None),
    roster_json: str | None = Form(default=None),
) -> UploadResponse:
    settings = get_settings()
    original_filename = file.filename or "upload.mp4"
    suffix = Path(original_filename).suffix.lower()
    if suffix not in settings.allowed_video_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported video type: {suffix or 'missing extension'}",
        )

    roster: list[DancerRosterEntry] = []
    if roster_json:
        try:
            raw_roster = json.loads(roster_json)
            if not isinstance(raw_roster, list):
                raise ValueError("roster_json must be a JSON array")
            roster = [DancerRosterEntry.model_validate(r) for r in raw_roster]
        except (json.JSONDecodeError, ValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid roster_json: {exc}",
            ) from exc
        names_seen = set()
        for entry in roster:
            if entry.name in names_seen:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Duplicate dancer name in roster: {entry.name}",
                )
            names_seen.add(entry.name)

    if settings.locateanything_backend != "disabled" and not roster:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "LocateAnything is enabled — please provide a dancer roster "
                "(name + appearance hint per dancer) on upload."
            ),
        )

    payload = await file.read()
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )

    job_id = str(uuid4())
    temp_path = settings.jobs_dir / f"{job_id}{suffix}"
    temp_path.write_bytes(payload)
    try:
        inspected = inspect_video_file(temp_path)
    except Exception as exc:
        temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid video file: {exc}",
        ) from exc
    video_meta = inspected.model_copy(update={"filename": original_filename})

    store = JobStore()
    sam2_enabled = settings.sam2_backend != "disabled"
    cotracker_enabled = settings.cotracker_backend != "disabled"
    needs_clicks = sam2_enabled or cotracker_enabled
    initial_status = "awaiting_clicks" if needs_clicks else "queued"
    try:
        store.create_job(
            job_id,
            video_meta,
            expected_dancer_count=expected_dancer_count,
            roster=roster,
            initial_status=initial_status,
        )
        store.save_upload(job_id, payload, video_meta.filename)
    finally:
        temp_path.unlink(missing_ok=True)

    if not needs_clicks:
        background_tasks.add_task(run_processing_job, job_id)
    return UploadResponse(
        job_id=job_id,
        status=initial_status,
        video_meta=video_meta,
        expected_dancer_count=expected_dancer_count,
        roster=roster,
    )


@router.get("/jobs/{job_id}/frame-jpeg")
def get_frame_jpeg(job_id: str, n: int = 0) -> Response:
    if n < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="frame index must be >= 0",
        )
    store = JobStore()
    try:
        store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    try:
        jpeg = store.extract_frame_jpeg(job_id, n)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return Response(content=jpeg, media_type="image/jpeg")


@router.post("/jobs/{job_id}/clicks", response_model=JobStatusResponse)
def submit_clicks(
    job_id: str,
    request: ClickSeedRequest,
    background_tasks: BackgroundTasks,
) -> JobStatusResponse:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status not in ("awaiting_clicks", "failed"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is in status '{job.status}'; clicks can only be set while awaiting_clicks.",
        )

    fw, fh = job.video_meta.width, job.video_meta.height
    for click in request.clicks:
        if click.x >= fw or click.y >= fh:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Click ({click.x},{click.y}) out of frame bounds ({fw}x{fh})",
            )
    # NOTE: duplicate names are allowed and treated as multiple click points on
    # the same dancer (e.g. head + torso). The CoTracker pipeline groups by
    # name and emits a dancer position if ANY of their points is visible per
    # frame; the SAM2 pipeline uses the first occurrence per name.
    if request.key_frame >= job.video_meta.frame_count:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"key_frame {request.key_frame} is out of bounds "
                f"(video has {job.video_meta.frame_count} frames)"
            ),
        )

    updated = store.update_job(
        job_id,
        status="queued",
        clicks=[c.model_dump(mode="json") for c in request.clicks],
        key_frame=request.key_frame,
        error=None,
    )
    background_tasks.add_task(run_processing_job, job_id)

    total_frames = max(updated.total_frames, 0)
    progress = 0.0
    return JobStatusResponse(
        job_id=updated.job_id,
        status=updated.status,
        processed_frames=updated.processed_frames,
        total_frames=updated.total_frames,
        progress=progress,
        error=updated.error,
        video_meta=updated.video_meta,
        expected_dancer_count=updated.expected_dancer_count,
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    total_frames = max(job.total_frames, 0)
    progress = (
        min(job.processed_frames / total_frames, 1.0)
        if total_frames > 0
        else 0.0
    )
    if job.status == "completed":
        progress = 1.0

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        processed_frames=job.processed_frames,
        total_frames=job.total_frames,
        progress=round(progress, 4),
        error=job.error,
        video_meta=job.video_meta,
        expected_dancer_count=job.expected_dancer_count,
    )


@router.get("/jobs/{job_id}/positions", response_model=PositionsResult)
def get_positions(job_id: str) -> PositionsResult:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    try:
        return store.load_positions(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        ) from exc


@router.get("/jobs/{job_id}/summary", response_model=PositionsSummaryResponse)
def get_positions_summary(job_id: str) -> PositionsSummaryResponse:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    try:
        positions = store.load_positions(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        ) from exc

    return PositionsSummaryResponse(
        job_id=job.job_id,
        status=job.status,
        video_meta=job.video_meta,
        summary=positions.summary,
    )


@router.get("/jobs/{job_id}/positions-file")
def download_positions_file(job_id: str) -> FileResponse:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    positions_path = store.positions_path(job_id)
    if not positions_path.exists():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        )

    return FileResponse(
        path=positions_path,
        media_type="application/json",
        filename=f"{job_id}_positions.json",
    )


@router.post("/jobs/{job_id}/merge-ids", response_model=PositionsResult)
def merge_ids(job_id: str, request: MergeIdsRequest) -> PositionsResult:
    if request.keep_id == request.remove_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="keep_id and remove_id must differ",
        )

    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    try:
        positions = store.load_positions(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        ) from exc

    updated = apply_id_merge(
        positions, keep_id=request.keep_id, remove_id=request.remove_id
    )
    store.save_positions(job_id, updated)
    return updated


@router.post("/jobs/{job_id}/swap-ids", response_model=PositionsResult)
def swap_ids(job_id: str, request: SwapIdsRequest) -> PositionsResult:
    if request.id_a == request.id_b:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="id_a and id_b must differ",
        )

    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    try:
        positions = store.load_positions(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        ) from exc

    updated = apply_id_swap(
        positions,
        id_a=request.id_a,
        id_b=request.id_b,
        from_frame=request.from_frame,
    )
    store.save_positions(job_id, updated)
    return updated


@router.get("/jobs/{job_id}/labels")
def get_labels(job_id: str) -> dict[int, str]:
    store = JobStore()
    try:
        store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    return store.load_labels(job_id)


@router.put("/jobs/{job_id}/labels", response_model=PositionsResult)
def set_labels(job_id: str, labels: dict[int, str]) -> PositionsResult:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    cleaned = {int(k): v.strip() for k, v in labels.items() if v and v.strip()}
    store.save_labels(job_id, cleaned)

    try:
        positions = store.load_positions(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        ) from exc

    histograms = store.load_histograms(job_id)
    if not histograms or not cleaned:
        return positions

    updated = apply_labels(positions, histograms, cleaned)
    store.save_positions(job_id, updated)
    return updated


@router.get("/jobs/{job_id}/debug-video")
def get_debug_video(job_id: str) -> FileResponse:
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    debug_video_path = store.find_debug_video(job_id)
    if debug_video_path is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                f"Missing debug video for job {job_id}. "
                f"This usually means processing happened before "
                f"FORMATIONAI_GENERATE_DEBUG_VIDEO=true was set, or the codec failed. "
                f"Re-upload to regenerate."
            ),
        )

    media_type = "video/mp4" if debug_video_path.suffix == ".mp4" else "video/x-msvideo"
    return FileResponse(
        path=debug_video_path,
        media_type=media_type,
        filename=f"{job_id}_debug_overlay{debug_video_path.suffix}",
    )


@router.get("/references", response_model=ReferencesListResponse)
def list_references() -> ReferencesListResponse:
    folder = _references_dir()
    files: list[ReferenceFile] = []
    for path in sorted(folder.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _REFERENCE_EXTS:
            continue
        files.append(ReferenceFile(filename=path.name, size_bytes=path.stat().st_size))
    return ReferencesListResponse(files=files)


def _find_comparison_video(job_dir: Path) -> Path | None:
    for ext in (".mp4", ".avi"):
        candidate = job_dir / f"comparison{ext}"
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


@router.post(
    "/jobs/{job_id}/compare-to-reference",
    response_model=CompareToReferenceResponse,
)
def compare_to_reference(
    job_id: str, request: CompareToReferenceRequest
) -> CompareToReferenceResponse:
    settings = get_settings()
    store = JobStore()
    try:
        job = store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if job.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Job {job_id} is not complete",
        )

    reference_path = _safe_reference_path(request.reference_filename)

    try:
        positions = store.load_positions(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Missing positions for job {job_id}",
        ) from exc

    labels = store.load_labels(job_id)
    job_dir = store.job_dir(job_id)
    stage_video_path = job_dir / "stage_view.mp4"
    try:
        stage_video_path = render_stage_video(
            positions,
            stage_video_path,
            width=800,
            height=450,
            labels=labels,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to render stage video: {exc}",
        ) from exc

    import cv2

    ref_cap = cv2.VideoCapture(str(reference_path))
    if not ref_cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to open reference video {reference_path.name}",
        )
    ref_w = int(ref_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    ref_h = int(ref_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    ref_cap.release()
    if ref_w <= 1 or ref_h <= 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Reference {reference_path.name} has invalid dimensions {ref_w}x{ref_h}",
        )
    half_w = ref_w // 2
    if request.reference_side == "left":
        crop = (0, 0, half_w, ref_h)
    else:
        crop = (ref_w - half_w, 0, half_w, ref_h)

    comparison_path = job_dir / "comparison.mp4"
    try:
        comparison_path = concat_side_by_side(
            left_video=reference_path,
            right_video=stage_video_path,
            output_path=comparison_path,
            left_crop=crop,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to build comparison video: {exc}",
        ) from exc

    metrics = _compute_compare_metrics(positions, job.video_meta.fps)

    return CompareToReferenceResponse(
        comparison_video_url=f"{settings.api_prefix}/jobs/{job_id}/comparison-video",
        metrics=metrics,
    )


@router.get("/jobs/{job_id}/comparison-video")
def get_comparison_video(job_id: str) -> FileResponse:
    store = JobStore()
    try:
        store.load_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    job_dir = store.job_dir(job_id)
    comparison = _find_comparison_video(job_dir)
    if comparison is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No comparison video for job {job_id} yet. "
                f"POST /jobs/{job_id}/compare-to-reference first."
            ),
        )
    media_type = "video/mp4" if comparison.suffix == ".mp4" else "video/x-msvideo"
    return FileResponse(
        path=comparison,
        media_type=media_type,
        filename=f"{job_id}_comparison{comparison.suffix}",
    )
