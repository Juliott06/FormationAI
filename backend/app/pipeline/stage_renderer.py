from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.schemas.jobs import PositionsResult


logger = logging.getLogger("uvicorn.error")

_CODEC_FALLBACK = [
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    """BGR color derived from track_id. Matches DebugVideoRenderer's scheme."""
    blue = (track_id * 97) % 255
    green = (track_id * 57 + 80) % 255
    red = (track_id * 137 + 40) % 255
    return int(blue), int(green), int(red)


def render_stage_video(
    positions: PositionsResult,
    output_path: Path,
    *,
    width: int = 800,
    height: int = 450,
    fps: float | None = None,
    labels: dict[int, str] | None = None,
) -> Path:
    """Write a stage-view video: blank canvas + colored named dots over time.

    Returns the actual output path (may differ from `output_path` if codec
    fallback chose a different extension). Mirrors DebugVideoRenderer's
    codec-fallback chain.
    """
    import cv2
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fps = fps if fps is not None else max(positions.video.fps, 1.0)
    labels = labels or {}

    writer: Any = None
    chosen_path = output_path
    chosen_codec = "none"
    for codec, ext in _CODEC_FALLBACK:
        candidate = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate_writer = cv2.VideoWriter(
            str(candidate),
            fourcc,
            max(fps, 1.0),
            (max(width, 1), max(height, 1)),
        )
        if candidate_writer.isOpened():
            writer = candidate_writer
            chosen_path = candidate
            chosen_codec = codec
            logger.info(
                "stage video writer ready: %s (codec=%s, %.1f fps, %dx%d)",
                candidate.name, codec, fps, width, height,
            )
            break
        logger.warning("stage video codec %s failed to open at %s", codec, candidate)

    if writer is None:
        raise ValueError(
            f"Unable to create stage video — no working codec found. "
            f"Tried: {[c for c, _ in _CODEC_FALLBACK]}"
        )

    canvas_bg = np.zeros((height, width, 3), dtype=np.uint8)
    canvas_bg[:] = (24, 28, 32)  # BGR ~ #1a1c20
    floor_y = height - 2
    try:
        for frame_positions in positions.frames:
            frame = canvas_bg.copy()
            cv2.line(frame, (0, floor_y), (width, floor_y), (90, 90, 90), 1)
            cv2.putText(
                frame,
                "front of stage",
                (8, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (140, 140, 140),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"f{frame_positions.frame}  t={frame_positions.timestamp_sec:.2f}s",
                (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 170),
                1,
                cv2.LINE_AA,
            )
            for dancer in frame_positions.dancers:
                cx = int(round(dancer.x * width))
                cy = int(round((1.0 - dancer.y) * height))
                color = _color_for_track(dancer.id)
                cv2.circle(frame, (cx, cy), 14, color, -1)
                cv2.circle(frame, (cx, cy), 14, (255, 255, 255), 2)
                text = labels.get(dancer.id, str(dancer.id))
                (tw, th), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.putText(
                    frame,
                    text,
                    (cx - tw // 2, cy + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            writer.write(frame)
    finally:
        writer.release()

    size = chosen_path.stat().st_size if chosen_path.exists() else 0
    logger.info(
        "stage video closed: %d frames, codec=%s, file=%s, size=%d",
        len(positions.frames), chosen_codec, chosen_path.name, size,
    )
    return chosen_path


def concat_side_by_side(
    left_video: Path,
    right_video: Path,
    output_path: Path,
    *,
    left_crop: tuple[int, int, int, int] | None = None,
    right_crop: tuple[int, int, int, int] | None = None,
) -> Path:
    """Read two videos frame by frame, scale to a common height, concat horizontally.

    left_crop/right_crop: optional (x, y, w, h) crop applied to each source.
    Returns the actual output path (codec-fallback may swap extension)."""
    import cv2
    import numpy as np

    left_cap = cv2.VideoCapture(str(left_video))
    right_cap = cv2.VideoCapture(str(right_video))
    if not left_cap.isOpened():
        raise FileNotFoundError(f"Unable to open left video: {left_video}")
    if not right_cap.isOpened():
        left_cap.release()
        raise FileNotFoundError(f"Unable to open right video: {right_video}")

    fps = max(
        left_cap.get(cv2.CAP_PROP_FPS) or 0.0,
        right_cap.get(cv2.CAP_PROP_FPS) or 0.0,
        1.0,
    )

    def _read(cap, crop):
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        if crop is not None:
            x, y, w, h = crop
            frame = frame[y : y + h, x : x + w]
        return frame

    first_left = _read(left_cap, left_crop)
    first_right = _read(right_cap, right_crop)
    if first_left is None or first_right is None:
        left_cap.release()
        right_cap.release()
        raise ValueError("One of the videos has no readable frames")

    target_h = max(first_left.shape[0], first_right.shape[0])

    def _resize_to_height(img, h):
        ratio = h / max(img.shape[0], 1)
        new_w = max(int(round(img.shape[1] * ratio)), 1)
        return cv2.resize(img, (new_w, h))

    first_left_r = _resize_to_height(first_left, target_h)
    first_right_r = _resize_to_height(first_right, target_h)
    out_w = first_left_r.shape[1] + first_right_r.shape[1]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: Any = None
    chosen_path = output_path
    for codec, ext in _CODEC_FALLBACK:
        candidate = output_path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        candidate_writer = cv2.VideoWriter(
            str(candidate), fourcc, fps, (out_w, target_h)
        )
        if candidate_writer.isOpened():
            writer = candidate_writer
            chosen_path = candidate
            logger.info(
                "comparison video writer ready: %s (codec=%s)", candidate.name, codec
            )
            break

    if writer is None:
        left_cap.release()
        right_cap.release()
        raise ValueError("No working codec for comparison video")

    try:
        combined = np.hstack([first_left_r, first_right_r])
        writer.write(combined)
        while True:
            l = _read(left_cap, left_crop)
            r = _read(right_cap, right_crop)
            if l is None or r is None:
                break
            l = _resize_to_height(l, target_h)
            r = _resize_to_height(r, target_h)
            if l.shape[1] + r.shape[1] != out_w:
                # Pad/truncate to keep output width stable
                r_target = out_w - l.shape[1]
                if r_target <= 0:
                    continue
                r = cv2.resize(r, (r_target, target_h))
            writer.write(np.hstack([l, r]))
    finally:
        writer.release()
        left_cap.release()
        right_cap.release()

    return chosen_path
