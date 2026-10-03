"""Upload normalization: make frame indices mean the same thing everywhere.

Phone recordings are usually variable-frame-rate (VFR). On a VFR file,
OpenCV's CAP_PROP_POS_FRAMES seek converts the index to a time using the
AVERAGE fps and lands on the wrong frame — measured: frame 300 of a clip with
one 0.37s timestamp gap came back as a different frame. That broke
click-to-track: the click picker (which seeks) showed frame N, CoTracker
(which decodes sequentially) seeded the user's clicks on its own frame N, a
different moment — so tracks started on the wrong spots. Timestamps computed
as frame/fps drifted from the real video too, desyncing reference compares.

Re-encoding VFR uploads to constant frame rate once, at upload, makes seek,
sequential decode and frame/fps timestamps all agree.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path


logger = logging.getLogger("uvicorn.error")


def is_variable_frame_rate(video_path: Path, *, tolerance: float = 0.5) -> bool:
    """True if any inter-frame gap deviates from the median gap by more than
    `tolerance` x median. Decodes only packets' timestamps via grab()."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False
    stamps: list[float] = []
    try:
        while cap.grab():
            stamps.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)))
    finally:
        cap.release()
    if len(stamps) < 3:
        return False
    gaps = [b - a for a, b in zip(stamps, stamps[1:])]
    ordered = sorted(gaps)
    median = ordered[len(ordered) // 2]
    if median <= 0:
        return False
    return any(abs(g - median) > tolerance * median for g in gaps)


def normalize_frame_rate(video_path: Path) -> bool:
    """If the video is VFR, re-encode it in place to constant frame rate at its
    average fps. Returns True if the file was rewritten. Any failure leaves the
    original untouched (the app still works, seek just may be imprecise)."""
    import cv2

    try:
        if not is_variable_frame_rate(video_path):
            return False
    except Exception:
        logger.exception("VFR check failed for %s", video_path.name)
        return False
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    if fps <= 0:
        return False
    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        logger.warning("VFR video %s but imageio-ffmpeg unavailable", video_path.name)
        return False
    tmp = video_path.with_name(video_path.stem + "_cfr.mp4")
    cmd = [
        ffmpeg, "-y", "-i", str(video_path),
        "-vf", f"fps={fps:.6f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", str(tmp),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=1800)
        if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            logger.warning(
                "CFR re-encode failed for %s: %s",
                video_path.name, result.stderr[-300:].decode(errors="replace"),
            )
            tmp.unlink(missing_ok=True)
            return False
    except Exception as exc:
        logger.warning("CFR re-encode error for %s: %s", video_path.name, exc)
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(video_path)
    logger.info("re-encoded variable-frame-rate upload %s to constant %.3f fps", video_path.name, fps)
    return True
