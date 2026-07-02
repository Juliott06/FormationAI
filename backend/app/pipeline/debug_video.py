from __future__ import annotations

import logging
from pathlib import Path

from app.schemas.jobs import DancerPosition


logger = logging.getLogger("uvicorn.error")

# Try codecs in order; first one that opens wins. mp4v is usually fine on macOS/Linux
# but flaky on Windows depending on installed codecs. XVID is the reliable fallback.
_CODEC_FALLBACK = [
    ("mp4v", ".mp4"),
    ("avc1", ".mp4"),
    ("XVID", ".avi"),
    ("MJPG", ".avi"),
]


def _color_for_track(track_id: int) -> tuple[int, int, int]:
    # Deterministic BGR color derived from the track ID for stable overlays.
    blue = (track_id * 97) % 255
    green = (track_id * 57 + 80) % 255
    red = (track_id * 137 + 40) % 255
    return int(blue), int(green), int(red)


class DebugVideoRenderer:
    def __init__(
        self,
        *,
        output_path: Path,
        fps: float,
        frame_size: tuple[int, int],
    ) -> None:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        width, height = frame_size

        self._writer = None
        self.output_path = output_path
        self._codec = "none"
        for codec, ext in _CODEC_FALLBACK:
            candidate = output_path.with_suffix(ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(
                str(candidate),
                fourcc,
                max(fps, 1.0),
                (max(width, 1), max(height, 1)),
            )
            if writer.isOpened():
                self._writer = writer
                self.output_path = candidate
                self._codec = codec
                logger.info(
                    "debug video writer ready: %s (codec=%s, %.1f fps, %dx%d)",
                    candidate.name, codec, fps, width, height,
                )
                break
            else:
                logger.warning("debug video codec %s failed to open at %s", codec, candidate)

        if self._writer is None:
            raise ValueError(
                f"Unable to create debug video — no working codec found. "
                f"Tried: {[c for c, _ in _CODEC_FALLBACK]}"
            )

        self._frames_written = 0

    def write_frame(self, frame: object, dancers: list[DancerPosition]) -> None:
        import cv2

        annotated_frame = frame.copy()
        for dancer in dancers:
            color = _color_for_track(dancer.id)
            x, y, width, height = dancer.bbox
            anchor_x, anchor_y = dancer.anchor_px

            cv2.rectangle(
                annotated_frame,
                (x, y),
                (x + width, y + height),
                color,
                2,
            )
            cv2.circle(annotated_frame, (anchor_x, anchor_y), 5, color, -1)
            cv2.putText(
                annotated_frame,
                f"ID {dancer.id}",
                (x, max(y - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated_frame,
                f"{dancer.confidence:.2f}",
                (x, min(y + height + 20, annotated_frame.shape[0] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        self._writer.write(annotated_frame)
        self._frames_written += 1

    def close(self) -> None:
        self._writer.release()
        exists = self.output_path.exists()
        size = self.output_path.stat().st_size if exists else 0
        logger.info(
            "debug video closed: %d frames written, codec=%s, file=%s, exists=%s, size=%d bytes",
            self._frames_written, self._codec, self.output_path.name, exists, size,
        )
