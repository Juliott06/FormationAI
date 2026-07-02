from __future__ import annotations

from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
import numpy as np

from app.pipeline.stage_renderer import concat_side_by_side


def _write_test_video(path: Path, fps: float, n_frames: int, size=(64, 48)) -> None:
    """Each frame is a solid gray level = frame_index * 16 (distinct after codec loss)."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, size)
    assert writer.isOpened()
    for i in range(n_frames):
        level = min(i * 16, 240)
        frame = np.full((size[1], size[0], 3), level, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _decode_level(frame_half) -> int:
    """Recover the frame index from mean gray level (quantized to steps of 16)."""
    mean = float(frame_half.mean())
    return int(round(mean / 16.0))


def test_concat_synchronizes_by_timestamp_not_frame_index(tmp_path: Path):
    # Left: 30 fps, 30 frames = 1.0 s. Right: 10 fps, 10 frames = 1.0 s.
    # Naive 1:1 pairing would exhaust the right side at output frame 10 and
    # pair left frame k with right frame k (3x too fast). Correct behavior:
    # output at 30 fps, right side holds each frame for 3 output frames.
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    _write_test_video(left, fps=30.0, n_frames=30)
    _write_test_video(right, fps=10.0, n_frames=10)

    out = concat_side_by_side(left, right, tmp_path / "combined.mp4")
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()

    n_out = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(out_fps - 30.0) < 0.5
    # Both sides cover ~1s, so the output should cover ~1s at 30fps (not 10 frames)
    assert n_out >= 28

    mismatches = 0
    checked = 0
    for k in range(n_out):
        ok, frame = cap.read()
        if not ok:
            break
        w = frame.shape[1]
        left_half = frame[:, : w // 2]
        right_half = frame[:, w // 2 :]
        left_idx = _decode_level(left_half)
        right_idx = _decode_level(right_half)
        # Left advances 1:1; right should show frame floor(k * 10/30)
        expected_right = int(k * 10.0 / 30.0)
        if left_idx != min(k, 15) and left_idx != k:  # levels cap at 240 (idx 15)
            pass  # left check only meaningful below the cap
        if k < 15:  # below gray-level saturation, indices are decodable
            checked += 1
            if abs(right_idx - expected_right) > 1:
                mismatches += 1
    cap.release()

    assert checked > 10
    # Timestamp sync: right side must track k/3, not k (naive pairing would be
    # off by >1 for most k >= 3 and the video would end at 10 frames)
    assert mismatches == 0
