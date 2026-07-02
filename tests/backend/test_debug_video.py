from pathlib import Path

import cv2
import numpy as np

from app.pipeline.debug_video import DebugVideoRenderer
from app.schemas.jobs import DancerPosition


def test_debug_video_renderer_writes_mp4(tmp_path: Path) -> None:
    output_path = tmp_path / "debug_overlay.mp4"
    renderer = DebugVideoRenderer(
        output_path=output_path,
        fps=30.0,
        frame_size=(320, 180),
    )

    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    renderer.write_frame(
        frame,
        [
            DancerPosition(
                id=1,
                bbox=[10, 20, 40, 80],
                anchor_px=[30, 100],
                x=0.09375,
                y=0.444444,
                confidence=0.92,
            )
        ],
    )
    renderer.close()

    assert output_path.exists()
    assert output_path.stat().st_size > 0

    capture = cv2.VideoCapture(str(output_path))
    assert capture.isOpened()
    ok, first_frame = capture.read()
    capture.release()
    assert ok
    assert first_frame is not None
