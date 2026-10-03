"""Camera motion compensation (pan / tilt / zoom).

The floor mapping (corners or auto-calibration) is solved in ONE camera view.
If the camera zooms or pans afterwards, a dancer standing still appears to
move — a zoom-in looks like everyone walking toward the camera. Practice
videos often reframe like this.

A camera that pans/tilts/zooms about a fixed position relates any two frames
by a homography — between consecutive frames, nearly a similarity (shift +
zoom + slight roll), which is what we fit, robustly, from the
BACKGROUND (walls, floor markings) with sparse optical flow, masking out the
people, and chain them into M_t: frame t pixels -> reference (key) frame
pixels. Positions are converted with M_t before floor projection.

Per-step motions below _STILL_PX are treated as zero so a fixed camera
accumulates no drift.
"""
from __future__ import annotations

import logging
from pathlib import Path


logger = logging.getLogger("uvicorn.error")

_WORK_WIDTH = 640
_STILL_PX = 0.35  # median background flow below this = camera didn't move
_MIN_POINTS = 25
# Most background points must agree on the camera motion. Moving dancers that
# slip past the person mask disagree with each other, so they can't reach
# this together — which is what stops a fixed camera from "drifting".
_MIN_INLIER_FRACTION = 0.6

Matrix = list[list[float]]
Box = tuple[int, int, int, int]


def _identity() -> Matrix:
    return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]


def apply_h(m: Matrix, x: float, y: float) -> tuple[float, float]:
    d = m[2][0] * x + m[2][1] * y + m[2][2]
    if abs(d) < 1e-9:
        return x, y
    return (
        (m[0][0] * x + m[0][1] * y + m[0][2]) / d,
        (m[1][0] * x + m[1][1] * y + m[1][2]) / d,
    )


class CameraMotionEstimator:
    """Feed frames in order with person boxes (when known); get per-frame
    step homographies (frame t -> frame t-1)."""

    def __init__(self, frame_size: tuple[int, int]) -> None:
        import numpy as np

        w, h = frame_size
        self.scale = _WORK_WIDTH / max(w, 1)
        self.size = (_WORK_WIDTH, max(int(round(h * self.scale)), 16))
        self.prev_gray = None
        self.steps: list = []  # numpy 3x3, full-res coords, frame t -> t-1
        self.moving_steps = 0
        self._np = np

    def _mask(self, boxes: list[Box]):
        np = self._np
        m = np.full((self.size[1], self.size[0]), 255, np.uint8)
        for x, y, bw, bh in boxes:
            pad_x, pad_y = int(bw * 0.15), int(bh * 0.08)
            x0 = max(int((x - pad_x) * self.scale), 0)
            y0 = max(int((y - pad_y) * self.scale), 0)
            x1 = min(int((x + bw + pad_x) * self.scale), self.size[0])
            y1 = min(int((y + bh + pad_y) * self.scale), self.size[1])
            m[y0:y1, x0:x1] = 0
        return m

    def add_frame(self, frame_bgr, boxes: list[Box]) -> None:
        import cv2

        np = self._np
        gray = cv2.cvtColor(cv2.resize(frame_bgr, self.size), cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.prev_gray = gray
            self.steps.append(np.eye(3))
            return
        step = np.eye(3)
        pts = cv2.goodFeaturesToTrack(
            self.prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8,
            mask=self._mask(boxes),
        )
        if pts is not None and len(pts) >= _MIN_POINTS:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts, None)
            ok = st.reshape(-1) == 1
            p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
            if len(p0) >= _MIN_POINTS:
                flow = np.median(np.linalg.norm(p1 - p0, axis=1))
                if flow * (1 / self.scale) >= _STILL_PX:
                    # Pan / tilt / zoom (+ slight roll) between consecutive
                    # frames is a similarity transform to good approximation;
                    # a full homography overfits to whatever moving people
                    # slip past the mask.
                    sim, inl = cv2.estimateAffinePartial2D(
                        p1, p0, method=cv2.RANSAC, ransacReprojThreshold=1.0
                    )
                    if sim is not None and inl is not None:
                        n_in = int(inl.sum())
                        zoom = float(np.hypot(sim[0, 0], sim[1, 0]))
                        shift = float(np.hypot(sim[0, 2], sim[1, 2]))
                        plausible = (
                            n_in >= _MIN_POINTS
                            and n_in >= _MIN_INLIER_FRACTION * len(p0)
                            and abs(zoom - 1.0) < 0.08
                            and shift < 0.08 * self.size[0]
                        )
                        if plausible:
                            hm = np.vstack([sim, [0.0, 0.0, 1.0]])
                            s_ = np.diag([self.scale, self.scale, 1.0])
                            step = np.linalg.inv(s_) @ hm @ s_
                            self.moving_steps += 1
        self.steps.append(step)
        self.prev_gray = gray

    def to_reference(self, key_frame: int) -> list[Matrix]:
        """Chain steps into M_t (frame t -> key frame) for every frame."""
        np = self._np
        n = len(self.steps)
        if n == 0:
            return []
        k = min(max(key_frame, 0), n - 1)
        mats = [np.eye(3) for _ in range(n)]
        for t in range(k + 1, n):
            mats[t] = mats[t - 1] @ self.steps[t]
        for t in range(k - 1, -1, -1):
            # steps[t+1] maps t+1 -> t; we need t -> t+1.
            mats[t] = mats[t + 1] @ np.linalg.inv(self.steps[t + 1])
        out = []
        for m in mats:
            m = m / m[2, 2]
            out.append([[float(v) for v in row] for row in m])
        return out


def camera_is_moving(mats: list[Matrix], frame_size: tuple[int, int]) -> bool:
    """True if any frame's view differs from the reference by more than a
    couple of pixels at the frame corners."""
    w, h = frame_size
    corners = [(0, 0), (w, 0), (w, h), (0, h)]
    for m in mats:
        for x, y in corners:
            u, v = apply_h(m, x, y)
            if abs(u - x) > 3 or abs(v - y) > 3:
                return True
    return False


def estimate_camera_motion(
    video_path: Path,
    key_frame: int,
    boxes_by_frame: dict[int, list[Box]] | None = None,
) -> list[Matrix]:
    """Standalone pass over the video (used when no per-frame loop exists)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video file: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    est = CameraMotionEstimator((w, h))
    last_boxes: list[Box] = []
    fi = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if boxes_by_frame and fi in boxes_by_frame:
                last_boxes = boxes_by_frame[fi]
            est.add_frame(frame, last_boxes)
            fi += 1
    finally:
        cap.release()
    return est.to_reference(key_frame)
