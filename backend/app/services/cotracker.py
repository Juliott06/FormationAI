"""CoTracker3 (Meta) click-to-track on CPU, in-process.

Loads the model lazily via torch hub on first use. The checkpoint is cached
under ~/.cache/torch/hub so first run downloads once (~100 MB).

This is a lighter alternative to SAM2:
- Tracks point queries (not segmentation masks)
- Runs on CPU at ~2 fps
- Identity is solved by construction (N click points = N tracks across whole clip)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import get_settings


logger = logging.getLogger("uvicorn.error")


@dataclass(frozen=True)
class TrackedPoint:
    """Per-frame xy position for one tracked dancer."""

    frame: int
    x: int
    y: int
    visible: bool


class CoTrackerError(RuntimeError):
    """Raised when CoTracker fails to load or run."""


class CoTrackerNotConfigured(CoTrackerError):
    """Raised when no backend is configured; surfaced as HTTP 503."""


class CoTrackerClient(ABC):
    @abstractmethod
    def track_video(
        self,
        video_path: Path,
        clicks: list[dict],
    ) -> list[list[TrackedPoint]]:
        """clicks: [{"name": str, "key_frame": int, "x": int, "y": int}, ...]
        Returns: per-input-click list of TrackedPoint (one per video frame).
        Same order as input clicks. The caller groups by name to support multiple
        click points per dancer.

        Coords are in the ORIGINAL video pixel space (the client handles resize).
        """


class DisabledClient(CoTrackerClient):
    def track_video(self, video_path, clicks) -> list[list[TrackedPoint]]:
        raise CoTrackerNotConfigured(
            "CoTracker backend is not configured. "
            "Set FORMATIONAI_COTRACKER_BACKEND=local."
        )


class LocalClient(CoTrackerClient):
    """Runs CoTracker3 online (sliding-window) in-process on CPU.

    Streams the video from disk to keep memory bounded (~1.5 GB peak for any
    clip length at 960x540). Uses the click points as query inputs at the
    user-chosen key frame."""

    _model: Any = None  # Loaded lazily and cached at class level

    def __init__(self, resize_width: int = 960) -> None:
        self._resize_width = max(int(resize_width), 320)

    @classmethod
    def _ensure_model(cls) -> Any:
        if cls._model is not None:
            return cls._model
        try:
            import torch
        except ImportError as exc:
            raise CoTrackerError(
                "torch is not installed; CoTracker local backend requires PyTorch"
            ) from exc
        logger.info("loading CoTracker3 online model via torch hub (cached)...")
        try:
            model = torch.hub.load(
                "facebookresearch/co-tracker",
                "cotracker3_online",
                trust_repo=True,
            )
        except Exception as exc:
            raise CoTrackerError(f"failed to load CoTracker3: {exc}") from exc
        model.eval()
        cls._model = model
        logger.info("CoTracker3 ready (online, step=%d)", model.step)
        return cls._model

    def track_video(self, video_path, clicks):
        import cv2
        import numpy as np
        import torch

        if not clicks:
            return {}

        model = self._ensure_model()
        step = int(model.step)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise CoTrackerError(f"cannot open video {video_path}")
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if src_w <= 0 or src_h <= 0 or n_frames <= 0:
            cap.release()
            raise CoTrackerError(
                f"invalid video dims {src_w}x{src_h} or frame count {n_frames}"
            )

        scale = self._resize_width / src_w
        rw = self._resize_width
        rh = max(int(round(src_h * scale)), 16)
        logger.info(
            "CoTracker: video=%s src=%dx%d resize=%dx%d frames=%d clicks=%d",
            video_path.name, src_w, src_h, rw, rh, n_frames, len(clicks),
        )

        queries: list[list[float]] = []
        for click in clicks:
            kf = int(click.get("key_frame", 0))
            qx = float(click["x"]) * scale
            qy = float(click["y"]) * scale
            queries.append([float(kf), qx, qy])
        queries_tensor = torch.tensor(queries, dtype=torch.float32).unsqueeze(0)

        # Stream all frames into memory as a single tensor. Online mode still
        # processes incrementally but the public API expects the full tensor;
        # at 960x540 RGB this is ~1.5 MB/frame so a 5000-frame clip is ~7.5 GB
        # — too much. We use a chunked feed via video_chunk argument instead.
        all_tracks: list[Any] = []
        all_vis: list[Any] = []

        # Read all frames first as a list of numpy arrays (~1.5 MB each).
        # For Gabriela 4919 frames at 960x540 → ~7.5 GB. We must instead
        # build a generator that yields chunks. CoTracker online API needs
        # video_chunk of shape [B, T_chunk, C, H, W] where T_chunk >= step*2.
        chunk_size = step * 2
        chunk_buffer: list[np.ndarray] = []
        frames_read = 0
        last_logged = 0
        import time as _time
        t0 = _time.perf_counter()

        with torch.no_grad():
            initialized = False
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.resize(frame, (rw, rh))
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                chunk_buffer.append(frame_rgb)
                frames_read += 1

                # When we have a full window, run a step
                if len(chunk_buffer) >= chunk_size:
                    arr = np.stack(chunk_buffer[:chunk_size])  # T, H, W, C
                    chunk_tensor = (
                        torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0)
                    )
                    if not initialized:
                        model(
                            video_chunk=chunk_tensor,
                            is_first_step=True,
                            queries=queries_tensor,
                        )
                        initialized = True
                    tracks, vis = model(video_chunk=chunk_tensor)
                    all_tracks.append(tracks.detach().cpu())
                    all_vis.append(vis.detach().cpu())
                    # advance by `step` (sliding window), keep last `step` for context
                    chunk_buffer = chunk_buffer[step:]

                    if frames_read - last_logged >= 200:
                        elapsed = _time.perf_counter() - t0
                        rate = frames_read / max(elapsed, 1e-6)
                        logger.info(
                            "CoTracker progress: %d/%d frames (%.1f fps)",
                            frames_read, n_frames, rate,
                        )
                        last_logged = frames_read

            # Final partial chunk: pad to window size and run
            if chunk_buffer and initialized:
                pad_needed = chunk_size - len(chunk_buffer)
                if pad_needed > 0:
                    last = chunk_buffer[-1]
                    chunk_buffer = chunk_buffer + [last] * pad_needed
                arr = np.stack(chunk_buffer)
                chunk_tensor = (
                    torch.from_numpy(arr).permute(0, 3, 1, 2).float().unsqueeze(0)
                )
                tracks, vis = model(video_chunk=chunk_tensor)
                all_tracks.append(tracks.detach().cpu())
                all_vis.append(vis.detach().cpu())

        cap.release()

        if not all_tracks:
            raise CoTrackerError("CoTracker produced no output")

        # CoTracker online returns CUMULATIVE tracks per call (the last call
        # contains the full history). Take the LAST result and truncate to
        # the actual frame count (we may have padded the last chunk).
        final_tracks = all_tracks[-1]
        final_vis = all_vis[-1]
        T = final_tracks.shape[1]
        actual_T = min(T, n_frames)
        final_tracks = final_tracks[:, :actual_T]
        final_vis = final_vis[:, :actual_T]

        N = len(clicks)
        if final_tracks.shape[2] != N:
            raise CoTrackerError(
                f"unexpected track count {final_tracks.shape[2]} for {N} queries"
            )

        inv_scale = 1.0 / scale
        tracks_np = final_tracks[0].numpy()
        vis_np = final_vis[0].numpy()
        result: list[list[TrackedPoint]] = []
        for ni in range(N):
            points: list[TrackedPoint] = []
            for ti in range(actual_T):
                xr = float(tracks_np[ti, ni, 0])
                yr = float(tracks_np[ti, ni, 1])
                v = float(vis_np[ti, ni])
                x_src = int(round(xr * inv_scale))
                y_src = int(round(yr * inv_scale))
                x_src = max(0, min(src_w - 1, x_src))
                y_src = max(0, min(src_h - 1, y_src))
                points.append(
                    TrackedPoint(frame=ti, x=x_src, y=y_src, visible=v >= 0.5)
                )
            result.append(points)
        elapsed = _time.perf_counter() - t0
        logger.info(
            "CoTracker done: %d frames @ %.1f fps over %.1fs",
            actual_T, actual_T / max(elapsed, 1e-6), elapsed,
        )
        return result


def build_client() -> CoTrackerClient:
    settings = get_settings()
    if settings.cotracker_backend == "local":
        return LocalClient(resize_width=settings.cotracker_resize_width)
    return DisabledClient()
