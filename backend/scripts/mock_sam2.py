"""Mock SAM2 server for dry-running the click-to-track flow without a GPU.

Reads the uploaded video to know its frame count, then for each click returns
tracks where each dancer stays at the click point across every frame (with a
tiny horizontal drift so you can see motion). Identity is rock-solid because
it's just the input echoed back — no real model in the loop.

Run:
    uvicorn backend.scripts.mock_sam2:app --port 9200

Then in backend/.env:
    FORMATIONAI_SAM2_BACKEND=http
    FORMATIONAI_SAM2_URL=http://localhost:9200/track

Restart the main uvicorn after editing .env (settings are cached).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

app = FastAPI()


def _frame_count(video_bytes: bytes) -> tuple[int, int, int]:
    """Return (frame_count, width, height) of the uploaded video."""
    import cv2

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = Path(tmp.name)
    try:
        capture = cv2.VideoCapture(str(tmp_path))
        if not capture.isOpened():
            return 0, 0, 0
        n = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        capture.release()
        return n, w, h
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/track")
async def track(
    video: UploadFile = File(...),
    clicks_json: str = Form(...),
) -> dict:
    clicks = json.loads(clicks_json)
    video_bytes = await video.read()
    n_frames, w, h = _frame_count(video_bytes)
    if n_frames <= 0:
        return {"tracks": {}}

    tracks: dict[str, list[dict]] = {}
    box_w, box_h = 80, 160
    for click in clicks:
        name = click["name"]
        cx0, cy0 = int(click["x"]), int(click["y"])
        per_frame: list[dict] = []
        for fi in range(n_frames):
            drift = (fi % 60) - 30
            cx = max(box_w // 2, min(w - box_w // 2, cx0 + drift))
            cy = cy0
            x1 = cx - box_w // 2
            y1 = max(0, cy - box_h // 2)
            x2 = cx + box_w // 2
            y2 = min(h, cy + box_h // 2)
            per_frame.append({"frame": fi, "bbox": [x1, y1, x2, y2], "score": 0.95})
        tracks[name] = per_frame
    return {"tracks": tracks}


@app.get("/health")
def health() -> dict:
    return {"ok": True}
