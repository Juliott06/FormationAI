"""Mock LocateAnything server for dry-running the per-frame flow without a GPU.

Returns fake boxes spread evenly across the frame width. Real detection won't match
these boxes, so the named-matcher will fall through to the YOLO fallback for every
dancer — but every code path (HTTP, parsing, matcher, fallback) gets exercised.

Run:
    uvicorn backend.scripts.mock_locateanything:app --port 9100

Then in backend/.env:
    FORMATIONAI_LOCATEANYTHING_BACKEND=http
    FORMATIONAI_LOCATEANYTHING_URL=http://localhost:9100/locate

Restart the main uvicorn after editing .env (settings are cached).
"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class LocateRequest(BaseModel):
    image_b64: str
    prompts: dict[str, str]


@app.post("/locate")
def locate(req: LocateRequest) -> dict:
    n = max(len(req.prompts), 1)
    boxes: dict[str, list[int]] = {}
    for i, name in enumerate(req.prompts.keys()):
        cx = int((i + 0.5) / n * 1000)
        boxes[name] = [max(0, cx - 60), 300, min(1000, cx + 60), 800]
    return {"boxes": boxes}


@app.get("/health")
def health() -> dict:
    return {"ok": True}
