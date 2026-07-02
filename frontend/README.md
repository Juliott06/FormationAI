# FormationAI Frontend

Minimal React + Vite + TypeScript single-page app for the Phase 1 backend.

## Setup

```
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173` and proxies `/api/*` to `http://localhost:8000`, so you also need the backend running:

```
cd backend
uvicorn app.main:app --reload
```

## What it does

- Uploads a video to `POST /api/v1/jobs/upload`.
- Polls `GET /api/v1/jobs/{id}` once per second while the job runs.
- On completion, fetches `GET /api/v1/jobs/{id}/positions` and renders a top-down stage view with a frame scrubber.
- Provides a link to the debug overlay video.

## What it does not do yet

- Formation segmentation.
- Editing / export.
- Rendering the original video alongside the stage view.
