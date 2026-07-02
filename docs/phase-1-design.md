# FormationAI Phase 1 Design

## Goal

Implement the smallest working milestone for FormationAI:

`dance_practice.mp4 -> tracked dancer positions JSON`

## Scope

Phase 1 includes:

- FastAPI upload and job-status endpoints
- local filesystem job storage
- OpenCV video metadata and frame iteration
- Ultralytics YOLO person detection
- identity tracking via Ultralytics built-in tracker (ByteTrack by default; BoT-SORT optional)
- normalized proxy stage coordinates
- automatic debug overlay video generation

Phase 1 excludes:

- production UI
- floor-plane homography
- beat alignment
- formation change-point detection
- editing and export workflows

## Processing Rules

- job states: `queued`, `processing`, `completed`, `failed`
- optional upload form field: `expected_dancer_count`
- anchor point fallback:
  1. midpoint of left and right ankles
  2. midpoint of left and right hips
  3. bounding-box bottom center
- normalized proxy coordinates:
  - `x = anchor_px_x / frame_width`
  - `y = 1 - (anchor_px_y / frame_height)`
- track persistence is handled by the configured Ultralytics tracker (`FORMATIONAI_TRACKER_CONFIG`, default `bytetrack.yaml`)

## MediaPipe Model Asset

The repository still includes MediaPipe-related code for later pose work, but the active Phase 1 detector is Ultralytics YOLO person detection.

Ultralytics downloads official pretrained detection weights automatically on first use, based on the configured model name such as `yolo26s.pt`.
