# FormationAI API Contract

## Endpoints

### `GET /health`

Returns service health.

### `POST /api/v1/jobs/upload`

Uploads one video file and creates a processing job.

Optional form field:

- `expected_dancer_count`

Response:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "expected_dancer_count": 9,
  "video_meta": {
    "filename": "dance_practice.mp4",
    "fps": 30.0,
    "frame_count": 5400,
    "width": 1920,
    "height": 1080,
    "duration_sec": 180.0
  }
}
```

### `GET /api/v1/jobs/{job_id}`

Returns job progress and terminal failure details.

### `GET /api/v1/jobs/{job_id}/positions`

Returns the canonical result JSON after completion.

### `GET /api/v1/jobs/{job_id}/summary`

Returns a lightweight summary payload for browser-safe inspection.

### `GET /api/v1/jobs/{job_id}/positions-file`

Downloads `positions.json` as a file instead of pretty-printing the full payload inline.

### `GET /api/v1/jobs/{job_id}/debug-video`

Returns an MP4 overlay video after completion with:

- bounding boxes
- anchor points
- track IDs
- per-detection confidence labels

## Result Shape

```json
{
  "job_id": "uuid",
  "video": {
    "filename": "dance_practice.mp4",
    "fps": 30.0,
    "frame_count": 5400,
    "width": 1920,
    "height": 1080,
    "duration_sec": 180.0
  },
  "coordinate_space": {
    "image_anchor_px": "pixel anchor point in the source frame",
    "normalized_stage_proxy": "heuristic top-down proxy derived from image-space anchors"
  },
  "summary": {
    "expected_dancer_count": 9,
    "unique_track_ids": 13,
    "max_dancers_in_frame": 4,
    "average_dancers_per_frame": 2.8,
    "frames_with_detections": 287,
    "frames_below_expected": 300,
    "frames_meeting_expected": 0
  },
  "frames": [
    {
      "frame": 1240,
      "timestamp_sec": 41.333,
      "dancers": [
        {
          "id": 1,
          "bbox": [100, 200, 80, 190],
          "anchor_px": [140, 390],
          "x": 0.31,
          "y": 0.22,
          "confidence": 0.95
        }
      ]
    }
  ]
}
```

The active detection stage uses a pretrained Ultralytics YOLO person detector and tracks only COCO `person` detections.
