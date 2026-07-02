# Samples

Place short local dance-practice clips here for manual testing.

Suggested manual test flow:

1. Start the backend service.
2. Upload a short `.mp4` clip to `POST /api/v1/jobs/upload`.
3. Poll `GET /api/v1/jobs/{job_id}` until `completed`.
4. Fetch `GET /api/v1/jobs/{job_id}/positions`.

