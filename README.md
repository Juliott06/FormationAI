# FormationAI

FormationAI is a senior-design-level ECE/CompE project for extracting K-pop dance formations from practice videos.

Phase 1 implements a backend-first vertical slice:

- upload a fixed-camera dance practice video
- detect people with a pretrained Ultralytics YOLO model
- track dancers over time
- return per-frame dancer positions as JSON
- generate a debug overlay video with bounding boxes, anchor points, and IDs
- optionally compare detections against an expected dancer count for diagnostics

See [docs/phase-1-design.md](/C:/Dev/Relationship%20Widget/kpop-formations/docs/phase-1-design.md) for the implementation scope and [docs/api-contract.md](/C:/Dev/Relationship%20Widget/kpop-formations/docs/api-contract.md) for the API contract.
