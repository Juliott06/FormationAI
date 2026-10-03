# FormationAI

FormationAI extracts K-pop dance formations from practice videos. You upload a
fixed-camera practice clip, the app tracks each dancer across the whole video,
and it renders a top-down "stage view" that snaps the dancers into the geometric
formations (V, line, grid, diamond, etc.) they move through — the kind of
formation map a choreographer would draw by hand.

The hard problem is **identity**: K-pop dancers wear near-identical outfits and
constantly cross paths, so a plain person-detector produces dozens of track IDs
for six dancers. FormationAI solves this with **click-to-track** — you click each
dancer once on a clear frame and the tracker follows those specific points for
the entire clip, so identity is fixed by construction.

---

## How it works

There are three tracking backends. Only one is active at a time, chosen by
`backend/.env`.

| Backend | How identity works | Hardware | When to use |
| --- | --- | --- | --- |
| **CoTracker3** (default) | You click each dancer; Meta's CoTracker3 follows the click points. Identity is deterministic. | CPU, in-process (~1.7 fps) | **Recommended.** Runs on any machine, no GPU, no extra server. |
| **YOLO + BoT-SORT** | Automatic person detection + appearance-ReID tracking, then post-processing to merge/repair IDs. No clicking. | CPU (~1.5 fps) | Quick automatic pass; struggles on tight clusters and long crossings. |
| **SAM2** | You click each dancer; a remote SAM2 server propagates masks. | Needs a GPU server (e.g. a DGX) running SAM2 | Highest quality, but requires standing up an inference server. |
| **LocateAnything** | Text prompts ("dancer in white top") ground each dancer per frame. | Needs a GPU server | Experimental / dormant; text descriptions can't distinguish identical uniforms. |

**Why CoTracker is the default:** on a full 3.5-minute clip it went from YOLO's
2.6% of frames with all 6 dancers to **73.9%**, with exactly 6 stable identities
(vs YOLO's 12 fragmented ones). See
[backend/data/references/ITERATION_LOG.md](backend/data/references/ITERATION_LOG.md)
for the full tuning history that led here.

### The click-to-track flow

1. Upload a video.
2. The app pauses and shows a **frame picker**. Scrub to a frame near the
   **start** where every dancer is clearly visible and spread out. (Tracking runs
   *forward* from the frame you click — earlier frames won't be tracked.)
3. Click each dancer once and type their name.
4. For a dancer who often gets hidden behind others (e.g. a back-row member),
   **Shift+Click** a second point on them (their head or torso). If any of a
   dancer's points is visible, they stay on the stage.
5. Click **Start processing**. The tracker runs, then the stage view renders.

After processing you can still hand-correct: merge two IDs that are the same
dancer, swap IDs at a crossing, rename dancers, and compare against a reference
video.

---

## Setup

Requires **Python 3.10+** and **Node 18+**.

### Backend

```powershell
cd "backend"
python -m venv ..\.venv
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The first time CoTracker runs it downloads its model weights (~100 MB) from
PyTorch Hub and caches them under `~/.cache/torch/hub`.

### Frontend

```powershell
cd "frontend"
npm install
```

---

## Running it

FormationAI needs **two long-running processes** in separate terminals. Both must
stay open while you use the app.

**Terminal 1 — backend API** (must run from inside `backend/`):

```powershell
cd "backend"
..\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Wait for the `FormationAI config: ...` log line. Sanity check:
`curl http://localhost:8000/health` → `{"status":"ok"}`.

**Terminal 2 — frontend dev server:**

```powershell
cd "frontend"
npm run dev
```

Open the URL it prints (usually **http://localhost:5173**). The frontend proxies
`/api` to the backend on port 8000.

> **After changing `backend/.env` or backend code:** stop uvicorn (Ctrl+C) and
> start it again. `--reload` re-imports changed Python files, but settings are
> cached and only reload on a fresh process.

---

## Configuration (`backend/.env`)

Edit `backend/.env` and restart uvicorn to apply. Key settings:

**Choose the tracker** (set exactly one click-based backend active):

```
FORMATIONAI_COTRACKER_BACKEND=local        # default: CPU click-to-track
FORMATIONAI_SAM2_BACKEND=disabled          # set to "http" + a URL for a GPU server
FORMATIONAI_LOCATEANYTHING_BACKEND=disabled
```

When SAM2 and CoTracker are both `disabled`, upload runs the automatic YOLO
pipeline (no click step). Precedence when several are enabled: SAM2 → CoTracker →
YOLO.

**Tuning:**

```
FORMATIONAI_COTRACKER_RESIZE_WIDTH=960     # CoTracker runs at 512x384 internally; above ~512 only costs decode time
FORMATIONAI_COTRACKER_AUTO_POINTS=true     # 2 extra tracked points per dancer, median-voted (robust to drift)
FORMATIONAI_YOLO_IMAGE_SIZE=1280           # YOLO detection resolution
FORMATIONAI_YOLO_CONFIDENCE_THRESHOLD=0.20 # lower = more detections (+ more false positives)
FORMATIONAI_TRACKER_CONFIG=botsort_reid.yaml
FORMATIONAI_FORMATION_TEMPLATE_SNAP_THRESHOLD=0.08  # how aggressively to snap to clean shapes
FORMATIONAI_FORMATION_MOVEMENT_THRESHOLD=25         # stage px/s below which dancers are "holding" a formation
```

The file has inline comments explaining every knob and why it's set where it is.

---

## Comparing to a reference

If you have a "ground truth" video (e.g. a side-by-side clip with the correct
formations rendered next to the dance), drop it in `backend/data/references/`.
After a job completes, the **Compare to reference** panel renders your result
next to the reference and reports metrics (ID stability, missing-dancer gaps,
formation counts). The panels are synchronized by timestamp, so differing frame
rates line up correctly.

---

## Testing

```powershell
..\.venv\Scripts\python.exe -m pytest tests/backend/ -q      # from repo root
cd frontend && npx tsc --noEmit                              # frontend type check
```

`backend/scripts/synthetic_check.py` is an end-to-end check that needs no model
weights: it renders a synthetic practice-room clip with known formations, runs
the real CoTracker pipeline with simulated CoTracker/YOLO (including occlusion,
drift and merged-box failures), and reports position error vs ground truth,
floor proportions and detected formations:

```powershell
cd backend
..\.venv\Scripts\python.exe -m scripts.synthetic_check --out ..\synthetic_out
```

---

## Project layout

```
backend/
  app/
    api/routes.py            # HTTP endpoints
    core/config.py           # Settings (reads backend/.env)
    pipeline/
      processor.py           # YOLO pipeline + shared post-processing (finalize_frames)
      cotracker_pipeline.py  # CoTracker click-to-track pipeline
      sam2_pipeline.py       # SAM2 click-to-track pipeline
      named_matcher.py       # LocateAnything → YOLO box matching
      templates.py           # formation shape library + fitting
      stage_renderer.py      # stage-view + side-by-side comparison video rendering
    services/
      cotracker.py           # CoTracker3 client (loads model via torch hub)
      sam2.py, locate_anything.py  # HTTP clients for GPU-server backends
      job_runner.py          # picks the pipeline, runs the job
      storage.py             # per-job files on disk (job.json, positions.json, ...)
  scripts/
    iterate.py, iterate_cotracker.py  # offline eval loop against a reference
    mock_sam2.py, mock_locateanything.py  # fake servers for local dry-runs
  data/                      # job outputs + reference videos (git-ignored, except the log)
frontend/
  src/App.tsx                # entire UI: upload, click-picker, stage view, compare
tests/backend/               # pytest suite
docs/                        # api-contract.md, phase-1-design.md
```

### Setting up a GPU backend (SAM2)

Both SAM2 and LocateAnything are wired up on the app side but need an inference
server you host. The request/response contract each expects is documented at the
top of `backend/app/services/sam2.py` and `.../locate_anything.py`. For a dry run
without a GPU, `backend/scripts/mock_sam2.py` is a stub server that returns fake
tracks so you can exercise the full click flow.
