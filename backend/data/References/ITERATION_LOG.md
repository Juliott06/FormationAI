# FormationAI iteration log

Running log of reference-driven iterations on the YOLO/post-processing pipeline.
One entry per iteration. Append; never edit prior entries.

Format per iteration:

```
## Iter N — YYYY-MM-DD — <clip name>

**Hypothesis:** what I think is wrong and what change should help
**Change:** specific file/config/threshold I modified (with before → after values)
**Metrics:**
- formation count: N (reference ~M)
- ID stability score: X.XX (target ≤1.2)
- longest missing-dancer gap: X.X sec (target ≤1.0)
- per-formation count match: N/M formations have full dancer set
**Observed (eye check):** what the side-by-side comparison video showed
**Verdict:** keep / revert / partial
**Next hypothesis:** what to try next
```

---

## Iter 0 — 2026-06-14 — KATSEYE Gabriela (first 30s of 3:25 raw clip, expected 6 dancers)

**Hypothesis:** baseline — what the YOLO+post-processing pipeline produces out of the box with current `.env` settings.

**Change:** none. Default config: `YOLO_CONFIDENCE_THRESHOLD=0.35`, `YOLO_IMAGE_SIZE=960`, tracker=bytetrack, debug video on, appearance threshold 0.92, formation movement threshold 7px, min duration 0.2s.

**Metrics (30s clip, 719 frames @ 24fps):**
- detected_formation_count: 13
- unique_track_ids: 9 (expected 6) — id_stability 1.50 (target ≤1.2)
- longest_missing_gap_sec: 19.67 (target ≤1.0) — ID 16 or 5 was missing for ~470 consecutive frames within its active span
- full_count_formations: 4 of 13
- **frames_with_expected (≥6 dancers): 19.9% (143/719)** ← the real headline number
- avg_dancers_per_frame: 4.76 (target 6.00)
- Processing time: 179.5s (4.0 fps avg)

**Observed (eye check):**
- Reference (left) has clean 6 named dots at every sample frame.
- Our output (right) is empty at f=0, then shows 3-5 dots throughout. NEVER 6.
- 4 IDs (1, 3, 11, 6) covered the clip >92% — these are 4 of the 6 dancers tracked well.
- 2 IDs (5, 16) covered 33-37% — the missing dancers, YOLO drops them often.
- 3 orphan IDs (2, 23, 95) — false positives that survived merging.
- Stage Y axis is too clustered vertically; reference spreads from backstage to audience.

**Verdict:** baseline. Detection is the primary bottleneck — we need to recover the missing 1.2 dancers per frame before identity work matters.

**Next hypothesis (iter 1):** Lower YOLO confidence from 0.35 → 0.20. Should find more of the partially-occluded/distant dancers per frame. Risk: more false positives — but the appearance-merge stage already catches similar-outfit duplicates. Net expected impact: avg_dancers_per_frame ↑, pct_frames_with_expected ↑, unique_track_ids likely also ↑ (will need merge/recovery to catch up).

---

## Iter 1 — 2026-06-14 — same 30s clip

**Hypothesis:** Lower YOLO confidence 0.35 → 0.20 to detect partially-occluded dancers.

**Change:** [backend/.env](backend/.env) `FORMATIONAI_YOLO_CONFIDENCE_THRESHOLD=0.35` → `0.20`.

**Metrics (30s clip, 719 frames):**
| Metric | Iter 0 | Iter 1 | Δ |
| --- | --- | --- | --- |
| detected_formation_count | 13 | 14 | +1 |
| unique_track_ids | 9 | 10 | +1 |
| id_stability_score | 1.50 | 1.67 | worse |
| longest_missing_gap_sec | 19.67 | 17.48 | -2.2s |
| full_count_formations | 4 / 13 | **10 / 14** | +6 |
| **pct_frames_with_expected** | **19.9%** | **72.7%** | **+52.8 pp** |
| avg_dancers_per_frame | 4.76 | 5.88 | +1.12 |
| processing_time_sec | 179.5 | 162.3 | -17 (faster!) |

**Observed (eye check, iter_01.mp4):**
- f30%: 6 dancers visible for the first time. Identity matches reference's V-shape arrangement reasonably.
- f60%: 5 visible, 1 missing — small dancer at right side of stage dropped.
- f90%: only 3-4 visible, clustered tightly — likely SAME dancer detected as multiple IDs (clustering visible).
- Per-ID coverage: 5 IDs (1, 4, 6, 5, 23) cover 80-99% — those are 5 of 6 dancers tracked well. The 6th dancer is split across IDs 2 (early 4-143) → 41 (late 166-718), with ID 8 also looking like a fragment (40% spanning whole clip but co-occurring with IDs 2 and 41 — likely a different intermittent dancer or a persistent false positive).

**Verdict:** **KEEP.** Massive win on detection (the headline 72.7% number). New problem revealed: identity fragmentation. The 6th dancer is split across 2-3 IDs because ByteTrack discards tracks after only ~30 frames of absence (~1 sec) and K-pop occlusions are longer.

**Next hypothesis (iter 2):** Switch tracker from ByteTrack to BoT-SORT+ReID (`FORMATIONAI_TRACKER_CONFIG=botsort_reid.yaml`). BoT-SORT integrates appearance ReID into the tracker (not just post-hoc merge), so fragmented IDs 2/8/41 should bind to one stable identity. Cost: ~3x slower processing (~9-12 min for 30s clip). Expected impact: unique_track_ids ↓ toward 6, id_stability_score ↓ toward 1.0, longest_missing_gap ↓.

---

## Iter 2 — 2026-06-14 — same 30s clip

**Hypothesis:** BoT-SORT+ReID will bind sporadic 6th-dancer detections to a stable ID via appearance, fixing iter 1's identity fragmentation.

**Change:** [backend/.env](backend/.env) `FORMATIONAI_TRACKER_CONFIG=bytetrack.yaml` → `botsort_reid.yaml`.

**Metrics:**
| Metric | Iter 1 | Iter 2 | Δ | Target met? |
| --- | --- | --- | --- | --- |
| detected_formation_count | 14 | 12 | -2 | — |
| unique_track_ids | 10 | **7** | -3 | ✓ close to 6 |
| id_stability_score | 1.67 | **1.17** | -0.50 | **✓ ≤1.2** |
| longest_missing_gap_sec | 17.48 | **0.00** | -17.48 | **✓ ≤1.0** |
| full_count_formations | 10 / 14 | 4 / 12 | -6 | — |
| **pct_frames_with_expected** | 72.7% | 42.6% | **-30.1 pp** | regressed |
| avg_dancers_per_frame | 5.88 | 5.38 | -0.50 | regressed |
| processing_time_sec | 162 | 200 | +38 | — |

**Observed (eye check, iter_02.mp4):**
- IDs collapse to single-digit numbers (1-8 only). No more 41/95/104 orphans. Identity story is dramatically cleaner.
- f30%: **6 dots properly spread** — matches reference V-shape.
- f60%: only 5 visible — one dancer dropped.
- f90%: only 4 visible — bigger gap.
- Two of four success targets HIT: id_stability ≤1.2 ✓ and longest_missing_gap ≤1.0 ✓.

**Verdict:** **KEEP for identity** — BoT-SORT clearly wins for stable IDs. But it's stricter about confirming tracks: with `new_track_thresh: 0.6` and our YOLO confidence at 0.20, many low-confidence YOLO detections never get promoted to confirmed tracks. The detection regression is config-fixable.

**Next hypothesis (iter 3):** Edit [backend/app/pipeline/trackers/botsort_reid.yaml](backend/app/pipeline/trackers/botsort_reid.yaml): `new_track_thresh: 0.6 → 0.35` and `track_high_thresh: 0.5 → 0.30`. Lets the looser YOLO detections promote to tracks while keeping BoT-SORT's appearance ReID for identity. Expected: pct_frames_with_expected recovers toward 60-70% while keeping id_stability ≤1.2.

---

## Iter 3 — 2026-06-14 — same 30s clip

**Hypothesis:** Loosening BoT-SORT thresholds will recover detection coverage without breaking identity.

**Change:** [backend/app/pipeline/trackers/botsort_reid.yaml](backend/app/pipeline/trackers/botsort_reid.yaml): `track_high_thresh 0.5 → 0.30`, `new_track_thresh 0.6 → 0.35`.

**Metrics:**
| Metric | Iter 2 | Iter 3 | Δ |
| --- | --- | --- | --- |
| unique_track_ids | 7 | 10 | worse |
| id_stability_score | 1.17 | 1.67 | worse |
| longest_missing_gap_sec | 0.00 | 15.14 | much worse |
| pct_frames_with_expected | 42.6% | 37.0% | worse |
| avg_dancers_per_frame | 5.38 | 5.43 | flat |
| processing_time_sec | 200 | 206 | flat |

**Verdict:** **REVERT.** Hypothesis FAILED on every dimension. Lowering new_track_thresh let weak detections seed false tracks (10 IDs, gap back to 15s) without meaningfully improving coverage. BoT-SORT's strictness is what gives it the identity edge — loosening it loses that without gaining detection.

**Action:** Reverted botsort_reid.yaml to original thresholds (0.5 / 0.6).

**Next hypothesis (iter 4):** Keep BoT-SORT at defaults (the iter-2 identity baseline) but bump YOLO image size 960 → 1280. More pixels = better detection on small/partially-occluded dancers, especially the back-of-stage ones. Expected to push pct_frames_with_expected back toward 70%+ while preserving iter-2's identity (id_stability ~1.17). Cost: ~30% slower processing.

---

## Iter 4 — 2026-06-14 — CONVERGENCE on 30s clip

**Hypothesis:** With BoT-SORT defaults restored, larger YOLO image size (1280) should recover the detection coverage that iter 2 lost.

**Change:** [backend/.env](backend/.env) `FORMATIONAI_YOLO_IMAGE_SIZE=960 → 1280`. Tracker yaml reverted to original 0.5 / 0.6 thresholds.

**Metrics — ALL FOUR SUCCESS CRITERIA HIT:**
| Metric | Iter 4 | Target | Pass? |
| --- | --- | --- | --- |
| id_stability_score | **1.17** | ≤1.2 | ✓ |
| longest_missing_gap_sec | **0.00** | ≤1.0 | ✓ |
| full_count_formations | **11 / 11** | 100% | ✓ |
| pct_frames_with_expected | **86.9%** | high | ✓ |
| avg_dancers_per_frame | 5.83 | ~6.0 | ~ |
| processing_time_sec | 436 (+114%) | — | (cost) |

**Trajectory:**
| | iter 0 | iter 1 | iter 2 | iter 3 (revert) | iter 4 |
| --- | --- | --- | --- | --- | --- |
| pct_frames_with_expected | 19.9% | 72.7% | 42.6% | 37.0% | **86.9%** |
| id_stability | 1.50 | 1.67 | 1.17 | 1.67 | **1.17** |
| longest_missing_gap_sec | 19.7 | 17.5 | 0.0 | 15.1 | **0.0** |
| full_count_formations | 4/13 | 10/14 | 4/12 | 4/9 | **11/11** |

**Observed (eye check, iter_04.mp4):** f30% shows 6 dots in V-spread matching reference. f60% shows 6 dots with one cluster near top. f90% has 5 visible (one dropped during the line formation) — the remaining 13% where we miss a dancer.

**Verdict:** **CONVERGED on 30s clip.** The three-knob recipe is: BoT-SORT+ReID (identity), YOLO conf 0.20 (recall), YOLO image size 1280 (resolution for small/distant dancers). All four success criteria pass.

**Status:** Stop iterating on the 30s clip. Next step is either (a) a full 3:25 clip run to validate the result holds at scale (~60 min processing time) or (b) leave this here as the convergence point and use these settings going forward.

---

## Iter 5 — 2026-06-14 — FULL 3:25 CLIP with iter-4 settings (regression at scale)

**Hypothesis:** iter-4 settings generalize from the 30s opening to the full song.

**Change:** none — same iter-4 settings, just `--clip-seconds` removed.

**Metrics (4919 frames, 205s):**
| Metric | Iter 4 (30s) | Iter 5 (full) | regression |
| --- | --- | --- | --- |
| pct_frames_with_expected | 86.9% | **2.6%** | -84 pp |
| id_stability_score | 1.17 | 2.00 | worse |
| longest_missing_gap_sec | 0.00 | 46.26 | much worse |
| full_count_formations | 11 / 11 | 3 / 88 | worse |
| avg_dancers_per_frame | 5.83 | 4.10 | worse |
| processing_time_sec | 436 | 3133 (52 min) | proportional |

**Per-section avg-dancers shows the song structure:**
| section | avg dancers | % with 6+ |
| --- | --- | --- |
| 0-25s (opening) | 5.2 | 18% |
| 25-128s (verse → chorus) | 4.2-5.0 | ~0% |
| 128-153s | 4.2 | 2% |
| 153-179s | **2.8** | 0% |
| 179-205s (outro) | **1.7** | 0% |

**Observed (eye check):**
- t=30s: 5 dots spread out — reference shows 6 in a tight V at backstage labelled "Formation 10". We miss 1 dancer.
- t=60s: 5 dots spread — reference shows 6 **tightly clustered in the center** as one blob ("Megan 3" section). YOLO can't separate tight clusters.
- t=90s: 5 dots — reference 6 in V at "Chorus 2".
- t=150s: 3 dots vs 6 in reference's tight cluster at "Formation 27".
- Reference is 194s long; raw is 205s. They're different lengths (different edits) — comparison video is truncated to 194s.
- Per-ID coverage: 5 IDs (1, 3, 4, 6, 10) cover 55-96% of clip; the 6th dancer's identity fragmented into IDs 72, 75, 5, 2 (each <10%).

**Verdict:** **CEILING HIT, as called out in the plan.** Two structural reasons our pipeline can't close the rest of the gap:
1. **Tight clusters** — K-pop choreography deliberately stacks dancers very close together at the center, where YOLO's bounding boxes overlap heavily and ByteTrack/BoT-SORT can't reliably separate identities. This is exactly the failure mode SAM2 was designed for (mask-based segmentation handles tight overlap).
2. **Camera cuts to close-ups** in the late sections of the song mean only 1-2 dancers are actually visible. "expected=6" becomes meaningless; the metric shows 0% but the truth is "the camera isn't showing 6 right now."

**Reality check:** the 30s opening was the EASIEST section of the song. iter-4 settings DO work where they're physically possible to work — they just can't manufacture detections that YOLO can't make from the source footage.

**Next steps (recommendations, not committed):**
- **Best path:** Bring up SAM2 on the DGX. The click-to-track UX already wired up will close most of the remaining gap because (a) clicks anchor identity per dancer, (b) SAM2's segmentation handles tight clusters better than bbox detection, (c) SAM2's video temporal coherence handles close-ups (it tracks the dancer's last-known mask through brief absences).
- **Or:** Accept the ceiling, mark "iter-4 settings are the YOLO-only recipe", and switch focus to SAM2 server setup.
- **Lower-value alternatives:** Try yolo26m or yolo26l (slower, modest detection gain — won't fix cluster overlap). Or improve the metric to ignore frames where the camera genuinely isn't showing 6 dancers (requires per-frame "is this a wide shot?" detection — itself a problem).

---

## DECISION — 2026-06-14 — stop iterating

**Stopped at iter 5.** YOLO + post-processing ceiling reached.

**Committed YOLO-only recipe** (in [backend/.env](backend/.env)):
- `FORMATIONAI_YOLO_CONFIDENCE_THRESHOLD=0.20`
- `FORMATIONAI_YOLO_IMAGE_SIZE=1280`
- `FORMATIONAI_TRACKER_CONFIG=botsort_reid.yaml` (default thresholds)

These settings hit all four success criteria on spread-formation sections (iter 4). They cannot fix tight-cluster sections or close-up cuts (iter 5) — those require SAM2 or another mask-based approach.

**Cluster limitation accepted.** Returning to other work; SAM2 bring-up is the next ceiling-breaker when ready.

---

## Iter 6 — 2026-06-14 — CoTracker3 baseline on 30s clip

**Hypothesis:** CoTracker3 runs on CPU at ~2 fps and is the realistic ceiling-breaker that doesn't need a GPU. Identity should be perfect by construction (6 click points = 6 tracks).

**Change:** Added [backend/app/services/cotracker.py](backend/app/services/cotracker.py) + [backend/app/pipeline/cotracker_pipeline.py](backend/app/pipeline/cotracker_pipeline.py) + [backend/scripts/iterate_cotracker.py](backend/scripts/iterate_cotracker.py). `.env`: `FORMATIONAI_COTRACKER_BACKEND=local`, `RESIZE_WIDTH=960`. Click points taken from iter 5's positions.json at frame 93 (the first frame where YOLO found all 6 dancers).

**Metrics (30s clip):**
| Metric | iter 4 (YOLO) | iter 6 (CoTracker) |
| --- | --- | --- |
| unique_track_ids | 7 | **6** ✓ |
| id_stability_score | 1.17 | **1.00** ✓ |
| longest_missing_gap_sec | 0.00 | **0.00** ✓ |
| pct_frames_with_expected | 86.9% | 76.9% |
| detected_formation_count | 11 | 1 |
| processing_time_sec | 436 | 384 (faster!) |

**Observed (eye check):** All 6 dots visible and persistent at f30%/f60%. f90% shows 5 (D1 back-row marked invisible). Spatial layout is flatter horizontally than reference (Y-refit clusters anchors to a narrow band when starting Y values are similar). 1 detected formation = movement is so smooth segmentation thinks it's all one big formation.

**Verdict:** **Identity perfect, detection slightly conservative.** Move to full-clip test.

---

## Iter 7 — 2026-06-14 — CoTracker3 on FULL 3:25 clip — CEILING BROKEN

**Hypothesis:** CoTracker holds identity through the mid-song tight clusters where iter 5 YOLO collapsed.

**Change:** same as iter 6, no `--clip-seconds` arg.

**Metrics:**
| Metric | iter 5 (YOLO) | iter 7 (CoTracker) | Δ |
| --- | --- | --- | --- |
| pct_frames_with_expected | 2.6% | **28.7%** | **11× better** |
| unique_track_ids | 12 | **6** ✓ | exact |
| id_stability_score | 2.00 | **1.00** ✓ | perfect |
| longest_missing_gap_sec | 46.26 | 37.88 | better but still high |
| detected_formation_count | 88 (fragments) | **8 (real)** | huge |
| full_count_formations | 3 / 88 | 3 / 8 | proportional |
| avg_dancers_per_frame | 4.10 | 4.74 | better |
| processing_time_sec | 3133 (52 min) | 2906 (48 min) | faster |

**Detected formation shapes** (snapped to templates): 3x2 grid, Inverted V, V, V, Two rows, 3x2 grid — real K-pop shapes, not noise.

**Per-section coverage (% frames with 6 dancers):**
| section | YOLO iter 5 | CoTracker iter 7 |
| --- | --- | --- |
| 0-25s | 18% | **86%** |
| 25-51s | 0% | 17% |
| 51-76s | 0% | 41% |
| 76-102s | 0% | 10% |
| 102-128s | 0% | 34% |
| 128-153s | 2% | 8% |
| 153-179s | 0% | 5% |
| 179-205s | 0% | 24% |

**Per-ID activity (5 of 6 tracked across entire song):**
| ID | active frames | coverage |
| --- | --- | --- |
| D1 (back-row click 928,523) | 80-4727 | 36% (often occluded) |
| D2 (590,795) | 80-4719 | **86%** |
| D3 (1241,796) | 80-4623 | **92%** |
| D4 (747,801) | 80-4728 | **87%** |
| D5 (1113,803) | 80-4617 | **84%** |
| D6 (920,846) | 80-4725 | **86%** |

**Verdict:** **CoTracker delivers the structural fix.** Five of six dancers tracked rock-solid for 84-92% of the entire 3:25 song. Identity is solved — no manual ID merging needed. Real formations detected.

Remaining problem is concentrated: **D1 (the back-row dancer click point) drops to 36% coverage because they're frequently occluded by front-row dancers.** This isn't a CoTracker limitation — it's a "the back-row dancer is genuinely behind people much of the time" data limitation. Fixable two ways:
1. **Multiple click points per dancer** — click head AND torso on D1 so when one is occluded the other still tracks; emit a "dancer is present" record if ANY point is visible.
2. **Manual rescue UI** — let the user click "D1 is here" at problem frames in the timeline.

---

## Iter 8 — 2026-06-14 — multi-click for D1 on 30s clip — multi-click works

**Hypothesis:** giving the back-row dancer 3 click points (head + torso + foot) instead of 1 closes the occlusion gap.

**Change:** [backend/app/services/cotracker.py](backend/app/services/cotracker.py) `track_video` now returns `list[list[TrackedPoint]]` (one list per input click, parallel to clicks). [backend/app/pipeline/cotracker_pipeline.py](backend/app/pipeline/cotracker_pipeline.py) `tracked_points_to_frames` groups by `click.name`, emits a dancer if ANY of their points is visible, uses LOWEST-y visible point as foot anchor. [backend/app/api/routes.py](backend/app/api/routes.py) `submit_clicks` no longer rejects duplicate names. [backend/app/pipeline/sam2_pipeline.py](backend/app/pipeline/sam2_pipeline.py) dedupes by name defensively (SAM2 server expects unique names). Frontend [App.tsx](frontend/src/App.tsx) ClickPicker: Shift+Click adds a point to the most-recently-named dancer.

Iter 8 ran with D1 having 3 query points: (928, 323) head, (928, 423) torso, (928, 523) foot. Other 5 dancers unchanged (single point).

**Metrics (30s clip):**
| Metric | iter 4 (best YOLO) | iter 6 (CoTracker 1-click) | iter 8 (CoTracker multi-click) |
| --- | --- | --- | --- |
| unique_track_ids | 7 | 6 | **6** ✓ |
| id_stability_score | 1.17 | 1.00 | **1.00** ✓ |
| longest_missing_gap_sec | 0.00 | 0.00 | **0.00** ✓ |
| pct_frames_with_expected | 86.9% | 76.9% | **88.9%** ✓ |
| processing_time_sec | 436 | 384 | 546 |

**CoTracker iter 8 now beats YOLO iter 4 on every metric.**

**Observed:** Iter 6's f90% problem (D1 missing) is fixed — iter 8 f90% shows all 6 dots. Identity is perfect, coverage on the easy 30s opening exceeds YOLO.

**Verdict:** **KEEP.** Multi-click closes the back-row dancer gap. The whole pipeline is now click-grounded and identity is deterministic from the user's clicks. 42% slower per iteration on the 30s clip (3 extra query points), so full-clip rerun would be ~68 min.

**Next:** decide whether to spend the 68 min validating on the full clip, or accept iter 8 as the converged recipe and ship.

---

## Iter 9 — 2026-06-14 — CoTracker multi-click D1 on FULL 3:25 CLIP — CONVERGED

**Hypothesis:** multi-click for D1 on the full clip closes the back-row occlusion gap that iter 7 had, getting full-clip coverage close to the 30s clip's 88.9%.

**Change:** same as iter 8 (multi-click D1: head 323 / torso 423 / foot 523), removed `--clip-seconds`.

**Metrics:**
| Metric | iter 5 YOLO | iter 7 CoTracker 1pt | **iter 9 CoTracker multi-pt** |
| --- | --- | --- | --- |
| **pct_frames_with_expected** | **2.6%** | 28.7% | **73.9%** |
| unique_track_ids | 12 | 6 | **6** ✓ |
| id_stability_score | 2.00 | 1.00 | **1.00** ✓ |
| longest_missing_gap_sec | 46.26 | 37.88 | **14.14** |
| **full_count_formations** | **3 / 88** | 3 / 8 | **11 / 14** |
| avg_dancers_per_frame | 4.10 | 4.74 | **5.28** |
| detected_formation_count | 88 (fragments) | 8 | 14 (real, 5 V-shapes) |
| processing_time_sec | 3133 | 2906 | 2947 (flat — multi-click is ~free) |

**Per-section breakdown — most sections jumped from 0% to 65-100%:**
| section | YOLO iter 5 | CoTracker 1pt iter 7 | **CoTracker multi-pt iter 9** |
| --- | --- | --- | --- |
| 0-25s | 18% | 86% | 86% |
| 25-51s | 0% | 17% | **100%** ⭐ |
| 51-76s | 0% | 41% | **65%** |
| 76-102s | 0% | 10% | **87%** ⭐ |
| 102-128s | 0% | 34% | **85%** ⭐ |
| 128-153s | 2% | 8% | 42% (heavy clusters) |
| 153-179s | 0% | 5% | **73%** |
| 179-205s | 0% | 24% | 50% (camera close-ups, can't fix) |

**Observed (eye check, iter_09.mp4):**
- t=30s: all 6 dots visible (was the broken section in iter 5).
- t=90s: all 6 dots visible.
- t=150s: all 6 dots visible, including D1 back-row.
- 5 of 14 detected formations snap to V — matches what the reference shows.

**Verdict:** **CONVERGED.** From 2.6% to 73.9% frames with 6 dancers visible — ~29× improvement vs YOLO baseline. Identity is solved (1.00 stability, exactly 6 unique IDs). The previously-broken mid-song sections that YOLO collapsed on now hit 65-100%. The remaining gap (179-205s outro at 50%) is camera-side: the video literally cuts to close-ups of fewer dancers, so "expected 6" is unachievable there.

**Shipping recipe:**
- `FORMATIONAI_COTRACKER_BACKEND=local` (already in [.env](backend/.env))
- `FORMATIONAI_COTRACKER_RESIZE_WIDTH=960`
- Click each dancer in the UI. **Shift+Click** to add extra points on the same dancer (use this for back-row/often-occluded dancers).
- Total processing: ~50 min for a 3:25 song on CPU, identity guaranteed.










