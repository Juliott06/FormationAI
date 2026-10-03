import { useEffect, useMemo, useRef, useState } from 'react';
import type {
  CompareToReferenceResponse,
  DancerClick,
  FramePositions,
  JobStatusResponse,
  PositionsResult,
  ReferenceFile,
  ReferencesListResponse,
  UploadResponse,
  VideoMetadata,
} from './types';

interface StageDot {
  id: number;
  x: number;
  y: number;
}

const API_BASE = '/api/v1';
const STAGE_WIDTH = 800;
const STAGE_HEIGHT = 450;

type Phase =
  | 'idle'
  | 'uploading'
  | 'awaiting_clicks'
  | 'processing'
  | 'completed'
  | 'failed';

function colorForTrack(id: number): string {
  const hue = (id * 137) % 360;
  return `hsl(${hue}, 80%, 55%)`;
}

export default function App() {
  const [phase, setPhase] = useState<Phase>('idle');
  const [error, setError] = useState<string | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [positions, setPositions] = useState<PositionsResult | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [expectedCount, setExpectedCount] = useState<string>('');
  const [frameIndex, setFrameIndex] = useState(0);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [merging, setMerging] = useState(false);
  const [showCleanFormation, setShowCleanFormation] = useState(true);
  const [labels, setLabels] = useState<Record<number, string>>({});
  const [labelDraft, setLabelDraft] = useState<Record<number, string>>({});
  const [applyingLabels, setApplyingLabels] = useState(false);
  const [videoMeta, setVideoMeta] = useState<VideoMetadata | null>(null);
  const [clickFrame, setClickFrame] = useState<number>(0);
  const [clicks, setClicks] = useState<DancerClick[]>([]);
  // The frame ALL clicks are bound to. Clicks and key_frame must agree, so
  // clicking on a different frame restarts the click set on that frame.
  const [seedFrame, setSeedFrame] = useState<number | null>(null);
  const [stageCorners, setStageCorners] = useState<[number, number][]>([]);
  const [submittingClicks, setSubmittingClicks] = useState(false);
  const [references, setReferences] = useState<ReferenceFile[]>([]);
  const [referenceFilename, setReferenceFilename] = useState<string>('reference.mp4');
  const [referenceSide, setReferenceSide] = useState<'left' | 'right'>('right');
  const [comparing, setComparing] = useState(false);
  const [compareResult, setCompareResult] = useState<CompareToReferenceResponse | null>(null);
  const [comparisonRefreshKey, setComparisonRefreshKey] = useState(0);
  const pollingRef = useRef<number | null>(null);

  useEffect(
    () => () => {
      if (pollingRef.current !== null) window.clearInterval(pollingRef.current);
    },
    [],
  );

  useEffect(() => {
    if (phase !== 'completed') return;
    let cancelled = false;
    (async () => {
      try {
        const resp = await fetch(`${API_BASE}/references`);
        if (!resp.ok) return;
        const data = (await resp.json()) as ReferencesListResponse;
        if (cancelled) return;
        setReferences(data.files);
        if (data.files.length > 0) {
          const preferred = data.files.find((f) => f.filename === 'reference.mp4');
          setReferenceFilename(preferred ? preferred.filename : data.files[0].filename);
        }
      } catch {
        // optional feature; ignore
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [phase, jobId]);

  async function handleCompare() {
    if (!jobId || !referenceFilename) return;
    setComparing(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/jobs/${jobId}/compare-to-reference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reference_filename: referenceFilename,
          reference_side: referenceSide,
        }),
      });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(`Compare failed (${resp.status}): ${detail}`);
      }
      const data = (await resp.json()) as CompareToReferenceResponse;
      setCompareResult(data);
      setComparisonRefreshKey((k) => k + 1);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setComparing(false);
    }
  }

  function stopPolling() {
    if (pollingRef.current !== null) {
      window.clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }

  async function loadPositions(id: string) {
    const response = await fetch(`${API_BASE}/jobs/${id}/positions`);
    if (!response.ok) throw new Error(`Positions request failed (${response.status})`);
    const data = (await response.json()) as PositionsResult;
    setPositions(data);
    // Tracking starts at the clicked key frame; open the stage view on the
    // first frame that actually has dancers instead of an empty stage.
    const firstPopulated = data.frames.findIndex((f) => f.dancers.length > 0);
    setFrameIndex(Math.max(firstPopulated, 0));
    try {
      const labelResp = await fetch(`${API_BASE}/jobs/${id}/labels`);
      if (labelResp.ok) {
        const labelData = (await labelResp.json()) as Record<string, string>;
        const numericLabels: Record<number, string> = {};
        for (const [k, v] of Object.entries(labelData)) {
          numericLabels[Number(k)] = v;
        }
        setLabels(numericLabels);
        setLabelDraft(numericLabels);
      }
    } catch {
      // labels are optional; ignore load failures
    }
  }

  function startPolling(id: string) {
    stopPolling();
    pollingRef.current = window.setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/jobs/${id}`);
        if (!response.ok) throw new Error(`Status request failed (${response.status})`);
        const data = (await response.json()) as JobStatusResponse;
        setStatus(data);
        if (data.status === 'completed') {
          stopPolling();
          await loadPositions(id);
          setPhase('completed');
        } else if (data.status === 'failed') {
          stopPolling();
          setError(data.error ?? 'Unknown processing error');
          setPhase('failed');
        } else if (data.status === 'awaiting_clicks') {
          stopPolling();
          setPhase('awaiting_clicks');
        } else if (data.status === 'processing' || data.status === 'queued') {
          setPhase('processing');
        }
      } catch (err) {
        stopPolling();
        setError(err instanceof Error ? err.message : String(err));
        setPhase('failed');
      }
    }, 2000);
  }

  async function handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (!file) {
      setError('Choose a video file.');
      return;
    }

    setPhase('uploading');
    setJobId(null);
    setStatus(null);
    setPositions(null);
    setFrameIndex(0);
    // Clear everything belonging to the previous job — stale selections or
    // comparison results must never carry across uploads (a leftover
    // selectedIds pair could fire a merge with the old job's track ids).
    setSelectedIds([]);
    setCompareResult(null);
    setLabels({});
    setLabelDraft({});
    setClicks([]);
    setClickFrame(0);
    setSeedFrame(null);
    setStageCorners([]);

    const formData = new FormData();
    formData.append('file', file);
    if (expectedCount.trim() !== '') {
      formData.append('expected_dancer_count', expectedCount.trim());
    }

    try {
      const response = await fetch(`${API_BASE}/jobs/upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Upload failed (${response.status}): ${detail}`);
      }
      const data = (await response.json()) as UploadResponse;
      setJobId(data.job_id);
      setVideoMeta(data.video_meta);
      if (data.status === 'awaiting_clicks') {
        setClickFrame(0);
        setClicks([]);
        setSeedFrame(null);
        setStageCorners([]);
        setPhase('awaiting_clicks');
      } else {
        setPhase('processing');
        startPolling(data.job_id);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase('failed');
    }
  }

  function handleAddClick(c: DancerClick) {
    // Clicks are only valid on the frame they were made on. First click pins
    // the seed frame; clicking on a different frame restarts the click set
    // there (coords from one frame + key_frame from another would seed the
    // trackers on the wrong dancers).
    if (seedFrame === null || clicks.length === 0) {
      setSeedFrame(clickFrame);
      setClicks([c]);
      return;
    }
    if (clickFrame !== seedFrame) {
      setSeedFrame(clickFrame);
      setClicks([c]);
      setError(null);
      return;
    }
    setClicks((prev) => [...prev, c]);
  }

  async function handleSubmitClicks() {
    if (!jobId) return;
    const cleaned = clicks
      .map((c) => ({ name: c.name.trim(), x: c.x, y: c.y }))
      .filter((c) => c.name);
    if (cleaned.length === 0) {
      setError('Click on each dancer and give them a name first.');
      return;
    }
    // Duplicate names are intentional — they mean multiple click points on
    // one dancer (e.g. head + torso for an often-occluded back-row dancer).
    setSubmittingClicks(true);
    setError(null);
    try {
      const resp = await fetch(`${API_BASE}/jobs/${jobId}/clicks`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // key_frame is the frame the clicks were MADE on, not the scrubber's
        // current position. stage_corners only sent when all 4 are marked.
        body: JSON.stringify({
          key_frame: seedFrame ?? clickFrame,
          clicks: cleaned,
          stage_corners: stageCorners.length === 4 ? stageCorners : undefined,
        }),
      });
      if (!resp.ok) {
        const detail = await resp.text();
        throw new Error(`Submit clicks failed (${resp.status}): ${detail}`);
      }
      setPhase('processing');
      startPolling(jobId);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setSubmittingClicks(false);
    }
  }

  const busy =
    phase === 'uploading' || phase === 'processing' || phase === 'awaiting_clicks';

  const currentFrame = useMemo<FramePositions | null>(() => {
    if (!positions || positions.frames.length === 0) return null;
    const clamped = Math.min(Math.max(frameIndex, 0), positions.frames.length - 1);
    return positions.frames[clamped];
  }, [positions, frameIndex]);

  const currentFormationIndex = useMemo<number | null>(() => {
    if (!positions || !currentFrame) return null;
    const found = positions.formations.findIndex(
      (f) => currentFrame.frame >= f.start_frame && currentFrame.frame <= f.end_frame,
    );
    return found >= 0 ? found : null;
  }, [positions, currentFrame]);

  const stageDots = useMemo<StageDot[]>(() => {
    if (!currentFrame) return [];
    if (
      showCleanFormation &&
      positions &&
      currentFormationIndex !== null &&
      positions.formations[currentFormationIndex].shape_name
    ) {
      return positions.formations[currentFormationIndex].dancers.map((d) => ({
        id: d.id,
        x: d.x,
        y: d.y,
      }));
    }
    return currentFrame.dancers.map((d) => ({
      id: d.id,
      x: d.x,
      y: d.y,
    }));
  }, [positions, currentFrame, currentFormationIndex, showCleanFormation]);

  function jumpToFrame(targetFrame: number) {
    if (!positions) return;
    const idx = positions.frames.findIndex((f) => f.frame === targetFrame);
    if (idx >= 0) setFrameIndex(idx);
  }

  function toggleDotSelection(id: number) {
    setSelectedIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  }

  async function handleMerge() {
    if (!jobId || selectedIds.length !== 2) return;
    setMerging(true);
    setError(null);
    try {
      const [a, b] = selectedIds;
      const keep_id = Math.min(a, b);
      const remove_id = Math.max(a, b);
      const response = await fetch(`${API_BASE}/jobs/${jobId}/merge-ids`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ keep_id, remove_id }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Merge failed (${response.status}): ${detail}`);
      }
      const data = (await response.json()) as PositionsResult;
      setPositions(data);
      setSelectedIds([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMerging(false);
    }
  }

  async function handleApplyLabels() {
    if (!jobId) return;
    const payload: Record<string, string> = {};
    for (const [k, v] of Object.entries(labelDraft)) {
      const trimmed = v.trim();
      if (trimmed) payload[k] = trimmed;
    }
    setApplyingLabels(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}/jobs/${jobId}/labels`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Label apply failed (${response.status}): ${detail}`);
      }
      const data = (await response.json()) as PositionsResult;
      setPositions(data);
      const numericLabels: Record<number, string> = {};
      for (const [k, v] of Object.entries(payload)) {
        numericLabels[Number(k)] = v;
      }
      setLabels(numericLabels);
      setLabelDraft(numericLabels);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setApplyingLabels(false);
    }
  }

  const visibleTrackIds = useMemo<number[]>(() => {
    if (!positions) return [];
    const ids = new Set<number>();
    for (const f of positions.frames) {
      for (const d of f.dancers) ids.add(d.id);
    }
    return Array.from(ids).sort((a, b) => a - b);
  }, [positions]);

  async function handleSwap() {
    if (!jobId || selectedIds.length !== 2 || !currentFrame) return;
    setMerging(true);
    setError(null);
    try {
      const [id_a, id_b] = selectedIds;
      const response = await fetch(`${API_BASE}/jobs/${jobId}/swap-ids`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id_a, id_b, from_frame: currentFrame.frame }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(`Swap failed (${response.status}): ${detail}`);
      }
      const data = (await response.json()) as PositionsResult;
      setPositions(data);
      setSelectedIds([]);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setMerging(false);
    }
  }

  return (
    <div className="app">
      <header>
        <h1>FormationAI</h1>
        <p>Upload a dance practice video and inspect detected dancer positions.</p>
      </header>

      <section className="upload">
        <form onSubmit={handleSubmit}>
          <label>
            Video file
            <input
              type="file"
              accept="video/*"
              disabled={busy}
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
          </label>
          <label>
            Expected dancers (optional)
            <input
              type="number"
              min={1}
              value={expectedCount}
              onChange={(e) => setExpectedCount(e.target.value)}
              disabled={busy}
            />
          </label>
          <button type="submit" disabled={busy}>
            {phase === 'uploading' ? 'Uploading…' : 'Upload & process'}
          </button>
          <p className="upload-hint">
            After uploading you'll mark the floor corners and click each dancer.
          </p>
        </form>
      </section>

      {error && <div className="error">{error}</div>}

      {phase === 'processing' && status && (
        <section className="progress">
          <p>
            Processing… {status.processed_frames}/{status.total_frames} frames (
            {Math.round(status.progress * 100)}%)
          </p>
        </section>
      )}

      {phase === 'awaiting_clicks' && jobId && videoMeta && (
        <ClickPicker
          jobId={jobId}
          videoMeta={videoMeta}
          frame={clickFrame}
          onFrameChange={setClickFrame}
          clicks={clicks}
          seedFrame={seedFrame}
          onAddClick={handleAddClick}
          onUpdateClick={(idx, partial) =>
            setClicks((prev) =>
              prev.map((c, i) => (i === idx ? { ...c, ...partial } : c)),
            )
          }
          onRemoveClick={(idx) =>
            setClicks((prev) => prev.filter((_, i) => i !== idx))
          }
          stageCorners={stageCorners}
          onAddCorner={(pt) =>
            setStageCorners((prev) => (prev.length < 4 ? [...prev, pt] : prev))
          }
          onClearCorners={() => setStageCorners([])}
          onSubmit={handleSubmitClicks}
          submitting={submittingClicks}
        />
      )}

      {phase === 'completed' && positions && currentFrame && (
        <section className="result">
          <div className="summary">
            <span>
              Frames: {positions.frames.length} · Unique IDs:{' '}
              {positions.summary.unique_track_ids} · Max dancers per frame:{' '}
              {positions.summary.max_dancers_in_frame}
            </span>
            {jobId && status?.debug_video_available && (
              <a
                href={`${API_BASE}/jobs/${jobId}/debug-video`}
                target="_blank"
                rel="noreferrer"
              >
                Debug overlay video
              </a>
            )}
          </div>

          <StageView
            dots={stageDots}
            selectedIds={selectedIds}
            onDotClick={toggleDotSelection}
            labels={labels}
          />

          <div className="merge-controls">
            {selectedIds.length === 0 && (
              <span>Click two dots to merge their IDs (e.g., two tracks that are really the same dancer).</span>
            )}
            {selectedIds.length === 1 && (
              <span>Selected ID {selectedIds[0]}. Click another dot to pair, or click it again to deselect.</span>
            )}
            {selectedIds.length === 2 && (
              <>
                <span>
                  IDs {selectedIds[0]} &amp; {selectedIds[1]} selected.
                  {' '}Merge = same dancer all clip. Swap = crossing happened at this frame.
                </span>
                <button type="button" onClick={handleMerge} disabled={merging}>
                  {merging ? 'Working…' : 'Merge (whole clip)'}
                </button>
                <button type="button" onClick={handleSwap} disabled={merging}>
                  {merging ? 'Working…' : `Swap from frame ${currentFrame.frame}`}
                </button>
                <button type="button" onClick={() => setSelectedIds([])} disabled={merging}>
                  Cancel
                </button>
              </>
            )}
          </div>

          <div className="scrubber">
            <input
              type="range"
              min={0}
              max={positions.frames.length - 1}
              value={frameIndex}
              onChange={(e) => setFrameIndex(Number(e.target.value))}
            />
            <span>
              Frame {currentFrame.frame} @ {currentFrame.timestamp_sec.toFixed(2)}s ·{' '}
              {currentFrame.dancers.length} dancer(s) visible
              {currentFormationIndex !== null
                ? ` · Formation ${currentFormationIndex + 1}${
                    positions.formations[currentFormationIndex].shape_name
                      ? ` (snapped to ${positions.formations[currentFormationIndex].shape_name})`
                      : ' (no shape match)'
                  }`
                : ' · Transition'}
            </span>
            <label className="clean-toggle">
              <input
                type="checkbox"
                checked={showCleanFormation}
                onChange={(e) => setShowCleanFormation(e.target.checked)}
              />
              Snap to detected shape when inside a formation
            </label>
          </div>

          {visibleTrackIds.length > 0 && (
            <div className="labels-panel">
              <div className="labels-header">
                Label dancers — names become anchors used to fix split tracks
              </div>
              <div className="labels-grid">
                {visibleTrackIds.map((tid) => (
                  <label key={tid} className="label-row">
                    <span
                      className="label-id-chip"
                      style={{ background: colorForTrack(tid) }}
                    >
                      {tid}
                    </span>
                    <input
                      type="text"
                      placeholder="dancer name"
                      value={labelDraft[tid] ?? ''}
                      onChange={(e) =>
                        setLabelDraft((prev) => ({ ...prev, [tid]: e.target.value }))
                      }
                      disabled={applyingLabels}
                    />
                  </label>
                ))}
              </div>
              <button
                type="button"
                onClick={handleApplyLabels}
                disabled={applyingLabels}
              >
                {applyingLabels ? 'Applying…' : 'Apply labels'}
              </button>
              {Object.keys(labels).length > 0 && (
                <span className="labels-status">
                  {Object.keys(labels).length} labeled
                </span>
              )}
            </div>
          )}

          {references.length > 0 && (
            <div className="compare-panel">
              <div className="compare-header">
                Compare to reference
                <span className="compare-subheader">
                  Renders our stage view next to your reference video, side by side, with
                  metrics for ID stability and missing-dancer gaps.
                </span>
              </div>
              <div className="compare-controls">
                <label>
                  Reference file
                  <select
                    value={referenceFilename}
                    onChange={(e) => setReferenceFilename(e.target.value)}
                    disabled={comparing}
                  >
                    {references.map((f) => (
                      <option key={f.filename} value={f.filename}>
                        {f.filename} ({(f.size_bytes / 1024 / 1024).toFixed(1)} MB)
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Formation panel side
                  <select
                    value={referenceSide}
                    onChange={(e) => setReferenceSide(e.target.value as 'left' | 'right')}
                    disabled={comparing}
                  >
                    <option value="left">left half</option>
                    <option value="right">right half</option>
                  </select>
                </label>
                <button type="button" onClick={handleCompare} disabled={comparing}>
                  {comparing ? 'Building comparison…' : 'Run comparison'}
                </button>
              </div>
              {compareResult && (
                <>
                  <div className="compare-metrics">
                    <Metric
                      label="Detected formations"
                      value={String(compareResult.metrics.detected_formation_count)}
                    />
                    <Metric
                      label="Unique track IDs"
                      value={`${compareResult.metrics.unique_track_ids} / ${compareResult.metrics.expected_dancer_count} expected`}
                      good={
                        compareResult.metrics.id_stability_score <= 1.2 &&
                        compareResult.metrics.id_stability_score >= 0.9
                      }
                    />
                    <Metric
                      label="ID stability"
                      value={compareResult.metrics.id_stability_score.toFixed(2)}
                      hint="target ≤ 1.2"
                      good={compareResult.metrics.id_stability_score <= 1.2}
                    />
                    <Metric
                      label="Longest missing gap"
                      value={`${compareResult.metrics.longest_missing_gap_sec.toFixed(1)} s`}
                      hint="target ≤ 1.0s"
                      good={compareResult.metrics.longest_missing_gap_sec <= 1.0}
                    />
                    <Metric
                      label="Full-count formations"
                      value={`${compareResult.metrics.full_count_formations} / ${compareResult.metrics.detected_formation_count}`}
                    />
                  </div>
                  <video
                    key={comparisonRefreshKey}
                    className="comparison-video"
                    src={`${compareResult.comparison_video_url}?t=${comparisonRefreshKey}`}
                    controls
                  />
                </>
              )}
            </div>
          )}

          {positions.formations.length > 0 && (
            <div className="formations">
              <div className="formations-header">
                Formations ({positions.formations.length})
              </div>
              <div className="formations-strip">
                {positions.formations.map((f, i) => (
                  <button
                    key={f.index}
                    type="button"
                    className={`formation-card${i === currentFormationIndex ? ' active' : ''}`}
                    onClick={() => jumpToFrame(f.start_frame)}
                  >
                    <strong>F{i + 1}{f.shape_name ? ` · ${f.shape_name}` : ''}</strong>
                    <span>{f.start_time_sec.toFixed(1)}s</span>
                    <span>{f.duration_sec.toFixed(1)}s long</span>
                    <span>{f.dancers.length} dancer(s)</span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

function StageView({
  dots,
  selectedIds,
  onDotClick,
  labels,
}: {
  dots: StageDot[];
  selectedIds: number[];
  onDotClick: (id: number) => void;
  labels: Record<number, string>;
}) {
  return (
    <svg
      className="stage"
      viewBox={`0 0 ${STAGE_WIDTH} ${STAGE_HEIGHT}`}
      role="img"
      aria-label="Top-down stage view"
    >
      <rect x={0} y={0} width={STAGE_WIDTH} height={STAGE_HEIGHT} fill="#101418" />
      <line
        x1={0}
        y1={STAGE_HEIGHT - 1}
        x2={STAGE_WIDTH}
        y2={STAGE_HEIGHT - 1}
        stroke="#444"
        strokeWidth={2}
      />
      <text x={8} y={STAGE_HEIGHT - 8} fill="#666" fontSize={12}>
        front of stage
      </text>
      {dots.map((dot) => {
        const cx = dot.x * STAGE_WIDTH;
        const cy = (1 - dot.y) * STAGE_HEIGHT;
        const selected = selectedIds.includes(dot.id);
        const label = labels[dot.id];
        const displayText = label ?? String(dot.id);
        return (
          <g
            key={dot.id}
            onClick={() => onDotClick(dot.id)}
            style={{ cursor: 'pointer' }}
          >
            {selected && (
              <circle
                cx={cx}
                cy={cy}
                r={22}
                fill="none"
                stroke="#ffd84a"
                strokeWidth={3}
                strokeDasharray="4 3"
              />
            )}
            <circle
              cx={cx}
              cy={cy}
              r={label ? 20 : 16}
              fill={colorForTrack(dot.id)}
              stroke={selected ? '#ffd84a' : '#fff'}
              strokeWidth={selected ? 3 : 2}
            />
            <text
              x={cx}
              y={cy + 4}
              textAnchor="middle"
              fill="#fff"
              fontSize={label ? 10 : 12}
              fontWeight={700}
              style={{ pointerEvents: 'none' }}
            >
              {displayText}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function Metric({
  label,
  value,
  hint,
  good,
}: {
  label: string;
  value: string;
  hint?: string;
  good?: boolean;
}) {
  const cls = `metric${good === undefined ? '' : good ? ' metric-good' : ' metric-bad'}`;
  return (
    <div className={cls}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {hint && <div className="metric-hint">{hint}</div>}
    </div>
  );
}

type PickMode = 'corners' | 'dancers';

// Corners can be clicked in any order; draw them as a non-self-intersecting
// quad (back-left, back-right, front-right, front-left) like the backend does.
function sortedQuad(corners: [number, number][]): [number, number][] {
  if (corners.length !== 4) return corners;
  const byY = [...corners].sort((a, b) => a[1] - b[1]);
  const back = byY.slice(0, 2).sort((a, b) => a[0] - b[0]);
  const front = byY.slice(2).sort((a, b) => a[0] - b[0]);
  return [back[0], back[1], front[1], front[0]];
}

function pointInQuad(px: number, py: number, quad: [number, number][]): boolean {
  // Standard ray-cast: count edge crossings of a horizontal ray from (px,py).
  let inside = false;
  for (let i = 0, j = quad.length - 1; i < quad.length; j = i++) {
    const [xi, yi] = quad[i];
    const [xj, yj] = quad[j];
    if (yi > py !== yj > py && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

// Clicks are on torsos/heads, which sit ABOVE the floor quad in the image, so
// testing the click itself flagged every dancer as "outside the floor". The
// dancer's feet are somewhere straight below the click: warn only if no point
// on that vertical line is inside the quad.
function floorBelowClick(
  x: number,
  y: number,
  quad: [number, number][],
  frameHeight: number,
): boolean {
  for (let yy = y; yy < frameHeight; yy += 4) {
    if (pointInQuad(x, yy, quad)) return true;
  }
  return false;
}

function ClickPicker({
  jobId,
  videoMeta,
  frame,
  onFrameChange,
  clicks,
  seedFrame,
  onAddClick,
  onUpdateClick,
  onRemoveClick,
  stageCorners,
  onAddCorner,
  onClearCorners,
  onSubmit,
  submitting,
}: {
  jobId: string;
  videoMeta: VideoMetadata;
  frame: number;
  onFrameChange: (n: number) => void;
  clicks: DancerClick[];
  seedFrame: number | null;
  onAddClick: (c: DancerClick) => void;
  onUpdateClick: (idx: number, partial: Partial<DancerClick>) => void;
  onRemoveClick: (idx: number) => void;
  stageCorners: [number, number][];
  onAddCorner: (pt: [number, number]) => void;
  onClearCorners: () => void;
  onSubmit: () => void;
  submitting: boolean;
}) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  // Two phases: first mark the floor corners, then click the dancers.
  const [mode, setMode] = useState<PickMode>('corners');
  const [cornersDone, setCornersDone] = useState(false);
  // The slider updates a local value on every tick; the actual frame (which
  // triggers a backend video-decode per change) commits after a short pause,
  // so dragging doesn't fire hundreds of frame-extraction requests.
  const [sliderValue, setSliderValue] = useState(frame);
  useEffect(() => {
    setSliderValue(frame);
  }, [frame]);
  useEffect(() => {
    if (sliderValue === frame) return;
    const t = window.setTimeout(() => onFrameChange(sliderValue), 200);
    return () => window.clearTimeout(t);
  }, [sliderValue, frame, onFrameChange]);
  const lastFrameUrl = useMemo(
    () => `${API_BASE}/jobs/${jobId}/frame-jpeg?n=${frame}`,
    [jobId, frame],
  );

  // Stable per-dancer-name ordering and color
  const nameOrder = useMemo(() => {
    const seen: string[] = [];
    for (const c of clicks) {
      const n = c.name.trim();
      if (n && !seen.includes(n)) seen.push(n);
    }
    return seen;
  }, [clicks]);

  function colorForName(name: string): string {
    const idx = nameOrder.indexOf(name.trim());
    return colorForTrack(idx >= 0 ? idx + 1 : clicks.length + 1);
  }

  function pointIndexForClick(clickIdx: number): number {
    const target = clicks[clickIdx].name.trim();
    let n = 0;
    for (let i = 0; i <= clickIdx; i++) {
      if (clicks[i].name.trim() === target) n++;
    }
    return n;
  }

  function lastNamedClick(): DancerClick | null {
    for (let i = clicks.length - 1; i >= 0; i--) {
      if (clicks[i].name.trim()) return clicks[i];
    }
    return null;
  }

  function eventToImagePx(event: React.MouseEvent<HTMLImageElement>): [number, number] | null {
    const img = imgRef.current;
    if (!img) return null;
    const rect = img.getBoundingClientRect();
    const scaleX = videoMeta.width / rect.width;
    const scaleY = videoMeta.height / rect.height;
    const px = Math.max(0, Math.min(videoMeta.width - 1, Math.round((event.clientX - rect.left) * scaleX)));
    const py = Math.max(0, Math.min(videoMeta.height - 1, Math.round((event.clientY - rect.top) * scaleY)));
    return [px, py];
  }

  function handleImgClick(event: React.MouseEvent<HTMLImageElement>) {
    const pt = eventToImagePx(event);
    if (!pt) return;
    if (mode === 'corners') {
      if (stageCorners.length < 4) onAddCorner(pt);
      return;
    }
    if (event.shiftKey) {
      const target = lastNamedClick();
      if (target) {
        onAddClick({ name: target.name, x: pt[0], y: pt[1] });
        return;
      }
    }
    onAddClick({ name: '', x: pt[0], y: pt[1] });
  }

  // Overlays are positioned in video-pixel space via an SVG viewBox and
  // percentage offsets, so they track the image at any size. (They used to
  // read the image's on-screen rect during render — undefined before the
  // first image load and stale after a resize, so markers drew in the wrong
  // place or not at all.)
  const toPct = (x: number, y: number): [string, string] => [
    `${(x / videoMeta.width) * 100}%`,
    `${(y / videoMeta.height) * 100}%`,
  ];

  return (
    <section className="click-picker">
      {mode === 'corners' ? (
        <div className="click-picker-header">
          <strong>Step 1 — Mark the dance floor ({stageCorners.length}/4 corners)</strong>
          <span>
            Click 4 corners outlining the ground the dancers stand on. The rectangle
            must <strong>include the floor under every dancer</strong> — put the two
            back corners where the <strong>wall meets the floor behind them</strong>{' '}
            (not in front of their feet), and the two front corners at the bottom of
            the frame. Click order doesn't matter — we sort automatically. This turns
            the angled camera view into a true top-down view.{' '}
            <strong>Can't see all four floor corners?</strong> Skip this step — the
            top-down view is then calibrated automatically from the dancers' heights.
          </span>
        </div>
      ) : (
        <div className="click-picker-header">
          <strong>Step 2 — Click each dancer, then name them.</strong>
          <span>
            Tracking runs from the clicked frame <strong>forward</strong> — pick a frame
            near the start where every dancer is visible. <strong>Shift+Click</strong>{' '}
            adds a second point on the most-recently-named dancer (e.g. head + torso) —
            useful when they get occluded. All dancer clicks must be on the same frame;
            clicking a different frame restarts the set there.
          </span>
          {seedFrame !== null && (
            <span className="seed-frame-note">
              Seeding on frame {seedFrame}
              {seedFrame > videoMeta.frame_count * 0.1 && (
                <strong>
                  {' '}
                  — warning: the first {(seedFrame / Math.max(videoMeta.fps, 1)).toFixed(1)}s
                  of the video will have no tracking. Consider an earlier frame.
                </strong>
              )}
            </span>
          )}
          {stageCorners.length === 4 &&
            clicks.some((c) => !floorBelowClick(c.x, c.y, stageCorners, videoMeta.height)) && (
              <span className="seed-frame-note">
                <strong>
                  ⚠ Some dancers are outside your marked floor — the view will be
                  auto-adjusted, but for best results redo the corners so the floor
                  includes the ground under every dancer.
                </strong>
              </span>
            )}
        </div>
      )}

      <div className="click-picker-stage">
        <img
          ref={imgRef}
          src={lastFrameUrl}
          alt={`frame ${frame}`}
          onClick={handleImgClick}
          draggable={false}
          style={{ cursor: 'crosshair' }}
        />
        {/* Floor quad overlay */}
        {stageCorners.length >= 2 && (
          <svg
            className="corner-overlay"
            viewBox={`0 0 ${videoMeta.width} ${videoMeta.height}`}
            preserveAspectRatio="none"
            style={{
              position: 'absolute',
              left: 0,
              top: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
            }}
          >
            <polygon
              points={sortedQuad(stageCorners)
                .map(([x, y]) => `${x},${y}`)
                .join(' ')}
              fill="rgba(74,123,255,0.15)"
              stroke="#4a7bff"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          </svg>
        )}
        {stageCorners.map(([x, y], i) => {
          const [left, top] = toPct(x, y);
          return (
            <div key={`corner-${i}`} className="corner-marker" style={{ left, top }}>
              {i + 1}
            </div>
          );
        })}
        {/* Dancer markers (only in dancer mode) */}
        {mode === 'dancers' &&
          clicks.map((c, i) => {
            const [left, top] = toPct(c.x, c.y);
            const name = c.name.trim();
            const ptIdx = name ? pointIndexForClick(i) : 0;
            const label = name ? (ptIdx > 1 ? `${name}·${ptIdx}` : name) : `#${i + 1}`;
            return (
              <div
                key={i}
                className="click-marker"
                style={{
                  left,
                  top,
                  background: name ? colorForName(name) : colorForTrack(clicks.length + i + 1),
                }}
                title={label}
              >
                {label}
              </div>
            );
          })}
      </div>

      <div className="click-picker-scrubber">
        <input
          type="range"
          min={0}
          max={Math.max(videoMeta.frame_count - 1, 0)}
          value={sliderValue}
          onChange={(e) => setSliderValue(Number(e.target.value))}
        />
        <span>
          Frame {sliderValue} / {videoMeta.frame_count - 1}
        </span>
      </div>

      {mode === 'corners' ? (
        <div className="click-picker-actions">
          <button
            type="button"
            className="click-submit"
            disabled={stageCorners.length !== 4}
            onClick={() => {
              setCornersDone(true);
              setMode('dancers');
            }}
          >
            {stageCorners.length === 4
              ? 'Use these corners → click dancers'
              : `Mark ${4 - stageCorners.length} more corner${4 - stageCorners.length === 1 ? '' : 's'}`}
          </button>
          {stageCorners.length > 0 && (
            <button type="button" className="click-secondary" onClick={onClearCorners}>
              Clear corners
            </button>
          )}
          <button
            type="button"
            className="click-secondary"
            onClick={() => {
              onClearCorners();
              setCornersDone(true);
              setMode('dancers');
            }}
          >
            Skip — corners not visible (auto-calibrate from the dancers)
          </button>
        </div>
      ) : (
        <>
          <div className="click-list">
            {clicks.length === 0 && (
              <div className="click-list-empty">
                No dancers yet. Click on a dancer in the image above to start.
              </div>
            )}
            {clicks.map((c, i) => {
              const name = c.name.trim();
              const ptIdx = name ? pointIndexForClick(i) : 0;
              const chipText = name ? (ptIdx > 1 ? `${name}·${ptIdx}` : name) : `#${i + 1}`;
              return (
                <div key={i} className="click-row">
                  <span
                    className="click-chip"
                    style={{ background: name ? colorForName(name) : colorForTrack(clicks.length + i + 1) }}
                  >
                    {chipText}
                  </span>
                  <input
                    type="text"
                    placeholder="dancer name (e.g. Yeji)"
                    value={c.name}
                    onChange={(e) => onUpdateClick(i, { name: e.target.value })}
                    disabled={submitting}
                  />
                  <span className="click-coords">
                    ({c.x}, {c.y})
                  </span>
                  <button
                    type="button"
                    className="click-remove"
                    disabled={submitting}
                    onClick={() => onRemoveClick(i)}
                    aria-label="Remove dancer"
                  >
                    ×
                  </button>
                </div>
              );
            })}
          </div>

          <div className="click-picker-actions">
            <button
              type="button"
              className="click-submit"
              disabled={submitting || nameOrder.length === 0}
              onClick={onSubmit}
            >
              {submitting
                ? 'Processing…'
                : `Start processing (${nameOrder.length} dancer${nameOrder.length === 1 ? '' : 's'}, ${clicks.length} point${clicks.length === 1 ? '' : 's'})`}
            </button>
            <button
              type="button"
              className="click-secondary"
              disabled={submitting}
              onClick={() => setMode('corners')}
            >
              {cornersDone && stageCorners.length === 4 ? 'Redo floor corners' : 'Mark floor corners'}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
