export type JobStatus = 'queued' | 'awaiting_clicks' | 'processing' | 'completed' | 'failed';

export interface DancerClick {
  name: string;
  x: number;
  y: number;
  /** Frame the click was made on. A click on a frame after the seed frame
   *  is an identity anchor: it pins that dancer to the clicked person there. */
  frame?: number;
}

export interface ClickSeedRequest {
  key_frame: number;
  clicks: DancerClick[];
  stage_corners?: [number, number][];
}

export interface ReferenceFile {
  filename: string;
  size_bytes: number;
}

export interface ReferencesListResponse {
  files: ReferenceFile[];
}

export interface CompareMetrics {
  detected_formation_count: number;
  unique_track_ids: number;
  expected_dancer_count: number;
  id_stability_score: number;
  longest_missing_gap_sec: number;
  full_count_formations: number;
}

export interface CompareToReferenceResponse {
  comparison_video_url: string;
  metrics: CompareMetrics;
}

export interface VideoMetadata {
  filename: string;
  fps: number;
  frame_count: number;
  width: number;
  height: number;
  duration_sec: number;
}

export interface DancerRosterEntry {
  name: string;
  hint: string;
}

export interface UploadResponse {
  job_id: string;
  status: JobStatus;
  video_meta: VideoMetadata;
  expected_dancer_count: number | null;
  roster: DancerRosterEntry[];
}

export interface JobStatusResponse {
  job_id: string;
  status: JobStatus;
  processed_frames: number;
  total_frames: number;
  progress: number;
  error: string | null;
  video_meta: VideoMetadata;
  expected_dancer_count: number | null;
  debug_video_available: boolean;
}

export interface DancerPosition {
  id: number;
  bbox: [number, number, number, number];
  anchor_px: [number, number];
  x: number;
  y: number;
  confidence: number;
}

export interface FramePositions {
  frame: number;
  timestamp_sec: number;
  dancers: DancerPosition[];
}

export interface DetectionSummary {
  expected_dancer_count: number | null;
  unique_track_ids: number;
  max_dancers_in_frame: number;
  average_dancers_per_frame: number;
  frames_with_detections: number;
  frames_below_expected: number | null;
  frames_meeting_expected: number | null;
}

export interface FormationDancerPosition {
  id: number;
  x: number;
  y: number;
}

export interface Formation {
  index: number;
  start_frame: number;
  end_frame: number;
  start_time_sec: number;
  end_time_sec: number;
  duration_sec: number;
  dancers: FormationDancerPosition[];
  shape_name: string | null;
}

export interface PositionsResult {
  job_id: string;
  video: VideoMetadata;
  coordinate_space: {
    image_anchor_px: string;
    normalized_stage_proxy: string;
  };
  summary: DetectionSummary;
  frames: FramePositions[];
  formations: Formation[];
  stage_calibrated?: boolean;
}
