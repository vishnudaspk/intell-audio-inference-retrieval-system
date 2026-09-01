// API types matching FastAPI Pydantic models

export interface SystemHealth {
  status: string;
  app_name: string;
  environment: string;
  asr_engine: string;
  vad_engine: string;
  speaker_embedding_engine: string;
  retrieval_engine: string;
  lm_studio: string;
  chat_model: string;
  embedding_model: string;
  qdrant: string;
  bm25_index: string;
  services: {
    database: string;
    data_directories: string;
    asr: string;
    vad: string;
    lm_studio: string;
    qdrant: string;
  };
}

export interface AudioAsset {
  id: string;
  filename: string;
  file_path: string;
  format: string;
  duration: number;
  source_type: 'upload' | 'youtube';
  created_at: string;
}

export interface ProcessingJob {
  id: string;
  audio_id: string;
  status: 'CREATED' | 'VALIDATING' | 'NORMALIZING' | 'TRANSCRIBING' | 'ALIGNING' | 'PERSISTING' | 'COMPLETED' | 'FAILED';
  error_message?: string | null;
  timings: Record<string, number>;
  created_at: string;
  updated_at: string;
}

export interface IngestResponse {
  asset: AudioAsset;
  job: ProcessingJob;
}

export interface TranscriptWord {
  id: string;
  word: string;
  start: number | null;
  end: number | null;
  confidence: number | null;
}

export interface TranscriptSegment {
  id: string;
  transcript_id?: string | null;
  sequence_order: number;
  start: number | null;
  end: number | null;
  text: string;
  words: TranscriptWord[];
}

export interface TranscriptResponse {
  audio_id: string;
  text: string;
  language: string;
  words: TranscriptWord[];
  segments?: TranscriptSegment[] | null;
}

export interface AcousticFeatures {
  f0_mean?: number | null;
  f0_median?: number | null;
  f0_min?: number | null;
  f0_max?: number | null;
  f0_std?: number | null;
  f0_voiced_fraction?: number | null;
  rms_mean?: number | null;
  rms_std?: number | null;
  rms_max?: number | null;
  spectral_centroid_mean?: number | null;
  spectral_bandwidth_mean?: number | null;
  spectral_rolloff_mean?: number | null;
  spectral_flux_mean?: number | null;
  zero_crossing_rate_mean?: number | null;
  mfcc_means?: number[];
  mfcc_deltas?: number[];
  mfcc_delta2?: number[];
  band_energy_low?: number | null;
  band_energy_low_mid?: number | null;
  band_energy_mid?: number | null;
  band_energy_high_mid?: number | null;
  band_energy_high?: number | null;
}

export interface AudioSegment {
  id: string;
  audio_id: string;
  sequence_order: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  vad_confidence: number;
  text: string;
  language: string;
  whisper_segment_id?: number | null;
  avg_logprob?: number | null;
  no_speech_prob?: number | null;
  speaker_label?: string | null;
  speaker_id?: string | null;
  speaker_embedding?: number[] | null;
  acoustic_features?: AcousticFeatures | null;
  created_at: string;
}

export interface AudioSegmentsResponse {
  audio_id: string;
  total_segments: number;
  segments: AudioSegment[];
}

export interface SearchResult {
  matched_text: string;
  start: number;
  end: number;
  confidence?: number | null;
  word_index?: number | null;
}

export interface SearchResponse {
  audio_id: string;
  query: string;
  results_count: number;
  results: SearchResult[];
}

export interface Citation {
  audio_id: string;
  chunk_id: string;
  start_time: number;
  end_time: number;
  text: string;
}

export interface RAGResponse {
  answer: string;
  confidence: number;
  grounded: boolean;
  citations: Citation[];
  query: string;
  processing_time: number;
  model: string;
}

// ---------------------------------------------------------------------------
// Canonical Platform AnalysisResult Types
// ---------------------------------------------------------------------------

export interface AudioQuality {
  rms_energy: number;
  silence_ratio: number;
  clipping_detected: boolean;
  dynamic_range_db: number;
  snr_estimate_db?: number | null;
  peak_amplitude?: number;
  noise_floor_db?: number;
  speech_segments_count?: number;
  avg_speech_segment_sec?: number;
  longest_silence_sec?: number;
  spectral_centroid_hz?: number;
  zero_crossing_rate?: number;
  audio_quality_score?: number;
  warnings: string[];
}

export interface SpeakerStatistics {
  total_speaking_sec: number;
  speaking_percentage: number;
  num_turns: number;
  avg_turn_sec: number;
  longest_turn_sec: number;
  shortest_turn_sec: number;
  avg_pause_sec?: number | null;
  response_latency_sec?: number | null;
  speaking_rate_wps?: number | null;
}

export interface SpeakerProfile {
  speaker_id: string;
  speaker_label: string;
  color: string;
  statistics: SpeakerStatistics;
  features: AcousticFeatures;
  confidence: number;
  segment_count: number;
}

export interface ConversationTurn {
  turn_index: number;
  speaker_label: string;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  text: string;
  word_count: number;
  is_short_response: boolean;
}

export interface SpeakerTransition {
  from_speaker: string;
  to_speaker: string;
  gap_sec: number;
  at_sec: number;
}

export interface ConversationAnalytics {
  total_duration_sec: number;
  num_turns: number;
  num_speakers: number;
  turns: ConversationTurn[];
  transitions: SpeakerTransition[];
  dominant_speaker: string;
  conversation_balance: Record<string, number>;
  short_responses: { start_sec: number; end_sec: number; text: string; speaker_label: string }[];
  silence_gaps: { start_sec: number; end_sec: number; duration_sec: number }[];
}

export interface ProcessingStageInfo {
  name: string;
  status: string;
  start_time?: string | null;
  end_time?: string | null;
  duration_sec?: number | null;
  error_message?: string | null;
  model_info?: Record<string, string>;
}

export interface ProcessingInfo {
  stages: ProcessingStageInfo[];
  total_duration_sec: number;
  audio_duration_sec: number;
  realtime_factor: number;
  hardware: {
    device: string;
    cuda_available: boolean;
    gpu_name?: string | null;
  };
}

export interface DiarizedSegmentItem {
  id: string;
  sequence_order: number;
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  text: string;
  speaker_label: string;
  confidence: number;
  attribution_decision?: string | null;
  provisional?: boolean;
  acoustic_features?: AcousticFeatures | null;
}

export interface DiarizationResultData {
  num_speakers: number;
  method: string;
  parameters: Record<string, any>;
  segments: DiarizedSegmentItem[];
  cluster_info: {
    num_clusters: number;
    cluster_sizes: Record<string, number>;
    mean_cosine_similarity: number;
    silhouette_score?: number | null;
  };
}

export interface AnalysisResult {
  metadata: {
    job_id: string;
    audio_id: string;
    created_at: string;
    schema_version: string;
  };
  audio: {
    filename: string;
    format: string;
    duration_sec: number;
    sample_rate: number;
    channels: number;
    source_type: 'upload' | 'youtube';
  };
  audio_quality: AudioQuality;
  vad: {
    engine: string;
    threshold: number;
    segments: { start_sec: number; end_sec: number; duration_sec: number; confidence: number }[];
    speech_duration_sec: number;
    silence_duration_sec: number;
    speech_ratio: number;
    total_segments: number;
  };
  transcription: {
    engine: string;
    model: string;
    language: string;
    full_text: string;
    duration_sec: number;
    processing_sec: number;
    segments: TranscriptSegment[];
    word_timestamps: TranscriptWord[];
  };
  diarization: DiarizationResultData;
  speakers: SpeakerProfile[];
  conversation: ConversationAnalytics;
  processing: ProcessingInfo;
}

export interface ProcessingEvent {
  job_id: string;
  stage: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  progress: number;
  overall_progress: number;
  processed?: number | null;
  total?: number | null;
  elapsed_ms: number;
  message?: string | null;
  error?: string | null;
  timestamp: string;
}
