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
  zero_crossing_rate_mean?: number | null;
  mfcc_means?: number[];
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
