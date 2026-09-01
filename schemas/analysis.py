"""
Canonical Pydantic models for structured audio intelligence results, analytics, and event contracts.
"""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from schemas.enums import JobStatus, LanguageCode, SourceType, StageStatus


# ---------------------------------------------------------------------------
# Metadata & Audio Characteristics
# ---------------------------------------------------------------------------

class AnalysisMetadata(BaseModel):
    job_id: str
    audio_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    schema_version: str = "3.2.0"


class AudioInfo(BaseModel):
    filename: str
    format: str
    duration_sec: float
    sample_rate: int = 16000
    channels: int = 1
    file_size_bytes: Optional[int] = None
    source_type: SourceType = SourceType.UPLOAD


class AudioQuality(BaseModel):
    rms_energy: float = 0.0
    silence_ratio: float = 0.0
    clipping_detected: bool = False
    dynamic_range_db: float = 0.0
    snr_estimate_db: Optional[float] = None
    peak_amplitude: float = 0.0
    noise_floor_db: float = -60.0
    speech_segments_count: int = 0
    avg_speech_segment_sec: float = 0.0
    longest_silence_sec: float = 0.0
    spectral_centroid_hz: float = 0.0
    zero_crossing_rate: float = 0.0
    audio_quality_score: float = 95.0
    warnings: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# VAD & ASR Schemas
# ---------------------------------------------------------------------------

class VADSegment(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float
    confidence: float = 1.0


class VADResult(BaseModel):
    engine: str = "silero-vad"
    threshold: float = 0.5
    segments: List[VADSegment] = Field(default_factory=list)
    speech_duration_sec: float = 0.0
    silence_duration_sec: float = 0.0
    speech_ratio: float = 0.0
    total_segments: int = 0
    config: Dict[str, Any] = Field(default_factory=dict)


class TranscriptWord(BaseModel):
    id: Optional[str] = None
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    confidence: Optional[float] = None


class TranscriptSegmentResult(BaseModel):
    id: str
    sequence_order: int
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str
    words: List[TranscriptWord] = Field(default_factory=list)
    avg_logprob: Optional[float] = None
    no_speech_prob: Optional[float] = None


class TranscriptionResult(BaseModel):
    engine: str = "faster-whisper"
    model: str = "base.en"
    language: str = "en"
    language_probability: float = 1.0
    full_text: str = ""
    duration_sec: float = 0.0
    processing_sec: float = 0.0
    segments: List[TranscriptSegmentResult] = Field(default_factory=list)
    word_timestamps: List[TranscriptWord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Acoustic Features
# ---------------------------------------------------------------------------

class BandEnergies(BaseModel):
    low_hz: float = 0.0        # 0 - 500 Hz
    low_mid_hz: float = 0.0    # 500 - 2000 Hz
    mid_hz: float = 0.0        # 2000 - 4000 Hz
    high_mid_hz: float = 0.0   # 4000 - 6000 Hz
    high_hz: float = 0.0       # 6000 - 8000 Hz


class AcousticFeatureSet(BaseModel):
    f0_mean: Optional[float] = None
    f0_median: Optional[float] = None
    f0_std: Optional[float] = None
    f0_range: Optional[float] = None
    f0_voiced_fraction: Optional[float] = None
    rms_mean: Optional[float] = None
    rms_std: Optional[float] = None
    spectral_centroid_mean: Optional[float] = None
    spectral_bandwidth_mean: Optional[float] = None
    spectral_rolloff_mean: Optional[float] = None
    spectral_flux_mean: Optional[float] = None
    zcr_mean: Optional[float] = None
    mfcc_means: List[float] = Field(default_factory=list)      # 13 coefficients
    mfcc_deltas: List[float] = Field(default_factory=list)     # 13 delta values
    mfcc_delta2: List[float] = Field(default_factory=list)     # 13 delta-delta values
    band_energies: Optional[BandEnergies] = None
    harmonicity: Optional[float] = None
    jitter: Optional[float] = None
    shimmer: Optional[float] = None
    formant_f1: Optional[float] = None
    formant_f2: Optional[float] = None
    formant_f3: Optional[float] = None


# ---------------------------------------------------------------------------
# Diarization, Clustering & Temporal Modeling
# ---------------------------------------------------------------------------

class DiarizedSegment(BaseModel):
    id: str
    sequence_order: int
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str = ""
    speaker_label: str
    speaker_id: Optional[str] = None
    confidence: float = 1.0
    attribution_decision: Optional[str] = None
    provisional: bool = False
    acoustic_features: Optional[AcousticFeatureSet] = None


class ClusterInfo(BaseModel):
    num_clusters: int = 1
    cluster_sizes: Dict[str, int] = Field(default_factory=dict)
    mean_cosine_similarity: float = 1.0
    silhouette_score: Optional[float] = None


class EmbeddingPoint(BaseModel):
    x: float
    y: float
    speaker_label: str
    confidence: float = 1.0


class EmbeddingViz(BaseModel):
    method: str = "PCA"
    points: List[EmbeddingPoint] = Field(default_factory=list)


class DiarizationResult(BaseModel):
    num_speakers: int = 1
    method: str = "AHC+eigengap+CASA"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    segments: List[DiarizedSegment] = Field(default_factory=list)
    cluster_info: ClusterInfo = Field(default_factory=ClusterInfo)
    embedding_visualization: Optional[EmbeddingViz] = None


class SmoothedSegment(BaseModel):
    start_sec: float
    end_sec: float
    raw_speaker: str
    smoothed_speaker: str


class TemporalModel(BaseModel):
    method: str = "HMM"
    num_states: int = 1
    speaker_sequence: List[str] = Field(default_factory=list)
    transition_matrix: List[List[float]] = Field(default_factory=list)
    smoothed_segments: List[SmoothedSegment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Speaker Profiles & Analytics
# ---------------------------------------------------------------------------

class SpeakerStatistics(BaseModel):
    total_speaking_sec: float = 0.0
    speaking_percentage: float = 0.0
    num_turns: int = 0
    avg_turn_sec: float = 0.0
    longest_turn_sec: float = 0.0
    shortest_turn_sec: float = 0.0
    avg_pause_sec: Optional[float] = None
    response_latency_sec: Optional[float] = None
    speaking_rate_wps: Optional[float] = None


class SpeakerProfile(BaseModel):
    speaker_id: str
    speaker_label: str
    color: str = "#4A90E2"
    statistics: SpeakerStatistics = Field(default_factory=SpeakerStatistics)
    features: AcousticFeatureSet = Field(default_factory=AcousticFeatureSet)
    confidence: float = 1.0
    segment_count: int = 0


# ---------------------------------------------------------------------------
# Conversation Analytics
# ---------------------------------------------------------------------------

class ConversationTurn(BaseModel):
    turn_index: int
    speaker_label: str
    start_sec: float
    end_sec: float
    duration_sec: float
    text: str = ""
    word_count: int = 0
    is_short_response: bool = False


class SpeakerTransition(BaseModel):
    from_speaker: str
    to_speaker: str
    gap_sec: float
    at_sec: float


class ShortResponse(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    speaker_label: str


class SilenceGap(BaseModel):
    start_sec: float
    end_sec: float
    duration_sec: float


class ConversationAnalytics(BaseModel):
    total_duration_sec: float = 0.0
    num_turns: int = 0
    num_speakers: int = 1
    turns: List[ConversationTurn] = Field(default_factory=list)
    transitions: List[SpeakerTransition] = Field(default_factory=list)
    dominant_speaker: str = "Speaker 1"
    conversation_balance: Dict[str, float] = Field(default_factory=dict)
    short_responses: List[ShortResponse] = Field(default_factory=list)
    silence_gaps: List[SilenceGap] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Processing Events & Status
# ---------------------------------------------------------------------------

class ProcessingStage(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_sec: Optional[float] = None
    error_message: Optional[str] = None
    warning_message: Optional[str] = None
    model_info: Dict[str, str] = Field(default_factory=dict)


class HardwareInfo(BaseModel):
    device: str = "cpu"
    cuda_available: bool = False
    gpu_name: Optional[str] = None


class ProcessingInfo(BaseModel):
    stages: List[ProcessingStage] = Field(default_factory=list)
    total_duration_sec: float = 0.0
    audio_duration_sec: float = 0.0
    realtime_factor: float = 0.0
    hardware: HardwareInfo = Field(default_factory=HardwareInfo)


class ProcessingEvent(BaseModel):
    job_id: str
    stage: str
    status: StageStatus = StageStatus.RUNNING
    progress: int = 0
    overall_progress: int = 0
    processed: Optional[int] = None
    total: Optional[int] = None
    elapsed_ms: int = 0
    message: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Canonical Analysis Result
# ---------------------------------------------------------------------------

class AnalysisResult(BaseModel):
    metadata: AnalysisMetadata
    audio: AudioInfo
    audio_quality: AudioQuality = Field(default_factory=AudioQuality)
    vad: VADResult = Field(default_factory=VADResult)
    transcription: TranscriptionResult = Field(default_factory=TranscriptionResult)
    diarization: DiarizationResult = Field(default_factory=DiarizationResult)
    speakers: List[SpeakerProfile] = Field(default_factory=list)
    temporal_model: Optional[TemporalModel] = None
    conversation: ConversationAnalytics = Field(default_factory=ConversationAnalytics)
    processing: ProcessingInfo = Field(default_factory=ProcessingInfo)
