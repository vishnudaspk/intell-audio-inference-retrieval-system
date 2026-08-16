import type {
  AudioAsset,
  AudioSegmentsResponse,
  IngestResponse,
  ProcessingJob,
  RAGResponse,
  SearchResponse,
  SystemHealth,
  TranscriptResponse,
} from '../types';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${API_BASE}${endpoint}`;
    const res = await fetch(url, {
      ...options,
      headers: {
        ...options.headers,
      },
    });

    if (!res.ok) {
      let errorMessage = `HTTP Error ${res.status}: ${res.statusText}`;
      try {
        const errorData = await res.json();
        if (errorData.detail) {
          errorMessage = typeof errorData.detail === 'string' ? errorData.detail : JSON.stringify(errorData.detail);
        }
      } catch {
        // use default error message
      }
      const err = new Error(errorMessage) as Error & { status?: number };
      err.status = res.status;
      throw err;
    }

    return res.json();
  }

  // Health
  async getHealth(): Promise<SystemHealth> {
    return this.request<SystemHealth>('/api/v1/health');
  }

  // Media Ingestion
  async uploadFile(file: File): Promise<IngestResponse> {
    const formData = new FormData();
    formData.append('file', file);
    return this.request<IngestResponse>('/api/v1/ingest/upload', {
      method: 'POST',
      body: formData,
    });
  }

  async ingestYouTube(url: string): Promise<IngestResponse> {
    return this.request<IngestResponse>('/api/v1/ingest/youtube', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
  }

  // Assets
  async listAssets(): Promise<AudioAsset[]> {
    return this.request<AudioAsset[]>('/api/v1/assets');
  }

  async getAsset(audioId: string): Promise<AudioAsset> {
    return this.request<AudioAsset>(`/api/v1/assets/${audioId}`);
  }

  getMediaUrl(audioId: string): string {
    return `${API_BASE}/api/v1/assets/${audioId}/media`;
  }

  async getJobStatus(audioId: string, jobId: string): Promise<ProcessingJob> {
    return this.request<ProcessingJob>(`/api/v1/assets/${audioId}/jobs/${jobId}`);
  }

  async deleteAsset(audioId: string): Promise<{ status: string; audio_id: string }> {
    return this.request<{ status: string; audio_id: string }>(`/api/v1/assets/${audioId}`, {
      method: 'DELETE',
    });
  }

  async reprocessAsset(audioId: string): Promise<IngestResponse> {
    return this.request<IngestResponse>(`/api/v1/assets/${audioId}/process`, {
      method: 'POST',
    });
  }

  // Transcripts & V3 Segments
  async getTranscript(audioId: string): Promise<TranscriptResponse | null> {
    try {
      return await this.request<TranscriptResponse>(`/api/v1/assets/${audioId}/transcript`);
    } catch (err: any) {
      if (err?.status === 404 || err?.message?.includes('404')) {
        return null;
      }
      throw err;
    }
  }

  async getAudioSegments(audioId: string): Promise<AudioSegmentsResponse> {
    return this.request<AudioSegmentsResponse>(`/api/v1/assets/${audioId}/segments`);
  }

  // Search & RAG
  async searchTranscript(audioId: string, query: string): Promise<SearchResponse> {
    return this.request<SearchResponse>('/api/v1/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ audio_id: audioId, query }),
    });
  }

  async askAudio(query: string, audioId?: string): Promise<RAGResponse> {
    return this.request<RAGResponse>('/api/v1/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, audio_id: audioId }),
    });
  }
}

export const api = new ApiService();
