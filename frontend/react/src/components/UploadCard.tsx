import React, { useState } from 'react';
import { Upload, Link2, FileAudio, FileVideo, AlertCircle, Loader2, ArrowRight } from 'lucide-react';
import type { IngestResponse } from '../types';
import { api } from '../services/api';

interface UploadCardProps {
  onIngestSuccess: (data: IngestResponse) => void;
}

export const UploadCard: React.FC<UploadCardProps> = ({ onIngestSuccess }) => {
  const [activeTab, setActiveTab] = useState<'upload' | 'youtube'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) { setFile(e.target.files[0]); setError(null); }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files?.[0]) { setFile(e.dataTransfer.files[0]); setError(null); }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);
    try {
      if (activeTab === 'upload') {
        if (!file) throw new Error('Select an audio or video file.');
        const res = await api.uploadFile(file);
        onIngestSuccess(res);
        setFile(null);
      } else {
        if (!youtubeUrl.trim()) throw new Error('Enter a valid YouTube URL.');
        const res = await api.ingestYouTube(youtubeUrl.trim());
        onIngestSuccess(res);
        setYoutubeUrl('');
      }
    } catch (err: any) {
      setError(err.message || 'Ingestion failed. Check backend logs.');
    } finally {
      setLoading(false);
    }
  };

  const isVideo = file && (file.name.endsWith('.mp4') || file.name.endsWith('.webm'));
  const fileSizeMB = file ? (file.size / 1024 / 1024).toFixed(1) : '0';

  return (
    <div>
      {/* Section header */}
      <div className="section-label" style={{ marginBottom: '10px' }}>
        <Upload size={10} />
        Ingest
      </div>

      {/* Tab switcher */}
      <div className="tab-bar" style={{ marginBottom: '10px' }}>
        <button
          type="button"
          onClick={() => { setActiveTab('upload'); setError(null); }}
          className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
        >
          File
        </button>
        <button
          type="button"
          onClick={() => { setActiveTab('youtube'); setError(null); }}
          className={`tab-btn ${activeTab === 'youtube' ? 'active' : ''}`}
        >
          YouTube
        </button>
      </div>

      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {activeTab === 'upload' ? (
          <div
            className={`dropzone ${isDragOver ? 'drag-over' : ''} ${file ? 'has-file' : ''}`}
            onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
            onDragLeave={() => setIsDragOver(false)}
            onDrop={handleDrop}
            style={{ padding: '18px 12px' }}
          >
            <input
              type="file"
              id="file-upload"
              accept=".mp3,.wav,.m4a,.flac,.ogg,.mp4,.webm"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
            <label htmlFor="file-upload" style={{ cursor: 'pointer', display: 'block', textAlign: 'center' }}>
              {file ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                  {isVideo
                    ? <FileVideo size={28} color="var(--c-purple)" />
                    : <FileAudio size={28} color="var(--c-cyan)" />
                  }
                  <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--c-text)' }}>{file.name}</div>
                  <span className="chip chip-dim">{fileSizeMB} MB</span>
                </div>
              ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '6px' }}>
                  <Upload size={22} color="var(--c-blue)" />
                  <div style={{ fontSize: '12px', color: 'var(--c-text-dim)' }}>Drop file or <span style={{ color: 'var(--c-blue)' }}>browse</span></div>
                  <div style={{ fontSize: '10px', color: 'var(--c-text-muted)', fontFamily: 'var(--mono)' }}>MP3 WAV M4A FLAC OGG MP4 · 500MB max</div>
                </div>
              )}
            </label>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: 'var(--c-text-muted)', fontFamily: 'var(--mono)', marginBottom: '2px' }}>
            <Link2 size={10} color="var(--c-red)" />
              YOUTUBE URL
            </div>
            <input
              type="url"
              placeholder="https://www.youtube.com/watch?v=..."
              value={youtubeUrl}
              onChange={(e) => setYoutubeUrl(e.target.value)}
              className="input"
            />
            <div style={{ fontSize: '10px', color: 'var(--c-text-muted)', fontFamily: 'var(--sans)', marginTop: '2px' }}>
              Paste public YouTube video link for automated transcription and speaker analysis.
            </div>
          </div>
        )}

        {error && (
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px', padding: '7px 10px', borderRadius: 'var(--radius)', background: 'rgba(255,77,106,0.08)', border: '1px solid rgba(255,77,106,0.25)', color: 'var(--c-red)', fontSize: '11px' }}>
            <AlertCircle size={12} style={{ flexShrink: 0 }} />
            <span>{error}</span>
          </div>
        )}

        <button
          type="submit"
          disabled={loading || (activeTab === 'upload' && !file) || (activeTab === 'youtube' && !youtubeUrl.trim())}
          className="btn btn-primary"
          style={{ width: '100%', justifyContent: 'center' }}
        >
          {loading ? (
            <><Loader2 size={13} className="spin" /> Ingesting media…</>
          ) : (
            <><ArrowRight size={13} /> Add to Audio Library</>
          )}
        </button>
      </form>
    </div>
  );
};
