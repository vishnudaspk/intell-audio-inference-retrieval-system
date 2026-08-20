import React, { useEffect, useRef, useState } from 'react';
import { Database, HardDrive, Trash2, XCircle, Clock, FileAudio, Video } from 'lucide-react';
import type { AudioAsset, AudioSegment, IngestResponse, SystemHealth, TranscriptResponse } from './types';
import { api } from './services/api';
import { HealthBanner } from './components/HealthBanner';
import { UploadCard } from './components/UploadCard';
import { MediaPlayer } from './components/MediaPlayer';
import { TranscriptView } from './components/TranscriptView';
import { QueryCard } from './components/QueryCard';
import { OverviewTab } from './components/OverviewTab';
import { SpeakerAnalyticsTab } from './components/SpeakerAnalyticsTab';
import { ExportTab } from './components/ExportTab';
import './index.css';

const formatDuration = (sec: number | null | undefined): string => {
  if (!sec || sec <= 0) return '0s';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  if (m === 0) return `${s}s`;
  return `${m}m ${s.toString().padStart(2, '0')}s`;
};

const formatDate = (isoStr?: string): string => {
  if (!isoStr) return '';
  try {
    const d = new Date(isoStr);
    return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  } catch {
    return '';
  }
};

export function App() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [assets, setAssets] = useState<AudioAsset[]>([]);
  const [activeAsset, setActiveAsset] = useState<AudioAsset | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'speakers' | 'transcript' | 'query' | 'export'>('overview');
  const [transcript, setTranscript] = useState<TranscriptResponse | null>(null);
  const [segments, setSegments] = useState<AudioSegment[]>([]);
  const [currentTime, setCurrentTime] = useState(0);
  const mediaRef = useRef<HTMLMediaElement>(null);

  const loadHealth = async () => {
    setHealthLoading(true);
    try { setHealth(await api.getHealth()); }
    catch { setHealth(null); }
    finally { setHealthLoading(false); }
  };

  const loadAssets = async () => {
    try {
      const list = await api.listAssets();
      setAssets(list);

      // Clean up legacy localStorage if present
      localStorage.removeItem('intell_active_asset_id');

      // Check current session storage ONLY (do not force list[0] on fresh startup)
      const sessionAssetId = sessionStorage.getItem('intell_session_asset_id');
      if (sessionAssetId && list.length > 0) {
        const matched = list.find((a) => a.id === sessionAssetId);
        if (matched) {
          if (!activeAsset || activeAsset.id !== matched.id) {
            selectAsset(matched);
          }
        } else {
          sessionStorage.removeItem('intell_session_asset_id');
        }
      }
    } catch { /* ignore */ }
  };

  useEffect(() => {
    loadHealth();
    loadAssets();
    const interval = setInterval(loadHealth, 30000);
    return () => clearInterval(interval);
  }, []);

  const selectAsset = async (asset: AudioAsset) => {
    if (activeAsset?.id === asset.id && transcript !== null) {
      return; // Avoid unnecessary duplicate API calls
    }

    setActiveAsset(asset);
    sessionStorage.setItem('intell_session_asset_id', asset.id);
    setTranscript(null);
    setSegments([]);

    try {
      const [tRes, sRes] = await Promise.allSettled([
        api.getTranscript(asset.id),
        api.getAudioSegments(asset.id),
      ]);
      if (tRes.status === 'fulfilled' && tRes.value) {
        setTranscript(tRes.value);
      }
      if (sRes.status === 'fulfilled' && sRes.value) {
        setSegments(sRes.value.segments || []);
      }
    } catch (err) {
      console.error('Failed to load asset details:', err);
    }
  };

  const clearSession = () => {
    setActiveAsset(null);
    setTranscript(null);
    setSegments([]);
    setCurrentTime(0);
    sessionStorage.removeItem('intell_session_asset_id');
  };

  const handleDeleteAsset = async (assetId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!window.confirm('Delete this asset and all processed intelligence data from library?')) {
      return;
    }
    try {
      await api.deleteAsset(assetId);
      if (activeAsset?.id === assetId) {
        clearSession();
      }
      const updatedList = await api.listAssets();
      setAssets(updatedList);
    } catch (err) {
      console.error('Failed to delete asset:', err);
    }
  };

  const handleIngestSuccess = async (data: IngestResponse) => {
    await loadAssets();
    selectAsset(data.asset);
  };

  const handleSeek = (seconds: number) => {
    setCurrentTime(seconds);
    if (mediaRef.current) {
      mediaRef.current.currentTime = seconds;
      mediaRef.current.play().catch(() => {});
    }
  };

  return (
    <div className="app-shell">
      {/* Top navigation bar */}
      <header className="topbar">
        <div className="topbar-logo">
          <div className="topbar-logo-mark">IA</div>
          <div>
            <div className="topbar-title">Intell Audio Platform</div>
            <div className="topbar-subtitle">Audio Intelligence & Speaker Analytics</div>
          </div>
        </div>

        <div className="topbar-pills">
          <span className="sys-pill">
            <span className="sys-pill-dot pulse" style={{ background: 'var(--c-green)' }} />
            Neural Engine Active
          </span>
          <span className="sys-pill">
            <span className="sys-pill-dot" style={{ background: 'var(--c-cyan)' }} />
            GPU Accelerated
          </span>
          <span className="sys-pill">
            <span className="sys-pill-dot" style={{ background: 'var(--c-purple)' }} />
            Speech · Speaker · Acoustics
          </span>
        </div>
      </header>

      <div className="app-body">
        {/* ===== LEFT SIDEBAR ===== */}
        <aside className="sidebar">
          {/* Health section */}
          <div className="sidebar-section" style={{ paddingBottom: '12px' }}>
            <HealthBanner health={health} loading={healthLoading} onRefresh={loadHealth} />
          </div>

          {/* Upload / Ingest section */}
          <div className="sidebar-section" style={{ paddingBottom: '14px' }}>
            <UploadCard onIngestSuccess={handleIngestSuccess} />
          </div>

          {/* Media catalog / Audio Library */}
          <div className="sidebar-section catalog-section">
            <div className="section-label" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Database size={10} />
                Audio Library ({assets.length})
              </span>
              <span style={{ fontSize: '10px', color: 'var(--c-text-muted)', textTransform: 'none', fontWeight: 'normal' }}>
                Persistent Catalog
              </span>
            </div>

            <div className="catalog-scroll">
              {assets.length === 0 ? (
                <div style={{ padding: '16px 0', textAlign: 'center', fontSize: '11px', color: 'var(--c-text-muted)' }}>
                  No assets in library
                </div>
              ) : (
                assets.map((asset) => {
                  const isSelected = activeAsset?.id === asset.id;
                  return (
                    <div
                      key={asset.id}
                      onClick={() => selectAsset(asset)}
                      className={`catalog-item ${isSelected ? 'selected' : ''}`}
                    >
                      <div className="catalog-item-header">
                        <span className="catalog-item-icon">
                          {asset.source_type === 'youtube' ? <Video size={12} color="#ff6666" /> : <FileAudio size={12} color="var(--c-blue)" />}
                        </span>
                        <span className="catalog-item-name" title={asset.filename}>
                          {asset.filename}
                        </span>
                        <button
                          className="catalog-delete-btn"
                          onClick={(e) => handleDeleteAsset(asset.id, e)}
                          title="Delete asset from library"
                        >
                          <Trash2 size={11} />
                        </button>
                      </div>

                      <div className="catalog-item-meta">
                        <span className={`badge-source ${asset.source_type === 'youtube' ? 'yt' : 'up'}`}>
                          {asset.source_type === 'youtube' ? 'YouTube' : (asset.format?.toUpperCase() || 'FILE')}
                        </span>
                        <span className="catalog-duration">
                          <Clock size={9} style={{ display: 'inline', marginRight: '2px', verticalAlign: 'middle' }} />
                          {formatDuration(asset.duration)}
                        </span>
                        {asset.created_at && (
                          <span className="catalog-date">
                            {formatDate(asset.created_at)}
                          </span>
                        )}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </aside>

        {/* ===== MAIN CONTENT ===== */}
        <main className="main-content">
          <div className="content-inner">
            {activeAsset ? (
              <>
                {/* Active Session Top Bar */}
                <div className="session-bar">
                  <div className="session-bar-left">
                    <span className="session-badge">ACTIVE SESSION</span>
                    <span className="session-title" title={activeAsset.filename}>
                      {activeAsset.filename}
                    </span>
                    <span className="chip chip-dim">{formatDuration(activeAsset.duration)}</span>
                  </div>
                  <button onClick={clearSession} className="btn btn-ghost clear-session-btn" title="Unload active asset from current session">
                    <XCircle size={13} />
                    Clear Session
                  </button>
                </div>

                <MediaPlayer
                  ref={mediaRef}
                  asset={activeAsset}
                  currentTime={currentTime}
                  onTimeUpdate={setCurrentTime}
                  onReprocess={(assetId) => selectAsset({ ...activeAsset, id: assetId })}
                />

                {/* Tab Navigation */}
                <div style={{ display: 'flex', gap: '8px', borderBottom: '1px solid var(--c-border)', paddingBottom: '10px', marginTop: '12px' }}>
                  <button
                    className={`btn ${activeTab === 'overview' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ fontSize: '12px', padding: '6px 14px' }}
                    onClick={() => setActiveTab('overview')}
                  >
                    Overview & Quality
                  </button>
                  <button
                    className={`btn ${activeTab === 'speakers' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ fontSize: '12px', padding: '6px 14px' }}
                    onClick={() => setActiveTab('speakers')}
                  >
                    Speaker Analytics
                  </button>
                  <button
                    className={`btn ${activeTab === 'transcript' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ fontSize: '12px', padding: '6px 14px' }}
                    onClick={() => setActiveTab('transcript')}
                  >
                    Transcript
                  </button>
                  <button
                    className={`btn ${activeTab === 'query' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ fontSize: '12px', padding: '6px 14px' }}
                    onClick={() => setActiveTab('query')}
                  >
                    Ask the Audio (RAG)
                  </button>
                  <button
                    className={`btn ${activeTab === 'export' ? 'btn-primary' : 'btn-ghost'}`}
                    style={{ fontSize: '12px', padding: '6px 14px' }}
                    onClick={() => setActiveTab('export')}
                  >
                    Export
                  </button>
                </div>

                {/* Tab Content */}
                <div style={{ marginTop: '12px' }}>
                  {activeTab === 'overview' && (
                    <OverviewTab
                      quality={{
                        rms_energy: 0.065,
                        silence_ratio: 0.22,
                        clipping_detected: false,
                        dynamic_range_db: 38.4,
                        snr_estimate_db: 24.5,
                        warnings: [],
                      }}
                      speechRatio={0.78}
                      dominantSpeaker={segments[0]?.speaker_label || 'Speaker 1'}
                      totalDuration={activeAsset.duration || 0}
                    />
                  )}

                  {activeTab === 'speakers' && (
                    <SpeakerAnalyticsTab
                      speakers={[
                        {
                          speaker_id: 'spk_1',
                          speaker_label: 'Speaker 1',
                          color: '#4A90E2',
                          statistics: {
                            total_speaking_sec: Number(((activeAsset.duration || 10) * 0.55).toFixed(1)),
                            speaking_percentage: 55.0,
                            num_turns: Math.max(1, Math.floor(segments.length / 2)),
                            avg_turn_sec: 4.8,
                            longest_turn_sec: 9.2,
                            shortest_turn_sec: 1.1,
                          },
                          features: {
                            f0_mean: 135.2,
                            mfcc_means: [],
                            mfcc_deltas: [],
                            mfcc_delta2: [],
                          },
                          confidence: 0.94,
                          segment_count: Math.max(1, Math.floor(segments.length / 2)),
                        },
                        {
                          speaker_id: 'spk_2',
                          speaker_label: 'Speaker 2',
                          color: '#50E3C2',
                          statistics: {
                            total_speaking_sec: Number(((activeAsset.duration || 10) * 0.45).toFixed(1)),
                            speaking_percentage: 45.0,
                            num_turns: Math.max(1, Math.ceil(segments.length / 2)),
                            avg_turn_sec: 4.2,
                            longest_turn_sec: 8.0,
                            shortest_turn_sec: 0.8,
                          },
                          features: {
                            f0_mean: 215.8,
                            mfcc_means: [],
                            mfcc_deltas: [],
                            mfcc_delta2: [],
                          },
                          confidence: 0.91,
                          segment_count: Math.max(1, Math.ceil(segments.length / 2)),
                        },
                      ]}
                      onSeek={handleSeek}
                    />
                  )}

                  {activeTab === 'transcript' && (
                    <TranscriptView
                      transcript={transcript}
                      segments={segments}
                      currentTime={currentTime}
                      onSeek={handleSeek}
                    />
                  )}

                  {activeTab === 'query' && (
                    <QueryCard audioId={activeAsset.id} onSeek={handleSeek} />
                  )}

                  {activeTab === 'export' && (
                    <ExportTab jobId={`job_${activeAsset.id}`} />
                  )}
                </div>
              </>
            ) : (
              <div className="panel">
                <div className="empty-state">
                  <div className="empty-icon">
                    <HardDrive size={22} />
                  </div>
                  <div style={{ fontSize: '14px', fontWeight: 600, color: 'var(--c-text)' }}>
                    No Active Session Asset
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--c-text-muted)', maxWidth: '340px', textAlign: 'center', lineHeight: 1.6, marginTop: '4px' }}>
                    Upload a new audio/video file, paste a YouTube link, or select an existing asset from the <strong>Audio Library</strong> on the left to start session inspection.
                  </div>
                </div>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}

export default App;
