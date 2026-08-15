import { useEffect, useRef, useState } from 'react';
import { Database, HardDrive, GitBranch } from 'lucide-react';
import type { AudioAsset, AudioSegment, IngestResponse, SystemHealth, TranscriptResponse } from './types';
import { api } from './services/api';
import { HealthBanner } from './components/HealthBanner';
import { UploadCard } from './components/UploadCard';
import { MediaPlayer } from './components/MediaPlayer';
import { TranscriptView } from './components/TranscriptView';
import { QueryCard } from './components/QueryCard';
import './index.css';

export function App() {
  const [health, setHealth] = useState<SystemHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(false);
  const [assets, setAssets] = useState<AudioAsset[]>([]);
  const [activeAsset, setActiveAsset] = useState<AudioAsset | null>(null);
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
      if (list.length > 0) {
        const savedId = localStorage.getItem('intell_active_asset_id');
        const matched = savedId ? list.find((a) => a.id === savedId) : null;
        const target = matched || list[0];
        if (!activeAsset || activeAsset.id !== target.id) {
          selectAsset(target);
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
    setActiveAsset(asset);
    localStorage.setItem('intell_active_asset_id', asset.id);
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
            <div className="topbar-title">Intell Audio</div>
            <div className="topbar-subtitle">Inference & Retrieval System · V3</div>
          </div>
        </div>

        <div className="topbar-pills">
          <span className="sys-pill">
            <span className="sys-pill-dot pulse" style={{ background: 'var(--c-green)' }} />
            RTX 4060
          </span>
          <span className="sys-pill">
            <span className="sys-pill-dot" style={{ background: 'var(--c-cyan)' }} />
            CUDA 12.6
          </span>
          <span className="sys-pill">
            <span className="sys-pill-dot" style={{ background: 'var(--c-blue)' }} />
            Whisper · Silero · ECAPA
          </span>
          <span className="sys-pill">
            <GitBranch size={9} />
            v3-multimodal-intelligence
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

          {/* Media catalog */}
          <div className="sidebar-section" style={{ paddingBottom: '6px', display: 'flex', flexDirection: 'column', flex: 1, minHeight: 0 }}>
            <div className="section-label">
              <Database size={10} />
              Media Catalog ({assets.length})
            </div>
          </div>

          <div className="catalog-scroll">
            {assets.length === 0 ? (
              <div style={{ padding: '16px 0', textAlign: 'center', fontSize: '11px', color: 'var(--c-text-muted)' }}>
                No assets yet
              </div>
            ) : (
              assets.map((asset) => (
                <button
                  key={asset.id}
                  onClick={() => selectAsset(asset)}
                  className={`catalog-item ${activeAsset?.id === asset.id ? 'selected' : ''}`}
                >
                  <span className="catalog-item-name">{asset.filename}</span>
                  <span className="catalog-item-meta">
                    {asset.duration > 0 ? `${asset.duration.toFixed(0)}s` : asset.format.toUpperCase()}
                  </span>
                </button>
              ))
            )}
          </div>
        </aside>

        {/* ===== MAIN CONTENT ===== */}
        <main className="main-content">
          <div className="content-inner">
            {activeAsset ? (
              <>
                <MediaPlayer
                  ref={mediaRef}
                  asset={activeAsset}
                  currentTime={currentTime}
                  onTimeUpdate={setCurrentTime}
                />
                <QueryCard audioId={activeAsset.id} onSeek={handleSeek} />
                <TranscriptView
                  transcript={transcript}
                  segments={segments}
                  currentTime={currentTime}
                  onSeek={handleSeek}
                />
              </>
            ) : (
              <div className="panel">
                <div className="empty-state">
                  <div className="empty-icon">
                    <HardDrive size={18} />
                  </div>
                  <div style={{ fontSize: '13px', fontWeight: 600, color: 'var(--c-text-dim)' }}>
                    No Asset Selected
                  </div>
                  <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', maxWidth: '280px', textAlign: 'center', lineHeight: 1.6 }}>
                    Upload an audio or video file, or select an existing asset from the catalog to inspect V3 speech intelligence.
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
