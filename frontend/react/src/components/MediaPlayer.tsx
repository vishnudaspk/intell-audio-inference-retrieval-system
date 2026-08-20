import React, { forwardRef, useState } from 'react';
import { Volume2, FileVideo, Clock, Play, Loader2, CheckCircle } from 'lucide-react';
import type { AudioAsset } from '../types';
import { api } from '../services/api';

interface MediaPlayerProps {
  asset: AudioAsset;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
  onReprocess?: (assetId: string) => void;
}

type PipelineState = 'idle' | 'running' | 'done' | 'error';

const fmt = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 100);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

export const MediaPlayer = forwardRef<HTMLMediaElement, MediaPlayerProps>(
  ({ asset, currentTime, onTimeUpdate, onReprocess }, ref) => {
    const isVideo = asset.format.toLowerCase() === 'mp4' || asset.filename.toLowerCase().endsWith('.mp4');
    const mediaUrl = api.getMediaUrl(asset.id);
    const [pipelineState, setPipelineState] = useState<PipelineState>('idle');
    const [pipelineError, setPipelineError] = useState<string | null>(null);
    const [progressPct, setProgressPct] = useState<number>(0);
    const [progressStage, setProgressStage] = useState<string>('');

    const handleRunPipeline = async () => {
      if (pipelineState === 'running') return;
      setPipelineState('running');
      setPipelineError(null);
      setProgressPct(10);
      setProgressStage('Initializing audio analysis…');

      // Simulate smooth progress increments while background task processes
      const timer = setInterval(() => {
        setProgressPct((prev) => {
          if (prev < 30) {
            setProgressStage('Voice activity & speech detection…');
            return prev + 15;
          }
          if (prev < 65) {
            setProgressStage('Speech transcription & alignment…');
            return prev + 10;
          }
          if (prev < 85) {
            setProgressStage('Speaker diarization & acoustics…');
            return prev + 5;
          }
          if (prev < 95) {
            setProgressStage('Synthesizing intelligence results…');
            return prev + 2;
          }
          return prev;
        });
      }, 500);

      try {
        await api.reprocessAsset(asset.id);
        clearInterval(timer);
        setProgressPct(100);
        setProgressStage('Complete');
        setPipelineState('done');
        if (onReprocess) onReprocess(asset.id);
        setTimeout(() => {
          setPipelineState('idle');
          setProgressPct(0);
          setProgressStage('');
        }, 3000);
      } catch (err: any) {
        clearInterval(timer);
        setPipelineError(err?.message || 'Processing failed');
        setPipelineState('error');
        setTimeout(() => {
          setPipelineState('idle');
          setProgressPct(0);
          setProgressStage('');
        }, 4000);
      }
    };

    return (
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            {isVideo ? <FileVideo size={12} color="var(--c-purple)" /> : <Volume2 size={12} color="var(--c-cyan)" />}
            <span style={{ maxWidth: '260px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'var(--sans)', textTransform: 'none', fontWeight: 500, letterSpacing: 0, fontSize: '12px', color: 'var(--c-text)' }}>
              {asset.filename}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '4px', fontFamily: 'var(--mono)', fontSize: '11px', color: 'var(--c-cyan)' }}>
              <Clock size={11} />
              {fmt(currentTime)}
            </span>
            <span style={{ color: 'var(--c-text-muted)', fontFamily: 'var(--mono)', fontSize: '11px' }}>
              / {fmt(asset.duration)}
            </span>
            <span className="chip chip-dim">{asset.format.toUpperCase()}</span>

            {/* Run Analysis button */}
            <button
              className={`run-pipeline-btn ${pipelineState}`}
              onClick={handleRunPipeline}
              disabled={pipelineState === 'running'}
              style={{ minWidth: pipelineState === 'running' ? '180px' : 'auto' }}
              title={
                pipelineState === 'running' ? progressStage
                : pipelineState === 'done' ? 'Analysis complete!'
                : pipelineState === 'error' ? (pipelineError || 'Error occurred')
                : 'Run complete audio intelligence & speaker analytics'
              }
            >
              {pipelineState === 'running' ? (
                <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <Loader2 size={11} className="spin" />
                  <span>Analyzing ({progressPct}%)</span>
                </div>
              ) : pipelineState === 'done' ? (
                <><CheckCircle size={11} /> Analysis Ready</>
              ) : pipelineState === 'error' ? (
                <><Play size={11} /> Error — Retry</>
              ) : (
                <><Play size={11} /> Analyze Audio</>
              )}
            </button>
          </div>
        </div>

        <div style={{ padding: '12px' }}>
          {isVideo ? (
            <video
              ref={ref as React.RefObject<HTMLVideoElement>}
              src={mediaUrl}
              controls
              onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
              style={{ width: '100%', maxHeight: '280px', borderRadius: 'var(--radius)', backgroundColor: '#000', display: 'block' }}
            />
          ) : (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '10px', color: 'var(--c-text-muted)', fontFamily: 'var(--mono)' }}>
                <span>16kHz Mono Normalized Stream</span>
                <span style={{ color: 'var(--c-text-muted)' }}>
                  {asset.duration > 0 ? `${asset.duration.toFixed(1)}s` : '—'}
                </span>
              </div>
              <audio
                ref={ref as React.RefObject<HTMLAudioElement>}
                src={mediaUrl}
                controls
                onTimeUpdate={(e) => onTimeUpdate(e.currentTarget.currentTime)}
                style={{ width: '100%' }}
              />
            </div>
          )}
        </div>
      </div>
    );
  }
);

MediaPlayer.displayName = 'MediaPlayer';
