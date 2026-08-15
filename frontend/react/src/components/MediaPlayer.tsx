import React, { forwardRef } from 'react';
import { Volume2, FileVideo, Clock } from 'lucide-react';
import type { AudioAsset } from '../types';
import { api } from '../services/api';

interface MediaPlayerProps {
  asset: AudioAsset;
  currentTime: number;
  onTimeUpdate: (time: number) => void;
}

const fmt = (sec: number) => {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 100);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

export const MediaPlayer = forwardRef<HTMLMediaElement, MediaPlayerProps>(
  ({ asset, currentTime, onTimeUpdate }, ref) => {
    const isVideo = asset.format.toLowerCase() === 'mp4' || asset.filename.toLowerCase().endsWith('.mp4');
    const mediaUrl = api.getMediaUrl(asset.id);

    return (
      <div className="panel">
        <div className="panel-header">
          <div className="panel-title">
            {isVideo ? <FileVideo size={12} color="var(--c-purple)" /> : <Volume2 size={12} color="var(--c-cyan)" />}
            <span style={{ maxWidth: '380px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', fontFamily: 'var(--sans)', textTransform: 'none', fontWeight: 500, letterSpacing: 0, fontSize: '12px', color: 'var(--c-text)' }}>
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
