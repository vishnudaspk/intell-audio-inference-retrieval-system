import React from 'react';
import { RefreshCw } from 'lucide-react';
import type { SystemHealth } from '../types';

interface HealthBannerProps {
  health: SystemHealth | null;
  loading: boolean;
  onRefresh: () => void;
}

export const HealthBanner: React.FC<HealthBannerProps> = ({ health, loading, onRefresh }) => {
  if (!health) {
    return (
      <div
        style={{
          background: 'rgba(255, 255, 255, 0.02)',
          border: '1px solid var(--c-border)',
          borderRadius: '10px',
          padding: '12px 14px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          marginBottom: '14px',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--c-text-muted)', fontSize: '11px', fontFamily: 'var(--mono)' }}>
          <RefreshCw size={12} className="spin" />
          <span>Connecting to Intelligence Engine...</span>
        </div>
        <button onClick={onRefresh} className="btn btn-ghost" style={{ padding: '4px 8px', fontSize: '10px' }}>
          <RefreshCw size={10} className={loading ? 'spin' : ''} />
        </button>
      </div>
    );
  }

  const isOk = health.status === 'ok';

  const modules = [
    { label: 'ASR', value: health.asr_engine || 'whisper', active: true },
    { label: 'VAD', value: health.vad_engine || 'silero', active: true },
    { label: 'EMBED', value: health.speaker_embedding_engine || 'ecapa-tdnn', active: true },
    { label: 'LLM', value: health.lm_studio, active: health.lm_studio === 'available' },
    { label: 'VECTOR', value: health.qdrant, active: health.qdrant === 'available' },
  ];

  return (
    <div
      style={{
        background: 'linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%)',
        border: '1px solid rgba(255, 255, 255, 0.08)',
        borderRadius: '10px',
        padding: '12px 14px',
        marginBottom: '14px',
        boxShadow: '0 4px 12px rgba(0,0,0,0.15)',
      }}
    >
      {/* Header bar inside banner */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '10px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <div
            style={{
              width: '8px',
              height: '8px',
              borderRadius: '50%',
              background: isOk ? '#22c55e' : '#f59e0b',
              boxShadow: isOk ? '0 0 8px rgba(34, 197, 94, 0.6)' : '0 0 8px rgba(245, 158, 11, 0.6)',
            }}
          />
          <span style={{ fontSize: '11px', fontWeight: 700, letterSpacing: '0.06em', color: isOk ? 'var(--c-text)' : '#f59e0b', textTransform: 'uppercase' }}>
            {isOk ? 'System Operational' : 'Degraded Services'}
          </span>
        </div>

        <button
          onClick={onRefresh}
          className="btn btn-ghost"
          style={{ padding: '3px 6px', fontSize: '10px', color: 'var(--c-text-muted)' }}
          title="Refresh service diagnostics"
        >
          <RefreshCw size={11} className={loading ? 'spin' : ''} />
        </button>
      </div>

      {/* Modules Status Pills */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(80px, 1fr))', gap: '6px' }}>
        {modules.map((m) => (
          <div
            key={m.label}
            style={{
              background: 'rgba(0, 0, 0, 0.3)',
              border: `1px solid ${m.active ? 'rgba(255, 255, 255, 0.08)' : 'rgba(255, 77, 106, 0.2)'}`,
              borderRadius: '6px',
              padding: '6px 8px',
              display: 'flex',
              flexDirection: 'column',
              gap: '2px',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <span style={{ fontSize: '9px', fontWeight: 600, color: 'var(--c-text-muted)', letterSpacing: '0.05em' }}>
                {m.label}
              </span>
              <span
                style={{
                  width: '5px',
                  height: '5px',
                  borderRadius: '50%',
                  background: m.active ? '#22c55e' : '#6b7280',
                }}
              />
            </div>
            <span
              style={{
                fontSize: '10px',
                fontWeight: 600,
                color: m.active ? 'var(--c-text)' : 'var(--c-text-muted)',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                whiteSpace: 'nowrap',
                fontFamily: 'var(--mono)',
              }}
            >
              {m.value || 'offline'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};
