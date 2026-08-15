import React from 'react';
import { CheckCircle, AlertTriangle, RefreshCw, Zap } from 'lucide-react';
import type { SystemHealth } from '../types';

interface HealthBannerProps {
  health: SystemHealth | null;
  loading: boolean;
  onRefresh: () => void;
}

export const HealthBanner: React.FC<HealthBannerProps> = ({ health, loading, onRefresh }) => {
  if (!health) {
    return (
      <div className="panel" style={{ marginBottom: '16px' }}>
        <div className="panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--c-text-muted)', fontSize: '12px', fontFamily: 'var(--mono)' }}>
            <RefreshCw size={13} className="spin" />
            Connecting to V3 backend at :8000…
          </div>
          <button onClick={onRefresh} className="btn btn-ghost" style={{ padding: '4px 10px', fontSize: '11px' }}>
            <RefreshCw size={12} className={loading ? 'spin' : ''} />
            Retry
          </button>
        </div>
      </div>
    );
  }

  const isOk = health.status === 'ok';

  const services = [
    { label: 'ASR', value: health.asr_engine, color: 'chip-blue' },
    { label: 'VAD', value: health.vad_engine, color: 'chip-cyan' },
    { label: 'EMBED', value: health.speaker_embedding_engine, color: 'chip-purple' },
    { label: 'LLM', value: health.lm_studio, color: health.lm_studio === 'available' ? 'chip-green' : 'chip-dim' },
    { label: 'QDRANT', value: health.qdrant, color: health.qdrant === 'available' ? 'chip-green' : 'chip-dim' },
  ];

  return (
    <div className="panel" style={{ marginBottom: '16px' }}>
      <div className="panel-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px', padding: '10px 14px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
          <span className={`chip ${isOk ? 'chip-green' : 'chip-amber'}`} style={{ gap: '5px' }}>
            {isOk ? <CheckCircle size={10} /> : <AlertTriangle size={10} />}
            {isOk ? 'V3 ACTIVE' : 'DEGRADED'}
          </span>
          <span style={{ fontSize: '10px', color: 'var(--c-text-muted)', fontFamily: 'var(--mono)' }}>
            {health.app_name} / {health.environment}
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap' }}>
            {services.map(s => (
              <span key={s.label} className={`chip ${s.color}`}>
                <Zap size={9} />
                {s.label}: {s.value}
              </span>
            ))}
          </div>
        </div>
        <button
          onClick={onRefresh}
          className="btn btn-ghost"
          style={{ padding: '4px 8px', fontSize: '11px' }}
          title="Refresh diagnostics"
        >
          <RefreshCw size={12} className={loading ? 'spin' : ''} />
        </button>
      </div>
    </div>
  );
};
