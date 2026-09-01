import React from 'react';
import type { SpeakerProfile } from '../types';
import { UserCheck, Clock, MessageSquare, Mic } from 'lucide-react';

interface Props {
  speakers: SpeakerProfile[];
  onSeek?: (seconds: number) => void;
}

export const SpeakerAnalyticsTab: React.FC<Props> = ({ speakers }) => {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '14px' }}>
        {speakers.map((spk) => (
          <div
            key={spk.speaker_id}
            style={{
              background: 'var(--c-surface)',
              border: `1px solid var(--c-border)`,
              borderTop: `3px solid ${spk.color || 'var(--c-blue)'}`,
              borderRadius: '8px',
              padding: '16px',
            }}
          >
            {/* Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <div
                  style={{
                    width: '12px',
                    height: '12px',
                    borderRadius: '50%',
                    background: spk.color || 'var(--c-blue)',
                  }}
                />
                <span style={{ fontWeight: 700, fontSize: '14px', color: 'var(--c-text)' }}>
                  {spk.speaker_label}
                </span>
              </div>
              <span className="chip" style={{ fontSize: '10px', background: 'rgba(255,255,255,0.06)' }}>
                {spk.statistics.speaking_percentage}% Talk Time
              </span>
            </div>

            {/* Metrics Breakdown */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px', fontSize: '11px' }}>
              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--c-text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Clock size={11} /> Total Speaking
                </div>
                <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--c-text)', marginTop: '2px' }}>
                  {spk.statistics.total_speaking_sec}s
                </div>
              </div>

              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--c-text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <MessageSquare size={11} /> Dialogue Turns
                </div>
                <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--c-text)', marginTop: '2px' }}>
                  {spk.statistics.num_turns} turns
                </div>
              </div>

              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--c-text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <Mic size={11} /> Avg Turn Duration
                </div>
                <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--c-text)', marginTop: '2px' }}>
                  {spk.statistics.avg_turn_sec}s
                </div>
              </div>

              <div style={{ background: 'rgba(0,0,0,0.2)', padding: '8px', borderRadius: '6px' }}>
                <div style={{ color: 'var(--c-text-muted)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                  <UserCheck size={11} /> Pitch Mean (F0)
                </div>
                <div style={{ fontWeight: 600, fontSize: '13px', color: 'var(--c-text)', marginTop: '2px' }}>
                  {spk.features.f0_mean ? `${spk.features.f0_mean} Hz` : 'N/A'}
                </div>
              </div>
            </div>

            {/* Speaking Rate / Confidence bar */}
            <div style={{ marginTop: '12px', paddingTop: '10px', borderTop: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--c-text-muted)' }}>
              <span>Attribution Confidence</span>
              <span style={{ fontWeight: 600, color: 'var(--c-green)' }}>{(spk.confidence * 100).toFixed(1)}%</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
