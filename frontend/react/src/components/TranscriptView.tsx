import React, { useState } from 'react';
import { FileText, Search, Clock, Mic, Activity, ChevronDown, ChevronUp } from 'lucide-react';
import type { AudioSegment, TranscriptResponse } from '../types';

interface TranscriptViewProps {
  transcript: TranscriptResponse | null;
  segments: AudioSegment[];
  currentTime: number;
  onSeek: (seconds: number) => void;
}

const fmt = (sec: number | null | undefined): string => {
  if (sec == null) return '--:--';
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 100);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

export const TranscriptView: React.FC<TranscriptViewProps> = ({
  transcript, segments, currentTime, onSeek,
}) => {
  const [activeTab, setActiveTab] = useState<'segments' | 'full' | 'matrix'>('segments');
  const [filterQuery, setFilterQuery] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);

  if (!transcript && segments.length === 0) {
    return (
      <div className="panel">
        <div className="empty-state">
          <div className="empty-icon"><FileText size={18} /></div>
          <div style={{ fontSize: '12px', color: 'var(--c-text-dim)' }}>No speech transcript for this asset</div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', maxWidth: '300px', lineHeight: 1.5 }}>
            This asset has not been transcribed yet or processing did not detect verbal speech intervals.
          </div>
        </div>
      </div>
    );
  }

  const filtered = segments.filter(s => s.text.toLowerCase().includes(filterQuery.toLowerCase()));

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <FileText size={12} color="var(--c-blue)" />
          Speech Intelligence
          {transcript?.language && (
            <span className="chip chip-blue" style={{ marginLeft: '2px' }}>{transcript.language.toUpperCase()}</span>
          )}
        </div>
        <div className="tab-bar">
          <button onClick={() => setActiveTab('segments')} className={`tab-btn ${activeTab === 'segments' ? 'active' : ''}`}>
            Segments {segments.length > 0 && `(${segments.length})`}
          </button>
          <button onClick={() => setActiveTab('full')} className={`tab-btn ${activeTab === 'full' ? 'active' : ''}`}>
            Full Text
          </button>
          <button onClick={() => setActiveTab('matrix')} className={`tab-btn ${activeTab === 'matrix' ? 'active' : ''}`}>
            Acoustic
          </button>
        </div>
      </div>

      {/* Segments Tab */}
      {activeTab === 'segments' && (
        <div style={{ display: 'flex', flexDirection: 'column' }}>
          {/* Filter bar */}
          <div style={{ padding: '10px 14px', borderBottom: '1px solid var(--c-border)', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <Search size={12} color="var(--c-text-muted)" style={{ flexShrink: 0 }} />
            <input
              type="text"
              placeholder="Filter segments…"
              value={filterQuery}
              onChange={(e) => setFilterQuery(e.target.value)}
              className="input"
              style={{ border: 'none', background: 'transparent', padding: '0', fontSize: '12px', boxShadow: 'none' }}
            />
          </div>

          {/* Segment list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '6px', maxHeight: '480px', overflowY: 'auto', padding: '10px 14px' }}>
            {filtered.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px', fontSize: '11px', color: 'var(--c-text-muted)' }}>
                No segments match filter.
              </div>
            ) : (
              filtered.map((seg) => {
                const isActive = currentTime >= seg.start_sec && currentTime <= seg.end_sec;
                const isExpanded = expanded === seg.id;

                return (
                  <div key={seg.id} className={`seg-card ${isActive ? 'active' : ''}`}>
                    <div className="seg-card-top">
                      <button
                        className="ts-btn"
                        onClick={() => onSeek(seg.start_sec)}
                        title="Seek to this segment"
                      >
                        <Clock size={10} />
                        {fmt(seg.start_sec)} → {fmt(seg.end_sec)}
                      </button>

                      {seg.speaker_label && (
                        <span className="chip chip-purple" style={{ fontWeight: 600 }}>
                          <Mic size={9} />
                          {seg.speaker_label}
                        </span>
                      )}

                      <div style={{ display: 'flex', alignItems: 'center', gap: '5px', flexWrap: 'wrap', marginLeft: 'auto' }}>
                        <span className="chip chip-dim">
                          VAD {(seg.vad_confidence * 100).toFixed(0)}%
                        </span>
                        {seg.speaker_embedding && !seg.speaker_label && (
                          <span className="chip chip-purple">
                            <Mic size={9} />
                            ECAPA-192
                          </span>
                        )}
                        {seg.acoustic_features?.f0_mean && (
                          <span className="chip chip-green">
                            <Activity size={9} />
                            {seg.acoustic_features.f0_mean.toFixed(0)} Hz
                          </span>
                        )}
                        <button
                          onClick={() => setExpanded(isExpanded ? null : seg.id)}
                          className="btn btn-ghost"
                          style={{ padding: '2px 5px', minWidth: 'auto' }}
                          title="Expand acoustic features"
                        >
                          {isExpanded ? <ChevronUp size={11} /> : <ChevronDown size={11} />}
                        </button>
                      </div>
                    </div>

                    <div className="seg-text">
                      {seg.text || (
                        <span style={{ fontStyle: 'italic', color: 'var(--c-text-muted)' }}>(non-verbal interval)</span>
                      )}
                    </div>

                    {isExpanded && (
                      <div className="seg-expand">
                        {/* Acoustics */}
                        <div className="seg-metric-group">
                          <div className="seg-metric-label" style={{ color: 'var(--c-cyan)' }}>
                            <Activity size={10} /> Librosa Acoustics
                          </div>
                          {[
                            ['F0 Mean', seg.acoustic_features?.f0_mean?.toFixed(1), 'Hz'],
                            ['F0 Range', seg.acoustic_features?.f0_min != null ? `${seg.acoustic_features.f0_min.toFixed(0)}–${seg.acoustic_features.f0_max?.toFixed(0)}` : null, 'Hz'],
                            ['RMS Energy', seg.acoustic_features?.rms_mean?.toFixed(4), ''],
                            ['Centroid', seg.acoustic_features?.spectral_centroid_mean?.toFixed(0), 'Hz'],
                            ['Rolloff', seg.acoustic_features?.spectral_rolloff_mean?.toFixed(0), 'Hz'],
                            ['ZCR', seg.acoustic_features?.zero_crossing_rate_mean?.toFixed(4), ''],
                          ].map(([k, v, u]) => (
                            <div key={k as string} className="seg-metric-row">
                              <span className="seg-metric-key">{k}</span>
                              <span className="seg-metric-val">{v ?? '—'} {v && u}</span>
                            </div>
                          ))}
                        </div>

                        {/* Embedding */}
                        <div className="seg-metric-group">
                          <div className="seg-metric-label" style={{ color: 'var(--c-purple)' }}>
                            <Mic size={10} /> SpeechBrain ECAPA
                          </div>
                          {[
                            ['Model', 'spkrec-ecapa-voxceleb'],
                            ['Dims', seg.speaker_embedding ? `${seg.speaker_embedding.length}-d L2` : '—'],
                            ['logprob', seg.avg_logprob?.toFixed(3) ?? '—'],
                            ['no_speech', seg.no_speech_prob?.toFixed(3) ?? '—'],
                          ].map(([k, v]) => (
                            <div key={k as string} className="seg-metric-row">
                              <span className="seg-metric-key">{k}</span>
                              <span className="seg-metric-val">{v}</span>
                            </div>
                          ))}
                          {seg.speaker_embedding && (
                            <div style={{ fontSize: '10px', color: 'var(--c-text-muted)', marginTop: '4px', wordBreak: 'break-all', fontFamily: 'var(--mono)' }}>
                              [{seg.speaker_embedding.slice(0, 5).map(v => v.toFixed(3)).join(', ')}, …]
                            </div>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </div>
      )}

      {/* Full Text Tab */}
      {activeTab === 'full' && (
        <div style={{ padding: '14px', maxHeight: '500px', overflowY: 'auto' }}>
          <div style={{
            background: 'var(--c-surface-1)',
            border: '1px solid var(--c-border)',
            borderRadius: 'var(--radius)',
            padding: '14px',
            fontSize: '13px',
            color: 'var(--c-text)',
            lineHeight: 1.75,
            fontFamily: 'var(--sans)',
            whiteSpace: 'pre-wrap',
          }}>
            {transcript?.text ||
              (segments.length > 0
                ? segments.map((s) => s.text).filter(Boolean).join(' ')
                : 'No text extracted.')}
          </div>
        </div>
      )}

      {/* Acoustic Matrix Tab */}
      {activeTab === 'matrix' && (
        <div style={{ maxHeight: '500px', overflowY: 'auto' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Interval</th>
                <th>Speaker</th>
                <th>Dur</th>
                <th>F0</th>
                <th>RMS</th>
                <th>Centroid</th>
                <th>Embed</th>
              </tr>
            </thead>
            <tbody>
              {segments.map((s, idx) => (
                <tr key={s.id || idx}>
                  <td className="col-ts" onClick={() => onSeek(s.start_sec)} title="Seek">
                    {fmt(s.start_sec)}
                  </td>
                  <td style={{ color: 'var(--c-purple)', fontWeight: 600 }}>{s.speaker_label ?? '—'}</td>
                  <td>{s.duration_sec.toFixed(2)}s</td>
                  <td className="col-f0">{s.acoustic_features?.f0_mean?.toFixed(1) ?? '—'}</td>
                  <td className="col-rms">{s.acoustic_features?.rms_mean?.toFixed(3) ?? '—'}</td>
                  <td className="col-sc">{s.acoustic_features?.spectral_centroid_mean?.toFixed(0) ?? '—'}</td>
                  <td className="col-emb">{s.speaker_embedding ? `${s.speaker_embedding.length}d` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};
