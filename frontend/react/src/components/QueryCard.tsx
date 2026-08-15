import React, { useState } from 'react';
import { Search, Cpu, Clock, AlertCircle, Loader2, BookOpen, Quote } from 'lucide-react';
import type { Citation, RAGResponse, SearchResponse } from '../types';
import { api } from '../services/api';

interface QueryCardProps {
  audioId: string | null;
  onSeek: (seconds: number) => void;
}

const fmt = (sec: number): string => {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 100);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
};

export const QueryCard: React.FC<QueryCardProps> = ({ audioId, onSeek }) => {
  const [mode, setMode] = useState<'rag' | 'search'>('rag');
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ragResult, setRagResult] = useState<RAGResponse | null>(null);
  const [searchResult, setSearchResult] = useState<SearchResponse | null>(null);

  const handleQuery = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      if (mode === 'rag') {
        const res = await api.askAudio(query.trim(), audioId || undefined);
        setRagResult(res);
        setSearchResult(null);
      } else {
        if (!audioId) throw new Error('Lexical search requires an active audio asset.');
        const res = await api.searchTranscript(audioId, query.trim());
        setSearchResult(res);
        setRagResult(null);
      }
    } catch (err: any) {
      setError(err.message || 'Query failed. Check LM Studio / Qdrant.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="panel">
      <div className="panel-header">
        <div className="panel-title">
          <Cpu size={12} color="var(--c-purple)" />
          Ask & Retrieve
        </div>
        <div className="tab-bar">
          <button
            type="button"
            onClick={() => { setMode('rag'); setError(null); }}
            className={`tab-btn ${mode === 'rag' ? 'active' : ''}`}
          >
            AI RAG
          </button>
          <button
            type="button"
            onClick={() => { setMode('search'); setError(null); }}
            className={`tab-btn ${mode === 'search' ? 'active' : ''}`}
          >
            Lexical
          </button>
        </div>
      </div>

      <div style={{ padding: '12px 14px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
        <form onSubmit={handleQuery} style={{ display: 'flex', gap: '6px', alignItems: 'stretch' }}>
          <input
            type="text"
            placeholder={
              mode === 'rag'
                ? "Ask anything about this audio…"
                : "Search for exact words or phrases…"
            }
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="input"
            style={{ flex: 1 }}
          />
          <button
            type="submit"
            disabled={loading || !query.trim()}
            className="btn btn-primary"
            style={{ flexShrink: 0 }}
          >
            {loading
              ? <Loader2 size={13} className="spin" />
              : <Search size={13} />
            }
            {mode === 'rag' ? 'Ask' : 'Search'}
          </button>
        </form>

        {error && (
          <div style={{ display: 'flex', alignItems: 'flex-start', gap: '6px', padding: '8px 10px', borderRadius: 'var(--radius)', background: 'rgba(255,77,106,0.08)', border: '1px solid rgba(255,77,106,0.25)', color: 'var(--c-red)', fontSize: '11px' }}>
            <AlertCircle size={12} style={{ flexShrink: 0, marginTop: '1px' }} />
            <span>{error}</span>
          </div>
        )}

        {/* RAG Result */}
        {ragResult && (
          <div style={{ border: '1px solid rgba(167,110,244,0.25)', borderRadius: 'var(--radius-lg)', background: 'rgba(167,110,244,0.05)', overflow: 'hidden' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', borderBottom: '1px solid rgba(167,110,244,0.15)', background: 'rgba(167,110,244,0.06)' }}>
              <span style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '10px', fontFamily: 'var(--mono)', fontWeight: 600, color: 'var(--c-purple)', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                <BookOpen size={10} />
                Grounded Response
              </span>
              <span style={{ fontSize: '10px', fontFamily: 'var(--mono)', color: 'var(--c-text-muted)' }}>
                {ragResult.model} · {ragResult.processing_time.toFixed(2)}s
              </span>
            </div>
            <div style={{ padding: '12px', fontSize: '13px', color: 'var(--c-text)', lineHeight: 1.7 }}>
              {ragResult.answer}
            </div>

            {ragResult.citations?.length > 0 && (
              <div style={{ borderTop: '1px solid rgba(167,110,244,0.15)', padding: '10px 12px', display: 'flex', flexDirection: 'column', gap: '5px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '5px', fontSize: '10px', fontFamily: 'var(--mono)', color: 'var(--c-text-muted)', fontWeight: 600, marginBottom: '2px' }}>
                  <Quote size={10} />
                  TIMESTAMP CITATIONS ({ragResult.citations.length})
                </div>
                {ragResult.citations.map((c: Citation, idx: number) => (
                  <div key={idx} className="citation-card">
                    <span className="citation-text">"{c.text}"</span>
                    <button
                      onClick={() => onSeek(c.start_time)}
                      className="ts-btn"
                      style={{ flexShrink: 0 }}
                    >
                      <Clock size={9} />
                      {fmt(c.start_time)}
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Search Result */}
        {searchResult && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '5px' }}>
            <div style={{ fontSize: '10px', fontFamily: 'var(--mono)', color: 'var(--c-text-muted)', padding: '2px 0' }}>
              {searchResult.results_count} result(s) for <span style={{ color: 'var(--c-text-dim)' }}>"{searchResult.query}"</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', maxHeight: '240px', overflowY: 'auto' }}>
              {searchResult.results.map((res, idx) => (
                <div
                  key={idx}
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: '10px', padding: '8px 10px', borderRadius: 'var(--radius)', border: '1px solid var(--c-border)', background: 'var(--c-surface-1)', fontSize: '12px' }}
                >
                  <span style={{ color: 'var(--c-text)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
                    {res.matched_text}
                  </span>
                  <button onClick={() => onSeek(res.start)} className="ts-btn" style={{ flexShrink: 0 }}>
                    <Clock size={9} />
                    {fmt(res.start)}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
