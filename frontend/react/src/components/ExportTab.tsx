import React from 'react';
import { api } from '../services/api';
import { Download, FileText, Table, FileCode, Layers } from 'lucide-react';

interface Props {
  jobId: string;
}

export const ExportTab: React.FC<Props> = ({ jobId }) => {
  const handleDownload = (format: string) => {
    const url = api.getExportUrl(jobId, format);
    window.open(url, '_blank');
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      <div style={{ fontSize: '13px', color: 'var(--c-text-muted)' }}>
        Export full audio intelligence results, transcripts, speaker metrics, and feature matrices directly for downstream research or production systems.
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '14px' }}>
        {/* Full JSON */}
        <div className="card" style={{ padding: '16px', background: 'var(--c-surface)', borderRadius: '8px', border: '1px solid var(--c-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px', color: 'var(--c-text)' }}>
            <FileCode size={16} color="var(--c-cyan)" />
            Canonical JSON
          </div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', margin: '8px 0 14px 0', lineHeight: 1.4 }}>
            Full structured AnalysisResult schema containing all acoustic, diarization, and conversational models.
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', border: '1px solid var(--c-border)' }} onClick={() => handleDownload('json')}>
            <Download size={13} style={{ marginRight: '6px' }} /> Download JSON
          </button>
        </div>

        {/* SRT Subtitles */}
        <div className="card" style={{ padding: '16px', background: 'var(--c-surface)', borderRadius: '8px', border: '1px solid var(--c-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px', color: 'var(--c-text)' }}>
            <FileText size={16} color="var(--c-green)" />
            SRT Subtitles
          </div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', margin: '8px 0 14px 0', lineHeight: 1.4 }}>
            SubRip format with speaker labels and aligned millisecond timestamps for video playback.
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', border: '1px solid var(--c-border)' }} onClick={() => handleDownload('srt')}>
            <Download size={13} style={{ marginRight: '6px' }} /> Download SRT
          </button>
        </div>

        {/* WebVTT */}
        <div className="card" style={{ padding: '16px', background: 'var(--c-surface)', borderRadius: '8px', border: '1px solid var(--c-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px', color: 'var(--c-text)' }}>
            <FileText size={16} color="var(--c-blue)" />
            WebVTT (.vtt)
          </div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', margin: '8px 0 14px 0', lineHeight: 1.4 }}>
            Browser-native subtitle tracks with speaker voice cues for HTML5 video elements.
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', border: '1px solid var(--c-border)' }} onClick={() => handleDownload('vtt')}>
            <Download size={13} style={{ marginRight: '6px' }} /> Download VTT
          </button>
        </div>

        {/* Segments CSV */}
        <div className="card" style={{ padding: '16px', background: 'var(--c-surface)', borderRadius: '8px', border: '1px solid var(--c-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px', color: 'var(--c-text)' }}>
            <Table size={16} color="var(--c-purple)" />
            Segments CSV
          </div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', margin: '8px 0 14px 0', lineHeight: 1.4 }}>
            Tabular speaker attribution segments, timestamps, confidence scores, and transcript text.
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', border: '1px solid var(--c-border)' }} onClick={() => handleDownload('csv')}>
            <Download size={13} style={{ marginRight: '6px' }} /> Download CSV
          </button>
        </div>

        {/* Feature Matrix */}
        <div className="card" style={{ padding: '16px', background: 'var(--c-surface)', borderRadius: '8px', border: '1px solid var(--c-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px', color: 'var(--c-text)' }}>
            <Layers size={16} color="var(--c-orange)" />
            Feature Matrix CSV
          </div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', margin: '8px 0 14px 0', lineHeight: 1.4 }}>
            Full acoustic matrix (Pitch, RMS, Centroid, ZCR, and 13 MFCC coefficients per segment).
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', border: '1px solid var(--c-border)' }} onClick={() => handleDownload('feature_matrix')}>
            <Download size={13} style={{ marginRight: '6px' }} /> Download Features
          </button>
        </div>

        {/* Markdown Report */}
        <div className="card" style={{ padding: '16px', background: 'var(--c-surface)', borderRadius: '8px', border: '1px solid var(--c-border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '14px', color: 'var(--c-text)' }}>
            <FileText size={16} color="#ffa500" />
            Executive Report
          </div>
          <div style={{ fontSize: '11px', color: 'var(--c-text-muted)', margin: '8px 0 14px 0', lineHeight: 1.4 }}>
            Comprehensive Markdown summary report including executive breakdown and signal statistics.
          </div>
          <button className="btn btn-ghost" style={{ width: '100%', border: '1px solid var(--c-border)' }} onClick={() => handleDownload('speaker_report')}>
            <Download size={13} style={{ marginRight: '6px' }} /> Download Markdown
          </button>
        </div>
      </div>
    </div>
  );
};
