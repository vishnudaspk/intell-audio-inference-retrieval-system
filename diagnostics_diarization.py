"""
Diarization Diagnostics Tool — V3.1 Root-Cause Inspector
Traces full pipeline execution on a single audio file without running full UI.
Outputs step-by-step mathematical & structural metrics.
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from collections import Counter

from database.sqlite_db import SQLiteRepository
from services.audio_service import AudioService
from services.vad_service import VADService
from services.transcription_service import TranscriptionService
from services.speaker_embedding_service import SpeakerEmbeddingService, ECAPA_EMBEDDING_DIM
from schemas.models import AudioAsset
from workers.audio_worker import AudioWorker

def run_diagnostics(audio_id_or_path: str, win_len: float = 2.0, win_hop: float = 1.0, max_k: int = 8):
    repo = SQLiteRepository()
    
    # 1. Resolve Audio File
    p = Path(audio_id_or_path)
    if p.exists():
        wav_path = p
        audio_id = "diagnostic_run"
        filename = p.name
        duration = 0.0
    else:
        audio_id = audio_id_or_path
        asset = repo.get_audio_asset(audio_id)
        if not asset:
            print(f"Error: Asset '{audio_id}' not found in database.")
            return
        filename = asset.filename
        duration = asset.duration or 0.0
        # Check normalized wav first
        wav_path = Path("data/audio") / f"{audio_id}.wav"
        if not wav_path.exists():
            wav_path = Path(asset.file_path)

    print("================================================================================")
    print("                      DIARIZATION ROOT-CAUSE DIAGNOSTICS                        ")
    print("================================================================================")
    print(f"Audio ID   : {audio_id}")
    print(f"Filename   : {filename}")
    print(f"WAV Path   : {wav_path} (exists={wav_path.exists()})")

    # Normalize audio to 16kHz mono if needed
    if not isinstance(asset, AudioAsset):
        asset = AudioAsset(id=audio_id, filename=filename, file_path=str(wav_path), format="wav", duration=duration)
    audio_service = AudioService()
    norm_wav = audio_service.normalize_to_wav(asset)
    import soundfile as sf
    info = sf.info(str(norm_wav))
    print(f"Duration   : {info.duration:.2f}s | Sample Rate: {info.samplerate} | Channels: {info.channels}")
    print("--------------------------------------------------------------------------------")

    # 2. Stage: VAD
    vad_service = VADService()
    vad_segs = vad_service.detect_segments(norm_wav)
    vad_segs = vad_service.filter_short_segments(vad_segs, min_duration_sec=0.25)
    vad_segs = vad_service.merge_close_segments(vad_segs, max_gap_sec=0.30)
    print(f"\n[STAGE 1: VAD SPEECH REGIONS] Count = {len(vad_segs)}")
    for i, (s, e, c) in enumerate(vad_segs):
        print(f"  VAD [{i:2d}]: {s:6.2f}s - {e:6.2f}s (dur={e-s:5.2f}s, conf={c:.2f})")

    # 3. Stage: Sliding Windows
    speech_intervals = [(s, e) for s, e, _ in vad_segs]
    windows = []
    for st, et in speech_intervals:
        dur = et - st
        if dur <= win_len:
            windows.append((round(st, 3), round(et, 3)))
        else:
            cur = st
            while cur + 0.8 <= et:
                w_end = min(cur + win_len, et)
                windows.append((round(cur, 3), round(w_end, 3)))
                cur += win_hop
            if windows and windows[-1][1] < et - 0.3:
                windows.append((round(max(st, et - win_len), 3), round(et, 3)))

    print(f"\n[STAGE 2: SPEAKER ANALYSIS WINDOWS] Count = {len(windows)} (win_len={win_len}s, hop={win_hop}s)")
    for i, (s, e) in enumerate(windows):
        print(f"  Win [{i:2d}]: {s:6.2f}s - {e:6.2f}s (dur={e-s:5.2f}s)")

    # 4. Stage: ECAPA-TDNN Embeddings & Normalization
    spk_service = SpeakerEmbeddingService()
    raw_embs = spk_service.embed_segments(norm_wav, windows)
    
    raw_norms = [float(np.linalg.norm(e)) for e in raw_embs]
    valid_mask = [n > 1e-6 for n in raw_norms]
    valid_indices = [i for i, v in enumerate(valid_mask) if v]
    valid_windows = [windows[i] for i in valid_indices]
    
    X = np.array([raw_embs[i] / raw_norms[i] for i in valid_indices])
    post_norms = [float(np.linalg.norm(v)) for v in X]

    print(f"\n[STAGE 3: EMBEDDINGS & NORMALIZATION]")
    print(f"  Total raw embeddings      : {len(raw_embs)} (Dim={ECAPA_EMBEDDING_DIM})")
    print(f"  Valid non-zero embeddings : {len(X)} / {len(raw_embs)}")
    print(f"  Raw norms min/mean/max    : {min(raw_norms):.4f} / {np.mean(raw_norms):.4f} / {max(raw_norms):.4f}")
    print(f"  Post-normalized norms     : min={min(post_norms):.4f}, max={max(post_norms):.4f}")

    # 5. Stage: Pairwise Cosine Similarity & Affinity Matrix
    from scipy.spatial.distance import pdist, squareform
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.metrics import silhouette_score

    dist_condensed = pdist(X, metric="cosine")
    dist_mat = squareform(dist_condensed)
    cos_sim_mat = 1.0 - dist_mat
    
    # Extract upper triangular off-diagonal similarities
    triu_idx = np.triu_indices(len(X), k=1)
    pairwise_sims = cos_sim_mat[triu_idx]

    print(f"\n[STAGE 4: PAIRWISE SIMILARITY & AFFINITY STATS]")
    print(f"  Cosine Sim min/mean/max/std : {np.min(pairwise_sims):.4f} / {np.mean(pairwise_sims):.4f} / {np.max(pairwise_sims):.4f} / {np.std(pairwise_sims):.4f}")
    
    # Cosine similarity distribution percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    perc_vals = np.percentile(pairwise_sims, percentiles)
    print(f"  Percentiles (p10..p99)      : " + ", ".join(f"p{p}={v:.3f}" for p, v in zip(percentiles, perc_vals)))

    # Affinity with cubic sharpening
    A = np.maximum(0, cos_sim_mat) ** 3
    np.fill_diagonal(A, 0.0)
    print(f"  Affinity A^3 mean/max       : {np.mean(A):.4f} / {np.max(A):.4f}")

    # 6. Stage: Speaker-Count Estimation (Eigengap & Silhouette)
    d_sum = np.sum(A, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(1e-8, d_sum))
    L_sym = np.eye(len(X)) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    eigenvalues, _ = np.linalg.eigh(L_sym)

    k_candidates = list(range(2, min(max_k + 1, len(X))))
    gaps = [float(eigenvalues[k] - eigenvalues[k - 1]) for k in k_candidates]
    best_k_eigengap = k_candidates[int(np.argmax(gaps))]

    Z = linkage(dist_condensed, method="average")

    print(f"\n[STAGE 5: SPEAKER-COUNT ESTIMATION (K Candidates)]")
    sil_scores = {}
    for k in k_candidates:
        cids = fcluster(Z, t=k, criterion="maxclust")
        if len(set(cids)) > 1:
            score = float(silhouette_score(dist_mat, cids, metric="precomputed"))
            sil_scores[k] = score
            print(f"  K={k}: Eigengap = {eigenvalues[k]-eigenvalues[k-1]:.4f} | Silhouette = {score:6.3f} | Cluster counts = {dict(Counter(cids))}")
        else:
            print(f"  K={k}: Eigengap = {eigenvalues[k]-eigenvalues[k-1]:.4f} | (Single cluster)")

    best_k_silhouette = max(sil_scores, key=sil_scores.get) if sil_scores else 1
    
    # Distance-based cluster count estimate (e.g. at 0.50 and 0.60)
    cids_dist_50 = fcluster(Z, t=0.50, criterion="distance")
    cids_dist_60 = fcluster(Z, t=0.60, criterion="distance")
    print(f"  Distance threshold (t=0.50) -> K = {len(set(cids_dist_50))}")
    print(f"  Distance threshold (t=0.60) -> K = {len(set(cids_dist_60))}")
    print(f"  Selected K (Eigengap)       -> K = {best_k_eigengap}")
    print(f"  Selected K (Max Silhouette) -> K = {best_k_silhouette}")

    # 7. Stage: Clustering & Centroid Similarities
    chosen_k = best_k_eigengap
    raw_cids = fcluster(Z, t=chosen_k, criterion="maxclust")
    
    # Chronological mapping
    cluster_order = {}
    next_num = 1
    for cid in raw_cids:
        if cid not in cluster_order:
            cluster_order[cid] = f"Speaker {next_num}"
            next_num += 1

    window_speaker_labels = [cluster_order[cid] for cid in raw_cids]
    cluster_counts = Counter(window_speaker_labels)

    # Centroids
    centroids = {}
    for cid in set(raw_cids):
        c_embs = X[raw_cids == cid]
        c_mean = np.mean(c_embs, axis=0)
        c_mean /= np.linalg.norm(c_mean)
        centroids[cluster_order[cid]] = c_mean

    print(f"\n[STAGE 6: CLUSTER CENTROIDS & SIZES (for K={chosen_k})]")
    for spk, cnt in sorted(cluster_counts.items()):
        print(f"  {spk:10s} : {cnt:2d} windows")

    print("\n  Pairwise Centroid Cosine Similarities:")
    spk_names = sorted(list(centroids.keys()))
    for i in range(len(spk_names)):
        for j in range(i + 1, len(spk_names)):
            s1, s2 = spk_names[i], spk_names[j]
            sim = float(np.dot(centroids[s1], centroids[s2]))
            print(f"    Sim({s1}, {s2}) = {sim:.4f}")

    # 8. Stage: Raw Diarization Timeline BEFORE Transcript Mapping
    print(f"\n[STAGE 7: RAW WINDOW SPEAKER TIMELINE (BEFORE TRANSCRIPT MAPPING)]")
    # Merge contiguous identical speaker windows
    merged_timeline = []
    cur_spk = window_speaker_labels[0]
    cur_st, cur_et = valid_windows[0]

    for (w_st, w_et), spk in zip(valid_windows[1:], window_speaker_labels[1:]):
        if spk == cur_spk and w_st <= cur_et + 0.5:
            cur_et = max(cur_et, w_et)
        else:
            merged_timeline.append((cur_st, cur_et, cur_spk))
            cur_st, cur_et, cur_spk = w_st, w_et, spk
    merged_timeline.append((cur_st, cur_et, cur_spk))

    for s, e, spk in merged_timeline:
        print(f"  {spk:10s}: {s:6.2f}s - {e:6.2f}s (dur={e-s:5.2f}s)")

    # 9. Stage: Whisper Transcription & Word Mapping
    print(f"\n[STAGE 8: WHISPER TRANSCRIPTION & WORD/PHRASE MAPPING]")
    transcription_service = TranscriptionService()
    transcript = transcription_service.transcribe_audio(audio_id=audio_id, wav_path=norm_wav)
    
    words = [
        {"word": w.word, "start_time": w.start or 0.0, "end_time": w.end or 0.0, "confidence": w.confidence}
        for w in (transcript.words or [])
    ]
    print(f"  Whisper raw segments = {len(transcript.segments)}, words = {len(words)}")

    diarized_segments, diag = spk_service.diarize_audio(
        wav_path=norm_wav,
        speech_intervals=speech_intervals,
        transcript_words=words if words else None,
        win_len=win_len,
        win_hop=win_hop,
        max_speakers=max_k,
    )

    mapped_spk_counts = Counter(s.get("speaker_label") for s in diarized_segments)
    print(f"\n  Mapped Dialogue Phrases Count = {len(diarized_segments)}")
    for i, seg in enumerate(diarized_segments):
        print(f"  [{i:2d}] [{seg['start_sec']:5.2f}s - {seg['end_sec']:5.2f}s] {seg.get('speaker_label', 'None'):10s} | {seg.get('text', '')}")

    # 10. Summary & Comparison
    print("\n================================================================================")
    print("                            DIARIZATION SUMMARY REPORT                          ")
    print("================================================================================")
    print(f"Duration                       : {info.duration:.2f}s")
    print(f"VAD Regions                    : {len(vad_segs)}")
    print(f"Speaker Windows                : {len(valid_windows)}")
    print(f"Embedding Dimension            : {ECAPA_EMBEDDING_DIM}")
    print(f"Pairwise Cosine Sim Mean / Max : {np.mean(pairwise_sims):.4f} / {np.max(pairwise_sims):.4f}")
    print(f"Selected K (Algorithm)         : {diag.get('estimated_speakers')}")
    print(f"Speakers BEFORE transcript map : {len(set(window_speaker_labels))} -> {sorted(list(set(window_speaker_labels)))}")
    print(f"Speakers AFTER transcript map  : {len(mapped_spk_counts)} -> {sorted(list(mapped_spk_counts.keys()))}")
    print(f"Cluster Distribution           : {dict(mapped_spk_counts)}")
    print("================================================================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarization Root-Cause Diagnostic Tool")
    parser.add_argument("audio_id", help="Audio ID or file path to evaluate")
    parser.add_argument("--win_len", type=float, default=2.0, help="Window duration in seconds")
    parser.add_argument("--win_hop", type=float, default=1.0, help="Window hop step in seconds")
    parser.add_argument("--max_k", type=int, default=8, help="Maximum candidate speakers")
    args = parser.parse_args()

    run_diagnostics(args.audio_id, win_len=args.win_len, win_hop=args.win_hop, max_k=args.max_k)
