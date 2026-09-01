import numpy as np
from pathlib import Path
from services.speaker_embedding_service import SpeakerEmbeddingService
from database.sqlite_db import SQLiteRepository
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

def run_phrase_centric_diarization(audio_id):
    repo = SQLiteRepository()
    asset = repo.get_audio_asset(audio_id)
    if not asset:
        print(f"Asset {audio_id} not found in DB")
        return
    from services.audio_service import AudioService
    audio_service = AudioService()
    wav_path = audio_service.normalize_to_wav(asset)

    with repo._get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT start_time, end_time, word FROM transcript_words WHERE audio_id = ? ORDER BY start_time", (audio_id,))
        words = c.fetchall()

    if not words:
        from services.transcription_service import TranscriptionService
        ts = TranscriptionService()
        transcript = ts.transcribe_audio(audio_id, wav_path)
        words = [
            {"start_time": w.start or 0.0, "end_time": w.end or 0.0, "word": w.word}
            for w in (transcript.words or [])
        ]
        repo.save_alignment_words(audio_id, transcript.words or [])

    if not words:
        print(f"No words for {audio_id}")
        return

    # 1. Extract dialogue phrases
    phrases = []
    cur_words = [words[0]]
    for w in words[1:]:
        st = w.get("start_time") if isinstance(w, dict) else w["start_time"]
        last_et = cur_words[-1].get("end_time") if isinstance(cur_words[-1], dict) else cur_words[-1]["end_time"]
        gap = st - last_et
        last_word = str(cur_words[-1].get("word") if isinstance(cur_words[-1], dict) else cur_words[-1]["word"]).strip()
        is_punct = last_word.endswith(('.', '?', '!', '...', ';', ':'))
        if gap >= 0.35 or (is_punct and gap >= 0.15):
            p_st = cur_words[0]['start_time']
            p_et = cur_words[-1]['end_time']
            p_txt = "".join(str(x['word']) for x in cur_words).strip()
            phrases.append({"start_sec": round(p_st, 3), "end_sec": round(p_et, 3), "text": p_txt})
            cur_words = [w]
        else:
            cur_words.append(w)
    if cur_words:
        p_st = cur_words[0]['start_time']
        p_et = cur_words[-1]['end_time']
        p_txt = "".join(str(x['word']) for x in cur_words).strip()
        phrases.append({"start_sec": round(p_st, 3), "end_sec": round(p_et, 3), "text": p_txt})

    # 2. Extract Speech Analysis Windows:
    # A) Phrase units themselves (with center padding if < 1.2s)
    # B) Sub-windows for long phrases (> 2.5s)
    windows = []
    win_sources = [] # 'phrase' or 'subwindow'
    for i, p in enumerate(phrases):
        st, et = p["start_sec"], p["end_sec"]
        dur = et - st
        if dur <= 2.5:
            # Pad up to 1.2s for ECAPA stability
            pad = max(0.0, (1.2 - dur) / 2.0)
            windows.append((max(0.0, round(st - pad, 3)), round(et + pad, 3)))
            win_sources.append(i)
        else:
            # Long monologue: slice into 1.5s sub-windows
            cur = st
            while cur + 0.8 <= et:
                w_end = min(cur + 1.5, et)
                windows.append((round(cur, 3), round(w_end, 3)))
                win_sources.append(i)
                cur += 0.75
            if windows[-1][1] < et - 0.2:
                windows.append((round(max(st, et - 1.5), 3), round(et, 3)))
                win_sources.append(i)

    service = SpeakerEmbeddingService()
    raw_embs = service.embed_segments(wav_path, windows)
    valid_mask = [np.linalg.norm(e) > 1e-6 for e in raw_embs]
    X = np.array([raw_embs[i] / np.linalg.norm(raw_embs[i]) for i, v in enumerate(valid_mask) if v])
    valid_sources = [win_sources[i] for i, v in enumerate(valid_mask) if v]

    # Cluster using Normalized Laplacian Eigengap
    A = np.maximum(0, np.dot(X, X.T)) ** 3
    np.fill_diagonal(A, 0.0)
    d_sum = np.sum(A, axis=1)
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(1e-8, d_sum))
    L_sym = np.eye(len(X)) - (A * d_inv_sqrt[:, None]) * d_inv_sqrt[None, :]
    eigenvalues, _ = np.linalg.eigh(L_sym)

    k_candidates = list(range(2, min(8 + 1, len(X))))
    gaps = [float(eigenvalues[k] - eigenvalues[k - 1]) for k in k_candidates]
    best_k = k_candidates[int(np.argmax(gaps))]

    Z = linkage(pdist(X, metric="cosine"), method="average")
    raw_cids = fcluster(Z, t=best_k, criterion="maxclust")

    # Map cluster IDs to phrases by majority of constituent windows
    phrase_cids = []
    for p_idx in range(len(phrases)):
        p_cids = [raw_cids[i] for i, src in enumerate(valid_sources) if src == p_idx]
        if p_cids:
            from collections import Counter
            phrase_cids.append(Counter(p_cids).most_common(1)[0][0])
        else:
            phrase_cids.append(raw_cids[0])

    # Chronological mapping
    cluster_order = {}
    next_num = 1
    for cid in phrase_cids:
        if cid not in cluster_order:
            cluster_order[cid] = f"Speaker {next_num}"
            next_num += 1

    labels = [cluster_order[cid] for cid in phrase_cids]

    print(f"\n================================================================================")
    print(f"Asset: {audio_id} | Total Phrases: {len(phrases)} | Discovered K: {best_k}")
    print(f"Distinct Speakers: {sorted(list(set(labels)))}")
    print("--------------------------------------------------------------------------------")
    for i, (p, lbl) in enumerate(zip(phrases, labels)):
        print(f"[{i:2d}] [{p['start_sec']:5.2f}s - {p['end_sec']:5.2f}s] {lbl:10s} | {p['text']}")

for aid, title in [
    ("8d1b1162-756e-4338-878a-9a4827e2acfa", "Asset 1: Jane Reading Lips (4 speakers)"),
    ("2a0482e1-7068-4f07-a7ce-c910356ce8d5", "Asset 2: Reverse Surveillance (3-4 speakers)"),
    ("c9b2671d-155f-4dae-9d28-25f2cf46790c", "Asset 3: Guide Dogs Hotel (2-3 speakers)"),
]:
    print(f"\n--- Testing {title} ---")
    run_phrase_centric_diarization(aid)
