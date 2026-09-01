import numpy as np
from pathlib import Path
from services.speaker_embedding_service import SpeakerEmbeddingService
from database.sqlite_db import SQLiteRepository
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

repo = SQLiteRepository()
audio_id = "34266ef2-87de-4b39-8535-00ca3c931fe7"
wav_path = Path(f"data/audio/{audio_id}.wav")

with repo._get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT start_time, end_time, word FROM transcript_words WHERE audio_id = ? ORDER BY start_time", (audio_id,))
    words = c.fetchall()

service = SpeakerEmbeddingService()

# 1. Inspect the dialogue phrases
phrases = []
cur_words = [words[0]]
for w in words[1:]:
    gap = w['start_time'] - cur_words[-1]['end_time']
    last_word = cur_words[-1]['word'].strip()
    is_punct = last_word.endswith(('.', '?', '!', '...', ';', ':'))
    if gap >= 0.35 or (is_punct and gap >= 0.15):
        p_st = cur_words[0]['start_time']
        p_et = cur_words[-1]['end_time']
        p_txt = "".join(x['word'] for x in cur_words).strip()
        phrases.append({"start_sec": round(p_st, 3), "end_sec": round(p_et, 3), "text": p_txt})
        cur_words = [w]
    else:
        cur_words.append(w)
if cur_words:
    p_st = cur_words[0]['start_time']
    p_et = cur_words[-1]['end_time']
    p_txt = "".join(x['word'] for x in cur_words).strip()
    phrases.append({"start_sec": round(p_st, 3), "end_sec": round(p_et, 3), "text": p_txt})

print(f"Total dialogue phrases: {len(phrases)}")

# 2. Extract phrase-level embeddings directly (with adaptive padding)
phrase_embs = []
for p in phrases:
    st, et = p["start_sec"], p["end_sec"]
    dur = et - st
    # Center padding to minimum 1.5s for ECAPA stability
    if dur < 1.5:
        pad = (1.5 - dur) / 2.0
        emb = service.embed_segment(wav_path, max(0.0, st - pad), et + pad)
    else:
        emb = service.embed_segment(wav_path, st, et)
    phrase_embs.append(emb)

phrase_norms = [np.linalg.norm(e) for e in phrase_embs]
valid_phrase_idx = [i for i, n in enumerate(phrase_norms) if n > 1e-6]
P_X = np.array([phrase_embs[i] / phrase_norms[i] for i in valid_phrase_idx])

# 3. Inspect Phrase-Level Cosine Distance Matrix
P_dist = squareform(pdist(P_X, metric="cosine"))
P_sim = 1.0 - P_dist

print("\n--- Phrase vs Phrase Cosine Similarity (Phrases 0..4) ---")
for i in range(min(5, len(phrases))):
    print(f"Phrase {i:2d} [{phrases[i]['start_sec']:5.2f}-{phrases[i]['end_sec']:5.2f}] \"{phrases[i]['text']}\"")
    row_sims = [f"{P_sim[i, j]:.2f}" for j in range(min(5, len(phrases)))]
    print(f"  Sims: {' '.join(row_sims)}")

# 4. Compare with Window-Level clustering and centroids
speech_intervals = [(0.0, 10.0), (19.4, 28.2), (29.9, 37.5), (39.0, 45.5), (48.0, 56.0)]
diar_segs, diag = service.diarize_audio(
    wav_path=wav_path,
    speech_intervals=speech_intervals,
    transcript_words=words,
    win_len=1.5,
    win_hop=0.75,
)

print("\n--- Mapped Diarization Results (win_len=1.5s, win_hop=0.75s) ---")
for i, s in enumerate(diar_segs):
    print(f"[{i:2d}] [{s['start_sec']:5.2f}s - {s['end_sec']:5.2f}s] {s['speaker_label']:10s} | {s['text']}")
