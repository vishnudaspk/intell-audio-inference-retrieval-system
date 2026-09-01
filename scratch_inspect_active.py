import numpy as np
from pathlib import Path
from services.speaker_embedding_service import SpeakerEmbeddingService
from database.sqlite_db import SQLiteRepository
from scipy.spatial.distance import pdist, squareform
from collections import Counter

repo = SQLiteRepository()
audio_id = "8d1b1162-756e-4338-878a-9a4827e2acfa"
wav_path = Path(f"data/audio/{audio_id}.wav")

with repo._get_connection() as conn:
    c = conn.cursor()
    c.execute("SELECT start_time, end_time, word FROM transcript_words WHERE audio_id = ? ORDER BY start_time", (audio_id,))
    words = c.fetchall()

print(f"Total words: {len(words)}")
service = SpeakerEmbeddingService()

# 1. Direct dialogue phrase extraction
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

print(f"\nTotal phrases: {len(phrases)}")

# 2. Extract phrase-level embeddings
phrase_embs = []
for p in phrases:
    st, et = p["start_sec"], p["end_sec"]
    dur = et - st
    pad = max(0.0, (1.5 - dur) / 2.0)
    emb = service.embed_segment(wav_path, max(0.0, st - pad), et + pad)
    phrase_embs.append(emb)

phrase_norms = [np.linalg.norm(e) for e in phrase_embs]
valid_idx = [i for i, n in enumerate(phrase_norms) if n > 1e-6]
P_X = np.array([phrase_embs[i] / phrase_norms[i] for i in valid_idx])

# Run current diarize_audio
speech_intervals = [(0.0, 58.0)]
diar_segs, diag = service.diarize_audio(
    wav_path=wav_path,
    speech_intervals=speech_intervals,
    transcript_words=words,
)

print(f"\nCurrent Diarization Diagnostics: {diag}")
print("\nFirst 10 Dialogue Phrases & Attribution:")
for i in range(min(10, len(diar_segs))):
    s = diar_segs[i]
    print(f"[{i:2d}] [{s['start_sec']:5.2f}s - {s['end_sec']:5.2f}s] {s['speaker_label']:10s} | {s['text']}")
