import numpy as np
from pathlib import Path
from services.speaker_embedding_service import SpeakerEmbeddingService
from database.sqlite_db import SQLiteRepository
from scipy.spatial.distance import pdist, squareform

repo = SQLiteRepository()
audio_id = "8d1b1162-756e-4338-878a-9a4827e2acfa"
wav_path = Path(f"data/audio/{audio_id}.wav")

service = SpeakerEmbeddingService()

# Let's inspect the first 6 phrases individually:
# Phrase 0: [0.00 - 0.34s] "What are you doing?" -> Lisbon (Female voice)
# Phrase 1: [0.54 - 1.50s] "I'm following the conversation." -> Jane (Male voice)
# Phrase 2: [1.94 - 4.42s] "The G-Man says that I don't have adequate security clearance." -> Jane (Male voice)
# Phrase 3: [4.64 - 5.40s] "Are you reading their lips?" -> Lisbon (Female voice)
# Phrase 4: [5.70 - 7.60s] "Yeah, I would be if I didn't have to answer any questions." -> Jane (Male voice)
# Phrase 5: [7.94 - 8.48s] "Oh, okay, boss." -> Lisbon/Jane

test_spans = [
    (0.00, 0.50, "What are you doing? (Lisbon)"),
    (0.54, 1.80, "I'm following the conversation. (Jane)"),
    (1.94, 4.42, "The G-Man says that I don't have... (Jane)"),
    (4.64, 5.50, "Are you reading their lips? (Lisbon)"),
    (5.70, 7.60, "Yeah, I would be if I didn't have to... (Jane)"),
]

print("--- Direct Embedding Extraction with Exact Word Boundaries ---")
embs = []
for st, et, label in test_spans:
    dur = et - st
    # Extract with precise temporal boundary vs padded
    pad = max(0.0, (1.2 - dur) / 2.0)
    emb = service.embed_segment(wav_path, max(0.0, st - pad), et + pad)
    embs.append(emb / np.linalg.norm(emb))

sim_mat = 1.0 - squareform(pdist(embs, metric="cosine"))
print("Pairwise Cosine Similarities (First 5 Dialogue Turns):")
for i in range(len(test_spans)):
    print(f"[{i}] {test_spans[i][2][:35]:35s} | " + " ".join(f"{sim_mat[i, j]:.3f}" for j in range(len(test_spans))))

# Also let's inspect the sliding windows that were generated in [0.0s - 8.0s]:
windows = []
cur = 0.0
while cur + 0.8 <= 8.0:
    windows.append((round(cur, 2), round(cur + 2.0, 2)))
    cur += 1.0

print("\n--- Sliding Windows in [0.0 - 8.0s] ---")
win_embs = service.embed_segments(wav_path, windows)
for w, e in zip(windows, win_embs):
    norm = np.linalg.norm(e)
    # Check similarity against Lisbon (Span 0) and Jane (Span 2)
    e_norm = e / norm if norm > 1e-6 else e
    sim_lisbon = float(np.dot(e_norm, embs[0]))
    sim_jane = float(np.dot(e_norm, embs[2]))
    print(f"Window [{w[0]:4.1f} - {w[1]:4.1f}s]: sim(Lisbon)={sim_lisbon:6.3f} | sim(Jane)={sim_jane:6.3f}")
