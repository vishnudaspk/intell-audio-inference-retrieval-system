import json
import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings
from database.sqlite_db import SQLiteRepository
from retrieval.bm25 import BM25Index
from retrieval.hybrid import RetrievalPipeline
from retrieval.lexical import LexicalRetrievalEngine
from retrieval.vector_store import QdrantVectorStore
from schemas.models import SourceType
from services.audio_service import AudioService
from services.embedding_service import LMStudioEmbeddingProvider
from services.llm_service import LMStudioLLMProvider
from services.reasoning_agent import ReasoningAgent
from workers.audio_worker import AudioWorker
from workers.indexing_worker import IndexingWorker


def main():
    print("=" * 70)
    print("INTELL AUDIO V3 — MILESTONE VALIDATION & BENCHMARK")
    print("=" * 70)

    # 1. Initialize services
    repo = SQLiteRepository()
    audio_service = AudioService()
    worker = AudioWorker(repository=repo, audio_service=audio_service)

    bm25_index = BM25Index()
    vector_store = QdrantVectorStore()
    embedding_provider = LMStudioEmbeddingProvider()
    llm_provider = LMStudioLLMProvider()

    indexing_worker = IndexingWorker(
        repository=repo,
        bm25_index=bm25_index,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
    )

    retrieval_pipeline = RetrievalPipeline(
        bm25_index=bm25_index,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        repository=repo,
    )

    reasoning_agent = ReasoningAgent(
        llm_provider=llm_provider,
        repository=repo,
    )

    lexical_engine = LexicalRetrievalEngine()

    # Find sample audio
    candidate_files = [
        Path("data/audio/sample 4.mp3"),
        Path("data/raw/sample 4.mp3"),
        Path("data/audio/sample2.mp3"),
    ]
    sample_file = None
    for cf in candidate_files:
        if cf.exists():
            sample_file = cf
            break
    if not sample_file:
        # search anywhere in data/
        for p in Path("data").glob("**/*.mp3"):
            sample_file = p
            break

    print(f"Using audio input: {sample_file}")
    if not sample_file:
        raise RuntimeError("No sample audio file found in data directory!")

    # 2. Ingest & Process Asset
    with open(sample_file, "rb") as f:
        content = f.read()

    asset = audio_service.save_uploaded_file(content, sample_file.name)
    print(f"Created AudioAsset id={asset.id}, format={asset.format}")

    t0_e2e = time.time()
    job = worker.process_asset(asset)
    t_pipeline = time.time() - t0_e2e

    print("\n--- Pipeline Timings ---")
    print(f"Status: {job.status.value}")
    print(f"Duration: {asset.duration:.2f}s")
    for stage, dur in job.timings.items():
        print(f"  {stage}: {dur:.3f}s")
    print(f"Pipeline Total Time: {t_pipeline:.3f}s")
    print(f"Realtime Factor (Pipeline): {t_pipeline / max(asset.duration, 0.001):.3f}x")

    # Verify speaker embeddings generated
    segments = repo.get_audio_segments(asset.id)
    print(f"\nTotal Assembled AudioSegments: {len(segments)}")
    for i, s in enumerate(segments):
        emb_dim = len(s.speaker_embedding) if s.speaker_embedding else 0
        f0 = s.acoustic_features.get("f0_mean") if s.acoustic_features else None
        print(f"  Seg {i} [{s.start_sec:.2f}–{s.end_sec:.2f}s]: words={len(s.words)}, emb_dim={emb_dim}, f0={f0}")

    # 3. Indexing
    t0_idx = time.time()
    idx_res = indexing_worker.index_audio(asset.id)
    t_index = time.time() - t0_idx
    print(f"\n--- Indexing ---")
    print(f"Chunks indexed: {idx_res.indexed_chunks}, Indexing time: {t_index:.3f}s")

    total_e2e = t_pipeline + t_index
    print(f"Total E2E (Processing + Indexing): {total_e2e:.3f}s")
    print(f"Overall Realtime Factor: {total_e2e / max(asset.duration, 0.001):.3f}x")

    # 4. Retrieval Queries
    print("\n" + "=" * 70)
    print("RETRIEVAL QUALITY TESTS")
    print("=" * 70)

    # Test 1: Direct Factual Question
    q1 = "What trend or behavior was mentioned regarding earphones?"
    print(f"\n[Test 1 - Direct Factual Question]: '{q1}'")
    t0 = time.time()
    res1_retrieval = retrieval_pipeline.search(q1, audio_id=asset.id, top_k=3)
    res1_rag = reasoning_agent.answer_question(q1, res1_retrieval, audio_id=asset.id)
    dur1 = time.time() - t0
    print(f"Answer: {res1_rag.answer}")
    print(f"Grounded: {res1_rag.grounded}, Citations: {len(res1_rag.citations)}, Time: {dur1:.2f}s")
    for c in res1_rag.citations:
        print(f"  Citation: [{c.start_time:.2f}-{c.end_time:.2f}s] \"{c.text}\"")

    # Test 2: Semantic Question
    q2 = "Why are people preferring wired headphones or what nostalgia was expressed?"
    print(f"\n[Test 2 - Semantic Question]: '{q2}'")
    t0 = time.time()
    res2_retrieval = retrieval_pipeline.search(q2, audio_id=asset.id, top_k=3)
    res2_rag = reasoning_agent.answer_question(q2, res2_retrieval, audio_id=asset.id)
    dur2 = time.time() - t0
    print(f"Answer: {res2_rag.answer}")
    print(f"Grounded: {res2_rag.grounded}, Citations: {len(res2_rag.citations)}, Time: {dur2:.2f}s")
    for c in res2_rag.citations:
        print(f"  Citation: [{c.start_time:.2f}-{c.end_time:.2f}s] \"{c.text}\"")

    # Test 3: Lexical Search
    q3 = "wired"
    print(f"\n[Test 3 - Lexical Exact Search]: '{q3}'")
    transcript = repo.get_transcript(asset.id)
    if transcript:
        lex_results = lexical_engine.search(words=transcript.words, query=q3)
        print(f"Lexical occurrences found: {len(lex_results)}")
        for r in lex_results[:3]:
            print(f"  Match: [{r.start:.2f}-{r.end:.2f}s] \"{r.matched_text}\"")

    # Test 4: Unanswerable Hallucination Test
    q4 = "What is the capital of Japan and who is the prime minister?"
    print(f"\n[Test 4 - Unanswerable Question (Hallucination Test)]: '{q4}'")
    t0 = time.time()
    res4_retrieval = retrieval_pipeline.search(q4, audio_id=asset.id, top_k=3)
    res4_rag = reasoning_agent.answer_question(q4, res4_retrieval, audio_id=asset.id)
    dur4 = time.time() - t0
    print(f"Answer: {res4_rag.answer}")
    print(f"Grounded: {res4_rag.grounded}, Citations: {len(res4_rag.citations)}, Time: {dur4:.2f}s")

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
