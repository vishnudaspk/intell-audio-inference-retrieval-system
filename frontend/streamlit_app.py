"""
Streamlit UI for Intell Audio Inference & Retrieval System — Phase 5 & Phase 6 Modernization.
Supports Audio Upload, YouTube Ingestion, Word Alignment, Lexical Search, and Grounded Ask the Audio RAG.
"""

from pathlib import Path

import streamlit as st

from database.sqlite_db import SQLiteRepository
from retrieval.bm25 import BM25Index
from retrieval.hybrid import RetrievalPipeline
from retrieval.lexical import LexicalRetrievalEngine
from retrieval.vector_store import QdrantVectorStore
from services.audio_service import AudioService
from services.embedding_service import LMStudioEmbeddingProvider
from services.health_service import HealthService
from services.llm_service import LMStudioLLMProvider
from services.reasoning_agent import ReasoningAgent
from utils.exceptions import IntellAudioError
from workers.audio_worker import AudioWorker
from workers.indexing_worker import IndexingWorker


def get_services():
    """Instantiate and return application services (stateless/cached)."""
    repo = SQLiteRepository()
    audio_service = AudioService()
    worker = AudioWorker(repository=repo, audio_service=audio_service)
    retrieval_engine = LexicalRetrievalEngine()

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

    return repo, audio_service, worker, retrieval_engine, indexing_worker, retrieval_pipeline, reasoning_agent


def format_timestamp(seconds: float) -> str:
    """Format seconds float into MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    st.set_page_config(
        page_title="VANS AUDIO SYSTEM — Temporal Audio Intelligence",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("VANS AUDIO SYSTEM")
    st.caption("Temporal Audio Intelligence & Grounded RAG Platform — Phase 6 Modernization")

    repo, audio_service, worker, retrieval_engine, indexing_worker, retrieval_pipeline, reasoning_agent = get_services()

    # Session State Initialization
    if "current_audio_id" not in st.session_state:
        st.session_state.current_audio_id = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "seek_bytes" not in st.session_state:
        st.session_state.seek_bytes = None
    if "rag_response" not in st.session_state:
        st.session_state.rag_response = None

    # Sidebar System Health Status
    with st.sidebar:
        st.subheader("System Health")
        health = HealthService.check_health()
        st.write(f"**Overall Status:** `{health['status']}`")
        st.write(f"**ASR Engine:** `{health['asr_engine']}`")

        st.write(f"**VAD Engine:** `{health.get('vad_engine', 'silero')}`")
        st.write(f"**Speaker Model:** `{health.get('speaker_embedding_engine', 'speechbrain')}`")
        st.caption("ℹ️ *Legacy Dev UI — Primary UI is React/Vite*")

        lm_st = health.get("lm_studio", "unavailable")
        if lm_st == "available":
            st.success("LM Studio Local API: Available")
        else:
            st.warning("LM Studio: Offline / Unreachable (http://localhost:1234)")

        qd_st = health.get("qdrant", "unavailable")
        if qd_st == "available":
            st.success("Qdrant Vector Store: Available")
        else:
            st.warning("Qdrant Server: Offline (http://localhost:6333)")

    # 1. Select Audio Source
    audio_source = st.radio(
        "Select audio source",
        ("Audio Upload", "YouTube Link"),
    )

    if audio_source == "Audio Upload":
        uploaded_file = st.file_uploader("Upload Audio", type=["mp3", "wav", "m4a", "flac"])

        if st.button("Process Audio"):
            if uploaded_file is None:
                st.warning("Please upload an audio file first.")
            else:
                try:
                    with st.spinner("Processing audio & running forced alignment..."):
                        asset = audio_service.save_uploaded_file(
                            uploaded_file.read(),
                            uploaded_file.name,
                        )
                        st.session_state.current_audio_id = asset.id
                        st.session_state.search_results = []
                        st.session_state.seek_bytes = None
                        st.session_state.rag_response = None

                        worker.process_asset(asset)

                    with st.spinner("Generating temporal chunks & Qwen3 embeddings..."):
                        try:
                            idx_status = indexing_worker.index_audio(asset.id)
                            st.success(
                                f"Audio processed & indexed! (Duration: {asset.duration:.1f}s, Chunks: {idx_status.total_chunks})"
                            )
                        except Exception as idx_err:
                            st.warning(f"Audio processed, but indexing failed/degraded: {idx_err}")

                except IntellAudioError as exc:
                    st.error(f"Audio processing error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    else:
        youtube_link = st.text_input("Enter YouTube link")

        if st.button("Process Audio"):
            if not youtube_link.strip():
                st.warning("Please enter a YouTube link first.")
            else:
                try:
                    with st.spinner("Downloading YouTube audio & processing..."):
                        asset = audio_service.download_youtube_audio(youtube_link)
                        st.session_state.current_audio_id = asset.id
                        st.session_state.search_results = []
                        st.session_state.seek_bytes = None
                        st.session_state.rag_response = None

                        worker.process_asset(asset)

                    with st.spinner("Generating temporal chunks & Qwen3 embeddings..."):
                        try:
                            idx_status = indexing_worker.index_audio(asset.id)
                            st.success(
                                f"YouTube audio processed & indexed! (Duration: {asset.duration:.1f}s, Chunks: {idx_status.total_chunks})"
                            )
                        except Exception as idx_err:
                            st.warning(f"YouTube audio processed, but indexing degraded: {idx_err}")

                except IntellAudioError as exc:
                    st.error(f"YouTube processing error: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected error: {exc}")

    # Display processing analysis UI when an audio asset is ready
    audio_id = st.session_state.current_audio_id
    if not audio_id:
        return

    asset = repo.get_audio_asset(audio_id)
    transcript = repo.get_transcript(audio_id)

    if not asset or not transcript:
        return

    st.markdown("---")

    # 2. Transcript Display
    st.subheader("Transcript")
    with st.expander("View Full Transcript Text", expanded=False):
        st.write(transcript.text if transcript.text else "*(No transcript text generated)*")

    # 3. Audio Player
    st.subheader("Audio Playback")
    if Path(asset.file_path).exists():
        with open(asset.file_path, "rb") as f:
            st.audio(f.read(), format=f"audio/{asset.format}")

    # Seeked Preview Player
    if st.session_state.seek_bytes:
        st.markdown("**Seeked Audio Preview:**")
        st.audio(st.session_state.seek_bytes, format="audio/wav")

    st.markdown("---")

    # 4. ASK THE AUDIO (PHASE 6 GROUNDED RAG)
    st.subheader("🤖 ASK THE AUDIO")
    st.caption("Ask natural language questions grounded strictly in the processed audio transcript.")

    rag_question = st.text_input("Ask a question about this audio...", key="rag_input_field")

    if st.button("Ask Audio", key="ask_button"):
        if not rag_question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Retrieving evidence & generating grounded answer..."):
                    # 1. Deterministic Hybrid Retrieval
                    retrieved_chunks = retrieval_pipeline.search(
                        query=rag_question,
                        audio_id=audio_id,
                    )

                    # 2. Grounded Reasoning & Citation Resolution
                    rag_res = reasoning_agent.answer_question(
                        query=rag_question,
                        retrieved_chunks=retrieved_chunks,
                        audio_id=audio_id,
                    )
                    st.session_state.rag_response = rag_res

            except Exception as exc:
                st.error(f"Ask the Audio failed: {exc}")

    rag_resp = st.session_state.rag_response
    if rag_resp:
        with st.container(border=True):
            st.markdown(f"**Answer:**")
            st.write(rag_resp.answer)

            if rag_resp.grounded:
                st.caption(f"✓ Grounded Answer (Confidence: {rag_resp.confidence:.2f} | Model: {rag_resp.model})")
            else:
                st.warning("⚠️ Low confidence / Insufficient evidence")

            # Display Timestamp Citations
            if rag_resp.citations:
                st.markdown("**Sources & Timestamps:**")
                cols = st.columns(min(4, max(1, len(rag_resp.citations))))
                for idx, cit in enumerate(rag_resp.citations):
                    col_idx = idx % len(cols)
                    label = f"[{format_timestamp(cit.start_time)} – {format_timestamp(cit.end_time)}]"
                    if cols[col_idx].button(label, key=f"cit_btn_{idx}"):
                        try:
                            wav_path = audio_service.audio_dir / f"{audio_id}.wav"
                            seek_file = wav_path if wav_path.exists() else Path(asset.file_path)
                            seek_bytes = audio_service.extract_audio_preview(seek_file, cit.start_time)
                            st.session_state.seek_bytes = seek_bytes
                            st.rerun()
                        except Exception as seek_err:
                            st.error(f"Could not seek to timestamp: {seek_err}")

            # Expandable Retrieved Evidence Section
            with st.expander("Show Retrieved Evidence Chunks", expanded=False):
                if not rag_resp.retrieved_chunks:
                    st.write("*(No candidate evidence chunks retrieved)*")
                else:
                    for cand in rag_resp.retrieved_chunks:
                        st.markdown(
                            f"**Chunk `{cand.chunk.chunk_id}`** "
                            f"(Rank #{cand.rank} | Source: `{cand.retrieval_source}` | Score: `{cand.score:.3f}`)  \n"
                            f"Time: `{format_timestamp(cand.start_time)} – {format_timestamp(cand.end_time)}`  \n"
                            f"Text: *\"{cand.chunk.text}\"*"
                        )
                        st.markdown("---")

    st.markdown("---")

    # 5. Word / Phrase Search (Phase 1 Baseline)
    st.subheader("Exact Word / Phrase Search")
    search_input = st.text_input("Enter word or phrase to search", key="lexical_input_field")

    if st.button("Search Exact Words", key="lexical_search_button"):
        if not search_input.strip():
            st.warning("Please enter a word or phrase to search.")
            st.session_state.search_results = []
        else:
            words = repo.get_alignment_words(audio_id)
            results = retrieval_engine.search(words, search_input)
            st.session_state.search_results = results

            if not results:
                st.warning("No results found.")
            else:
                st.success(f"Found {len(results)} result(s).")

    search_results = st.session_state.search_results

    if search_results:
        st.write("Search results:")
        for idx, res in enumerate(search_results):
            st.markdown(f"**Option {idx + 1}:** '{res.matched_text}' found at `{res.start:.2f}s`")

        selected_option = st.number_input(
            "Enter option to seek",
            min_value=1,
            max_value=len(search_results),
            step=1,
            value=1,
            key="seek_option",
        )

        if st.button("Seek Option", key="seek_lexical_button"):
            selected_res = search_results[selected_option - 1]
            try:
                wav_path = audio_service.audio_dir / f"{audio_id}.wav"
                seek_file = wav_path if wav_path.exists() else Path(asset.file_path)

                seek_bytes = audio_service.extract_audio_preview(seek_file, selected_res.start)
                st.session_state.seek_bytes = seek_bytes
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to seek audio: {exc}")

    # 6. Reset / Done Action
    if st.button("Reset Session", key="reset_button"):
        st.session_state.current_audio_id = None
        st.session_state.search_results = []
        st.session_state.seek_bytes = None
        st.session_state.rag_response = None
        st.success("Session reset successfully!")
        st.rerun()


if __name__ == "__main__":
    main()
