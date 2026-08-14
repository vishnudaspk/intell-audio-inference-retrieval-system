"""
Streamlit UI for Intell Audio Inference & Retrieval System - Phase 7B Modernization.
Supports Audio Upload, YouTube Ingestion, Automatic Chapters, Grounded Ask the Audio RAG,
and Instant Audio Seeking to Exact Temporal Locations.
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
from services.query_understanding import QueryUnderstanding
from services.reasoning_agent import ReasoningAgent
from services.temporal_context_expander import TemporalContextExpander
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

    query_understanding = QueryUnderstanding(llm_provider=llm_provider)
    context_expander = TemporalContextExpander()

    retrieval_pipeline = RetrievalPipeline(
        bm25_index=bm25_index,
        vector_store=vector_store,
        embedding_provider=embedding_provider,
        repository=repo,
        query_understanding=query_understanding,
        context_expander=context_expander,
    )

    reasoning_agent = ReasoningAgent(
        llm_provider=llm_provider,
        repository=repo,
    )

    return (
        repo,
        audio_service,
        worker,
        retrieval_engine,
        indexing_worker,
        retrieval_pipeline,
        reasoning_agent,
    )


def format_timestamp(seconds: float) -> str:
    """Format seconds float into MM:SS format."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def main():
    st.set_page_config(
        page_title="VANS AUDIO SYSTEM - Temporal Audio Intelligence",
        page_icon="🎙️",
        layout="wide",
    )

    st.title("🎙️ VANS AUDIO SYSTEM")
    st.caption("Temporal Audio Intelligence, Semantic Chapters & Intent-Aware Grounded RAG")

    (
        repo,
        audio_service,
        worker,
        retrieval_engine,
        indexing_worker,
        retrieval_pipeline,
        reasoning_agent,
    ) = get_services()

    # Session State Initialization
    if "current_audio_id" not in st.session_state:
        st.session_state.current_audio_id = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "seek_bytes" not in st.session_state:
        st.session_state.seek_bytes = None
    if "rag_response" not in st.session_state:
        st.session_state.rag_response = None
    if "seek_label" not in st.session_state:
        st.session_state.seek_label = ""

    audio_id = st.session_state.current_audio_id

    # Sidebar
    with st.sidebar:
        st.subheader("System Health")
        health = HealthService.check_health()
        st.write(f"**Status:** `{health['status']}`")
        st.write(f"**ASR:** `{health['asr_engine']}`")

        lm_st = health.get("lm_studio", "unavailable")
        if lm_st == "available":
            st.success("LM Studio: Online (Qwen3-8B)")
        else:
            st.warning("LM Studio: Offline (http://localhost:1234)")

        qd_st = health.get("qdrant", "unavailable")
        if qd_st == "available":
            st.success("Qdrant: Connected")
        else:
            st.warning("Qdrant: Offline (http://localhost:6333)")

        # Chapters Section in Sidebar
        if audio_id:
            st.markdown("---")
            st.subheader("📑 Chapters")
            try:
                chapters = repo.get_chapters(audio_id)
                if chapters:
                    for idx, chap in enumerate(chapters):
                        btn_label = f"▶ {chap.title} ({format_timestamp(chap.start_time)} – {format_timestamp(chap.end_time)})"
                        if st.button(btn_label, key=f"sidebar_chap_{idx}", use_container_width=True):
                            try:
                                asset = repo.get_audio_asset(audio_id)
                                wav_path = audio_service.audio_dir / f"{audio_id}.wav"
                                seek_file = wav_path if wav_path.exists() else Path(asset.file_path)
                                st.session_state.seek_bytes = audio_service.extract_audio_preview(seek_file, chap.start_time)
                                st.session_state.seek_label = f"Chapter: {chap.title} ({format_timestamp(chap.start_time)})"
                                st.rerun()
                            except Exception as e:
                                st.error(f"Could not seek chapter: {e}")
                else:
                    st.caption("No chapters generated for this audio.")
            except Exception as exc:
                st.caption(f"Chapters unavailable: {exc}")

            # Speaker Turns Info
            try:
                segments = repo.get_speaker_segments(audio_id)
                if segments:
                    st.markdown("---")
                    st.subheader("👥 Speaker Turns")
                    st.caption(f"Detected {len(segments)} turn boundaries (Heuristic Energy/Silence)")
            except Exception:
                pass

    # 1. Ingestion Section
    col_input, col_info = st.columns([2, 1])

    with col_input:
        audio_source = st.radio(
            "Select Audio Source",
            ("Audio Upload", "YouTube Link"),
            horizontal=True,
        )

        if audio_source == "Audio Upload":
            uploaded_file = st.file_uploader("Upload Audio File", type=["mp3", "wav", "m4a", "flac"])
            if st.button("Process & Index Audio", type="primary"):
                if uploaded_file is None:
                    st.warning("Please upload an audio file first.")
                else:
                    try:
                        with st.spinner("Processing audio, turn segmentation & alignment..."):
                            asset = audio_service.save_uploaded_file(
                                uploaded_file.read(),
                                uploaded_file.name,
                            )
                            st.session_state.current_audio_id = asset.id
                            st.session_state.search_results = []
                            st.session_state.seek_bytes = None
                            st.session_state.rag_response = None
                            st.session_state.seek_label = ""
                            worker.process_asset(asset)

                        with st.spinner("Generating semantic metadata, embeddings & chapters..."):
                            try:
                                idx_status = indexing_worker.index_audio(asset.id)
                                st.success(
                                    f"Audio indexed! ({asset.duration:.1f}s | {idx_status.total_chunks} chunks)"
                                )
                            except Exception as idx_err:
                                st.warning(f"Audio processed, indexing degraded: {idx_err}")
                        st.rerun()

                    except IntellAudioError as exc:
                        st.error(f"Processing error: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")

        else:
            youtube_link = st.text_input("Enter YouTube URL")
            if st.button("Download, Process & Index", type="primary"):
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
                            st.session_state.seek_label = ""
                            worker.process_asset(asset)

                        with st.spinner("Generating semantic metadata, embeddings & chapters..."):
                            try:
                                idx_status = indexing_worker.index_audio(asset.id)
                                st.success(
                                    f"YouTube audio indexed! ({asset.duration:.1f}s | {idx_status.total_chunks} chunks)"
                                )
                            except Exception as idx_err:
                                st.warning(f"Audio processed, indexing degraded: {idx_err}")
                        st.rerun()

                    except IntellAudioError as exc:
                        st.error(f"YouTube processing error: {exc}")
                    except Exception as exc:
                        st.error(f"Unexpected error: {exc}")

    if not audio_id:
        return

    asset = repo.get_audio_asset(audio_id)
    transcript = repo.get_transcript(audio_id)

    if not asset or not transcript:
        return

    with col_info:
        st.markdown("**Active Asset:**")
        st.write(f"• **Filename:** `{asset.filename}`")
        st.write(f"• **Duration:** `{asset.duration:.1f}s` (`{format_timestamp(asset.duration)}`)")
        st.write(f"• **Format:** `{asset.format}`")

    st.markdown("---")

    # 2. Main Audio Player & Seek Preview
    col_player, col_transcript = st.columns([1, 1])

    with col_player:
        st.subheader("🎧 Full Audio Playback")
        if Path(asset.file_path).exists():
            with open(asset.file_path, "rb") as f:
                st.audio(f.read(), format=f"audio/{asset.format}")

        if st.session_state.seek_bytes:
            st.markdown(f"**🎯 Exact Seek Preview ({st.session_state.seek_label}):**")
            st.audio(st.session_state.seek_bytes, format="audio/wav")

    with col_transcript:
        st.subheader("📝 Transcript")
        with st.expander("View Full Transcript", expanded=False):
            st.write(transcript.text if transcript.text else "*(No transcript text generated)*")

    st.markdown("---")

    # 3. PRIMARY INTERACTION: ASK THE AUDIO (PHASE 7B INTENT-AWARE RAG)
    st.subheader("🤖 Ask the Audio")
    st.caption("Ask natural-language questions (e.g. 'Which bolt do I need to unscrew to remove the turbo?'). The system understands intent, actions, and objects to retrieve the exact procedure and audio timestamp.")

    rag_question = st.text_input(
        "Ask a question about this audio...",
        key="rag_input_field",
        placeholder="e.g. Which bolt do I need to unscrew to remove the turbo?",
    )

    if st.button("Ask Audio", key="ask_button", type="primary"):
        if not rag_question.strip():
            st.warning("Please enter a question first.")
        else:
            try:
                with st.spinner("Extracting intent, retrieving evidence & synthesizing grounded answer..."):
                    # 1. Deterministic Hybrid Retrieval with Query Understanding
                    retrieved_chunks = retrieval_pipeline.search(
                        query=rag_question,
                        audio_id=audio_id,
                    )

                    # Extract query_intent if available
                    query_intent = None
                    if retrieved_chunks and "query_intent" in retrieved_chunks[0].metadata:
                        from schemas.models import QueryIntent
                        try:
                            query_intent = QueryIntent(**retrieved_chunks[0].metadata["query_intent"])
                        except Exception:
                            pass

                    # 2. Grounded Reasoning & Citation Resolution
                    rag_res = reasoning_agent.answer_question(
                        query=rag_question,
                        retrieved_chunks=retrieved_chunks,
                        audio_id=audio_id,
                        query_intent=query_intent,
                    )
                    st.session_state.rag_response = rag_res

            except Exception as exc:
                st.error(f"Ask the Audio failed: {exc}")

    rag_resp = st.session_state.rag_response
    if rag_resp:
        with st.container(border=True):
            st.markdown("### 💡 Grounded Answer")
            st.markdown(rag_resp.answer)

            # Prominent Jump-to button for primary timestamp
            if rag_resp.primary_timestamp:
                p_ts = rag_resp.primary_timestamp
                jump_label = f"▶ Jump to Exact Moment: {format_timestamp(p_ts.start_time)} – {format_timestamp(p_ts.end_time)}"
                if st.button(jump_label, key="btn_primary_jump", type="primary"):
                    try:
                        wav_path = audio_service.audio_dir / f"{audio_id}.wav"
                        seek_file = wav_path if wav_path.exists() else Path(asset.file_path)
                        seek_bytes = audio_service.extract_audio_preview(seek_file, p_ts.start_time)
                        st.session_state.seek_bytes = seek_bytes
                        st.session_state.seek_label = f"Primary Result: {format_timestamp(p_ts.start_time)}"
                        st.rerun()
                    except Exception as seek_err:
                        st.error(f"Could not seek to timestamp: {seek_err}")

            # Badges / metadata row
            badge_cols = st.columns(4)
            with badge_cols[0]:
                if rag_resp.grounded:
                    st.success(f"✓ Grounded ({rag_resp.confidence:.2f})")
                else:
                    st.warning("⚠️ Low Confidence")
            with badge_cols[1]:
                if rag_resp.intent:
                    st.info(f"Intent: `{rag_resp.intent}`")
            with badge_cols[2]:
                if rag_resp.chapter:
                    st.info(f"Chapter: `{rag_resp.chapter}`")
            with badge_cols[3]:
                if rag_resp.speaker:
                    st.info(f"Speaker: `{rag_resp.speaker}`")

            # Related sections & citations
            if rag_resp.citations:
                st.markdown("**Citations & Time Spans:**")
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
                            st.session_state.seek_label = f"Citation #{idx+1} ({format_timestamp(cit.start_time)})"
                            st.rerun()
                        except Exception as seek_err:
                            st.error(f"Could not seek to timestamp: {seek_err}")

            # Expandable Evidence Chunks with Semantic Tags
            with st.expander("🔍 Show Retrieved Evidence Chunks & Semantic Metadata", expanded=False):
                if not rag_resp.retrieved_chunks:
                    st.write("*(No candidate evidence chunks retrieved)*")
                else:
                    for cand in rag_resp.retrieved_chunks:
                        chunk = cand.chunk
                        st.markdown(
                            f"**Chunk `{chunk.chunk_id}`** "
                            f"(Rank #{cand.rank} | Source: `{cand.retrieval_source}` | Score: `{cand.score:.3f}`)  \n"
                            f"Time: `{format_timestamp(cand.start_time)} – {format_timestamp(cand.end_time)}`"
                        )
                        if chunk.content_type:
                            st.caption(f"Content Type: `{chunk.content_type}` | Topic: `{chunk.topic or 'N/A'}`")
                        if chunk.actions or chunk.objects or chunk.targets:
                            st.caption(f"Actions: `{chunk.actions}` | Objects: `{chunk.objects}` | Targets: `{chunk.targets}`")
                        st.write(f"*\"{chunk.text}\"*")
                        st.markdown("---")

    st.markdown("---")

    # 4. Advanced / Exact Lexical Search (Baseline Feature)
    with st.expander("🔎 Advanced: Exact Word / Phrase Search", expanded=False):
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
                    st.session_state.seek_bytes = audio_service.extract_audio_preview(seek_file, selected_res.start)
                    st.session_state.seek_label = f"Exact Match: '{selected_res.matched_text}' ({selected_res.start:.2f}s)"
                    st.rerun()
                except Exception as exc:
                    st.error(f"Unable to seek audio: {exc}")

    # 5. Reset Session
    if st.button("Reset Session", key="reset_button"):
        st.session_state.current_audio_id = None
        st.session_state.search_results = []
        st.session_state.seek_bytes = None
        st.session_state.rag_response = None
        st.session_state.seek_label = ""
        st.success("Session reset successfully!")
        st.rerun()


if __name__ == "__main__":
    main()
