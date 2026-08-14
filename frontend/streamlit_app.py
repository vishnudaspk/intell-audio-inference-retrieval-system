"""
Refactored Streamlit UI for Intell Audio Inference & Retrieval System.
Zero business logic — delegates all operations to application services.
"""

from pathlib import Path

import streamlit as st

from database.sqlite_db import SQLiteRepository
from retrieval.lexical import LexicalRetrievalEngine
from services.audio_service import AudioService
from services.health_service import HealthService
from utils.exceptions import IntellAudioError
from workers.audio_worker import AudioWorker


def get_services():
    """Instantiate and return application services (stateless/cached)."""
    repo = SQLiteRepository()
    audio_service = AudioService()
    worker = AudioWorker(repository=repo, audio_service=audio_service)
    retrieval_engine = LexicalRetrievalEngine()
    return repo, audio_service, worker, retrieval_engine


def main():
    st.set_page_config(
        page_title="VANS AUDIO SYSTEM — Temporal Audio Retrieval",
        page_icon="🎙️",
        layout="centered",
    )

    st.title("VANS AUDIO SYSTEM")
    st.caption("Temporal Audio Intelligence & Retrieval Platform — Phase 1 Baseline")

    repo, audio_service, worker, retrieval_engine = get_services()

    # Session State Initialization
    if "current_audio_id" not in st.session_state:
        st.session_state.current_audio_id = None
    if "search_results" not in st.session_state:
        st.session_state.search_results = []
    if "seek_bytes" not in st.session_state:
        st.session_state.seek_bytes = None

    # Sidebar System Health Status
    with st.sidebar:
        st.subheader("System Status")
        health = HealthService.check_health()
        st.write(f"**Status:** `{health['status']}`")
        st.write(f"**ASR Engine:** `{health['asr_engine']}`")

        gentle_st = health["services"]["gentle"]
        if gentle_st == "available":
            st.success("Gentle Server: Available")
        else:
            st.warning("Gentle Server: Unavailable (Forced alignment disabled)")

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
                    with st.spinner("Processing audio asset..."):
                        asset = audio_service.save_uploaded_file(
                            uploaded_file.read(),
                            uploaded_file.name,
                        )
                        st.session_state.current_audio_id = asset.id
                        st.session_state.search_results = []
                        st.session_state.seek_bytes = None

                        worker.process_asset(asset)
                        st.success(f"Audio processed successfully! (Duration: {asset.duration:.1f}s)")
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
                    with st.spinner("Downloading YouTube audio and processing..."):
                        asset = audio_service.download_youtube_audio(youtube_link)
                        st.session_state.current_audio_id = asset.id
                        st.session_state.search_results = []
                        st.session_state.seek_bytes = None

                        worker.process_asset(asset)
                        st.success(f"YouTube audio processed successfully! (Duration: {asset.duration:.1f}s)")
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

    # 4. Word / Phrase Search
    st.subheader("Transcript Search")
    search_input = st.text_input("Enter word or phrase to search")

    if st.button("Search"):
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

        # 5. Timestamp Seek Selection
        selected_option = st.number_input(
            "Enter the option to seek",
            min_value=1,
            max_value=len(search_results),
            step=1,
            value=1,
            key="seek_option",
        )

        if st.button("Seek"):
            selected_res = search_results[selected_option - 1]
            try:
                # Use WAV file for slicing
                wav_path = audio_service.audio_dir / f"{audio_id}.wav"
                seek_file = wav_path if wav_path.exists() else Path(asset.file_path)

                seek_bytes = audio_service.extract_audio_preview(seek_file, selected_res.start)
                st.session_state.seek_bytes = seek_bytes
                st.rerun()
            except Exception as exc:
                st.error(f"Unable to seek audio: {exc}")

    # 6. Reset / Done Action
    if st.button("Done"):
        st.session_state.current_audio_id = None
        st.session_state.search_results = []
        st.session_state.seek_bytes = None
        st.success("Session reset successfully!")
        st.rerun()


if __name__ == "__main__":
    main()
