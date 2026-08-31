# -*- coding: utf-8 -*-
"""
VANS Audio System — 2022 Legacy Compatibility Application

This application preserves the original 2022 workflow:

    Audio / YouTube
        ↓
    PocketSphinx Speech Recognition
        ↓
    Gentle Forced Alignment
        ↓
    Word-level timestamps
        ↓
    Word Search
        ↓
    Timestamp-based Audio Seeking

IMPORTANT:
This is intentionally kept separate from the V3 architecture.
V3 uses FastAPI + React + Whisper + Silero VAD + SpeechBrain, etc.

Legacy runtime dependencies:
    - Streamlit
    - pytube
    - moviepy
    - SpeechRecognition
    - PocketSphinx
    - Gentle Docker service
    - pydub
    - NLTK
"""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path

import requests
import streamlit as st
import speech_recognition as sr

from moviepy.editor import AudioFileClip
from pydub import AudioSegment
from pytube import YouTube

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

LEGACY_DATA_DIR = PROJECT_ROOT / "legacy_data"

AUDIO_PATH = LEGACY_DATA_DIR / "my_audio.mp3"
WAV_PATH = LEGACY_DATA_DIR / "my_audio.wav"
TEXT_PATH = LEGACY_DATA_DIR / "audio.txt"
ALIGNMENT_PATH = LEGACY_DATA_DIR / "alignment_result.csv"

GENTLE_URL = "http://localhost:8888/transcriptions?async=false"

SUPPORTED_UPLOAD_TYPES = ["mp3", "wav"]


# ---------------------------------------------------------------------------
# Directory initialization
# ---------------------------------------------------------------------------

LEGACY_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def file_exists(path: Path) -> bool:
    """Return True if a file exists and has non-zero size."""
    return path.exists() and path.stat().st_size > 0


def clear_legacy_files() -> None:
    """Delete all generated 2022 pipeline artifacts."""

    files = [
        AUDIO_PATH,
        WAV_PATH,
        TEXT_PATH,
        ALIGNMENT_PATH,
    ]

    deleted = []

    for path in files:
        try:
            if path.exists():
                path.unlink()
                deleted.append(path.name)
        except OSError as exc:
            st.warning(f"Could not delete {path.name}: {exc}")

    if deleted:
        st.success(
            "Deleted legacy files: "
            + ", ".join(deleted)
        )
    else:
        st.info("No legacy files to delete.")


def gentle_is_available() -> bool:
    """Check whether the local Gentle server is reachable."""

    try:
        response = requests.get(
            "http://localhost:8888",
            timeout=3,
        )

        return response.status_code < 500

    except requests.RequestException:
        return False


# ---------------------------------------------------------------------------
# YouTube
# ---------------------------------------------------------------------------

def download_audio(youtube_link: str) -> Path | None:
    """
    Download the audio stream from a YouTube URL.

    The resulting file is stored as:
        legacy_data/my_audio.mp3
    """

    if not youtube_link.strip():
        st.error("Please enter a YouTube URL.")
        return None

    try:
        st.info("Connecting to YouTube...")

        video = YouTube(youtube_link)

        audio_streams = (
            video.streams
            .filter(only_audio=True)
            .order_by("abr")
        )

        if not audio_streams:
            st.error("No audio stream was found.")
            return None

        audio_stream = audio_streams.last()

        st.info("Downloading audio...")

        downloaded_path = audio_stream.download(
            output_path=str(LEGACY_DATA_DIR),
            filename="my_audio.mp4",
        )

        downloaded_path = Path(downloaded_path)

        # Convert downloaded audio to MP3 using pydub/ffmpeg.
        st.info("Converting downloaded audio to MP3...")

        audio = AudioSegment.from_file(str(downloaded_path))

        audio.export(
            str(AUDIO_PATH),
            format="mp3",
        )

        # Remove intermediate downloaded file.
        try:
            downloaded_path.unlink()
        except OSError:
            pass

        st.success("YouTube audio downloaded successfully.")

        return AUDIO_PATH

    except Exception as exc:
        st.error(
            f"An error occurred during YouTube download: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Audio normalization
# ---------------------------------------------------------------------------

def convert_to_wav(audio_path: Path) -> Path | None:
    """
    Convert the input audio into a WAV file suitable for
    SpeechRecognition and Gentle.
    """

    try:
        st.info("Converting audio to WAV...")

        # If source is already WAV, still normalize through pydub.
        audio = AudioSegment.from_file(str(audio_path))

        audio = audio.set_channels(1)
        audio = audio.set_frame_rate(16000)

        audio.export(
            str(WAV_PATH),
            format="wav",
        )

        st.success("Audio converted to WAV successfully.")

        return WAV_PATH

    except Exception as exc:
        st.error(
            f"Failed to convert audio to WAV: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Speech recognition
# ---------------------------------------------------------------------------

def transcribe_with_pocketsphinx(wav_path: Path) -> str | None:
    """
    Transcribe audio using the legacy PocketSphinx backend.
    """

    try:
        st.info("Converting audio to text with PocketSphinx...")

        recognizer = sr.Recognizer()

        with sr.AudioFile(str(wav_path)) as source:
            audio_data = recognizer.record(source)

        text = recognizer.recognize_sphinx(audio_data)

        text = text.strip()

        if not text:
            st.warning(
                "PocketSphinx returned an empty transcript."
            )
            return None

        st.success("Audio converted to text successfully.")

        return text

    except sr.UnknownValueError:
        st.error(
            "PocketSphinx could not understand the audio."
        )
        return None

    except sr.RequestError as exc:
        st.error(
            f"PocketSphinx request error: {exc}"
        )
        return None

    except Exception as exc:
        st.error(
            f"Speech recognition failed: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# Save transcript
# ---------------------------------------------------------------------------

def save_transcript(text: str) -> bool:
    """Save the legacy transcript as audio.txt."""

    try:
        TEXT_PATH.write_text(
            text,
            encoding="utf-8",
        )

        st.success(
            f"Transcript saved to {TEXT_PATH.name}."
        )

        return True

    except OSError as exc:
        st.error(
            f"Could not save transcript: {exc}"
        )
        return False


# ---------------------------------------------------------------------------
# Gentle forced alignment
# ---------------------------------------------------------------------------

def perform_gentle_alignment(
    wav_path: Path,
    transcript: str,
) -> bool:
    """
    Send the transcript and WAV audio to Gentle.

    Gentle must be running at:

        http://localhost:8888
    """

    if not gentle_is_available():
        st.error(
            "Gentle is not reachable at "
            "http://localhost:8888"
        )

        st.info(
            "Start Gentle with:\n\n"
            "docker start intell_gentle"
        )

        return False

    try:
        st.info("Performing Gentle forced alignment...")

        with wav_path.open("rb") as audio_file:

            response = requests.post(
                GENTLE_URL,
                files={
                    "audio": (
                        wav_path.name,
                        audio_file,
                        "audio/wav",
                    )
                },
                data={
                    "transcript": transcript,
                },
                timeout=300,
            )

        if response.status_code != 200:
            st.error(
                "Gentle alignment failed.\n\n"
                f"HTTP status: {response.status_code}\n"
                f"Response: {response.text[:500]}"
            )

            return False

        alignment_data = response.json()

        words = alignment_data.get("words", [])

        if not words:
            st.warning(
                "Gentle returned no aligned words."
            )
            return False

        with ALIGNMENT_PATH.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            writer = csv.writer(csv_file)

            writer.writerow(
                ["word", "start", "end"]
            )

            for word in words:

                if "start" in word:
                    start_time = word["start"]

                elif "startOffset" in word:
                    start_time = (
                        word["startOffset"] / 1000
                    )

                else:
                    start_time = ""

                if "end" in word:
                    end_time = word["end"]

                elif "endOffset" in word:
                    end_time = (
                        word["endOffset"] / 1000
                    )

                else:
                    end_time = ""

                writer.writerow(
                    [
                        word.get("word", ""),
                        start_time,
                        end_time,
                    ]
                )

        st.success(
            f"Gentle alignment saved to "
            f"{ALIGNMENT_PATH.name}."
        )

        return True

    except requests.RequestException as exc:
        st.error(
            f"Could not communicate with Gentle: {exc}"
        )
        return False

    except Exception as exc:
        st.error(
            f"Forced alignment failed: {exc}"
        )
        return False


# ---------------------------------------------------------------------------
# Complete processing pipeline
# ---------------------------------------------------------------------------

def process_audio(audio_path: Path) -> bool:
    """
    Execute the complete 2022 pipeline:

        audio
        ↓
        WAV
        ↓
        PocketSphinx
        ↓
        audio.txt
        ↓
        Gentle
        ↓
        alignment_result.csv
    """

    st.divider()
    st.subheader("Processing Audio")

    # ---------------------------------------------------------
    # Step 1 — Convert to WAV
    # ---------------------------------------------------------

    wav_path = convert_to_wav(audio_path)

    if wav_path is None:
        return False

    # ---------------------------------------------------------
    # Step 2 — Speech recognition
    # ---------------------------------------------------------

    transcript = transcribe_with_pocketsphinx(wav_path)

    if transcript is None:
        return False

    # ---------------------------------------------------------
    # Step 3 — Save transcript
    # ---------------------------------------------------------

    if not save_transcript(transcript):
        return False

    # ---------------------------------------------------------
    # Step 4 — Gentle alignment
    # ---------------------------------------------------------

    if not perform_gentle_alignment(
        wav_path,
        transcript,
    ):
        return False

    st.success(
        "2022 audio processing pipeline completed."
    )

    return True


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_word(
    csv_file_path: Path,
    search_input: str,
) -> list[list[str]]:
    """
    Search aligned words.

    Unlike the original application, this function does NOT
    modify the alignment CSV while searching.
    """

    if not file_exists(csv_file_path):
        return []

    search_input = search_input.strip().lower()

    if not search_input:
        return []

    search_words = search_input.split()

    results = []

    try:

        with csv_file_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as csv_file:

            rows = list(csv.reader(csv_file))

        # Skip CSV header.
        if rows and rows[0][0].lower() == "word":
            rows = rows[1:]

        normalized_words = [
            row[0].strip().lower()
            if row
            else ""
            for row in rows
        ]

        for index, row in enumerate(rows):

            if not row:
                continue

            current_word = normalized_words[index]

            if current_word != search_words[0]:
                continue

            # Single-word search.
            if len(search_words) == 1:
                results.append(row)
                continue

            # Multi-word sequential search.
            matches = True

            for offset, search_word_value in enumerate(
                search_words
            ):

                target_index = index + offset

                if target_index >= len(rows):
                    matches = False
                    break

                if (
                    normalized_words[target_index]
                    != search_word_value
                ):
                    matches = False
                    break

            if matches:
                results.append(row)

        return results

    except Exception as exc:
        st.error(
            f"Search failed: {exc}"
        )
        return []


# ---------------------------------------------------------------------------
# Audio seeking
# ---------------------------------------------------------------------------

def modify_audio(start_time: float) -> None:
    """
    Create a temporary audio stream starting at the
    requested timestamp.
    """

    if not file_exists(AUDIO_PATH):
        st.error(
            "No processed audio is available."
        )
        return

    try:

        audio = AudioSegment.from_file(
            str(AUDIO_PATH)
        )

        start_ms = max(
            0,
            int(start_time * 1000),
        )

        new_audio = audio[start_ms:]

        output = io.BytesIO()

        new_audio.export(
            output,
            format="wav",
        )

        output.seek(0)

        st.audio(
            output,
            format="audio/wav",
        )

    except Exception as exc:
        st.error(
            f"Could not seek audio: {exc}"
        )


# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------

def main() -> None:

    st.set_page_config(
        page_title="VANS Audio System — 2022",
        page_icon="🎙️",
        layout="wide",
    )

    st.title("🎙️ VANS Audio System")

    st.caption(
        "Legacy 2022 audio transcription, "
        "Gentle forced alignment and timestamp search."
    )

    # ---------------------------------------------------------
    # Legacy architecture warning
    # ---------------------------------------------------------

    st.warning(
        "Legacy 2022 application — this interface uses "
        "PocketSphinx + Gentle and is separate from the V3 "
        "FastAPI/React architecture."
    )

    # ---------------------------------------------------------
    # Service status
    # ---------------------------------------------------------

    with st.expander(
        "Legacy Service Status",
        expanded=True,
    ):

        if gentle_is_available():

            st.success(
                "Gentle: ONLINE — localhost:8888"
            )

        else:

            st.error(
                "Gentle: OFFLINE — localhost:8888"
            )

            st.code(
                "docker start intell_gentle",
                language="powershell",
            )

    # ---------------------------------------------------------
    # Input source
    # ---------------------------------------------------------

    st.subheader("1. Select Audio Source")

    audio_source = st.radio(
        "Audio source",
        [
            "Audio Upload",
            "YouTube Link",
        ],
        horizontal=True,
    )

    # ---------------------------------------------------------
    # Upload
    # ---------------------------------------------------------

    if audio_source == "Audio Upload":

        uploaded_file = st.file_uploader(
            "Upload audio",
            type=SUPPORTED_UPLOAD_TYPES,
        )

        if uploaded_file is not None:

            st.audio(
                uploaded_file,
                format=uploaded_file.type,
            )

            if st.button(
                "▶ Process Audio",
                type="primary",
            ):

                try:

                    with AUDIO_PATH.open(
                        "wb"
                    ) as output_file:

                        output_file.write(
                            uploaded_file.getbuffer()
                        )

                    st.success(
                        "Audio uploaded successfully."
                    )

                    process_audio(
                        AUDIO_PATH
                    )

                except Exception as exc:

                    st.error(
                        f"Could not save uploaded audio: "
                        f"{exc}"
                    )

    # ---------------------------------------------------------
    # YouTube
    # ---------------------------------------------------------

    else:

        youtube_link = st.text_input(
            "Enter YouTube link",
            placeholder="https://www.youtube.com/watch?v=...",
        )

        if st.button(
            "▶ Download & Process",
            type="primary",
        ):

            downloaded_audio = download_audio(
                youtube_link
            )

            if downloaded_audio:

                process_audio(
                    downloaded_audio
                )

    # ---------------------------------------------------------
    # No processed audio yet
    # ---------------------------------------------------------

    if not file_exists(AUDIO_PATH):

        st.info(
            "No audio has been processed yet. "
            "Upload an audio file or provide a YouTube link."
        )

        return

    # ---------------------------------------------------------
    # Transcript
    # ---------------------------------------------------------

    st.divider()

    st.subheader("2. Transcript")

    if file_exists(TEXT_PATH):

        try:

            transcript = TEXT_PATH.read_text(
                encoding="utf-8"
            )

            st.text_area(
                "PocketSphinx transcript",
                transcript,
                height=180,
            )

        except OSError as exc:

            st.error(
                f"Could not read transcript: {exc}"
            )

    else:

        st.info(
            "Transcript has not been generated yet."
        )

    # ---------------------------------------------------------
    # Original audio
    # ---------------------------------------------------------

    st.subheader("3. Original Audio")

    try:

        with AUDIO_PATH.open("rb") as audio_file:

            st.audio(
                audio_file.read(),
                format="audio/mp3",
            )

    except OSError as exc:

        st.error(
            f"Could not open audio: {exc}"
        )

    # ---------------------------------------------------------
    # Word search
    # ---------------------------------------------------------

    st.divider()

    st.subheader(
        "4. Search Word / Phrase"
    )

    if not file_exists(ALIGNMENT_PATH):

        st.info(
            "No Gentle alignment exists yet. "
            "Process the audio first."
        )

    else:

        search_input = st.text_input(
            "Enter word or phrase",
            placeholder="example: artificial intelligence",
        )

        if st.button("🔎 Search"):

            results = search_word(
                ALIGNMENT_PATH,
                search_input,
            )

            if not results:

                st.warning(
                    "No matching words found."
                )

            else:

                st.success(
                    f"Found {len(results)} match(es)."
                )

                st.session_state[
                    "search_results"
                ] = results

    # ---------------------------------------------------------
    # Search results
    # ---------------------------------------------------------

    search_results = st.session_state.get(
        "search_results",
        [],
    )

    if search_results:

        st.subheader(
            "5. Search Results"
        )

        for index, result in enumerate(
            search_results,
            start=1,
        ):

            if len(result) < 2:
                continue

            word = result[0]

            try:
                start_time = float(result[1])
            except (ValueError, TypeError):
                continue

            if len(result) >= 3:
                try:
                    end_time = float(result[2])
                except (ValueError, TypeError):
                    end_time = None
            else:
                end_time = None

            if end_time is not None:

                label = (
                    f"**Option {index}:** "
                    f"`{word}` — "
                    f"{start_time:.3f}s → "
                    f"{end_time:.3f}s"
                )

            else:

                label = (
                    f"**Option {index}:** "
                    f"`{word}` — "
                    f"{start_time:.3f}s"
                )

            st.markdown(label)

        # -----------------------------------------------------
        # Seek
        # -----------------------------------------------------

        st.subheader(
            "6. Timestamp Seeking"
        )

        selected_option = st.number_input(
            "Select result",
            min_value=1,
            max_value=len(search_results),
            value=1,
            step=1,
        )

        if st.button("⏩ Seek to Timestamp"):

            selected_result = (
                search_results[
                    selected_option - 1
                ]
            )

            try:

                start_time = float(
                    selected_result[1]
                )

                st.write(
                    f"Starting audio at "
                    f"`{start_time:.3f}` seconds."
                )

                modify_audio(
                    start_time
                )

            except (
                ValueError,
                TypeError,
                IndexError,
            ) as exc:

                st.error(
                    f"Invalid timestamp: {exc}"
                )

    # ---------------------------------------------------------
    # Legacy files
    # ---------------------------------------------------------

    st.divider()

    with st.expander(
        "Legacy Pipeline Files"
    ):

        files = [
            AUDIO_PATH,
            WAV_PATH,
            TEXT_PATH,
            ALIGNMENT_PATH,
        ]

        for path in files:

            if file_exists(path):

                size_mb = (
                    path.stat().st_size
                    / (1024 * 1024)
                )

                st.write(
                    f"✅ `{path.name}` "
                    f"({size_mb:.2f} MB)"
                )

            else:

                st.write(
                    f"⬜ `{path.name}`"
                )

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    if st.button(
        "🗑️ Clear Legacy Session"
    ):

        clear_legacy_files()

        st.session_state.pop(
            "search_results",
            None,
        )

        st.rerun()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()