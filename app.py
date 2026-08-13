# -*- coding: utf-8 -*-
"""
Intell Audio Inference & Retrieval System
Legacy V1 - cleaned and stabilized baseline.

Original project was generated from a Google Colab notebook.
"""

import csv
import io
import os

import requests
import speech_recognition as sr
import streamlit as st
from moviepy.editor import AudioFileClip
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pydub import AudioSegment
from pytube import YouTube


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AUDIO_FILE = "my_audio.mp3"
WAV_FILE = "my_audio.mp3.wav"
TRANSCRIPT_FILE = "audio.txt"
ALIGNMENT_FILE = "alignment_result.csv"
GENTLE_URL = "http://localhost:8888/transcriptions?async=false"

STOP_WORDS = set(stopwords.words("english"))


# ---------------------------------------------------------------------------
# Audio acquisition
# ---------------------------------------------------------------------------

def download_audio(youtube_link):
    """Download the lowest-bitrate audio stream from a YouTube URL."""
    try:
        video = YouTube(youtube_link)
        audio_streams = video.streams.filter(only_audio=True)
        audio_streams = sorted(audio_streams, key=lambda stream: stream.bitrate)

        if not audio_streams:
            st.error("No audio stream was found for this YouTube video.")
            return None

        audio = audio_streams[0]

        st.write("Downloading audio...")
        audio.download(filename=AUDIO_FILE)
        st.success("Audio downloaded successfully!")

        return AUDIO_FILE

    except Exception as exc:
        st.error(f"An error occurred during audio download: {exc}")
        return None


# ---------------------------------------------------------------------------
# Speech recognition + forced alignment
# ---------------------------------------------------------------------------

def process_audio(audio_filename):
    """Convert audio to text and generate word-level alignment with Gentle."""
    try:
        st.write("Converting audio to text...")

        audio_clip = AudioFileClip(audio_filename)
        try:
            audio_clip.write_audiofile(WAV_FILE)
        finally:
            audio_clip.close()

        recognizer = sr.Recognizer()

        with sr.AudioFile(WAV_FILE) as source:
            audio_data = recognizer.record(source)

        # Legacy V1 uses PocketSphinx for offline speech recognition.
        text = recognizer.recognize_sphinx(audio_data)

        st.success("Audio converted to text successfully!")
        st.write("Text:")
        st.write(text)

        with open(TRANSCRIPT_FILE, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        st.success("Text saved successfully as a text file!")

        # Gentle forced alignment
        st.write("Performing forced alignment...")

        with open(WAV_FILE, "rb") as audio_file:
            files = {"audio": audio_file}
            data = {"transcript": text}

            response = requests.post(
                GENTLE_URL,
                files=files,
                data=data,
                timeout=300,
            )

        if response.status_code != 200:
            st.error(
                f"Failed to perform forced alignment. "
                f"Gentle returned HTTP {response.status_code}."
            )
            st.code(response.text[:2000])
            return False

        alignment_data = response.json()

        with open(ALIGNMENT_FILE, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["word", "start", "end"])

            for word in alignment_data.get("words", []):
                if "start" in word:
                    start_time = word["start"]
                elif "startOffset" in word:
                    start_time = word["startOffset"] / 1000
                else:
                    start_time = ""

                if "end" in word:
                    end_time = word["end"]
                elif "endOffset" in word:
                    end_time = word["endOffset"] / 1000
                else:
                    end_time = ""

                writer.writerow([word.get("word", ""), start_time, end_time])

        st.success("Forced alignment result saved as CSV.")
        return True

    except sr.UnknownValueError:
        st.error("PocketSphinx could not understand the audio.")
    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to Gentle at "
            f"{GENTLE_URL}. Make sure the Gentle Docker container is running."
        )
    except requests.exceptions.RequestException as exc:
        st.error(f"Gentle request failed: {exc}")
    except Exception as exc:
        st.error(f"An error occurred during audio processing: {exc}")

    return False


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_word(csv_file_path, search_input):
    """
    Search the alignment CSV for a word or consecutive phrase.

    The original implementation permanently removed stop-word rows from
    alignment_result.csv before searching. This version does not modify the
    alignment file, preserving the original timestamps.
    """
    if not search_input or not search_input.strip():
        return []

    if not os.path.exists(csv_file_path):
        return []

    search_words = [
        word.lower()
        for word in word_tokenize(search_input.strip())
        if word.strip()
    ]

    if not search_words:
        return []

    with open(csv_file_path, "r", newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))

    # Remove the CSV header.
    if rows and rows[0] and rows[0][0].lower() == "word":
        rows = rows[1:]

    # Ignore malformed/empty rows.
    rows = [row for row in rows if len(row) >= 3 and row[0].strip()]

    search_results = []

    for index, row in enumerate(rows):
        current_word = row[0].strip().lower()

        # Single-word search.
        if len(search_words) == 1:
            if current_word == search_words[0]:
                search_results.append(row)
            continue

        # Consecutive phrase search.
        candidate_words = [
            rows[index + offset][0].strip().lower()
            for offset in range(len(search_words))
            if index + offset < len(rows)
        ]

        if candidate_words == search_words:
            search_results.append(row)

    return search_results


# ---------------------------------------------------------------------------
# File/session cleanup
# ---------------------------------------------------------------------------

def delete_files(file_paths):
    """Delete generated working files if they exist."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except OSError as exc:
            st.error(f"Error deleting file '{file_path}': {exc}")


def clear_session_results():
    """Clear cached search results."""
    st.session_state.search_results = []


# ---------------------------------------------------------------------------
# Timestamp playback
# ---------------------------------------------------------------------------

def modify_audio(start_time):
    """Create an audio preview beginning at the selected timestamp."""
    try:
        audio = AudioSegment.from_file(AUDIO_FILE)
        new_audio = audio[int(float(start_time) * 1000):]
        new_audio_bytes = new_audio.export(format="wav").read()

        st.audio(io.BytesIO(new_audio_bytes), format="audio/wav")

    except Exception as exc:
        st.error(f"Unable to seek audio: {exc}")


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def main():
    st.title("VANS AUDIO SYSTEM")

    # Initialize persistent search state.
    if "search_results" not in st.session_state:
        st.session_state.search_results = []

    # Select audio source.
    audio_source = st.radio(
        "Select audio source",
        ("Audio Upload", "YouTube Link"),
    )

    if audio_source == "Audio Upload":
        uploaded_file = st.file_uploader(
            "Upload Audio",
            type=["mp3", "wav"],
        )

        if st.button("Process Audio"):
            if uploaded_file is None:
                st.warning("Please upload an audio file first.")
            else:
                with open(AUDIO_FILE, "wb") as audio_file:
                    audio_file.write(uploaded_file.read())

                st.success("Audio uploaded successfully!")
                clear_session_results()
                process_audio(AUDIO_FILE)

    else:
        youtube_link = st.text_input("Enter YouTube link")

        if st.button("Process Audio"):
            if not youtube_link.strip():
                st.warning("Please enter a YouTube link first.")
            else:
                audio_filename = download_audio(youtube_link)

                if audio_filename:
                    clear_session_results()
                    process_audio(audio_filename)

    # Only show the analysis/search UI when processing has produced
    # all required files.
    files_ready = all(
        os.path.exists(path)
        for path in (TRANSCRIPT_FILE, AUDIO_FILE, ALIGNMENT_FILE)
    )

    if not files_ready:
        return

    # -----------------------------------------------------------------------
    # Transcript
    # -----------------------------------------------------------------------

    with open(TRANSCRIPT_FILE, "r", encoding="utf-8") as text_file:
        text = text_file.read()

    st.subheader("Transcript")
    st.write(text)

    # -----------------------------------------------------------------------
    # Audio player
    # -----------------------------------------------------------------------

    with open(AUDIO_FILE, "rb") as audio_file:
        audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/mp3")

    # -----------------------------------------------------------------------
    # Search
    # -----------------------------------------------------------------------

    st.subheader("Transcript Search")

    search_input = st.text_input("Enter word or phrase to search")

    if st.button("Search"):
        if not search_input.strip():
            st.warning("Please enter a word or phrase to search.")
            clear_session_results()
        else:
            st.session_state.search_results = search_word(
                ALIGNMENT_FILE,
                search_input,
            )

            if not st.session_state.search_results:
                st.warning("No results found.")
            else:
                st.success(
                    f"Found {len(st.session_state.search_results)} result(s)."
                )

    search_results = st.session_state.search_results

    # Display persistent search results after Streamlit reruns.
    if search_results:
        st.write("Search results:")

        for index, result in enumerate(search_results):
            try:
                start_time = float(result[1])
            except (ValueError, IndexError):
                continue

            st.markdown(
                f"**Option {index + 1}:** "
                f"'{search_input}' found at "
                f"`{start_time:.2f}s`"
            )

        # -------------------------------------------------------------------
        # Seek
        # -------------------------------------------------------------------

        selected_option = st.number_input(
            "Enter the option to seek",
            min_value=1,
            max_value=len(search_results),
            step=1,
            value=1,
            key="seek_option",
        )

        if st.button("Seek"):
            selected_result = search_results[selected_option - 1]
            start_time = float(selected_result[1])
            modify_audio(start_time)

    # -----------------------------------------------------------------------
    # Done / cleanup
    # -----------------------------------------------------------------------

    if st.button("Done"):
        delete_files(
            [
                AUDIO_FILE,
                WAV_FILE,
                TRANSCRIPT_FILE,
                ALIGNMENT_FILE,
            ]
        )

        clear_session_results()
        st.success("All generated files deleted successfully!")
        st.rerun()


if __name__ == "__main__":
    main()