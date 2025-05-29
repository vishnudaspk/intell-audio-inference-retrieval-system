# Intell.-Audio_Inference-and-Retrieval_Sys.

An intelligent audio processing system that extracts, transcribes, aligns, and allows word-level search and playback within audio files or YouTube audio streams.

---

## Overview

This project implements a smart audio inference and retrieval system using Streamlit, enabling users to:

- Upload audio files or extract audio from YouTube videos.
- Convert audio to text using speech recognition.
- Perform forced alignment between audio and text with Gentle.
- Search for specific words in the audio transcript.
- Seek playback to exact timestamps where searched words occur.
- Interact via a user-friendly web interface.

---

## Features

1. **Audio Upload:** Upload audio files (MP3 or WAV) for processing.  
2. **YouTube Link:** Extract and process audio from YouTube links.  
3. **Audio to Text:** Speech-to-text conversion using the SpeechRecognition library.  
4. **Forced Alignment:** Precise alignment of text with audio timestamps via Gentle.  
5. **Word Search:** Search for words in the transcript with contextual filtering.  
6. **Playback Seek:** Jump to exact points in audio based on word search results.  
7. **Streamlit UI:** Interactive, real-time web application interface.

---

## Getting Started

### Prerequisites

- Python 3.6 or higher

### Install dependencies

```bash
pip install -r requirements.txt
