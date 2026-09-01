# Intell Audio Inference & Retrieval System

> **Legacy 2022 implementation — preserved for historical comparison**

This repository contains the original **VANS Audio System**, an early audio transcription, forced-alignment, search, and timestamp-based playback prototype developed in 2022.

The project demonstrates an early approach to turning spoken audio into searchable, time-aligned text using traditional speech-recognition and forced-alignment technologies.

A significantly modernized **2026 architecture** has since been developed on separate branches, introducing Whisper, Silero VAD, speaker intelligence, acoustic analysis, vector retrieval, RAG, Qdrant, and a React/TypeScript frontend.

The legacy implementation remains preserved here so the evolution of the system can be demonstrated clearly.

---

## 🎥 Legacy System Demo

The following video demonstrates the original 2022 system and its processing pipeline:

**Legacy Model Demonstration**
The demonstration shows:

1. Audio ingestion
2. Speech recognition
3. Transcript generation
4. Gentle forced alignment
5. Word-level timestamps
6. Timestamp-based audio navigation
7. Lexical word search

### Demo Video

[▶️ Watch the Legacy 2022 System Demo](assets/legacy%20model.mp4)

---

## What Was the 2022 System?

The original system was designed around a relatively straightforward pipeline:

```text
Audio
   │
   ▼
Audio Extraction
   │
   ▼
Speech Recognition
(PocketSphinx)
   │
   ▼
Transcript
   │
   ▼
Gentle Forced Alignment
   │
   ▼
Word + Timestamp CSV
   │
   ▼
Search
   │
   ▼
Timestamp-based Audio Playback
```

The goal was to make spoken content searchable and allow the user to jump to the point in the recording where a particular word was spoken.

---

## Core Technologies

| Component          | Technology      |
| ------------------ | --------------- |
| Frontend           | Streamlit       |
| Speech Recognition | PocketSphinx    |
| Forced Alignment   | Gentle          |
| Alignment Server   | Docker          |
| Audio Processing   | MoviePy / Pydub |
| YouTube Audio      | PyTube          |
| Search             | CSV + Python    |
| NLP Processing     | NLTK            |
| Language           | Python          |

---

## Gentle Forced Alignment

A central component of the original system is **Gentle**, an open-source forced-alignment system.

Gentle takes:

* an audio file
* an existing transcript

and attempts to determine **when each word occurs in the audio**.

Conceptually:

```text
Audio:
───────────────────────────────────────────────►

Transcript:
"hello this is an audio system"

Gentle:
hello      0.42s ── 0.81s
this       0.86s ── 1.04s
is         1.07s ── 1.15s
an         1.18s ── 1.28s
audio      1.31s ── 1.68s
system     1.72s ── 2.20s
```

These timestamps are then written into:

```text
alignment_result.csv
```

The application can subsequently search this file and use the timestamp to jump into the corresponding portion of the audio.

---

## How the Legacy Search Works

The original application performs a simple lexical search over the alignment CSV.

For example:

```text
User searches:
audio
```

The system looks through the aligned words and finds matching entries.

It can then return something similar to:

```text
Option 1:
Word 'audio' found at t=1.31s
```

Selecting the result allows the application to create an audio playback starting from that timestamp.

This was an early implementation of **timestamp-grounded audio retrieval**.

---

## Why Gentle Was Important

Gentle provided something that ordinary speech recognition did not provide reliably:

> **Temporal information about individual words.**

A conventional transcript might produce:

```text
"this is an audio system"
```

Gentle attempted to produce:

```text
this   → 0.42s
is     → 1.07s
an     → 1.18s
audio  → 1.31s
system → 1.72s
```

This temporal representation became an important foundation for the later versions of the project.

---

## Limitations of the 2022 Approach

The original implementation was useful as a prototype, but it had several significant limitations.

### 1. PocketSphinx Accuracy

PocketSphinx is a lightweight traditional speech-recognition engine, but its recognition accuracy is substantially lower than modern neural ASR systems such as Whisper, particularly with:

* accents
* background noise
* conversational speech
* multiple speakers
* spontaneous dialogue

Incorrect transcription directly affects downstream alignment.

---

### 2. Gentle Depends on an Existing Transcript

Gentle is primarily a **forced-alignment system**, not a modern end-to-end speech recognizer.

It expects a transcript and attempts to align that transcript to the audio.

Therefore:

```text
Bad Transcript
      ↓
Poor Alignment
      ↓
Incorrect Timestamps
```

The quality of the final result is therefore dependent on the quality of the preceding transcription stage.

---

### 3. Multiple Speakers

The original pipeline does not perform modern speaker diarization.

It does not robustly answer:

```text
Who is speaking?
When did Speaker 1 stop?
When did Speaker 2 start?
```

It primarily operates on the supplied transcript and audio signal.

---

### 4. Overlapping Speech

When two people speak simultaneously, traditional forced alignment can struggle to determine the correct temporal boundaries.

This becomes especially problematic for:

* interviews
* meetings
* films
* podcasts
* conversational recordings

---

### 5. Short Responses

Very short utterances such as:

```text
"Yeah."

"Okay."

"Right."

"No."
```

can be difficult to align reliably, particularly when surrounded by other speech or noise.

---

### 6. Legacy Architecture

The original application was essentially a Streamlit monolith.

Several responsibilities were coupled together:

```text
UI
│
├── Audio processing
├── Speech recognition
├── Gentle API communication
├── CSV generation
├── Search
└── Audio playback
```

This made the system harder to extend and maintain as additional intelligence was introduced.

---

# Project Evolution

The project subsequently evolved into a substantially different architecture.

### 2022 — Legacy System

```text
PocketSphinx
     ↓
Transcript
     ↓
Gentle
     ↓
Word Alignment
     ↓
CSV Search
     ↓
Timestamp Playback
```

### 2026 — Modernized System

```text
Audio / Video
      ↓
Silero VAD
      ↓
Whisper ASR
      ↓
Temporal Transcript
      ↓
Speaker Intelligence
      ↓
Acoustic Analysis
      ↓
Hybrid Retrieval
      ↓
Qdrant
      ↓
RAG / Qwen3
      ↓
Grounded Answers
      ↓
Timestamp Citations
```

The 2026 implementation is maintained separately so that the original system can remain reproducible and historically intact.

---

## Branch Structure

The repository contains separate development histories.

The **legacy `main` branch** represents the original 2022 implementation.

The newer development branches contain the 2026 modernization work.

This separation allows the project to demonstrate the progression from:

**traditional speech processing → modern AI audio intelligence**

without rewriting the original implementation.

---

## Running the Legacy Application

### Requirements

* Python 3.11+
* Streamlit
* PocketSphinx / SpeechRecognition
* Gentle
* Docker
* MoviePy
* Pydub
* NLTK

### Start Gentle

The legacy application expects Gentle to be available at:

```text
http://localhost:8888
```

Start the Gentle Docker container:

```powershell
docker run -d `
  --name intell_gentle `
  -p 8888:8765 `
  lowerquality/gentle:latest
```

Verify:

```powershell
docker ps --filter "name=intell_gentle"
```

### Start Streamlit

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## Original Workflow

1. Select **Audio Upload** or **YouTube Link**
2. Provide an audio source
3. Download or load the audio
4. Convert the audio to WAV
5. Generate a transcript using PocketSphinx
6. Send the transcript and audio to Gentle
7. Generate word-level alignment
8. Store alignment results in CSV
9. Search for words
10. Select a timestamp
11. Play audio from the selected location

---

## Historical Significance

Although technically simple by modern standards, this prototype established several concepts that continued into later versions of the project:

* speech-to-text processing
* temporal representation of speech
* searchable transcripts
* word-level timestamps
* audio navigation through textual search
* retrieval of information from recorded speech

The 2026 system builds upon these ideas using modern neural speech processing and retrieval technologies.

---

## Project Status

**2022 Legacy Implementation — Preserved**

This branch is maintained primarily for:

* historical reference
* reproducibility
* demonstration
* architecture comparison
* understanding the evolution of the project

The modern implementation should be evaluated from its dedicated 2026 development branch.

---

## License

See the repository license for details.
