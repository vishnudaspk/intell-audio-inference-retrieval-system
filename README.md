# Intell Audio Inference & Retrieval System

Short project description

## What Is Intell Audio?

Explain the original 2022 project and the 2026 modernization.

## Why This Project Was Rebuilt

2022:
- PocketSphinx
- basic transcript generation
- Gentle alignment
- CSV-based search
- Streamlit monolith

2026:
- modular architecture
- modern ASR-ready engine abstraction
- temporal transcript representation
- BM25
- semantic embeddings
- Qdrant
- hybrid retrieval
- RAG
- Qwen3
- timestamp-grounded answers
- FastAPI + Streamlit

## Architecture

[architecture diagram]

## Core Pipeline

Audio
→ ASR
→ Temporal Transcript
→ Chunking
→ BM25 + Embeddings
→ Qdrant
→ Hybrid Retrieval
→ Reranking
→ Qwen3
→ Grounded Answer
→ Timestamp Citation

## Features

### Audio ingestion
### Transcription
### Forced alignment
### Temporal indexing
### Hybrid retrieval
### Ask the Audio
### Timestamp citations

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| API | FastAPI |
| Storage | SQLite |
| Lexical retrieval | BM25 |
| Vector DB | Qdrant |
| Embeddings | Qwen3-Embedding-0.6B |
| LLM | Qwen3-8B |
| Local inference | LM Studio |
| Alignment | Gentle |
| Audio | MoviePy / Pydub |
| Testing | Pytest |

## Project Structure

[tree]

## Requirements

Hardware:
- NVIDIA RTX 4060 8GB recommended
- 16GB+ system RAM recommended
- Docker Desktop
- Python 3.11+

Software:
- LM Studio
- Qdrant
- Gentle
- Python

## Installation

git clone ...
cd ...
python -m venv .venv
...
pip install -r requirements.txt

## Local Services

### 1. Gentle

docker ...

### 2. Qdrant

docker ...

### 3. LM Studio

Explain model configuration.

## Configuration

.env example

## Running

Terminal 1:
Gentle

Terminal 2:
Qdrant

Terminal 3:
LM Studio

Terminal 4:
FastAPI

Terminal 5:
Streamlit

## Usage

1. Upload audio
2. Process
3. Index
4. Ask questions
5. Open timestamp citation

## API

Health
Ingest
Index
Search
Ask

## Testing

pytest

## Example Questions

"What topics were discussed?"
"When was X mentioned?"
"What did the speaker say about Y?"
"Summarize the discussion about Z."

## Design Principles

### Local-first
### Evidence-grounded
### Temporal retrieval
### Deterministic citations
### Modular providers

## Current Limitations

Be honest about:
- ASR engine
- speaker diarization
- model context limits
- local inference performance
- YouTube dependency
- Gentle limitations

## Roadmap

Phase 1 — Architecture ✓
Phase 2 — Modern ASR
Phase 3 — Temporal Transcript ✓
Phase 4 — Intelligence
Phase 5 — Intelligent Retrieval ✓
Phase 6 — Ask the Audio ✓
Phase 7 — Speaker Intelligence
Phase 8 — Productionization

## Project Evolution

2022 → 2026


🔵 V3.2 — Speaker Intelligence




=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

🔵 V3.2 — Speaker Intelligence
CASA
dialogue consistency
speaker-attribution confidence
acoustic + temporal + linguistic fusion
short-response handling
early-dialogue stabilization
attribution evaluation benchmark


🔵 V3.3 — Temporal Intelligence
word/phrase/turn synchronization
speaker turn boundaries
interruptions
overlaps
pause analysis
timeline intelligence


🔵 V3.4 — Multimodal Reasoning
transcript reasoning
audio reasoning
visual context
cross-modal evidence
conversational understanding
event/scene understanding


🏁 V3.5 — V3 Final / Release Candidate
Then we freeze the architecture and perform:
regression testing
accuracy benchmarking
performance benchmarking
UI cleanup
documentation
failure-case analysis
reproducibility testing